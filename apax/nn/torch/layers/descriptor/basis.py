from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn


def cosine_cutoff(dr, r_max):
    # shape: neighbors
    dr_clipped = torch.clamp(dr, max=r_max)
    cos_cutoff = 0.5 * (torch.cos(np.pi * dr_clipped / r_max) + 1.0)
    cutoff = cos_cutoff[:, None]
    return cutoff


class GaussianBasisT(nn.Module):
    def __init__(
        self,
        n_basis: int = 7,
        r_min: float = 0.5,
        r_max: float = 6.0,
        dtype: Any = torch.float32,
    ) -> None:
        super().__init__()
        self.n_basis = n_basis
        self.r_min = r_min
        self.r_max = r_max
        self.dtype = dtype

        self.betta = self.n_basis**2 / self.r_max**2
        self.rad_norm = (2.0 * self.betta / np.pi) ** 0.25
        shifts = self.r_min + (self.r_max - self.r_min) / self.n_basis * np.arange(
            self.n_basis
        )

        self.betta = torch.tensor(self.betta)
        self.rad_norm = torch.tensor(self.rad_norm)

        # shape: 1 x n_basis
        shifts = einops.repeat(shifts, "n_basis -> 1 n_basis")
        self.shifts = torch.tensor(shifts, dtype=self.dtype)

    def forward(self, dr: torch.Tensor) -> torch.Tensor:
        # dr shape: neighbors
        # neighbors -> neighbors x 1
        dr = dr[:, None].type(self.dtype)
        # 1 x n_basis, neighbors x 1 -> neighbors x n_basis
        distances = self.shifts - dr

        # shape: neighbors x n_basis
        basis = torch.exp(-self.betta * (distances**2))
        basis = self.rad_norm * basis
        return basis


class RadialFunctionT(nn.Module):
    def __init__(
        self,
        n_radial: int = 5,
        basis_fn: nn.Module = GaussianBasisT(),
        emb_init: str = "uniform",
        n_species: int = 119,
        params=None,
        dtype: Any = torch.float32,
    ) -> None:
        super().__init__()
        self.n_radial = n_radial
        self.basis_fn = basis_fn
        self.n_species = n_species
        self.emb_init = emb_init
        self.dtype = dtype

        self.r_max = torch.tensor(self.basis_fn.r_max)
        norm = 1.0 / np.sqrt(self.basis_fn.n_basis)
        self.embed_norm = torch.tensor(norm, dtype=self.dtype)
        self.embeddings = None

        if params:
            emb = params["atomic_type_embedding"]
            emb = torch.from_numpy(np.array(emb))
            self.embeddings = nn.Parameter(emb)
        elif self.emb_init is not None:
            self.n_radial = n_radial
            emb = torch.rand(
                (self.n_species, self.n_species, self.n_radial, self.basis_fn.n_basis)
            )
            self.embeddings = nn.Parameter(emb)
        else:
            self.n_radial = self.basis_fn.n_basis

    def forward(self, dr, Z_i, Z_j):
        dr = dr.type(self.dtype)
        # basis shape: neighbors x n_basis
        basis = self.basis_fn(dr)

        if self.embeddings is None:
            radial_function = basis
        else:
            # coeffs shape: n_neighbors x n_radialx n_basis
            # reverse convention to match original
            species_pair_coeffs = self.embeddings[Z_j, Z_i, ...]
            species_pair_coeffs = self.embed_norm * species_pair_coeffs

            radial_function = torch.einsum(
                "nrb, nb -> nr",
                species_pair_coeffs,
                basis,
            )
        cutoff = cosine_cutoff(dr, self.r_max)
        radial_function = radial_function * cutoff

        assert radial_function.dtype == self.dtype
        return radial_function
