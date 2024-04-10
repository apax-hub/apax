from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from apax.nn.impl.basis import cosine_cutoff, gaussian_basis_impl, radial_basis_impl


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
        basis = gaussian_basis_impl(
            dr.type(self.dtype), self.shifts, self.betta, self.rad_norm
        )
        return basis


class RadialFunctionT(nn.Module):
    def __init__(
        self,
        n_radial: int = 5,
        basis_fn: nn.Module = GaussianBasisT(),
        emb_init: str = "uniform",
        n_species: int = 119,
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
        if self.emb_init is not None:
            self.n_radial = n_radial
            emb = torch.rand((self.n_species,
                        self.n_species,
                        self.n_radial,
                        self.basis_fn.n_basis))
            self.embeddings = nn.Parameter(emb)
        else:
            self.n_radial = self.basis_fn.n_basis

    def forward(self, dr, Z_i, Z_j):
        dr = dr.type(self.dtype)
        # basis shape: neighbors x n_basis
        basis = self.basis_fn(dr)

        radial_function = radial_basis_impl(
            basis, Z_i, Z_j, self.embeddings, self.embed_norm
        )
        cutoff = cosine_cutoff(dr, self.r_max)
        radial_function = radial_function * cutoff

        assert radial_function.dtype == self.dtype
        return radial_function
