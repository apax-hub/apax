"""MACE descriptor for apax.

Exposes :class:`MaceRepresentation`, a Flax linen ``nn.Module`` that consumes
pair displacement vectors and atomic numbers, and returns per-atom scalar
features compatible with apax's :class:`AtomisticReadout`.

Matches the apax descriptor contract exactly:
``__call__(dr_vec, Z, idx) -> (n_atoms, n_features)``.

The heavy equivariant math lives in :mod:`apax.layers.descriptor.mace_blocks`
and is added progressively over P1–P2. This module starts as a typed skeleton
that returns random-but-finite scalar features so the rest of apax (builder,
config, readout wiring) can be validated end-to-end first.
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
from flax import linen as nn

from apax.utils.convert import str_to_dtype

InteractionKind = Literal[
    "RealAgnostic",
    "RealAgnosticResidual",
    "RealAgnosticDensity",
    "RealAgnosticDensityResidual",
]


class MaceRepresentation(nn.Module):
    """MACE descriptor producing per-atom scalar features.

    Parameters
    ----------
    r_max : float
        Interaction cutoff in Angstrom.
    num_bessel : int
        Number of Bessel radial basis functions.
    num_polynomial_cutoff : int
        Polynomial order of the envelope cutoff.
    max_ell : int
        Maximum spherical-harmonic degree used for edge features.
    hidden_irreps : str
        e3nn irreps string for node features, e.g. ``"128x0e + 128x1o"``.
    num_interactions : int
        Number of (interaction, product) layer pairs.
    correlation : int
        Symmetric-contraction correlation order.
    interaction_cls : str
        Which MACE interaction block variant to use.
    num_elements : int
        Size of the chemical-element embedding table.
    use_cueq : bool
        If True, use cuequivariance-jax kernels where available.
    apply_mask : bool
        If True, zero out masked atoms in the output.
    dtype : Any
        Floating-point dtype for features.
    """

    r_max: float = 5.0
    num_bessel: int = 8
    num_polynomial_cutoff: int = 5
    max_ell: int = 3
    hidden_irreps: str = "128x0e + 128x1o"
    num_interactions: int = 2
    correlation: int = 3
    interaction_cls: InteractionKind = "RealAgnosticResidual"
    num_elements: int = 119
    use_cueq: bool = False
    apply_mask: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, dr_vec, Z, idx):
        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)
        n_atoms = Z.shape[0]
        # P0 skeleton: return a random-but-finite per-atom feature tensor
        # so the rest of apax (readout, scale/shift, train, MD) can be wired.
        # Replaced by real forward pass in P1.
        n_scalar = _scalar_feature_dim(self.hidden_irreps) * self.num_interactions
        w = self.param(
            "skeleton_w",
            nn.initializers.normal(stddev=0.01),
            (self.num_elements, n_scalar),
            dtype,
        )
        features = w[Z]
        return features


def _scalar_feature_dim(irreps_str: str) -> int:
    """Parse irreps string and return the multiplicity of the 0e component.

    Parameters
    ----------
    irreps_str : str
        e3nn irreps string, e.g. ``"128x0e + 128x1o"``.

    Returns
    -------
    int
        Multiplicity of the ``0e`` irrep, or 0 if not found.
    """
    for part in irreps_str.split("+"):
        part = part.strip()
        if part.endswith("x0e"):
            return int(part.split("x")[0])
    return 0
