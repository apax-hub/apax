"""MACE descriptor for apax.

Exposes :class:`MaceRepresentation`, a Flax linen ``nn.Module`` that consumes
pair displacement vectors and atomic numbers, and returns per-atom scalar
features compatible with apax's :class:`AtomisticReadout`.

Matches the apax descriptor contract exactly:
``__call__(dr_vec, Z, idx) -> (n_atoms, n_features)``.

The heavy equivariant math lives in :mod:`apax.layers.descriptor.mace_blocks`
and is composed here into the full MACE pipeline:
``LinearNodeEmbedding -> N x (InteractionBlock -> ProductBlock) -> concat scalars``.
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

    Notes
    -----
    Output is the concatenation of per-layer scalar (``0e``) features. The
    per-layer layout is relied upon by the planned ``MaceReadout`` layer
    (see plan P3.3) to reproduce the foundation MACE forward pass via
    per-layer linear / non-linear heads.
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
        from apax.layers.descriptor.mace_blocks import (
            InteractionBlock,
            LinearNodeEmbedding,
            ProductBlock,
            assemble_edge_features,
        )

        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)
        i, j = idx[0], idx[1]

        pair_mask = _get_neighbor_mask(idx) if self.apply_mask else 1.0
        node_mask = _get_node_mask(Z) if self.apply_mask else 1.0

        radial, sph = assemble_edge_features(
            dr_vec,
            self.r_max,
            self.num_bessel,
            self.num_polynomial_cutoff,
            self.max_ell,
        )
        if self.apply_mask:
            radial = radial * pair_mask[..., None]

        # Initial node features: scalars only
        scalar_init = _scalar_irreps_only(self.hidden_irreps)
        node_feats = LinearNodeEmbedding(
            num_elements=self.num_elements,
            irreps_out=scalar_init,
        )(Z)

        per_layer_scalars = []
        for _ in range(self.num_interactions):
            node_feats = InteractionBlock(
                irreps_out=self.hidden_irreps,
                interaction_cls=self.interaction_cls,
            )(node_feats, sph, radial, i, j)
            node_feats = ProductBlock(
                hidden_irreps=self.hidden_irreps,
                correlation=self.correlation,
                num_elements=self.num_elements,
                use_cueq=self.use_cueq,
            )(node_feats, Z)
            per_layer_scalars.append(node_feats.filter(keep="0e").array)

        features = jnp.concatenate(per_layer_scalars, axis=-1)
        if self.apply_mask:
            features = features * node_mask[..., None]
        features = features.astype(dtype)
        return features


def _get_node_mask(Z):
    """Return an int16 mask of real (non-padding) atoms.

    Parameters
    ----------
    Z : Array
        Atomic-number vector; zeros mark padding atoms.

    Returns
    -------
    Array
        1 where ``Z != 0``, else 0, as ``int16``.
    """
    return (Z != 0).astype(jnp.int16)


def _get_neighbor_mask(idx):
    """Return an int16 mask of real (non-self-pair) neighbor edges.

    Parameters
    ----------
    idx : Array, shape (2, n_edges)
        Edge index array with rows ``(receivers, senders)``.

    Returns
    -------
    Array
        1 where ``idx[0] != idx[1]``, else 0, as ``int16``.
    """
    return ((idx[0] - idx[1]) != 0).astype(jnp.int16)


def _scalar_irreps_only(irreps_str: str) -> str:
    """Return the 0e subset of an irreps string.

    Parameters
    ----------
    irreps_str : str
        Full irreps string, e.g. ``"128x0e + 128x1o"``.

    Returns
    -------
    str
        Only the ``0e`` component(s), e.g. ``"128x0e"``.

    Raises
    ------
    ValueError
        If no ``0e`` component is present.
    """
    parts = [p.strip() for p in irreps_str.split("+")]
    scalar = [p for p in parts if p.endswith("x0e")]
    if not scalar:
        raise ValueError(f"No 0e component in irreps {irreps_str!r}")
    return " + ".join(scalar)
