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

from typing import Any, Literal, Union

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from flax import linen as nn

from apax.utils.convert import str_to_dtype

# Mirrors :attr:`apax.config.model_config.MaceModelConfig.interaction_cls`.
InteractionKind = Literal[
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
    interaction_cls : Literal["RealAgnosticResidual"]
        Which MACE interaction block variant to use. Pinned to
        ``"RealAgnosticResidual"`` until other variants land — see
        :class:`apax.config.model_config.MaceModelConfig`.
    num_elements : int
        Size of the chemical-element embedding table.
    use_cueq : bool
        If True, use cuequivariance-jax kernels where available.
    apply_mask : bool
        If True, zero out masked atoms in the output.
    dtype : Any
        Floating-point dtype for features.
    avg_num_neighbors : float
        Per-message normaliser forwarded to every :class:`InteractionBlock`.
        Foundation models burn in a per-dataset average (~62 for MACE-MP-0);
        defaults to ``1.0`` to leave freshly trained apax models unchanged.

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
    interaction_cls: Union[InteractionKind, tuple[InteractionKind, ...]] = (
        "RealAgnosticResidual"
    )
    num_elements: int = 119
    use_cueq: bool = False
    apply_mask: bool = True
    dtype: Any = jnp.float32
    avg_num_neighbors: float = 1.0
    distance_transform: Any = None

    @nn.compact
    def __call__(self, dr_vec, Z, idx):
        from apax.layers.descriptor.basis_functions import MaceRadialEmbedding
        from apax.layers.descriptor.mace_blocks import (
            _INTERACTION_BLOCK_CLS,
            LinearNodeEmbedding,
            ProductBlock,
        )

        # Resolve ``interaction_cls`` to a concrete per-layer list. A single
        # str is broadcast; a list/tuple must match ``num_interactions``.
        if isinstance(self.interaction_cls, str):
            per_layer_cls = [self.interaction_cls] * self.num_interactions
        else:
            per_layer_cls = list(self.interaction_cls)
            if len(per_layer_cls) != self.num_interactions:
                raise ValueError(
                    f"interaction_cls list length {len(per_layer_cls)} does not "
                    f"match num_interactions {self.num_interactions}"
                )

        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)
        i, j = idx[0], idx[1]

        pair_mask = _get_neighbor_mask(idx) if self.apply_mask else 1.0
        node_mask = _get_node_mask(Z) if self.apply_mask else 1.0

        radial, sph = MaceRadialEmbedding(
            r_max=self.r_max,
            num_bessel=self.num_bessel,
            num_polynomial_cutoff=self.num_polynomial_cutoff,
            max_ell=self.max_ell,
            distance_transform=self.distance_transform,
            name="radial_embedding",
        )(dr_vec, Z, idx)
        if self.apply_mask:
            radial = radial * pair_mask[..., None]

        # Compute the full multi-irrep target (matches torch-mace + mace-jax):
        # interaction_irreps = (sh_irreps × num_features).sort().simplify()
        sh_irreps = e3nn.Irreps.spherical_harmonics(self.max_ell)
        sh_irreps_str = str(sh_irreps)
        hidden_irreps = e3nn.Irreps(self.hidden_irreps)
        # Validate up-front so the error matches the legacy contract test.
        _scalar_irreps_only(self.hidden_irreps)
        num_features = hidden_irreps.count(e3nn.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort().irreps.simplify()
        interaction_irreps_str = str(interaction_irreps)
        node_attrs_irreps_str = f"{self.num_elements}x0e"
        init_node_irreps_str = f"{num_features}x0e"

        # Initial node features: scalars only (LinearNodeEmbedding(num_features x 0e)).
        node_feats = LinearNodeEmbedding(
            num_elements=self.num_elements,
            irreps_out=init_node_irreps_str,
        )(Z)
        # One-hot element attributes for the per-element skip in InteractionBlock.
        Z_one_hot = e3nn.IrrepsArray(
            node_attrs_irreps_str,
            jax.nn.one_hot(Z, self.num_elements).astype(dtype),
        )

        prev_irreps_str = init_node_irreps_str
        per_layer_scalars = []
        for k in range(self.num_interactions):
            is_last = k == self.num_interactions - 1
            this_hidden_str = (
                str(e3nn.Irreps([hidden_irreps[0]])) if is_last else self.hidden_irreps
            )
            Block = _INTERACTION_BLOCK_CLS[per_layer_cls[k]]
            # Pass ``name=`` explicitly so the converter sees a uniform
            # ``InteractionBlock_{k}`` path regardless of which variant class
            # is used; otherwise Linen names the block after the class.
            message, sc = Block(
                node_feats_irreps=prev_irreps_str,
                node_attrs_irreps=node_attrs_irreps_str,
                edge_attrs_irreps=sh_irreps_str,
                target_irreps=interaction_irreps_str,
                hidden_irreps=this_hidden_str,
                avg_num_neighbors=self.avg_num_neighbors,
                name=f"InteractionBlock_{k}",
            )(node_feats, sph, radial, Z_one_hot, i, j)
            node_feats = ProductBlock(
                node_feats_irreps=interaction_irreps_str,
                target_irreps=this_hidden_str,
                correlation=self.correlation,
                num_elements=self.num_elements,
                # Density (non-residual) variant returns sc=None.
                use_sc=(sc is not None),
                use_cueq=self.use_cueq,
            )(message, sc, Z)
            per_layer_scalars.append(node_feats.filter(keep="0e").array)
            prev_irreps_str = this_hidden_str

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
