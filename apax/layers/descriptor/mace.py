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

from typing import Any

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from flax import linen as nn

from apax.layers.descriptor.mace_blocks import (
    _INTERACTION_BLOCK_CLS,
    InteractionBlockDensity,
    LinearNodeEmbedding,
    ProductBlock,
)
from apax.utils.convert import str_to_dtype


class MaceRepresentation(nn.Module):
    """MACE descriptor producing per-atom scalar features.

    Parameters
    ----------
    radial_embedding : nn.Module
        Pre-built :class:`apax.layers.descriptor.basis_functions.MaceRadialEmbedding`
        instance, constructed by :class:`apax.nn.builder.MaceBuilder` from
        ``model.basis`` + ``model.radial_embedding`` config groups. Mirrors the
        injection pattern used by GMNN / EquivMP / So3krates.
    distance_transform : nn.Module or None
        Same Linen instance referenced by ``radial_embedding.distance_transform``.
        Held here as a field so its parameter slot lands at
        ``representation/distance_transform/...`` in the apax pytree — the
        slot key the torch→jax converter targets. ``None`` for foundations
        without a transform (small/medium MP-0).
    max_ell : int
        Maximum spherical-harmonic degree used for edge features. Spherical
        harmonics are computed inline here (not in the radial submodule).
    hidden_irreps : str
        e3nn irreps string for node features, e.g. ``"128x0e + 128x1o"``.
        Must include a 0e component.
    correlation : int
        Symmetric-contraction correlation order.
    interactions : tuple[dict, ...]
        Per-layer interaction-block configs. Each element is a discriminated
        dict keyed by ``"name"``; the dispatch table
        :data:`~apax.layers.descriptor.mace_blocks._INTERACTION_BLOCK_CLS`
        resolves the name to a Linen block class.
    avg_num_neighbors : float
        Per-message normaliser forwarded to every interaction block.
    num_elements : int
        Size of the chemical-element embedding table.
    apply_mask : bool
        If ``True``, zero out masked atoms in the output.
    dtype : Any
        Floating-point dtype for features.
    """

    radial_embedding: Any
    distance_transform: Any
    max_ell: int
    hidden_irreps: str
    correlation: int
    interactions: tuple
    avg_num_neighbors: float
    num_elements: int
    apply_mask: bool
    dtype: Any

    @nn.compact
    def __call__(self, dr_vec, Z, idx):
        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)
        i, j = idx[0], idx[1]

        pair_mask = _get_neighbor_mask(idx) if self.apply_mask else 1.0
        node_mask = _get_node_mask(Z) if self.apply_mask else 1.0

        radial = self.radial_embedding(dr_vec, Z, idx)
        if self.apply_mask:
            radial = radial * pair_mask[..., None]

        sh_irreps = e3nn.Irreps.spherical_harmonics(self.max_ell)
        sh_irreps_str = str(sh_irreps)
        sph = e3nn.spherical_harmonics(
            sh_irreps,
            dr_vec,
            normalize=True,
            normalization="component",
        )

        hidden_irreps = e3nn.Irreps(self.hidden_irreps)
        _scalar_irreps_only(self.hidden_irreps)
        num_features = hidden_irreps.count(e3nn.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort().irreps.simplify()
        interaction_irreps_str = str(interaction_irreps)
        node_attrs_irreps_str = f"{self.num_elements}x0e"
        init_node_irreps_str = f"{num_features}x0e"

        node_feats = LinearNodeEmbedding(
            num_elements=self.num_elements,
            irreps_out=init_node_irreps_str,
        )(Z)
        Z_one_hot = e3nn.IrrepsArray(
            node_attrs_irreps_str,
            jax.nn.one_hot(Z, self.num_elements).astype(dtype),
        )

        prev_irreps_str = init_node_irreps_str
        per_layer_scalars = []
        n_layers = len(self.interactions)
        for k, inter_cfg in enumerate(self.interactions):
            is_last = k == n_layers - 1
            this_hidden_str = (
                str(e3nn.Irreps([hidden_irreps[0]])) if is_last else self.hidden_irreps
            )
            Block = _INTERACTION_BLOCK_CLS[inter_cfg["name"]]
            kwargs = dict(
                node_feats_irreps=prev_irreps_str,
                node_attrs_irreps=node_attrs_irreps_str,
                edge_attrs_irreps=sh_irreps_str,
                target_irreps=interaction_irreps_str,
                avg_num_neighbors=self.avg_num_neighbors,
                name=f"InteractionBlock_{k}",
            )
            if Block is not InteractionBlockDensity:
                kwargs["hidden_irreps"] = this_hidden_str
            message, sc = Block(**kwargs)(node_feats, sph, radial, Z_one_hot, i, j)
            node_feats = ProductBlock(
                node_feats_irreps=interaction_irreps_str,
                target_irreps=this_hidden_str,
                correlation=self.correlation,
                num_elements=self.num_elements,
                use_sc=(sc is not None),
                name=f"ProductBlock_{k}",
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
