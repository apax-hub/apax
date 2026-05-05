"""Building blocks for :class:`~apax.layers.descriptor.mace.MaceRepresentation`.

Each block is a Flax linen ``nn.Module`` (or a pure function for stateless
utilities) and is independently testable. Layer order inside
``MaceRepresentation`` is::

    LinearNodeEmbedding → N× (InteractionBlock → ProductBlock) → concat scalar features

Equivariant primitives:
- Irreps / IrrepsArray / tensor products: ``e3nn_jax``
- Symmetric contraction: ``cuequivariance`` / ``cuequivariance_jax``

``cuequivariance-jax`` has a pure-JAX path so blocks work on CPU; CUDA kernels
are an optional acceleration when ``use_cueq=True`` and GPUs are present.
"""

from __future__ import annotations

from functools import lru_cache

import cuequivariance as cue
import cuequivariance_jax as cuex
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction as _cue_mace_symmetric_contraction,
)
from flax import linen as nn

from apax.utils.parity_debug import is_parity_debug_enabled


# torch-mace's :class:`e3nn.nn.FullyConnectedNet` wraps every hidden layer's
# activation in :class:`e3nn.math.normalize2mom`, which estimates the L2 moment
# of the activation under a unit Gaussian via Monte Carlo (1M samples, seed 0).
# For ``torch.nn.functional.silu`` that yields the constant below.  We hard-code
# the exact torch value so the apax radial MLP matches torch bit-for-bit; in
# contrast, e3nn-jax's :func:`normalize_function` uses an ICDF-based
# discretisation that gives ~1.6766 — a ~0.16% drift per activation, which
# compounds across the 3 hidden silu activations to a ~0.5% drift on the radial
# MLP output and ~3% on per-atom forces.  See
# ``e3nn.math._normalize_activation.normalize2mom``.
_TORCH_NORMALIZE2MOM_SILU_CST = 1.6791767923989418


def _silu_torch_normalized(x):
    """Apply ``torch.nn.functional.silu`` * torch's ``normalize2mom`` constant.

    Parameters
    ----------
    x : Array
        Input tensor.

    Returns
    -------
    Array
        ``silu(x) * 1.6791767923989418``, matching torch-mace's
        ``e3nn.nn._fc._Layer`` activation.
    """
    return jax.nn.silu(x) * _TORCH_NORMALIZE2MOM_SILU_CST


class _MaceFullyConnectedNet(nn.Module):
    """Bit-for-bit replica of torch-mace's :class:`e3nn.nn.FullyConnectedNet`.

    Mirrors the per-layer formula::

        w = weight / sqrt(h_in)         # var_in/var_out are 1 throughout MACE
        x = x @ w
        x = silu(x) * 1.6791767...      # only on hidden layers (out_act=False)

    Used as the radial MLP inside :class:`InteractionBlock` so the converted
    weights from ``conv_tp_weights.layerN.weight`` produce the same gating
    coefficients as torch.

    Parameters
    ----------
    list_neurons : tuple of int
        Output sizes of each layer (excluding the input layer).
    """

    list_neurons: tuple

    @nn.compact
    def __call__(self, x):
        n_layers = len(self.list_neurons)
        for i, h_out in enumerate(self.list_neurons):
            # Use a Dense sub-module with no bias so the param tree is
            # ``radial_mlp/Dense_{i}/kernel`` — matches the layout produced by
            # ``e3nn.flax.MultiLayerPerceptron`` (and what
            # :func:`apax.transfer_learning.mace_foundation._map_interactions`
            # already targets).
            h_in = x.shape[-1]
            x_after_w = nn.Dense(
                features=h_out,
                use_bias=False,
                kernel_init=nn.initializers.normal(stddev=1.0),
                param_dtype=x.dtype,
                name=f"Dense_{i}",
            )(x)
            # Same algebra as torch's ``_Layer.forward``: ``x @ (w / sqrt(h_in))``.
            x = x_after_w / jnp.sqrt(jnp.asarray(h_in, x.dtype))
            if i < n_layers - 1:
                x = _silu_torch_normalized(x)
        return x


def assemble_edge_features(dr_vec, r_max, num_bessel, num_poly_cutoff, max_ell):
    """Back-compat shim that wraps :class:`MaceRadialEmbedding` (no transform).

    Existing callers (and tests) construct radial features without a
    distance transform; this thin wrapper preserves their signature by
    instantiating :class:`MaceRadialEmbedding` with
    ``distance_transform=None`` and applying it to ``dr_vec``. The dummy
    ``Z`` / ``idx`` arrays are never consumed because the no-transform path
    does not read them.

    Parameters
    ----------
    dr_vec : Array, shape (n_edges, 3)
        Relative position vectors ``r_j - r_i`` for each neighbor pair.
    r_max : float
        Interaction cutoff in the same units as ``dr_vec``.
    num_bessel : int
        Number of Bessel radial basis functions.
    num_poly_cutoff : int
        Polynomial order of the smooth envelope cutoff.
    max_ell : int
        Maximum spherical-harmonic degree (inclusive). Output irreps are
        ``spherical_harmonics(max_ell)`` which is ``0e + 1o + ... + max_ell(e|o)``.

    Returns
    -------
    radial : Array, shape (n_edges, num_bessel)
        Bessel basis multiplied element-wise by the polynomial cutoff envelope.
        Zero for ``|r| >= r_max``.
    sph : e3nn_jax.IrrepsArray
        Spherical harmonics of ``dr_vec``, normalized on the unit sphere with
        the ``"component"`` normalization that MACE uses.

    Notes
    -----
    The dtype of both outputs matches the dtype of ``dr_vec``.
    """
    from apax.layers.descriptor.basis_functions import (
        MaceBesselBasis,
        MaceRadialEmbedding,
    )

    basis_fn = MaceBesselBasis(
        n_basis=num_bessel, r_max=r_max, dtype=dr_vec.dtype,
    )
    module = MaceRadialEmbedding(
        basis_fn=basis_fn,
        num_polynomial_cutoff=num_poly_cutoff,
        r_max=r_max,
        distance_transform=None,
    )
    Z_dummy = jnp.zeros((1,), dtype=jnp.int32)
    idx_dummy = jnp.zeros((2, dr_vec.shape[0]), dtype=jnp.int32)
    radial = module.apply({}, dr_vec, Z_dummy, idx_dummy)
    sph = e3nn.spherical_harmonics(
        e3nn.Irreps.spherical_harmonics(max_ell),
        dr_vec,
        normalize=True,
        normalization="component",
    )
    return radial, sph


class LinearNodeEmbedding(nn.Module):
    """One-hot element embedding followed by an irreps-linear.

    Produces node features in ``irreps_out``. Initial features are pure
    scalars (parity even), so ``irreps_out`` must contain only ``0e`` components.
    """

    num_elements: int
    irreps_out: str  # must be scalar irreps (e.g. "128x0e")

    @nn.compact
    def __call__(self, Z):
        irreps = e3nn.Irreps(self.irreps_out).filter("0e")
        one_hot = jax.nn.one_hot(Z, self.num_elements)  # (n, E)
        # Linear projection E -> irreps.dim, wrapped as IrrepsArray
        w = self.param(
            "weight",
            nn.initializers.normal(stddev=1.0 / jnp.sqrt(self.num_elements)),
            (self.num_elements, irreps.dim),
        )
        feats = one_hot @ w  # (n, irreps.dim)
        return e3nn.IrrepsArray(irreps, feats)


def tp_out_irreps_with_instructions(irreps_in1, irreps_in2, target_irreps):
    """Compute the simplified intersection of ``irreps_in1 ⊗ irreps_in2`` with ``target_irreps``.

    Adapted (port) from
    ``mace_jax/modules/irreps_tools.py:tp_out_irreps_with_instructions``
    (MIT-licensed). Used by :class:`InteractionBlock` to build the
    ``conv_tp`` tensor-product graph: ``irreps_mid`` is the set of output
    irreps reachable from ``in1 ⊗ in2`` and present in ``target_irreps``,
    sorted to allow simplification by the downstream ``Linear``.

    Parameters
    ----------
    irreps_in1 : e3nn.Irreps or str
        First operand irreps (e.g. node-feature irreps).
    irreps_in2 : e3nn.Irreps or str
        Second operand irreps (e.g. spherical-harmonic irreps).
    target_irreps : e3nn.Irreps or str
        Filter for the output irreps; only paths whose ``ir_out`` is in
        ``target_irreps`` are kept.

    Returns
    -------
    irreps_mid : e3nn.Irreps
        The collapsed intermediate irreps reachable in ``target_irreps``,
        sorted by ``(l, p, mul)``.
    instructions : list[tuple]
        e3nn-style ``(i_in1, i_in2, i_out, "uvu", trainable=True)`` paths
        with output indices permuted to match the sorted ``irreps_mid``.
    """
    irreps_in1 = e3nn.Irreps(irreps_in1)
    irreps_in2 = e3nn.Irreps(irreps_in2)
    target_irreps = e3nn.Irreps(target_irreps)
    trainable = True

    irreps_out_list: list[tuple[int, e3nn.Irrep]] = []
    instructions: list[tuple] = []
    for i, (mul, ir_in) in enumerate(irreps_in1):
        for j, (_, ir_edge) in enumerate(irreps_in2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    irreps_mid = e3nn.Irreps(irreps_out_list)
    irreps_mid, permut, _ = irreps_mid.sort()
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]
    instructions = sorted(instructions, key=lambda x: x[2])
    return irreps_mid, instructions


def _interaction_scaffold(
    node_feats,
    edge_attrs,
    edge_feats,
    receivers,
    senders,
    *,
    node_feats_irreps: e3nn.Irreps,
    edge_attrs_irreps: e3nn.Irreps,
    target_irreps: e3nn.Irreps,
    radial_mlp: tuple,
    sow_fn=None,
):
    """Common interaction-block prefix shared by all three variants.

    Runs ``linear_up → conv_tp → radial-MLP gate → scatter-sum → linear``.
    Returns the un-normalised aggregated message in ``target_irreps`` so each
    variant can divide by either ``avg_num_neighbors`` (Residual) or
    ``density + 1`` (Density variants).

    Submodules are created with literal names (``linear_up``, ``radial_mlp``,
    ``linear``) so the calling block's param tree path matches torch-mace's
    ``state_dict`` segments exactly. This keeps the converter's per-block
    mapping logic identical across all three variants.

    Parameters
    ----------
    node_feats : e3nn.IrrepsArray
        Per-atom node features.
    edge_attrs : e3nn.IrrepsArray
        Per-edge spherical harmonics.
    edge_feats : Array
        Per-edge radial basis × cutoff features.
    receivers : Array
        Per-edge receiver indices.
    senders : Array
        Per-edge sender indices.
    node_feats_irreps, edge_attrs_irreps, target_irreps : e3nn.Irreps
        Pre-parsed irreps. ``target_irreps`` is the multi-irrep message target
        (typically ``(sh_irreps × num_features).sort().simplify()``).
    radial_mlp : tuple
        Hidden widths of the radial MLP gating tensor-product channels.

    Returns
    -------
    e3nn.IrrepsArray
        Aggregated message in ``target_irreps``, before any per-message
        normalisation.
    """
    x = e3nn.flax.Linear(node_feats_irreps, name="linear_up")(node_feats)
    if sow_fn is not None:
        sow_fn("linear_up", x.array)
    irreps_mid, _instructions = tp_out_irreps_with_instructions(
        node_feats_irreps, edge_attrs_irreps, target_irreps,
    )
    x_j = x[senders]
    tp = e3nn.tensor_product(x_j, edge_attrs, filter_ir_out=irreps_mid)

    # Per-edge per-path scalar gates from the radial MLP.  We use the
    # bit-for-bit ``_MaceFullyConnectedNet`` rather than e3nn-jax's
    # MultiLayerPerceptron so the silu normalisation constant matches torch.
    n_paths = tp.irreps.num_irreps
    weights = _MaceFullyConnectedNet(
        list_neurons=tuple(radial_mlp) + (n_paths,), name="radial_mlp",
    )(edge_feats)
    weighted = tp * weights
    if sow_fn is not None:
        sow_fn("conv_tp", weighted.array)

    agg = e3nn.scatter_sum(weighted, dst=receivers, output_size=node_feats.shape[0])
    out = e3nn.flax.Linear(target_irreps, name="linear")(agg)
    if sow_fn is not None:
        sow_fn("linear", out.array)
    return out


def _edge_density(edge_feats, receivers, n_atoms):
    """Per-atom density gate used by the Density / DensityResidual variants.

    Mirrors torch-mace's ``edge_density = tanh(density_fn(edge_feats) ** 2)``
    followed by a scatter-sum into receivers. The ``density_fn`` is a single
    Dense layer with no activation (matches torch's
    ``FullyConnectedNet([input_dim, 1], silu)`` — with only one layer the
    activation never fires; see torch's ``_Layer.forward``).

    Parameters
    ----------
    edge_feats : Array, shape (n_edges, num_bessel)
        Bessel × cutoff features.
    receivers : Array, shape (n_edges,)
        Per-edge receiver index.
    n_atoms : int
        Number of receivers (sets ``output_size``).

    Returns
    -------
    e3nn.IrrepsArray with irreps ``"1x0e"``
        Per-atom density gate of shape ``(n_atoms, 1)``.
    """
    edge_density = jnp.tanh(
        _MaceFullyConnectedNet(list_neurons=(1,), name="density_fn")(edge_feats) ** 2
    )
    return e3nn.scatter_sum(
        e3nn.IrrepsArray("0e", edge_density), dst=receivers, output_size=n_atoms,
    )


class InteractionBlockResidual(nn.Module):
    """MACE Residual interaction block — multi-irrep target with per-element skip.

    Mirrors torch-mace ``RealAgnosticResidualInteractionBlock``. Internally
    runs ``linear_up → tensor_product(node_feats × sph) → radial-weighted
    scatter sum → linear / avg_num_neighbors`` and computes a parallel per-element
    skip ``Linear(node_feats × node_attrs) → hidden_irreps`` (the
    ``FullyConnectedTensorProduct`` factorises into an irreps-Linear over the
    tensor product because ``node_attrs`` are scalars). Returns the tuple
    ``(message, sc)`` so the downstream :class:`ProductBlock` can apply the
    skip after the symmetric contraction.

    Parameters
    ----------
    node_feats_irreps : str
        Input node-feature irreps (e.g. ``"128x0e"`` for the first layer,
        ``hidden_irreps`` for subsequent layers).
    node_attrs_irreps : str
        One-hot element-attribute irreps (``"<num_elements>x0e"``).
    edge_attrs_irreps : str
        Spherical-harmonics irreps (``Irreps.spherical_harmonics(max_ell)``).
    target_irreps : str
        Multi-irrep target for ``message`` — typically
        ``(sh_irreps * num_features).sort().simplify()``.
    hidden_irreps : str
        Target irreps for the per-element skip ``sc``. Matches the post-product
        ``this_layer_hidden`` of :class:`MaceRepresentation`.
    radial_mlp : tuple
        Hidden widths of the radial MLP gating tensor-product channels.
    avg_num_neighbors : float
        Per-atom message normaliser ``message = linear(agg) / avg_num_neighbors``
        applied after the post-aggregation ``Linear``. Mirrors torch-mace
        (``mace.modules.blocks.RealAgnosticResidualInteractionBlock`` divides
        by the same scalar). Defaults to ``1.0`` so freshly built apax models
        without a precomputed average are unaffected.

    Returns
    -------
    message : e3nn.IrrepsArray
        Aggregated message in ``target_irreps``.
    sc : e3nn.IrrepsArray
        Per-element skip in ``hidden_irreps``.
    """

    node_feats_irreps: str
    node_attrs_irreps: str
    edge_attrs_irreps: str
    target_irreps: str
    hidden_irreps: str
    radial_mlp: tuple = (64, 64, 64)
    avg_num_neighbors: float = 1.0
    layer_idx: int = -1  # Must be set by the parent module for sow naming.

    @nn.compact
    def __call__(
        self, node_feats, edge_attrs, edge_feats, node_attrs, receivers, senders,
    ):
        if self.layer_idx < 0:
            raise ValueError(
                f"{type(self).__name__} requires layer_idx to be set by the parent module"
            )
        node_feats_irreps = e3nn.Irreps(self.node_feats_irreps)
        edge_attrs_irreps = e3nn.Irreps(self.edge_attrs_irreps)
        target_irreps = e3nn.Irreps(self.target_irreps)
        hidden_irreps = e3nn.Irreps(self.hidden_irreps)

        def _sow(slot, arr):
            # Gated so plain ``model.init`` doesn't sprout a ``debug`` branch.
            if is_parity_debug_enabled():
                self.sow("debug", f"interactions[{self.layer_idx}].{slot}", arr)

        message = _interaction_scaffold(
            node_feats, edge_attrs, edge_feats, receivers, senders,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            target_irreps=target_irreps,
            radial_mlp=self.radial_mlp,
            sow_fn=_sow,
        )
        message = message / self.avg_num_neighbors

        # skip_tp: per-element FCTP (node_feats × node_attrs → hidden_irreps).
        # Since node_attrs is scalar-only, the FCTP is equivalent to
        # Linear(tensor_product(node_feats, node_attrs)) — force_irreps_out
        # zero-pads channels not reachable from the input.
        skip_input = e3nn.tensor_product(node_feats, node_attrs)
        sc = e3nn.flax.Linear(
            hidden_irreps, name="skip_tp", force_irreps_out=True,
        )(skip_input)
        _sow("skip_tp", sc.array)
        return message, sc


# Back-compat alias: the existing ``InteractionBlock`` symbol is referenced by
# tests, the converter (``rep_params["InteractionBlock_{k}"]``), and external
# imports. Keep it pointing at the Residual variant so existing param trees
# round-trip unchanged.
InteractionBlock = InteractionBlockResidual


class InteractionBlockDensity(nn.Module):
    """MACE Density interaction block (non-residual).

    Mirrors torch-mace ``RealAgnosticDensityInteractionBlock``
    (``mace/modules/blocks.py:745-862``). Differs from
    :class:`InteractionBlockResidual` in two ways:

    1. The per-message normaliser is ``density + 1`` rather than
       ``avg_num_neighbors``, where ``density`` is the per-atom scatter-sum of
       ``tanh(density_fn(edge_feats) ** 2)``.
    2. ``skip_tp`` is applied to the post-density message in
       ``target_irreps`` (not to raw ``node_feats``); there is no separate
       ``sc`` returned. The signature is
       ``FullyConnectedTensorProduct(target_irreps, node_attrs, target_irreps)``.

    Returns ``(message, None)``; the downstream :class:`ProductBlock` is
    constructed with ``use_sc=False``.
    """

    node_feats_irreps: str
    node_attrs_irreps: str
    edge_attrs_irreps: str
    target_irreps: str
    # Unused — kept for parity with InteractionBlockResidual signature so the
    # MaceRepresentation dispatch can pass identical kwargs to every variant.
    hidden_irreps: str = ""
    radial_mlp: tuple = (64, 64, 64)
    avg_num_neighbors: float = 1.0  # ignored; mirrors torch (no /avg)
    layer_idx: int = -1  # Must be set by the parent module for sow naming.

    @nn.compact
    def __call__(
        self, node_feats, edge_attrs, edge_feats, node_attrs, receivers, senders,
    ):
        if self.layer_idx < 0:
            raise ValueError(
                f"{type(self).__name__} requires layer_idx to be set by the parent module"
            )
        node_feats_irreps = e3nn.Irreps(self.node_feats_irreps)
        edge_attrs_irreps = e3nn.Irreps(self.edge_attrs_irreps)
        target_irreps = e3nn.Irreps(self.target_irreps)

        def _sow(slot, arr):
            # Gated so plain ``model.init`` doesn't sprout a ``debug`` branch.
            if is_parity_debug_enabled():
                self.sow("debug", f"interactions[{self.layer_idx}].{slot}", arr)

        pre = _interaction_scaffold(
            node_feats, edge_attrs, edge_feats, receivers, senders,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            target_irreps=target_irreps,
            radial_mlp=self.radial_mlp,
            sow_fn=_sow,
        )
        density = _edge_density(edge_feats, receivers, node_feats.shape[0])
        # density is an IrrepsArray("0e", (n_atoms, 1)); broadcast over message irreps.
        message = pre / (density.array + 1.0)

        # skip_tp: target_irreps × node_attrs_irreps → target_irreps. Operates on
        # the post-density message (not raw node_feats), so there is no separate
        # ``sc`` returned.
        skip_input = e3nn.tensor_product(message, node_attrs)
        message = e3nn.flax.Linear(
            target_irreps, name="skip_tp", force_irreps_out=True,
        )(skip_input)
        _sow("skip_tp", message.array)
        return message, None


class InteractionBlockDensityResidual(nn.Module):
    """MACE Density interaction block with parent-style residual skip.

    Mirrors torch-mace ``RealAgnosticDensityResidualInteractionBlock``
    (``mace/modules/blocks.py:866-988``). Combines the Density per-message
    normaliser (``/(density + 1)``) with the Residual block's per-element skip
    computed up front from raw ``node_feats``:
    ``skip_tp = FullyConnectedTensorProduct(node_feats_irreps, node_attrs_irreps,
    hidden_irreps)``.

    Returns ``(message, sc)`` exactly like :class:`InteractionBlockResidual`;
    the downstream :class:`ProductBlock` runs with ``use_sc=True``.
    """

    node_feats_irreps: str
    node_attrs_irreps: str
    edge_attrs_irreps: str
    target_irreps: str
    hidden_irreps: str
    radial_mlp: tuple = (64, 64, 64)
    avg_num_neighbors: float = 1.0  # ignored; mirrors torch (no /avg)
    layer_idx: int = -1  # Must be set by the parent module for sow naming.

    @nn.compact
    def __call__(
        self, node_feats, edge_attrs, edge_feats, node_attrs, receivers, senders,
    ):
        if self.layer_idx < 0:
            raise ValueError(
                f"{type(self).__name__} requires layer_idx to be set by the parent module"
            )
        node_feats_irreps = e3nn.Irreps(self.node_feats_irreps)
        edge_attrs_irreps = e3nn.Irreps(self.edge_attrs_irreps)
        target_irreps = e3nn.Irreps(self.target_irreps)
        hidden_irreps = e3nn.Irreps(self.hidden_irreps)

        def _sow(slot, arr):
            # Gated so plain ``model.init`` doesn't sprout a ``debug`` branch.
            if is_parity_debug_enabled():
                self.sow("debug", f"interactions[{self.layer_idx}].{slot}", arr)

        # Skip is computed from raw node_feats BEFORE linear_up — same shape
        # and placement as the Residual variant.
        skip_input = e3nn.tensor_product(node_feats, node_attrs)
        sc = e3nn.flax.Linear(
            hidden_irreps, name="skip_tp", force_irreps_out=True,
        )(skip_input)
        _sow("skip_tp", sc.array)

        pre = _interaction_scaffold(
            node_feats, edge_attrs, edge_feats, receivers, senders,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            target_irreps=target_irreps,
            radial_mlp=self.radial_mlp,
            sow_fn=_sow,
        )
        density = _edge_density(edge_feats, receivers, node_feats.shape[0])
        message = pre / (density.array + 1.0)
        return message, sc


# Maps :attr:`MaceModelConfig.interaction_cls` literal to the Linen module.
# Single source of truth used by :class:`MaceRepresentation` dispatch. New
# variants are added here and to the ``Literal`` in ``MaceModelConfig``.
_INTERACTION_BLOCK_CLS = {
    "RealAgnosticResidual": InteractionBlockResidual,
    "RealAgnosticDensity": InteractionBlockDensity,
    "RealAgnosticDensityResidual": InteractionBlockDensityResidual,
}


class LinearReadoutBlock(nn.Module):
    """Single-atom linear readout: IrrepsArray -> (n_out,).

    Mirrors torch-mace's :class:`LinearReadoutBlock`: a single
    ``e3nn.flax.Linear`` projecting per-atom node features down to
    ``n_out x 0e``. Designed to be vmapped over atoms inside
    :class:`~apax.layers.descriptor.mace.MaceReadout`.

    Parameters
    ----------
    n_out : int
        Number of scalar output channels. ``1`` (default) gives a single
        per-atom energy; values ``>1`` support shallow ensembles.
    """

    n_out: int = 1

    @nn.compact
    def __call__(self, feat):
        """Apply linear readout to a single-atom feature vector.

        Parameters
        ----------
        feat : e3nn.IrrepsArray
            Per-atom node features (no batch dimension).

        Returns
        -------
        jnp.ndarray, shape (n_out,)
            Scalar output for this atom.
        """
        out = e3nn.flax.Linear(f"{self.n_out}x0e", name="linear")(feat)
        return out.array


class NonLinearReadoutBlock(nn.Module):
    """Single-atom non-linear readout: Linear -> SiLU -> Linear -> (n_out,).

    Mirrors torch-mace's :class:`NonLinearReadoutBlock`:
    ``linear_1 -> silu -> linear_2``. The hidden layer is scalar-only for the
    foundation models considered here (``MLP_irreps = "16x0e"``). Designed to
    be vmapped over atoms inside
    :class:`~apax.layers.descriptor.mace.MaceReadout`.

    Parameters
    ----------
    MLP_irreps : str
        Hidden-layer irreps (scalar-only), e.g. ``"16x0e"``.
    silu_normalization : float
        Multiplicative constant applied after SiLU to match torch-e3nn's
        ``normalize2mom`` wrapper on ``torch.nn.functional.silu``. Default is
        the exact numerical value ``1.6791767923989418`` — required for
        bit-for-bit parity with torch-mace foundation readouts.
    n_out : int
        Number of scalar output channels. ``1`` (default) gives a single
        per-atom energy; values ``>1`` support shallow ensembles.
    """

    MLP_irreps: str = "16x0e"
    silu_normalization: float = 1.6791767923989418
    n_out: int = 1

    @nn.compact
    def __call__(self, feat):
        """Apply non-linear readout to a single-atom feature vector.

        Parameters
        ----------
        feat : e3nn.IrrepsArray
            Per-atom node features (no batch dimension).

        Returns
        -------
        jnp.ndarray, shape (n_out,)
            Scalar output for this atom.
        """
        irreps_hidden = e3nn.Irreps(self.MLP_irreps)
        if not all(ir.l == 0 and ir.p == 1 for _, ir in irreps_hidden):
            raise NotImplementedError(
                "NonLinearReadoutBlock only supports scalar hidden irreps "
                f"(got {self.MLP_irreps!r})."
            )
        x = e3nn.flax.Linear(self.MLP_irreps, name="linear_1")(feat)
        x = e3nn.IrrepsArray(x.irreps, jax.nn.silu(x.array) * self.silu_normalization)
        out = e3nn.flax.Linear(f"{self.n_out}x0e", name="linear_2")(x)
        return out.array


class ScaleShift(nn.Module):
    """Per-atom affine ``scale * x + shift``.

    Mirrors torch-mace's :class:`ScaleShiftBlock`. Both parameters are scalars
    (single head); the multi-head case is out of scope for the small MP-0
    parity milestone.
    """

    scale_init: float = 1.0
    shift_init: float = 0.0

    @nn.compact
    def __call__(self, x):
        scale = self.param(
            "scale", lambda rng: jnp.array(self.scale_init, dtype=x.dtype)
        )
        shift = self.param(
            "shift", lambda rng: jnp.array(self.shift_init, dtype=x.dtype)
        )
        return scale * x + shift


class ProductBlock(nn.Module):
    """MACE product block: symmetric contraction → Linear → optional ``+ sc``.

    Mirrors torch-mace ``EquivariantProductBasisBlock``. Reduces a multi-irrep
    input ``node_feats_irreps`` to the post-product ``target_irreps`` via
    per-element symmetric contraction (cuequivariance), then applies a final
    irreps ``Linear`` and optionally adds the per-element skip ``sc`` from the
    parent :class:`InteractionBlock`.

    Parameters
    ----------
    node_feats_irreps : str
        Input irreps — typically :attr:`InteractionBlock.target_irreps`.
        All entries must share a common multiplicity.
    target_irreps : str
        Output irreps — typically ``this_layer_hidden`` from
        :class:`MaceRepresentation` (last-layer collapse-aware).
    correlation : int
        Maximum tensor-product order (MACE commonly uses 3).
    num_elements : int
        Number of chemical elements (per-species weight table row count).
    use_sc : bool
        Whether to add the parent skip ``sc`` after the post-SC Linear.
    use_cueq : bool
        Reserved for P2 (CUDA kernel dispatch). For now, all paths go through
        :func:`cuex.equivariant_polynomial` with ``method='naive'``.

    Notes
    -----
    The descriptor built by
    :func:`cuequivariance.group_theory.experimental.mace.symmetric_contractions.symmetric_contraction`
    is static (does not depend on any traced values), so it is built lazily per
    configuration via a module-level ``lru_cache`` keyed on
    ``(node_feats_irreps, target_irreps, correlation)``. Weight initialisation
    uses ``normal(stddev=1.0)``; parity with torch-mace is achieved by the
    state-dict mapper in :mod:`apax.transfer_learning.mace_foundation`.
    """

    node_feats_irreps: str
    target_irreps: str
    correlation: int = 3
    num_elements: int = 119
    use_sc: bool = True
    use_cueq: bool = False
    layer_idx: int = -1  # Must be set by the parent module for sow naming.

    @nn.compact
    def __call__(self, node_feats, sc, Z):
        if self.layer_idx < 0:
            raise ValueError(
                f"{type(self).__name__} requires layer_idx to be set by the parent module"
            )
        if self.use_cueq:
            raise NotImplementedError(
                "use_cueq=True is reserved for P2 (cuequivariance CUDA dispatch); "
                "not yet implemented."
            )
        irreps_in_e3 = e3nn.Irreps(self.node_feats_irreps)
        irreps_out_e3 = e3nn.Irreps(self.target_irreps)

        out_muls = {mul for mul, _ in irreps_out_e3}
        in_muls = {mul for mul, _ in irreps_in_e3}
        if len(out_muls | in_muls) != 1:
            raise ValueError(
                "ProductBlock requires all input/output irreps to share the same multiplicity; "
                f"got input {irreps_in_e3!r} output {irreps_out_e3!r}"
            )
        mul = next(iter(out_muls | in_muls))

        # Build the (static) descriptor. The cache keeps jit-recompilation cheap.
        descriptor, weight_irreps, weight_numel = _get_symmetric_contraction_descriptor(
            str(irreps_in_e3),
            str(irreps_out_e3),
            int(self.correlation),
        )
        weight_basis_dim = weight_numel // mul

        array = node_feats.array
        dtype = array.dtype
        n_atoms = array.shape[0]
        feature_dim_in = sum(ir.dim for _, ir in irreps_in_e3)
        expected_total = mul * feature_dim_in
        if array.shape[-1] != expected_total:
            raise ValueError(
                "ProductBlock expected a flat feature dim of "
                f"{expected_total} (mul={mul} * feature_dim_in={feature_dim_in}); "
                f"got {array.shape[-1]}"
            )

        # The flat e3nn ``mul_ir`` layout stores each ``(mul, ir)`` chunk as a
        # contiguous ``mul * ir.dim`` block in mul-major order:
        # ``[m0_d0, m0_d1, ..., m0_d_{ir.dim-1}, m1_d0, ...]``.  A single
        # ``reshape(n_atoms, mul, feature_dim_in)`` would treat the flat axis as
        # row-major ``(mul, feature_dim_in)`` which interleaves irreps incorrectly.
        # We instead slice per-irrep, reshape to ``(n_atoms, mul, ir.dim)`` and
        # concatenate along the last axis to obtain the canonical
        # ``(n_atoms, mul, feature_dim_in)`` tensor expected by
        # :func:`_features_to_rep`.
        offset = 0
        per_irrep = []
        for mul_i, ir in irreps_in_e3:
            block = array[..., offset : offset + mul_i * ir.dim]
            offset += mul_i * ir.dim
            per_irrep.append(block.reshape(n_atoms, mul_i, ir.dim))
        x_mul_ir = jnp.concatenate(per_irrep, axis=-1)

        # Per-element weight table.
        weight = self.param(
            "weight",
            nn.initializers.normal(stddev=1.0),
            (self.num_elements, weight_basis_dim, mul),
            dtype,
        )
        weight_flat = weight.reshape(self.num_elements, weight_numel)
        selected = weight_flat[Z.astype(jnp.int32)]  # (n_atoms, weight_numel)
        weight_rep = cuex.RepArray(weight_irreps, selected, cue.ir_mul)

        # Wrap the node features as a RepArray in ir_mul layout.
        irreps_in_cue = cue.Irreps(cue.O3, str(irreps_in_e3))
        x_rep = _features_to_rep(x_mul_ir, irreps_in_cue, mul, dtype)

        out_rep = cuex.equivariant_polynomial(
            descriptor,
            [weight_rep, x_rep],
            math_dtype=dtype,
            method="naive",
        )

        out_ir_mul = out_rep.change_layout(cue.ir_mul).array
        out_mul_ir = _ir_mul_to_mul_ir(out_ir_mul, irreps_out_e3)
        out = e3nn.IrrepsArray(irreps_out_e3, out_mul_ir)

        # Post-SC Linear matches torch's products.k.linear.weight slot.
        out = e3nn.flax.Linear(irreps_out_e3, name="linear")(out)

        if self.use_sc and sc is not None:
            out = out + sc
        # Gated so plain ``model.init`` doesn't sprout a ``debug`` branch.
        if is_parity_debug_enabled():
            self.sow("debug", f"products[{self.layer_idx}]", out.array)
        return out


# ---------------------------------------------------------------------------
# Private helpers (ported from mace-jax's cuequivariance adapter utilities)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _get_symmetric_contraction_descriptor(
    input_irreps_str: str, output_irreps_str: str, correlation: int,
):
    """Cache the MACE symmetric-contraction descriptor build.

    Parameters
    ----------
    input_irreps_str : str
        String form of the node-feature irreps (input to the contraction).
    output_irreps_str : str
        String form of the target irreps (output of the contraction).
    correlation : int
        Maximum tensor-product order; degrees ``range(1, correlation + 1)``.

    Returns
    -------
    descriptor : cuequivariance descriptor
        The polynomial descriptor passed to
        :func:`cuex.equivariant_polynomial`.
    weight_irreps : cue.Irreps
        The cue irreps of the weight input.
    weight_numel : int
        Total size of the flattened weight vector per element.

    Notes
    -----
    The underlying cuequivariance helper also returns a projection matrix used
    only by the ``use_reduced_cg=False`` path. That path is deferred to a
    future milestone; the projection is discarded here and the cache
    signature kept minimal.
    """
    irreps_in_cue = cue.Irreps(cue.O3, input_irreps_str)
    irreps_out_cue = cue.Irreps(cue.O3, output_irreps_str)
    degrees = tuple(range(1, correlation + 1))
    descriptor, _projection = _cue_mace_symmetric_contraction(
        irreps_in_cue, irreps_out_cue, degrees
    )
    weight_irreps = descriptor.inputs[0].irreps
    weight_numel = weight_irreps.dim
    return descriptor, weight_irreps, weight_numel


def _features_to_rep(
    x_mul_ir: jnp.ndarray,
    irreps_cue: cue.Irreps,
    mul: int,
    dtype: jnp.dtype,
) -> cuex.RepArray:
    """Pack ``mul_ir`` features into a cuequivariance ``RepArray``.

    Parameters
    ----------
    x_mul_ir : jnp.ndarray
        Feature tensor of shape ``(batch, mul, feature_dim)`` in the e3nn
        ``mul_ir`` layout.
    irreps_cue : cue.Irreps
        cue irreps of the input (full multiplicity).
    mul : int
        Common multiplicity shared by every irrep entry.
    dtype : jnp.dtype
        Desired dtype of the resulting ``RepArray``.

    Returns
    -------
    cuex.RepArray
        Features in ``ir_mul`` layout, segmented per irrep, ready to be passed
        to :func:`cuex.equivariant_polynomial`.
    """
    base_irreps = irreps_cue.set_mul(1)
    segments: list[jnp.ndarray] = []
    offset = 0
    for mul_ir in base_irreps:
        width = mul_ir.ir.dim
        seg = x_mul_ir[:, :, offset : offset + width]
        # swap (..., mul, ir_dim) -> (..., ir_dim, mul)
        segments.append(jnp.swapaxes(seg, -2, -1))
        offset += width
    return cuex.from_segments(
        irreps_cue,
        segments,
        (x_mul_ir.shape[0], mul),
        cue.ir_mul,
        dtype=dtype,
    )


def _ir_mul_to_mul_ir(array: jnp.ndarray, irreps: e3nn.Irreps) -> jnp.ndarray:
    """Reorder the last axis of ``array`` from ``ir_mul`` back to ``mul_ir``.

    Parameters
    ----------
    array : jnp.ndarray
        Array whose last axis equals ``irreps.dim`` in cue's ``ir_mul`` order.
    irreps : e3nn.Irreps
        Irreps describing the last axis.

    Returns
    -------
    jnp.ndarray
        Array of the same shape, last axis reordered to e3nn's ``mul_ir``.
    """
    if irreps.dim == 0:
        return array
    leading_shape = array.shape[:-1]
    array = array.reshape(*leading_shape, irreps.dim)
    segments: list[jnp.ndarray] = []
    offset = 0
    for mul, ir in irreps:
        block = array[..., offset : offset + mul * ir.dim]
        offset += mul * ir.dim
        block = block.reshape(*leading_shape, ir.dim, mul)
        block = jnp.swapaxes(block, -1, -2)  # -> (..., mul, ir_dim)
        block = block.reshape(*leading_shape, mul * ir.dim)
        segments.append(block)
    return jnp.concatenate(segments, axis=-1) if segments else array
