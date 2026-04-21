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


def assemble_edge_features(dr_vec, r_max, num_bessel, num_poly_cutoff, max_ell):
    """Compute per-edge radial × cutoff basis and spherical harmonics.

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
    from apax.layers.descriptor.basis_functions import BesselBasis, PolynomialCutoff

    dtype = dr_vec.dtype
    r_ij = jnp.linalg.norm(dr_vec, axis=-1)

    bessel_module = BesselBasis(n_basis=num_bessel, r_max=r_max, dtype=dtype)
    bessel = bessel_module.apply({}, r_ij)

    cutoff_module = PolynomialCutoff(p=num_poly_cutoff, r_max=r_max)
    cutoff = cutoff_module.apply({}, r_ij)

    radial = (bessel * cutoff[..., None]).astype(dtype)

    irreps = e3nn.Irreps.spherical_harmonics(max_ell)
    sph = e3nn.spherical_harmonics(
        irreps,
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


# InteractionBlock (RealAgnosticResidual) — port notes
# Inputs:  node_feats [n_atoms, irreps_in], edge_attrs (sph) [n_edges, Ylm],
#          edge_feats (radial) [n_edges, n_bessel], i (receivers), j (senders), pair_mask
# Layout:
#   1. source_linear:    node_feats[j] -> irreps_in (e3nn Linear)
#   2. conv_tp:          source @ edge_attrs via FullyConnectedTensorProduct
#                        with per-edge MLP weights from radial features
#   3. scatter_sum:      aggregate messages into receiver index i
#   4. target_linear:    aggregated -> irreps_out (e3nn Linear)
#   5. residual:         output + skip connection from original node_feats
# Output: new node_feats [n_atoms, irreps_out]
class InteractionBlock(nn.Module):
    """MACE interaction + residual.

    RealAgnosticResidual variant: one tensor product between node features and
    edge (spherical-harmonic) attributes, weighted by a radial MLP, aggregated
    into receivers via scatter-sum, plus a skip connection.

    Parameters
    ----------
    irreps_out : str
        Target irreps for node features after this block.
    interaction_cls : str
        Which interaction variant; for now only RealAgnosticResidual is
        implemented. Other variants raise NotImplementedError.
    radial_mlp : tuple[int, ...]
        Hidden layer widths for the radial MLP that gates tensor-product
        channels. Defaults to ``(64, 64, 64)``.
    """

    irreps_out: str
    interaction_cls: str = "RealAgnosticResidual"
    radial_mlp: tuple = (64, 64, 64)

    @nn.compact
    def __call__(self, node_feats, edge_attrs, edge_feats, receivers, senders):
        if self.interaction_cls != "RealAgnosticResidual":
            raise NotImplementedError(
                f"Interaction variant {self.interaction_cls!r} not yet implemented; "
                "only 'RealAgnosticResidual' is supported at this phase."
            )
        irreps_in = node_feats.irreps
        irreps_out = e3nn.Irreps(self.irreps_out)

        # 1. Linear pre-mix
        x = e3nn.flax.Linear(irreps_in, name="linear_up")(node_feats)

        # 2. Gather source node features at senders
        x_j = x[senders]

        # 3. Tensor product with edge spherical harmonics
        #    Output irreps = full tensor-square subset reachable in irreps_out
        tp = e3nn.tensor_product(x_j, edge_attrs, filter_ir_out=irreps_out)

        # 4. Radial MLP producing a scalar per TP path per edge
        n_paths = tp.irreps.num_irreps
        mlp_widths = (*self.radial_mlp, n_paths)
        weights = e3nn.flax.MultiLayerPerceptron(
            list(mlp_widths), act=jax.nn.silu, name="radial_mlp"
        )(edge_feats)
        weighted = tp * weights                                 # broadcast-safe

        # 5. Scatter-sum into receivers
        out = e3nn.scatter_sum(weighted, dst=receivers, output_size=node_feats.shape[0])

        # 6. Post-mix and residual.
        # force_irreps_out=True on skip_linear zero-pads channels unreachable
        # from node_feats.irreps (e.g. the first layer has scalar-only input
        # so 1o/2e channels can't arise via a pure Linear), keeping the sum
        # with the tensor-product path shape-consistent across layers.
        out = e3nn.flax.Linear(irreps_out, name="linear_down")(out)
        skip = e3nn.flax.Linear(
            irreps_out, name="skip_linear", force_irreps_out=True
        )(node_feats)
        return out + skip


class LinearReadoutBlock(nn.Module):
    """Scalar linear readout.

    Mirrors torch-mace's :class:`LinearReadoutBlock`: a single
    ``e3nn.flax.Linear`` projecting node features down to ``1x0e``. Returns a
    per-atom scalar (shape ``(n_atoms,)``).

    Parameters
    ----------
    irreps_in : str
        e3nn irreps string for the input node features (typically scalar-only,
        e.g. ``"128x0e"``, for the foundation models).
    """

    irreps_in: str

    @nn.compact
    def __call__(self, node_feats):
        out = e3nn.flax.Linear("1x0e", name="linear")(node_feats)
        return out.array.squeeze(-1)


class NonLinearReadoutBlock(nn.Module):
    """Non-linear (MLP) scalar readout.

    Mirrors torch-mace's :class:`NonLinearReadoutBlock`:
    ``linear_1 -> silu -> linear_2``. The hidden layer is scalar-only for the
    foundation models considered here (``MLP_irreps = "16x0e"``). The hidden
    activation is the *normalised* SiLU (``silu(x) * normalize2mom(silu).cst``)
    that torch-e3nn bakes into ``nn.Activation`` — we must reproduce this
    factor for bit-for-bit parity.

    Parameters
    ----------
    irreps_in : str
        e3nn irreps string for the input node features.
    MLP_irreps : str
        Hidden-layer irreps (scalar-only).
    silu_normalization : float
        Multiplicative constant applied after SiLU to match torch-e3nn's
        ``normalize2mom`` wrapper. Default is the exact numerical value
        ``1.6791767923989418`` used by torch-e3nn for ``torch.nn.functional.silu``.
    """

    irreps_in: str
    MLP_irreps: str = "16x0e"
    silu_normalization: float = 1.6791767923989418

    @nn.compact
    def __call__(self, node_feats):
        irreps_hidden = e3nn.Irreps(self.MLP_irreps)
        if not all(ir.l == 0 and ir.p == 1 for _, ir in irreps_hidden):
            raise NotImplementedError(
                "NonLinearReadoutBlock only supports scalar hidden irreps "
                f"(got {self.MLP_irreps!r})."
            )
        h = e3nn.flax.Linear(self.MLP_irreps, name="linear_1")(node_feats)
        h_act = jax.nn.silu(h.array) * self.silu_normalization
        h = e3nn.IrrepsArray(h.irreps, h_act)
        out = e3nn.flax.Linear("1x0e", name="linear_2")(h)
        return out.array.squeeze(-1)


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
    """MACE product block: high-body-order symmetric contraction.

    Wraps cuequivariance's MACE symmetric-contraction descriptor, evaluated
    through :func:`cuequivariance_jax.equivariant_polynomial`. Produces node
    features in the same irreps as its input, after per-element weighted
    self-tensor-products up to ``correlation`` order.

    Parameters
    ----------
    hidden_irreps : str
        Irreps string for both input and output (MACE keeps them equal).
        All entries must share a common multiplicity.
    correlation : int
        Maximum tensor-product order (MACE commonly uses 3).
    num_elements : int
        Number of chemical elements (per-species weight table row count).
    use_cueq : bool
        Reserved for P2 (CUDA kernel dispatch). For now, all paths go through
        :func:`cuex.equivariant_polynomial` with ``method='naive'``.

    Notes
    -----
    The descriptor built by
    :func:`cuequivariance.group_theory.experimental.mace.symmetric_contractions.symmetric_contraction`
    is static (does not depend on any traced values), so it is built lazily per
    configuration via a module-level ``lru_cache``. Weight initialisation uses
    ``normal(stddev=1.0)`` as a placeholder for the smoke test; parity with
    torch-mace is deferred to P3.
    """

    hidden_irreps: str
    correlation: int = 3
    num_elements: int = 119
    use_cueq: bool = False

    @nn.compact
    def __call__(self, node_feats, Z):
        if self.use_cueq:
            raise NotImplementedError(
                "use_cueq=True is reserved for P2 (cuequivariance CUDA dispatch); "
                "not yet implemented."
            )
        irreps_out_e3 = e3nn.Irreps(self.hidden_irreps)

        muls = {mul for mul, _ in irreps_out_e3}
        if len(muls) != 1:
            raise ValueError(
                "ProductBlock requires all irreps to share the same multiplicity; "
                f"got multiplicities {muls} in {self.hidden_irreps!r}"
            )
        mul = next(iter(muls))

        # Build the (static) descriptor. The cache keeps jit-recompilation cheap.
        descriptor, weight_irreps, weight_numel = _get_symmetric_contraction_descriptor(
            str(irreps_out_e3),
            int(self.correlation),
        )
        weight_basis_dim = weight_numel // mul

        array = node_feats.array
        dtype = array.dtype
        n_atoms = array.shape[0]
        feature_dim = sum(ir.dim for _, ir in irreps_out_e3)
        expected_total = mul * feature_dim
        if array.shape[-1] != expected_total:
            raise ValueError(
                "ProductBlock expected a flat feature dim of "
                f"{expected_total} (mul={mul} * feature_dim={feature_dim}); "
                f"got {array.shape[-1]}"
            )

        # Reshape (n_atoms, mul*feature_dim) -> (n_atoms, mul, feature_dim) in mul_ir.
        x_mul_ir = array.reshape(n_atoms, mul, feature_dim)

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
        irreps_in_cue = cue.Irreps(cue.O3, str(irreps_out_e3))
        x_rep = _features_to_rep(x_mul_ir, irreps_in_cue, mul, dtype)

        out_rep = cuex.equivariant_polynomial(
            descriptor,
            [weight_rep, x_rep],
            math_dtype=dtype,
            method="naive",
        )

        out_ir_mul = out_rep.change_layout(cue.ir_mul).array
        out_mul_ir = _ir_mul_to_mul_ir(out_ir_mul, irreps_out_e3)
        return e3nn.IrrepsArray(irreps_out_e3, out_mul_ir)


# ---------------------------------------------------------------------------
# Private helpers (ported from mace-jax's cuequivariance adapter utilities)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _get_symmetric_contraction_descriptor(hidden_irreps_str: str, correlation: int):
    """Cache the MACE symmetric-contraction descriptor build.

    Parameters
    ----------
    hidden_irreps_str : str
        String form of the node irreps (input == output for MACE product blocks).
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
    only by the ``use_reduced_cg=False`` path. That path is deferred to P3; the
    projection is discarded here and the cache signature kept minimal. When P3
    adds support, extend this function's return to include it.
    """
    irreps_cue = cue.Irreps(cue.O3, hidden_irreps_str)
    degrees = tuple(range(1, correlation + 1))
    descriptor, _projection = _cue_mace_symmetric_contraction(
        irreps_cue, irreps_cue, degrees
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
