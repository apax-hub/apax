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
    target_irreps : str or None
        Target irreps used for the inner ``linear`` (post-TP). If ``None`` this
        equals ``irreps_out``. Foundation parity requires ``target_irreps`` to
        be the full irreps emitted by the tensor product (e.g.
        ``"128x0e+128x1o+128x2e+128x3o"``), independent of the final
        ``hidden_irreps`` which is scalar-only.
    interaction_cls : str
        Which interaction variant; for now only RealAgnosticResidual is
        implemented. Other variants raise NotImplementedError.
    radial_mlp : tuple[int, ...]
        Hidden layer widths for the radial MLP that gates tensor-product
        channels. Defaults to ``(64, 64, 64)``.
    foundation_mode : bool
        If True, mirrors torch-mace foundation semantics:
        - skip connection becomes per-element
          ``FullyConnectedTensorProduct(node_feats_irreps, one_hot(Z), hidden_irreps)``;
          ``num_elements`` and ``hidden_irreps_final`` must be provided.
        - post-conv ``linear`` output is divided by ``avg_num_neighbors``.
        - radial MLP has no activation on the output layer.
        If False, behaviour is unchanged (existing training codepath).
    num_elements : int
        Number of chemical species for the per-element skip. Required when
        ``foundation_mode=True``.
    hidden_irreps_final : str or None
        Irreps of the skip-connection output in foundation mode (e.g.
        ``"128x0e"`` for small MP-0). Not used if ``foundation_mode=False``.
    avg_num_neighbors : float
        Normalisation factor; post-conv message is divided by this. Only used
        when ``foundation_mode=True``.
    """

    irreps_out: str
    target_irreps: str | None = None
    interaction_cls: str = "RealAgnosticResidual"
    radial_mlp: tuple = (64, 64, 64)
    foundation_mode: bool = False
    num_elements: int = 0
    hidden_irreps_final: str | None = None
    avg_num_neighbors: float = 1.0

    @nn.compact
    def __call__(self, node_feats, edge_attrs, edge_feats, receivers, senders, Z=None):
        if self.interaction_cls != "RealAgnosticResidual":
            raise NotImplementedError(
                f"Interaction variant {self.interaction_cls!r} not yet implemented; "
                "only 'RealAgnosticResidual' is supported at this phase."
            )
        irreps_in = node_feats.irreps
        irreps_out = e3nn.Irreps(self.irreps_out)
        target_irreps = (
            e3nn.Irreps(self.target_irreps) if self.target_irreps is not None
            else irreps_out
        )

        # 1. Linear pre-mix
        x = e3nn.flax.Linear(irreps_in, name="linear_up")(node_feats)

        # 2. Gather source node features at senders
        x_j = x[senders]

        # 3. Tensor product with edge spherical harmonics
        #    Output irreps = full tensor-square subset reachable in target_irreps
        tp = e3nn.tensor_product(x_j, edge_attrs, filter_ir_out=target_irreps)

        # 4. Radial MLP producing a scalar per TP path per edge
        n_paths = tp.irreps.num_irreps
        mlp_widths = (*self.radial_mlp, n_paths)
        mlp_kwargs = {}
        if self.foundation_mode:
            mlp_kwargs["output_activation"] = False
        weights = e3nn.flax.MultiLayerPerceptron(
            list(mlp_widths), act=jax.nn.silu, name="radial_mlp", **mlp_kwargs
        )(edge_feats)
        weighted = tp * weights                                 # broadcast-safe

        # 5. Scatter-sum into receivers
        out = e3nn.scatter_sum(weighted, dst=receivers, output_size=node_feats.shape[0])

        # 6. Post-mix and residual.
        out = e3nn.flax.Linear(target_irreps, name="linear_down")(out)
        if self.foundation_mode:
            out = out / self.avg_num_neighbors

        if self.foundation_mode:
            if self.num_elements <= 0 or self.hidden_irreps_final is None or Z is None:
                raise ValueError(
                    "foundation_mode=True requires num_elements>0, "
                    "hidden_irreps_final, and a Z argument."
                )
            skip = PerElementSkipTP(
                num_elements=self.num_elements,
                hidden_irreps=self.hidden_irreps_final,
                name="skip_tp",
            )(node_feats, Z)
            # Expand skip to target_irreps layout by zero-padding non-scalar channels
            skip = _pad_to_irreps(skip, target_irreps)
        else:
            # Backward-compatible per-irrep linear skip with zero-padded extras.
            skip = e3nn.flax.Linear(
                target_irreps, name="skip_linear", force_irreps_out=True
            )(node_feats)

        return out + skip


def _pad_to_irreps(src, target_irreps):
    """Expand a scalar-only IrrepsArray to ``target_irreps`` with zero padding.

    Parameters
    ----------
    src : e3nn.IrrepsArray
        Scalar-only input, irreps ``"Mx0e"``.
    target_irreps : e3nn.Irreps
        Target irreps, must start with ``Mx0e`` (same multiplicity) and contain
        additional non-scalar entries to be zero-padded.

    Returns
    -------
    e3nn.IrrepsArray
        Array with irreps ``target_irreps``; scalar block copied from ``src``,
        non-scalar blocks filled with zeros.
    """
    target_irreps = e3nn.Irreps(target_irreps)
    # Pad: concatenate zeros for the non-0e part.
    n_leading = src.array.shape[:-1]
    total_dim = target_irreps.dim
    src_dim = src.array.shape[-1]
    if total_dim == src_dim:
        return e3nn.IrrepsArray(target_irreps, src.array)
    pad = jnp.zeros((*n_leading, total_dim - src_dim), dtype=src.array.dtype)
    arr = jnp.concatenate([src.array, pad], axis=-1)
    return e3nn.IrrepsArray(target_irreps, arr)


class PerElementSkipTP(nn.Module):
    """Per-element skip connection mirroring torch-mace's ``skip_tp``.

    Implements ``FullyConnectedTensorProduct((M_in x 0e), (E x 0e), (M_out x 0e))``
    where the second argument is a one-hot over chemical species ``Z``. This is
    equivalent to gathering a per-element linear weight matrix of shape
    ``(num_elements, M_in, M_out)`` and applying it to the scalar-only node
    features. The e3nn path weight ``1/sqrt(M_in * num_elements)`` is baked
    into the forward pass so that torch weights can be copied verbatim.

    Parameters
    ----------
    num_elements : int
        Number of chemical species (must match the torch model's
        ``atomic_numbers`` table length).
    hidden_irreps : str
        Output irreps, e.g. ``"128x0e"``.
    """

    num_elements: int
    hidden_irreps: str

    @nn.compact
    def __call__(self, node_feats, Z):
        in_irreps = node_feats.irreps
        out_irreps = e3nn.Irreps(self.hidden_irreps)

        if not all(ir.l == 0 for _, ir in in_irreps):
            raise NotImplementedError(
                "PerElementSkipTP currently supports scalar-only node features "
                f"(got irreps={in_irreps})."
            )
        if not all(ir.l == 0 for _, ir in out_irreps):
            raise NotImplementedError(
                "PerElementSkipTP currently supports scalar-only output "
                f"(got {self.hidden_irreps!r})."
            )

        in_mul = sum(mul for mul, ir in in_irreps if ir.l == 0)
        out_mul = sum(mul for mul, ir in out_irreps if ir.l == 0)
        dtype = node_feats.array.dtype

        path_weight = 1.0 / jnp.sqrt(
            jnp.asarray(in_mul * self.num_elements, dtype=dtype)
        )
        weight = self.param(
            "weight",
            nn.initializers.normal(stddev=1.0),
            (self.num_elements, in_mul, out_mul),
            dtype,
        )
        w_selected = weight[Z.astype(jnp.int32)]          # (n_atoms, in_mul, out_mul)
        x = node_feats.array                              # (n_atoms, in_mul)
        y = jnp.einsum("ni,nij->nj", x, w_selected) * path_weight
        return e3nn.IrrepsArray(out_irreps, y)


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
    """MACE product block: high-body-order symmetric contraction.

    Wraps cuequivariance's MACE symmetric-contraction descriptor, evaluated
    through :func:`cuequivariance_jax.equivariant_polynomial`. Produces node
    features after per-element weighted self-tensor-products up to
    ``correlation`` order. Optionally followed by an ``e3nn.flax.Linear`` post
    map (``post_linear=True``) to mirror torch-mace's ``EquivariantProductBasisBlock``.

    Parameters
    ----------
    hidden_irreps : str
        Output (target) irreps string. For the small MP-0 model this is
        ``"128x0e"``. All entries must share a common multiplicity.
    input_irreps : str or None
        Input irreps for the symmetric contraction. When ``None`` (default) this
        equals ``hidden_irreps``; the foundation-model path uses different
        input/output irreps (e.g. ``"128x0e+128x1o+128x2e+128x3o"`` → ``"128x0e"``).
    correlation : int
        Maximum tensor-product order (MACE commonly uses 3).
    num_elements : int
        Number of chemical elements (per-species weight table row count).
    post_linear : bool
        If True, apply an ``e3nn.flax.Linear(hidden_irreps → hidden_irreps)`` after
        the symmetric contraction (matches torch-mace foundation layout).
    use_cueq : bool
        Reserved for P2 (CUDA kernel dispatch). For now, all paths go through
        :func:`cuex.equivariant_polynomial` with ``method='naive'``.

    Notes
    -----
    The descriptor built by
    :func:`cuequivariance.group_theory.experimental.mace.symmetric_contractions.symmetric_contraction`
    is static (does not depend on any traced values), so it is built lazily per
    configuration via a module-level ``lru_cache``. Weight initialisation uses
    ``normal(stddev=1.0)`` as a placeholder; parity with torch-mace is achieved
    by the state-dict mapper in :mod:`apax.transfer_learning.mace_foundation`.
    """

    hidden_irreps: str
    input_irreps: str | None = None
    correlation: int = 3
    num_elements: int = 119
    post_linear: bool = False
    use_cueq: bool = False

    @nn.compact
    def __call__(self, node_feats, Z):
        if self.use_cueq:
            raise NotImplementedError(
                "use_cueq=True is reserved for P2 (cuequivariance CUDA dispatch); "
                "not yet implemented."
            )
        irreps_out_e3 = e3nn.Irreps(self.hidden_irreps)
        irreps_in_e3 = (
            e3nn.Irreps(self.input_irreps)
            if self.input_irreps is not None
            else irreps_out_e3
        )

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

        # Reshape (n_atoms, mul*feature_dim) -> (n_atoms, mul, feature_dim) in mul_ir.
        x_mul_ir = array.reshape(n_atoms, mul, feature_dim_in)

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
        if self.post_linear:
            out = e3nn.flax.Linear(irreps_out_e3, name="linear")(out)
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
