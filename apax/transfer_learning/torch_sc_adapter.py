"""Convert torch-mace symmetric-contraction weights into the apax layout.

Vendored from ``mace_jax.adapters.cuequivariance.symmetric_contraction``
(MIT-licensed). The mace-jax dependency was dropped because its public surface
is unstable (the helper used here is private) and not on PyPI; the dense
linear-algebra portion is straightforward to reproduce directly on the
cuequivariance descriptor that apax already builds.

Required at conversion time only — torch and mace-torch are still needed to
read the source state-dict and compute the reduced-CG projection.
"""
from __future__ import annotations

from functools import cache

import cuequivariance as cue
import cuequivariance_jax as cuex
import jax.numpy as jnp
import numpy as np
from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction as cue_mace_symmetric_contraction,
)
from e3nn_jax import Irreps


def convert_native_weights(torch_module, *, target_template: jnp.ndarray) -> jnp.ndarray:
    """Convert native MACE weights into the layout expected by apax's ProductBlock.

    Parameters
    ----------
    torch_module
        A torch ``mace.modules.symmetric_contraction.SymmetricContraction`` instance.
    target_template
        Array matching the desired output shape ``(num_elements, basis_dim, mul)``
        and dtype. Only its metadata is consumed.

    Returns
    -------
    jnp.ndarray
        Weight tensor expressed in the basis required by apax's ProductBlock.
    """
    from mace.tools.cg_cueq_tools import (  # noqa: PLC0415
        symmetric_contraction_proj as mace_symmetric_contraction_proj,
    )

    irreps_in = Irreps(str(torch_module.irreps_in))
    irreps_out = Irreps(str(torch_module.irreps_out))
    correlation = int(torch_module.contractions[0].correlation)
    num_elements = int(torch_module.contractions[0].weights_max.shape[0])

    mul = irreps_in[0].mul
    feature_dim = sum(term.ir.dim for term in irreps_in)
    if mul == 0 or feature_dim == 0:
        return target_template

    weight_shape = target_template.shape
    _, basis_dim, mul_dim = weight_shape
    if basis_dim == 0 or mul_dim == 0:
        return target_template

    native_weight = _gather_native_reduced_weights(
        torch_module,
        correlation=correlation,
        mul_dim=mul_dim,
        num_elements=num_elements,
    )

    cue_irreps_in = cue.Irreps(cue.O3, str(irreps_in))
    cue_irreps_out = cue.Irreps(cue.O3, str(irreps_out))
    degrees = tuple(range(1, correlation + 1))
    _, reduced_projection = mace_symmetric_contraction_proj(
        cue_irreps_in, cue_irreps_out, degrees,
    )
    reduced_projection = np.asarray(reduced_projection, dtype=native_weight.dtype)

    _, descriptor_projection = cue_mace_symmetric_contraction(
        cue_irreps_in, cue_irreps_out, degrees,
    )
    descriptor_projection = np.asarray(descriptor_projection, dtype=native_weight.dtype)

    reduced_dim = reduced_projection.shape[1]
    full_dim = descriptor_projection.shape[0]
    native_dim = native_weight.shape[1]

    if basis_dim == reduced_dim:
        if native_dim == reduced_dim:
            converted = np.einsum(
                "zau,ab->zbu", native_weight, reduced_projection, optimize=True,
            )
        elif native_dim == full_dim:
            transform = _full_cg_transform(irreps_in, irreps_out, correlation)
            transform = np.asarray(transform, dtype=native_weight.dtype)
            canonical = np.einsum("ab,zbu->zau", transform, native_weight, optimize=True)
            converted = np.einsum(
                "zau,ab->zbu", canonical, descriptor_projection, optimize=True,
            )
        else:
            raise ValueError(
                "Native SymmetricContraction weight shape mismatch during import."
            )
    elif basis_dim == full_dim:
        if native_dim == full_dim:
            transform = _full_cg_transform(irreps_in, irreps_out, correlation)
            transform = np.asarray(transform, dtype=native_weight.dtype)
            converted = np.einsum("ab,zbu->zau", transform, native_weight, optimize=True)
        elif native_dim == reduced_dim:
            lift = np.linalg.pinv(descriptor_projection, rcond=1e-12).astype(
                native_weight.dtype, copy=False,
            )
            converted = np.einsum("zau,ab->zbu", native_weight, lift, optimize=True)
        else:
            raise ValueError(
                "Native SymmetricContraction weight shape mismatch during import."
            )
    else:
        raise ValueError(
            "Native SymmetricContraction weight shape mismatch during import."
        )
    return jnp.asarray(converted, dtype=target_template.dtype)


def _gather_native_reduced_weights(
    torch_module,
    *,
    correlation: int,
    mul_dim: int,
    num_elements: int,
) -> np.ndarray:
    """Stack native torch weights in the order expected by the cue projection."""
    base_array = torch_module.contractions[0].weights_max.detach().cpu().numpy()
    dtype = np.asarray(base_array).dtype

    if correlation <= 0:
        return np.zeros((num_elements, 0, mul_dim), dtype=dtype)

    native_blocks: list[np.ndarray] = []
    for contraction in torch_module.contractions:
        degree_blocks: list[np.ndarray] = []
        for degree in range(correlation, 0, -1):
            if degree == correlation:
                weight_param = contraction.weights_max
                zeroed = bool(getattr(contraction, "weights_max_zeroed", False))
            else:
                idx = correlation - degree - 1
                weight_param = contraction.weights[idx]
                zeroed = bool(getattr(contraction, f"weights_{idx}_zeroed", False))

            array = np.asarray(weight_param.detach().cpu().numpy(), dtype=dtype)
            if zeroed:
                array = np.zeros_like(array)
            if array.shape[1] == 0:
                continue
            degree_blocks.append(array)

        if degree_blocks:
            native_blocks.append(np.concatenate(degree_blocks, axis=1))

    if native_blocks:
        stacked = np.concatenate(native_blocks, axis=1)
    else:
        stacked = np.zeros((num_elements, 0, mul_dim), dtype=dtype)

    if stacked.shape[0] != num_elements or stacked.shape[2] != mul_dim:
        raise ValueError("Native SymmetricContraction weights shape mismatch.")
    return stacked


def _full_cg_transform(
    irreps_in: Irreps, irreps_out: Irreps, correlation: int,
) -> np.ndarray:
    """Native -> canonical change-of-basis for full-CG weights."""
    base_in = Irreps(str(irreps_in)).set_mul(1)
    base_out = Irreps(str(irreps_out)).set_mul(1)
    return _cached_full_cg_transform(str(base_in), str(base_out), int(correlation))


@cache
def _cached_full_cg_transform(
    irreps_in_str: str, irreps_out_str: str, correlation: int,
) -> np.ndarray:
    """Solve for the native -> canonical change-of-basis via design matrices.

    Builds the canonical design matrix by applying the cuequivariance descriptor
    directly to one-hot canonical weight excitations. Builds the native design
    matrix by exciting the torch ``SymmetricContraction`` one native basis
    vector at a time. A least-squares solve recovers the transform.

    Cached because the linear-algebra cost is non-trivial and the result depends
    only on (irreps_in, irreps_out, correlation).
    """
    from e3nn import o3  # noqa: PLC0415
    from mace.modules.wrapper_ops import (  # noqa: PLC0415
        SymmetricContractionWrapper as TorchSymmetricContraction,
    )

    irreps_in_o3 = o3.Irreps(irreps_in_str)
    irreps_out_o3 = o3.Irreps(irreps_out_str)

    torch_module = (
        TorchSymmetricContraction(
            irreps_in=irreps_in_o3,
            irreps_out=irreps_out_o3,
            correlation=correlation,
            num_elements=1,
            use_reduced_cg=False,
        )
        .float()
        .eval()
    )

    cue_irreps_in = cue.Irreps(cue.O3, irreps_in_str)
    cue_irreps_out = cue.Irreps(cue.O3, irreps_out_str)
    degrees = tuple(range(1, correlation + 1))
    descriptor, descriptor_projection = cue_mace_symmetric_contraction(
        cue_irreps_in, cue_irreps_out, degrees,
    )
    descriptor_projection = np.asarray(descriptor_projection)
    weight_irreps = descriptor.inputs[0].irreps

    mul = Irreps(irreps_in_str)[0].mul
    feature_dim = sum(term.ir.dim for term in Irreps(irreps_in_str))
    # The "canonical" basis here is the full CG basis (projection's first axis);
    # the cue descriptor consumes the projected reduced basis.
    canonical_dim = descriptor_projection.shape[0]

    native_dim = _gather_native_reduced_weights(
        torch_module, correlation=correlation, mul_dim=mul, num_elements=1,
    ).shape[1]

    batch = max(canonical_dim, native_dim)
    rng = np.random.default_rng(0)
    inputs_np = rng.standard_normal((batch, mul, feature_dim)).astype(np.float32)

    canonical_matrix = _canonical_design_matrix(
        descriptor=descriptor,
        weight_irreps=weight_irreps,
        descriptor_projection=descriptor_projection,
        cue_irreps_in=cue_irreps_in,
        cue_irreps_out=cue_irreps_out,
        canonical_dim=canonical_dim,
        mul=mul,
        feature_dim=feature_dim,
        inputs=jnp.asarray(inputs_np),
    )
    native_matrix = _native_design_matrix(
        torch_module,
        correlation=correlation,
        basis_dim=native_dim,
        inputs_np=inputs_np,
    )

    transform = np.linalg.lstsq(canonical_matrix, native_matrix, rcond=1e-12)[0]
    return transform.astype(np.float64)


def _canonical_design_matrix(
    *,
    descriptor,
    weight_irreps,
    descriptor_projection: np.ndarray,
    cue_irreps_in,
    cue_irreps_out,
    canonical_dim: int,
    mul: int,
    feature_dim: int,
    inputs: jnp.ndarray,
) -> np.ndarray:
    """Apply the cuequivariance descriptor to each canonical (full-CG) basis excitation.

    Mirrors the forward pass of mace-jax's ``SymmetricContraction`` with
    ``use_reduced_cg=False``: a one-hot weight in the full CG basis is
    projected through ``descriptor_projection`` into the reduced descriptor
    basis before being fed to ``cuex.equivariant_polynomial``.
    """
    irreps_out_o3 = Irreps(str(cue_irreps_out))
    base_irreps_in = cue_irreps_in.set_mul(1)
    weight_numel = weight_irreps.dim
    proj = descriptor_projection.astype(np.float32)

    x_rep = _features_to_rep(inputs, base_irreps_in, cue_irreps_in, mul, inputs.dtype)
    batch = inputs.shape[0]

    outputs: list[np.ndarray] = []
    for idx in range(canonical_dim):
        # One-hot in the full CG basis at slot 0 of the mul axis.
        w_full = np.zeros((1, canonical_dim, mul), dtype=np.float32)
        w_full[0, idx, 0] = 1.0
        # Project full -> reduced descriptor basis: 'zau,ab->zbu'.
        w_proj = np.einsum("zau,ab->zbu", w_full, proj, optimize=True)
        w_flat = w_proj.reshape(1, weight_numel)
        selected = jnp.broadcast_to(jnp.asarray(w_flat), (batch, weight_numel))
        weight_rep = cuex.RepArray(weight_irreps, selected, cue.ir_mul)
        out_rep = cuex.equivariant_polynomial(
            descriptor, [weight_rep, x_rep], math_dtype=inputs.dtype, method="naive",
        )
        out_ir_mul = out_rep.change_layout(cue.ir_mul).array
        out_mul_ir = _ir_mul_to_mul_ir(out_ir_mul, irreps_out_o3)
        outputs.append(np.asarray(out_mul_ir).reshape(-1))
    return np.stack(outputs, axis=1)


def _native_design_matrix(
    torch_module, *, correlation: int, basis_dim: int, inputs_np: np.ndarray,
) -> np.ndarray:
    import torch  # noqa: PLC0415

    torch_inputs = torch.tensor(inputs_np, dtype=torch.float32)
    num_elements = torch_module.contractions[0].weights_max.shape[0]
    torch_attrs = torch.ones((inputs_np.shape[0], num_elements), dtype=torch.float32)

    outputs: list[np.ndarray] = []
    for idx in range(basis_dim):
        basis = np.zeros(basis_dim, dtype=np.float32)
        basis[idx] = 1.0
        _assign_native_basis(torch_module, basis_vector=basis, correlation=correlation)
        with torch.no_grad():
            out = torch_module(torch_inputs, torch_attrs).cpu().numpy().reshape(-1)
        outputs.append(out)
    return np.stack(outputs, axis=1)


def _assign_native_basis(
    torch_module, *, basis_vector: np.ndarray, correlation: int,
) -> None:
    import torch  # noqa: PLC0415

    offset = 0
    with torch.no_grad():
        for contraction in torch_module.contractions:
            for degree in range(correlation, 0, -1):
                if degree == correlation:
                    target = contraction.weights_max
                else:
                    idx = correlation - degree - 1
                    target = contraction.weights[idx]
                width = target.shape[1]
                slice_vals = basis_vector[offset : offset + width]
                target.zero_()
                target[0, :width, 0] = torch.tensor(slice_vals, dtype=target.dtype)
                offset += width
    if offset != basis_vector.size:
        raise ValueError("Basis vector length mismatch while scattering weights.")


def _features_to_rep(x_mul_ir, base_irreps, full_irreps, mul: int, dtype):
    """Pack mul_ir features into a cuequivariance ir_mul RepArray."""
    segments: list[jnp.ndarray] = []
    offset = 0
    for mul_ir in base_irreps:
        width = mul_ir.ir.dim
        seg = x_mul_ir[:, :, offset : offset + width]
        segments.append(jnp.swapaxes(seg, -2, -1))
        offset += width
    return cuex.from_segments(
        full_irreps,
        segments,
        (x_mul_ir.shape[0], mul),
        cue.ir_mul,
        dtype=dtype,
    )


def _ir_mul_to_mul_ir(array: jnp.ndarray, irreps: Irreps) -> jnp.ndarray:
    """Reorder the last axis from ir_mul back to e3nn mul_ir."""
    if irreps.dim == 0:
        return array
    leading = array.shape[:-1]
    array = array.reshape(*leading, irreps.dim)
    segments: list[jnp.ndarray] = []
    offset = 0
    for mul, ir in irreps:
        block = array[..., offset : offset + mul * ir.dim]
        offset += mul * ir.dim
        block = block.reshape(*leading, ir.dim, mul)
        block = jnp.swapaxes(block, -1, -2)
        block = block.reshape(*leading, mul * ir.dim)
        segments.append(block)
    return jnp.concatenate(segments, axis=-1) if segments else array
