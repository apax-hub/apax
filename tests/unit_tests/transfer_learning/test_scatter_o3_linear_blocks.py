"""Pin the multi-irrep slot-key vs torch instruction-order invariant.

``_scatter_o3_linear_blocks`` (apax/transfer_learning/mace_foundation.py)
maps torch-mace ``o3.Linear`` weights into apax's ``e3nn.flax.Linear``
slot-keyed parameter tree. For multi-irrep targets (e.g. layer 1 of
MPA-0 medium), the slot-key sort order must agree with torch's
instruction-order or weights silently land in the wrong slots.

This test locks the contract by constructing both Linears with multi-irrep
input and output, scattering torch weights into apax via the converter
helper, and asserting both produce identical output for the same input.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.mace_parity


def test_scatter_o3_linear_blocks_multi_irrep_round_trip():
    """Multi-irrep round-trip: same input then same output bit-by-bit at float64.

    Constructs ``o3.Linear(8x0e+8x1o+8x2e -> 4x0e+4x1o+4x2e)`` on the
    torch side, scatters its weights into the matching ``e3nn.flax.Linear``
    on the apax side via ``_scatter_o3_linear_blocks``, and asserts both
    yield identical output for a random input. The test gates on
    ``mace_parity`` because torch + e3nn + e3nn-jax are required.
    """
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    pytest.importorskip("e3nn_jax")

    import e3nn_jax as e3j  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import torch  # noqa: PLC0415

    # torch >=2.6 sets ``weights_only=True`` by default, which trips on the
    # ``slice`` global baked into e3nn's precomputed Wigner constants. Allow-
    # list it so ``from e3nn import o3`` succeeds; the constants ship with
    # e3nn itself and are safe.
    torch.serialization.add_safe_globals([slice])
    from e3nn import o3  # noqa: PLC0415

    from apax.transfer_learning.mace_foundation import (  # noqa: PLC0415
        _scatter_o3_linear_blocks,
    )

    # Multi-irrep input AND output. Each output irrep is reachable from a
    # single input irrep of the same l (so the converter must place each
    # (mul_in, mul_out) block in the matching slot in instruction order).
    # 0e, 1o, 2e cover scalar, vector, and rank-2 tensor channels — the
    # combination that surfaces the slot-key sort vs instruction-order
    # invariant for production foundation models.
    # Multiplicities are asymmetric (`mul_in=8 != mul_out=4`) so a transposed `(mul_in, mul_out)` block in the converter would not coincidentally pass.
    irreps_in = "8x0e + 8x1o + 8x2e"
    irreps_out = "4x0e + 4x1o + 4x2e"

    torch_lin = o3.Linear(
        irreps_in=o3.Irreps(irreps_in),
        irreps_out=o3.Irreps(irreps_out),
        biases=False,
    )
    torch_lin.double()
    flat_w = torch_lin.weight.detach().double().cpu().numpy()

    rng = np.random.default_rng(0)
    n_atoms = 5
    x_np = rng.standard_normal(
        size=(n_atoms, o3.Irreps(irreps_in).dim),
    ).astype(np.float64)

    with torch.no_grad():
        y_torch = torch_lin(torch.from_numpy(x_np)).cpu().numpy()

    # apax-side Linear. Mirrors the call shape used in
    # apax/layers/descriptor/mace_blocks.py for the per-layer ``linear``
    # block (no force_irreps_out — every output irrep here is reachable).
    apax_lin = e3j.flax.Linear(e3j.Irreps(irreps_out), name="linear")
    x_jax = e3j.IrrepsArray(e3j.Irreps(irreps_in), jnp.asarray(x_np))
    params = apax_lin.init(jax.random.PRNGKey(0), x_jax)

    # Convert to mutable numpy tree, scatter, and re-apply. The converter
    # mutates the slot-keyed sub-dict in place; ``params["params"]`` is
    # exactly that sub-dict (the top-level ``Linear`` module has its
    # slot-keys directly under ``params``).
    apax_params = jax.tree_util.tree_map(np.asarray, params)
    _scatter_o3_linear_blocks(apax_params["params"], flat_w)

    y_apax = apax_lin.apply(apax_params, x_jax).array
    y_apax_np = np.asarray(y_apax)

    # Bit-identical agreement at float64. Loosening this tolerance would
    # mask the I4 invariant (slot-key vs instruction-order disagreement)
    # this test exists to catch.
    np.testing.assert_allclose(y_apax_np, y_torch, atol=1e-12, rtol=1e-12)
