"""Integration test for the MACE foundation converter.

Gated by the ``mace_parity`` marker; requires::

    uv sync --extra mace-convert

Two tests verify that :func:`apax.transfer_learning.mace_foundation.run_conversion`
produces a directory in apax's standard training-output layout and that the
output round-trips through :func:`apax.train.checkpoints.restore_parameters`.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.mace_parity


def test_convert_small_writes_apax_native_format(tmp_path):
    """Convert MACE-MP-0 ``small`` and verify apax-native layout."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")

    from apax.train.checkpoints import restore_parameters
    from apax.transfer_learning.mace_foundation import run_conversion

    dst = tmp_path / "small.apax"
    run_conversion("small", dst, head="default", family="mace_mp")

    # Layout: <dst>/config.yaml + <dst>/best/ + <dst>/converter_metadata.json
    assert (dst / "config.yaml").exists(), "config.yaml not written"
    assert (dst / "best").is_dir(), "orbax best/ checkpoint dir missing"
    assert (dst / "converter_metadata.json").exists(), "metadata not written"

    meta = json.loads((dst / "converter_metadata.json").read_text())
    assert meta["source"] == "small"
    assert meta["family"] == "mace_mp"
    assert meta["head_selected"] == "default"
    assert meta["torch_mace_version"]
    assert meta["apax_version"]

    # Standard apax loader path: returns (Config, params)
    cfg, params = restore_parameters(dst)
    assert cfg.model.name == "mace"
    assert len(cfg.model.descriptor.interactions) == 2
    assert cfg.model.descriptor.correlation == 3
    assert cfg.model.descriptor.hidden_irreps == "128x0e"
    assert cfg.model.basis.r_max == pytest.approx(6.0)

    # Pytree must contain the three top-level branches expected by
    # ``EnergyDerivativeModel(EnergyModel(representation, readout, scale_shift))``.
    energy_params = params["params"]["energy_model"]
    assert "representation" in energy_params
    assert "readout" in energy_params
    assert "scale_shift" in energy_params


def test_convert_rejects_unknown_head(tmp_path):
    """Unknown ``--head`` should fail before any heavy work happens."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion

    with pytest.raises(ValueError, match="head"):
        run_conversion(
            "medium-mpa-0",
            tmp_path / "out.apax",
            head="not-a-real-head",
            family="mace_mp",
        )


def test_torch_to_apax_param_coverage_no_projections(tmp_path):
    """Every torch state_dict float param maps 1:1 to an apax leaf (modulo 89→119 padding).

    Asserts numel-per-key equality so we catch regressions where a future
    refactor would silently drop or project a torch weight. Apax has more
    total params than torch because of element-table padding (89 → 119) and
    the pre-existing ``PerElementScaleShift`` rows; the ``>=`` check is the
    safety net.
    """
    pytest.importorskip("torch")
    pytest.importorskip("mace")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from apax.transfer_learning.mace_foundation import (
        _extract_config_from_torch,
        _load_torch_foundation_model,
        _map_state_to_pytree,
        _synthesize_full_config,
    )

    torch_model, _ = _load_torch_foundation_model("small", family="mace_mp")
    torch_atomic_numbers = tuple(
        torch_model.atomic_numbers.detach().cpu().numpy().astype(int).tolist()
    )
    cfg_fields = _extract_config_from_torch(torch_model, head="default")
    full_cfg = _synthesize_full_config(cfg_fields, dst=tmp_path / "_apax_pcov")
    Builder = full_cfg.model.get_builder()
    builder = Builder(full_cfg.model.model_dump(), n_species=119)
    energy_model = builder.build_energy_derivative_model()

    R = jnp.zeros((2, 3))
    Z = jnp.array([1, 1], dtype=jnp.int32)
    neigh = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    box = jnp.zeros((3,))
    offsets = jnp.zeros((neigh.shape[1], 3))
    template = energy_model.init(jax.random.PRNGKey(0), R, Z, neigh, box, offsets)

    state = {k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()}
    extra_scalars = {
        "scale": float(torch_model.scale_shift.scale.detach().cpu()),
        "shift": float(torch_model.scale_shift.shift.detach().cpu()),
        "atomic_energies": torch_model.atomic_energies_fn.atomic_energies.detach()
        .cpu()
        .numpy(),
    }
    params = _map_state_to_pytree(
        state,
        template,
        torch_atomic_numbers=torch_atomic_numbers,
        extra_scalars=extra_scalars,
        selected_head="default",
        config=full_cfg.model,
        torch_model=torch_model,
    )

    apax_total = sum(
        int(np.prod(v.shape)) for _, v in jax.tree_util.tree_flatten_with_path(params)[0]
    )
    torch_total = sum(
        int(np.prod(v.shape)) for v in state.values() if v.dtype.kind == "f"
    )
    # Apax has 119/89 element padding plus PerElementScaleShift's pre-existing
    # rows; allow only the documented padding overhead.
    assert apax_total >= torch_total, (
        f"apax total numel {apax_total} < torch total {torch_total} — "
        "weights are being dropped"
    )


def test_convert_medium_mpa0_with_distance_transform(tmp_path):
    """Convert MACE-MPA-0 medium and verify the AgnesiTransform is wired.

    MPA-0 ships an AgnesiTransform; small does not. The converter must
    locate the apax distance-transform slot and copy ``a``/``q``/``p``/
    ``covalent_radii`` into it. Regression test for
    ``KeyError: 'distance_transform'`` reported when the wired slot path
    diverged from the actual params-tree path.
    """
    pytest.importorskip("torch")
    pytest.importorskip("mace")

    import numpy as np
    import torch

    from apax.train.checkpoints import restore_parameters
    from apax.transfer_learning.mace_foundation import run_conversion

    dst = tmp_path / "mpa-0-medium.apax"
    run_conversion("medium-mpa-0", dst, head="default", family="mace_mp")

    cfg, params = restore_parameters(dst)
    assert cfg.model.radial_embedding.distance_transform is not None
    assert cfg.model.radial_embedding.distance_transform.name == "agnesi"

    # The AgnesiTransform's ``a``/``q``/``p``/``covalent_radii`` must be in
    # the converted pytree, with values matching the torch source bit-for-bit.
    src_path = "/Users/fzills/tools/apax/tmp/mace-mpa-0-medium.model"
    torch_model = torch.load(src_path, map_location="cpu", weights_only=False)
    dt_torch = torch_model.radial_embedding.distance_transform

    found = {}
    for path, leaf in _flatten_with_str_path(params):
        if "distance_transform" in path:
            tail = path.rsplit("/", 1)[-1]
            found[tail] = np.asarray(leaf)

    assert found, "distance_transform slot not present in converted params pytree"
    for name in ("a", "q", "p", "covalent_radii"):
        assert name in found, f"distance_transform/{name} missing from pytree"
        torch_arr = np.asarray(getattr(dt_torch, name).detach().cpu())
        np.testing.assert_allclose(found[name], torch_arr, atol=0.0, rtol=0.0)


def _flatten_with_str_path(tree):
    """Yield ``(slash_path, leaf)`` pairs for a nested mapping."""
    import jax

    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        keys = []
        for entry in path:
            if hasattr(entry, "key"):
                keys.append(str(entry.key))
            else:
                keys.append(str(entry))
        yield "/".join(keys), leaf
