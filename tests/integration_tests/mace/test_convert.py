"""Integration test for the MACE foundation converter.

Gated by the ``mace_parity`` marker; requires::

    uv sync --extra mace-convert

Two tests verify that :func:`apax.transfer_learning.mace_foundation.run_conversion`
produces a directory in apax's standard training-output layout and that the
output round-trips through :func:`apax.train.checkpoints.restore_parameters`.
"""
from __future__ import annotations

import json
from pathlib import Path

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
    assert cfg.model.r_max == pytest.approx(6.0)
    assert cfg.model.num_interactions == 2
    assert cfg.model.correlation == 3
    assert cfg.model.hidden_irreps == "128x0e"

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
