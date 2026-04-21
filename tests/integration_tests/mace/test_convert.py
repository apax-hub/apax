"""Integration test for convert-mace. Gated by mace_parity marker.

Requires:
    uv sync --group mace-convert --extra mace

These tests resolve MACE foundation models via the upstream
``mace.calculators.foundations_models.mace_mp`` interface, which handles
the bundled-local model, HTTP download, and caching under ``~/.cache/mace/``.
"""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.mace_parity


@pytest.mark.parametrize("model_name", [
    "medium-mpa-0",      # default; bundled with mace-torch package
    "medium",            # MACE-MP-0 medium; first-run download, cached thereafter
])
def test_convert_canonical_name(tmp_path, model_name):
    """Convert a canonical foundation model fetched via mace_mp()."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion

    dst = tmp_path / f"{model_name}.apax"
    run_conversion(model_name, dst, head="mp", family="mace_mp")

    assert (dst / "params.msgpack").exists()
    assert (dst / "config.json").exists()
    assert (dst / "metadata.json").exists()

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["name"] == "mace"
    assert cfg["num_interactions"] >= 1

    meta = json.loads((dst / "metadata.json").read_text())
    assert meta["source"] == model_name            # records the canonical name
    assert meta["source_resolved_path"]             # records where it actually came from


def test_convert_local_path(tmp_path):
    """Convert from an explicit .model path (no network)."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    # Pre-resolve the cached path, then feed it as a local file input
    local_path = Path(download_mace_mp_checkpoint("medium-mpa-0"))
    assert local_path.exists()

    dst = tmp_path / "local.apax"
    run_conversion(str(local_path), dst, head="mp", family="mace_mp")

    assert (dst / "params.msgpack").exists()


def test_convert_rejects_unknown_head(tmp_path):
    """Reject unknown --head before any mapping happens."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion

    with pytest.raises(ValueError, match="head"):
        run_conversion("medium-mpa-0", tmp_path / "out.apax", head="does-not-exist")
