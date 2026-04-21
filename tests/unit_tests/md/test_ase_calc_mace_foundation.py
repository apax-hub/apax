"""ASECalculator dispatch for MACE foundation directories. Gated."""
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.mace_parity


def _write_fake_foundation_dir(tmp_path: Path) -> Path:
    """Fabricate a minimal apax MACE foundation-model directory.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.

    Returns
    -------
    Path
        Path to the created fake foundation directory.
    """
    dst = tmp_path / "fake.apax"
    dst.mkdir()
    (dst / "config.json").write_text(json.dumps({"name": "mace", "r_max": 5.0}))
    (dst / "params.msgpack").write_bytes(b"\x80")  # non-empty placeholder
    (dst / "metadata.json").write_text("{}")
    return dst


def test_ase_calc_rejects_foundation_dir_with_clear_hint(tmp_path):
    """Until P3.4 Step 3 lands, the calculator must raise with guidance."""
    from apax.md.ase_calc import ASECalculator

    dst = _write_fake_foundation_dir(tmp_path)
    with pytest.raises(NotImplementedError, match="MaceFoundationEnergyModel"):
        ASECalculator(dst)
