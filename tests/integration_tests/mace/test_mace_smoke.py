"""End-to-end smoke test for the MACE descriptor pipeline (P0 skeleton).

Trains a tiny MACE model (skeleton features) for 1 epoch on MD22 stachyose.
Correctness is not checked — only that the entire pipeline runs: config
validation → builder → data pipeline → trainer → checkpoint writing.
"""

import pathlib
import uuid

import pytest

from tests.conftest import load_config_and_run_training

TEST_PATH = pathlib.Path(__file__).parent.resolve()


@pytest.mark.slow
def test_mace_smoke_trains_one_epoch(get_md22_stachyose, get_tmp_path):
    """The MACE P0 skeleton can be built from YAML and trained end-to-end.

    Uses the same MD22 fixture as the GMNN regression test. 1 epoch, tiny
    model, no metric assertions — just pipeline integrity.
    """
    config_path = TEST_PATH / "apax_config.yaml"
    working_dir = get_tmp_path / str(uuid.uuid4())
    file_path = get_md22_stachyose

    data_config_mods = {
        "data": {
            "directory": working_dir.as_posix(),
            "data_path": file_path.as_posix(),
            "energy_unit": "kcal/mol",
        }
    }

    load_config_and_run_training(config_path, data_config_mods)

    # Verify expected outputs exist — matches what the GMNN test looks for
    assert (working_dir / "mace_smoke").exists()
