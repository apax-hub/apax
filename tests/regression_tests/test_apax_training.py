import pathlib
import uuid

import numpy as np
import pytest

from tests.conftest import load_config_and_run_training

TEST_PATH = pathlib.Path(__file__).parent.resolve()


def load_csv(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)  # Skip the header row

    with open(filename, "r") as file:
        header = file.readline().strip().split(",")

    data_dict = {header[i]: data[:, i].tolist() for i in range(len(header))}

    return data_dict


@pytest.mark.slow
def test_regression_model_training(get_md22_stachyose, get_tmp_path):
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

    current_metrics = load_csv(working_dir / "test/log.csv")

    comparison_metrics = {
        "val_energy_mae": 0.075,
        "val_forces_mae": 0.045,
        "val_forces_mse": 0.004,
        "val_loss": 0.045,
    }

    for key in comparison_metrics.keys():
        print((np.array(current_metrics[key])[-1]))

    for key in comparison_metrics.keys():
        assert abs((np.array(current_metrics[key])[-1] - comparison_metrics[key])) < 0.01
