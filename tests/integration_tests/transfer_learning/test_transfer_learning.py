import pathlib
import uuid

import jax
import numpy as np
import pytest
from ase.io import write

from apax.train.checkpoints import restore_parameters
from tests.conftest import load_config_and_run_training

TEST_PATH = pathlib.Path(__file__).parent.resolve()


def l2_param_diff(p1, p2):
    p1, _ = jax.tree.flatten(p1)
    p2, _ = jax.tree.flatten(p2)
    diff = 0.0
    for i in range(len(p1)):
        diff += np.sum((p1[i] - p2[i]) ** 2)
    return diff


@pytest.mark.parametrize("num_data", (30,))
def test_transfer_learning(get_tmp_path, example_dataset):
    config_path = TEST_PATH / "config_base.yaml"
    config_ft_path = TEST_PATH / "config_ft.yaml"
    working_dir = get_tmp_path / str(uuid.uuid4())
    data_path = get_tmp_path / "ds.extxyz"

    write(data_path, example_dataset)

    data_config_mods = {
        "data": {
            "directory": working_dir.as_posix(),
            "experiment": "base",
            "data_path": data_path.as_posix(),
        },
    }
    load_config_and_run_training(config_path, data_config_mods)

    data_config_mods = {
        "data": {
            "directory": working_dir.as_posix(),
            "experiment": "fine_tune",
            "data_path": data_path.as_posix(),
        },
        "checkpoints": {"base_model_checkpoint": (working_dir / "base").as_posix()},
    }
    load_config_and_run_training(config_ft_path, data_config_mods)

    data_config_mods = {
        "data": {
            "directory": working_dir.as_posix(),
            "experiment": "fine_tune_no_pre_training",
            "data_path": data_path.as_posix(),
        },
    }
    load_config_and_run_training(config_ft_path, data_config_mods)

    # Compare parameters
    _, base_params = restore_parameters(working_dir / "base")
    _, ft_params = restore_parameters(working_dir / "fine_tune")
    _, ft_no_pre_params = restore_parameters(working_dir / "fine_tune_no_pre_training")

    diff_base_ft = l2_param_diff(base_params, ft_params)
    diff_base_no_pre = l2_param_diff(base_params, ft_no_pre_params)

    assert diff_base_ft < diff_base_no_pre
