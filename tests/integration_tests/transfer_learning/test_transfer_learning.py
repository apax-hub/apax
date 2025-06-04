import pathlib
import uuid

import jax
import numpy as np
import pytest
from ase.io import write
from flax.traverse_util import flatten_dict

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
        "transfer_learning": {"base_model_checkpoint": (working_dir / "base").as_posix()},
    }
    load_config_and_run_training(config_ft_path, data_config_mods)

    # Compare parameters
    _, base_params = restore_parameters(working_dir / "base")
    _, ft_params = restore_parameters(working_dir / "fine_tune")

    flat_before = flatten_dict(base_params, sep="/")
    flat_after = flatten_dict(ft_params, sep="/")

    layer = "dense_0"

    for path, p_before in flat_before.items():
        p_after = flat_after[path]
        same = np.allclose(p_before, p_after)
        should_be_same = layer in path

        if should_be_same:
            assert same
        elif not should_be_same:
            assert not same
