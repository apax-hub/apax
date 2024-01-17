import pathlib

import pytest
from flax.training import checkpoints

from apax.md import ASECalculator
from tests.conftest import initialize_model, load_and_dump_config

TEST_PATH = pathlib.Path(__file__).parent.resolve()


def test_model_loading(get_tmp_path, get_sample_input):
    pytest.MonkeyPatch().chdir(get_tmp_path)
    model_config_path = TEST_PATH / "config.yaml"

    model_config = load_and_dump_config(model_config_path, get_tmp_path)

    inputs, _ = get_sample_input

    _, params = initialize_model(model_config, inputs)

    ckpt = {"model": {"params": params}, "epoch": 0}
    ckpt_dir1 = pathlib.Path("models/apax_dummy/best")
    ckpt_dir2 = pathlib.Path("../models/apax_dummy/best")
    ckpt_dir3 = pathlib.Path("models/best")

    ckpt_dirs = [ckpt_dir1, ckpt_dir2, ckpt_dir3]
    experiments = ["apax_dummy", "apax_dummy", ""]
    for ckpt_dir, exp in zip(ckpt_dirs, experiments):
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        model_config.data.experiment = exp
        model_config.dump_config(ckpt_dir.parent)

        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir.resolve(),
            target=ckpt,
            step=0,
            overwrite=True,
        )
        # attempt to load models
        _ = ASECalculator(ckpt_dir.parent)
