import pathlib
import shutil

import orbax.checkpoint as ocp

from apax.md import ASECalculator
from tests.conftest import initialize_model, load_and_dump_config

TEST_PATH = pathlib.Path(__file__).parent.resolve()


def test_moved_model_loading(get_tmp_path, get_sample_input, monkeypatch):
    monkeypatch.chdir(get_tmp_path)
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

        options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=1)
        with ocp.CheckpointManager(ckpt_dir.resolve(), options=options) as mngr:
            mngr.save(0, args=ocp.args.StandardSave(ckpt))
        # attempt to load models
        _ = ASECalculator(ckpt_dir.parent)


def test_model_loading(get_tmp_path, get_sample_input, monkeypatch):
    monkeypatch.chdir(get_tmp_path)
    model_config_path = TEST_PATH / "config.yaml"

    # 1. Create original directory and save model
    original_path = get_tmp_path / "original"
    original_model_dir = original_path / "models" / "my_experiment"
    original_model_dir.mkdir(parents=True)

    model_config = load_and_dump_config(model_config_path, original_model_dir)
    # The following two lines will be written to the config.yaml,
    # but they will be invalid after moving the directory.
    model_config.data.directory = str(original_path / "models")
    model_config.data.experiment = "my_experiment"
    model_config.dump_config(original_model_dir)

    inputs, _ = get_sample_input
    _, params = initialize_model(model_config, inputs)

    ckpt = {"model": {"params": params}, "epoch": 0}
    ckpt_dir = original_model_dir / "best"
    ckpt_dir.mkdir()

    options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=1)
    with ocp.CheckpointManager(str(ckpt_dir.resolve()), options=options) as mngr:
        mngr.save(0, args=ocp.args.StandardSave(ckpt))

    # 2. Move the directory
    new_path = get_tmp_path / "new"
    shutil.move(original_path, new_path)

    # 3. Try to load from the new path
    moved_model_dir = new_path / "models" / "my_experiment"
    # Without the fix, this will fail because the config points to the old path.
    calc = ASECalculator(moved_model_dir)
    assert calc.n_models == 1
