import pathlib

import pytest
from flax.training import checkpoints

from apax.bal.api import kernel_selection
from tests.conftest import initialize_model, load_and_dump_config

TEST_PATH = pathlib.Path(__file__).parent.resolve()


@pytest.mark.parametrize(
    "num_data, pbc, calc_results",
    ([20, False, ["energy", "forces"]],),
)
def test_kernel_selection(example_atoms, get_tmp_path, get_sample_input):
    model_config_path = TEST_PATH / "config.yaml"

    model_config = load_and_dump_config(model_config_path, get_tmp_path)

    inputs, _ = get_sample_input

    _, params = initialize_model(model_config, inputs)

    ckpt = {"model": {"params": params}, "epoch": 0}
    best_dir = model_config.data.best_model_path()
    checkpoints.save_checkpoint(
        ckpt_dir=best_dir,
        target=ckpt,
        step=0,
        overwrite=True,
    )

    num_data = len(example_atoms)
    n_train = num_data // 2
    train_atoms = example_atoms[:n_train]
    pool_atoms = example_atoms[n_train:]

    base_fm_options = {"name": "ll_grad", "layer_name": "dense_2"}
    selection_method = "max_dist"
    bs = 5

    selected_indices = kernel_selection(
        model_config.data.model_version_path,
        train_atoms,
        pool_atoms,
        base_fm_options,
        selection_method,
        selection_batch_size=bs,
        processing_batch_size=bs,
    )
    assert len(selected_indices) == bs
