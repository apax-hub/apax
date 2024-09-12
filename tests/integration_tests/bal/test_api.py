import pathlib

import pytest
from flax.training import checkpoints

from apax.bal.api import kernel_selection
from tests.conftest import initialize_model, load_and_dump_config

TEST_PATH = pathlib.Path(__file__).parent.resolve()


@pytest.mark.parametrize(
    "config",
    ["config.yaml", "config_shallow.yaml"],  # ["config_shallow.yaml"]
)
@pytest.mark.parametrize(
    "features",
    [
        {
            "name": "ll_grad",
            "layer_name": "dense_2",
        },
        {
            "name": "ll_force_feat",
            "strategy": "flatten",
        },
        {
            "name": "full_grad_rp",
            "num_rp": 100,
        },
    ],
)
@pytest.mark.parametrize(
    "num_data, pbc, calc_results",
    ([20, False, ["energy", "forces"]],),
)
def test_kernel_selection(
    config, features, example_atoms, get_tmp_path, get_sample_input
):
    model_config_path = TEST_PATH / config  # "config.yaml"

    model_config = load_and_dump_config(model_config_path, get_tmp_path)

    inputs, _ = get_sample_input

    _, params = initialize_model(model_config, inputs)

    ckpt = {"model": {"params": params}, "epoch": 0}
    best_dir = model_config.data.best_model_path
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

    base_fm_options = features
    selection_method = "max_dist"
    bs = 5

    ranking_indices, distances, g_train, g_pool = kernel_selection(
        model_config.data.model_version_path,
        train_atoms,
        pool_atoms,
        base_fm_options,
        selection_method,
        selection_batch_size=bs,
        processing_batch_size=1,
    )
    assert len(ranking_indices) == bs
