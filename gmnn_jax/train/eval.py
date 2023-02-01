import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.training import checkpoints
from tqdm import trange

from gmnn_jax.config import Config
from gmnn_jax.data.input_pipeline import InputPipeline
from gmnn_jax.model.gmnn import get_training_model
from gmnn_jax.train.run import (
    initialize_callbacks,
    initialize_loss_fn,
    initialize_metrics,
)
from gmnn_jax.train.trainer import make_step_fns
from gmnn_jax.utils.data import load_data, split_atoms, split_label

log = logging.getLogger(__name__)


def get_test_idxs(atoms_list, used_idxs, n_test):
    idxs = np.arange(len(atoms_list))
    test_idxs = np.setdiff1d(used_idxs, idxs)
    np.random.shuffle(idxs)
    test_idxs = idxs[:n_test]

    return test_idxs


def get_test_data(
    config, model_version_path, eval_path, n_test: str = None
):  # TODO double code run.py in progress
    log.info("Running Input Pipeline")
    if config.data.data_path is not None:
        log.info(f"Read data file {config.data.data_path}")
        atoms_list, label_dict = load_data(config.data.data_path)

        idxs_dict = np.load(model_version_path / "train_val_idxs.npz")

        used_idxs = idxs_dict["train_idxs"]
        np.append(used_idxs, idxs_dict["val_idxs"])

        if n_test is not None:
            test_idxs = get_test_idxs(atoms_list, used_idxs, n_test)
        else:
            raise ValueError("n_test number of test structures not defined")

        os.makedirs(eval_path, exist_ok=True)
        np.savez(
            os.path.join(eval_path, "test_idxs"),
            test_idxs=test_idxs,
        )

        test_atoms_list, _ = split_atoms(atoms_list, test_idxs)
        test_label_dict, _ = split_label(label_dict, test_idxs)

    elif config.data.test_data_path is not None:
        log.info(f"Read test data file {config.data.test_data_path}")
        test_atoms_list, test_label_dict = load_data(config.data.test_data_path)
    else:
        raise ValueError("input data path/paths not defined")

    return test_atoms_list, test_label_dict


def load_params(model_version_path):
    best_dir = model_version_path / "best"
    log.info(f"load checkpoint from {best_dir}")
    raw_restored = checkpoints.restore_checkpoint(best_dir, target=None, step=None)
    params = jax.tree_map(jnp.asarray, raw_restored["model"]["params"])

    return params


def init_metrics(config):  # TODO double code run.py in progress
    keys = []
    reductions = []
    for metric in config.metrics:
        for reduction in metric.reductions:
            keys.append(metric.name)
            reductions.append(reduction)
    Metrics = initialize_metrics(keys, reductions)

    return Metrics


def predict(model, params, Metrics, loss_fn, test_ds, callbacks):
    callbacks.on_train_begin()
    _, test_step_fn = make_step_fns(loss_fn, Metrics, model=model)

    test_steps_per_epoch = test_ds.steps_per_epoch()
    batch_test_ds = test_ds.shuffle_and_batch()

    epoch_loss = {}
    epoch_start_time = time.time()

    epoch_loss.update({"test_loss": 0.0})
    test_metrics = Metrics.empty()
    with trange(
        0, test_steps_per_epoch, desc="Batches", ncols=100, disable=False, leave=True
    ) as batch_pbar:
        for batch_idx in range(test_steps_per_epoch):
            inputs, labels = next(batch_test_ds)

            test_metrics, batch_loss = test_step_fn(params, inputs, labels, test_metrics)

            epoch_loss["test_loss"] += batch_loss
            batch_pbar.set_postfix(test_loss=epoch_loss["test_loss"] / batch_idx)
            batch_pbar.update()

    epoch_loss["test_loss"] /= test_steps_per_epoch
    epoch_loss["test_loss"] = float(epoch_loss["test_loss"])
    epoch_metrics = {
        f"test_{key}": float(val) for key, val in test_metrics.compute().items()
    }
    epoch_metrics.update({**epoch_loss})
    epoch_end_time = time.time()
    epoch_metrics.update({"epoch_time": epoch_end_time - epoch_start_time})
    callbacks.on_epoch_end(epoch=1, logs=epoch_metrics)
    callbacks.on_train_end()
    pass


def eval_model(config_path, n_test=None):
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    config = Config.parse_obj(config)
    model_version_path = Path(config.data.model_path) / config.data.model_name
    eval_path = model_version_path / "testset_eval"

    callbacks = initialize_callbacks(config, eval_path)

    loss_fn = initialize_loss_fn(config)
    Metrics = init_metrics(config)

    test_atoms_list, test_label_dict = get_test_data(
        config, model_version_path, eval_path, n_test
    )

    test_ds = InputPipeline(
        config.model.r_max,
        1,
        config.data.valid_batch_size,
        test_atoms_list,
        test_label_dict,
        config.data.shuffle_buffer_size,
        disable_pbar=config.progress_bar.disable_nl_pbar,
    )

    numbers = [atoms.numbers for atoms in test_atoms_list]
    system_sizes = [num.shape[0] for num in numbers]
    n_atoms = np.max(system_sizes)
    n_species = max([max(n) for n in numbers]) + 1

    gmnn = get_training_model(
        n_atoms=n_atoms,
        n_species=n_species,
        displacement_fn=test_ds.displacement_fn,
        **config.model.dict(),
    )
    model = jax.vmap(gmnn.apply, in_axes=(None, 0, 0, 0))

    params = load_params(model_version_path)

    predict(model, params, Metrics, loss_fn, test_ds, callbacks)