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
from gmnn_jax.data.input_pipeline import (
    TFPipeline,
    create_dict_dataset,
    initialize_nbr_displacement_fns,
)
from gmnn_jax.data.statistics import energy_per_element
from gmnn_jax.model.gmnn import get_training_model
from gmnn_jax.train.metrics import initialize_metrics
from gmnn_jax.train.run import (
    find_largest_system,
    initialize_callbacks,
    initialize_loss_fn,
)
from gmnn_jax.train.trainer import make_step_fns
from gmnn_jax.utils.data import load_data, split_atoms, split_label
from gmnn_jax.utils.random import seed_py_np_tf

log = logging.getLogger(__name__)


def get_test_idxs(atoms_list, used_idxs, n_test=-1):
    idxs = np.arange(len(atoms_list))
    test_idxs = np.setdiff1d(idxs, used_idxs)
    np.random.shuffle(test_idxs)
    if n_test != -1:
        test_idxs = test_idxs[:n_test]

    return test_idxs


def load_test_data(
    config, model_version_path, eval_path, n_test=-1
):  # TODO double code run.py in progress
    log.info("Running Input Pipeline")
    if config.data.data_path is not None:
        log.info(f"Read data file {config.data.data_path}")
        atoms_list, label_dict = load_data(config.data.data_path)

        idxs_dict = np.load(model_version_path / "train_val_idxs.npz")

        used_idxs = idxs_dict["train_idxs"]
        np.append(used_idxs, idxs_dict["val_idxs"])

        test_idxs = get_test_idxs(atoms_list, used_idxs, n_test)

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


def initialize_test_dataset(test_atoms_list, test_label_dict, config):
    ds_stats = energy_per_element(
        test_atoms_list, lambd=config.data.energy_regularisation
    )
    displacement_fn, neighbor_fn = initialize_nbr_displacement_fns(
        test_atoms_list[0], config.model.r_max
    )
    ds_stats.displacement_fn = displacement_fn

    test_inputs, test_labels = create_dict_dataset(
        test_atoms_list,
        neighbor_fn,
        test_label_dict,
        disable_pbar=config.progress_bar.disable_nl_pbar,
        pos_unit=config.data.pos_unit,
        energy_unit=config.data.energy_unit,
    )

    max_atoms, max_nbrs = find_largest_system([test_inputs])
    ds_stats.n_atoms = max_atoms

    test_ds = TFPipeline(
        test_inputs,
        test_labels,
        1,
        config.data.batch_size,
        max_atoms=max_atoms,
        max_nbrs=max_nbrs,
        buffer_size=config.data.shuffle_buffer_size,
    )

    return test_ds, ds_stats


def load_params(model_version_path):
    best_dir = model_version_path / "best"
    log.info(f"load checkpoint from {best_dir}")
    raw_restored = checkpoints.restore_checkpoint(best_dir, target=None, step=None)
    params = jax.tree_map(jnp.asarray, raw_restored["model"]["params"])

    return params


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


def eval_model(config_path, n_test=-1, log_file="eval.log", log_level="error"):
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logging.basicConfig(filename=log_file, level=log_levels[log_level])

    log.info("Start model evaluation")
    log.info("Loading user config")
    if isinstance(config_path, (str, os.PathLike)):
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)

    config = Config.parse_obj(config)

    seed_py_np_tf(config.seed)

    model_version_path = Path(config.data.model_path) / config.data.model_name
    eval_path = model_version_path / "eval"

    callbacks = initialize_callbacks(config.callbacks, eval_path)
    loss_fn = initialize_loss_fn(config.loss)
    Metrics = initialize_metrics(config.metrics)

    test_atoms_list, test_label_dict = load_test_data(
        config, model_version_path, eval_path, n_test
    )

    test_ds, ds_stats = initialize_test_dataset(test_atoms_list, test_label_dict, config)

    model_dict = config.model.get_dict()

    gmnn = get_training_model(
        n_atoms=ds_stats.n_atoms,
        n_species=ds_stats.n_species,
        displacement_fn=ds_stats.displacement_fn,
        elemental_energies_mean=ds_stats.elemental_shift,
        elemental_energies_std=ds_stats.elemental_scale,
        **model_dict,
    )

    model = jax.vmap(gmnn.apply, in_axes=(None, 0, 0, 0))

    params = load_params(model_version_path)

    predict(model, params, Metrics, loss_fn, test_ds, callbacks)
