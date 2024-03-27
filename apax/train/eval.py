import logging
import os
import time
from pathlib import Path

import jax
import numpy as np
from tqdm import trange

from apax.config import parse_config
from apax.data.input_pipeline import InMemoryDataset
from apax.model import ModelBuilder
from apax.train.callbacks import initialize_callbacks
from apax.train.checkpoints import restore_single_parameters
from apax.train.metrics import initialize_metrics
from apax.train.run import initialize_loss_fn, setup_logging
from apax.train.trainer import make_step_fns
from apax.utils.data import load_data, split_atoms
from apax.utils.random import seed_py_np_tf

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
    os.makedirs(eval_path, exist_ok=True)
    if config.data.data_path is not None:
        log.info(f"Read data file {config.data.data_path}")
        atoms_list = load_data(config.data.data_path)

        idxs_dict = np.load(model_version_path / "train_val_idxs.npz")

        used_idxs = idxs_dict["train_idxs"]
        np.append(used_idxs, idxs_dict["val_idxs"])

        test_idxs = get_test_idxs(atoms_list, used_idxs, n_test)

        np.savez(
            os.path.join(eval_path, "test_idxs"),
            test_idxs=test_idxs,
        )

        atoms_list, _ = split_atoms(atoms_list, test_idxs)

    elif config.data.test_data_path is not None:
        log.info(f"Read test data file {config.data.test_data_path}")
        atoms_list, label_dict = load_data(config.data.test_data_path)
        atoms_list = atoms_list[:n_test]
        for key, val in label_dict.items():
            label_dict[key] = val[:n_test]
    else:
        raise ValueError("input data path/paths not defined")

    return atoms_list


def predict(model, params, Metrics, loss_fn, test_ds, callbacks, is_ensemble=False):
    callbacks.on_train_begin()
    _, test_step_fn = make_step_fns(
        loss_fn, Metrics, model=model, sam_rho=0.0, is_ensemble=is_ensemble
    )

    test_steps_per_epoch = test_ds.steps_per_epoch()
    batch_test_ds = test_ds.shuffle_and_batch()

    epoch_loss = {}
    epoch_start_time = time.time()

    epoch_loss.update({"test_loss": 0.0})
    test_metrics = Metrics.empty()

    batch_pbar = trange(
        0, test_steps_per_epoch, desc="Batches", ncols=100, disable=False, leave=True
    )
    for batch_idx in range(test_steps_per_epoch):
        inputs, labels = next(batch_test_ds)

        batch_loss, test_metrics = test_step_fn(params, inputs, labels, test_metrics)

        epoch_loss["test_loss"] += batch_loss
        batch_pbar.set_postfix(test_loss=epoch_loss["test_loss"] / batch_idx)
        batch_pbar.update()
    batch_pbar.close()

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
    # TODO currently this has no informative output


def eval_model(config_path, n_test=-1, log_file="eval.log", log_level="error"):
    setup_logging(log_file, log_level)
    log.info("Starting model evaluation")
    config = parse_config(config_path)

    seed_py_np_tf(config.seed)

    model_version_path = Path(config.data.directory) / config.data.experiment
    eval_path = model_version_path / "eval"

    callbacks = initialize_callbacks(config.callbacks, eval_path)
    loss_fn = initialize_loss_fn(config.loss)
    Metrics = initialize_metrics(config.metrics)

    atoms_list = load_test_data(config, model_version_path, eval_path, n_test)
    test_ds = InMemoryDataset(
        atoms_list, config.model.r_max, config.data.valid_batch_size
    )

    _, init_box = test_ds.init_input()

    builder = ModelBuilder(config.model.get_dict())
    model = builder.build_energy_derivative_model(
        apply_mask=True,
        init_box=init_box,
    )

    model = jax.vmap(model.apply, in_axes=(None, 0, 0, 0, 0, 0))
    config, params = restore_single_parameters(model_version_path)

    predict(
        model,
        params,
        Metrics,
        loss_fn,
        test_ds,
        callbacks,
        is_ensemble=config.n_models > 1,
    )
