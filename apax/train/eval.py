import logging
import os
import time
from pathlib import Path

import jax
import numpy as np
from tqdm import trange

from apax.config import parse_config
from apax.data.input_pipeline import OTFInMemoryDataset
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
    """
    Load test data for evaluation.

    Parameters
    ----------
    config : object
        Configuration object.
    model_version_path : str
        Path to the model version.
    eval_path : str
        Path to evaluation directory.
    n_test : int, default = -1
        Number of test samples to load, by default -1 (load all).

    Returns
    -------
    atoms_list
        List of ase.Atoms containing the test data.
    """

    log.info("Running Input Pipeline")
    os.makedirs(eval_path, exist_ok=True)

    if config.data.test_data_path is not None:
        log.info(f"Read test data file {config.data.test_data_path}")
        atoms_list = load_data(config.data.test_data_path)
        atoms_list = atoms_list[:n_test]

    elif config.data.data_path is not None:
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

    else:
        raise ValueError("input data path/paths not defined")

    return atoms_list


def predict(model, params, Metrics, loss_fn, test_ds, callbacks, is_ensemble=False):
    """
    Perform predictions on the test dataset.

    Parameters
    ----------
    model :
        Trained model.
    params :
        Model parameters.
    Metrics :
        Collection of metrics.
    loss_fn :
        Loss function.
    test_ds :
        Test dataset.
    callbacks :
        Callback functions.
    is_ensemble : bool, default = False
        Whether the model is an ensemble.
    """

    callbacks.on_train_begin()
    _, test_step_fn = make_step_fns(
        loss_fn, Metrics, model=model, is_ensemble=is_ensemble
    )

    batch_test_ds = test_ds.batch()

    test_metrics = Metrics.empty()

    batch_pbar = trange(
        0, test_ds.n_data, desc="Structure", ncols=100, disable=False, leave=True
    )
    for batch_idx in range(test_ds.n_data):
        batch = next(batch_test_ds)
        batch_start_time = time.time()

        batch_loss, test_metrics = test_step_fn(params, batch, test_metrics)
        batch_metrics = {"test_loss": float(batch_loss)}
        batch_metrics.update(
            {f"test_{key}": float(val) for key, val in test_metrics.compute().items()}
        )
        batch_end_time = time.time()
        batch_metrics.update({"time": batch_end_time - batch_start_time})

        callbacks.on_test_batch_end(batch=batch_idx, logs=batch_metrics)

        batch_pbar.set_postfix(test_loss=batch_metrics["test_loss"])
        batch_pbar.update()
    batch_pbar.close()
    callbacks.on_train_end()


def eval_model(config_path, n_test=-1, log_file="eval.log", log_level="error"):
    """
    Evaluate the model using the provided configuration.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    n_test : int, default = -1
        Number of test samples to evaluate, by default -1 (evaluate all).
    log_file : str, default = "eval.log"
        Path to the log file.
    log_level : str, default = "error"
        Logging level.
    """

    setup_logging(log_file, log_level)
    log.info("Starting model evaluation")
    config = parse_config(config_path)

    seed_py_np_tf(config.seed)

    model_version_path = Path(config.data.directory) / config.data.experiment
    eval_path = model_version_path / "eval"

    callbacks = initialize_callbacks(config, eval_path)
    loss_fn = initialize_loss_fn(config.loss)
    Metrics = initialize_metrics(config.metrics)

    atoms_list = load_test_data(config, model_version_path, eval_path, n_test)
    test_ds = OTFInMemoryDataset(
        atoms_list,
        config.model.basis.r_max,
        1,
        config.data.valid_batch_size,
        pos_unit=config.data.pos_unit,
        energy_unit=config.data.energy_unit,
    )

    _, init_box = test_ds.init_input()

    if config.model.ensemble and config.model.ensemble.kind == "full":
        n_full_models = config.model.ensemble.n_members
    else:
        n_full_models = 1

    Builder = config.model.get_builder()
    builder = Builder(config.model.model_dump(), n_species=119)
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
        is_ensemble=n_full_models > 1,
    )
