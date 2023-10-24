import dataclasses
import logging
import os
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from ase import Atoms
from flax.core.frozen_dict import freeze
from keras.callbacks import CSVLogger, TensorBoard

from apax.config import parse_config
from apax.data.input_pipeline import TFPipeline, create_dict_dataset
from apax.data.statistics import compute_scale_shift_parameters
from apax.model import ModelBuilder
from apax.optimizer import get_opt
from apax.train.checkpoints import load_params
from apax.train.loss import Loss, LossCollection
from apax.train.metrics import initialize_metrics
from apax.train.trainer import fit
from apax.transfer_learning import param_transfer
from apax.utils.data import load_data, split_atoms, split_idxs, split_label
from apax.utils.random import seed_py_np_tf

log = logging.getLogger(__name__)


def initialize_directories(model_version_path: Path) -> None:
    log.info("Initializing directories")
    os.makedirs(model_version_path, exist_ok=True)


@dataclasses.dataclass
class RawDataset:
    atoms_list: list[Atoms]
    additional_labels: Optional[dict] = None


def load_data_files(
    data_config, model_version_path, train_atoms_list=None, val_atoms_list=None
):
    log.info("Running Input Pipeline")
    if train_atoms_list is not None and val_atoms_list is not None:
        train_label_dict = None
        val_label_dict = None

    elif data_config.data_path is not None:
        log.info(f"Read data file {data_config.data_path}")
        atoms_list, label_dict = load_data(data_config.data_path)

        train_idxs, val_idxs = split_idxs(
            atoms_list, data_config.n_train, data_config.n_valid
        )
        train_atoms_list, val_atoms_list = split_atoms(atoms_list, train_idxs, val_idxs)
        train_label_dict, val_label_dict = split_label(label_dict, train_idxs, val_idxs)

        np.savez(
            os.path.join(model_version_path, "train_val_idxs"),
            train_idxs=train_idxs,
            val_idxs=val_idxs,
        )

    elif data_config.train_data_path and data_config.val_data_path is not None:
        log.info(f"Read training data file {data_config.train_data_path}")
        log.info(f"Read validation data file {data_config.val_data_path}")
        train_atoms_list, train_label_dict = load_data(data_config.train_data_path)
        val_atoms_list, val_label_dict = load_data(data_config.val_data_path)
    else:
        raise ValueError("eighter define input data path/paths or atoms_lists")

    train_raw_ds = RawDataset(
        atoms_list=train_atoms_list, additional_labels=train_label_dict
    )
    val_raw_ds = RawDataset(atoms_list=val_atoms_list, additional_labels=val_label_dict)

    return train_raw_ds, val_raw_ds


def initialize_dataset(config, raw_ds, calc_stats: bool = True):
    inputs, labels = create_dict_dataset(
        raw_ds.atoms_list,
        r_max=config.model.r_max,
        external_labels=raw_ds.additional_labels,
        disable_pbar=config.progress_bar.disable_nl_pbar,
        pos_unit=config.data.pos_unit,
        energy_unit=config.data.energy_unit,
    )

    if calc_stats:
        ds_stats = compute_scale_shift_parameters(
            inputs,
            labels,
            config.data.shift_method,
            config.data.scale_method,
            config.data.shift_options,
            config.data.scale_options,
        )

    dataset = TFPipeline(
        inputs,
        labels,
        config.n_epochs,
        config.data.batch_size,
        buffer_size=config.data.shuffle_buffer_size,
    )

    if calc_stats:
        return dataset, ds_stats
    else:
        return dataset


def maximize_l2_cache():
    import ctypes

    _libcudart = ctypes.CDLL("libcudart.so")
    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128


def initialize_callbacks(callback_configs, model_version_path):
    log.info("Initializing Callbacks")

    dummy_model = tf.keras.Model()
    callback_dict = {
        "csv": {
            "class": CSVLogger,
            "log_path": model_version_path / "log.csv",
            "path_arg_name": "filename",
            "kwargs": {"append": True},
            "model": dummy_model,
        },
        "tensorboard": {
            "class": TensorBoard,
            "log_path": model_version_path,
            "path_arg_name": "log_dir",
            "kwargs": {},
            "model": dummy_model,
        },
    }

    callback_configs = [config.name for config in callback_configs]
    if "csv" in callback_configs and "tensorboard" in callback_configs:
        csv_idx, tb_idx = callback_configs.index("csv"), callback_configs.index(
            "tensorboard"
        )
        msg = (
            "Using both csv and tensorboard callbacks is not supported at the moment."
            " Only the first of the two will be used."
        )
        print("Warning: " + msg)
        log.warning(msg)
        if csv_idx < tb_idx:
            callback_configs.pop(tb_idx)
        else:
            callback_configs.pop(csv_idx)

    callbacks = []
    for callback_config in callback_configs:
        callback_info = callback_dict[callback_config]

        path_arg_name = callback_info["path_arg_name"]
        path = {path_arg_name: callback_info["log_path"]}

        kwargs = callback_info["kwargs"]
        callback = callback_info["class"](**path, **kwargs)
        callback.set_model(callback_info["model"])
        callbacks.append(callback)

    return tf.keras.callbacks.CallbackList([callback])


def initialize_loss_fn(loss_config_list):
    log.info("Initializing Loss Function")
    loss_funcs = []
    for loss in loss_config_list:
        loss_funcs.append(Loss(**loss.model_dump()))
    return LossCollection(loss_funcs)


def setup_logging(log_file, log_level):
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])

    logging.basicConfig(filename=log_file, level=log_levels[log_level])


def run(
    user_config,
    log_file="train.log",
    log_level="error",
    train_atoms_list=None,
    val_atoms_list=None,
):
    setup_logging(log_file, log_level)
    log.info("Loading user config")
    config = parse_config(user_config)

    seed_py_np_tf(config.seed)
    rng_key = jax.random.PRNGKey(config.seed)
    if config.maximize_l2_cache:
        maximize_l2_cache()

    experiment = Path(config.data.experiment)
    directory = Path(config.data.directory)
    model_version_path = directory / experiment

    initialize_directories(model_version_path)
    config.dump_config(model_version_path)

    callbacks = initialize_callbacks(config.callbacks, model_version_path)
    loss_fn = initialize_loss_fn(config.loss)
    Metrics = initialize_metrics(config.metrics)

    train_raw_ds, val_raw_ds = load_data_files(
        config.data, model_version_path, train_atoms_list, val_atoms_list
    )
    train_ds, ds_stats = initialize_dataset(config, train_raw_ds)
    val_ds = initialize_dataset(config, val_raw_ds, calc_stats=False)

    log.info("Initializing Model")
    init_input = train_ds.init_input()
    R, Z, idx, init_box, offsets = (
        jnp.asarray(init_input["positions"][0]),
        jnp.asarray(init_input["numbers"][0]),
        jnp.asarray(init_input["idx"][0]),
        np.array(init_input["box"][0]),
        jnp.array(init_input["offsets"][0]),
    )

    # TODO n_species should be optional since it's already
    # TODO determined by the shape of shift and scale
    builder = ModelBuilder(config.model.get_dict(), n_species=ds_stats.n_species)
    model = builder.build_energy_derivative_model(
        scale=ds_stats.elemental_scale,
        shift=ds_stats.elemental_shift,
        apply_mask=True,
        init_box=init_box,
    )

    rng_key, model_rng_key = jax.random.split(rng_key, num=2)
    params = model.init(model_rng_key, R, Z, idx, init_box, offsets)
    params = freeze(params)

    base_checkpoint = config.checkpoints.base_model_checkpoint
    do_transfer_learning = base_checkpoint is not None
    if do_transfer_learning:
        source_params = load_params(base_checkpoint)
        log.info("Transferring parameters from %s", base_checkpoint)
        params = param_transfer(source_params, params, config.checkpoints.reset_layers)

    batched_model = jax.vmap(model.apply, in_axes=(None, 0, 0, 0, 0, 0))

    steps_per_epoch = train_ds.steps_per_epoch()
    n_epochs = config.n_epochs
    transition_steps = steps_per_epoch * n_epochs - config.optimizer.transition_begin
    tx = get_opt(
        params,
        transition_steps=transition_steps,
        **config.optimizer.model_dump(),
    )

    fit(
        batched_model,
        params,
        tx,
        train_ds,
        loss_fn,
        Metrics,
        callbacks,
        n_epochs,
        ckpt_dir=os.path.join(config.data.directory, config.data.experiment),
        ckpt_interval=config.checkpoints.ckpt_interval,
        val_ds=val_ds,
        sam_rho=config.optimizer.sam_rho,
        patience=config.patience,
        disable_pbar=config.progress_bar.disable_epoch_pbar,
    )
