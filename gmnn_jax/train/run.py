import dataclasses
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import yaml
from keras.callbacks import CSVLogger, TensorBoard

from gmnn_jax.config import Config
from gmnn_jax.data.input_pipeline import InputPipeline
from gmnn_jax.data.statistics import energy_per_element
from gmnn_jax.model.gmnn import get_training_model
from gmnn_jax.optimizer import get_opt
from gmnn_jax.train.loss import Loss, LossCollection
from gmnn_jax.train.metrics import initialize_metrics
from gmnn_jax.train.trainer import fit
from gmnn_jax.utils.data import load_data, split_atoms, split_idxs, split_label
from gmnn_jax.utils.random import seed_py_np_tf

log = logging.getLogger(__name__)


def init_directories(model_name, model_path):
    log.info("Initializing directories")
    model_version_path = os.path.join(model_path, model_name)
    os.makedirs(model_version_path, exist_ok=True)
    return model_version_path


@dataclasses.dataclass
class TFModelSpoof:
    stop_training = False


def initialize_callbacks(config, model_version_path):
    log.info("Initializing Callbacks")
    callback_dict = {
        "csv": {
            "class": CSVLogger,
            "path": "log.csv",
            "path_arg_name": "filename",
            "kwargs": {"append": True},
        },
        "tensorboard": {
            "class": TensorBoard,
            "path": "tb_logs",
            "path_arg_name": "log_dir",
            "kwargs": {},
        },
    }
    callbacks = []
    for callback_config in config.callbacks:
        callback_info = callback_dict[callback_config.name]

        log_path = os.path.join(model_version_path, callback_info["path"])
        path_arg_name = callback_info["path_arg_name"]
        path = {path_arg_name: log_path}

        kwargs = callback_info["kwargs"]
        callback = callback_info["class"](**path, **kwargs)
        callbacks.append(callback)

    return tf.keras.callbacks.CallbackList([callback], model=TFModelSpoof())


def initialize_loss_fn(config):
    log.info("Initializing Loss Function")
    loss_funcs = []
    for loss in config.loss:
        loss_funcs.append(Loss(**loss.dict()))
    return LossCollection(loss_funcs)


def run(user_config, log_file="train.log", log_level="error"):
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logging.basicConfig(filename=log_file, level=log_levels[log_level])

    log.info("Loading user config")
    if isinstance(user_config, (str, os.PathLike)):
        with open(user_config, "r") as stream:
            user_config = yaml.safe_load(stream)

    config = Config.parse_obj(user_config)

    seed_py_np_tf(config.seed)
    rng_key = jax.random.PRNGKey(config.seed)

    model_version_path = init_directories(config.data.model_name, config.data.model_path)
    config.dump_config(model_version_path)

    callbacks = initialize_callbacks(config, model_version_path)

    if config.maximize_l2_cache:
        import ctypes

        _libcudart = ctypes.CDLL("libcudart.so")
        # Set device limit on the current device
        # cudaLimitMaxL2FetchGranularity = 0x05
        pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
        assert pValue.contents.value == 128

    loss_fn = initialize_loss_fn(config)

    keys = []
    reductions = []
    for metric in config.metrics:
        for reduction in metric.reductions:
            keys.append(metric.name)
            reductions.append(reduction)
    Metrics = initialize_metrics(keys, reductions)

    log.info("Running Input Pipeline")
    if config.data.data_path is not None:
        log.info(f"Read data file {config.data.data_path}")
        atoms_list, label_dict = load_data(config.data.data_path)

        train_idxs, val_idxs = split_idxs(
            atoms_list, config.data.n_train, config.data.n_valid
        )
        train_atoms_list, val_atoms_list = split_atoms(atoms_list, train_idxs, val_idxs)
        train_label_dict, val_label_dict = split_label(label_dict, train_idxs, val_idxs)

        np.savez(
            os.path.join(model_version_path, "train_val_idxs"),
            train_idxs=train_idxs,
            val_idxs=val_idxs,
        )

    elif config.data.train_data_path and config.data.val_data_path is not None:
        log.info(f"Read training data file {config.data.train_data_path}")
        log.info(f"Read validation data file {config.data.val_data_path}")
        train_atoms_list, train_label_dict = load_data(config.data.train_data_path)
        val_atoms_list, val_label_dict = load_data(config.data.val_data_path)
    else:
        raise ValueError("input data path/paths not defined")

    ds_stats = energy_per_element(
        train_atoms_list, lambd=config.data.energy_regularisation
    )

    train_ds = InputPipeline(
        config.model.r_max,
        config.n_epochs,
        config.data.batch_size,
        train_atoms_list,
        train_label_dict,
        config.data.shuffle_buffer_size,
        disable_pbar=config.progress_bar.disable_nl_pbar,
    )
    val_ds = InputPipeline(
        config.model.r_max,
        config.n_epochs,
        config.data.valid_batch_size,
        val_atoms_list,
        val_label_dict,
        config.data.shuffle_buffer_size,
        disable_pbar=config.progress_bar.disable_nl_pbar,
    )

    n_atoms = ds_stats.n_atoms
    n_species = ds_stats.n_species
    model_dict = config.model.get_dict()

    gmnn = get_training_model(
        n_atoms=n_atoms,
        # ^This is going to make problems when training on differently sized molecules.
        # we may need to check batch shapes and manually initialize a new model
        # when a new size is encountered...
        n_species=n_species,
        displacement_fn=train_ds.displacement_fn,
        elemental_energies_mean=ds_stats.elemental_shift,
        elemental_energies_std=ds_stats.elemental_scale,
        **model_dict,
    )
    log.info("Initializing Model")
    init_input, _ = train_ds.init_input()
    R, Z, idx = (
        jnp.asarray(init_input["positions"][0]),
        jnp.asarray(init_input["numbers"][0]),
        jnp.asarray(init_input["idx"][0]),
    )

    rng_key, model_rng_key = jax.random.split(rng_key, num=2)
    params = gmnn.init(model_rng_key, R, Z, idx)
    batched_model = jax.vmap(gmnn.apply, in_axes=(None, 0, 0, 0))

    steps_per_epoch = train_ds.steps_per_epoch()
    n_epochs = config.n_epochs
    n_warmup = config.optimizer.transition_begin
    transition_steps = steps_per_epoch * n_epochs - n_warmup
    tx = get_opt(transition_steps=transition_steps, **config.optimizer.dict())

    fit(
        batched_model,
        params,
        tx,
        train_ds,
        loss_fn,
        Metrics,
        callbacks,
        n_epochs,
        ckpt_dir=os.path.join(config.data.model_path, config.data.model_name),
        ckpt_interval=config.checkpoints.ckpt_interval,
        val_ds=val_ds,
        disable_pbar=config.progress_bar.disable_epoch_pbar,
    )
