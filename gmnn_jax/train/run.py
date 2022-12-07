import logging
import os
import uuid
from pathlib import Path

import jax
import yaml
from keras.callbacks import CSVLogger, TensorBoard

from gmnn_jax.config import Config
from gmnn_jax.data.statistics import energy_per_element
from gmnn_jax.model.gmnn import get_training_model
from gmnn_jax.optimizer import get_opt
from gmnn_jax.train.loss import Loss, LossCollection
from gmnn_jax.train.metrics import initialize_metrics
from gmnn_jax.train.trainer import Trainer
from gmnn_jax.utils.random import seed_py_np_tf

log = logging.getLogger(__name__)


def init_directories(model_name, model_path):
    log.info("Initializing directories")
    if model_name is None:
        # creates an unique id for job
        directory = str(uuid.uuid4())
    else:
        directory = model_name

    model_version_path = os.path.join(model_path, directory)
    os.makedirs(model_version_path, exist_ok=True)
    return model_version_path


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

    return callbacks


def initialize_loss_fn(config):
    log.info("Initializing Loss Function")
    loss_funcs = []
    for loss in config.loss:
        loss_funcs.append(Loss(**loss.dict()))
    return LossCollection(loss_funcs)


def run(user_config):
    log.info("Loading user config")
    if isinstance(user_config, str):
        with open(user_config, "r") as stream:
            user_config = yaml.safe_load(stream)

    config = Config.parse_obj(user_config)

    seed_py_np_tf(config.seed)
    rng_key = jax.random.PRNGKey(config.seed)

    model_version_path = init_directories(config.data.model_name, config.data.model_path)
    config.dump_config(model_version_path)

    callbacks = initialize_callbacks(config, model_version_path)

    loss_fn = initialize_loss_fn(config)

    keys = []
    reductions = []
    for metric in config.metrics:
        for reduction in metric.reductions:
            keys.append(metric.name)
            reductions.append(reduction)
    Metrics = initialize_metrics(keys, reductions)

    log.info("Running Input Pipeline")
    # input pipeline
    # read atoms_list
    atoms_list = None
    datasets = None
    elemental_energies_mean, elemental_energies_std = energy_per_element(
        atoms_list,
        lambd=config.data.energy_regularisation
    )

    # preliminary, needs to be retrievable from the ds
    n_atoms = datasets.n_atoms
    n_species = datasets.n_species
    displacement_fn = datasets.displacement_fn
    model_init, model = get_training_model(
        n_atoms=n_atoms,
        n_species=n_species,
        displacement_fn=displacement_fn,
        elemental_energies_mean=elemental_energies_mean,
        elemental_energies_std=elemental_energies_std,
        **config.model.dict()
    )
    # sample_inputs, _ = next(train_ds.take(1).as_numpy_iterator())
    # rng_key, model_rng_key = jax.random.split(rng_key, num=2)
    # params = model_init(model_rng_key, sample_data)

    # preliminary, need to get steps per epoch from ds
    n_steps = datasets.steps_per_epoch
    n_epochs = config.num_epochs
    n_warmup = config.optimizer.transition_begin
    transition_steps = n_steps * n_epochs - n_warmup
    tx = get_opt(transition_steps=transition_steps, **config.optimizer.dict())

    log.info("Begining Training")
    # , params # needs to be passed to trainer
    trainer = Trainer(model, tx, datasets, loss_fn, Metrics, callbacks)
    trainer.fit()
