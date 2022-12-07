import logging
import os
import uuid
from pathlib import Path

import jax
import yaml
from keras.callbacks import CSVLogger, TensorBoard

from gmnn_jax.config import Config
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

    # init model
    model = None
    transition_steps = 5  # preliminary, need to get steps per epoch from ds
    tx = get_opt(transition_steps=transition_steps, **config.optimizer.dict())

    callbacks = initialize_callbacks(config, model_version_path)
    # init checkpoints? maybe in trainer

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
    datasets = None

    log.info("Begining Training")
    trainer = Trainer(model, tx, datasets, loss_fn, Metrics, callbacks)
    trainer.fit()
