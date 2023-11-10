import logging
import os
from pathlib import Path
from typing import List

import jax

from apax.config import LossConfig, parse_config
from apax.data.initialization import initialize_dataset, load_data_files
from apax.model import ModelBuilder
from apax.optimizer import get_opt
from apax.train.callbacks import initialize_callbacks
from apax.train.checkpoints import create_params, create_train_state, load_params
from apax.train.loss import Loss, LossCollection
from apax.train.metrics import initialize_metrics
from apax.train.trainer import fit
from apax.transfer_learning import param_transfer
from apax.utils.random import seed_py_np_tf

log = logging.getLogger(__name__)


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


def initialize_loss_fn(loss_config_list: List[LossConfig]) -> LossCollection:
    log.info("Initializing Loss Function")
    loss_funcs = []
    for loss in loss_config_list:
        loss_funcs.append(Loss(**loss.model_dump()))
    return LossCollection(loss_funcs)


def run(user_config, log_file="train.log", log_level="error"):
    setup_logging(log_file, log_level)
    log.info("Loading user config")
    config = parse_config(user_config)

    seed_py_np_tf(config.seed)
    rng_key = jax.random.PRNGKey(config.seed)

    experiment = Path(config.data.experiment)
    directory = Path(config.data.directory)
    model_version_path = directory / experiment
    log.info("Initializing directories")
    model_version_path.mkdir(parents=True, exist_ok=True)
    config.dump_config(model_version_path)

    callbacks = initialize_callbacks(config.callbacks, model_version_path)
    loss_fn = initialize_loss_fn(config.loss)
    Metrics = initialize_metrics(config.metrics)

    train_raw_ds, val_raw_ds = load_data_files(config.data, model_version_path)
    train_ds, ds_stats = initialize_dataset(config, train_raw_ds)
    val_ds = initialize_dataset(config, val_raw_ds, calc_stats=False)

    train_ds.set_batch_size(config.data.batch_size)
    val_ds.set_batch_size(config.data.valid_batch_size)

    log.info("Initializing Model")
    sample_input, init_box = train_ds.init_input()

    builder = ModelBuilder(config.model.get_dict())
    model = builder.build_energy_derivative_model(
        scale=ds_stats.elemental_scale,
        shift=ds_stats.elemental_shift,
        apply_mask=True,
        init_box=init_box,
    )
    batched_model = jax.vmap(model.apply, in_axes=(None, 0, 0, 0, 0, 0))

    params, rng_key = create_params(model, rng_key, sample_input, config.n_models)

    # TODO rework optimizer initialization and lr keywords
    steps_per_epoch = train_ds.steps_per_epoch()
    n_epochs = config.n_epochs
    transition_steps = steps_per_epoch * n_epochs - config.optimizer.transition_begin
    tx = get_opt(
        params,
        transition_steps=transition_steps,
        **config.optimizer.model_dump(),
    )

    state = create_train_state(batched_model, params, tx)

    base_checkpoint = config.checkpoints.base_model_checkpoint
    do_transfer_learning = base_checkpoint is not None
    if do_transfer_learning:
        source_params = load_params(base_checkpoint)
        log.info("Transferring parameters from %s", base_checkpoint)
        params = param_transfer(
            source_params, state.params, config.checkpoints.reset_layers
        )
        state.replace(params=params)

    fit(
        state,
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
        is_ensemble=config.n_models > 1,
    )
