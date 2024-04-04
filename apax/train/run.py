import logging
import sys
from typing import List

import jax

from apax.config import Config, LossConfig, parse_config
from apax.data.initialization import load_data_files
from apax.data.input_pipeline import dataset_dict
from apax.data.statistics import compute_scale_shift_parameters
from apax.model import ModelBuilder
from apax.optimizer import get_opt
from apax.train.callbacks import initialize_callbacks
from apax.train.checkpoints import create_params, create_train_state
from apax.train.loss import Loss, LossCollection
from apax.train.metrics import initialize_metrics
from apax.train.trainer import fit
from apax.transfer_learning import transfer_parameters
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

    logging.getLogger("absl").setLevel(logging.WARNING)

    logging.basicConfig(
        level=log_levels[log_level],
        format="%(levelname)s | %(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stderr)],
    )


def initialize_loss_fn(loss_config_list: List[LossConfig]) -> LossCollection:
    log.info("Initializing Loss Function")
    loss_funcs = []
    for loss in loss_config_list:
        loss_funcs.append(Loss(**loss.model_dump()))
    return LossCollection(loss_funcs)


def initialize_datasets(config: Config):
    train_raw_ds, val_raw_ds = load_data_files(config.data)

    Dataset = dataset_dict[config.data.ds_type]

    train_ds = Dataset(
        train_raw_ds,
        config.model.r_max,
        config.data.batch_size,
        config.n_epochs,
        config.data.shuffle_buffer_size,
        config.n_jitted_steps,
        config.data.pos_unit,
        config.data.energy_unit,
        pre_shuffle=True,
        cache_path=config.data.model_version_path,
    )
    val_ds = Dataset(
        val_raw_ds,
        config.model.r_max,
        config.data.valid_batch_size,
        config.n_epochs,
        pos_unit=config.data.pos_unit,
        energy_unit=config.data.energy_unit,
        cache_path=config.data.model_version_path,
    )
    ds_stats = compute_scale_shift_parameters(
        train_ds.inputs,
        train_ds.labels,
        config.data.shift_method,
        config.data.scale_method,
        config.data.shift_options,
        config.data.scale_options,
    )
    return train_ds, val_ds, ds_stats


def run(user_config, log_level="error"):
    config = parse_config(user_config)

    seed_py_np_tf(config.seed)
    rng_key = jax.random.PRNGKey(config.seed)

    config.data.model_version_path.mkdir(parents=True, exist_ok=True)
    setup_logging(config.data.model_version_path / "train.log", log_level)
    config.dump_config(config.data.model_version_path)
    log.info(f"Running on {jax.devices()}")

    callbacks = initialize_callbacks(config, config.data.model_version_path)
    loss_fn = initialize_loss_fn(config.loss)
    Metrics = initialize_metrics(config.metrics)

    train_ds, val_ds, ds_stats = initialize_datasets(config)

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
        state = transfer_parameters(state, config.checkpoints)

    fit(
        state,
        train_ds,
        loss_fn,
        Metrics,
        callbacks,
        n_epochs,
        ckpt_dir=config.data.model_version_path,
        ckpt_interval=config.checkpoints.ckpt_interval,
        val_ds=val_ds,
        sam_rho=config.optimizer.sam_rho,
        patience=config.patience,
        disable_pbar=config.progress_bar.disable_epoch_pbar,
        disable_batch_pbar=config.progress_bar.disable_batch_pbar,
        is_ensemble=config.n_models > 1,
    )
    log.info("Finished training")
