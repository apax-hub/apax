import logging
import os
import sys
from typing import List, Union

import jax

from apax.config import Config, LossConfig, parse_config
from apax.data.initialization import load_data_files
from apax.data.input_pipeline import dataset_dict
from apax.data.statistics import compute_scale_shift_parameters
from apax.optimizer import get_opt
from apax.train.callbacks import initialize_callbacks
from apax.train.checkpoints import create_params, create_train_state
from apax.train.loss import Loss, LossCollection
from apax.train.metrics import initialize_metrics
from apax.train.parameters import EMAParameters
from apax.train.trainer import fit
from apax.transfer_learning import transfer_parameters
from apax.utils.random import seed_py_np_tf

log = logging.getLogger(__name__)


def setup_logging(log_file, log_level):
    """
    Setup logging configuration.

    Parameters
    ----------
    log_file : str
        Path to the log file.
    log_level : str
        Logging level. Options: {'debug', 'info', 'warning', 'error', 'critical'}.
    """

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
    """
    Initialize loss functions based on configuration.

    Parameters
    ----------
    loss_config_list : List[LossConfig]
        List of loss configurations.

    Returns
    -------
    LossCollection
        Collection of initialized loss functions.
    """

    log.info("Initializing Loss Function")
    loss_funcs = []
    for loss in loss_config_list:
        loss_funcs.append(Loss(**loss.model_dump()))
    return LossCollection(loss_funcs)


def compute_property_shapes(config: Config):
    property_configs = [p.model_dump() for p in config.model.property_heads]

    additional_properties = []

    if len(property_configs) == 0:
        return additional_properties

    loss_names = [loss.name for loss in config.loss]
    for pconf in property_configs:
        name = pconf["name"]
        if name not in loss_names:
            continue
        shape = []
        if pconf["aggregation"] == "none":
            shape.append("natoms")

        feature_shapes = {"l0": [1], "l1": [3], "symmetric_traceless_l2": [3, 3]}

        shape.extend(feature_shapes[pconf["mode"]])

        additional_properties.append((name, shape))

    return additional_properties


def initialize_datasets(config: Config):
    """
    Initialize training and validation datasets based on the provided configuration.

    Parameters
    ----------
    config : Config
        Configuration object all parameters.

    Returns
    -------
    train_ds : Dataset
        Training dataset.
    val_ds : Dataset
        Validation dataset.
    ds_stats : Dict[str, Tuple[float, float]]
        Dictionary containing scale and shift parameters for normalization.
    """

    train_raw_ds, val_raw_ds = load_data_files(config.data)

    Dataset = dataset_dict[config.data.dataset.processing]

    dataset_kwargs = dict(config.data.dataset)
    processing = dataset_kwargs.pop("processing")

    if processing == "cached":
        dataset_kwargs["cache_path"] = config.data.model_version_path

    additional_properties = compute_property_shapes(config)

    train_ds = Dataset(
        train_raw_ds,
        config.model.basis.r_max,
        config.data.batch_size,
        config.n_epochs,
        config.n_jitted_steps,
        pos_unit=config.data.pos_unit,
        energy_unit=config.data.energy_unit,
        pre_shuffle=True,
        additional_properties=additional_properties,
        **dataset_kwargs,
    )
    val_ds = Dataset(
        val_raw_ds,
        config.model.basis.r_max,
        config.data.valid_batch_size,
        config.n_epochs,
        pos_unit=config.data.pos_unit,
        energy_unit=config.data.energy_unit,
        additional_properties=additional_properties,
        **dataset_kwargs,
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


def run(user_config: Union[str, os.PathLike, dict], log_level="error"):
    """
    Starts the training of a model with parameters provided by a the config.

    Parameters
    ----------
    user_config : str | os.PathLike | dict
        training config full example can be find :ref:`here <train_config>`:

    """
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

    sample_input, init_box = train_ds.init_input()
    Builder = config.model.get_builder()
    builder = Builder(config.model.model_dump())
    model = builder.build_energy_derivative_model(
        scale=ds_stats.elemental_scale,
        shift=ds_stats.elemental_shift,
        apply_mask=True,
        init_box=init_box,
    )
    batched_model = jax.vmap(model.apply, in_axes=(None, 0, 0, 0, 0, 0))

    if config.model.ensemble and config.model.ensemble.kind == "full":
        n_full_models = config.model.ensemble.n_members
    else:
        n_full_models = 1
    params, rng_key = create_params(model, rng_key, sample_input, n_full_models)

    # TODO rework optimizer initialization and lr keywords
    steps_per_epoch = train_ds.steps_per_epoch()
    tx = get_opt(
        params,
        config.n_epochs,
        steps_per_epoch,
        **config.optimizer.model_dump(),
    )

    state = create_train_state(batched_model, params, tx)

    base_checkpoint = config.checkpoints.base_model_checkpoint
    do_transfer_learning = base_checkpoint is not None
    if do_transfer_learning:
        state = transfer_parameters(state, config.checkpoints)

    if config.weight_average:
        ema_handler = EMAParameters(
            config.weight_average.ema_start, config.weight_average.alpha
        )
    else:
        ema_handler = None

    fit(
        state,
        train_ds,
        loss_fn,
        Metrics,
        callbacks,
        config.n_epochs,
        ckpt_dir=config.data.model_version_path,
        ckpt_interval=config.checkpoints.ckpt_interval,
        val_ds=val_ds,
        patience=config.patience,
        disable_pbar=config.progress_bar.disable_epoch_pbar,
        disable_batch_pbar=config.progress_bar.disable_batch_pbar,
        is_ensemble=n_full_models > 1,
        data_parallel=config.data_parallel,
        ema_handler=ema_handler,
    )
    log.info("Finished training")
