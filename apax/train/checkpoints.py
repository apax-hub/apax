import logging
from pathlib import Path
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.training import checkpoints, train_state
from flax.traverse_util import flatten_dict, unflatten_dict

from apax.config.common import parse_config
from apax.config.train_config import Config

log = logging.getLogger(__name__)


def check_for_ensemble(params: FrozenDict) -> int:
    """Checks if a set of parameters belongs to an ensemble model.
    This is the case if all parameters share the same first dimension (parameter batch)
    """
    flat_params = flatten_dict(params)
    shapes = [v.shape[0] for v in flat_params.values()]
    is_ensemble = len(set(shapes)) == 1

    if is_ensemble:
        return shapes[0]
    else:
        return 1


def create_train_state(model, params: FrozenDict, tx):
    n_models = check_for_ensemble(params)

    def create_single_train_state(params):
        state = train_state.TrainState.create(
            apply_fn=model,
            params=params,
            tx=tx,
        )
        return state

    if n_models > 1:
        train_state_fn = jax.vmap(
            create_single_train_state,
            axis_name="ensemble"
        )
    else:
        train_state_fn = create_single_train_state

    return train_state_fn(params)


def create_params(model, rng_key, sample_input: tuple, n_models: int):
    keys = jax.random.split(rng_key, num=n_models + 1)
    rng_key, model_rng = keys[0], keys[1:]

    log.info(f"initializing {n_models} models")

    if n_models == 1:
        params = model.init(model_rng[0], *sample_input)
    elif n_models > 1:
        num_args = len(sample_input)
        # vmap only over parameters, not over any data from the input
        in_axes = (0, *[None] * num_args)
        params = jax.vmap(model.init, in_axes=in_axes)(model_rng, *sample_input)
    else:
        raise ValueError(f"n_models should be a positive integer, found {n_models}")

    params = freeze(params)

    return params, rng_key


def load_state(state, ckpt_dir):
    start_epoch = 0
    target = {"model": state, "epoch": 0}
    checkpoints_exist = Path(ckpt_dir).is_dir()
    if checkpoints_exist:
        log.info("Loading checkpoint")
        raw_restored = checkpoints.restore_checkpoint(ckpt_dir, target=target, step=None)
        state = raw_restored["model"]
        start_epoch = raw_restored["epoch"] + 1
        log.info("Successfully restored checkpoint from epoch %d", raw_restored["epoch"])

    return state, start_epoch


class CheckpointManager:
    def __init__(self) -> None:
        self.async_manager = checkpoints.AsyncManager()

    def save_checkpoint(self, ckpt, epoch: int, path: str) -> None:
        checkpoints.save_checkpoint(
            ckpt_dir=path,
            target=ckpt,
            step=epoch,
            overwrite=True,
            keep=2,
            async_manager=self.async_manager,
        )


def stack_parameters(param_list: List[FrozenDict]) -> FrozenDict:
    """Combine a list of parameter sets into a stacked version.
    Used for model ensembles.
    """
    flat_param_list = []
    for params in param_list:
        params = unfreeze(params)
        flat_params = flatten_dict(params)
        flat_param_list.append(flat_params)

    stacked_flat_params = flat_params
    for p in flat_param_list[0].keys():
        stacked_flat_params[p] = jnp.stack(
            [flat_param[p] for flat_param in flat_param_list]
        )

    stacked_params = unflatten_dict(stacked_flat_params)
    stack_params = freeze(stacked_params)
    return stack_params


def load_params(model_version_path: Path, best=True) -> FrozenDict:
    if best:
        model_version_path = model_version_path / "best"
    log.info(f"loading checkpoint from {model_version_path}")
    try:
        raw_restored = checkpoints.restore_checkpoint(
            model_version_path,
            target=None,
            step=None
        )
    except FileNotFoundError:
        print(f"No checkpoint found at {model_version_path}")
    params = jax.tree_map(jnp.asarray, raw_restored["model"]["params"])

    return params


def restore_single_parameters(model_dir: Path) -> Tuple[Config, FrozenDict]:
    """Load the config and parameters of a single model
    """
    model_config = parse_config(Path(model_dir) / "config.yaml")
    ckpt_dir = model_config.data.model_version_path()
    return model_config, load_params(ckpt_dir)


def restore_parameters(model_dir: Union[Path, List[Path]]) -> Tuple[Config, FrozenDict]:
    """Restores one or more model configs and parameters.
    Parameters are stacked for ensembling.
    """
    if isinstance(model_dir, Path) or isinstance(model_dir, str):
        config, params = restore_single_parameters(model_dir)

    elif isinstance(model_dir, list):
        param_list = []
        for path in model_dir:
            config, params = restore_single_parameters(path)
            param_list.append(params)

        params = stack_parameters(param_list)
    else:
        raise NotImplementedError(
            "Please provide either a path or list of paths to trained models"
        )

    return config, params
