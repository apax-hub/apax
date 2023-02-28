import logging
from typing import Any, Callable

import optax
from flax import traverse_util
from flax.core.frozen_dict import freeze

log = logging.getLogger(__name__)


def map_nested_fn(fn: Callable[[str, Any], dict]) -> Callable[[dict], dict]:
    """
    Recursively apply `fn` to the key-value pairs of a nested dict
    See
    https://optax.readthedocs.io/en/latest/api.html?highlight=multitransform#multi-transform
    """

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def get_schedule(
    lr: float, transition_begin: int, transition_steps: int
) -> optax._src.base.Schedule:
    """
    builds a linear learning rate schedule.
    """
    lr_schedule = optax.linear_schedule(
        init_value=lr,
        end_value=1e-6,
        transition_begin=transition_begin,
        transition_steps=transition_steps,
    )
    return lr_schedule


def make_optimizer(opt, lr, transition_begin, transition_steps, opt_kwargs):
    if lr <= 1e-7:
        optimizer = optax.set_to_zero()
    else:
        schedule = get_schedule(lr, transition_begin, transition_steps)
        optimizer = opt(schedule, **opt_kwargs)
    return optimizer


def get_opt(
    params,
    transition_begin: int,
    transition_steps: int,
    emb_lr: float = 0.02,
    nn_lr: float = 0.03,
    scale_lr: float = 0.001,
    shift_lr: float = 0.05,
    opt_name: str = "adam",
    opt_kwargs: dict = {},
    use_flax: bool = False,
) -> optax._src.base.GradientTransformation:
    """
    Builds an optimizer with different learning rates for each parameter group.
    Several `optax` optimizers are supported.
    """
    log.info("Initializing Optimizer")
    opt = getattr(optax, opt_name)

    if use_flax:
        nn_opt = make_optimizer(
            opt, nn_lr, transition_begin, transition_steps, opt_kwargs
        )
        emb_opt = make_optimizer(
            opt, emb_lr, transition_begin, transition_steps, opt_kwargs
        )
        scale_opt = make_optimizer(
            opt, scale_lr, transition_begin, transition_steps, opt_kwargs
        )
        shift_opt = make_optimizer(
            opt, shift_lr, transition_begin, transition_steps, opt_kwargs
        )

        partition_optimizers = {
            "w": nn_opt,
            "b": nn_opt,
            "atomic_type_embedding": emb_opt,
            "scale_per_element": scale_opt,
            "shift_per_element": shift_opt,
        }
        param_partitions = freeze(
            traverse_util.path_aware_map(lambda path, v: path[-1], params)
        )
        tx = optax.multi_transform(partition_optimizers, param_partitions)

    else:
        emb_schedule = get_schedule(emb_lr, transition_begin, transition_steps)
        nn_schedule = get_schedule(nn_lr, transition_begin, transition_steps)
        scale_schedule = get_schedule(scale_lr, transition_begin, transition_steps)
        shift_schedule = get_schedule(shift_lr, transition_begin, transition_steps)
        label_fn = map_nested_fn(lambda k, _: k)
        tx = optax.multi_transform(
            {
                "w": opt(nn_schedule, **opt_kwargs),
                "b": opt(nn_schedule, **opt_kwargs),
                "atomic_type_embedding": opt(emb_schedule, **opt_kwargs),
                "scale_per_element": opt(scale_schedule, **opt_kwargs),
                "shift_per_element": opt(shift_schedule, **opt_kwargs),
            },
            label_fn,
        )
    return tx
