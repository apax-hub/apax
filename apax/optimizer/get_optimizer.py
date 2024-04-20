import logging
from typing import Any, Callable

import optax
from optax import contrib
from flax import traverse_util
from flax.core.frozen_dict import freeze

log = logging.getLogger(__name__)


def sam(lr=1e-3, b1=0.9, b2=0.999, rho=0.001, sync_period=2):
    """A SAM optimizer using Adam for the outer optimizer."""
    opt = optax.adam(lr, b1=b1, b2=b2)
    adv_opt = optax.chain(contrib.normalize(), optax.sgd(rho))
    return contrib.sam(opt, adv_opt, sync_period=sync_period)


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
    zbl_lr: float = 0.001,
    opt_name: str = "adam",
    opt_kwargs: dict = {},
    **kwargs,
) -> optax._src.base.GradientTransformation:
    """
    Builds an optimizer with different learning rates for each parameter group.
    Several `optax` optimizers are supported.
    """
    log.info("Initializing Optimizer")
    if opt_name == "sam":
        opt = sam
    else:
        print("optname")
        opt = getattr(optax, opt_name)

    nn_opt = make_optimizer(opt, nn_lr, transition_begin, transition_steps, opt_kwargs)
    emb_opt = make_optimizer(opt, emb_lr, transition_begin, transition_steps, opt_kwargs)
    scale_opt = make_optimizer(
        opt, scale_lr, transition_begin, transition_steps, opt_kwargs
    )
    shift_opt = make_optimizer(
        opt, shift_lr, transition_begin, transition_steps, opt_kwargs
    )
    zbl_opt = make_optimizer(opt, zbl_lr, transition_begin, transition_steps, opt_kwargs)

    partition_optimizers = {
        "w": nn_opt,
        "b": nn_opt,
        "atomic_type_embedding": emb_opt,
        "scale_per_element": scale_opt,
        "shift_per_element": shift_opt,
        "a_exp": zbl_opt,
        "a_num": zbl_opt,
        "coefficients": zbl_opt,
        "exponents": zbl_opt,
        "rep_scale": zbl_opt,
    }

    param_partitions = freeze(
        traverse_util.path_aware_map(lambda path, v: path[-1], params)
    )
    tx = optax.multi_transform(partition_optimizers, param_partitions)

    return tx
