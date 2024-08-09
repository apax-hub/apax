import logging

import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util
from flax.core.frozen_dict import freeze
from optax import contrib
from optax._src import base

log = logging.getLogger(__name__)


def sam(lr=1e-3, b1=0.9, b2=0.999, rho=0.001, sync_period=2):
    """A SAM optimizer using Adam for the outer optimizer."""
    opt = optax.adam(lr, b1=b1, b2=b2)
    adv_opt = optax.chain(contrib.normalize(), optax.sgd(rho))
    return contrib.sam(opt, adv_opt, sync_period=sync_period)


def cyclic_cosine_decay_schedule(
    init_value: float,
    steps_per_epoch,
    period: int,
    decay_factor: float = 0.9,
) -> base.Schedule:
    r"""Returns a function which implements cyclic cosine learning rate decay.

    Args:
        init_value: An initial value for the learning rate.

    Returns:
        schedule
        A function that maps step counts to values.
    """

    def schedule(count):
        cycle = count // (period * steps_per_epoch)
        step_in_period = jnp.mod(count, period * steps_per_epoch)
        lr = (
            init_value
            / 2
            * (jnp.cos(np.pi * step_in_period / (period * steps_per_epoch)) + 1)
        )
        lr = lr * (decay_factor**cycle)
        return lr

    return schedule


def get_schedule(
    lr: float,
    n_epochs: int,
    steps_per_epoch: int,
    schedule_kwargs: dict,
) -> optax._src.base.Schedule:
    """
    builds a linear learning rate schedule.
    """
    schedule_kwargs = schedule_kwargs.copy()
    name = schedule_kwargs.pop("name")
    if name == "linear":
        lr_schedule = optax.linear_schedule(
            init_value=lr, transition_steps=n_epochs * steps_per_epoch, **schedule_kwargs
        )
    elif name == "cyclic_cosine":
        lr_schedule = cyclic_cosine_decay_schedule(lr, steps_per_epoch, **schedule_kwargs)
    else:
        raise KeyError(f"unknown learning rate schedule: {name}")
    return lr_schedule


def make_optimizer(opt, lr, n_epochs, steps_per_epoch, kwargs, schedule):
    if lr <= 1e-7:
        optimizer = optax.set_to_zero()
    else:
        schedule = get_schedule(lr, n_epochs, steps_per_epoch, schedule)
        optimizer = optax.chain(
            opt(schedule, **kwargs),
            optax.zero_nans(),
        )
    return optimizer


def get_opt(
    params,
    n_epochs: int,
    steps_per_epoch: int,
    emb_lr: float = 0.02,
    nn_lr: float = 0.03,
    scale_lr: float = 0.001,
    shift_lr: float = 0.05,
    zbl_lr: float = 0.001,
    name: str = "adam",
    kwargs: dict = {},
    schedule: dict = {},
) -> optax._src.base.GradientTransformation:
    """
    Builds an optimizer with different learning rates for each parameter group.
    Several `optax` optimizers are supported.
    """

    log.info("Initializing Optimizer")
    if name == "sam":
        opt = sam
    else:
        opt = getattr(optax, name)

    nn_opt = make_optimizer(opt, nn_lr, n_epochs, steps_per_epoch, kwargs, schedule)
    emb_opt = make_optimizer(opt, emb_lr, n_epochs, steps_per_epoch, kwargs, schedule)
    scale_opt = make_optimizer(opt, scale_lr, n_epochs, steps_per_epoch, kwargs, schedule)
    shift_opt = make_optimizer(opt, shift_lr, n_epochs, steps_per_epoch, kwargs, schedule)
    zbl_opt = make_optimizer(opt, zbl_lr, n_epochs, steps_per_epoch, kwargs, schedule)

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
        "kernel": nn_opt,
        "bias": nn_opt,
        "embedding": emb_opt,
        "weights_K": nn_opt,
        "weights_Q": nn_opt,
        "weights_V": nn_opt,
        "scale": scale_opt,
    }

    param_partitions = freeze(
        traverse_util.path_aware_map(lambda path, v: path[-1], params)
    )
    tx = optax.multi_transform(partition_optimizers, param_partitions)

    return tx
