import logging

import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util
from flax.core.frozen_dict import freeze
from optax._src import base

from apax.optimizer.optimizers import ademamix, sam

log = logging.getLogger(__name__)


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
        arg = np.pi * step_in_period / (period * steps_per_epoch)
        lr = init_value / 2 * (jnp.cos(arg) + 1)
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


class OptimizerFactory:
    def __init__(
        self, opt, n_epochs, steps_per_epoch, gradient_clipping, kwargs, schedule
    ) -> None:
        self.opt = opt
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.gradient_clipping = gradient_clipping
        self.kwargs = kwargs
        self.schedule = schedule

    def create(self, lr):
        if lr <= 1e-7:
            optimizer = optax.set_to_zero()
        else:
            schedule = get_schedule(
                lr, self.n_epochs, self.steps_per_epoch, self.schedule
            )
            optimizer = optax.chain(
                optax.clip(self.gradient_clipping),
                self.opt(schedule, **self.kwargs),
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
    rep_scale_lr: float = 0.001,
    rep_prefactor_lr: float = 0.0001,
    gradient_clipping=1000.0,
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
    elif name == "ademamix":
        opt = ademamix
    else:
        opt = getattr(optax, name)

    opt_fac = OptimizerFactory(
        opt, n_epochs, steps_per_epoch, gradient_clipping, kwargs, schedule
    )

    nn_opt = opt_fac.create(nn_lr)
    emb_opt = opt_fac.create(emb_lr)
    scale_opt = opt_fac.create(scale_lr)
    shift_opt = opt_fac.create(shift_lr)
    zbl_opt = opt_fac.create(zbl_lr)
    rep_scale_opt = opt_fac.create(rep_scale_lr)
    rep_prefactor_opt = opt_fac.create(rep_prefactor_lr)

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
        "rep_scale": rep_scale_opt,
        "rep_prefactor": rep_prefactor_opt,
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
