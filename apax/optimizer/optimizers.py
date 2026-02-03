from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from jax import tree_util
from optax import bias_correction, contrib, update_moment, update_moment_per_elem_norm
from optax._src import base, combine, numerics, transform
from optax.tree_utils import tree_zeros_like


class ScaleByAdemamixState(NamedTuple):
    count: jax.Array
    count_m2: jax.Array
    m1: base.Updates
    m2: base.Updates
    nu: base.Updates


def ademamix(
    lr: float,
    b1: float = 0.9,
    b2: float = 0.999,
    b3: float = 0.9999,
    alpha: float = 5.0,
    b3_scheduler: Optional[Callable] = None,
    alpha_scheduler: Optional[Callable] = None,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> base.GradientTransformation:
    """AdEMAmix implementation directly taken from the original implementation:
    2409.03137
    """
    return combine.chain(
        scale_by_ademamix(b1, b2, b3, alpha, b3_scheduler, alpha_scheduler, eps),
        transform.add_decayed_weights(weight_decay),
        transform.scale_by_learning_rate(lr),
    )


def scale_by_ademamix(
    b1: float,
    b2: float,
    b3: float,
    alpha: float,
    b3_scheduler: Optional[Callable],
    alpha_scheduler: Optional[Callable],
    eps: float,
) -> base.GradientTransformation:
    def init_fn(params: base.Params) -> ScaleByAdemamixState:
        m1 = tree_zeros_like(params)  # fast EMA
        m2 = tree_zeros_like(params)  # slow EMA
        nu = tree_zeros_like(params)  # second moment estimate
        return ScaleByAdemamixState(
            count=jnp.zeros([], jnp.int32),
            count_m2=jnp.zeros([], jnp.int32),
            m1=m1,
            m2=m2,
            nu=nu,
        )

    def update_fn(
        updates: base.Updates, state: ScaleByAdemamixState, params: Optional[base.Params] = None
    ) -> tuple[base.Updates, ScaleByAdemamixState]:
        del params
        c_b3 = b3_scheduler(state.count_m2) if b3_scheduler is not None else b3
        c_alpha = (
            alpha_scheduler(state.count_m2) if alpha_scheduler is not None else alpha
        )
        m1 = update_moment(updates, state.m1, b1, 1)  # m1 = b1 * m1 + (1-b1) * updates
        m2 = update_moment(updates, state.m2, c_b3, 1)
        nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)
        count_m2_inc = numerics.safe_int32_increment(state.count_m2)
        m1_hat = bias_correction(m1, b1, count_inc)
        nu_hat = bias_correction(nu, b2, count_inc)
        updates = tree_util.tree_map(
            lambda m1_, m2_, v_: (m1_ + c_alpha * m2_) / (jnp.sqrt(v_) + eps),
            m1_hat,
            m2,
            nu_hat,
        )
        return updates, ScaleByAdemamixState(
            count=count_inc, count_m2=count_m2_inc, m1=m1, m2=m2, nu=nu
        )

    return base.GradientTransformation(init_fn, update_fn)


def sam(
    lr: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    rho: float = 0.001,
    sync_period: int = 2,
) -> base.GradientTransformation:
    """A SAM optimizer using Adam for the outer optimizer."""
    opt = optax.adam(lr, b1=b1, b2=b2)
    adv_opt = optax.chain(contrib.normalize(), optax.sgd(rho))
    return contrib.sam(opt, adv_opt, sync_period=sync_period)
