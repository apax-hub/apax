import logging

import jax
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

    def create(self, lr, no_weight_decay=False):
        if lr <= 1e-7:
            optimizer = optax.set_to_zero()
        else:
            schedule = get_schedule(
                lr, self.n_epochs, self.steps_per_epoch, self.schedule
            )
            kwargs = self.kwargs
            # weight decay acts on a param even when its loss-gradient is ~0
            # (e.g. ZBL rep_scale), which would silently decay the repulsion
            # wall away. Let selected groups opt out.
            if no_weight_decay and "weight_decay" in kwargs:
                kwargs = {**kwargs, "weight_decay": 0.0}
            optimizer = optax.chain(
                optax.clip(self.gradient_clipping),
                self.opt(schedule, **kwargs),
                optax.zero_nans(),
            )
        return optimizer


def rank_stacked(inner, rank):
    """Run `inner` on (n_basis, rank * n_radial) leaves viewed as (n_basis, rank, n_radial)."""

    def to3(t):
        return jax.tree_util.tree_map(lambda x: x.reshape(x.shape[0], rank, -1), t)

    def init(params):
        return inner.init(to3(params))

    def update(updates, state, params=None):
        shapes = jax.tree_util.tree_map(jnp.shape, updates)
        new, state = inner.update(to3(updates), state, None if params is None else to3(params))
        new = jax.tree_util.tree_map(lambda x, s: x.reshape(s), new, shapes)
        return new, state

    return optax.GradientTransformation(init, update)


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
    freeze_layers: list[str] = [],
    name: str = "adam",
    kwargs: dict = {},
    schedule: dict = {},
) -> optax._src.base.GradientTransformation:
    """
    Builds an optimizer with different learning rates for each parameter group.
    Several `optax` optimizers are supported.
    """

    log.info("Initializing Optimizer")
    muon_lr = None
    if name == "muon":
        # Muon on hidden weight matrices, Adam (at the usual group lrs) on everything else
        kwargs = dict(kwargs)
        muon_lr = kwargs.pop("muon_lr", 0.02)
        muon_groups = kwargs.pop("muon_groups", ["w", "kernel", "pair_core"])
        muon_kwargs = {"weight_decay": kwargs.pop("muon_weight_decay", 0.0)}
        name = "adam"
    if name == "sam":
        opt = sam
    elif name == "ademamix":
        opt = ademamix
    else:
        opt = getattr(optax, name)

    opt_fac = OptimizerFactory(
        opt, n_epochs, steps_per_epoch, gradient_clipping, kwargs, schedule
    )

    frozen_opt = opt_fac.create(0.0)

    nn_opt = opt_fac.create(nn_lr)
    emb_opt = opt_fac.create(emb_lr)
    scale_opt = opt_fac.create(scale_lr)
    shift_opt = opt_fac.create(shift_lr)
    zbl_opt = opt_fac.create(zbl_lr, no_weight_decay=True)
    rep_scale_opt = opt_fac.create(rep_scale_lr, no_weight_decay=True)
    rep_prefactor_opt = opt_fac.create(rep_prefactor_lr, no_weight_decay=True)

    partition_optimizers = {
        "w": nn_opt,
        "b": nn_opt,
        "atomic_type_embedding": emb_opt,
        "pair_emb_centre": emb_opt,
        "pair_emb_nbr": emb_opt,
        "pair_core": emb_opt,
        "scale_per_element": scale_opt,
        "shift_per_element": shift_opt,
        "a_exp": zbl_opt,
        "a_num": zbl_opt,
        "coefficients": zbl_opt,
        "exponents": zbl_opt,
        "rep_scale": rep_scale_opt,
        "rep_prefactor": rep_prefactor_opt,
        "envelope_amp": scale_opt,
        "envelope_amp_frozen": frozen_opt,
        "kernel": nn_opt,
        "bias": nn_opt,
        "embedding": emb_opt,
        "weights_K": nn_opt,
        "weights_Q": nn_opt,
        "weights_V": nn_opt,
        "scale": scale_opt,
        "frozen": frozen_opt,
    }

    if muon_lr is not None:

        def hidden_matrices(p):
            return jax.tree_util.tree_map(
                lambda x: optax.contrib.MuonDimensionNumbers()
                if x.ndim == 2 and min(x.shape) > 1
                else None,
                p,
            )

        muon_fac = OptimizerFactory(
            optax.contrib.muon,
            n_epochs,
            steps_per_epoch,
            gradient_clipping,
            {"muon_weight_dimension_numbers": hidden_matrices, **muon_kwargs},
            schedule,
        )
        muon_opt = muon_fac.create(muon_lr)
        partition_optimizers["w_out"] = nn_opt  # readout output layer stays on Adam
        for k in muon_groups:
            partition_optimizers[k] = muon_opt
        # CP pair_core is orthogonalised per rank slice (n_basis x n_radial): beats flat on 8/8 seeds;
        # rank = width of the CP pair embeddings (n_species, rank)
        ranks = {
            x.shape[-1]
            for path, x in traverse_util.flatten_dict(params).items()
            if path[-1] in ("pair_emb_centre", "pair_emb_nbr")
        }
        if ranks and "pair_core" in muon_groups:
            (muon_core_rank,) = ranks
            per_slice = OptimizerFactory(
                optax.contrib.muon,
                n_epochs,
                steps_per_epoch,
                gradient_clipping,
                {
                    "muon_weight_dimension_numbers": optax.contrib.MuonDimensionNumbers(
                        reduction_axis=0, output_axis=2
                    ),
                    **muon_kwargs,
                },
                schedule,
            ).create(muon_lr)
            partition_optimizers["pair_core"] = rank_stacked(per_slice, muon_core_rank)

    param_groups = list(partition_optimizers.keys())

    def get_param_group(path, x):
        """
        Assigns each parameter to a group based on its path.
        - Freezes layers with matching name
        - Uses last element of path (e.g., 'w') to match group
        """
        path_str = "/".join(path)
        for frozen in freeze_layers:
            if frozen in path_str:
                return "frozen"

        p_name = path[-1]
        if muon_lr is not None and p_name == "w" and x.shape[-1] == 1:
            return "w_out"
        p_name = p_name if p_name in param_groups else "default"
        return p_name

    param_partitions = freeze(traverse_util.path_aware_map(get_param_group, params))
    tx = optax.multi_transform(partition_optimizers, param_partitions)

    return tx
