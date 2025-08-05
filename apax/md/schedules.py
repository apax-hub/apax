import jax
import jax.numpy as jnp
import numpy as np
from ase import units


class TSchedule:
    def __init__(self, T0: int):
        self.T0 = T0

    def __call__(self, step) -> float:
        raise NotImplementedError


class ConstantTSchedule(TSchedule):
    def __call__(self, step) -> float:
        return self.T0 * units.kB


class PieceWiseLinearTSchedule(TSchedule):
    def __init__(self, T0: int, temperatures: list[float], durations: list[int]):
        self.T0 = T0
        self.temperatures = jnp.array(temperatures)
        steps = np.cumsum(durations)
        self.steps = jnp.array(steps)

    def __call__(self, step) -> float:
        T = jnp.interp(
            step, self.steps, self.temperatures, left=self.T0, right=self.temperatures[-1]
        )
        return T * units.kB


class OscillatingRampTSchedule(TSchedule):
    def __init__(
        self,
        T0: int,
        Tend: float,
        amplitude: float,
        num_oscillations: int,
        total_steps: int,
    ):
        self.T0 = T0
        self.Tend = Tend
        self.amplitude = amplitude
        self.num_oscillations = num_oscillations
        self.total_steps = total_steps

    def __call__(self, step) -> float:
        ramp = step / self.total_steps * (self.Tend - self.T0)
        oscillation = self.amplitude * jnp.sin(
            2 * np.pi * step / self.total_steps * self.num_oscillations
        )
        T = self.T0 + ramp + oscillation

        T = jnp.maximum(0, T)  # prevent negative temperature
        T = jnp.where(step < self.total_steps, T, self.Tend)

        return T * units.kB


def additive_impacted(state, box, switched, **switch_kwargs):
    pos = state.position @ box
    pos = pos[-1, -1]

    condition = pos < switch_kwargs["impact_hight"]
    switched = jnp.logical_or(condition, switched)
    return switched


def instant_switching(state, switched, **switch_kwargs):
    return True


CONDITION_FN = {
    "additive_impacted": additive_impacted,
    "instant_switching": instant_switching,
}


def linear_switching(n_steps, steps_after_switching, **switch_kwargs):
    switching_factor = steps_after_switching / n_steps
    return switching_factor


def sigmoid_switching(
    n_steps, steps_after_switching, **switch_kwargs
):  # adjust k as needed: higher = sharper transition
    k = switch_kwargs["k"]
    t = steps_after_switching / n_steps
    switching_factor = 1 / (1 + jnp.exp(-k * (t - 0.5)))
    return switching_factor


SWITCHING_FN = {"linear": linear_switching, "sigmoid": sigmoid_switching}


class SwitchSchedule:
    def __init__(
        self,
        switching_fn,
        n_steps,
        condition,
        switch_kwargs: dict,
    ):
        self.switching_fn = switching_fn
        self.n_steps = n_steps
        self.condition = condition
        self.switch_kwargs = switch_kwargs

    def __call__(self, state, box, sim_step, switched, switching_step) -> float:
        def no_switch(sim_step, switching_step):
            switching_factor = 0.0
            switching_step = sim_step
            return switching_factor, switching_step

        def do_switch(sim_step, switching_step):
            steps_after_switching = sim_step - switching_step
            switching_factor = SWITCHING_FN[self.switching_fn](
                self.n_steps, steps_after_switching, **self.switch_kwargs
            )
            return switching_factor, switching_step

        switched = CONDITION_FN[self.condition](
            state, box, switched, **self.switch_kwargs
        )
        switch_factor, switching_step = jax.lax.cond(
            switched, do_switch, no_switch, sim_step, switching_step
        )

        return switch_factor, switched, switching_step
