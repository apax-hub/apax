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
