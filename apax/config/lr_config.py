from typing import Literal

from pydantic import BaseModel, NonNegativeFloat


class LRSchedule(BaseModel, frozen=True, extra="forbid"):
    name: str


class LinearLR(LRSchedule, frozen=True, extra="forbid"):
    """
    Configuration of the optimizer.
    Learning rates of 0 will freeze the respective parameters.

    Parameters
    ----------
    name : str, default = "adam"
    transition_begin: int = 0
        Number of steps after which to start decreasing
    end_value: NonNegativeFloat = 1e-6
        Final LR at the end of training.
    """

    name: Literal["linear"]
    transition_begin: int = 0
    end_value: NonNegativeFloat = 1e-6


class CyclicCosineLR(LRSchedule, frozen=True, extra="forbid"):
    """
    Configuration of the optimizer.
    Learning rates of 0 will freeze the respective parameters.

    Parameters
    ----------
    period: int = 40
        Length of a cycle in epochs.
    decay_factor: NonNegativeFloat = 0.93
        Factor by which to decrease the LR after each cycle.
        1.0 means no decrease.
    """

    name: Literal["cyclic_cosine"]
    period: int = 40
    decay_factor: NonNegativeFloat = 0.93
