from typing import Any, Callable, Tuple


def make_energy_only_model(energy_properties_model: Callable[..., Tuple[Any, ...]]) -> Callable[..., Any]:
    energy_model = lambda *args, **kwargs: energy_properties_model(*args, **kwargs)[0]
    return energy_model
