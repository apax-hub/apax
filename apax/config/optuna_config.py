from pathlib import Path
from typing import Any, Optional, Type

import optuna
from pydantic import BaseModel


def get_pruner(name: str, **kwargs) -> Type[optuna.pruners.BasePruner]:
    if name not in optuna.pruners.__all__:
        raise ValueError(f"pruner with name {name} not in optuna.pruners")
    return getattr(optuna.pruners, name)


def get_sampler(name: str, **kwargs) -> Type[optuna.samplers.BaseSampler]:
    if name == "AutoSampler":
        try:
            import optunahub

            # Requires optunahub, scipy, cmaes and torch.
            # See https://hub.optuna.org/samplers/auto_sampler/requirements.txt
            return optunahub.load_module("samplers/auto_sampler").AutoSampler
        except ImportError as e:
            raise ImportError(
                f"pruner {name} requires optunahub. Set pruner to other or install optunahub"
            ) from e

    if name not in optuna.samplers.__all__:
        raise ValueError(f"sampler with name {name} not in optuna.samplers")
    return getattr(optuna.samplers, name)


class OptunaPrunerConfig(BaseModel, extra="forbid"):
    name: str
    interval: int = 1
    kwargs: dict[str, Any] = {}


class OptunaSamplerConfig(BaseModel, extra="forbid"):
    name: str = "AutoSampler"
    kwargs: dict[str, Any] = {}


class OptunaConfig(BaseModel, extra="forbid"):
    n_trials: int
    search_space: dict[str, dict[str, Any]]
    seed: int = 1
    monitor: str = "val_loss"
    study_name: str = "study"
    study_log_file: str | Path = "study.log"
    sampler_config: OptunaSamplerConfig = OptunaSamplerConfig()
    pruner_config: Optional[OptunaPrunerConfig] = None


def get_pruner_from_config(
    optuna_config: OptunaConfig,
) -> Optional[optuna.pruners.BasePruner]:
    # See https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results

    if optuna_config.pruner_config is None:
        return None

    pruner_kwargs = optuna_config.pruner_config.kwargs.copy()
    # pruner_kwargs["seed"] = optuna_config.seed

    pruner_class = get_pruner(optuna_config.pruner_config.name)
    return pruner_class(**pruner_kwargs)


def get_sampler_from_config(optuna_config: OptunaConfig) -> optuna.samplers.BaseSampler:
    # See https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results
    sampler_kwargs = optuna_config.sampler_config.kwargs.copy()
    sampler_kwargs["seed"] = optuna_config.seed

    sampler_class = get_sampler(optuna_config.sampler_config.name)
    return sampler_class(**sampler_kwargs)
