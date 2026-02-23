from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Type

from pydantic import BaseModel, PositiveInt

if TYPE_CHECKING:
    import optuna


def get_pruner(name: str) -> Type["optuna.pruners.BasePruner"]:
    """Get the pruner class from the name.

    Args:
        name (str): name of pruner

    Returns:
        Type[optuna.pruners.BasePruner]: uninstantiated pruner
    """

    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "optuna is required for hyperparameter optimisation. "
            "Install it via `pip install optuna`."
        ) from e

    if name not in optuna.pruners.__all__:
        raise ValueError(f"pruner with name {name} not in optuna.pruners")
    return getattr(optuna.pruners, name)


def get_sampler(name: str) -> Type["optuna.samplers.BaseSampler"]:
    """Get the sampler class from the name.

    Args:
        name (str): name of sampler

    Returns:
        Type[optuna.pruners.BaseSampler]: uninstantiated sampler

    Notes:
        Also can include the AutoSampler, which automatically infer the "best"
            sampler based on the study parameters, see
            https://hub.optuna.org/samplers/auto_sampler
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "optuna is required for hyperparameter optimisation. "
            "Install it via `pip install optuna`."
        ) from e

    if name == "AutoSampler":
        try:
            import optunahub

            # Requires optunahub, scipy, cmaes and torch.
            return optunahub.load_module("samplers/auto_sampler").AutoSampler
        except ImportError as e:
            raise ImportError(
                f"pruner {name} requires optunahub. Set pruner to other or install optunahub"
            ) from e

    if name not in optuna.samplers.__all__:
        raise ValueError(f"sampler with name {name} not in optuna.samplers")
    return getattr(optuna.samplers, name)


class OptunaPrunerConfig(BaseModel, extra="forbid"):
    """Config for optuna pruner.

    Attributes:
        name (str): Name of pruner
        interval (PositiveInt): Interval to check whether a trial should be pruned
        kwargs (dict[str, Any]): kwargs passed to pruner
    """

    name: str
    interval: PositiveInt = 1
    kwargs: dict[str, Any] = {}


class OptunaSamplerConfig(BaseModel, extra="forbid"):
    """Config for optuna sampler.

    Attributes:
        name (str): Name of sampler. Default: "AutoSampler"
        kwargs (str): kwargs passed to sampler
    """

    name: str = "AutoSampler"
    kwargs: dict[str, Any] = {}


class OptunaConfig(BaseModel, extra="forbid"):
    """Configuration of optuna study.

    Attributes:
        n_trials (PositiveInt): number of trials
        search_space (dict[str, dict[str, Any]]): dictionary indicating the
            search space to use for the hyperparameter optimization. Keys should
            be keys in the training configuration, and if the keys are nested,
            they should be prefixed with the parent key, and an underscore,
            see the example below.
        seed (int): seed to use for sampling
        monitor (str): metric to monitor. Default: "val_loss"
        study_name (str): name of study. Default: "study"
        study_log_file (str): path to study log file. Default: "study.log"
        sampler_config (OptunaSamplerConfig): sampler configuration
        pruner_config (Optional[OptunaPrunerConfig]): pruner configuration
            Default: None

    Example:
        For a search space with varying radius for the environment between 3
            and 8 Angstrom, the number of tensor contractions between
            5 and 8, and number of epochs between 100 and 200.

        .. code-block:: python

            search_space = {
                'model_basis_r_max': {'type': 'float', 'low': 3, 'high': 8},
                'model_n_contr': {'type': 'int', 'low': 5, 'high': 8},
                'n_epochs': {'type': 'int', 'low': 100, 'high': 200},
            }

    """

    n_trials: PositiveInt
    search_space: dict[str, dict[str, Any]]
    seed: int = 1
    monitor: str = "val_loss"
    study_name: str = "study"
    study_log_file: str | Path = "study.log"
    sampler_config: OptunaSamplerConfig = OptunaSamplerConfig()
    pruner_config: Optional[OptunaPrunerConfig] = None


def get_pruner_from_config(
    optuna_config: OptunaConfig,
) -> Optional["optuna.pruners.BasePruner"]:
    """Get the instantiated pruner from the optuna configuration

    Args:
        optuna_config (OptunaConfig): configuration for study

    Returns:
        Optional[optuna.pruners.BasePruner]: instantiated pruner.
            None if optuna_config.pruner_config is None
    """

    if optuna_config.pruner_config is None:
        return None

    pruner_kwargs = optuna_config.pruner_config.kwargs.copy()

    pruner_class = get_pruner(optuna_config.pruner_config.name)
    return pruner_class(**pruner_kwargs)


def get_sampler_from_config(optuna_config: OptunaConfig) -> "optuna.samplers.BaseSampler":
    """Get the instantiated sampler from the optuna configuration

    Args:
        optuna_config (OptunaConfig): configuration for study

    Returns:
        Optional[optuna.pruners.BaseSampler]: instantiated sampler.
    """

    # See https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results
    sampler_kwargs = optuna_config.sampler_config.kwargs.copy()
    sampler_kwargs["seed"] = optuna_config.seed

    sampler_class = get_sampler(optuna_config.sampler_config.name)
    return sampler_class(**sampler_kwargs)
