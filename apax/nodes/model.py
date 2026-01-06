import json
import logging
import pathlib
import typing as t
from copy import deepcopy

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import znh5md
import zntrack
from ase import Atoms

import apax
from apax.calibration import compute_calibration_factors
from apax.config.common import flatten, parse_config, unflatten
from apax.config.optuna_config import (
    OptunaConfig,
    get_pruner_from_config,
    get_sampler_from_config,
)
from apax.md import ASECalculator
from apax.md.function_transformations import (
    GaussianAcceleratedMolecularDynamics,
    GlobalCalibration,
    UncertaintyDrivenDynamics,
    available_transformations,
)
from apax.nodes.utils import check_duplicate_keys
from apax.train.run import run as apax_run
from apax.utils.helpers import (
    load_csv_metrics,
    update_nested_dictionary,
)

if t.TYPE_CHECKING:
    import optuna

try:
    import optuna
except ImportError:
    optuna = None

log = logging.getLogger(__name__)


class ApaxBase(zntrack.Node):
    def get_calculator(self, **kwargs):
        raise NotImplementedError


class Apax(ApaxBase):
    """Class for traing Apax models

    Parameters
    ----------
    config: str
        path to the apax config file
    data: list[ase.Atoms]
        the training data set
    validation_data: list[ase.Atoms]
        atoms object with the validation data set
    model: t.Optional[Apax]
        model to be used as a base model for transfer learning
    log_level: str
        verbosity of logging during training
    """

    data: list[ase.Atoms] = zntrack.deps()
    config: str = zntrack.params_path()
    validation_data: list[ase.Atoms] = zntrack.deps()
    model: t.Optional[ApaxBase] = zntrack.deps(None)
    nl_skin: float = zntrack.params(0.5)
    log_level: str = zntrack.params("info")

    model_directory: pathlib.Path = zntrack.outs_path(zntrack.nwd / "apax_model")

    train_data_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "train_atoms.h5")
    validation_data_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "val_atoms.h5")

    metrics: dict = zntrack.metrics()

    @property
    def parameter(self) -> dict:
        parameter = yaml.safe_load(self.state.fs.read_text(self.config))

        custom_parameters = {
            "directory": self.model_directory.as_posix(),
            "experiment": "",
            "train_data_path": self.train_data_file.as_posix(),
            "val_data_path": self.validation_data_file.as_posix(),
        }

        if self.model is not None:
            param_files = self.model.parameter["data"]["directory"]
            base_path = {"base_model_checkpoint": param_files}
            try:
                parameter["transfer_learning"].update(base_path)
            except KeyError:
                parameter["transfer_learning"] = base_path

        check_duplicate_keys(custom_parameters, parameter["data"], log)
        parameter["data"].update(custom_parameters)
        return parameter

    def train_model(self):
        """Train the model using `apax.train.run`"""
        apax_run(self.parameter, log_level=self.log_level)

    def get_metrics(self):
        """In addition to the plots write a model metric"""
        metrics_df = pd.read_csv(self.model_directory / "log.csv")
        best_epoch = np.argmin(metrics_df["val_loss"])
        self.metrics = metrics_df.iloc[best_epoch].to_dict()

    def run(self):
        """Primary method to run which executes all steps of the model training"""

        if not self.state.restarted:
            train_db = znh5md.IO(self.train_data_file.as_posix())
            train_db.extend(self.data)

            val_db = znh5md.IO(self.validation_data_file.as_posix())
            val_db.extend(self.validation_data)

        csv_path = self.model_directory / "log.csv"
        if self.state.restarted and csv_path.is_file():
            metrics_df = pd.read_csv(self.model_directory / "log.csv")

            if metrics_df["epoch"].iloc[-1] >= self.parameter["n_epochs"] - 1:
                return

        self.train_model()
        self.get_metrics()

    def get_calculator(self, **kwargs):
        """Get an apax ase calculator"""
        with self.state.use_tmp_path():
            calc = ASECalculator(
                model_dir=self.model_directory,
                dr_threshold=self.nl_skin,
            )
            return calc


class ApaxApplyTransformation(ApaxBase):
    """Apply transformation to an Apax model."""

    model: ApaxBase = zntrack.deps()
    transformations: list[
        UncertaintyDrivenDynamics | GaussianAcceleratedMolecularDynamics
    ] = zntrack.deps(default_factory=list)

    def run(self):
        pass

    @property
    def model_directory(self):
        return self.model.model_directory

    def get_calculator(self, **kwargs):
        with self.model.state.use_tmp_path():
            calc = ASECalculator(
                model_dir=self.model_directory,
                dr_threshold=self.model.nl_skin,
                transformations=self.transformations,
            )
            return calc


class ApaxEnsemble(ApaxBase):
    """Parallel apax model ensemble in ASE.

    Parameters
    ----------
    models: list
        List of `ApaxModel` nodes to ensemble.
    nl_skin: float
        Neighborlist skin.
    """

    models: list[Apax] = zntrack.deps()
    nl_skin: float = zntrack.params(0.5)

    def run(self) -> None:
        pass

    def get_calculator(self, **kwargs) -> ase.calculators.calculator.Calculator:
        """Property to return a model specific ase calculator object.

        Returns
        -------
        calc:
            ase calculator object
        """
        param_files = [m.parameter["data"]["directory"] for m in self.models]

        with self.state.use_tmp_path():
            calc = ASECalculator(
                param_files,
                dr_threshold=self.nl_skin,
            )
            return calc


class ApaxImport(zntrack.Node):
    """Parallel apax model ensemble in ASE.

    Parameters
    ----------
    models: list
        List of `ApaxModel` nodes to ensemble.
    nl_skin: float
        Neighborlist skin.
    """

    config: str = zntrack.params_path()
    nl_skin: float = zntrack.params(0.5)

    @property
    def parameter(self) -> dict:
        return yaml.safe_load(self.state.fs.read_text(self.config))

    def get_calculator(self, **kwargs) -> ase.calculators.calculator.Calculator:
        """Property to return a model specific ase calculator object.

        Returns
        -------
        calc:
            ase calculator object
        """

        directory = self.parameter["data"]["directory"]
        exp = self.parameter["data"]["experiment"]
        model_dir = directory + "/" + exp

        with self.state.use_tmp_path():
            calc = ASECalculator(
                model_dir,
                dr=self.nl_skin,
            )
            return calc

    def run(self) -> None:
        pass


class ApaxCalibrate(ApaxBase):
    """Globally calibrate the energy and force uncertainties of an Apax model.

    Parameters
    ----------
    model: Apax
        Model to calibrate
    validation_data: list[ase.Atoms]
        Calibration atoms
    batch_size: int, default = 32
        Processing batch size. Choose the largest allowed by your VRAM.
    criterion: str, default = "ma_cal
        Calibration criterion. See uncertainty_toolbox for more details.
    shared_factor: bool, default = False
        Whether or not to calibrate energies and forces separately.
    optimizer_bounds: Tuple[float, float], default = (1e-2, 1e2)
        Search value bounds.
    transformations: dict
        Key-parameter dict with function transformations applied
        to the model function within the ASE calculator.
        See the apax documentation for available methods.
    """

    model: ApaxBase = zntrack.deps()
    validation_data: list[Atoms] = zntrack.deps()
    batch_size: int = zntrack.params(32)
    criterion: str = zntrack.params("ma_cal")
    shared_factor: bool = zntrack.params(False)
    optimizer_bounds: t.Tuple[float, float] = zntrack.params((1e-2, 1e2))

    transformations: t.Optional[list[dict[str, dict]]] = zntrack.params(None)

    nl_skin: float = zntrack.params(0.5)

    metrics: dict = zntrack.metrics()

    def run(self):
        """Primary method to run which executes all steps of the model training"""

        calc = self.model.get_calculator()
        self.e_factor, self.f_factor = compute_calibration_factors(
            calc,
            self.validation_data,
            batch_size=self.batch_size,
            criterion=self.criterion,
            shared_factor=self.shared_factor,
            optimizer_bounds=self.optimizer_bounds,
        )

        self.metrics = {
            "e_factor": self.e_factor,
            "f_factor": self.f_factor,
        }

    @property
    def model_directory(self):
        return self.model.model_directory

    @property
    def parameter(self) -> dict:
        return self.model.parameter

    def get_calculator(self, **kwargs):
        """Property to return a model specific ase calculator object.

        Returns
        -------
        calc:
            ase calculator object
        """

        e_factor = self.metrics["e_factor"]
        f_factor = self.metrics["f_factor"]

        config_file = self.model.parameter["data"]["directory"]

        calibration = GlobalCalibration(
            energy_factor=e_factor,
            forces_factor=f_factor,
        )
        transformations = [calibration]
        if self.transformations:
            for transform, params in self.transformations.items():
                transformations.append(available_transformations[transform](**params))
        with self.model.state.use_tmp_path():
            calc = ASECalculator(
                model_dir=self.model_directory,
                dr_threshold=self.model.nl_skin,
                transformations=transformations,
            )
            return calc


class ApaxOptimizeHyperparameters(ApaxBase):
    """Optimize the hyperparameters of the Apax model.

    config: str
        path to the apax config file
    data: list[ase.Atoms]
        the training data set
    validation_data: list[ase.Atoms]
        atoms object with the validation data set
    log_level: str
        verbosity of logging during training
    """

    data: list[ase.Atoms] = zntrack.deps()
    validation_data: list[ase.Atoms] = zntrack.deps()
    config: str = zntrack.params_path()
    optuna_config_path: str = zntrack.params_path()

    train_log_level: str = zntrack.params("info")
    optuna_log_level: str = zntrack.params("info")
    nl_skin: float = zntrack.params(0.5)

    train_data_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "train_atoms.h5")
    validation_data_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "val_atoms.h5")

    img_optimization: pathlib.Path = zntrack.outs_path(zntrack.nwd / "optimization.png")
    img_importance: pathlib.Path = zntrack.outs_path(zntrack.nwd / "importances.png")
    img_slices: pathlib.Path = zntrack.outs_path(zntrack.nwd / "slices.png")
    img_trials: pathlib.Path = zntrack.outs_path(zntrack.nwd / "trials.png")

    metrics: dict = zntrack.metrics()

    def __post_init__(self):
        if optuna is None:
            raise ImportError(f"{self.__name__} requires optuna")
        optuna.logging.enable_propagation()
        optuna.logging.set_verbosity(self.optuna_log_level.upper())

    def check_search_space(self) -> None:
        full_config = yaml.safe_load(
            self.state.fs.read_text(
                f"{pathlib.Path(apax.__file__).parent}/cli/templates/train_config_full.yaml"
            )
        )
        flat_full_config = flatten(full_config)
        for key in self.optuna_config.search_space:
            if key in ["model_n_layers", "model_n_nodes"]:
                continue
            if key not in flat_full_config:
                raise ValueError(f"key {key} in search config was not in full config")

        # TODO: Make number of nodes possible vary per layer.

    def _get_trial_configuration(self, trial: optuna.trial.Trial) -> dict[str, t.Any]:
        params = trial.params.copy()

        # Since "model_nn" is a list of number of nodes for each layer, we
        # create this list by taking the suggested number of layers, and the
        # suggested number of nodes.
        if "model_n_layers" in params:
            n_layers = params["model_n_layers"]
            del params["model_n_layers"]
        else:
            n_layers = len(self.default_config["model"]["nn"])
        if "model_n_nodes" in params:
            n_nodes = params["model_n_nodes"]
            del params["model_n_nodes"]
        else:
            n_nodes = self.default_config["model"]["nn"][0]
        params["model_nn"] = [n_nodes] * n_layers

        # This needs to be updated for each level in the dictionary, with new keys
        # if necessary.
        nested_keys = {
            "optimizer": (None,),
            "model": ({"basis": (None,)}, None),
        }
        dct = unflatten(params, nested_keys)
        dct["data"] = {"directory": self.get_trial_directory(trial.number).as_posix()}
        return dct

    def _update_configuration(self, trial: optuna.trial.Trial) -> dict[str, t.Any]:
        trial_config = self._get_trial_configuration(trial)
        log.debug(f"Trial {trial.number} configuration: {trial_config}")

        # Need to deepcopy, otherwise pruning parameters will be appended to trials
        # and trial 1 will have callback of trial 0 and callback of trial 1.
        parameters = deepcopy(self.default_config)

        check_duplicate_keys(flatten(trial_config), flatten(parameters), log)
        parameters = update_nested_dictionary(parameters, trial_config)
        if self.optuna_config.pruner_config is None:
            return parameters

        pruning_config = {
            "name": "pruning",
            "study_name": self.optuna_config.study_name,
            "trial_id": trial._trial_id,
            "study_log_file": (self.nwd / self.optuna_config.study_log_file).as_posix(),
            "interval": self.optuna_config.pruner_config.interval,
        }
        if "callbacks" in parameters:
            parameters["callbacks"].append(pruning_config)
        else:
            parameters["callback"] = [pruning_config]
        return parameters

    def run_trial(
        self, trial: optuna.trial.Trial
    ) -> tuple[float, optuna.trial.TrialState]:
        config = self._update_configuration(trial)

        try:
            apax_run(config, log_level=self.train_log_level)
            state = optuna.trial.TrialState.COMPLETE
        except optuna.TrialPruned:
            state = optuna.trial.TrialState.PRUNED

        # Get the metrics
        metrics = load_csv_metrics(self.get_trial_directory(trial.number) / "log.csv")
        return np.min(metrics[self.optuna_config.monitor]), state

    def _get_n_trials_run_previous_study(self) -> int:
        if not any(
            study.study_name == self.optuna_config.study_name
            for study in self.storage.get_all_studies()
        ):
            return 0

        study_id = self.storage.get_study_id_from_name(self.optuna_config.study_name)
        log.debug(
            f"Found study with name {self.optuna_config.study_name} and id {study_id} in storage"
        )

        first_trial_distributions = self.storage.get_all_trials(study_id)[0].distributions
        for param, distribution in self.distributions.items():
            if (
                param not in first_trial_distributions
                or first_trial_distributions[param] != distribution
            ):
                log.warning(
                    f"Found distribution {param} with old distribution that is incompatible with new distribution. Deleting study"
                )
                self.storage.delete_study(study_id)
                return 0

        ntrials_done = self.storage.get_n_trials(
            study_id,
            state=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
        )
        log.debug(f"{ntrials_done} done in previous study")
        return ntrials_done

    def _setup(self):
        self.optuna_config: OptunaConfig = parse_config(
            self.optuna_config_path, mode="optuna"
        )
        self.check_search_space()

        self.distributions = {}
        for param, dct in self.optuna_config.search_space.items():
            self.distributions[param] = optuna.distributions.json_to_distribution(
                json.dumps(dct)
            )

        self.storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                (self.nwd / self.optuna_config.study_log_file).as_posix()
            )
        )

        self.default_config = yaml.safe_load(self.state.fs.read_text(self.config))
        custom_parameters = {
            "experiment": "",
            "train_data_path": self.train_data_file.as_posix(),
            "val_data_path": self.validation_data_file.as_posix(),
        }
        self.default_config["data"].update(custom_parameters)

        if not self.state.restarted:
            train_db = znh5md.IO(self.default_config["data"]["train_data_path"])
            train_db.extend(self.data)

            val_db = znh5md.IO(self.default_config["data"]["val_data_path"])
            val_db.extend(self.validation_data)

    def run(self):
        self._setup()
        ntrials_done = self._get_n_trials_run_previous_study()

        study = optuna.create_study(
            study_name=self.optuna_config.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction="minimize",
            sampler=get_sampler_from_config(self.optuna_config),
            pruner=get_pruner_from_config(self.optuna_config),
        )
        # Write pruner configuration so it is visible to the trial.
        study.set_user_attr("pruner", self.optuna_config.pruner_config.dict())

        log.info("Starting hyperparameter optimization")
        for _ in range(ntrials_done, self.optuna_config.n_trials):
            trial = study.ask(fixed_distributions=self.distributions)
            # Somehow add pruning into this.
            value, trial_state = self.run_trial(trial)
            if trial_state == optuna.trial.TrialState.COMPLETE:
                study.tell(trial, values=value, state=trial_state)
            elif trial_state == optuna.trial.TrialState.PRUNED:
                study.tell(trial, state=trial_state)
            else:
                raise NotImplementedError()
            tf.keras.backend.clear_session()
            log.info(
                f"Trial {trial.number} finished with value {value} and state {trial_state}"
            )

            log.debug(
                f"Trial {trial.number} finished with value {value}. Best trial: {study.best_trial.number} with value {study.best_value}"
            )
        log.info("Hyperparameter optimization finished")
        log.debug(
            f"Best trial: {study.best_trial.number}, with configuration {study.best_trial.params}"
        )

        self._get_plots(study)
        self.get_metrics()
        optuna.logging.disable_propagation()

    def _get_plots(self, study: optuna.study.Study):
        plt.figure()
        ax = optuna.visualization.matplotlib.plot_optimization_history(
            study, target_name="Loss"
        )
        plt.savefig(self.img_optimization, bbox_inches="tight")
        plt.close()

        plt.figure()
        ax = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(self.img_importance, bbox_inches="tight")
        plt.close()

        plt.figure()
        ax = optuna.visualization.matplotlib.plot_slice(study)
        plt.savefig(self.img_slices, bbox_inches="tight")
        plt.close()

        plt.figure()
        for trial_number in range(self.optuna_config.n_trials):
            metrics = load_csv_metrics(self.get_trial_directory(trial_number) / "log.csv")
            plt.plot(metrics["epoch"], metrics[self.optuna_config.monitor])
        plt.xlabel("Epoch")
        plt.ylabel(self.optuna_config.monitor)
        plt.savefig(self.img_trials, bbox_inches="tight")
        plt.close()

    def get_metrics(self):
        """In addition to the plots write a model metric"""
        best_trial_number = self.get_best_trial_number()
        metrics_df = pd.read_csv(self.get_trial_directory(best_trial_number) / "log.csv")
        best_epoch = np.argmin(metrics_df["val_loss"])
        self.metrics = metrics_df.iloc[best_epoch].to_dict()

    def get_calculator(self, **kwargs) -> ASECalculator:
        best_trial_number = self.get_best_trial_number()
        log.debug(f"Loading best trial with trial number {best_trial_number}")
        with self.state.use_tmp_path():
            calc = ASECalculator(
                model_dir=self.get_trial_directory(best_trial_number),
                dr_threshold=self.nl_skin,
            )
            return calc

    def get_best_trial_number(self) -> int:
        study_id = self.storage.get_study_id_from_name(self.optuna_config.study_name)
        best_trial_number = self.storage.get_best_trial(study_id).number
        return best_trial_number

    def get_trial_directory(self, trial_number: int) -> pathlib.Path:
        return self.nwd / f"trial_{trial_number}"
