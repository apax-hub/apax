import json
import logging
import pathlib
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yaml
import znh5md
import zntrack
from ase import Atoms
from tensorflow.keras.backend import clear_session

import apax
from apax.config.common import flatten, parse_config, unflatten
from apax.config.optuna_config import (
    OptunaConfig,
    get_pruner_from_config,
    get_sampler_from_config,
)
from apax.md.ase_calc import ASECalculator
from apax.nodes.utils import check_duplicate_keys
from apax.train.run import run as apax_run
from apax.utils.helpers import (
    load_csv_metrics,
    update_nested_dictionary,
)

log = logging.getLogger(__name__)


class ApaxOptimizeHyperparameters(zntrack.Node):
    """Optimize the hyperparameters of the Apax model.

    config: str
        path to the apax config file
    data: list[Atoms]
        the training data set
    validation_data: list[Atoms]
        atoms object with the validation data set
    optuna_config_path: str
        path to optuna configuration file

    train_log_level: str, default = 'info'
        log level for training logs
    optuna_log_level: str, default = 'info'
        log level for logs from optuna
    """

    config: str = zntrack.params_path()
    data: list[Atoms] = zntrack.deps()
    validation_data: list[Atoms] = zntrack.deps()
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

    def _get_trial_configuration(self, trial: optuna.trial.Trial) -> dict[str, Any]:
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

    def _update_configuration(self, trial: optuna.trial.Trial) -> dict[str, Any]:
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
            log.info(
                f"Trial {trial.number} finished with value {value} and state {trial_state}"
            )
            clear_session()

            log.info(
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
