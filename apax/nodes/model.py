import logging
import pathlib
import shutil
import typing as t
from typing import Optional

import ase.io
import numpy as np
import pandas as pd
import yaml
import zntrack.utils

from apax.md import ASECalculator
from apax.md.function_transformations import available_transformations
from apax.train.run import run as apax_run
from jax import config
from matplotlib import pyplot as plt
from zntrack import dvc, zn

from .utils import check_duplicate_keys

log = logging.getLogger(__name__)


class Apax(zntrack.Node):
    """Class for the implementation of the apax model

    Attributes
    ----------
    config: str
        path to the apax config file
    validation_data: ase.Atoms
        atoms object with the validation data set
    model_directory: pathlib.Path
        model directory
    train_data_file: pathlib.Path
        path to the training data
    validation_data_file: pathlib.Path
        path to the valdidation data
    """

    config: str = dvc.params("apax.yaml")
    validation_data = zntrack.deps()
    model: Optional[t.Any] = zntrack.deps(None)

    model_directory: pathlib.Path = dvc.outs(zntrack.nwd / "apax_model")

    train_data_file: pathlib.Path = dvc.outs(zntrack.nwd / "train_atoms.extxyz")
    validation_data_file: pathlib.Path = dvc.outs(zntrack.nwd / "val_atoms.extxyz")

    jax_enable_x64: bool = zn.params(True)

    metrics_epoch = dvc.plots(
        zntrack.nwd / "log.csv",
        # template=STATIC_PATH / "y_log.json",
        # x="epoch",
        # x_label="epochs",
        # y="val_loss",
        # y_label="validation loss",
    )
    metrics = zn.metrics()

    _parameter: dict = None

    # def _post_init_(self):
    #     self.data = utils.helpers.get_deps_if_node(self.data, "atoms")
    #     self.validation_data = utils.helpers.get_deps_if_node(
    #         self.validation_data, "atoms"
    #     )
    #     self._handle_parameter_file()

    def _post_load_(self) -> None:
        self._handle_parameter_file()

    def _handle_parameter_file(self):
        with self.state.use_tmp_path():
            self._parameter = yaml.safe_load(pathlib.Path(self.config).read_text())

            custom_parameters = {
                "directory": self.model_directory.resolve().as_posix(),
                "experiment": "",
                "train_data_path": self.train_data_file.as_posix(),
                "val_data_path": self.validation_data_file.as_posix(),
            }

            if self.model is not None:
                param_files = self.model._parameter["data"]["directory"]
                base_path = {"base_model_checkpoint": param_files}
                try:
                    self._parameter["checkpoints"].update(base_path)
                except KeyError:
                    self._parameter["checkpoints"] = base_path

            check_duplicate_keys(custom_parameters, self._parameter["data"], log)
            self._parameter["data"].update(custom_parameters)

    def train_model(self):
        """Train the model using `apax.train.run`"""
        apax_run(self._parameter)

    def move_metrics(self):
        """Move the metrics to the correct directories for DVC"""
        path = self.model_directory / self.metrics_epoch.name
        shutil.move(path, self.metrics_epoch)

    def get_metrics_from_plots(self):
        """In addition to the plots write a model metric"""
        metrics_df = pd.read_csv(self.metrics_epoch)
        self.metrics = metrics_df.iloc[-1].to_dict()

    def run(self):
        """Primary method to run which executes all steps of the model training"""

        config.update("jax_enable_x64", self.jax_enable_x64)

        ase.io.write(self.train_data_file, self.data)
        ase.io.write(self.validation_data_file, self.validation_data)

        self.train_model()
        self.move_metrics()
        self.get_metrics_from_plots()

    def get_calculator(self, **kwargs):
        """Get an apax ase calculator"""
        with self.state.use_tmp_path():
            return ASECalculator(model_dir=self.model_directory)


class ApaxEnsemble(zntrack.Node):
    """Parallel apax model ensemble in ASE.

    Attributes
    ----------
    models: list
        List of `ApaxModel` nodes to ensemble.
    nl_skin: float
        Neighborlist skin.
    transformations: dict
        Key-parameter dict with function transformations applied
        to the model function within the ASE calculator.
        See the apax documentation for available methods.
    """

    models: list[Apax] = zntrack.deps()
    nl_skin: float = zntrack.params(0.5)
    transformations: dict[str, dict] = zntrack.params(None)

    def run(self) -> None:
        pass

    def get_calculator(self, **kwargs) -> ase.calculators.calculator.Calculator:
        """Property to return a model specific ase calculator object.

        Returns
        -------
        calc:
            ase calculator object
        """

        param_files = [m._parameter["data"]["directory"] for m in self.models]

        transformations = []
        if self.transformations:
            for transform, params in self.transformations.items():
                transformations.append(available_transformations[transform](**params))

        calc = ASECalculator(
            param_files,
            dr=self.nl_skin,
            transformations=transformations,
        )
        return calc
