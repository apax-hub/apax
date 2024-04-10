import logging
import pathlib
import typing as t

import ase.io
import pandas as pd
import yaml
import zntrack.utils

from apax.md import ASECalculator
from apax.md.function_transformations import available_transformations
from apax.train.run import run as apax_run

from .utils import check_duplicate_keys

log = logging.getLogger(__name__)


class Apax(zntrack.Node):
    """Class for the implementation of the apax model

    Parameters
    ----------
    config: str
        path to the apax config file
    data: list[ase.Atoms]
        the training data set
    validation_data: list[ase.Atoms]
        atoms object with the validation data set
    model: t.Optional[Apax]
        model to be used as a base model
    model_directory: pathlib.Path
        model directory
    train_data_file: pathlib.Path
        output path to the training data
    validation_data_file: pathlib.Path
        output path to the validation data
    """

    data: list = zntrack.deps()
    config: str = zntrack.params_path()
    validation_data = zntrack.deps()
    model: t.Optional[t.Any] = zntrack.deps(None)

    model_directory: pathlib.Path = zntrack.outs_path(zntrack.nwd / "apax_model")

    train_data_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "train_atoms.extxyz")
    validation_data_file: pathlib.Path = zntrack.outs_path(
        zntrack.nwd / "val_atoms.extxyz"
    )

    metrics = zntrack.metrics()

    _parameter: dict = None

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

    def get_metrics_from_plots(self):
        """In addition to the plots write a model metric"""
        metrics_df = pd.read_csv(self.model_directory / "log.csv")
        self.metrics = metrics_df.iloc[-1].to_dict()

    def run(self):
        """Primary method to run which executes all steps of the model training"""
        ase.io.write(self.train_data_file, self.data)
        ase.io.write(self.validation_data_file, self.validation_data)

        self.train_model()
        self.get_metrics_from_plots()

    def get_calculator(self, **kwargs):
        """Get an apax ase calculator"""
        with self.state.use_tmp_path():
            return ASECalculator(model_dir=self.model_directory)


class ApaxEnsemble(zntrack.Node):
    """Parallel apax model ensemble in ASE.

    Parameters
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
