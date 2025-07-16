import logging
import pathlib
import typing as t

import ase
import numpy as np
import pandas as pd
import yaml
import znh5md
import zntrack
from ase import Atoms

from apax.calibration import compute_calibration_factors
from apax.md import ASECalculator
from apax.md.function_transformations import (
    GaussianAcceleratedMolecularDynamics,
    GlobalCalibration,
    UncertaintyDrivenDynamics,
    available_transformations,
)
from apax.train.run import run as apax_run

from .utils import check_duplicate_keys

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
