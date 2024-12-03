import logging
import os
from pathlib import Path
from typing import List, Literal, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
    create_model,
    model_validator,
)
from typing_extensions import Annotated

from apax.config.lr_config import CyclicCosineLR, LinearLR
from apax.config.model_config import GMNNConfig, ModelConfig
from apax.data.statistics import scale_method_list, shift_method_list

log = logging.getLogger(__name__)


class DatasetConfig(BaseModel, extra="forbid"):
    processing: str


class CachedDataset(DatasetConfig, extra="forbid"):
    """Dataset which pads everything (atoms, neighbors)
    to the largest system in the dataset.
    The NL is computed on the fly during the first epoch and stored to disk using
    tf.data's cache.
    Most performant option for datasets with samples of very similar size.

    Parameters
    ----------
    shuffle_buffer_size : int
        | Size of the buffer that is shuffled by tf.data.
        | Larger values require more RAM.
    """

    processing: Literal["cached"] = "cached"
    shuffle_buffer_size: PositiveInt = 1000


class OTFDataset(DatasetConfig, extra="forbid"):
    """Dataset which pads everything (atoms, neighbors)
    to the largest system in the dataset.
    The NL is computed on the fly and fed into a tf.data generator.
    Mostly for internal purposes.

    Parameters
    ----------
    shuffle_buffer_size : int
        | Size of the buffer that is shuffled by tf.data.
        | Larger values require more RAM.
    """

    processing: Literal["otf"] = "otf"
    shuffle_buffer_size: PositiveInt = 1000


class PBPDatset(DatasetConfig, extra="forbid"):
    """Dataset which pads everything (atoms, neighbors)
    to the next larges power of two.
    This limits the compute wasted due to padding at the (negligible)
    cost of some recompilations.
    The NL is computed on-the-fly in parallel for `num_workers` of batches.
    Does not use tf.data.

    Most performant option for datasets with significantly differently sized systems
    (e.g. MP, SPICE).

    Parameters
    ----------
    num_workers : int
        | Number of batches to be processed in parallel.
    atom_padding : int
        | Next nearest integer to which to pad per-atom arrays (positions, forces, ...).
    nl_padding: int
        | Next nearest integer to which to pad neighborlists.
    """

    processing: Literal["pbp"] = "pbp"
    num_workers: PositiveInt = 10
    atom_padding: PositiveInt = 10
    nl_padding: PositiveInt = 2000


class DataConfig(BaseModel, extra="forbid"):
    """
    Configuration for data loading, preprocessing and training.

    Parameters
    ----------
    directory : str, required
        | Path to directory where training results and checkpoints are written.
    experiment : str, required
        | Model name distinguishing from others in directory.
    data_path : str, required if train_data_path and val_data_path is not specified
        | Path to single dataset file.
    train_data_path : str, required if data_path is not specified
        | Path to training dataset.
    val_data_path : str, required if data_path is not specified
        | Path to validation dataset.
    test_data_path : str, optional
        | Path to test dataset.
    n_train : int, default = 1000
        | Number of training datapoints from `data_path`.
    n_valid : int, default = 100
        | Number of validation datapoints from `data_path`.
    batch_size : int, default = 32
        | Number of training examples to be evaluated at once.
    valid_batch_size : int, default = 100
        | Number of validation examples to be evaluated at once.
    shuffle_buffer_size : int, default = 1000
        | Size of the `tf.data` shuffle buffer.
    energy_regularisation :
        | Magnitude of the regularization in the per-element energy regression.
    pos_unit : str, default = "Ang"
        unit of length
    energy_unit : str, default = "eV"
        unit of energy
    """

    directory: str
    experiment: str
    dataset: Union[CachedDataset, OTFDataset, PBPDatset] = Field(
        CachedDataset(processing="cached"), discriminator="processing"
    )

    data_path: Optional[str] = None
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    test_data_path: Optional[str] = None

    n_train: PositiveInt = 1000
    n_valid: PositiveInt = 100
    batch_size: PositiveInt = 32
    valid_batch_size: PositiveInt = 100

    shift_method: str = "per_element_regression_shift"
    shift_options: dict = {"energy_regularisation": 1.0}

    scale_method: str = "per_element_force_rms_scale"
    scale_options: Optional[dict] = {}

    pos_unit: Optional[str] = "Ang"
    energy_unit: Optional[str] = "eV"

    @model_validator(mode="after")
    def set_data_or_train_val_path(self):
        not_data_path = self.data_path is None
        not_train_path = self.train_data_path is None

        neither_set = not_data_path and not_train_path
        both_set = not not_data_path and not not_train_path

        if neither_set or both_set:
            raise ValueError("Please specify either data_path or train_data_path")

        return self

    @model_validator(mode="after")
    def validate_shift_scale_methods(self):
        method_lists = [shift_method_list, scale_method_list]
        requested_methods = [self.shift_method, self.scale_method]
        requested_options = [self.shift_options, self.scale_options]

        cases = zip(method_lists, requested_methods, requested_options)
        for method_list, requested_method, requested_params in cases:
            methods = {method.name: method for method in method_list}

            # check if method exists
            if requested_method not in methods.keys():
                raise KeyError(
                    f"The initialization method '{requested_method}' is not among the"
                    f" implemented methods. Choose from {methods.keys()}"
                )

            # check if parameters names are complete and correct
            method = methods[requested_method]
            fields = {
                name: (dtype, ...)
                for name, dtype in zip(method.parameters, method.dtypes)
            }
            MethodConfig = create_model(
                f"{method.name}Config", __config__=ConfigDict(extra="forbid"), **fields
            )

            _ = MethodConfig(**requested_params)

        return self

    @property
    def model_version_path(self):
        version_path = Path(self.directory) / self.experiment
        return version_path

    @property
    def best_model_path(self):
        return self.model_version_path / "best"


class OptimizerConfig(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration of the optimizer.
    Learning rates of 0 will freeze the respective parameters.

    Parameters
    ----------
    name : str, default = "adam"
        Name of the optimizer. Can be any `optax` optimizer.
    emb_lr : NonNegativeFloat, default = 0.001
        Learning rate of the elemental embedding contraction coefficients.
    nn_lr : NonNegativeFloat, default = 0.001
        Learning rate of the neural network parameters.
    scale_lr : NonNegativeFloat, default = 0.0001
        Learning rate of the elemental output scaling factors.
    shift_lr : NonNegativeFloat, default = 0.003
        Learning rate of the elemental output shifts.
    zbl_lr : NonNegativeFloat, default = 0.0001
        Learning rate of the ZBL correction parameters.
    rep_scale_lr : NonNegativeFloat, default = 0.001
        LR for the length scale of thes exponential repulsion potential.
    rep_prefactor_lr : NonNegativeFloat, default = 0.0001
        LR for the strength of the exponential repulsion potential.
    gradient_clipping: NonNegativeFloat, default = 1000.0
        Per element Gradient clipping value.
        Default is so high that it effectively disabled.
    schedule : LRSchedule = LinearLR
        Learning rate schedule.
    kwargs : dict, default = {}
        Optimizer keyword arguments. Passed to the `optax` optimizer.
    """

    name: str = "adam"
    emb_lr: NonNegativeFloat = 0.001
    nn_lr: NonNegativeFloat = 0.001
    scale_lr: NonNegativeFloat = 0.0001
    shift_lr: NonNegativeFloat = 0.003
    zbl_lr: NonNegativeFloat = 0.0001
    rep_scale_lr: NonNegativeFloat = 0.001
    rep_prefactor_lr: NonNegativeFloat = 0.0001

    gradient_clipping: NonNegativeFloat = 1000.0

    schedule: Union[LinearLR, CyclicCosineLR] = Field(
        LinearLR(name="linear"), discriminator="name"
    )
    kwargs: dict = {}


class MetricsConfig(BaseModel, extra="forbid"):
    """
    Configuration for the metrics collected during training.

    Parameters
    ----------
    name : str
        Keyword of the quantity, e.g., 'energy'.
    reductions : List[str]
        List of reductions performed on the difference between target and predictions.
        Can be 'mae', 'mse', 'rmse' for energies and forces.
        For forces, 'angle' can also be used.
    """

    name: str
    reductions: List[str]


class LossConfig(BaseModel, extra="forbid"):
    """
    Configuration of the loss functions used during training.

    Parameters
    ----------
    name : str
        Keyword of the quantity, e.g., 'energy'.
    loss_type : str, optional
        Weighting scheme for atomic contributions. See the MLIP package
        for reference 10.1088/2632-2153/abc9fe for details, by default "mse".
    weight : NonNegativeFloat, optional
        Weighting factor in the overall loss function, by default 1.0.
    atoms_exponent : NonNegativeFloat, optional
        Exponent for atomic contributions weighting, by default 1.
    parameters : dict, optional
        Additional parameters for configuring the loss function, by default {}.

    Notes
    -----
    This class specifies the configuration of the loss functions used during training.
    """

    name: str
    loss_type: str = "mse"
    weight: NonNegativeFloat = 1.0
    atoms_exponent: NonNegativeFloat = 1
    parameters: dict = {}


class CSVCallback(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration of the CSVCallback.

    Parameters
    ----------
    name: Keyword of the callback used..
    """

    name: Literal["csv"]


class TBCallback(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration of the TensorBoard callback.

    Parameters
    ----------
    name: Keyword of the callback used..
    """

    name: Literal["tensorboard"]


class MLFlowCallback(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration of the MLFlow callback.

    Parameters
    ----------
    name: Keyword of the callback used.
    experiment: Path to the MLFlow experiment, e.g. /Users/<user>/<my_experiment>
    """

    name: Literal["mlflow"]
    experiment: str


CallBack = Annotated[
    Union[CSVCallback, TBCallback, MLFlowCallback], Field(discriminator="name")
]


class TrainProgressbarConfig(BaseModel, extra="forbid"):
    """
    Configuration of progressbars.

    Parameters
    ----------
    disable_epoch_pbar: Set to True to disable the epoch progress bar.
    disable_batch_pbar: Set to True to disable the batch progress bar.
    """

    disable_epoch_pbar: bool = False
    disable_batch_pbar: bool = True


class CheckpointConfig(BaseModel, extra="forbid"):
    """
    Checkpoint configuration.

    Parameters
    ----------
    ckpt_interval: Number of epochs between checkpoints.
    base_model_checkpoint: Path to the folder containing a pre-trained model ckpt.
    reset_layers: List of layer names for which the parameters will be reinitialized.
    """

    ckpt_interval: PositiveInt = 500
    base_model_checkpoint: Optional[str] = None
    reset_layers: List[str] = []


class WeightAverage(BaseModel, extra="forbid"):
    """Applies an exponential moving average to model parameters.

    Parameters
    ----------
    ema_start : int, default = 1
        Epoch at which to start averaging models.
    alpha : float, default = 0.9
        How much of the new model to use. 1.0 would mean no averaging, 0.0 no updates.
    """

    ema_start: int = 0
    alpha: float = 0.9


class Config(BaseModel, frozen=True, extra="forbid"):
    """
    Main configuration of a apax training run. Parameter that are config classes will
    be generated by parsing the config.yaml file and are specified
    as shown :ref:`here <train_config>`:

    Example
    -------
    .. code-block:: yaml

        data:
            directory: models/
            experiment: apax
                .
                .

    Parameters
    ----------
    n_epochs : int, required
        | Number of training epochs.
    patience : int, optional
        | Number of epochs without improvement before trainings gets terminated.
    seed : int, default = 1
        | Random seed.
    n_jitted_steps : int, default = 1
        | Number of train batches to be processed in a compiled loop.
        | Can yield significant speedups for small structures or small batch sizes.
    data : :class:`.DataConfig`
        | Data configuration.
    model : :class:`.ModelConfig`
        | Model configuration.
    metrics : List of :class:`.MetricsConfig`
        | Metrics configuration.
    loss : List of :class:`.LossConfig`
        | Loss configuration.
    optimizer : :class:`.OptimizerConfig`
        | Loss optimizer configuration.
    weight_average : :class:`.WeightAverage`, optional
        | Options for averaging weights between epochs.
    callbacks : List of various CallBack classes
        | Possible callbacks are :class:`.CSVCallback`,
        | :class:`.TBCallback`, :class:`.MLFlowCallback`
    progress_bar : :class:`.TrainProgressbarConfig`
        | Progressbar configuration.
    checkpoints : :class:`.CheckpointConfig`
        | Checkpoint configuration.
    data_parallel : bool, default = True
        | Automatically uses all available GPUs for data parallel training.
        | Set to false to force single device training.
    """

    n_epochs: PositiveInt
    patience: Optional[PositiveInt] = None
    seed: int = 1
    n_jitted_steps: int = 1
    data_parallel: bool = True

    data: DataConfig
    model: ModelConfig = Field(GMNNConfig(name="gmnn"), discriminator="name")
    metrics: List[MetricsConfig] = []
    loss: List[LossConfig]
    optimizer: OptimizerConfig = OptimizerConfig()
    weight_average: Optional[WeightAverage] = None
    callbacks: List[CallBack] = [CSVCallback(name="csv")]
    progress_bar: TrainProgressbarConfig = TrainProgressbarConfig()
    checkpoints: CheckpointConfig = CheckpointConfig()

    def dump_config(self, save_path):
        """
        Writes the current config file to the specified directory.

        Parameters
        ----------
        save_path: Path to the directory.
        """
        with open(os.path.join(save_path, "config.yaml"), "w") as conf:
            yaml.dump(self.model_dump(), conf, default_flow_style=False)
