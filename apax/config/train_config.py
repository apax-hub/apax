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
    PositiveFloat,
    PositiveInt,
    create_model,
    model_validator,
)
from typing_extensions import Annotated

from apax.config.lr_config import CyclicCosineLR, LinearLR
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
    reset_every : int
        | Number of epochs before reinitializing the ProcessPoolExcecutor.
        | Avoids memory leaks.
    """

    processing: Literal["pbp"] = "pbp"
    num_workers: PositiveInt = 10
    reset_every: PositiveInt = 10


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
    additional_properties_info : dict, optional
        | dict of property name, shape (ragged or fixed) pairs. Currently unused.
    energy_regularisation :
        | Magnitude of the regularization in the per-element energy regression.

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
    additional_properties_info: dict[str, str] = {}

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


class GaussianBasisConfig(BaseModel, extra="forbid"):
    """
    Gaussian primitive basis functions.

    Parameters
    ----------
    n_basis : PositiveInt, default = 7
        Number of uncontracted basis functions.
    r_min : NonNegativeFloat, default = 0.5
        Position of the first uncontracted basis function's mean.
    r_max : PositiveFloat, default = 6.0
        Cutoff radius of the descriptor.
    """

    name: Literal["gaussian"] = "gaussian"
    n_basis: PositiveInt = 7
    r_min: NonNegativeFloat = 0.5
    r_max: PositiveFloat = 6.0


class BesselBasisConfig(BaseModel, extra="forbid"):
    """
    Gaussian primitive basis functions.

    Parameters
    ----------
    n_basis : PositiveInt, default = 7
        Number of uncontracted basis functions.
    r_max : PositiveFloat, default = 6.0
        Cutoff radius of the descriptor.
    """

    name: Literal["bessel"] = "bessel"
    n_basis: PositiveInt = 7
    r_max: PositiveFloat = 6.0


BasisConfig = Union[GaussianBasisConfig, BesselBasisConfig]


class FullEnsembleConfig(BaseModel, extra="forbid"):
    """
    Configuration for full model ensembles.
    Usage can improve accuracy and stability at the cost of slower inference.
    Uncertainties will generally not be calibrated.

    Parameters
    ----------
    n_members : int
        Number of ensemble members.
    """

    kind: Literal["full"] = "full"
    n_members: int


class ShallowEnsembleConfig(BaseModel, extra="forbid"):
    """
    Configuration for shallow (last layer) ensembles.
    Allows use of probabilistic loss functions.
    The predicted uncertainties should be well calibrated.
    See 10.1088/2632-2153/ad594a for details.

    Parameters
    ----------
    n_members : int
        Number of ensemble members.
    force_variance : bool, default = True
        Whether or not to compute force uncertainties.
        Required for probabilistic force loss and calibration of force uncertainties.
        Can lead to better force metrics but but enabling it introduces some non-negligible cost.
    chunk_size : Optional[int], default = None
        If set to an integer, the jacobian of ensemble energies wrt. to positions will be computed
        in chunks of that size. This sacrifices some performance for the possibility to use relatively
        large ensemble sizes.
    """

    kind: Literal["shallow"] = "shallow"
    n_members: int
    force_variance: bool = True
    chunk_size: Optional[int] = None


EnsembleConfig = Union[FullEnsembleConfig, ShallowEnsembleConfig]



class BaseModelConfig(BaseModel, extra="forbid"):

    nn: List[PositiveInt] = [512, 512]
    w_init: Literal["normal", "lecun"] = "normal"
    b_init: Literal["normal", "zeros"] = "normal"
    use_ntk: bool = True

    ensemble: Optional[EnsembleConfig] = None

    # corrections
    use_zbl: bool = False

    calc_stress: bool = False

    descriptor_dtype: Literal["fp32", "fp64"] = "fp64"
    readout_dtype: Literal["fp32", "fp64"] = "fp32"
    scale_shift_dtype: Literal["fp32", "fp64"] = "fp32"

    def get_dict(self):
        import jax.numpy as jnp

        model_dict = self.model_dump()
        prec_dict = {"fp32": jnp.float32, "fp64": jnp.float64}
        model_dict["descriptor_dtype"] = prec_dict[model_dict["descriptor_dtype"]]
        model_dict["readout_dtype"] = prec_dict[model_dict["readout_dtype"]]
        model_dict["scale_shift_dtype"] = prec_dict[model_dict["scale_shift_dtype"]]

        return model_dict


class GMNNConfig(BaseModelConfig, extra="forbid"):
    """
    Configuration for the model.

    Parameters
    ----------
    basis : BasisConfig, default = GaussianBasisConfig()
        Configuration for primitive basis funtions.
    n_radial : PositiveInt, default = 5
        Number of contracted basis functions.
    n_contr : int, default = 8
        How many gaussian moment contractions to use.
    emb_init : Optional[str], default = "uniform"
        Initialization scheme for embedding layer weights.
    nn : List[PositiveInt], default = [512, 512]
        Number of hidden layers and units in those layers.
    w_init : Literal["normal", "lecun"], default = "normal"
        Initialization scheme for the neural network weights.
    b_init : Literal["normal", "zeros"], default = "normal"
        Initialization scheme for the neural network biases.
    use_ntk : bool, default = True
        Whether or not to use NTK parametrization.
    ensemble : Optional[EnsembleConfig], default = None
        What kind of model ensemble to use (optional).
    use_zbl : bool, default = False
        Whether to include the ZBL correction.
    calc_stress : bool, default = False
        Whether to calculate stress during model evaluation.
    descriptor_dtype : Literal["fp32", "fp64"], default = "fp64"
        Data type for descriptor calculations.
    readout_dtype : Literal["fp32", "fp64"], default = "fp32"
        Data type for readout calculations.
    scale_shift_dtype : Literal["fp32", "fp64"], default = "fp32"
        Data type for scale and shift parameters.
    """


    name: Literal["gmnn"] = "gmnn"

    basis: BasisConfig = Field(GaussianBasisConfig(name="gaussian"), discriminator="name")
    n_radial: PositiveInt = 5
    n_contr: int = 8
    emb_init: Optional[str] = "uniform"
    
    def get_builder(self):
        from apax.nn.builder import GMNNBuilder
        return GMNNBuilder


class EquivMPConfig(BaseModelConfig, extra="forbid"):
    """
    Configuration for the model.

    Parameters
    ----------
    basis : BasisConfig, default = GaussianBasisConfig()
        Configuration for primitive basis funtions.
    n_radial : PositiveInt, default = 5
        Number of contracted basis functions.
    n_contr : int, default = 8
        How many gaussian moment contractions to use.
    emb_init : Optional[str], default = "uniform"
        Initialization scheme for embedding layer weights.
    nn : List[PositiveInt], default = [512, 512]
        Number of hidden layers and units in those layers.
    w_init : Literal["normal", "lecun"], default = "normal"
        Initialization scheme for the neural network weights.
    b_init : Literal["normal", "zeros"], default = "normal"
        Initialization scheme for the neural network biases.
    use_ntk : bool, default = True
        Whether or not to use NTK parametrization.
    ensemble : Optional[EnsembleConfig], default = None
        What kind of model ensemble to use (optional).
    use_zbl : bool, default = False
        Whether to include the ZBL correction.
    calc_stress : bool, default = False
        Whether to calculate stress during model evaluation.
    descriptor_dtype : Literal["fp32", "fp64"], default = "fp64"
        Data type for descriptor calculations.
    readout_dtype : Literal["fp32", "fp64"], default = "fp32"
        Data type for readout calculations.
    scale_shift_dtype : Literal["fp32", "fp64"], default = "fp32"
        Data type for scale and shift parameters.
    """
    
    name: Literal["equiv-mp"] = "equiv-mp"

    basis: BasisConfig = Field(GaussianBasisConfig(name="gaussian"), discriminator="name")

    features: PositiveInt = 32
    max_degree: PositiveInt = 2
    num_iterations: PositiveInt = 1
    
    def get_builder(self):
        from apax.nn.builder import EquivMPBuilder
        return EquivMPBuilder



ModelConfig = Union[GMNNConfig, EquivMPConfig]



class OptimizerConfig(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration of the optimizer.
    Learning rates of 0 will freeze the respective parameters.

    Parameters
    ----------
    name : str, default = "adam"
        Name of the optimizer. Can be any `optax` optimizer.
    emb_lr : NonNegativeFloat, default = 0.02
        Learning rate of the elemental embedding contraction coefficients.
    nn_lr : NonNegativeFloat, default = 0.03
        Learning rate of the neural network parameters.
    scale_lr : NonNegativeFloat, default = 0.001
        Learning rate of the elemental output scaling factors.
    shift_lr : NonNegativeFloat, default = 0.05
        Learning rate of the elemental output shifts.
    zbl_lr : NonNegativeFloat, default = 0.001
        Learning rate of the ZBL correction parameters.
    schedule : LRSchedule = LinearLR
        Learning rate schedule.
    kwargs : dict, default = {}
        Optimizer keyword arguments. Passed to the `optax` optimizer.
    """

    name: str = "adam"
    emb_lr: NonNegativeFloat = 0.02
    nn_lr: NonNegativeFloat = 0.03
    scale_lr: NonNegativeFloat = 0.001
    shift_lr: NonNegativeFloat = 0.05
    zbl_lr: NonNegativeFloat = 0.001
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

    ckpt_interval: PositiveInt = 1
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
    model: ModelConfig = Field(
        GMNNConfig(name="gmnn"), discriminator="name"
    )
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
