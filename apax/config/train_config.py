import os
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Extra, NonNegativeFloat, PositiveFloat, PositiveInt


class DataConfig(BaseModel, extra=Extra.forbid):
    """
    Configuration for data loading, preprocessing and training.

    Parameters
    ----------
    model_path: Path to the directory where the training results and
        checkpoints will be written.
    model_name: Name of  the model. Distinguishes it from the other models
        trained in the same `model_path`.
    data_path: Path to a single dataset file. Set either this or `val_data_path` and
        `train_data_path`.
    train_data_path: Path to a training dataset. Set this and `val_data_path`
        if your data comes pre-split.
    val_data_path: Path to a validation dataset. Set this and `train_data_path`
        if your data comes pre-split.
    test_data_path: Path to a test dataset. Set this, `train_data_path` and
        `val_data_path` if your data comes pre-split.
    n_train: Number of training datapoints from `data_path`.
    n_valid: Number of validation datapoints from `data_path`.
    batch_size: Number of training examples to be evaluated at once.
    valid_batch_size: Number of validation examples to be evaluated at once.
    shuffle_buffer_size: Size of the `tf.data` shuffle buffer.
    energy_regularisation: Magnitude of the regularization in the per-element
        energy regression.
    """

    model_path: str
    model_name: str
    data_path: Optional[str] = None
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    pos_unit: Optional[str] = "Ang"
    energy_unit: Optional[str] = "eV"

    n_train: PositiveInt = 1000
    n_valid: PositiveInt = 100
    batch_size: PositiveInt = 32
    valid_batch_size: PositiveInt = 100
    shuffle_buffer_size: PositiveInt = 1000

    energy_regularisation: NonNegativeFloat = 1.0


class ModelConfig(BaseModel, extra=Extra.forbid):
    """
    Configuration for the model.

    Parameters
    ----------
    n_basis: Number of uncontracted gaussian basis functions.
    n_radial: Number of contracted basis functions.
    r_min: Position of the first uncontracted basis function's mean.
    r_max: Cutoff radius of the descriptor.
    nn: Number of hidden layers and units in those layers.
    b_init: Initialization scheme for the neural network biases.
        Either `normal` or `zeros`.
    """

    n_basis: PositiveInt = 7
    n_radial: PositiveInt = 5
    r_min: NonNegativeFloat = 0.5
    r_max: PositiveFloat = 6.0

    nn: List[PositiveInt] = [512, 512]
    b_init: Literal["normal", "zeros"] = "normal"

    descriptor_dtype: Literal["fp32", "fp64"] = "fp32"
    readout_dtype: Literal["fp32", "fp64"] = "fp32"
    scale_shift_dtype: Literal["fp32", "fp64"] = "fp32"

    def get_dict(self):
        import jax.numpy as jnp

        model_dict = self.dict()
        prec_dict = {"fp32": jnp.float32, "fp64": jnp.float64}
        model_dict["descriptor_dtype"] = prec_dict[model_dict["descriptor_dtype"]]
        model_dict["readout_dtype"] = prec_dict[model_dict["readout_dtype"]]
        model_dict["scale_shift_dtype"] = prec_dict[model_dict["scale_shift_dtype"]]

        return model_dict


class OptimizerConfig(BaseModel, frozen=True, extra=Extra.forbid):
    """
    Configuration of the optimizer.
    Learning rates of 0 will freeze the respective parameters.

    Parameters
    ----------
    opt_name: Name of the optimizer. Can be any `optax` optimizer.
    emb_lr: Learning rate of the elemental embedding contraction coefficients.
    nn_lr: Learning rate of the neural network parameters.
    scale_lr: Learning rate of the elemental output scaling factors.
    shift_lr: Learning rate of the elemental output shifts.
    transition_begin: Number of training steps (not epochs) before the start of the
        linear learning rate schedule.
    opt_kwargs: Optimizer keyword arguments. Passed to the `optax` optimizer.
    """

    opt_name: str = "adam"
    emb_lr: NonNegativeFloat = 0.02
    nn_lr: NonNegativeFloat = 0.03
    scale_lr: NonNegativeFloat = 0.001
    shift_lr: NonNegativeFloat = 0.05
    transition_begin: int = 0
    opt_kwargs: dict = {}


class MetricsConfig(BaseModel, extra=Extra.forbid):
    """
    Configuration for the metrics collected during training.

    Parameters
    ----------
    name: Keyword of the quantity e.g `energy`.
    reductions: List of reductions performed on the difference between
        target and predictions. Can be mae, mse, rmse for energies and forces.
        For forces it is also possible to use `angle`.
    """

    name: str
    reductions: List[str]


class LossConfig(BaseModel, extra=Extra.forbid):
    """
    Confuration of the loss functions used during training.

    Parameters
    ----------
    name: Keyword of the quantity e.g `energy`.
    loss_type: Weighting scheme for atomic contributions. See the MLIP package
        for reference 10.1088/2632-2153/abc9fe for details
    weight: Weighting factor in the overall loss function.
    """

    name: str
    loss_type: str = "molecules"
    weight: NonNegativeFloat = 1.0


class CallbackConfig(BaseModel, frozen=True, extra=Extra.forbid):
    """
    Configuraton of the training callbacks.

    Parameters
    ----------
    name: Keyword of the callback used. Currently we implement "csv" and "tensorboard".
    """

    name: str


class TrainProgressbarConfig(BaseModel, extra=Extra.forbid):
    """
    Configuration of progressbars.

    Parameters
    ----------
    disable_epoch_pbar: Set to True to disable the epoch progress bar.
    disable_nl_pbar: Set to True to disable the NL precomputation progress bar.
    """

    disable_epoch_pbar: bool = False
    disable_nl_pbar: bool = False


class CheckpointConfig(BaseModel, extra=Extra.forbid):
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


class Config(BaseModel, frozen=True, extra=Extra.forbid):
    """
    Main configuration of a apax training run.

    Parameters
    ----------

    n_epochs: Number of training epochs.
    seed: Random seed.
    data: :class: `Data` <config.DataConfig> configuration.
    model: :class: `Model` <config.ModelConfig> configuration.
    metrics: List of :class: `metric` <config.MetricsConfig> configurations.
    loss: List of :class: `loss` <config.LossConfig> function configurations.
    optimizer: :class: `Optimizer` <config.OptimizerConfig> configuration.
    callbacks: List of :class: `callback` <config.CallbackConfig> configurations.
    progress_bar: Progressbar configuration.
    checkpoints: Checkpoint configuration.
    maximize_l2_cache: Whether or not to maximize GPU L2 cache.
    """

    n_epochs: PositiveInt
    seed: int = 1
    use_flax: bool = True

    data: DataConfig
    model: ModelConfig = ModelConfig()
    metrics: List[MetricsConfig] = []
    loss: List[LossConfig]
    optimizer: OptimizerConfig = OptimizerConfig()
    callbacks: List[CallbackConfig] = [CallbackConfig(name="csv")]
    progress_bar: TrainProgressbarConfig = TrainProgressbarConfig()
    checkpoints: CheckpointConfig = CheckpointConfig()
    maximize_l2_cache: bool = False

    def dump_config(self, save_path):
        """
        Writes the current config file to the specified directory.

        Parameters
        ----------
        save_path: Path to the directory.
        """
        with open(os.path.join(save_path, "config.yaml"), "w") as conf:
            yaml.dump(self.dict(), conf, default_flow_style=False)