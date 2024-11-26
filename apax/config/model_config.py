from typing import List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)


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
    n_basis : PositiveInt, default = 16
        Number of uncontracted basis functions.
    r_max : PositiveFloat, default = 5.0
        Cutoff radius of the descriptor.
    """

    name: Literal["bessel"] = "bessel"
    n_basis: PositiveInt = 16
    r_max: PositiveFloat = 5.0


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

    Hint
    ----------
    Loss type hase to be changed to a probabalistic loss like 'nll' or 'crps'

    """

    kind: Literal["shallow"] = "shallow"
    n_members: int
    force_variance: bool = True
    chunk_size: Optional[int] = None


EnsembleConfig = Union[FullEnsembleConfig, ShallowEnsembleConfig]


class Correction(BaseModel, extra="forbid"):
    name: str


class ZBLRepulsion(Correction, extra="forbid"):
    name: Literal["zbl"]
    r_max: NonNegativeFloat = 1.5


class ExponentialRepulsion(Correction, extra="forbid"):
    name: Literal["exponential"]
    r_max: NonNegativeFloat = 1.5


class LatentEwald(Correction, extra="forbid"):
    name: Literal["latent_ewald"]
    kgrid: list
    sigma: float = 1.0


EmpiricalCorrection = Union[ZBLRepulsion, ExponentialRepulsion, LatentEwald]


class PropertyHead(BaseModel, extra="forbid"):
    """ """

    name: str
    aggregation: str = "none"
    mode: str = "l0"

    nn: List[PositiveInt] = [128, 128]
    n_shallow_members: int = 0
    w_init: Literal["normal", "lecun"] = "lecun"
    b_init: Literal["normal", "zeros"] = "zeros"
    use_ntk: bool = False
    dtype: Literal["fp32", "fp64"] = "fp32"


class BaseModelConfig(BaseModel, extra="forbid"):
    """
    Configuration for the model.

    Parameters
    ----------
    basis : BasisConfig, default = GaussianBasisConfig()
        Configuration for primitive basis funtions.
    nn : List[PositiveInt], default = [256, 256]
        Number of hidden layers and units in those layers.
    w_init : Literal["normal", "lecun"], default = "lecun"
        Initialization scheme for the neural network weights.
    b_init : Literal["normal", "zeros"], default = "zeros"
        Initialization scheme for the neural network biases.
    use_ntk : bool, default = False
        Whether or not to use NTK parametrization.
    ensemble : Optional[EnsembleConfig], default = None
        What kind of model ensemble to use (optional).
    use_zbl : bool, default = False
        Whether to include the ZBL correction.
    calc_stress : bool, default = False
        Whether to calculate stress during model evaluation.
    descriptor_dtype : Literal["fp32", "fp64"], default = "fp32"
        Data type for descriptor calculations.
    readout_dtype : Literal["fp32", "fp64"], default = "fp32"
        Data type for readout calculations.
    scale_shift_dtype : Literal["fp32", "fp64"], default = "fp64"
        Data type for scale and shift parameters.
    """

    basis: BasisConfig = Field(BesselBasisConfig(name="bessel"), discriminator="name")

    nn: List[PositiveInt] = [256, 256]
    w_init: Literal["normal", "lecun"] = "lecun"
    b_init: Literal["normal", "zeros"] = "zeros"
    use_ntk: bool = False

    ensemble: Optional[EnsembleConfig] = None

    property_heads: list[PropertyHead] = []

    # corrections
    empirical_corrections: list[EmpiricalCorrection] = []

    calc_stress: bool = False

    descriptor_dtype: Literal["fp32", "fp64"] = "fp32"
    readout_dtype: Literal["fp32", "fp64"] = "fp32"
    scale_shift_dtype: Literal["fp32", "fp64"] = "fp64"


class GMNNConfig(BaseModelConfig, extra="forbid"):
    """
    Configuration for the model.

    Parameters
    ----------
    n_radial : PositiveInt, default = 5
        Number of contracted basis functions.
    n_contr : int, default = 8
        How many gaussian moment contractions to use.
    emb_init : Optional[str], default = "uniform"
        Initialization scheme for embedding layer weights.
    """

    name: Literal["gmnn"] = "gmnn"

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
    features: PositiveInt = 32
        Feature dimension of the linear layers
    max_degree: PositiveInt = 2
        Maximal rotation order for features and tensorproducts
    num_iterations: PositiveInt = 1
        Number of message passing steps.
    """

    name: Literal["equiv-mp"] = "equiv-mp"

    features: PositiveInt = 32
    max_degree: PositiveInt = 2
    num_iterations: PositiveInt = 1

    def get_builder(self):
        from apax.nn.builder import EquivMPBuilder

        return EquivMPBuilder


class So3kratesConfig(BaseModelConfig, extra="forbid"):
    """
    Configuration for the model.

    Parameters
    ----------
    num_layers: PositiveInt = 1
        Number of message passing layers
    max_degree: PositiveInt = 3
        Maximum rotation order
    num_features: PositiveInt = 128
        Feature dimension
    num_heads: PositiveInt = 4
        Number of attention heads
    use_layer_norm_1: bool = False
        Layer norm in transformer block
    use_layer_norm_2: bool = False
        Layer norm in transformer block
    use_layer_norm_final: bool = False
        Layer norm before readout
    activation: str = "silu"
        Activation function
    cutoff_fn: str = "cosine_cutoff"
        Smooth cutoff function
    transform_input_features: bool = False
        Whether or not to apply a dense layer to transformer input features

    """

    name: Literal["so3krates"] = "so3krates"

    num_layers: PositiveInt = 1
    max_degree: PositiveInt = 3
    num_features: PositiveInt = 128
    num_heads: PositiveInt = 4
    use_layer_norm_1: bool = False
    use_layer_norm_2: bool = False
    use_layer_norm_final: bool = False
    activation: str = "silu"
    cutoff_fn: str = "cosine_cutoff"
    transform_input_features: bool = False

    def get_builder(self):
        from apax.nn.builder import So3kratesBuilder

        return So3kratesBuilder


ModelConfig = Union[GMNNConfig, EquivMPConfig, So3kratesConfig]
