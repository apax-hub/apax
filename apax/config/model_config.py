from typing import List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)


class GaussianBasisConfig(BaseModel, extra="forbid"):
    """
    Gaussian primitive basis functions.

    Parameters
    ----------
    n_basis : PositiveInt, default = 12
        Number of uncontracted basis functions.
    r_min : NonNegativeFloat, default = 0.5
        Position of the first uncontracted basis function's mean.
    r_max : PositiveFloat, default = 6.0
        Cutoff radius of the descriptor.
    spacing: Literal['linear', 'exponential'], default = 'linear'
        Spacing of centers of Gaussians. `"exponential"` results in more basis
        functions closer to `r_min`, and less at `r_max`.
        See https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181, Figure 2
    """

    name: Literal["gaussian"] = "gaussian"
    n_basis: PositiveInt = 12
    r_min: NonNegativeFloat = 0.5
    r_max: PositiveFloat = 6.0
    spacing: Literal["linear", "exponential"] = "linear"


class BesselBasisConfig(BaseModel, extra="forbid"):
    """
    Bessel basis functions.

    Parameters
    ----------
    n_basis : PositiveInt, default = 12
        Number of uncontracted basis functions.
    r_max : PositiveFloat, default = 5.0
        Cutoff radius of the descriptor.
    """

    name: Literal["bessel"] = "bessel"
    n_basis: PositiveInt = 12
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
    Loss type hase to be changed to a probabilistic loss like 'nll' or 'crps'

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
    # Coulomb constant e^2/(4*pi*eps0) = 14.4 eV*Ang (=1 in Hartree/Bohr).
    initial_rep_scale: PositiveFloat = 14.4


class ExponentialRepulsion(Correction, extra="forbid"):
    name: Literal["exponential"]
    r_max: NonNegativeFloat = 1.5


class LatentEwald(Correction, extra="forbid"):
    name: Literal["latent_ewald"]
    kgrid: list
    sigma: float = 1.0
    use_property: str = "charges"


class NLHRepulsion(Correction, extra="forbid"):
    name: Literal["nlh"]
    r_max: PositiveFloat = 6.0
    # path to triple-exponential coeffs; None -> bundled apax/data/nlh_coeffs.dat
    coeffs_file: Optional[str] = None


EmpiricalCorrection = Union[ZBLRepulsion, ExponentialRepulsion, LatentEwald, NLHRepulsion]


class PropertyHead(BaseModel, extra="forbid"):
    """
    Configuration for property heads.

    Parameters
    ----------
    name : str
        Name of the property.
    aggregation : str, default = "none"
        Aggregation method for atomic contributions.
    mode : str, default = "l0"
        Rotation order of the property.
    nn : List[PositiveInt], default = [128, 128]
        Number of hidden layers and units in those layers.
    n_shallow_members : int, default = 0
        Number of shallow ensemble members for this head.
    w_init : Literal["normal", "lecun"], default = "lecun"
        Initialization scheme for the neural network weights.
    b_init : Literal["normal", "zeros"], default = "zeros"
        Initialization scheme for the neural network biases.
    use_ntk : bool, default = False
        Whether or not to use NTK parametrization.
    dtype : Literal["fp32", "fp64"], default = "fp32"
        Data type for property head calculations.
    """

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
    basis : BasisConfig, default = BesselBasisConfig()
        Configuration for primitive basis functions.
    radial_transform : Literal["identity", "covalent"], default = "identity"
        Transform applied to interatomic distances before featurization.
        `"identity"` featurizes the physical distance unchanged (original
        behavior). `"covalent"` applies a per-element-pair tanh transform
        centered at the sum of covalent radii, reallocating basis resolution
        toward the bonding region.
    nn : List[PositiveInt], default = [256, 256]
        Number of hidden layers and units in those layers.
    w_init : Literal["normal", "lecun"], default = "lecun"
        Initialization scheme for the neural network weights.
    b_init : Literal["normal", "zeros"], default = "zeros"
        Initialization scheme for the neural network biases.
    activation_fn: str, default = "variance_preserving_swish"
        Activation function to use. Options are those shown at
        https://docs.jax.dev/en/latest/jax.nn.html and `variance_preserving_swish`,
        which is a variant of swish that preserves the second moment of the
        input.
    use_ntk : bool, default = False
        Whether or not to use NTK parametrization.
    use_bias : bool, default = False
        Whether the readout layers have biases. Without biases an atom with no
        neighbours gets exactly its isolated-atom energy shift.
    readout_activation : Literal["identity", "swish", "elu"], default = "elu"
        Activation applied to the readout output before the per-element scale.
        All options are 0 at 0. `"elu"` is unbounded above and keeps the scale
        small when isolated-atom energy shifts are used.
    ensemble : Optional[EnsembleConfig], default = None
        What kind of model ensemble to use (optional).
    property_heads : list[PropertyHead], default = []
        List of property heads to include.
    empirical_corrections : list[EmpiricalCorrection], default = []
        List of empirical corrections to include.
    calc_stress : bool, default = False
        Whether to calculate stress during model evaluation.
    calc_hessian : bool, default = False
        Whether to calculate Hessians during model evaluation.
    descriptor_dtype : Literal["fp32", "fp64"], default = "fp32"
        Data type for descriptor calculations.
    readout_dtype : Literal["fp32", "fp64"], default = "fp32"
        Data type for readout calculations.
    scale_shift_dtype : Literal["fp32", "fp64"], default = "fp64"
        Data type for scale and shift parameters.
    """

    basis: BasisConfig = Field(BesselBasisConfig(name="bessel"), discriminator="name")
    radial_transform: Literal["identity", "covalent"] = "identity"

    nn: List[PositiveInt] = [256, 256]
    w_init: Literal["normal", "lecun"] = "lecun"
    b_init: Literal["normal", "zeros"] = "zeros"
    activation_fn: str = "variance_preserving_swish"
    use_ntk: bool = False
    use_bias: bool = False
    readout_activation: Literal["identity", "swish", "elu"] = "elu"

    ensemble: Optional[EnsembleConfig] = None

    property_heads: list[PropertyHead] = []

    # corrections
    empirical_corrections: list[EmpiricalCorrection] = []

    calc_stress: bool = False
    calc_hessian: bool = False

    descriptor_dtype: Literal["fp32", "fp64"] = "fp32"
    readout_dtype: Literal["fp32", "fp64"] = "fp32"
    scale_shift_dtype: Literal["fp32", "fp64"] = "fp64"


def _drop_removed(values, key, off_value, what):
    """Saved configs of older models still carry removed fields; accept them only if the feature was off."""
    if isinstance(values, dict) and key in values:
        if values[key] not in (off_value, None):
            raise ValueError(
                f"`{key}` was removed ({what}); this config enables it and cannot be loaded."
            )
        values = {k: v for k, v in values.items() if k != key}
    return values


class GMNNConfig(BaseModelConfig, extra="forbid"):
    """
    Configuration for the model.

    Parameters
    ----------
    n_radial : PositiveInt, default = 12
        Number of contracted radial channels (used by the l=0 moment).
    n_radial_tensor : Optional[PositiveInt], default = 4
        Radial channels for the l>0 moments, a learned linear combination of the
        `n_radial` channels (applied per pair). None uses all `n_radial` channels.
    n_contr : int, default = 8
        How many gaussian moment contractions to use.
    emb_init : Optional[str], default = "uniform"
        Initialization of the dense species-pair table (`radial_function: dense`).
    radial_function : Literal["dense", "factorized"], default = "factorized"
        How the radial coefficients depend on the element pair. `"dense"` learns an
        independent table per element pair. `"factorized"` uses a rank-`radial_rank`
        CP factorization over per-element embeddings, so pairs share radial shapes
        (much better on unseen or rare element pairs; can underfit on data-rich
        sets with few elements).
    radial_rank : PositiveInt, default = 8
        Rank of the factorized radial function.
    radial_emb_jitter : NonNegativeFloat, default = 0.1
        Spread of the factorized embeddings at init around 1, i.e. how far apart
        the element pairs' radial functions start.
    """

    name: Literal["gmnn"] = "gmnn"

    n_radial: PositiveInt = 12
    n_radial_tensor: Optional[PositiveInt] = 4
    n_contr: int = 8
    emb_init: Optional[str] = "uniform"
    radial_function: Literal["dense", "factorized"] = "factorized"
    radial_rank: PositiveInt = 8
    radial_emb_jitter: NonNegativeFloat = 0.1

    @model_validator(mode="before")
    @classmethod
    def _removed_fields(cls, values):
        values = _drop_removed(
            values, "radial_residual", False, "the per-pair radial residual did nothing"
        )
        values = _drop_removed(
            values, "radial_factor_mode", "cp", "only the CP factorization is kept"
        )
        if (
            isinstance(values, dict)
            and "radial_rank" in values
            and values["radial_rank"] is None
        ):
            # saved before `radial_function` existed: no rank meant the dense table
            values = {k: v for k, v in values.items() if k != "radial_rank"}
            values.setdefault("radial_function", "dense")
        elif isinstance(values, dict) and "radial_rank" in values:
            values.setdefault("radial_function", "factorized")
        return values

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
