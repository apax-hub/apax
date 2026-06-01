from typing import Annotated, List, Literal, Optional, Union

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
    spacing: Literal['linear', 'exponential'], default = 'linear'
        Spacing of centers of Gaussians. `"exponential"` results in more basis
        functions closer to `r_min`, and less at `r_max`.
        See https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181, Figure 2
    """

    name: Literal["gaussian"] = "gaussian"
    n_basis: PositiveInt = 7
    r_min: NonNegativeFloat = 0.5
    r_max: PositiveFloat = 6.0
    spacing: Literal["linear", "exponential"] = "linear"


class BesselBasisConfig(BaseModel, extra="forbid"):
    """
    Bessel basis functions.

    Parameters
    ----------
    variant : Literal["kocer", "standard"], default = "kocer"
        ``kocer`` selects the Kocer 2019 symmetrised form
        (:class:`~apax.layers.descriptor.basis_functions.BesselBasis`).
        ``standard`` selects the textbook spherical-Bessel form used by
        torch-mace (:class:`~apax.layers.descriptor.basis_functions.MaceBesselBasis`).
    n_basis : PositiveInt, default = 16
        Number of uncontracted basis functions.
    r_max : PositiveFloat, default = 5.0
        Cutoff radius of the descriptor.
    """

    name: Literal["bessel"] = "bessel"
    variant: Literal["kocer", "standard"] = "kocer"
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


class ExponentialRepulsion(Correction, extra="forbid"):
    name: Literal["exponential"]
    r_max: NonNegativeFloat = 1.5


class LatentEwald(Correction, extra="forbid"):
    name: Literal["latent_ewald"]
    kgrid: list
    sigma: float = 1.0
    use_property: str = "charges"


class MaceZBLPairRepulsion(Correction, extra="forbid"):
    """Faithful torch-mace ``ZBLBasis`` correction.

    Distinct from :class:`ZBLRepulsion` (apax's existing cosine-cutoff +
    softplus-coefficient flavour). This one matches torch-mace's
    polynomial-cutoff ZBL with per-pair ``r_max`` from ASE covalent radii —
    the format MACE foundation models (MACE-MPA-0, MatPES, OMAT) ship.

    Parameters
    ----------
    p : int, default = 6
        Polynomial-cutoff order.
    trainable : bool, default = False
        If ``True``, ``a_exp`` and ``a_prefactor`` become trainable parameters.
        Foundation models in scope use ``trainable=False``.
    """

    name: Literal["mace_zbl"]
    p: int = 6
    trainable: bool = False
    output_scale: float = 1.0


EmpiricalCorrection = Union[
    ZBLRepulsion,
    ExponentialRepulsion,
    LatentEwald,
    MaceZBLPairRepulsion,
]


class DistanceTransform(BaseModel, extra="forbid"):
    name: str


class AgnesiTransformConfig(DistanceTransform, extra="forbid"):
    """Faithful port of ``mace.modules.radial.AgnesiTransform``.

    Per-edge length transform driven by element-pair covalent radii. Used by
    MACE foundations such as MACE-MPA-0 and MACE-matpes-r2scan-omat-ft, which
    insert this transform between the polynomial cutoff and the Bessel basis
    in the radial embedding.

    Parameters
    ----------
    a : float, default = 1.0805
        Multiplicative coefficient in the Agnesi denominator. Default matches
        the foundation-model value.
    q : float, default = 0.9183
        Numerator exponent of the scaled distance. Default matches the
        foundation-model value.
    p : float, default = 4.5791
        Denominator exponent offset (``q - p`` is the actual exponent in the
        inner term). Default matches the foundation-model value.
    trainable : bool, default = False
        If ``True``, ``a``/``q``/``p`` become trainable parameters. Foundation
        models in scope ship them as non-trainable buffers.
    """

    name: Literal["agnesi"]
    a: float = 1.0805
    q: float = 0.9183
    p: float = 4.5791
    trainable: bool = False


DistanceTransformConfig = Annotated[
    Union[AgnesiTransformConfig], Field(discriminator="name")
]


class PropertyHead(BaseModel, extra="forbid"):
    """
    Configuration for property heads.

    The readout architecture is dictated by the parent model: GMNN / EquivMP /
    So3krates use :class:`apax.layers.readout.AtomisticReadout` (configured by
    ``nn``); MACE uses :class:`apax.layers.readout.MaceReadout` (configured by
    ``MLP_irreps``). The unused field is ignored by the corresponding builder.

    Parameters
    ----------
    name : str
        Name of the property.
    aggregation : str, default = "none"
        Aggregation method for atomic contributions.
    mode : str, default = "l0"
        Rotation order of the property.
    nn : List[PositiveInt], default = [128, 128]
        Hidden layers / units for the AtomisticReadout (non-MACE models).
    n_shallow_members : int, default = 0
        Number of shallow ensemble members for this head.
    MLP_irreps : str, default = "16x0e"
        e3nn irreps string for the MaceReadout's intermediate MLP (MACE only).
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
    MLP_irreps: str = "16x0e"
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
        Configuration for primitive basis functions.
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

    nn: List[PositiveInt] = [256, 256]
    w_init: Literal["normal", "lecun"] = "lecun"
    b_init: Literal["normal", "zeros"] = "zeros"
    activation_fn: str = "variance_preserving_swish"
    use_ntk: bool = False

    ensemble: Optional[EnsembleConfig] = None

    property_heads: list[PropertyHead] = []

    # corrections
    empirical_corrections: list[EmpiricalCorrection] = []

    calc_stress: bool = False
    calc_hessian: bool = False

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


class MaceRadialEmbeddingConfig(BaseModel, extra="forbid"):
    """MACE radial-embedding cutoff envelope + optional length transform.

    Parameters
    ----------
    num_polynomial_cutoff : PositiveInt, default = 5
        Polynomial order of the smooth-cutoff envelope.
    distance_transform : Optional[DistanceTransformConfig], default = None
        Optional per-edge length transform applied between the polynomial
        cutoff and the basis. ``None`` (default) preserves the existing
        ``radial = bessel(r) * cutoff(r)`` pipeline; foundations that ship
        a non-trivial transform (MACE-MPA-0, MACE-matpes-r2scan-omat-ft)
        set this to an :class:`AgnesiTransformConfig`.
    """

    num_polynomial_cutoff: PositiveInt = 5
    distance_transform: Optional[DistanceTransformConfig] = None


class RealAgnosticResidualConfig(BaseModel, extra="forbid"):
    """MACE-MP-0 small/medium baseline interaction-block variant."""

    name: Literal["RealAgnosticResidual"] = "RealAgnosticResidual"


class RealAgnosticDensityConfig(BaseModel, extra="forbid"):
    """Density-normalised, no residual skip (MACE-MPA-0 / MatPES layer 0)."""

    name: Literal["RealAgnosticDensity"] = "RealAgnosticDensity"


class RealAgnosticDensityResidualConfig(BaseModel, extra="forbid"):
    """Density-normalised with parent-style residual skip (MPA-0 / MatPES layer 1)."""

    name: Literal["RealAgnosticDensityResidual"] = "RealAgnosticDensityResidual"


InteractionConfig = Annotated[
    Union[
        RealAgnosticResidualConfig,
        RealAgnosticDensityConfig,
        RealAgnosticDensityResidualConfig,
    ],
    Field(discriminator="name"),
]


class MaceDescriptorConfig(BaseModel, extra="forbid"):
    """MACE message-passing architecture knobs.

    Parameters
    ----------
    max_ell : PositiveInt, default = 3
        Maximum spherical-harmonic degree.
    hidden_irreps : str, default = "128x0e + 128x1o"
        e3nn irreps string for node features. Must include a ``0e`` term.
    correlation : PositiveInt, default = 3
        Symmetric-contraction correlation order.
    interactions : list[InteractionConfig], min_length = 1, default = two RealAgnosticResidual
        Per-layer interaction-block configs. The number of layers is
        ``len(interactions)``; there is no separate ``num_interactions``.
    avg_num_neighbors : PositiveFloat, default = 1.0
        Per-message normaliser used by every interaction block.
    """

    max_ell: PositiveInt = 3
    hidden_irreps: str = "128x0e + 128x1o"
    correlation: PositiveInt = 3
    interactions: list[InteractionConfig] = Field(
        default_factory=lambda: [
            RealAgnosticResidualConfig(),
            RealAgnosticResidualConfig(),
        ],
        min_length=1,
    )
    avg_num_neighbors: PositiveFloat = 1.0


class MaceReadoutConfig(BaseModel, extra="forbid"):
    """MACE energy-head readout configuration.

    Parameters
    ----------
    MLP_irreps : str, default = "16x0e"
        e3nn irreps string for the MaceReadout's intermediate MLP.
    """

    MLP_irreps: str = "16x0e"


class MaceModelConfig(BaseModelConfig, extra="forbid"):
    """Configuration for a MACE model.

    Grouped into four sub-configs mirroring the forward pass:
    ``basis -> radial_embedding -> descriptor -> readout``. The cutoff comes
    from ``model.basis.r_max``, the bessel count from ``model.basis.n_basis``,
    and the layer count from ``len(model.descriptor.interactions)``.

    Parameters
    ----------
    basis : BesselBasisConfig
        Defaults to ``(variant="standard", n_basis=8, r_max=5.0)`` — torch-mace's
        bessel formula plus apax's MACE defaults.
    radial_embedding : MaceRadialEmbeddingConfig
        Cutoff envelope + optional length transform.
    descriptor : MaceDescriptorConfig
        Message-passing architecture knobs.
    readout : MaceReadoutConfig
        Readout block configuration.
    """

    name: Literal["mace"] = "mace"

    basis: BesselBasisConfig = Field(
        default_factory=lambda: BesselBasisConfig(
            variant="standard",
            n_basis=8,
            r_max=5.0,
        ),
        discriminator="name",
    )
    radial_embedding: MaceRadialEmbeddingConfig = Field(
        default_factory=MaceRadialEmbeddingConfig
    )
    descriptor: MaceDescriptorConfig = Field(default_factory=MaceDescriptorConfig)
    readout: MaceReadoutConfig = Field(default_factory=MaceReadoutConfig)

    def get_builder(self):
        from apax.nn.builder import MaceBuilder

        return MaceBuilder


ModelConfig = Union[GMNNConfig, EquivMPConfig, So3kratesConfig, MaceModelConfig]
