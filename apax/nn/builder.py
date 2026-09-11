import logging
from pathlib import Path

import numpy as np

# bundled triple-exponential repulsion coefficients
DEFAULT_NLH_COEFFS = Path(__file__).parent.parent / "data" / "nlh_coeffs.dat"


def load_nlh_coeffs(path, n_species: int | None = None):
    """Load triple-exponential repulsion coeffs into (n_species, n_species, 3)
    arrays a, b, symmetric in the two atomic numbers. Sized to the data
    (max Z + 1) unless n_species is given. File rows: Z1 Z2 a1 b1 a2 b2 a3 b3."""
    d = np.loadtxt(path, usecols=range(8))
    z1, z2 = d[:, 0].astype(int), d[:, 1].astype(int)
    if n_species is None:
        n_species = int(max(z1.max(), z2.max())) + 1  # 93 for the bundled Z<=92 set
    a = np.zeros((n_species, n_species, 3))
    b = np.zeros((n_species, n_species, 3))
    a[z1, z2] = a[z2, z1] = d[:, [2, 4, 6]]
    b[z1, z2] = b[z2, z1] = d[:, [3, 5, 7]]
    return a, b


from apax.config import ModelConfig
from apax.layers.activation import get_activation_fn
from apax.layers.descriptor import (
    EquivMPRepresentation,
    GaussianMomentDescriptor,
    So3kratesRepresentation,
)
from apax.layers.descriptor.basis_functions import (
    BesselBasis,
    CovalentRadialTransform,
    GaussianBasis,
    IdentityRadialTransform,
    RadialFunction,
)
from apax.layers.empirical import all_corrections
from apax.layers.properties import PropertyHead
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.nn.models import (
    EnergyDerivativeModel,
    EnergyModel,
    FeatureModel,
    ShallowEnsembleModel,
)

log = logging.getLogger(__name__)


class ModelBuilder:
    def __init__(self, model_config: ModelConfig, n_species: int = 119):
        self.config = model_config
        self.n_species = n_species

    def build_basis_function(self):
        basis_config = self.config["basis"]
        name = basis_config["name"]

        if name == "gaussian":
            basis_fn = GaussianBasis(
                n_basis=basis_config["n_basis"],
                r_min=basis_config["r_min"],
                r_max=basis_config["r_max"],
                dtype=self.config["descriptor_dtype"],
                spacing=basis_config["spacing"],
            )
        elif name == "bessel":
            basis_fn = BesselBasis(
                n_basis=basis_config["n_basis"],
                r_max=basis_config["r_max"],
                dtype=self.config["descriptor_dtype"],
            )
        else:
            raise ValueError("unknown basis requested")
        return basis_fn

    def build_radial_transform(self):
        name = self.config.get("radial_transform", "identity")
        if name == "identity":
            return IdentityRadialTransform()
        elif name == "covalent":
            return CovalentRadialTransform(dtype=self.config["descriptor_dtype"])
        else:
            raise ValueError(f"unknown radial transform requested: {name}")

    def build_radial_function(self):
        basis_fn = self.build_basis_function()
        radial_transform = self.build_radial_transform()

        if self.config["basis"]["name"] == "gaussian":
            use_embed_norm = True
            one_sided_dist = False
        else:
            use_embed_norm = False
            one_sided_dist = True

        radial_fn = RadialFunction(
            n_radial=self.config["n_radial"],
            basis_fn=basis_fn,
            radial_transform=radial_transform,
            n_species=self.n_species,
            emb_init=self.config["emb_init"],
            use_embed_norm=use_embed_norm,
            one_sided_dist=one_sided_dist,
            dtype=self.config["descriptor_dtype"],
        )
        return radial_fn

    def build_descriptor(
        self,
        apply_mask,
    ):
        raise NotImplementedError("use a subclass to facilitate this")

    def build_readout(
        self, head_config, is_feature_fn=False, only_use_n_layers: None | int = None
    ):
        has_ensemble = "ensemble" in head_config.keys() and head_config["ensemble"]
        if has_ensemble and head_config["ensemble"]["kind"] == "shallow":
            n_shallow_ensemble = head_config["ensemble"]["n_members"]
        elif "n_shallow_members" in head_config.keys():
            n_shallow_ensemble = head_config["n_shallow_members"]
        else:
            n_shallow_ensemble = 0

        if "readout_dtype" in head_config:
            dtype = head_config["readout_dtype"]
        elif "dtype" in head_config:
            dtype = head_config["dtype"]
        else:
            raise KeyError("No dtype specified in config")

        nn_layers = head_config["nn"]
        if only_use_n_layers is not None:
            nn_layers = nn_layers[:only_use_n_layers]
            if len(nn_layers) == 0:
                return None

        activation_fn = get_activation_fn(self.config["activation_fn"])

        readout = AtomisticReadout(
            units=nn_layers,
            b_init=head_config["b_init"],
            w_init=head_config["w_init"],
            use_ntk=head_config["use_ntk"],
            use_bias=head_config.get("use_bias", False),
            output_activation=head_config.get("readout_activation", "swish"),
            is_feature_fn=is_feature_fn,
            n_shallow_ensemble=n_shallow_ensemble,
            dtype=dtype,
            activation_fn=activation_fn,
        )
        return readout

    def build_scale_shift(self, scale, shift):
        scale_shift = PerElementScaleShift(
            n_species=self.n_species,
            scale=scale,
            shift=shift,
            dtype=self.config["scale_shift_dtype"],
        )
        return scale_shift

    def build_property_heads(self, apply_mask: bool = True):
        property_heads = []
        for head in self.config["property_heads"]:
            readout = self.build_readout(head)
            phead = PropertyHead(
                pname=head["name"],
                aggregation=head["aggregation"],
                mode=head["mode"],
                readout=readout,
                apply_mask=apply_mask,
            )
            property_heads.append(phead)
        return property_heads

    def build_corrections(self, apply_mask: bool = True):
        corrections = []
        for correction in self.config["empirical_corrections"]:
            correction = correction.copy()
            name = correction.pop("name")
            if name == "nlh":
                # dependency injection: load per-pair coeffs into arrays here
                path = correction.pop("coeffs_file", None) or DEFAULT_NLH_COEFFS
                a, b = load_nlh_coeffs(path)
                correction["a"], correction["b"] = a, b
            Correction = all_corrections[name]
            corr = Correction(
                **correction,
                apply_mask=apply_mask,
            )
            corrections.append(corr)

        return corrections

    def build_energy_model(
        self,
        scale=1.0,
        shift=0.0,
        apply_mask=True,
        init_box: np.array = np.array([0.0, 0.0, 0.0]),
        inference_disp_fn=None,
    ):
        log.debug("Building atomistic model")

        descriptor = self.build_descriptor(apply_mask)
        readout = self.build_readout(self.config)
        scale_shift = self.build_scale_shift(scale, shift)

        property_heads = self.build_property_heads(apply_mask=apply_mask)
        corrections = self.build_corrections(apply_mask=apply_mask)

        model = EnergyModel(
            representation=descriptor,
            readout=readout,
            scale_shift=scale_shift,
            property_heads=property_heads,
            corrections=corrections,
            init_box=init_box,
            inference_disp_fn=inference_disp_fn,
        )
        return model

    def build_energy_derivative_model(
        self,
        scale=1.0,
        shift=0.0,
        apply_mask=True,
        init_box: np.array = np.array([0.0, 0.0, 0.0]),
        inference_disp_fn=None,
        calc_stress: bool | None = None,
        calc_hessian: bool | None = None,
        force_variance: bool | None = None,
    ):
        energy_model = self.build_energy_model(
            scale,
            shift,
            apply_mask,
            init_box=init_box,
            inference_disp_fn=inference_disp_fn,
        )

        if calc_stress is None:
            calc_stress = self.config["calc_stress"]
        if calc_hessian is None:
            calc_hessian = self.config["calc_hessian"]

        if (
            self.config["ensemble"]
            and self.config["ensemble"]["kind"] == "shallow"
            and self.config["ensemble"]["n_members"] > 1
        ):
            if force_variance is None:
                force_variance = self.config["ensemble"]["force_variance"]

            log.info("Building ShallowEnsemble model")
            model = ShallowEnsembleModel(
                energy_model,
                calc_stress=calc_stress,
                calc_hessian=calc_hessian,
                force_variance=force_variance,
                chunk_size=self.config["ensemble"]["chunk_size"],
            )
        else:
            log.info("Building Standard model")
            model = EnergyDerivativeModel(
                energy_model,
                calc_stress=calc_stress,
                calc_hessian=calc_hessian,
            )
        return model

    def build_feature_model(
        self,
        only_use_n_layers=None,
        apply_mask=True,
        init_box: np.array = np.array([0.0, 0.0, 0.0]),
        inference_disp_fn=None,
        should_average: bool = True,
    ):
        log.info("Building feature model")
        descriptor = self.build_descriptor(apply_mask)
        readout = self.build_readout(
            self.config, is_feature_fn=True, only_use_n_layers=only_use_n_layers
        )

        model = FeatureModel(
            descriptor,
            readout,
            should_average=should_average,
            init_box=init_box,
            inference_disp_fn=inference_disp_fn,
            mask_atoms=True,
        )
        return model


class GMNNBuilder(ModelBuilder):
    def build_descriptor(
        self,
        apply_mask,
    ):
        radial_fn = self.build_radial_function()
        descriptor = GaussianMomentDescriptor(
            radial_fn=radial_fn,
            n_contr=self.config["n_contr"],
            dtype=self.config["descriptor_dtype"],
            apply_mask=apply_mask,
        )
        return descriptor


class EquivMPBuilder(ModelBuilder):
    def build_descriptor(
        self,
        apply_mask,
    ):
        descriptor = EquivMPRepresentation(
            features=self.config["features"],
            max_degree=self.config["max_degree"],
            num_iterations=self.config["num_iterations"],
            basis_fn=self.build_basis_function(),
            dtype=self.config["descriptor_dtype"],
            apply_mask=apply_mask,
        )
        return descriptor


class So3kratesBuilder(ModelBuilder):
    def build_descriptor(
        self,
        apply_mask,
    ):
        descriptor = So3kratesRepresentation(
            basis_fn=self.build_basis_function(),
            num_layers=self.config["num_layers"],
            max_degree=self.config["max_degree"],
            num_features=self.config["num_features"],
            num_heads=self.config["num_heads"],
            use_layer_norm_1=self.config["use_layer_norm_1"],
            use_layer_norm_2=self.config["use_layer_norm_2"],
            use_layer_norm_final=self.config["use_layer_norm_final"],
            activation=self.config["activation"],
            cutoff_fn=self.config["cutoff_fn"],
            transform_input_features=self.config["transform_input_features"],
            dtype=self.config["descriptor_dtype"],
        )
        return descriptor
