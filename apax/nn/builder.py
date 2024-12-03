import logging

import numpy as np

from apax.config import ModelConfig
from apax.layers.descriptor import (
    EquivMPRepresentation,
    GaussianMomentDescriptor,
    So3kratesRepresentation,
)
from apax.layers.descriptor.basis_functions import (
    BesselBasis,
    GaussianBasis,
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

    def build_radial_function(self):
        basis_fn = self.build_basis_function()

        if self.config["basis"]["name"] == "gaussian":
            use_embed_norm = True
            one_sided_dist = False
        else:
            use_embed_norm = False
            one_sided_dist = True

        radial_fn = RadialFunction(
            n_radial=self.config["n_radial"],
            basis_fn=basis_fn,
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

    def build_readout(self, head_config, is_feature_fn=False):
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

        readout = AtomisticReadout(
            units=head_config["nn"],
            b_init=head_config["b_init"],
            w_init=head_config["w_init"],
            use_ntk=head_config["use_ntk"],
            is_feature_fn=is_feature_fn,
            n_shallow_ensemble=n_shallow_ensemble,
            dtype=dtype,
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
    ):
        energy_model = self.build_energy_model(
            scale,
            shift,
            apply_mask,
            init_box=init_box,
            inference_disp_fn=inference_disp_fn,
        )
        if (
            self.config["ensemble"]
            and self.config["ensemble"]["kind"] == "shallow"
            and self.config["ensemble"]["n_members"] > 1
        ):
            log.info("Building ShallowEnsemble model")
            model = ShallowEnsembleModel(
                energy_model,
                calc_stress=self.config["calc_stress"],
                force_variance=self.config["ensemble"]["force_variance"],
                chunk_size=self.config["ensemble"]["chunk_size"],
            )
        else:
            log.info("Building Standard model")
            model = EnergyDerivativeModel(
                energy_model,
                calc_stress=self.config["calc_stress"],
            )
        return model

    def build_ll_feature_model(
        self,
        apply_mask=True,
        init_box: np.array = np.array([0.0, 0.0, 0.0]),
        inference_disp_fn=None,
    ):
        log.info("Building feature model")
        descriptor = self.build_descriptor(apply_mask)
        readout = self.build_readout(is_feature_fn=True)

        model = FeatureModel(
            descriptor,
            readout,
            should_average=True,
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
