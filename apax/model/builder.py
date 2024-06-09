import numpy as np

from apax.config import ModelConfig
from apax.layers.descriptor.basis_functions import (
    BesselBasis,
    GaussianBasis,
    RadialFunction,
)
from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor
from apax.layers.empirical import ZBLRepulsion
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.model.gmnn import AtomisticModel, EnergyDerivativeModel, EnergyModel


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
            raise NotImplementedError("unknown basis requested")
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
        radial_fn = self.build_radial_function()
        descriptor = GaussianMomentDescriptor(
            radial_fn=radial_fn,
            n_contr=self.config["n_contr"],
            dtype=self.config["descriptor_dtype"],
            apply_mask=apply_mask,
        )
        return descriptor

    def build_readout(self):
        readout = AtomisticReadout(
            units=self.config["nn"],
            b_init=self.config["b_init"],
            dtype=self.config["readout_dtype"],
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

    def build_atomistic_model(
        self,
        scale,
        shift,
        apply_mask,
    ):
        descriptor = self.build_descriptor(apply_mask)
        readout = self.build_readout()
        scale_shift = self.build_scale_shift(scale, shift)

        atomistic_model = AtomisticModel(descriptor, readout, scale_shift)
        return atomistic_model

    def build_energy_model(
        self,
        scale=1.0,
        shift=0.0,
        apply_mask=True,
        init_box: np.array = np.array([0.0, 0.0, 0.0]),
        inference_disp_fn=None,
    ):
        atomistic_model = self.build_atomistic_model(
            scale,
            shift,
            apply_mask,
        )
        corrections = []
        if self.config["use_zbl"]:
            repulsion = ZBLRepulsion(
                apply_mask=apply_mask,
                r_max=self.config["basis"]["r_max"],
            )
            corrections.append(repulsion)

        model = EnergyModel(
            atomistic_model,
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

        model = EnergyDerivativeModel(
            energy_model,
            calc_stress=self.config["calc_stress"],
        )
        return model
