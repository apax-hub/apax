import numpy as np

from apax.config import ModelConfig
from apax.nn.torch.layers.descriptor.basis import GaussianBasisT, RadialFunctionT
from apax.nn.torch.layers.descriptor.gaussian_moment_descriptor import (
    GaussianMomentDescriptorT,
)

# from apax.nn.torch.layers.empirical import ZBLRepulsion
from apax.nn.torch.layers.readout import AtomisticReadoutT
from apax.nn.torch.layers.scaling import PerElementScaleShiftT
from apax.nn.torch.model.gmnn import AtomisticModelT, EnergyDerivativeModelT, EnergyModelT


class ModelBuilderT:
    def __init__(self, model_config: ModelConfig, n_species: int = 119):
        self.config = model_config
        self.n_species = n_species

    def build_basis_function(self):
        basis_fn = GaussianBasisT(
            n_basis=self.config["n_basis"],
            r_min=self.config["r_min"],
            r_max=self.config["r_max"],
            dtype=self.config["descriptor_dtype"],
        )
        return basis_fn

    def build_radial_function(self):
        basis_fn = self.build_basis_function()
        radial_fn = RadialFunctionT(
            n_radial=self.config["n_radial"],
            basis_fn=basis_fn,
            n_species=self.n_species,
            emb_init=self.config["emb_init"],
            dtype=self.config["descriptor_dtype"],
        )
        return radial_fn

    def build_descriptor(
        self,
        apply_mask,
    ):
        radial_fn = self.build_radial_function()
        descriptor = GaussianMomentDescriptorT(
            radial_fn=radial_fn,
            n_contr=self.config["n_contr"],
            dtype=self.config["descriptor_dtype"],
            apply_mask=apply_mask,
        )
        return descriptor

    def build_readout(self):
        readout = AtomisticReadoutT(
            units=self.config["nn"],
            b_init=self.config["b_init"],
            dtype=self.config["readout_dtype"],
        )
        return readout

    def build_scale_shift(self, scale, shift):
        scale_shift = PerElementScaleShiftT(
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

        atomistic_model = AtomisticModelT(descriptor, readout, scale_shift)
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
        # if self.config["use_zbl"]:
        #     repulsion = ZBLRepulsion(
        #         apply_mask=apply_mask,
        #         r_max=self.config["r_max"],
        #     )
        #     corrections.append(repulsion)

        model = EnergyModelT(
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
        corrections = []
        if self.config["use_zbl"]:
            repulsion = ZBLRepulsion(
                apply_mask=apply_mask,
                r_max=self.config["r_max"],
            )
            corrections.append(repulsion)

        model = EnergyDerivativeModelT(
            energy_model,
            corrections=corrections,
            calc_stress=self.config["calc_stress"],
        )
        return model
