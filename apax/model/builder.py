import numpy as np

from apax.config import ModelConfig
from apax.layers.descriptor.basis_functions import GaussianBasis, RadialFunction
from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor
from apax.layers.empirical import ReaxBonded, ZBLRepulsion
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.model.gmnn import AtomisticModel, EnergyDerivativeModel, EnergyModel


class ModelBuilder:
    def __init__(self, model_config: ModelConfig, n_species):
        self.config = model_config
        self.n_species = n_species

    def build_basis_function(self):
        basis_fn = GaussianBasis(
            n_basis=self.config["n_basis"],
            r_min=self.config["r_min"],
            r_max=self.config["r_max"],
            dtype=self.config["descriptor_dtype"],
        )
        return basis_fn

    def build_radial_function(self):
        basis_fn = self.build_basis_function()
        radial_fn = RadialFunction(
            n_radial=self.config["n_radial"],
            basis_fn=basis_fn,
            n_species=self.n_species,
            emb_init=self.config["emb_init"],
            dtype=self.config["descriptor_dtype"],
        )
        return radial_fn

    def build_descriptor(
        self, displacement_fn, apply_mask, init_box: np.array = np.array([0.0, 0.0, 0.0])
    ):
        radial_fn = self.build_radial_function()
        descriptor = GaussianMomentDescriptor(
            displacement_fn=displacement_fn,
            radial_fn=radial_fn,
            n_contr=self.config["n_contr"],
            dtype=self.config["descriptor_dtype"],
            apply_mask=apply_mask,
            init_box=init_box,
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
        displacement_fn,
        scale,
        shift,
        apply_mask,
        init_box: np.array = np.array([0.0, 0.0, 0.0]),
    ):
        descriptor = self.build_descriptor(displacement_fn, apply_mask, init_box=init_box)
        readout = self.build_readout()
        scale_shift = self.build_scale_shift(scale, shift)

        atomistic_model = AtomisticModel(descriptor, readout, scale_shift)
        return atomistic_model

    def build_energy_model(
        self,
        displacement_fn,
        scale=1.0,
        shift=0.0,
        apply_mask=True,
        init_box: np.array = np.array([0.0, 0.0, 0.0]),
    ):
        atomistic_model = self.build_atomistic_model(
            displacement_fn, scale, shift, apply_mask, init_box=init_box
        )
        repulsion, bonded = None, None
        if self.config["use_zbl"]:
            repulsion = ZBLRepulsion(
                displacement_fn,
                apply_mask=apply_mask,
                r_max=self.config["r_max"],
                init_box=init_box,
            )
        if self.config["use_reax"]:
            bonded = ReaxBonded(
                displacement_fn,
                apply_mask=apply_mask,
                r_max=self.config["r_max"],
                init_box=init_box,
            )
        model = EnergyModel(atomistic_model, repulsion=repulsion, bonded=bonded)
        return model

    def build_energy_derivative_model(
        self,
        displacement_fn,
        scale=1.0,
        shift=0.0,
        apply_mask=True,
        init_box: np.array = np.array([0.0, 0.0, 0.0]),
    ):
        atomistic_model = self.build_atomistic_model(
            displacement_fn, scale, shift, apply_mask, init_box=init_box
        )
        repulsion, bonded = None, None
        if self.config["use_zbl"]:
            repulsion = ZBLRepulsion(
                displacement_fn,
                apply_mask=apply_mask,
                r_max=self.config["r_max"],
                init_box=init_box,
            )
        if self.config["use_reax"]:
            bonded = ReaxBonded(
                displacement_fn,
                apply_mask=apply_mask,
                r_max=self.config["r_max"],
                init_box=init_box,
            )
        model = EnergyDerivativeModel(
            atomistic_model,
            repulsion=repulsion,
            bonded=bonded,
            calc_stress=self.config["calc_stress"],
        )
        return model
