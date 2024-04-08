from typing import Any
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from apax.nn.torch.layers.descriptor import GaussianMomentDescriptor
from apax.nn.torch.layers.readout import AtomisticReadout
from apax.nn.torch.layers.scaling import PerElementScaleShift


class AtomisticModel(nn.Module):
    def __init__(
        self,
        descriptor: nn.Module = GaussianMomentDescriptor(),
        readout: nn.Module = AtomisticReadout(),
        scale_shift: nn.Module = PerElementScaleShift(),
    ):
        super().__init__()
        self.descriptor = descriptor
        self.readout = torch.vmap(readout)
        self.scale_shift = scale_shift

    def forward(
        self,
        dr_vec: torch.tensor,
        Z: torch.tensor,
        idx: torch.tensor,
    ) -> torch.tensor:
        gm = self.descriptor(dr_vec, Z, idx)
        h = self.readout(gm)
        output = self.scale_shift(h, Z)

        return output


def free_displacement(Ri, Rj, box, perturbation):
    return Ri - Rj


def get_displacement(init_box, inference_disp_fn):
    if np.all(init_box < 1e-6):
        # gas phase training and predicting
        displacement = free_displacement
    # elif inference_disp_fn is None:
    #     # for training on periodic systems
    #     displacement = vmap(disp_fn, (0, 0, None, None), 0)
    # else:
    #     mappable_displacement_fn = get_disp_fn(self.inference_disp_fn)
    #     displacement = vmap(mappable_displacement_fn, (0, 0, None, None), 0)

    return displacement


class EnergyModel(nn.Module):
    def __init__(
        self,
        atomistic_model: AtomisticModel = AtomisticModel(),
        # corrections: list[EmpiricalEnergyTerm] = field(default_factory=lambda: []),
        init_box: np.array = np.array([0.0, 0.0, 0.0]),
        inference_disp_fn: Any = None,
    ):
        super().__init__()
        self.atomistic_model = atomistic_model
        # self.corrections = corrections
        self.init_box = init_box
        self.inference_disp_fn = inference_disp_fn

        self.displacement = get_displacement(init_box, inference_disp_fn)

    def forward(
        self,
        R: torch.Tensor,
        Z: torch.Tensor,
        idx: torch.Tensor,
        box,
        offsets,
        perturbation=None,
    ):
        # Distances
        idx_i, idx_j = idx[0], idx[1]

        # R shape n_atoms x 3
        R = R.type(torch.float64)
        Ri = R[idx_i]
        Rj = R[idx_j]

        # dr_vec shape: neighbors x 3
        if np.all(self.init_box < 1e-6):
            dr_vec = self.displacement(Rj, Ri)
        else:
            dr_vec = self.displacement(Rj, Ri, perturbation, box)
            dr_vec += offsets

        # Model Core
        atomic_energies = self.atomistic_model(dr_vec, Z, idx)
        total_energy = torch.sum(atomic_energies, dtype=torch.float64)

        # Corrections
        # for correction in self.corrections:
        #     energy_correction = correction(dr_vec, Z, idx)
        #     total_energy = total_energy + energy_correction

        return total_energy


class EnergyDerivativeModel(nn.Module):
    def __init__(
        self,
        energy_model: EnergyModel = EnergyModel(),
        calc_stress: bool = False,
    ):
        super().__init__()

        self.energy_model = energy_model
        self.calc_stress = False# calc_stress


    def forward(
        self,
        R: torch.Tensor,
        Z: torch.Tensor,
        neighbor: torch.Tensor,
        box: torch.Tensor,
        offsets: torch.Tensor,
    ):
        R.requires_grad = True
        requires_grad = [R]
        if self.calc_stress:
            eps = torch.zeros((3, 3), torch.float64)
            eps.requires_grad = True
            eps_sym = 0.5 * (eps + eps.T)
            identity = torch.eye(3, dtype=torch.float64)
            perturbation = identity + eps_sym
            requires_grad.append(eps)
        else:
            perturbation = None
        
        energy = self.energy_model(R, Z, neighbor, box, offsets, perturbation)
                    

        grads = autograd.grad(energy, requires_grad,
                            grad_outputs=torch.ones_like(energy),
                            create_graph=True)
        
        neg_forces = grads[0]
        forces = -neg_forces

        prediction = {"energy": energy, "forces": forces}

        if self.calc_stress:
            stress = grads[-1]
            prediction["stress"] = stress

        return prediction
