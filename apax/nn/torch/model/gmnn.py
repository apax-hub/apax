from typing import Any, List, Optional
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from apax.nn.torch.layers.descriptor import GaussianMomentDescriptorT
from apax.nn.torch.layers.readout import AtomisticReadoutT
from apax.nn.torch.layers.scaling import PerElementScaleShiftT


class AtomisticModelT(nn.Module):
    def __init__(
        self,
        descriptor: nn.Module = GaussianMomentDescriptorT(),
        readout: nn.Module = AtomisticReadoutT(),
        scale_shift: nn.Module = PerElementScaleShiftT(),
        params = None
    ):
        super().__init__()

        if params:
            self.descriptor = GaussianMomentDescriptorT(params=params["descriptor"])
            readout = AtomisticReadoutT(params_list=params["readout"])

            def readout_fn(x):
                return readout(x)

            self.readout = readout#torch.vmap(readout_fn)
            self.scale_shift = PerElementScaleShiftT(params=params["scale_shift"])
        else:
            self.descriptor = descriptor
            self.readout = readout
            self.scale_shift = scale_shift

    def forward(
        self,
        dr_vec: torch.Tensor,
        Z: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        gm = self.descriptor(dr_vec, Z, idx)
        
        h = []
        for g in gm:
            hi = self.readout(g)
            h.append(hi)
        h = torch.concat(h, dim=0)
        output = self.scale_shift(h, Z)
        return output


def free_displacement(Ri, Rj):
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


class EnergyModelT(nn.Module):
    def __init__(
        self,
        atomistic_model: AtomisticModelT = AtomisticModelT(),
        # corrections: list[EmpiricalEnergyTerm] = field(default_factory=lambda: []),
        params = None,
        init_box: np.array = np.array([0.0, 0.0, 0.0]),
        inference_disp_fn: Any = None,
    ):
        super().__init__()
        if params:
            self.atomistic_model = AtomisticModelT(params=params["atomistic_model"])
        else:
            self.atomistic_model = atomistic_model
        # self.corrections = corrections
        self.init_box = torch.tensor(init_box)
        self.inference_disp_fn = inference_disp_fn

        self.displacement = get_displacement(init_box, inference_disp_fn)

    def forward(
        self,
        R: torch.Tensor,
        Z: torch.Tensor,
        idx: torch.Tensor,
        box,
        offsets,
        # perturbation=None,
    ):
        # Distances
        idx_i, idx_j = idx[0], idx[1]

        # R shape n_atoms x 3
        R = R.type(torch.float64)
        Ri = R[idx_i]
        Rj = R[idx_j]

        # dr_vec shape: neighbors x 3
        dr_vec = self.displacement(Rj, Ri)
        # uncomment once pbc are implemented
        # if torch.all(self.init_box < 1e-6):
        #     dr_vec = self.displacement(Rj, Ri)
        # else:
        #     dr_vec = self.displacement(Rj, Ri, perturbation, box)
        #     dr_vec += offsets

        # Model Core
        atomic_energies = self.atomistic_model(dr_vec, Z, idx)
        total_energy = torch.sum(atomic_energies, dtype=torch.float64)

        # Corrections
        # for correction in self.corrections:
        #     energy_correction = correction(dr_vec, Z, idx)
        #     total_energy = total_energy + energy_correction

        return total_energy


class EnergyDerivativeModelT(nn.Module):
    def __init__(
        self,
        energy_model: EnergyModelT = EnergyModelT(),
        calc_stress: bool = False,
        params=None,
    ):
        super().__init__()

        if params:
            self.energy_model = EnergyModelT(params=params["energy_model"])
        else:
            self.energy_model = energy_model
        self.calc_stress = False  # calc_stress

    def forward(
        self,
        R: torch.Tensor,
        Z: torch.Tensor,
        neighbor: torch.Tensor,
        box: torch.Tensor,
        offsets: torch.Tensor,
    ):
        R.requires_grad_(True)
        requires_grad = [R]

        energy = self.energy_model(R, Z, neighbor, box, offsets)
        grad_outputs : List[Optional[torch.Tensor]] = [torch.ones_like(energy)]

        forces = autograd.grad(
            [energy],
            requires_grad,
            grad_outputs=grad_outputs,
            create_graph=True,
        )[0]
        assert forces is not None
        forces = - forces

        prediction = {"energy": energy, "forces": forces}

        return prediction
