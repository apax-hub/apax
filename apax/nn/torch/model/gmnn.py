from typing import Any
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        if np.all(self.init_box < 1e-6):
            # gas phase training and predicting
            displacement_fn = space.free()[0]
            self.displacement = space.map_bond(displacement_fn)
        elif self.inference_disp_fn is None:
            # for training on periodic systems
            self.displacement = vmap(disp_fn, (0, 0, None, None), 0)
        else:
            mappable_displacement_fn = get_disp_fn(self.inference_disp_fn)
            self.displacement = vmap(mappable_displacement_fn, (0, 0, None, None), 0)

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
        total_energy = fp64_sum(atomic_energies)

        # Corrections
        # for correction in self.corrections:
        #     energy_correction = correction(dr_vec, Z, idx)
        #     total_energy = total_energy + energy_correction

        return total_energy