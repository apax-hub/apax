import einops
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
