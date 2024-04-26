import torch
import torch.nn as nn
import numpy as np

from typing import Any, Union


class PerElementScaleShiftT(nn.Module):
    def __init__(
        self,
        scale: Union[torch.Tensor, float] = 1.0,
        shift: Union[torch.Tensor, float] = 0.0,
        params = None,
        n_species: int = 119,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.n_species = n_species

        if params:

            scale = params["scale_per_element"]
            shift = params["shift_per_element"]
            scale = torch.from_numpy(np.array(scale))
            shift = torch.from_numpy(np.array(shift))
        
        else:
            scale = np.repeat(scale, n_species)
            shift = np.repeat(shift, n_species)

            scale = torch.from_numpy(scale)
            shift = torch.from_numpy(shift)

        self.scale_param = nn.Parameter(scale)
        self.shift_param = nn.Parameter(shift)
        self.dtype = dtype

    def forward(self, x: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        # x shape: n_atoms x 1
        # Z shape: n_atoms
        # scale[Z] shape: n_atoms x 1
        x = x.type(self.dtype)

        out = self.scale_param[Z] * x + self.shift_param[Z]
        return out
