import torch
import torch.nn as nn

from typing import Any, Union


class PerElementScaleShiftT(nn.Module):
    def __init__(
        self,
        scale: Union[torch.Tensor, float] = 1.0,
        shift: Union[torch.Tensor, float] = 0.0,
        n_species: int = 119,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.n_species = n_species

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
