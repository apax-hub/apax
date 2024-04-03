import torch
import torch.nn as nn
import torch.nn.functional as F


class NTKLinear(nn.Module):
    def __init__(self, units) -> None:
        super().__init__()

        self.bias_factor = 0.1
        # self.weight_factor = torch.sqrt(1.0 / dim_in)

        self.w = nn.Parameter()
        self.b = nn.Parameter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_factor = torch.sqrt(1.0 / x.shape[0])
        out = F.linear(x, weight_factor * self.w, self.bias_factor * self.b)
        return out
