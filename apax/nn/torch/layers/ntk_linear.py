import torch
import torch.nn as nn
import torch.nn.functional as F


class NTKLinearT(nn.Module):
    def __init__(self, units_in, units_out) -> None:
        super().__init__()

        self.bias_factor = 0.1
        # self.weight_factor = torch.sqrt(1.0 / dim_in)

        self.w = nn.Parameter(torch.rand((units_out, units_in)))
        self.b = nn.Parameter(torch.rand((units_out)))
        self.one = torch.tensor(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_factor = torch.sqrt(self.one / x.size(0))
        out = F.linear(x, weight_factor * self.w, self.bias_factor * self.b)
        return out
