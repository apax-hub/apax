import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NTKLinearT(nn.Module):
    def __init__(self, units_in=None, units_out=None, params=None) -> None:
        super().__init__()

        self.bias_factor = 0.1
        # self.weight_factor = torch.sqrt(1.0 / dim_in)
        if params:
            w = torch.from_numpy(np.array(params["w"]).T)
            b = torch.from_numpy(np.array(params["b"]))
        else:
            w = torch.rand((units_out, units_in))
            b = torch.rand((units_out))

        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)
        self.one = torch.tensor(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_factor = torch.sqrt(self.one / x.size(0))
        out = F.linear(x, weight_factor * self.w, self.bias_factor * self.b)
        return out
