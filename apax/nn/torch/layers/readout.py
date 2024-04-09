import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List

from apax.nn.torch.layers.activation import SwishT
from apax.nn.torch.layers.ntk_linear import NTKLinear


class AtomisticReadoutT(nn.Module):
    def __init__(
        self, units: List[int] = [512, 512], activation_fn: Callable = SwishT
    ) -> None:
        super().__init__()

        units = [360] + [u for u in units] + [1]
        dense = []
        for ii in range(len(units)-1):
            units_in, units_out = units[ii], units[ii+1]
            dense.append(NTKLinear(units_in, units_out))
            if ii < len(units) - 2:
                dense.append(activation_fn())
        self.sequential = nn.Sequential(*dense)

    def forward(self, x):
        h = self.sequential(x)
        return h
