import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List

from apax.nn.impl.activation import swish
from apax.nn.torch.layers.ntk_linear import NTKLinear


class AtomisticReadoutT(nn.Module):
    def __init__(
        self, units: List[int] = [512, 512], activation_fn: Callable = swish
    ) -> None:
        super().__init__()

        units = [u for u in self.units] + [1]
        dense = []
        for ii, n_hidden in enumerate(units):
            dense.append(NTKLinear(n_hidden))
            if ii < len(units) - 1:
                dense.append(activation_fn)
        self.sequential = nn.Sequential(dense)

    def forward(self, x):
        h = self.sequential(x)
        return h
