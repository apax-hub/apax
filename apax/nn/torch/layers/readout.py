import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List, Optional

from apax.nn.torch.layers.activation import SwishT
from apax.nn.torch.layers.ntk_linear import NTKLinearT


class AtomisticReadoutT(nn.Module):
    def __init__(
        self, units: Optional[List[int]] = [512, 512], params_list = None, activation_fn: Callable = SwishT
    ) -> None:
        super().__init__()
        dense = []
        if params_list:
            param_list = (params_list.values())
            for ii, params in enumerate(param_list):
                dense.append(NTKLinearT(params=params))
                if ii < len(param_list) - 1:
                    dense.append(activation_fn())
        else:
            units = [360] + [u for u in units] + [1]
            for ii in range(len(units) - 1):
                units_in, units_out = units[ii], units[ii + 1]
                dense.append(NTKLinearT(units_in, units_out))
                if ii < len(units) - 2:
                    dense.append(activation_fn())

        self.sequential = nn.Sequential(*dense)

    def forward(self, x):
        h = self.sequential(x)
        return h
