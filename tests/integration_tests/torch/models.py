import jax
from apax.layers.descriptor.basis_functions import GaussianBasis, RadialFunction
from apax.layers.ntk_linear import NTKLinear
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.nn.torch.layers.descriptor.basis import GaussianBasisT, RadialFunctionT
from apax.nn.torch.layers.ntk_linear import NTKLinearT
from apax.nn.torch.layers.readout import AtomisticReadoutT
import jax.numpy as jnp
import numpy as np
import torch

from apax.nn.torch.layers.scaling import PerElementScaleShiftT
