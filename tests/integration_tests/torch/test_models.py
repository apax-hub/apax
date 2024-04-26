import jax
from apax.layers.descriptor.basis_functions import GaussianBasis, RadialFunction
from apax.layers.ntk_linear import NTKLinear
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.model.gmnn import AtomisticModel, EnergyModel
from apax.nn.torch.layers.descriptor.basis import GaussianBasisT, RadialFunctionT
from apax.nn.torch.layers.ntk_linear import NTKLinearT
from apax.nn.torch.layers.readout import AtomisticReadoutT
import jax.numpy as jnp
import numpy as np
import torch

from apax.nn.torch.layers.scaling import PerElementScaleShiftT
from apax.nn.torch.model.gmnn import AtomisticModelT, EnergyModelT


def test_i_torch_energy_model():

    linj = AtomisticModel()

    inputj = jnp.array(np.random.randn(8))

    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, inputj)

    inputt = torch.from_numpy(np.asarray(inputj, dtype=np.float32))
    lint = AtomisticModelT(params_list=params["params"])

    outj = linj.apply(params, inputj)
    outt = lint(inputt)

    outj = np.array(outj, np.float32)
    outt = outt.detach().numpy()

    assert np.allclose(outj, outt)
    assert outj.dtype == outt.dtype



# def test_i_torch_energy_model():

#     linj = EnergyModel()

#     inputj = jnp.array(np.random.randn(8))

#     rng_key = jax.random.PRNGKey(0)
#     params = linj.init(rng_key, inputj)

#     inputt = torch.from_numpy(np.asarray(inputj, dtype=np.float32))
#     lint = EnergyModelT(params_list=params["params"])

#     outj = linj.apply(params, inputj)
#     outt = lint(inputt)

#     outj = np.array(outj, np.float32)
#     outt = outt.detach().numpy()

#     assert np.allclose(outj, outt)
#     assert outj.dtype == outt.dtype
