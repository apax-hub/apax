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


def test_i_torch_gaussian_basis():
    linj = GaussianBasis(16)

    inputj = jnp.array(np.random.randn(8))

    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, inputj)

    inputt = torch.from_numpy(np.asarray(inputj, dtype=np.float32))
    lint = GaussianBasisT(params=params["params"])

    outj = linj.apply(params, inputj)
    outt = lint(inputt)

    assert np.allclose(outj, outt)
    assert outj.dtype == outt.dtype


def test_i_torch_radial_basis():
    linj = RadialFunction(16)

    inputj = jnp.array(np.random.randn(8))

    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, inputj)

    inputt = torch.from_numpy(np.asarray(inputj, dtype=np.float32))
    lint = RadialFunctionT(params=params["params"])

    outj = linj.apply(params, inputj)
    outt = lint(inputt)

    assert np.allclose(outj, outt)
    assert outj.dtype == outt.dtype



def test_i_torch_ntk_linear():
    linj = NTKLinear(16)

    inputj = jnp.array(np.random.randn(8))

    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, inputj)

    inputt = torch.from_numpy(np.asarray(inputj, dtype=np.float32))
    lint = NTKLinearT(params=params["params"])

    outj = linj.apply(params, inputj)
    outt = lint(inputt)

    assert np.allclose(outj, outt)
    assert outj.dtype == outt.dtype


def test_i_torch_readout():

    linj = AtomisticReadout(16)

    inputj = jnp.array(np.random.randn(8))


    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, inputj)

    inputt = torch.from_numpy(np.asarray(inputj, dtype=np.float32))
    lint = AtomisticReadoutT(params=params["params"])

    outj = linj.apply(params, inputj)
    outt = lint(inputt)

    assert np.allclose(outj, outt)
    assert outj.dtype == outt.dtype


def test_i_torch_scaling():

    linj = PerElementScaleShift(16)

    inputj = jnp.array(np.random.randn(8))

    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, inputj)

    inputt = torch.from_numpy(np.asarray(inputj, dtype=np.float32))
    lint = PerElementScaleShiftT(params=params["params"])

    outj = linj.apply(params, inputj)
    outt = lint(inputt)

    assert np.allclose(outj, outt)
    assert outj.dtype == outt.dtype

