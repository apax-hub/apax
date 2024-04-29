import jax
import pytest
from apax.data.preprocessing import compute_nl
from apax.layers.descriptor.basis_functions import GaussianBasis, RadialFunction
from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor
from apax.layers.ntk_linear import NTKLinear
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.nn.torch.layers.descriptor.basis import GaussianBasisT, RadialFunctionT
from apax.nn.torch.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptorT
from apax.nn.torch.layers.ntk_linear import NTKLinearT
from apax.nn.torch.layers.readout import AtomisticReadoutT
import jax.numpy as jnp
import numpy as np
import torch

from apax.nn.torch.layers.scaling import PerElementScaleShiftT
from apax.utils.convert import atoms_to_inputs


def test_i_torch_gaussian_basis():
    linj = GaussianBasis()

    inputj = jnp.array(np.random.randn(8), dtype=jnp.float32)

    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, inputj)

    inputt = torch.from_numpy(np.asarray(inputj, dtype=np.float32))
    lint = GaussianBasisT()

    outj = linj.apply(params, inputj)
    outt = lint(inputt)

    outj = np.array(outj, np.float32)
    outt = outt.detach().numpy()

    assert np.allclose(outj, outt)
    assert outj.dtype == outt.dtype


def test_i_torch_radial_basis():
    linj = RadialFunction(16)

    inputj = (np.random.rand(9,), np.random.randint(0,119, (9,)), np.random.randint(0,119, (9,)))

    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, *inputj)

    inputt = (
        torch.from_numpy(np.asarray(inputj[0], dtype=np.float32)),
        torch.from_numpy(np.asarray(inputj[1], dtype=np.int64)),
        torch.from_numpy(np.asarray(inputj[2], dtype=np.int64)),
    )
    lint = RadialFunctionT(params=params["params"])

    outj = linj.apply(params, *inputj)
    outt = lint(*inputt)
    
    outj = np.array(outj, np.float32)
    outt = outt.detach().numpy()

    assert np.allclose(outj, outt)
    assert outj.dtype == outt.dtype


@pytest.mark.parametrize(
    "num_data, pbc, calc_results",
    (
        [1, False, ["energy", "forces"]],
        # [1, True, ["energy", "forces"]],
    ),
)
def test_i_torch_descriptor(example_atoms):
    inputs = atoms_to_inputs(example_atoms)
    R = inputs["positions"][0]
    Z = inputs["numbers"][0]
    box = inputs["box"][0]
    idxs, offsets = compute_nl(R, box, 10.0)

    dr_vec = R[idxs[0]] - R[idxs[1]]

    inputj = dr_vec, Z, idxs
    inputt = (
        torch.from_numpy(np.asarray(dr_vec, dtype=np.float64)),
        torch.from_numpy(np.asarray(Z, dtype=np.int64)),
        torch.from_numpy(np.asarray(idxs, dtype=np.int64)),
    )

    linj = GaussianMomentDescriptor(
        dtype=jnp.float64,
        )
    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, *inputj)

    lint = GaussianMomentDescriptorT(
        params=params["params"],
        dtype=torch.float64,
    )

    outj = linj.apply(params, *inputj)
    outt = lint(*inputt)

    outj = np.array(outj, np.float64)
    outt = outt.detach().numpy()

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

    outj = np.array(outj, np.float32)
    outt = outt.detach().numpy()

    assert np.allclose(outj, outt)
    assert outj.dtype == outt.dtype


def test_i_torch_readout():

    linj = AtomisticReadout([16,16])

    inputj = jnp.array(np.random.randn(8))
    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, inputj)

    inputt = torch.from_numpy(np.asarray(inputj, dtype=np.float32))
    lint = AtomisticReadoutT(params_list=params["params"])

    outj = linj.apply(params, inputj)
    outt = lint(inputt)

    outj = np.array(outj, np.float32)
    outt = outt.detach().numpy()

    assert np.allclose(outj, outt)
    assert outj.dtype == outt.dtype


def test_i_torch_scaling():

    linj = PerElementScaleShift()

    inputj = (np.random.rand(9,), np.random.randint(0,119, (9,)))

    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, *inputj)

    inputt = (
        torch.from_numpy(np.asarray(inputj[0], dtype=np.float32)),
        torch.from_numpy(np.asarray(inputj[1], dtype=np.int64)),
    )
    lint = PerElementScaleShiftT(params=params["params"])

    outj = linj.apply(params, *inputj)
    outt = lint(*inputt)

    outj = np.array(outj, np.float32)
    outt = outt.detach().numpy()

    assert np.allclose(outj, outt)
    assert outj.dtype == outt.dtype

