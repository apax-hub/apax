import jax
import pytest
from apax.data.preprocessing import compute_nl
from apax.layers.descriptor.basis_functions import GaussianBasis, RadialFunction
from apax.layers.ntk_linear import NTKLinear
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.model.gmnn import AtomisticModel, EnergyDerivativeModel, EnergyModel
from apax.nn.torch.layers.descriptor.basis import GaussianBasisT, RadialFunctionT
from apax.nn.torch.layers.ntk_linear import NTKLinearT
from apax.nn.torch.layers.readout import AtomisticReadoutT
import jax.numpy as jnp
import numpy as np
import torch

from apax.nn.torch.layers.scaling import PerElementScaleShiftT
from apax.nn.torch.model.gmnn import AtomisticModelT, EnergyDerivativeModelT, EnergyModelT
from apax.utils.convert import atoms_to_inputs

@pytest.mark.parametrize(
    "num_data, pbc, calc_results",
    (
        [1, False, ["energy", "forces"]],
        # [1, True, ["energy", "forces"]],
    ),
)
def test_i_torch_atomistic_model(example_atoms):
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

    inputj = dr_vec, Z, idxs
    inputt = (
        torch.from_numpy(np.asarray(dr_vec, dtype=np.float64)),
        torch.from_numpy(np.asarray(Z, dtype=np.int64)),
        torch.from_numpy(np.asarray(idxs, dtype=np.int64)),
    )

    linj = AtomisticModel()

    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, *inputj)
    lint = AtomisticModelT(params=params["params"])

    outj = linj.apply(params, *inputj)
    outt = lint(*inputt)

    outj = np.array(outj)
    outt = outt.detach().numpy()

    assert np.allclose(outj, outt, rtol=0.01)
    assert outj.dtype == outt.dtype



@pytest.mark.parametrize(
    "num_data, pbc, calc_results",
    (
        [1, False, ["energy", "forces"]],
        # [1, True, ["energy", "forces"]],
    ),
)
def test_i_torch_energy_model(example_atoms):
    inputs = atoms_to_inputs(example_atoms)
    R = inputs["positions"][0]
    Z = inputs["numbers"][0]
    box = inputs["box"][0]
    idxs, offsets = compute_nl(R, box, 10.0)

    # dr_vec = R[idxs[0]] - R[idxs[1]]

    inputj = R, Z, idxs, box, offsets
    inputt = (
        torch.from_numpy(np.asarray(R, dtype=np.float64)),
        torch.from_numpy(np.asarray(Z, dtype=np.int64)),
        torch.from_numpy(np.asarray(idxs, dtype=np.int64)),
        torch.from_numpy(np.asarray(box, dtype=np.float64)),
        torch.from_numpy(np.asarray(offsets, dtype=np.float64)),
    )

    linj = EnergyModel()

    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, *inputj)
    lint = EnergyModelT(params=params["params"])

    outj = linj.apply(params, *inputj)
    outt = lint(*inputt)

    outj = np.array(outj)
    outt = outt.detach().numpy()

    assert np.allclose(outj, outt, rtol=0.01)
    assert outj.dtype == outt.dtype


@pytest.mark.parametrize(
    "num_data, pbc, calc_results",
    (
        [1, False, ["energy", "forces"]],
        # [1, True, ["energy", "forces"]],
    ),
)
def test_i_torch_energy_model(example_atoms):
    inputs = atoms_to_inputs(example_atoms)
    R = inputs["positions"][0]
    Z = inputs["numbers"][0]
    box = inputs["box"][0]
    idxs, offsets = compute_nl(R, box, 10.0)

    # dr_vec = R[idxs[0]] - R[idxs[1]]

    inputj = R, Z, idxs, box, offsets
    inputt = (
        torch.from_numpy(np.asarray(R, dtype=np.float64)),
        torch.from_numpy(np.asarray(Z, dtype=np.int64)),
        torch.from_numpy(np.asarray(idxs, dtype=np.int64)),
        torch.from_numpy(np.asarray(box, dtype=np.float64)),
        torch.from_numpy(np.asarray(offsets, dtype=np.float64)),
    )

    linj = EnergyDerivativeModel()

    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, *inputj)
    lint = EnergyDerivativeModelT(params=params["params"])

    outj = linj.apply(params, *inputj)
    outt = lint(*inputt)

    outj = np.array(outj)
    outt = outt.detach().numpy()

    assert np.allclose(outj, outt, rtol=0.01)
    assert outj.dtype == outt.dtype



@pytest.mark.parametrize(
    "num_data, pbc, calc_results",
    (
        [1, False, ["energy", "forces"]],
        # [1, True, ["energy", "forces"]],
    ),
)
def test_i_torch_energy_derivative_model(example_atoms):
    inputs = atoms_to_inputs(example_atoms)
    R = inputs["positions"][0]
    Z = inputs["numbers"][0]
    box = inputs["box"][0]
    idxs, offsets = compute_nl(R, box, 10.0)

    inputj = R, Z, idxs, box, offsets
    inputt = (
        torch.from_numpy(np.asarray(R, dtype=np.float64)),
        torch.from_numpy(np.asarray(Z, dtype=np.int64)),
        torch.from_numpy(np.asarray(idxs, dtype=np.int64)),
        torch.from_numpy(np.asarray(box, dtype=np.float64)),
        torch.from_numpy(np.asarray(offsets, dtype=np.float64)),
    )

    linj = EnergyDerivativeModel()
    rng_key = jax.random.PRNGKey(0)
    params = linj.init(rng_key, *inputj)
    lint = EnergyDerivativeModelT(params=params["params"])
    outj = linj.apply(params, *inputj)
    outt = lint(*inputt)

    energyj = outj["energy"]
    forcesj = outj["forces"]

    energyt = outt["energy"]
    forcest = outt["forces"]

    energyj = np.array(energyj)
    energyt = energyt.detach().numpy()

    forcesj = np.array(forcesj)
    forcest = forcest.detach().numpy()

    assert np.allclose(energyj, energyt, rtol=0.01)
    assert energyj.dtype == energyt.dtype

    assert np.allclose(energyj, energyt, rtol=0.01)
    assert energyj.dtype == energyt.dtype
    
    # test jit script
    traced = torch.jit.script(lint)

    traced.save('torchmodel.pt')
    loaded = torch.jit.load('torchmodel.pt')

    print(loaded(*inputt))
