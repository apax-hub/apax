import jax
import jax.numpy as jnp
import numpy as np

from apax.layers.descriptor import GaussianMomentDescriptor
from apax.layers.scaling import PerElementScaleShift
from apax.model.gmnn import AtomisticModel, EnergyForceModel, EnergyModel, NeighborSpoof


def test_apax_variable_size():
    R = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    Z = np.array([1, 2, 2])

    idx = np.array(
        [
            [1, 2, 0, 2, 0, 1],
            [0, 0, 1, 1, 2, 2],
        ]
    )

    box = np.array([0, 0, 0])

    R_padded = np.concatenate([R, np.zeros((1, 3))], axis=0)
    Z_padded = np.concatenate([Z, [0]])
    idx_padded = np.concatenate([idx, [[0], [0]]], axis=1)

    shift = jnp.array([0.0, 100.0, 200.0])
    scale = jnp.array([1.0, 1.2, 1.6])[..., None]

    model = EnergyForceModel(
        AtomisticModel(
            descriptor=GaussianMomentDescriptor(apply_mask=False),
            scale_shift=PerElementScaleShift(scale=scale, shift=shift),
            mask_atoms=False,
        )
    )
    model_padded = EnergyForceModel(
        AtomisticModel(
            descriptor=GaussianMomentDescriptor(apply_mask=True),
            scale_shift=PerElementScaleShift(scale=scale, shift=shift),
            mask_atoms=True,
        )
    )

    rng_key = jax.random.PRNGKey(1)

    params = model.init(rng_key, R, Z, idx, box)

    results = model.apply(params, R, Z, idx, box)
    results_padded = model_padded.apply(params, R_padded, Z_padded, idx_padded, box)

    print(results["forces"])
    print(results_padded["forces"])

    assert (results["energy"] - results_padded["energy"]) < 1e-6
    assert np.all(results["forces"] - results_padded["forces"][:-1, :] < 1e-6)  # 1e-6


def test_atomistic_model():
    key = jax.random.PRNGKey(0)

    R = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    Z = np.array([1, 2, 2])

    idx = np.array(
        [
            [1, 2, 0, 2, 0, 1],
            [0, 0, 1, 1, 2, 2],
        ]
    )
    neighbor = NeighborSpoof(idx=idx)

    box = np.array([0.0, 0.0, 0.0])

    model = AtomisticModel(mask_atoms=False)

    params = model.init(key, R, Z, neighbor, box)
    result = model.apply(params, R, Z, neighbor, box)

    assert result.shape == (3, 1)


def test_energy_model():
    key = jax.random.PRNGKey(0)

    R = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    Z = np.array([1, 2, 2])

    idx = np.array(
        [
            [1, 2, 0, 2, 0, 1],
            [0, 0, 1, 1, 2, 2],
        ]
    )
    neighbor = NeighborSpoof(idx=idx)

    box = np.array([0.0, 0.0, 0.0])

    model = EnergyModel()

    params = model.init(key, R, Z, neighbor, box)
    result = model.apply(params, R, Z, neighbor, box)

    assert result.shape == ()


def test_energy_force_model():
    key = jax.random.PRNGKey(0)

    R = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    Z = np.array([1, 2, 2])

    idx = np.array(
        [
            [1, 2, 0, 2, 0, 1],
            [0, 0, 1, 1, 2, 2],
        ]
    )

    box = np.array([0.0, 0.0, 0.0])

    model = EnergyForceModel()

    params = model.init(key, R, Z, idx, box)
    result = model.apply(params, R, Z, idx, box)

    assert result["energy"].shape == ()
    assert result["forces"].shape == (3, 3)
