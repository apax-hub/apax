import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from apax.model import get_training_model
from apax.model.apax import AtomisticModel, EnergyForceModel, EnergyModel, NeighborSpoof


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

    displacement_fn, _ = space.free()
    shift = jnp.array([0.0, 100.0, 200.0])
    scale = jnp.array([1.0, 1.2, 1.6])[..., None]

    apax = get_training_model(
        3,
        3,
        displacement_fn,
        [512, 512],
        b_init="zeros",
        elemental_energies_mean=shift,
        elemental_energies_std=scale,
    )
    apax_padded = get_training_model(
        4,
        3,
        displacement_fn,
        [512, 512],
        b_init="zeros",
        elemental_energies_mean=shift,
        elemental_energies_std=scale,
    )

    rng_key = jax.random.PRNGKey(1)

    params = apax.init(rng_key, R, Z, idx, box)

    results = apax.apply(params, R, Z, idx, box)
    results_padded = apax_padded.apply(params, R_padded, Z_padded, idx_padded, box)

    assert (results["energy"] - results_padded["energy"]) < 1e-6
    assert np.allclose(results["forces"], results_padded["forces"][:-1, :])


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
