import jax.numpy as jnp
import pytest

from apax.layers.empirical import LatentEwald


@pytest.fixture
def setup_data():
    n_atoms = 4

    R = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    dr_vec = None
    Z = jnp.array([1, 1, 1, 1])
    idx = None
    box = jnp.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    properties = {
        "charge": jnp.array([1.0, -1.0, 1.0, -1.0])[:, None]
    }  # Charges on atoms

    return R, dr_vec, Z, idx, box, properties


def test_latent_ewald(setup_data):
    R, dr_vec, Z, idx, box, properties = setup_data

    latent_ewald = LatentEwald(apply_mask=True, sigma=1.0, kgrid=[2, 2, 2])

    # Check for KeyError if "charge" property is missing
    with pytest.raises(KeyError, match="property 'charge' not found"):
        latent_ewald(R, dr_vec, Z, idx, box, {})

    energy = latent_ewald(R, dr_vec, Z, idx, box, properties)

    # Validate the energy output
    assert isinstance(energy, jnp.ndarray), "Output should be a JAX array"
    assert energy.ndim == 0, "Energy should be a scalar value"

    # Test with a larger kgrid
    latent_ewald = LatentEwald(apply_mask=False, sigma=1.5, kgrid=[4, 4, 4])
    energy = latent_ewald(R, dr_vec, Z, idx, box, properties)
    assert isinstance(energy, jnp.ndarray)
