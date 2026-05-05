import jax
import jax.numpy as jnp
import numpy as np

from apax.nn.models import EnergyDerivativeModel, EnergyModel


def test_hessian_prediction():
    # Use float64 for better precision in Hessian
    R = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    Z = np.array([1, 2, 2])

    idx = np.array(
        [
            [1, 2, 0, 2, 0, 1],
            [0, 0, 1, 1, 2, 2],
        ]
    )
    offsets = jnp.full((6, 3), 0.0, dtype=jnp.float64)
    box = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    energy_model = EnergyModel()
    model = EnergyDerivativeModel(energy_model=energy_model, calc_hessian=True)

    rng_key = jax.random.PRNGKey(1)
    # Initialize with proper R, Z, etc.
    params = model.init(rng_key, R, Z, idx, box, offsets)

    results = model.apply(params, R, Z, idx, box, offsets)

    assert "hessian" in results
    hessian = results["hessian"]
    assert hessian.shape == (3, 3, 3, 3)

    # Numerical Hessian check using jax.hessian
    def energy_wrapper(R_flat):
        R_reshaped = R_flat.reshape(3, 3)
        # Use EnergyModel directly to get the energy
        # Correct way to apply with nested params
        e, _ = energy_model.apply(
            {"params": params["params"]["energy_model"]}, R_reshaped, Z, idx, box, offsets
        )
        return e

    R_flat = R.flatten()
    numerical_hessian = jax.hessian(energy_wrapper)(R_flat)
    numerical_hessian = numerical_hessian.reshape(3, 3, 3, 3)

    assert np.allclose(hessian, numerical_hessian, atol=1e-5)
