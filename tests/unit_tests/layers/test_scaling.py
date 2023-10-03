import jax
import jax.numpy as jnp

from apax.layers.scaling import PerElementScaleShift


def test_per_element_scale_shift():
    key = jax.random.PRNGKey(0)

    x = jnp.array([1.0, 1.0, 1.0])[:, None]
    Z = jnp.array([1, 2, 2])
    n_species = 119  # 3

    global_shift = 2.0
    global_scale = 1.0

    scale_shift = PerElementScaleShift(
        n_species=n_species, scale=global_scale, shift=global_shift
    )

    params = scale_shift.init(key, x, Z)
    result = scale_shift.apply(params, x, Z)

    assert params["params"]["scale_per_element"].shape == (n_species, 1)
    assert params["params"]["shift_per_element"].shape == (n_species, 1)

    assert jnp.allclose(result, jnp.array([3.0, 3.0, 3.0])[:, None])

    indiv_scale = jnp.array([10.0, 2.0, 3.0])
    indiv_shift = jnp.array([0.0, -2.0, 2.0])
    scale_shift = PerElementScaleShift(
        n_species=n_species, scale=indiv_scale, shift=indiv_shift
    )

    params = scale_shift.init(key, x, Z)
    result = scale_shift.apply(params, x, Z)

    assert params["params"]["scale_per_element"].shape == (3, 1)
    assert params["params"]["shift_per_element"].shape == (3, 1)

    assert jnp.allclose(result, jnp.array([0.0, 5.0, 5.0])[:, None])
