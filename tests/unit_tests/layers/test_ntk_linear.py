import jax
import jax.numpy as jnp

from apax.layers.ntk_linear import NTKLinear


def test_ntk_linear():
    key = jax.random.PRNGKey(0)

    x = jnp.linspace(-1, 1, num=20)

    linear = NTKLinear(32, dtype=jnp.float32)
    params = linear.init(key, x)
    result = linear.apply(params, x)

    assert params["params"]["w"].shape == (20, 32)
    assert params["params"]["b"].shape == (32,)

    assert result.shape == (32,)
