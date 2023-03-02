import jax
import jax.numpy as jnp

from apax.layers.readout import AtomisticReadout


def test_atomistic_readout():
    key = jax.random.PRNGKey(0)

    x = jnp.linspace(-1, 1, num=10)

    linear = AtomisticReadout([32, 32])
    params = linear.init(key, x)
    result = linear.apply(params, x)

    assert result.shape == (1,)
