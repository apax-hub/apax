from gmnn_jax.layers.readout import AtomisticReadout
import jax
import jax.numpy as jnp


def test_atomistic_readout():
    key = jax.random.PRNGKey(0)

    x = jnp.linspace(-1,1, num=10)


    linear = AtomisticReadout([32,32])
    params = linear.init(key, x)
    result = linear.apply(params, x)

    assert result.shape == (1,)

test_atomistic_readout()