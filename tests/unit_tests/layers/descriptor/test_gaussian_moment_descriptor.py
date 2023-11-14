import jax
import jax.numpy as jnp
import numpy as np

from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor


def test_gaussian_moment_descriptor():
    dR = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    Z = jnp.array([8, 1, 1])
    neighbor = jnp.array([[1, 2, 0, 2, 0, 1], [0, 0, 1, 1, 2, 2]])

    descriptor = GaussianMomentDescriptor()

    key = jax.random.PRNGKey(0)
    params = descriptor.init(key, dR, Z, neighbor)
    result = descriptor.apply(params, dR, Z, neighbor)
    result_jit = jax.jit(descriptor.apply)(params, dR, Z, neighbor)

    assert result.shape == (3, 360)
    assert result_jit.shape == (3, 360)
