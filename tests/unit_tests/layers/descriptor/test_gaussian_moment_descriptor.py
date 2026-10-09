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


def test_radial_compression():
    dR = jax.random.normal(jax.random.PRNGKey(1), (6, 3))
    Z = jnp.array([8, 1, 1])
    neighbor = jnp.array([[1, 2, 0, 2, 0, 1], [0, 0, 1, 1, 2, 2]])

    # identity-initialised compression to the full width is the plain descriptor
    full = GaussianMomentDescriptor(n_radial_tensor=5)
    params = full.init(jax.random.PRNGKey(0), dR, Z, neighbor)
    plain_params = {"params": {"radial_fn": params["params"]["radial_fn"]}}
    np.testing.assert_allclose(
        full.apply(params, dR, Z, neighbor),
        GaussianMomentDescriptor().apply(plain_params, dR, Z, neighbor),
        rtol=1e-6,
    )

    # 5 scalar channels, 3 for all l>0 moments: 5 + 3*6 + 10 + 18 + 27 + 18 features
    small = GaussianMomentDescriptor(n_radial_tensor=3)
    params = small.init(jax.random.PRNGKey(0), dR, Z, neighbor)
    assert small.apply(params, dR, Z, neighbor).shape == (3, 96)
