import jax
import jax.numpy as jnp
from apax.bal.transforms import batch_features, ensemble_features


def test_ensemble_features():
    key = jax.random.PRNGKey(0)
    params_ensemble = jnp.ones((5, 2))  # 5 models, 2 params each
    x = jax.random.normal(key, (3,))  # 3 features for one structure

    # mock_feature_fn takes params for ONE model and x for ONE structure
    # and returns a feature vector
    def mock_feature_fn(params, x):
        return x * params[0]

    ens_ff = ensemble_features(mock_feature_fn)
    result = ens_ff(params_ensemble, x)

    assert result.shape == (3,)

    g = jnp.stack([x * p[0] for p in params_ensemble])
    expected = jnp.sqrt(1 / 5) * jnp.sum(g, axis=0)
    assert jnp.allclose(result, expected)


def test_batch_features():
    key = jax.random.PRNGKey(0)
    params = jnp.zeros(5)  # dummy
    x_batch = jax.random.normal(key, (4, 3))  # batch, features

    def mock_feature_fn(params, x):
        return x

    batch_ff = batch_features(mock_feature_fn)
    result = batch_ff(params, x_batch)

    assert result.shape == (4, 3)
    assert jnp.allclose(result, x_batch)
