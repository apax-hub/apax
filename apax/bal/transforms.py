import jax
import jax.numpy as jnp


def ensemble_features(feature_fn):
    ensemble_feature_fn = jax.vmap(feature_fn, (0, None), 0)

    def aberaged_feature_fn(params, x):
        g = ensemble_feature_fn(params, x)

        if len(g.shape) != 2:
            # models, features
            raise ValueError(
                "Dimension mismatch for input features. Expected shape (models, data,"
                f" features), got {g.shape}"
            )

        n_models = g.shape[0]
        feature_scale_factor = jnp.sqrt(1 / n_models)
        g_ens = feature_scale_factor * jnp.sum(g, axis=0)  # shape: n_features
        return g_ens

    return aberaged_feature_fn


def batch_features(feature_fn):
    batched_feature_fn = jax.vmap(feature_fn, (None, 0), 0)
    return batched_feature_fn
