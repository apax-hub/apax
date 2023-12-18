import jax
import jax.numpy as jnp

from apax.bal.feature_maps import FeatureMap


def ensemble_features(feature_fn: FeatureMap) -> FeatureMap:
    """
    Feature map transformation which averages the kernels of a model ensemble.
    """
    ensemble_feature_fn = jax.vmap(feature_fn, (0, None), 0)

    def averaged_feature_fn(params, x):
        g = ensemble_feature_fn(params, x)

        if len(g.shape) != 2:
            # models, features
            raise ValueError(
                "Dimension mismatch for input features. Expected shape (models,"
                f" features), got {g.shape}"
            )

        n_models = g.shape[0]
        # sqrt since the kernel is K = g^T g
        feature_scale_factor = jnp.sqrt(1 / n_models)
        g_ens = feature_scale_factor * jnp.sum(g, axis=0)  # shape: n_features
        return g_ens

    return averaged_feature_fn


def batch_features(feature_fn: FeatureMap) -> FeatureMap:
    """
    Vectorizes a feature map over structures.
    Should be the last transformation applied to a feature map.
    """
    batched_feature_fn = jax.vmap(feature_fn, (None, 0), 0)
    return batched_feature_fn
