from typing import Literal, Tuple, Union

import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from pydantic import BaseModel, Field


def extract_feature_params(params: dict, layer_name: str) -> Tuple[dict, dict]:
    """Seprate params into those belonging to a selected layer
    and the remaining ones.
    """
    p_flat = flatten_dict(params)

    feature_layer_params = {k: v for k, v in p_flat.items() if layer_name in k}
    remaining_params = {k: v for k, v in p_flat.items() if layer_name not in k}

    if len(feature_layer_params.keys()) > 2:  # w and b
        print(feature_layer_params.keys())
        raise ValueError("Found more than one layer of the specified name")

    return feature_layer_params, remaining_params


class LastLayerGradientFeatures(BaseModel, extra="forbid"):
    """
    Model transfomration which computes the gradient of the output
    wrt. the specified layer.
    https://arxiv.org/pdf/2203.09410
    """

    name: Literal["ll_grad"]
    layer_name: str = "dense_2"

    def apply(self, model):
        def ll_grad(params, inputs):
            ll_params, remaining_params = extract_feature_params(params, self.layer_name)

            def inner(ll_params):
                ll_params.update(remaining_params)
                full_params = unflatten_dict(ll_params)

                # TODO find better abstraction for inputs
                R, Z, idx, box, offsets = (
                    inputs["positions"],
                    inputs["numbers"],
                    inputs["idx"],
                    inputs["box"],
                    inputs["offsets"],
                )
                return model.apply(full_params, R, Z, idx, box, offsets)

            g_ll = jax.grad(inner)(ll_params)
            g_ll = unflatten_dict(g_ll)

            g_flat = jax.tree_map(lambda arr: jnp.reshape(arr, (-1,)), g_ll)
            (gw, gb), _ = jax.tree_util.tree_flatten(g_flat)

            bias_factor = 0.1
            weight_factor = jnp.sqrt(1 / gw.shape[-1])
            g_scaled = [weight_factor * gw, bias_factor * gb]

            g = jnp.concatenate(g_scaled)

            return g

        return ll_grad


class IdentityFeatures(BaseModel, extra="forbid"):
    """Identity feature map. For debugging purposes"""

    name: Literal["identity"]

    def apply(self, model):
        return model.apply


class FeatureMapOptions(BaseModel, extra="forbid"):
    base_feature_map: Union[LastLayerGradientFeatures, IdentityFeatures] = Field(
        LastLayerGradientFeatures(name="ll_grad"), discriminator="name"
    )
