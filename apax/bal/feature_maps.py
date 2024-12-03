from typing import Callable, Literal, Tuple, Union

import jax
import jax.ad_checkpoint
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict
from pydantic import BaseModel, TypeAdapter

from apax.nn.models import EnergyModel

FeatureMap = Callable[[FrozenDict, dict], jax.Array]


class FeatureTransformation(BaseModel):
    def apply(self, model: EnergyModel) -> FeatureMap:
        return model


def extract_feature_params(params: dict, layer_name: str) -> Tuple[dict, dict]:
    """Separate params into those belonging to a selected layer
    and the remaining ones.
    """
    p_flat = flatten_dict(params)

    feature_layer_params = {k: v for k, v in p_flat.items() if layer_name in k}
    remaining_params = {k: v for k, v in p_flat.items() if layer_name not in k}

    if len(feature_layer_params.keys()) > 2:  # w and b
        print(feature_layer_params.keys())
        raise ValueError("Found more than one layer of the specified name")

    return feature_layer_params, remaining_params


class LastLayerGradientFeatures(FeatureTransformation, extra="forbid"):
    """
    Model transfomration which computes the gradient of the output
    wrt. the specified layer.
    https://arxiv.org/pdf/2203.09410

    Parameters
    ----------
    layer_name: str
        Name of the layer wrt. which to take the gradient.
    """

    name: Literal["ll_grad"] = "ll_grad"
    layer_name: str = "dense_2"

    def apply(self, model: EnergyModel) -> FeatureMap:
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
                out = model(full_params, R, Z, idx, box, offsets)
                # take mean in case of shallow ensemble
                # no effect for single model
                out = jnp.mean(out)
                return out

            g_ll = jax.grad(inner)(ll_params)
            g_ll = unflatten_dict(g_ll)
            g_ll = jax.tree_map(lambda arr: jnp.mean(arr, axis=-1, keepdims=True), g_ll)
            g_flat = jax.tree_map(lambda arr: jnp.reshape(arr, (-1,)), g_ll)
            (gb, gw), _ = jax.tree_util.tree_flatten(g_flat)

            g = [gw, gb]
            g = jnp.concatenate(g)

            return g

        return ll_grad


class LastLayerForceFeatures(FeatureTransformation, extra="forbid"):
    """
    Model transformation which computes the jacobian of the forces
    wrt. the specified layer.
    For BAL the strategy "flatten" has to be selected.

    Parameters
    ----------
    layer_name: str
        Name of the layer wrt. which to take the jacobian.
    strategy: str
        one of raw, sum, flatten. Only flatten seems to work
        for BAL. raw is required for LLPR.
    """

    name: Literal["ll_force_feat"] = "ll_force_feat"
    layer_name: str = "dense_2"
    strategy: str = "raw"

    def apply(self, model: EnergyModel) -> FeatureMap:
        def ll_grad(params, inputs):
            ll_params, remaining_params = extract_feature_params(params, self.layer_name)

            energy_fn = lambda *inputs: jnp.mean(model(*inputs))
            force_fn = jax.grad(energy_fn, 1)

            def inner(ll_params):
                ll_params.update(remaining_params)
                full_params = unflatten_dict(ll_params)

                R, Z, idx, box, offsets = (
                    inputs["positions"],
                    inputs["numbers"],
                    inputs["idx"],
                    inputs["box"],
                    inputs["offsets"],
                )
                out = force_fn(full_params, R, Z, idx, box, offsets)
                return out

            g_ll = jax.jacfwd(inner)(ll_params)
            g_ll = unflatten_dict(g_ll)

            # shapes:
            # b: n_atoms, 3, 1
            # w: n_atoms, 3, n_features, 1

            if self.strategy == "raw":
                (gb, gw), _ = jax.tree_util.tree_flatten(g_ll)

                # g: n_atoms, 3, n_features
                g = gw[:, :, :, 0]
            elif self.strategy == "sum":
                g_summed = jax.tree_map(
                    lambda arr: jnp.reshape(jnp.sum(jnp.sum(arr, 0), 0), (-1,)), g_ll
                )
                (gb, gw), _ = jax.tree_util.tree_flatten(g_summed)
                g = [gw, gb]
                g = jnp.concatenate(g)

            elif self.strategy == "flatten":
                g_flat = jax.tree_map(lambda arr: jnp.reshape(arr, (-1,)), g_ll)
                (gb, gw), _ = jax.tree_util.tree_flatten(g_flat)
                g = gw
            else:
                raise ValueError(f"unknown strategy: {self.strategy}")

            return g

        return ll_grad


class FullGradientRPFeatures(FeatureTransformation, extra="forbid"):
    """
    Model transfomration which computes the gradient of the output
    wrt. all parameters and applies a gaussian random projection for
    dimensionality reduction.
    https://arxiv.org/pdf/2203.09410

    Parameters
    ----------
    num_rp: int
        Dimensionality to reduce the features to.
    """

    name: Literal["full_grad_rp"] = "full_grad_rp"
    num_rp: int = 512

    def apply(self, model: EnergyModel) -> FeatureMap:
        def full_grad(params, inputs):
            def inner(params):
                # TODO find better abstraction for inputs
                R, Z, idx, box, offsets = (
                    inputs["positions"],
                    inputs["numbers"],
                    inputs["idx"],
                    inputs["box"],
                    inputs["offsets"],
                )
                out = model(params, R, Z, idx, box, offsets)
                # take mean in case of shallow ensemble
                # no effect for single model
                out = jnp.mean(out)
                return out

            grads = jax.grad(inner)(params)
            grads = jax.tree_map(lambda arr: jnp.mean(arr, axis=-1, keepdims=True), grads)
            g_flat = jax.tree_map(lambda arr: jnp.reshape(arr, (-1,)), grads)
            gs, _ = jax.tree_util.tree_flatten(g_flat)
            g = jnp.concatenate(gs)

            with jax.ensure_compile_time_eval():
                n_features = g.shape[0]
                RP = np.random.randn(n_features, self.num_rp) / np.sqrt(self.num_rp)
                RP = jnp.array(RP)

            g_rp = g @ RP

            return g_rp

        return full_grad


class IdentityFeatures(FeatureTransformation, extra="forbid"):
    """Identity feature map. For debugging purposes"""

    name: Literal["identity"]

    def apply(self, model: EnergyModel) -> FeatureMap:
        return model


FeatureMapOptions = TypeAdapter(
    Union[
        LastLayerGradientFeatures,
        LastLayerForceFeatures,
        FullGradientRPFeatures,
        IdentityFeatures,
    ]
).validate_python
