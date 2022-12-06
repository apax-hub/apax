from functools import partial
from typing import Any

import jax.numpy as jnp
from clu import metrics


class RootAverage(metrics.Average):
    def compute(self) -> Any:
        return jnp.sqrt(self.total / self.count)


def mae_fn(label, prediction, key):
    return jnp.mean(jnp.abs(label[key] - prediction[key]))


def mse_fn(label, prediction, key):
    return jnp.mean((label[key] - prediction[key]) ** 2)


def cosine_sim(label, prediction, key):
    F_0_norm = jnp.linalg.norm(label[key], ord=2, axis=2, keepdims=True)
    F_p_norm = jnp.linalg.norm(prediction[key], ord=2, axis=2, keepdims=True)

    F_0_n = label[key] / F_0_norm
    F_p_n = prediction[key] / F_p_norm

    dotp = jnp.einsum("bai, bai -> ba", F_0_n, F_p_n)
    F_angle_loss = jnp.mean(1.0 - dotp)
    return F_angle_loss


def make_single_metric(key, reduction):
    reduction_fns = {
        "mae": mae_fn,
        "mse": mse_fn,
        "rmse": mse_fn,
        "cosine_sim": cosine_sim,
    }
    if reduction == "rmse":
        metric = RootAverage
    else:
        metric = metrics.Average

    reduction_fn = reduction_fns[reduction]
    reduction_fn = partial(reduction_fn, key=key)

    return metric.from_fun(reduction_fn)


def initialize_metrics(keys, reductions):
    metric_dict = {}
    for key, reduction in zip(keys, reductions):
        metric = make_single_metric(key, reduction)
        metric_identifier = f"{key}_{reduction}"
        metric_dict[metric_identifier] = metric

    metrics_collection = metrics.Collection.create(**metric_dict)

    return metrics_collection
