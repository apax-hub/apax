import logging
from functools import partial

import jax.numpy as jnp
from clu import metrics

from apax.utils.math import normed_dotp

log = logging.getLogger(__name__)


class Averagefp64(metrics.Average):
    @classmethod
    def empty(cls) -> metrics.Metric:
        return cls(total=jnp.array(0, jnp.float64), count=jnp.array(0, jnp.int64))


class RootAverage(Averagefp64):
    """
    Modifies the `compute` method of `metrics.Average` to obtain the root of the average.
    Meant to be used with `mse_fn`.
    """

    def compute(self) -> jnp.array:
        return jnp.sqrt(self.total / self.count)


def mae_fn(label: dict[jnp.array], prediction: dict[jnp.array], key: str) -> jnp.array:
    """
    Computes the Mean Absolute Error of two arrays.
    """
    return jnp.mean(jnp.abs(label[key] - prediction[key]))


def mse_fn(label: dict[jnp.array], prediction: dict[jnp.array], key: str) -> jnp.array:
    """
    Computes the Mean Squared Error of two arrays.
    """
    return jnp.mean((label[key] - prediction[key]) ** 2)


def cosine_sim(
    label: dict[jnp.array], prediction: dict[jnp.array], key: str
) -> jnp.array:
    """
    Computes the cosine similarity of two arrays.
    """
    dotp = normed_dotp(label[key], prediction[key])
    F_angle_loss = jnp.mean(1.0 - dotp)
    return F_angle_loss


def make_single_metric(key: str, reduction: str) -> metrics.Average:
    """
    Builds a single `clu` metric where the key picks out the quantity from
    the model predictions dict.
    Metric functions (like `mae_fn`) are curried with the `key`.
    """
    reduction_fns = {
        "mae": mae_fn,
        "mse": mse_fn,
        "rmse": mse_fn,
        "cosine_sim": cosine_sim,
    }
    if reduction == "rmse":
        metric = RootAverage
    else:
        metric = Averagefp64

    reduction_fn = reduction_fns[reduction]
    reduction_fn = partial(reduction_fn, key=key)

    return metric.from_fun(reduction_fn)


def initialize_metrics(metrics_list) -> metrics.Collection:
    """
    Builds a `clu` metrics `Collection` by looping over all `keys` and `reductions`.
    the metrics are named according to `key_reduction`.
    See `make_single_metric` for details on the individual metrics.
    """
    log.info("Initializing Metrics")
    keys = []
    reductions = []
    for metric in metrics_list:
        for reduction in metric.reductions:
            keys.append(metric.name)
            reductions.append(reduction)

    metric_dict = {}
    for key, reduction in zip(keys, reductions):
        metric = make_single_metric(key, reduction)
        metric_identifier = f"{key}_{reduction}"
        metric_dict[metric_identifier] = metric

    metrics_collection = metrics.Collection.create(**metric_dict)

    return metrics_collection
