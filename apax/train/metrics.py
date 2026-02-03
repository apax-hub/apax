import logging
from functools import partial
from typing import Dict, List

import jax.numpy as jnp
from clu import metrics
from jax import Array

from apax.config.train_config import MetricsConfig
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

    def compute(self) -> Array:
        return jnp.sqrt(self.total / self.count)


def mae_fn(
    inputs: Dict, label: Dict[str, Array], prediction: Dict[str, Array], key: str
) -> Array:
    """
    Computes the Mean Absolute Error of two arrays.
    """
    return jnp.mean(jnp.abs(label[key] - prediction[key]))


def mse_fn(
    inputs: Dict, label: Dict[str, Array], prediction: Dict[str, Array], key: str
) -> Array:
    """
    Computes the Mean Squared Error of two arrays.
    """
    return jnp.mean((label[key] - prediction[key]) ** 2)


def cosine_sim(
    inputs: Dict, label: Dict[str, Array], prediction: Dict[str, Array], key: str
) -> Array:
    """
    Computes the cosine similarity of two arrays.
    """
    dotp = normed_dotp(label[key], prediction[key])
    F_angle_loss = jnp.mean(1.0 - dotp)
    return F_angle_loss


def per_atom_mae_fn(
    inputs: Dict, label: Dict[str, Array], prediction: Dict[str, Array], key: str
) -> Array:
    """
    Computes the per atom Mean Absolute Error of two arrays.
    Only reasanable when using with structural
    properties like 'energy'.
    """
    err_abs = jnp.abs(label[key] - prediction[key])
    return jnp.sum(err_abs) / jnp.sum(inputs["n_atoms"])


def per_atom_mse_fn(
    inputs: Dict, label: Dict[str, Array], prediction: Dict[str, Array], key: str
) -> Array:
    """
    Computes the per atom Mean Squared Error of two arrays.
    Only reasanable when using with structural
    properties like 'energy'.
    """
    err_sq = (label[key] - prediction[key]) ** 2
    return jnp.sum(err_sq) / jnp.sum(inputs["n_atoms"])


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
        "per_atom_mae": per_atom_mae_fn,
        "per_atom_mse": per_atom_mse_fn,
        "per_atom_rmse": per_atom_mse_fn,
    }
    if reduction in ["rmse", "per_atom_rmse"]:
        metric = RootAverage
    else:
        metric = Averagefp64

    reduction_fn = reduction_fns[reduction]
    reduction_fn = partial(reduction_fn, key=key)

    return metric.from_fun(reduction_fn)


def initialize_metrics(metrics_list: List[MetricsConfig]) -> metrics.Collection:
    """
    Builds a `clu` metrics `Collection` by looping over all `keys` and `reductions`.
    the metrics are named according to `key_reduction`.
    See `make_single_metric` for details on the individual metrics.
    """
    log.info("Initializing Metrics")
    keys: List[str] = []
    reductions: List[str] = []
    for metric in metrics_list:
        for reduction in metric.reductions:
            keys.append(metric.name)
            reductions.append(reduction)

    metric_dict: Dict[str, metrics.Average] = {}
    for key, reduction in zip(keys, reductions):
        metric = make_single_metric(key, reduction)
        metric_identifier = f"{key}_{reduction}"
        metric_dict[metric_identifier] = metric

    metrics_collection = metrics.Collection.create(**metric_dict)

    return metrics_collection
