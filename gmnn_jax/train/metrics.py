from typing import Any

import jax.numpy as jnp
from clu import metrics


def mae_fn(label, prediction):
    return jnp.mean(jnp.abs(label - prediction))


def mse_fn(label, prediction):
    return jnp.mean((label - prediction) ** 2)


MeanAbsoluteError = metrics.Average.from_fun(mae_fn)
MeanSquaredError = metrics.Average.from_fun(mse_fn)


class RootMeanSquaredError(metrics.Average):
    def compute(self) -> Any:
        return jnp.sqrt(self.total / self.count)


# class MeanAbsoluteError(metrics.Metric):
#     mae = jnp.array

#     @classmethod
#     def from_model_output(cls, *, label, prediction) -> metrics.Metric:
#         return cls(mae = mae_fn(label, prediction))

#     def merge(self, other: metrics.Metric) -> metrics.Metric:
#         return type(self)(mae=self.mae + other.mae)

#     def compute(self):
