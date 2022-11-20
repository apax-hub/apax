# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import haiku as hk
import jax
import tensorflow as tf

# IMPORTANT
# If both TF and Jax attempt to reserve GPU memory,
# then we are immediately OOM!!!
tf.config.experimental.set_visible_devices([], "GPU")

from typing import Optional

import numpy as np


class MyLayer(hk.Module):
    def __init__(self, hidden, name: Optional[str] = None):
        super().__init__(name)

        self.hidden = hidden

    def __call__(self, inputs):
        print("recompiling!")

        dense = hk.Sequential(
            [
                hk.Linear(self.hidden),
                jax.nn.swish,
                hk.Linear(1),
            ]
        )
        prediction = dense(inputs)
        return prediction


def get_model(hidden=3):
    @hk.without_apply_rng
    @hk.transform
    def model(x):
        lin_reg = MyLayer(hidden=hidden)
        return lin_reg(x)

    return model.init, model.apply


x = np.random.normal(0.0, 1.0, (100, 2))
y = np.random.normal(0.0, 1.0, (100, 2))

print(x.shape, y.shape)


print("sample inputs")
sample_inputs = x[0]
print(sample_inputs.shape)

ds = tf.data.Dataset.from_tensor_slices((x, y))
sample_inputs2, _ = next(ds.take(1).as_numpy_iterator())
print(sample_inputs2.shape)
print()


model_init, model = get_model(3)
model_init2, model2 = get_model(3)

rng_key = jax.random.PRNGKey(42)
params = model_init(rng=rng_key, x=sample_inputs)
params2 = model_init(rng=rng_key, x=sample_inputs2)


print("sample predictions")
prediction = model(params, sample_inputs)
print(prediction.shape)

prediction2 = model2(params2, sample_inputs2)
print(prediction2.shape)
print()

print("batches")
batch = x[:4]
print(type(batch))
print(batch.shape)
print(batch)
print()

ds = ds.batch(4)
batch2, _ = next(ds.take(1).as_numpy_iterator())
print(type(batch2))
print(batch2.shape)
print(batch2)

print(batch == batch2)

print()


print("predictions")
prediction = model(params, batch)
print(prediction.shape)

prediction2 = model2(params2, batch2)
print(prediction2.shape)
