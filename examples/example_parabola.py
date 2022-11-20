import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

# IMPORTANT
# If both TF and Jax attempt to reserve GPU memory,
# then we are immediately OOM!!!
tf.config.experimental.set_visible_devices([], "GPU")

from typing import Optional

import numpy as np
from clu import metrics
from tensorflow.keras.callbacks import CSVLogger


class MyLayer(hk.Module):
    def __init__(self, hidden, name: Optional[str] = None):
        super().__init__(name)

        self.dense = hk.Sequential(
            [
                hk.Linear(hidden),
                jax.nn.swish,
                hk.Linear(hidden),
                jax.nn.swish,
                hk.Linear(1),
            ]
        )

    def __call__(self, inputs):
        print("recompiling!")
        prediction = self.dense(inputs)
        return prediction


def get_model(hidden=3):
    @hk.without_apply_rng
    @hk.transform
    def model(x):
        lin_reg = MyLayer(hidden=hidden)
        return lin_reg(x)

    return model.init, model.apply


data = np.load("./raw_data/parabola.npz")
x = data["x"][:, None]  # Add batch dim
y = data["y"][:, None]
print(x.shape, y.shape)

ds = tf.data.Dataset.from_tensor_slices((x, y))
print(ds.element_spec)
sample_inputs, _ = next(ds.take(1).as_numpy_iterator())
ds = ds.batch(8)


model_init, model = get_model(32)

rng_key = jax.random.PRNGKey(42)
params = model_init(rng=rng_key, x=sample_inputs)


def loss_fn(params, model, inputs, labels):
    prediction = model(params, inputs)
    loss = optax.l2_loss(prediction, labels)
    return jnp.mean(loss), prediction


@partial(jax.jit, static_argnames=["model"])
def step(model, params, opt_state, inputs, labels):
    grad_fn = jax.value_and_grad(loss_fn, 0, has_aux=True)
    (loss, prediction), grads = grad_fn(params, model, inputs, labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, prediction


schedule = optax.linear_schedule(
    init_value=1e-3, end_value=1e-5, transition_begin=128 * 2, transition_steps=128 * 5
)
optimizer = optax.adamw(learning_rate=1e-3)
opt_state = optimizer.init(params)

AverageLoss = metrics.Average.from_output("loss")

from dataclasses import dataclass


@dataclass
class TFModelSpoof:
    stop_training = False


callback = CSVLogger("log.csv", append=True)
callbacks = tf.keras.callbacks.CallbackList([callback], model=TFModelSpoof())

callbacks.on_train_begin()
epoch_losses = []
for epoch in range(3):
    callbacks.on_epoch_begin(epoch=epoch)
    batch_idx = 0
    for inputs, labels in ds:
        inputs = jnp.asarray(inputs)
        labels = jnp.asarray(labels)

        params, opt_state, loss, prediction = step(
            model, params, opt_state, inputs, labels
        )

        if batch_idx == 0:
            batch_loss = AverageLoss.from_model_output(loss=loss)
        else:
            new_batch_loss = AverageLoss.from_model_output(loss=loss)
            batch_loss = batch_loss.merge(new_batch_loss)

        batch_idx += 1
    epoch_loss = batch_loss.compute()

    metrics_dict = {"loss": np.asarray(epoch_loss)}
    callbacks.on_epoch_end(epoch=epoch, logs=metrics_dict)
    print(epoch_loss)

    epoch_losses.append(epoch_loss)


import matplotlib.pyplot as plt

plt.plot(x, y)

preds = model(params, x)
plt.plot(x, preds)
plt.show()
