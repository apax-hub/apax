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

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
from clu import metrics
from flax.training import checkpoints, train_state
from tensorflow.keras.callbacks import CSVLogger

from apax.utils.random import seed_py_np_tf

seed_py_np_tf()


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


def load_dataset():
    data = np.load("./parabola.npz")
    x = data["x"][:, None]  # Add batch dim
    y = data["y"][:, None]

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    return ds


@dataclasses.dataclass
class TFModelSpoof:
    stop_training = False


def init_callbacks(path=None):
    callback = CSVLogger("log.csv", append=True)
    callbacks = tf.keras.callbacks.CallbackList([callback], model=TFModelSpoof())
    return callbacks


def map_nested_fn(fn):
    """Recursively apply `fn` to the key-value pairs of a nested dict"""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def get_opt():
    schedule = optax.linear_schedule(
        init_value=1e-1,
        end_value=1e-1,
        transition_begin=128 * 2,
        transition_steps=128 * 5,
    )
    label_fn = map_nested_fn(lambda k, _: k)
    tx = optax.multi_transform(
        {"w": optax.adam(0.01), "b": optax.sgd(schedule)}, label_fn
    )
    return tx


def loss_fn(params, model, inputs, labels):
    prediction = model(params, inputs)
    loss = optax.l2_loss(prediction, labels)
    return jnp.mean(loss), prediction


@partial(jax.jit, static_argnames=["model"])
def step(model, state, inputs, labels, batch_loss):
    grad_fn = jax.value_and_grad(loss_fn, 0, has_aux=True)
    (loss, prediction), grads = grad_fn(state.params, model, inputs, labels)

    new_state = state.apply_gradients(grads=grads)

    new_batch_loss = AverageLoss.from_model_output(loss=loss)
    batch_loss = batch_loss.merge(new_batch_loss)
    return new_state, loss, prediction, batch_loss


def load_state(ckpt_dir, start_epoch):
    state = train_state.TrainState.create(
        apply_fn=model,
        params=params,
        tx=tx,
    )
    target = {"model": state, "config": config, "epoch": 0}
    checkpoints_exist = Path(ckpt_dir).is_dir()
    if checkpoints_exist:
        raw_restored = checkpoints.restore_checkpoint(ckpt_dir, target=target, step=None)
        state = raw_restored["model"]
        start_epoch = raw_restored["epoch"] + 1

    return state, start_epoch


ds = load_dataset()
sample_inputs, _ = next(ds.take(1).as_numpy_iterator())
ds = ds.batch(8)

rng_key = jax.random.PRNGKey(42)


config = {"hidden": 32}
model_init, model = get_model(**config)
params = model_init(rng=rng_key, x=sample_inputs)

tx = get_opt()

AverageLoss = metrics.Average.from_output("loss")
callbacks = init_callbacks()

start_epoch = 0
num_epochs = 5

ckpt_dir = "tmp/checkpoints"

state, start_epoch = load_state(ckpt_dir, start_epoch)
async_manager = checkpoints.AsyncManager()
print(f"start epoch: {start_epoch}, num epochs: {num_epochs}")
callbacks.on_train_begin()

epoch_losses = []
for epoch in range(start_epoch, num_epochs):
    print(f"EPOCH {epoch}")
    callbacks.on_epoch_begin(epoch=epoch)
    batch_loss = AverageLoss.empty()

    batch_idx = 0
    for inputs, labels in ds:
        inputs = jnp.asarray(inputs)
        labels = jnp.asarray(labels)

        state, loss, prediction, batch_loss = step(
            model, state, inputs, labels, batch_loss
        )
        batch_idx += 1

    # Metrics
    epoch_loss = batch_loss.compute()
    metrics_dict = {"loss": np.asarray(epoch_loss)}
    callbacks.on_epoch_end(epoch=epoch, logs=metrics_dict)
    epoch_losses.append(epoch_loss)
    print(epoch_loss)

    ckpt = {"model": state, "config": config, "epoch": epoch}
    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir,
        target=ckpt,
        step=epoch,
        overwrite=True,
        keep=2,
        async_manager=async_manager,
    )


import matplotlib.pyplot as plt

data = np.load("./parabola.npz")
x = data["x"][:, None]
y = data["y"][:, None]
plt.scatter(x, y)

preds = model(state.params, x)
plt.scatter(x, preds)
plt.show()
