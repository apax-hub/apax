import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from functools import partial
from typing import NamedTuple
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


data = np.load("./parabola.npz")
x = data["x"][:, None]  # Add batch dim
y = data["y"][:, None]

ds = tf.data.Dataset.from_tensor_slices((x, y))
sample_inputs, _ = next(ds.take(1).as_numpy_iterator())
ds = ds.batch(8)

config = {"hidden": 32}
model_init, model = get_model(**config)

rng_key = jax.random.PRNGKey(42)
params = model_init(rng=rng_key, x=sample_inputs)


schedule = optax.linear_schedule(
    init_value=1e-1, end_value=1e-1, transition_begin=128 * 2, transition_steps=128 * 5
)

is_bias_mask_fn = partial(hk.data_structures.map, lambda mname, name, val: name != 'b')
not_bias_mask_fn = partial(hk.data_structures.map, lambda mname, name, val: name == 'b')

tx = optax.chain(
    optax.masked(optax.sgd(learning_rate=schedule), is_bias_mask_fn),
    optax.masked(optax.adam(learning_rate=0.01), not_bias_mask_fn))


AverageLoss = metrics.Average.from_output("loss")

@dataclasses.dataclass
class TFModelSpoof:
    stop_training = False


callback = CSVLogger("log.csv", append=True)
callbacks = tf.keras.callbacks.CallbackList([callback], model=TFModelSpoof())


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


start_epoch = 0
num_epochs = 5

ckpt_dir = "tmp/checkpoints"

checkpoints_exist = Path(ckpt_dir).is_dir()
print(checkpoints_exist)
state = train_state.TrainState.create(
    apply_fn=model,
    params=params,
    tx=tx,
)
target = {"model": state, "config": config, "epoch":0}

if checkpoints_exist:
    raw_restored = checkpoints.restore_checkpoint(ckpt_dir, target=target, step=None)
    state = raw_restored["model"]
    start_epoch = raw_restored["epoch"] + 1

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
    print(epoch_loss)

    epoch_losses.append(epoch_loss)

    ckpt = {"model": state, "config": config, "epoch": epoch}
    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir, target=ckpt, step=epoch, overwrite=True, keep=2
    )


import matplotlib.pyplot as plt

plt.scatter(x, y)

preds = model(state.params, x)
plt.scatter(x, preds)
plt.show()
