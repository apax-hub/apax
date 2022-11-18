import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from ase.io import read

from gmnn_jax.model.lin_reg import LinReg
from gmnn_jax.utils.convert import convert_atoms_to_arrays, tf_to_jax_dict


def _forward_linreg(x):
    model = LinReg(n_species=15)
    return model(x)


def loss_fn(params, model, inputs, labels):
    prediction = model(x=inputs, params=params)
    # jax.debug.print("{prediction}", prediction=prediction)
    loss = optax.l2_loss(prediction, labels["energy"])
    return jnp.mean(loss)


atoms_list = read("raw_data/buoh.traj", index=":")
inputs, labels = convert_atoms_to_arrays(atoms_list)


ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
sample_inputs, _ = list(ds.take(1).as_numpy_iterator())[0]  # No batch dim!

ds = ds.batch(2)

rng_key = jax.random.PRNGKey(42)
forward_linreg = hk.without_apply_rng(hk.transform(_forward_linreg))

params = forward_linreg.init(rng=rng_key, x=sample_inputs)


optimizer = optax.adam(learning_rate=1e-2)
opt_state = optimizer.init(params)


@partial(jax.jit, static_argnames=["model"])
def step(model, params, opt_state, inputs, labels):
    loss_grad_fn = jax.value_and_grad(loss_fn, 0)  # grad wrt params
    loss, grads = loss_grad_fn(params, model, inputs, labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


for epoch in range(1):
    for inputs, labels in ds:
        inputs = tf_to_jax_dict(inputs)
        labels = tf_to_jax_dict(labels)

        params, opt_state, loss = step(
            forward_linreg.apply, params, opt_state, inputs, labels
        )

        print(loss)
