import logging
import time
from functools import partial

import jax
from flax.training import checkpoints

from gmnn_jax.train.checkpoints import load_state
from gmnn_jax.utils.convert import tf_to_jax_dict

log = logging.getLogger(__name__)


def fit(
    model,
    params,
    tx,
    train_ds,
    loss_fn,
    Metrics,
    callbacks,
    n_epochs,
    ckpt_dir,
    val_ds=None,
):
    log.info("Begining Training")
    step_fn = get_step_fn(loss_fn, Metrics)
    callbacks.on_train_begin()
    state, start_epoch = load_state(model, params, tx, ckpt_dir)
    async_manager = checkpoints.AsyncManager()

    if start_epoch >= n_epochs:
        raise ValueError(
            f"n_epochs <= current epoch from checkpoint ({n_epochs} <= {start_epoch})"
        )

    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()
        print("EPOCH", epoch)
        callbacks.on_epoch_begin(epoch=epoch)

        train_batch_metrics = Metrics.empty()
        val_batch_metrics = Metrics.empty()
        epoch_loss = {"train_loss": 0, "val_loss": 0}

        for batch_idx, data in enumerate(train_ds):
            callbacks.on_train_batch_begin(batch=batch_idx)

            inputs, labels = data
            inputs = tf_to_jax_dict(inputs)
            labels = tf_to_jax_dict(labels)

            train_batch_metrics, batch_loss, state = step_fn(
                model, state, inputs, labels, train_batch_metrics
            )
            epoch_loss["train_loss"] += batch_loss

            train_batch_step = batch_idx
            callbacks.on_train_batch_end(batch=batch_idx)
        epoch_loss["train_loss"] /= train_batch_step
        epoch_loss["train_loss"] = float(epoch_loss["train_loss"])

        # if val_ds is not None:
        #     for batch_idx, data in enumerate(val_ds):
        #         inputs, labels = data
        #         inputs = tf_to_jax_dict(inputs)
        #         labels = tf_to_jax_dict(labels)

        #         batch_loss, predictions = get_loss(
        #             model, state, Metrics, inputs, labels, loss_fn
        #         )
        #         epoch_loss["val_loss"] += batch_loss
        #         new_val_metrics = Metrics.single_from_model_output(labels, predictions)
        #         val_batch_metrics = val_batch_metrics.merge(new_val_metrics)
        #         val_batch_step = batch_idx
        #     epoch_loss["val_loss"] /= val_batch_step

        train_epoch_metrics = {
            f"train_{key}": float(val)
            for key, val in train_batch_metrics.compute().items()
        }
        val_epoch_metrics = {
            f"val_{key}": float(val) for key, val in val_batch_metrics.compute().items()
        }
        epoch_metrics = {**train_epoch_metrics, **val_epoch_metrics, **epoch_loss}

        epoch_end_time = time.time()
        epoch_metrics.update({"epoch_time": epoch_end_time - epoch_start_time})

        ckpt = {"model": state, "epoch": epoch}
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=ckpt,
            step=epoch,  # Should we use the true step here?
            overwrite=True,
            keep=2,
            async_manager=async_manager,
        )
        # TODO Save best
        print(epoch_metrics)
        callbacks.on_epoch_end(epoch=epoch, logs=epoch_metrics)


def get_loss(model, params, inputs, labels, loss_fn):
    R, Z, idx = inputs["positions"], inputs["numbers"], inputs["idx"]
    predictions = model(params, R, Z, idx)
    loss = loss_fn(inputs, labels, predictions)
    return loss, predictions


def get_step_fn(loss_fn, Metrics):
    # MS: I would suggest to rename these from get_* to something else
    # as to not cause confusion with getter methods
    get_loss_fn = partial(get_loss, loss_fn=loss_fn)

    @partial(jax.jit, static_argnames=["model"])
    def step(model, state, inputs, labels, batch_metrics):
        grad_fn = jax.value_and_grad(get_loss_fn, 1, has_aux=True)
        (loss, predictions), grads = grad_fn(model, state.params, inputs, labels)

        new_state = state.apply_gradients(grads=grads)

        new_batch_metrics = Metrics.single_from_model_output(
            label=labels, prediction=predictions
        )
        batch_metrics = batch_metrics.merge(new_batch_metrics)
        return batch_metrics, loss, new_state

    return step
