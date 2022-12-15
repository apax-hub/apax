import logging
import time
from functools import partial

import jax
from flax.training import checkpoints

from gmnn_jax.train.checkpoints import load_state

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
    callbacks.on_train_begin()

    train_step, val_step = make_step_fns(loss_fn, Metrics)

    state, start_epoch = load_state(model, params, tx, ckpt_dir)

    train_steps_per_epoch = train_ds.steps_per_epoch()
    batch_train_ds = train_ds.shuffle_and_batch()
    epoch_loss = {}

    if val_ds is not None:
        val_steps_per_epoch = val_ds.steps_per_epoch()
        batch_val_ds = val_ds.shuffle_and_batch()

    async_manager = checkpoints.AsyncManager()

    if start_epoch >= n_epochs:
        raise ValueError(
            f"n_epochs <= current epoch from checkpoint ({n_epochs} <= {start_epoch})"
        )

    for epoch in range(start_epoch, n_epochs):
        epoch += 1
        epoch_start_time = time.time()
        callbacks.on_epoch_begin(epoch=epoch)
        epoch_loss.update({"train_loss": 0.0})

        train_batch_metrics = Metrics.empty()

        for batch_idx in range(train_steps_per_epoch):
            callbacks.on_train_batch_begin(batch=batch_idx)

            inputs, labels = next(batch_train_ds)
            train_batch_metrics, batch_loss, state = train_step(
                model, state, inputs, labels, train_batch_metrics
            )

            epoch_loss["train_loss"] += batch_loss
            callbacks.on_train_batch_end(batch=batch_idx)

        epoch_loss["train_loss"] /= train_steps_per_epoch
        epoch_metrics = {
            f"train_{key}": float(val)
            for key, val in train_batch_metrics.compute().items()
        }

        if val_ds is not None:
            epoch_loss.update({"val_loss": 0.0})
            val_batch_metrics = Metrics.empty()
            for batch_idx in range(val_steps_per_epoch):
                inputs, labels = next(batch_val_ds)

                batch_loss, val_batch_metrics = val_step(
                    model, state, inputs, labels, val_batch_metrics
                )
                epoch_loss["val_loss"] += batch_loss

            epoch_loss["val_loss"] /= val_steps_per_epoch
            epoch_metrics.update(
                {
                    f"val_{key}": float(val)
                    for key, val in val_batch_metrics.compute().items()
                }
            )

        epoch_metrics.update({**epoch_loss})

        epoch_end_time = time.time()
        epoch_metrics.update({"epoch_time": epoch_end_time - epoch_start_time})

        ckpt = {"model": state, "epoch": epoch}
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=ckpt,
            step=epoch,
            overwrite=True,
            keep=2,
            async_manager=async_manager,
        )
        # TODO Save best
        callbacks.on_epoch_end(epoch=epoch, logs=epoch_metrics)


def calc_loss(model, params, inputs, labels, loss_fn):
    R, Z, idx = inputs["positions"], inputs["numbers"], inputs["idx"]
    predictions = model(params, R, Z, idx)
    loss = loss_fn(inputs, labels, predictions)
    return loss, predictions


def make_step_fns(loss_fn, Metrics, model):
    loss_calculator = partial(calc_loss, model=model, loss_fn=loss_fn)

    @jax.jit
    def train_step(state, inputs, labels, batch_metrics):
        grad_fn = jax.value_and_grad(loss_calculator, 1, has_aux=True)
        (loss, predictions), grads = grad_fn(state.params, inputs, labels)

        new_state = state.apply_gradients(grads=grads)

        new_batch_metrics = Metrics.single_from_model_output(
            label=labels, prediction=predictions
        )
        batch_metrics = batch_metrics.merge(new_batch_metrics)
        return batch_metrics, loss, new_state

    @jax.jit
    def val_step(state, inputs, labels, batch_metrics):
        loss, predictions = loss_calculator(state.params, inputs, labels)

        new_batch_metrics = Metrics.single_from_model_output(
            label=labels, prediction=predictions
        )
        batch_metrics = batch_metrics.merge(new_batch_metrics)
        return batch_metrics, loss

    return train_step, val_step
