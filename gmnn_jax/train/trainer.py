import time
from functools import partial

import jax

from gmnn_jax.utils.convert import tf_to_jax_dict


def fit(
    self,
    model,
    tx,
    train_ds,
    loss_fn,
    Metrics,
    callbacks,
    max_epoch,
    val_ds=None,
):
    step_fn = get_step_fn(loss_fn)
    callbacks.on_train_begin()
    # TODO epoch from load state / state
    epoch = 0
    if epoch >= self.max_epoch:
        raise ValueError(
            f"max_epoch <= current epoch from checkpoint ({self.max_epoch} <= {epoch})"
        )

    epoch_start_time = time.time()
    for epoch in range(max_epoch):
        callbacks.on_epoch_begin(epoch=epoch)

        epoch += 1
        train_batch_metrics = Metrics.empty()
        val_batch_metrics = Metrics.empty()
        epoch_loss = {"train_loss": 0, "val_loss": 0}

        for batch_idx, data in enumerate(train_ds):
            callbacks.on_train_batch_begin(batch=batch_idx)

            inputs, labels = data
            inputs = tf_to_jax_dict(inputs)
            labels = tf_to_jax_dict(labels)

            train_batch_metrics, batch_loss, state = step_fn(
                model, state, Metrics, inputs, labels, train_batch_metrics
            )
            epoch_loss["train_loss"] += batch_loss

            train_batch_step = batch_idx
            callbacks.on_train_batch_end(batch=batch_idx)
        epoch_loss["train_loss"] /= train_batch_step

        if val_ds is not None:
            for batch_idx, data in enumerate(val_ds):
                inputs, labels = data
                inputs = tf_to_jax_dict(inputs)
                labels = tf_to_jax_dict(labels)

                batch_loss, predictions = get_loss(
                    model, state, Metrics, inputs, labels, loss_fn
                )
                epoch_loss["val_loss"] += batch_loss
                new_val_metrics = Metrics.single_from_model_output(labels, predictions)
                val_batch_metrics = val_batch_metrics.merge(new_val_metrics)
                val_batch_step = batch_idx
            epoch_loss["val_loss"] /= val_batch_step

        train_epoch_metrics = {
            f"train_{key}": val for key, val in train_batch_metrics.compute().items()
        }
        val_epoch_metrics = {
            f"val_{key}": val for key, val in val_batch_metrics.compute().items()
        }
        epoch_metrics = {**train_epoch_metrics, **val_epoch_metrics, **epoch_loss}

        epoch_end_time = time.time()
        epoch_metrics.update({"epoch_time": epoch_end_time - epoch_start_time})

        callbacks.on_epoch_end(epoch=epoch, logs=epoch_metrics)


def get_loss(model, params, inputs, labels, loss_fn):
    predictions = model(params, inputs)
    loss = loss_fn(labels, predictions)
    return loss, predictions


def get_step_fn(loss_fn):
    get_loss_fn = partial(get_loss, loss_fn=loss_fn)

    @partial(jax.jit, static_argnames=["model"])
    def step(model, state, Metrics, inputs, labels, batch_metrics):
        grad_fn = jax.grad(get_loss_fn, 1, has_aux=True)
        loss, predictions, grads = grad_fn(model, state.params, inputs, labels)

        new_state = state.apply_gradients(grads=grads)

        new_batch_metrics = Metrics.single_from_model_output(labels, predictions)
        batch_metrics = batch_metrics.merge(new_batch_metrics)
        return batch_metrics, loss, new_state

    return step
