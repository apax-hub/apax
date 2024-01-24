import functools
import logging
import time
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from clu import metrics
from tqdm import trange

from apax.data.input_pipeline import AtomisticDataset
from apax.train.checkpoints import CheckpointManager, load_state

log = logging.getLogger(__name__)


def fit(
    state,
    train_ds: AtomisticDataset,
    loss_fn,
    Metrics: metrics.Collection,
    callbacks: list,
    n_epochs: int,
    ckpt_dir,
    ckpt_interval: int = 1,
    val_ds: Optional[AtomisticDataset] = None,
    sam_rho=0.0,
    patience: Optional[int] = None,
    disable_pbar: bool = False,
    is_ensemble=False,
    n_jitted_steps=1,
):
    log.info("Beginning Training")
    callbacks.on_train_begin()

    latest_dir = ckpt_dir / "latest"
    best_dir = ckpt_dir / "best"
    ckpt_manager = CheckpointManager()

    train_step, val_step = make_step_fns(
        loss_fn, Metrics, model=state.apply_fn, sam_rho=sam_rho, is_ensemble=is_ensemble
    )
    if n_jitted_steps > 1:
        train_step = jax.jit(functools.partial(jax.lax.scan, train_step))

    state, start_epoch = load_state(state, latest_dir)
    if start_epoch >= n_epochs:
        raise ValueError(
            f"n_epochs <= current epoch from checkpoint ({n_epochs} <= {start_epoch})"
        )

    train_ds.batch_multiple_steps(n_jitted_steps)
    train_steps_per_epoch = train_ds.steps_per_epoch()
    batch_train_ds = train_ds.shuffle_and_batch()

    if val_ds is not None:
        val_steps_per_epoch = val_ds.steps_per_epoch()
        batch_val_ds = val_ds.shuffle_and_batch()

    best_loss = np.inf
    early_stopping_counter = 0
    epoch_loss = {}
    epoch_pbar = trange(
        start_epoch, n_epochs, desc="Epochs", ncols=100, disable=disable_pbar, leave=True
    )
    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()
        callbacks.on_epoch_begin(epoch=epoch + 1)

        epoch_loss.update({"train_loss": 0.0})
        train_batch_metrics = Metrics.empty()

        for batch_idx in range(train_steps_per_epoch):
            callbacks.on_train_batch_begin(batch=batch_idx)

            batch = next(batch_train_ds)
            ((state, train_batch_metrics), batch_loss,) = train_step(
                (state, train_batch_metrics),
                batch,
            )

            epoch_loss["train_loss"] += jnp.mean(batch_loss)
            callbacks.on_train_batch_end(batch=batch_idx)

        epoch_loss["train_loss"] /= train_steps_per_epoch
        epoch_loss["train_loss"] = float(epoch_loss["train_loss"])

        epoch_metrics = {
            f"train_{key}": float(val)
            for key, val in train_batch_metrics.compute().items()
        }

        if val_ds is not None:
            epoch_loss.update({"val_loss": 0.0})
            val_batch_metrics = Metrics.empty()
            for batch_idx in range(val_steps_per_epoch):
                batch = next(batch_val_ds)

                batch_loss, val_batch_metrics = val_step(
                    state.params, batch, val_batch_metrics
                )
                epoch_loss["val_loss"] += batch_loss

            epoch_loss["val_loss"] /= val_steps_per_epoch
            epoch_loss["val_loss"] = float(epoch_loss["val_loss"])

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
        if epoch % ckpt_interval == 0:
            ckpt_manager.save_checkpoint(ckpt, epoch, latest_dir)

        if epoch_metrics["val_loss"] < best_loss:
            best_loss = epoch_metrics["val_loss"]
            ckpt_manager.save_checkpoint(ckpt, epoch, best_dir)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        callbacks.on_epoch_end(epoch=epoch, logs=epoch_metrics)

        epoch_pbar.set_postfix(val_loss=epoch_metrics["val_loss"])
        epoch_pbar.update()

        if patience is not None and early_stopping_counter >= patience:
            log.info(
                "Early stopping patience exceeded. Stopping training after"
                f" {epoch} epochs."
            )
            break
    epoch_pbar.close()
    callbacks.on_train_end()


def global_norm(updates) -> jnp.ndarray:
    """Returns the l2 norm of the input.
    Args:
      updates: A pytree of ndarrays representing the gradient.
    """
    norm = jax.tree_map(lambda u: jnp.sqrt(jnp.sum(jnp.square(u))), updates)
    return norm


def calc_loss(params, inputs, labels, loss_fn, model):
    R, Z, idx, box, offsets = (
        inputs["positions"],
        inputs["numbers"],
        inputs["idx"],
        inputs["box"],
        inputs["offsets"],
    )
    predictions = model(params, R, Z, idx, box, offsets)
    loss = loss_fn(inputs, labels, predictions)
    return loss, predictions


def make_ensemble_update(update_fn: Callable) -> Callable:
    # vmap over train state
    v_update_fn = jax.vmap(update_fn, (0, None, None), (0, 0, 0))

    def ensemble_update_fn(state, inputs, labels):
        loss, predictions, state = v_update_fn(state, inputs, labels)

        mean_predictions = jax.tree_map(lambda x: jnp.mean(x, axis=0), predictions)
        mean_loss = jnp.mean(loss)
        # Should we add std to predictions?
        return mean_loss, mean_predictions, state

    return ensemble_update_fn


def make_ensemble_eval(update_fn: Callable) -> Callable:
    # vmap over train state
    v_update_fn = jax.vmap(update_fn, (0, None, None), (0, 0))

    def ensemble_eval_fn(state, inputs, labels):
        loss, predictions = v_update_fn(state, inputs, labels)

        mean_predictions = jax.tree_map(lambda x: jnp.mean(x, axis=0), predictions)
        mean_loss = jnp.mean(loss)
        return mean_loss, mean_predictions

    return ensemble_eval_fn


def make_step_fns(loss_fn, Metrics, model, sam_rho, is_ensemble):
    loss_calculator = partial(calc_loss, loss_fn=loss_fn, model=model)
    grad_fn = jax.value_and_grad(loss_calculator, 0, has_aux=True)
    rho = sam_rho

    def update_step(state, inputs, labels):
        (loss, predictions), grads = grad_fn(state.params, inputs, labels)

        if rho > 1e-6:
            # SAM step
            grad_norm = global_norm(grads)
            eps = jax.tree_map(lambda g, n: g * rho / n, grads, grad_norm)
            params_eps = jax.tree_map(lambda p, e: p + e, state.params, eps)
            (loss, _), grads = grad_fn(params_eps, inputs, labels)  # maybe get rid of SAM

        state = state.apply_gradients(grads=grads)
        return loss, predictions, state

    if is_ensemble:
        update_fn = make_ensemble_update(update_step)
        eval_fn = make_ensemble_eval(loss_calculator)
    else:
        update_fn = update_step
        eval_fn = loss_calculator

    @jax.jit
    def train_step(carry, batch):
        state, batch_metrics = carry
        inputs, labels = batch
        loss, predictions, state = update_fn(state, inputs, labels)

        new_batch_metrics = Metrics.single_from_model_output(
            label=labels, prediction=predictions
        )
        batch_metrics = batch_metrics.merge(new_batch_metrics)

        new_carry = (state, batch_metrics)
        return new_carry, loss

    @jax.jit
    def val_step(params, batch, batch_metrics):
        inputs, labels = batch
        loss, predictions = eval_fn(params, inputs, labels)

        new_batch_metrics = Metrics.single_from_model_output(
            label=labels, prediction=predictions
        )
        batch_metrics = batch_metrics.merge(new_batch_metrics)
        return loss, batch_metrics

    return train_step, val_step
