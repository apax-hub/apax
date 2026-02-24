import logging
import time
from functools import partial
from pathlib import Path
from typing import Any, Any, Callable, Dict, Optional, Union, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from clu import metrics
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jax import tree_util
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import trange

from apax.data.input_pipeline import InMemoryDataset
from apax.train.callbacks import CallbackCollection
from apax.train.checkpoints import load_state
from apax.train.loss import LossCollection
from apax.train.parameters import EMAParameters

log = logging.getLogger(__name__)


class EarlyStop(Exception):
    pass


def fit(
    state: TrainState,
    train_ds: InMemoryDataset,
    loss_fn: LossCollection,
    Metrics: metrics.Collection,
    callbacks: CallbackCollection,
    n_epochs: int,
    ckpt_dir: Path,
    ckpt_interval: int = 1,
    val_ds: Optional[InMemoryDataset] = None,
    patience: Optional[int] = None,
    patience_min_delta: float = 0.0,
    disable_pbar: bool = False,
    disable_batch_pbar: bool = True,
    is_ensemble: bool = False,
    data_parallel: bool = True,
    ema_handler: Optional[EMAParameters] = None,
) -> None:
    """
    Trains the model using the provided training dataset.

    Parameters
    ----------
    state :
        The initial state of the model.
    train_ds : InMemoryDataset
        The training dataset.
    loss_fn :
        The loss function to be minimized.
    Metrics metrics.Collection :
        Collection of metrics to evaluate during training.
    callbacks : list
        List of callback functions to be executed during training.
    n_epochs : int
        Number of epochs for training.
    ckpt_dir:
        Directory to save checkpoints.
    ckpt_interval : int, default = 1
        Interval for saving checkpoints.
    val_ds : InMemoryDataset, default = None
        Validation dataset.
    patience : int, default = None
        Patience for early stopping.
    disable_pbar : bool, default = False
        Whether to disable progress bar for epochs..
    disable_batch_pbar : bool, default = True
        Whether to disable progress bar for batches.
    is_ensemble : bool, default = False
        Whether the model is an ensemble.
    data_parallel : bool, default = True
        Whether to use data parallelism.
    """

    log.info("Beginning Training")
    callbacks.on_train_begin()

    latest_dir = ckpt_dir / "latest"
    best_dir = ckpt_dir / "best"

    options = ocp.CheckpointManagerOptions(max_to_keep=2, save_interval_steps=1)

    train_step, val_step = make_step_fns(
        loss_fn, Metrics, model=state.apply_fn, is_ensemble=is_ensemble
    )

    state, start_epoch = load_state(state, latest_dir)
    if start_epoch >= n_epochs:
        print(
            f"Training has already completed ({start_epoch} >= {n_epochs}). Nothing to be done"
        )
        return

    devices = len(jax.devices())
    if devices > 1 and data_parallel:
        devices = mesh_utils.create_device_mesh((jax.device_count(),))
        mesh = Mesh(devices, axis_names=("data",))
        replicated_sharding = NamedSharding(mesh, P())
        state = jax.device_put(state, replicated_sharding)
    else:
        mesh = None

    train_steps_per_epoch = train_ds.steps_per_epoch()
    batch_train_ds = train_ds.shuffle_and_batch(mesh)

    if val_ds is not None:
        val_steps_per_epoch = val_ds.steps_per_epoch()
        batch_val_ds = val_ds.batch(mesh)

    best_loss = np.inf
    early_stopping_counter = 0
    epoch_loss: Dict[str, float] = {}
    epoch_pbar = trange(
        start_epoch, n_epochs, desc="Epochs", ncols=100, disable=disable_pbar, leave=True
    )
    try:
        with (
            ocp.CheckpointManager(
                latest_dir.resolve(), options=options
            ) as latest_ckpt_manager,
            ocp.CheckpointManager(
                best_dir.resolve(), options=options
            ) as best_ckpt_manager,
        ):
            for epoch in range(start_epoch, n_epochs):
                epoch_start_time = time.time()
                callbacks.on_epoch_begin(epoch=epoch + 1)

                if ema_handler:
                    ema_handler.update(state.params, epoch)

                epoch_loss.update({"train_loss": 0.0})
                train_batch_metrics = Metrics.empty()

                batch_pbar = trange(
                    0,
                    train_steps_per_epoch,
                    desc="Batches",
                    ncols=100,
                    mininterval=1.0,
                    disable=disable_batch_pbar,
                    leave=False,
                )

                for batch_idx in range(train_steps_per_epoch):
                    callbacks.on_train_batch_begin(batch=batch_idx)

                    batch = next(batch_train_ds)
                    (
                        (state, train_batch_metrics),
                        batch_loss,
                    ) = train_step(
                        (state, train_batch_metrics),
                        batch,
                    )

                    epoch_loss["train_loss"] += jnp.mean(batch_loss)
                    callbacks.on_train_batch_end(batch=batch_idx)
                    batch_pbar.update()

                epoch_loss["train_loss"] /= train_steps_per_epoch
                epoch_loss["train_loss"] = float(epoch_loss["train_loss"])

                epoch_metrics: Dict[str, Any] = {
                    f"train_{key}": float(val)
                    for key, val in train_batch_metrics.compute().items()
                }

                if ema_handler:
                    ema_handler.update(state.params, epoch)
                    val_params = ema_handler.ema_params
                else:
                    val_params = state.params

                if val_ds is not None:
                    epoch_loss.update({"val_loss": 0.0})
                    val_batch_metrics = Metrics.empty()

                    batch_pbar = trange(
                        0,
                        val_steps_per_epoch,
                        desc="Batches",
                        ncols=100,
                        mininterval=1.0,
                        disable=disable_batch_pbar,
                        leave=False,
                    )
                    for batch_idx in range(val_steps_per_epoch):
                        batch = next(batch_val_ds)

                        batch_loss, val_batch_metrics = val_step(
                            val_params, batch, val_batch_metrics
                        )
                        epoch_loss["val_loss"] += batch_loss
                        batch_pbar.update()

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
                    latest_ckpt_manager.save(epoch, args=ocp.args.StandardSave(ckpt))

                if epoch_metrics["val_loss"] < best_loss:
                    best_ckpt_manager.save(epoch, args=ocp.args.StandardSave(ckpt))
                    if abs(epoch_metrics["val_loss"] - best_loss) < patience_min_delta:
                        early_stopping_counter += 1
                    else:
                        early_stopping_counter = 0

                    best_loss = epoch_metrics["val_loss"]
                else:
                    early_stopping_counter += 1

                callbacks.on_epoch_end(epoch=epoch, logs=epoch_metrics)

                epoch_pbar.set_postfix(val_loss=epoch_metrics["val_loss"])
                epoch_pbar.update()

                if patience is not None and early_stopping_counter >= patience:
                    raise EarlyStop()
    except EarlyStop:
        log.info(
            f"Early stopping patience exceeded. Stopping training after {epoch} epochs."
        )

    epoch_pbar.close()
    callbacks.on_train_end()

    train_ds.cleanup()
    if val_ds:
        val_ds.cleanup()


def calc_loss(
    params: FrozenDict,
    inputs: Dict[str, jnp.ndarray],
    labels: Dict[str, jnp.ndarray],
    loss_fn: LossCollection,
    model: Callable,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    R, Z, idx, box, offsets = (
        inputs["positions"],
        inputs["numbers"],
        inputs["idx"],
        inputs["box"],
        inputs["offsets"],
    )
    predictions = model(params, R, Z, idx, box, offsets)
    loss = loss_fn(inputs, predictions, labels)
    return loss, predictions


def make_ensemble_update(update_fn: Callable) -> Callable:
    # vmap over train state
    v_update_fn = jax.vmap(update_fn, (0, None, None), (0, 0, 0))

    def ensemble_update_fn(
        state: TrainState,
        inputs: Dict[str, jnp.ndarray],
        labels: Dict[str, jnp.ndarray],
    ) -> Tuple[float, Dict[str, jnp.ndarray], TrainState]:
        loss, predictions, state = v_update_fn(state, inputs, labels)

        mean_predictions = tree_util.tree_map(lambda x: jnp.mean(x, axis=0), predictions)
        mean_loss = jnp.mean(loss)
        # Should we add std to predictions?
        return mean_loss, mean_predictions, state

    return ensemble_update_fn


def make_ensemble_eval(update_fn: Callable) -> Callable:
    # vmap over train state
    v_update_fn = jax.vmap(update_fn, (0, None, None), (0, 0))

    def ensemble_eval_fn(
        state: TrainState,
        inputs: Dict[str, jnp.ndarray],
        labels: Dict[str, jnp.ndarray],
    ) -> Tuple[float, Dict[str, jnp.ndarray]]:
        loss, predictions = v_update_fn(state, inputs, labels)

        mean_predictions = tree_util.tree_map(lambda x: jnp.mean(x, axis=0), predictions)
        mean_loss = jnp.mean(loss)
        return mean_loss, mean_predictions

    return ensemble_eval_fn


def make_step_fns(
    loss_fn: LossCollection,
    Metrics: metrics.Collection,
    model: Callable,
    is_ensemble: bool,
    return_predictions: bool = False,
) -> tuple[Callable, Callable]:
    """
    Creates JIT-compiled training and validation step functions.

    This factory handles the boilerplate for gradient calculation, state updates,
    metric aggregation, and optional ensemble logic.

    Parameters
    ----------
        loss_fn: Callable
            A callable that takes (predictions, labels) and returns a scalar loss.
        Metrics: metrics.Collection
            A class (typically a clu.metrics.Collection) used to track
            and merge batch statistics. Must implement `single_from_model_output`.
        model: Any
            The model architecture (e.g., a flax.linen.Module).
        is_ensemble: bool
            If True, wraps the update and eval functions with
            ensemble-specific handling logic.
        return_predictions: bool, default = False
            If True, the validation step will return the
            raw model predictions in addition to metrics and loss.

    Returns
    -------
        Tuple[Callable, Callable]
            A tuple of (train_step, val_step), where:
            - train_step: (carry, batch) -> (new_carry, loss)
            - val_step: (params, batch, metrics) -> (loss, metrics, [predictions])
    """
    loss_calculator = partial(calc_loss, loss_fn=loss_fn, model=model)
    grad_fn = jax.value_and_grad(loss_calculator, 0, has_aux=True)

    def update_step(
        state: TrainState,
        inputs: Dict[str, jnp.ndarray],
        labels: Dict[str, jnp.ndarray],
    ) -> Tuple[float, Dict[str, jnp.ndarray], TrainState]:
        (loss, predictions), grads = grad_fn(state.params, inputs, labels)
        state = state.apply_gradients(grads=grads)
        return loss, predictions, state

    if is_ensemble:
        update_fn = make_ensemble_update(update_step)
        eval_fn = make_ensemble_eval(loss_calculator)
    else:
        update_fn = update_step
        eval_fn = loss_calculator

    @jax.jit
    def train_step(
        carry: tuple[TrainState, metrics.Collection],
        batch: tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
    ) -> tuple[tuple[TrainState, metrics.Collection], float]:
        state, batch_metrics = carry
        inputs, labels = batch
        loss, predictions, state = update_fn(state, inputs, labels)

        new_batch_metrics = Metrics.single_from_model_output(
            inputs=inputs, label=labels, prediction=predictions
        )
        batch_metrics = batch_metrics.merge(new_batch_metrics)

        new_carry = (state, batch_metrics)
        return new_carry, loss

    @jax.jit
    def val_step(
        params: FrozenDict,
        batch: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        batch_metrics: metrics.Collection,
    ) -> Union[tuple[float, metrics.Collection], tuple[float, metrics.Collection, Any]]:
        inputs, labels = batch
        loss, predictions = eval_fn(params, inputs, labels)

        new_batch_metrics = Metrics.single_from_model_output(
            inputs=inputs, label=labels, prediction=predictions
        )
        batch_metrics = batch_metrics.merge(new_batch_metrics)
        if return_predictions:
            return loss, batch_metrics, predictions
        else:
            return loss, batch_metrics

    return train_step, val_step
