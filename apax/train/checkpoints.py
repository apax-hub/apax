import logging
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.training import checkpoints, train_state

log = logging.getLogger(__name__)


def load_state(model, params, tx, ckpt_dir):
    start_epoch = 0
    state = train_state.TrainState.create(
        apply_fn=model,
        params=params,
        tx=tx,
    )
    target = {"model": state, "epoch": 0}
    checkpoints_exist = Path(ckpt_dir).is_dir()
    if checkpoints_exist:
        log.info("Loading checkpoint")
        raw_restored = checkpoints.restore_checkpoint(ckpt_dir, target=target, step=None)
        state = raw_restored["model"]
        start_epoch = raw_restored["epoch"] + 1
        log.info("Successfully restored checkpoint from epoch %d", raw_restored["epoch"])

    return state, start_epoch


class CheckpointManager:
    def __init__(self) -> None:
        self.async_manager = checkpoints.AsyncManager()

    def save_checkpoint(self, ckpt, epoch: int, path: str) -> None:
        checkpoints.save_checkpoint(
            ckpt_dir=path,
            target=ckpt,
            step=epoch,
            overwrite=True,
            keep=2,
            async_manager=self.async_manager,
        )


def load_params(model_version_path, best=True):
    if best:
        model_version_path = model_version_path / "best"
    log.info(f"loading checkpoint from {model_version_path}")
    try:
        raw_restored = checkpoints.restore_checkpoint(
            model_version_path,
            target=None,
            step=None
        )
    except FileNotFoundError:
        print(f"No checkpoint found at {model_version_path}")
    params = jax.tree_map(jnp.asarray, raw_restored["model"]["params"])

    return params
