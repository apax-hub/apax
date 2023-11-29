import logging
from pathlib import Path

from flax.training import checkpoints
from jaxtyping import PyTree

log = logging.getLogger(__name__)


def load_md_state(state: PyTree, ckpt_dir: Path) -> tuple[PyTree, int]:
    try:
        log.info(f"loading MD state from {ckpt_dir}")
        target = {"state": state, "step": 0}
        restored_ckpt = checkpoints.restore_checkpoint(ckpt_dir, target=target, step=None)
    except FileNotFoundError:
        print(f"No checkpoint found at {ckpt_dir}")
    state = restored_ckpt["state"]
    step = restored_ckpt["step"]
    return state, step
