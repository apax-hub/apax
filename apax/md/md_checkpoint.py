import logging
import typing as t
from pathlib import Path

import orbax.checkpoint as ocp

log = logging.getLogger(__name__)


def load_md_state(state: t.Any, ckpt_dir: Path) -> tuple[t.Any, int]:
    try:
        log.info(f"loading MD state from {ckpt_dir}")
        target = {"state": state, "step": 0}
        with ocp.CheckpointManager(ckpt_dir) as mngr:
            restored_ckpt = mngr.restore(step=None, args=ocp.args.StandardRestore(target))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Checkpoint directory {ckpt_dir} does not exist or no checkpoint found."
        )
    state = restored_ckpt["state"]
    step = restored_ckpt["step"]
    return state, step
