import logging
import typing as t
from pathlib import Path

import orbax.checkpoint as ocp

log = logging.getLogger(__name__)


def load_md_state(state: t.Any, ckpt_dir: Path) -> tuple[t.Any, int]:
    try:
        log.info(f"loading MD state from {ckpt_dir}")
        target = {"state": state, "step": 0}
        restored_ckpt = ocp.CheckpointManager(ckpt_dir).restore(
            step=None, args=ocp.args.StandardRestore(), items=target
        )
    except FileNotFoundError:
        print(f"No checkpoint found at {ckpt_dir}")
    state = restored_ckpt["state"]
    step = restored_ckpt["step"]
    return state, step
