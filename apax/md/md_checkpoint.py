import logging

from flax.training import checkpoints

log = logging.getLogger(__name__)


def load_md_state(sim_dir):
    # TODO: not functional yet
    try:
        log.info("loading previous md state")
        raw_restored = checkpoints.restore_checkpoint(sim_dir, target=None, step=None)
    except FileNotFoundError:
        print(f"No checkpoint found at {sim_dir}")
    state = raw_restored["state"]
    step = raw_restored["step"]
    return state, step
