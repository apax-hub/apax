import glob

from flax.training import checkpoints


def look_for_checkpoints(directory):
    checkpoints_exist = glob.glob(f"{directory}/checkpoint*")
    if checkpoints_exist:
        return True
    else:
        return False


def load_md_state(sim_dir):
    # TODO: not functional yet
    raw_restored = checkpoints.restore_checkpoint(sim_dir, target=None, step=None)
    state = raw_restored["state"]
    step = raw_restored["step"]
    return state, step
