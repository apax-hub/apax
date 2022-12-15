from pathlib import Path

from flax.training import checkpoints, train_state


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
        raw_restored = checkpoints.restore_checkpoint(ckpt_dir, target=target, step=None)
        state = raw_restored["model"]
        start_epoch = raw_restored["epoch"] + 1

    return state, start_epoch