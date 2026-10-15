import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from apax.config.train_config import WeightAverage
from apax.train.checkpoints import load_state
from apax.train.parameters import EMAParameters


def test_ema_per_step_average():
    ema = EMAParameters(ema_start=1, decay=0.5)
    ema.update({"w": jnp.array(4.0)}, epoch=0)  # before ema_start: follows the raw params
    ema.update({"w": jnp.array(8.0)}, epoch=0)
    assert float(ema.ema_params["w"]) == 8.0
    ema.update({"w": jnp.array(0.0)}, epoch=1)
    ema.update({"w": jnp.array(0.0)}, epoch=1)
    assert np.isclose(float(ema.ema_params["w"]), 2.0)  # 8 * 0.5 * 0.5


def test_ema_checkpoint_roundtrip(tmp_path):
    import orbax.checkpoint as ocp

    def make_state(w):
        return train_state.TrainState.create(
            apply_fn=None, params={"w": jnp.array(w)}, tx=optax.sgd(0.1)
        )

    raw, average = make_state(1.0), {"w": jnp.array(3.0)}
    ckpt = {"model": raw.replace(params=average), "train_params": raw.params, "epoch": 4}
    with ocp.CheckpointManager((tmp_path / "latest").resolve()) as mngr:
        mngr.save(4, args=ocp.args.StandardSave(ckpt))

    ema = EMAParameters(ema_start=0)
    state, start_epoch = load_state(make_state(0.0), tmp_path / "latest", ema)
    assert start_epoch == 5
    assert float(state.params["w"]) == 1.0  # training resumes from the raw params
    assert float(ema.ema_params["w"]) == 3.0


def test_old_alpha_config_still_loads():
    wa = WeightAverage.model_validate({"ema_start": 2, "alpha": 0.9})
    assert wa.ema_start == 2 and wa.decay == 0.999
