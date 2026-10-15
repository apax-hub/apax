import jax.numpy as jnp

from apax.config.train_config import MetricsConfig
from apax.train.metrics import cosine_sim, initialize_metrics


def test_cosine_sim():
    prediction = {
        "forces": jnp.array(
            [
                [
                    [0.5, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                ]
            ]
        )
    }

    label = {
        "forces": jnp.array(
            [
                [
                    [0.0, 0.5, 0.0],
                    [0.5, 0.0, 0.0],
                ]
            ]
        )
    }

    inputs = {"numbers": jnp.array([[1, 1]])}
    out = cosine_sim(inputs=inputs, label=label, prediction=prediction, key="forces")
    angle_error = out["values"][out["mask"]].mean()
    ref = 0.5
    assert abs(angle_error - ref) < 1e-6


def test_initialize_metrics_collection():
    prediction = {
        "energy": jnp.array([1.0, 1.0]),
        "forces": jnp.array(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ]
        ),
    }

    label = {
        "energy": jnp.array([1.0, 2.0]),
        "forces": jnp.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            ]
        ),
    }
    inputs = {
        "n_atoms": jnp.array([2, 2]),
        "numbers": jnp.array([[1, 1], [1, 1]]),
    }
    metrics_list = [
        MetricsConfig(
            name="energy", reductions=["mae", "rmse", "per_atom_mae", "per_atom_mse"]
        ),
        MetricsConfig(name="forces", reductions=["mae", "mse"]),
    ]
    Metrics = initialize_metrics(metrics_list)
    batch_metrics = Metrics.single_from_model_output(
        inputs=inputs, label=label, prediction=prediction
    )

    epoch_metrics = batch_metrics.compute()

    assert abs(epoch_metrics["energy_mae"] - 0.5) < 1e-6
    assert abs(epoch_metrics["energy_per_atom_mae"] - 0.25) < 1e-6
    assert abs(epoch_metrics["energy_per_atom_mse"] - 0.125) < 1e-6
    assert abs(epoch_metrics["energy_rmse"] - jnp.sqrt(0.5)) < 1e-6
    assert abs(epoch_metrics["forces_mae"] - 1 / 3) < 1e-6
    assert abs(epoch_metrics["forces_mse"] - 1 / 3) < 1e-6


def test_metrics_ignore_padding_and_weight_batches():
    """Padded atom slots must not dilute force metrics, and batches merge by count."""
    Metrics = initialize_metrics([MetricsConfig(name="forces", reductions=["mae", "mse"])])
    zeros = jnp.zeros((1, 4, 3))
    # batch 1: one real atom (error 3 in x) padded to 4 slots
    b1 = Metrics.single_from_model_output(
        inputs={"n_atoms": jnp.array([1]), "numbers": jnp.array([[6, 0, 0, 0]])},
        label={"forces": zeros.at[0, 0, 0].set(3.0)},
        prediction={"forces": zeros},
    )
    # batch 2: four real atoms, zero error
    b2 = Metrics.single_from_model_output(
        inputs={"n_atoms": jnp.array([4]), "numbers": jnp.array([[6, 6, 6, 6]])},
        label={"forces": zeros},
        prediction={"forces": zeros},
    )
    m = b1.merge(b2).compute()
    # 15 real components, one with |err| = 3
    assert abs(m["forces_mae"] - 3 / 15) < 1e-6
    assert abs(m["forces_mse"] - 9 / 15) < 1e-6
