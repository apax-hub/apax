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

    angle_error = cosine_sim(inputs={}, label=label, prediction=prediction, key="forces")
    assert angle_error.shape == ()
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
    assert abs(epoch_metrics["energy_per_atom_mse"] - 0.25) < 1e-6
    assert abs(epoch_metrics["energy_rmse"] - jnp.sqrt(0.5)) < 1e-6
    assert abs(epoch_metrics["forces_mae"] - 1 / 3) < 1e-6
    assert abs(epoch_metrics["forces_mse"] - 1 / 3) < 1e-6
