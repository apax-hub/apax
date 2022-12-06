import jax.numpy as jnp
import numpy as np

from gmnn_jax.train.loss import (
    Loss,
    force_angle_div_force_label,
    force_angle_exponential_weight,
    force_angle_loss,
    weighted_squared_error,
)


def test_weighted_squared_error():
    energy_label = jnp.array([[0.1, 0.4, 0.2, -0.5], [0.1, -0.1, 0.8, 0.6]])

    loss = weighted_squared_error(energy_label, energy_label, divisor=1.0)
    loss = jnp.sum(loss)
    ref = 0.0
    assert loss.shape == ()
    assert abs(loss - ref) < 1e-6

    pred = jnp.array(
        [
            [0.6, 0.4, 0.2, -0.5],
            [0.1, -0.1, 0.8, 0.6],
        ]
    )
    loss = weighted_squared_error(energy_label, pred, divisor=1.0)
    loss = jnp.sum(loss)
    ref = 0.25
    assert abs(loss - ref) < 1e-6

    loss = weighted_squared_error(energy_label, pred, divisor=2.0)
    loss = jnp.sum(loss)
    ref = 0.125
    assert abs(loss - ref) < 1e-6


def test_force_angle_loss():
    F_pred = jnp.array(
        [
            [
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0],
            ]
        ]
    )

    F_0 = jnp.array(
        [
            [
                [0.5, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.9, 0.0, 0.0],
            ]
        ]
    )

    F_angle_loss = force_angle_loss(F_pred, F_0)
    F_angle_loss = jnp.arccos(-F_angle_loss + 1) * 360 / (2 * np.pi)
    assert F_angle_loss.shape == (1, 5)
    ref = jnp.array([0.0, 0.0, 45.0, 90.0, 90.0])
    assert jnp.all(abs(F_angle_loss - ref) < 1e-5)

    F_angle_loss = force_angle_div_force_label(F_pred, F_0)
    assert F_angle_loss.shape == (1, 5)

    F_angle_loss = force_angle_exponential_weight(F_pred, F_0)
    assert F_angle_loss.shape == (1, 5)


def test_force_loss():
    key = "forces"
    loss_type = "structures"
    weight = 1
    label = {
        key: jnp.array(
            [
                [
                    [0.4, 0.2, 0.5],
                    [0.3, 0.8, 0.1],
                ]
            ]
        ),
        "n_atoms": jnp.array([[2]]),
    }

    pred = {
        key: jnp.array(
            [
                [
                    [0.4, 0.2, 0.5],
                    [0.3, 0.8, 0.1],
                ]
            ]
        ),
    }
    loss_func = Loss(key=key, loss_type=loss_type, weight=weight)
    loss = loss_func(label=label, prediction=pred)
    ref_loss = 0.0
    assert loss.shape == ()
    assert abs(loss - ref_loss) < 1e-6

    pred = {
        key: jnp.array(
            [
                [
                    [0.4, 0.2, 0.5],
                    [0.3, 0.8, 0.6],
                ]
            ]
        ),
    }
    loss_func = Loss(key=key, loss_type=loss_type, weight=weight)
    loss = loss_func(label=label, prediction=pred)
    ref_loss = 0.125
    assert abs(loss - ref_loss) < 1e-6

    weight = 2
    loss_func = Loss(key=key, loss_type=loss_type, weight=weight)
    loss = loss_func(label=label, prediction=pred)
    ref_loss = 0.25
    assert abs(loss - ref_loss) < 1e-6
