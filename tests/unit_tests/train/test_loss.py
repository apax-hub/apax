import jax.numpy as jnp
import numpy as np

from apax.train.loss import (
    Loss,
    force_angle_div_force_label,
    force_angle_exponential_weight,
    force_angle_loss,
    weighted_squared_error,
)


def test_weighted_squared_error():
    name = "energy"
    label = {"energy": jnp.array([[0.1, 0.4, 0.2, -0.5], [0.1, -0.1, 0.8, 0.6]])}

    loss = weighted_squared_error(label, label, name)
    loss = jnp.sum(loss)
    ref = 0.0
    assert loss.shape == ()
    assert abs(loss - ref) < 1e-6

    pred = {
        "energy": jnp.array(
            [
                [0.6, 0.4, 0.2, -0.5],
                [0.1, -0.1, 0.8, 0.6],
            ]
        )
    }
    loss = weighted_squared_error(label, pred, name)
    loss = jnp.sum(loss)
    ref = 0.25
    assert abs(loss - ref) < 1e-6


def test_force_angle_loss():
    name = "forces"
    F_pred = {
        "forces": jnp.array(
            [
                [
                    [0.5, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.0],  # padding
                ]
            ]
        )
    }

    F_0 = {
        "forces": jnp.array(
            [
                [
                    [0.5, 0.0, 0.0],
                    [0.9, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.9, 0.0, 0.0],
                    [0.0, 0.0, 0.0],  # padding
                ]
            ]
        )
    }

    F_angle_loss = force_angle_loss(F_pred, F_0, name)
    F_angle_loss = jnp.arccos(-F_angle_loss + 1) * 360 / (2 * np.pi)
    assert F_angle_loss.shape == (1, 6)
    ref = jnp.array([0.0, 0.0, 45.0, 90.0, 90.0, 90.0])
    assert jnp.allclose(F_angle_loss, ref)

    F_angle_loss = force_angle_div_force_label(F_pred, F_0, name)
    assert F_angle_loss.shape == (1, 6)

    F_angle_loss = force_angle_exponential_weight(F_pred, F_0, name)
    assert F_angle_loss.shape == (1, 6)


def test_force_loss():
    name = "forces"
    loss_type = "mse"
    weight = 1
    inputs = {
        "n_atoms": jnp.array([2]),
    }
    label = {
        name: jnp.array(
            [
                [
                    [0.4, 0.2, 0.5],
                    [0.3, 0.8, 0.1],
                ]
            ]
        ),
    }

    pred = {
        name: jnp.array(
            [
                [
                    [0.4, 0.2, 0.5],
                    [0.3, 0.8, 0.1],
                ]
            ]
        ),
    }
    loss_func = Loss(name=name, loss_type=loss_type, weight=weight)
    loss = loss_func(inputs=inputs, label=label, prediction=pred)
    ref_loss = 0.0
    assert loss.shape == ()
    assert abs(loss - ref_loss) < 1e-6

    pred = {
        name: jnp.array(
            [
                [
                    [0.4, 0.2, 0.5],
                    [0.3, 0.8, 0.6],
                ]
            ]
        ),
    }
    loss_func = Loss(name=name, loss_type=loss_type, weight=weight)
    loss = loss_func(inputs=inputs, label=label, prediction=pred)
    ref_loss = 0.125
    assert abs(loss - ref_loss) < 1e-6

    weight = 2
    loss_func = Loss(name=name, loss_type=loss_type, weight=weight)
    loss = loss_func(inputs=inputs, label=label, prediction=pred)
    ref_loss = 0.25
    assert abs(loss - ref_loss) < 1e-6
