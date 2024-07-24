import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import freeze

from apax.optimizer import get_opt


def test_get_opt():
    params = {
        "my_linear": {"w": jnp.ones((3, 3)), "b": jnp.ones((3,))},
        "my_embed": {
            "atomic_type_embedding": jnp.ones((3, 3)),
        },
        "my_scale_shift": {
            "scale_per_element": jnp.ones((3,)),
            "shift_per_element": jnp.ones((3,)),
        },
    }
    params = freeze(params)

    grads = jax.tree_util.tree_map(lambda x: x * 0.01, tree=params)

    opt = get_opt(
        params,
        0,
        500,
        emb_lr=0.05,
        nn_lr=0.01,
        scale_lr=0.001,
        shift_lr=0.1,
        schedule={"name": "linear", "transition_begin": 0, "end_value": 1e-6},
    )
    opt_state = opt.init(params=params)

    updates, new_opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    filtered_params = jax.tree_util.tree_map(
        lambda v: v[0] if len(v.shape) == 1 else v[0, 0], new_params
    )
    flat_params, treedef = jax.tree_util.tree_flatten(filtered_params)

    assert jnp.allclose(flat_params[1], flat_params[2])
    assert flat_params[-2] > flat_params[1]
    assert flat_params[0] < flat_params[1]
    assert flat_params[-1] < flat_params[1]
