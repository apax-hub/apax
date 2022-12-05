import jax
import jax.numpy as jnp

from gmnn_jax.optimizer import get_opt
import optax


def test_get_opt():
    params = {
        "my_linear": {
            "w": jnp.ones((3,3)),
            "b": jnp.ones((3,))
        },
        "my_embed": {
            "atomic_type_embedding": jnp.ones((3,3)),
        },
        "my_scale_shift": {
            "scale": jnp.ones((3,)),
            "shift": jnp.ones((3,)),
        }
    }

    grads = jax.tree_util.tree_map(lambda x: x*0.01, tree=params)
    
    opt = get_opt(0, 500, emb_lr=0.01, nn_lr=0.05, scale_lr=0.001, shift_lr=0.1)
    opt_state = opt.init(params=params)

    updates, new_opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    filtered_params = jax.tree_util.tree_map(lambda v: v[0] if len(v.shape) == 1 else v[0,0], new_params)
    flat_params, treedef = jax.tree_util.tree_flatten(filtered_params)

    assert jnp.allclose(flat_params[1], flat_params[2])
    assert flat_params[-2] > flat_params[1]
    assert flat_params[0] < flat_params[1]
    assert flat_params[-1] < flat_params[1]



test_get_opt()





