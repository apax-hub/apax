import jax.numpy as jnp
import numpy as np

from gmnn_jax.layers.masking import mask_by_atom, mask_by_neighbor

def test_mask_by_atom():

    n_atoms = 5
    n_padding = 3
    n_total = n_atoms + n_padding
    Z = jnp.concatenate([np.ones(n_atoms), np.zeros(n_padding)])
    atomic_prediction = jnp.ones((1, n_total))

    assert (jnp.sum(atomic_prediction) - n_total) < 1e-6

    masked_prediction = mask_by_atom(atomic_prediction, Z)

    assert (jnp.sum(masked_prediction) - n_atoms) < 1e-6


def test_mask_by_neighbor():
    pass



if __name__ == "__main__":
    print("hello")
    test_mask_by_atom()