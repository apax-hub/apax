import jax
import jax.numpy as jnp


@jax.jit
def extract_nl(neighbors, positions):
    # vmapped neighborlist probably only useful for larger structures
    neighbors = neighbors.update(positions)
    return neighbors


def dataset_neighborlist(neighbor_fn, positions, extra_capacity=5):
    num_data = positions.shape[0]

    neighbors = neighbor_fn.allocate(positions[0], extra_capacity=extra_capacity)

    idx = []
    for i in range(0, num_data):
        neighbors = extract_nl(neighbors, positions[i])
        if neighbors.did_buffer_overflow:
            print("Neighbor list overflowed, reallocating.")
            neighbors = neighbor_fn.allocate(positions[i])

        idx.append(neighbors.idx)

    # TODO this currently doesn't work if the NL needs to be reallocated
    # idx needs to be padded before stacking
    # To Do this, I need to figure out how padding / masking works in jaxmd
    return jnp.stack(idx, axis=0)
