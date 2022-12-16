import collections
import itertools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax_md.partition import NeighborFn

log = logging.getLogger(__name__)


@jax.jit
def extract_nl(neighbors, positions):
    # vmapped neighborlist probably only useful for larger structures
    neighbors = neighbors.update(positions)
    return neighbors


def dataset_neighborlist(
    neighbor_fn: NeighborFn, positions: np.array, n_atoms: list[int]
) -> list[int]:
    """Calculates the neighbor list of all systems within positions using
    a jax_md.partition.NeighborFn.

    Parameters
    ----------
    neighbor_fn :
        Neighbor list function (jax_md.partition.NeighborFn).
    positions :
        Cartesian coordinates of all atoms in all structures.
    n_atoms :
        List of number of Atoms per structure.

    Returns
    -------
    idx :
        Neighbor list of all structures.
    """
    log.info("Precomputing neighborlists")
    positions = [jnp.asarray(pos) for pos in positions]
    neighbors = neighbor_fn.allocate(positions[0])
    idx = []
    num_atoms = n_atoms[0]
    for i, position in enumerate(positions):
        if n_atoms[i] != num_atoms:
            neighbors = neighbor_fn.allocate(position)
            num_atoms = n_atoms[i]

        neighbors = extract_nl(neighbors, position)
        if neighbors.did_buffer_overflow:
            log.info("Neighbor list overflowed, reallocating.")
            neighbors = neighbor_fn.allocate(position)

        idx.append(neighbors.idx)

    return idx


def prefetch_to_single_device(iterator, size):
    """
    inspired by
    https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
    except it does not shard the data.
    """
    queue = collections.deque()

    def _prefetch(x):
        return jnp.asarray(x)

    def enqueue(n):
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree_util.tree_map(_prefetch, data))

    enqueue(size)
    while queue:
        yield queue.popleft()
        enqueue(1)
