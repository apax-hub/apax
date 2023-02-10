import collections
import itertools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax_md.partition import NeighborFn
from tqdm import trange

log = logging.getLogger(__name__)


@jax.jit
def extract_nl(neighbors, position):
    # vmapped neighborlist probably only useful for larger structures
    neighbors = neighbors.update(position)
    return neighbors


@jax.jit
def extract_periodic_nl(neighbors, position, box):
    # vmapped neighborlist probably only useful for larger structures
    neighbors = neighbors.update(position, box=box)
    return neighbors


def allocate_nl(neighbor_fn, position, box):
    if np.all(box < 1e-6):
        neighbors = neighbor_fn.allocate(position)
    else:
        neighbors = neighbor_fn.allocate(position, box=box)
    return neighbors


def dataset_neighborlist(
    neighbor_fn: NeighborFn,
    positions: list[np.array],
    n_atoms: list[int],
    box: list[np.array],
    disable_pbar: bool = False,
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
    # The JaxMD NL throws an error if np arrays are passed to it in the CPU version
    positions = [jnp.asarray(pos) for pos in positions]
    neighbors = neighbor_fn.allocate(positions[0])
    idx = []
    num_atoms = n_atoms[0]
        
    pbar_update_freq = 10
    with trange(
        len(positions),
        desc="Precomputing NL",
        ncols=100,
        disable=disable_pbar,
        leave=True,
    ) as nl_pbar:
        for i, position in enumerate(positions):
            if n_atoms[i] != num_atoms:
                neighbors = allocate_nl(neighbor_fn, position, box[i])
                num_atoms = n_atoms[i]

            if np.all(box[i] < 1e-6):
                neighbors = extract_nl(neighbors, position)
            else:
                neighbors = extract_periodic_nl(neighbors, position, box[i])

            if neighbors.did_buffer_overflow:
                log.info("Neighbor list overflowed, reallocating.")
                neighbors = allocate_nl(neighbor_fn, position, box[i])

            idx.append(neighbors.idx)
            if i % pbar_update_freq == 0:
                nl_pbar.update(pbar_update_freq)

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
