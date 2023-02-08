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
def extract_nl(neighbors, positions, cell=None): #mit **kwargs lösen if cell in kwargs
    # vmapped neighborlist probably only useful for larger structures
    if cell is not None:
        neighbors = neighbors.update(positions, box=cell)
    else:
        neighbors = neighbors.update(positions)
    return neighbors


def dataset_neighborlist(
    neighbor_fn: NeighborFn,
    positions: np.array,
    n_atoms: list[int],
    cells: np.array = None,
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
    if cells is not None:
        cells = jnp.asarray(cells)
        
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
                neighbors = neighbor_fn.allocate(position)
                num_atoms = n_atoms[i]

            if cells is not None:
                neighbors = extract_nl(neighbors, position, cells[i])
            else:
                neighbors = extract_nl(neighbors, position)
            if neighbors.did_buffer_overflow:
                log.info("Neighbor list overflowed, reallocating.")
                neighbors = neighbor_fn.allocate(position)

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
