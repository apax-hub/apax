import collections
import itertools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition, space
from matscipy.neighbours import neighbour_list
from tqdm import trange

from apax.utils import jax_md_reduced

log = logging.getLogger(__name__)


def initialize_nbr_fn(atoms, cutoff):
    neighbor_fn = None
    default_box = 100
    box = jnp.asarray(atoms.cell.array)

    if np.all(box < 1e-6):
        displacement_fn, _ = space.free()
        box = default_box

        neighbor_fn = jax_md_reduced.partition.neighbor_list(
            displacement_or_metric=displacement_fn,
            box=box,
            r_cutoff=cutoff,
            format=partition.Sparse,
            fractional_coordinates=False,
        )

    return neighbor_fn


@jax.jit
def extract_nl(neighbors, position):
    # vmapped neighborlist probably only useful for larger structures
    neighbors = neighbors.update(position)
    return neighbors


def dataset_neighborlist(
    positions: list[np.array],
    n_atoms: list[int],
    box: list[np.array],
    r_max: float,
    atoms_list,
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
    idxs :
        Neighbor list of all structures.
    """
    log.info("Precomputing neighborlists")
    # The JaxMD NL throws an error if np arrays are passed to it in the CPU version
    idx_list = []
    offset_list = []
    neighbors = None
    last_n_atoms = -1

    neighbor_fn = initialize_nbr_fn(
        atoms_list[0],
        r_max,
    )

    nl_pbar = trange(
        len(positions),
        desc="Precomputing NL",
        ncols=100,
        mininterval=0.25,
        disable=disable_pbar,
        leave=True,
    )
    for i, position in enumerate(positions):
        if np.all(box[i] < 1e-6):
            position = jnp.asarray(position)
            if n_atoms[i] != last_n_atoms:
                neighbors = neighbor_fn.allocate(position)
                last_n_atoms = n_atoms[i]

            neighbors = extract_nl(neighbors, position)

            if neighbors.did_buffer_overflow:
                log.info("Neighbor list overflowed, reallocating.")
                neighbors = neighbor_fn.allocate(position)

            neighbor_idxs = np.asarray(neighbors.idx)
            n_neighbors = neighbor_idxs.shape[1]
            offsets = np.full([n_neighbors, 3], 0)
        else:
            idxs_i, idxs_j, offsets = neighbour_list("ijS", atoms_list[i], r_max)
            offsets = np.matmul(offsets, box[i])
            neighbor_idxs = np.array([idxs_i, idxs_j], dtype=np.int32)

        offset_list.append(offsets)
        idx_list.append(neighbor_idxs)
        nl_pbar.update()
    nl_pbar.close()

    return idx_list, offset_list


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
