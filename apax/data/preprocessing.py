import collections
import itertools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from matscipy.neighbours import neighbour_list

log = logging.getLogger(__name__)


def compute_nl(position, box, r_max):
    if np.all(box < 1e-6):
        cell, cell_origin = get_shrink_wrapped_cell(position)
        idxs_i, idxs_j = neighbour_list(
            "ij",
            positions=position,
            cutoff=r_max,
            cell=cell,
            cell_origin=cell_origin,
            pbc=[False, False, False],
        )

        neighbor_idxs = np.array([idxs_i, idxs_j], dtype=np.int32)

        n_neighbors = neighbor_idxs.shape[1]
        offsets = np.full([n_neighbors, 3], 0)

    else:
        idxs_i, idxs_j, offsets = neighbour_list(
            "ijS",
            positions=position,
            cutoff=r_max,
            cell=box,
        )
        neighbor_idxs = np.array([idxs_i, idxs_j], dtype=np.int32)
        offsets = np.matmul(offsets, box)
    return neighbor_idxs, offsets


def get_shrink_wrapped_cell(positions):
    rmin = np.min(positions, axis=0)
    rmax = np.max(positions, axis=0)
    cell_origin = rmin
    cell = np.diag(rmax - rmin)
    for idx in range(3):
        if cell[idx, idx] < 10e-1:
            cell[idx, idx] = 1.0

    cell[np.diag_indices_from(cell)] += 1

    return cell, cell_origin


def prefetch_to_single_device(iterator, size: int):
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
