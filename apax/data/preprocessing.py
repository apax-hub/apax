import collections
import itertools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from matscipy.neighbours import neighbour_list

log = logging.getLogger(__name__)


def compute_nl(positions, box, r_max):
    """
    Computes the neighbor list for a single structure.
    For periodic systems, positions are assumed to be in
    fractional coordinates.

    Parameters
    ----------
    positions : np.ndarray
        Positions of atoms.
    box : np.ndarray
        Simulation box dimensions.
    r_max : float
        Maximum interaction radius.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing neighbor indices array and offsets array.

    """
    if np.all(box < 1e-6):
        box, box_origin = get_shrink_wrapped_cell(positions)
        idxs_i, idxs_j = neighbour_list(
            "ij",
            positions=positions,
            cutoff=r_max,
            cell=box,
            cell_origin=box_origin,
            pbc=[False, False, False],
        )

        neighbor_idxs = np.array([idxs_i, idxs_j], dtype=np.int16)

        n_neighbors = neighbor_idxs.shape[1]
        offsets = np.full([n_neighbors, 3], 0)

    else:
        positions = positions @ box
        idxs_i, idxs_j, offsets = neighbour_list(
            "ijS", positions=positions, cutoff=r_max, cell=box, pbc=[True, True, True]
        )
        neighbor_idxs = np.array([idxs_i, idxs_j], dtype=np.int16)
        offsets = np.matmul(offsets, box)
    return neighbor_idxs, offsets


def get_shrink_wrapped_cell(positions):
    """
    Get the shrink-wrapped simulation cell based on atomic positions.

    Parameters
    ----------
    positions : np.ndarray
        Atomic positions.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the shrink-wrapped cell matrix and origin.
    """
    rmin = np.min(positions, axis=0)
    rmax = np.max(positions, axis=0)
    cell_origin = rmin
    cell = np.diag(rmax - rmin)
    for idx in range(3):
        if cell[idx, idx] < 10e-1:
            cell[idx, idx] = 1.0

    cell[np.diag_indices_from(cell)] += 1

    return cell, cell_origin


def prefetch_to_single_device(iterator, size: int, sharding=None, n_step_jit=False):
    """
    inspired by
    https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
    except it does not shard the data.
    """
    queue = collections.deque()

    if sharding:
        n_devices = len(sharding._devices)
        slice_start = 1
        shape = [n_devices]
        if n_step_jit:
            # replicate over multi-batch axis
            # data shape: njit x bs x ...
            slice_start = 2
            shape.insert(0, 1)

    def _prefetch(x: jax.Array):
        if sharding:
            remaining_axes = [1] * len(x.shape[slice_start:])
            final_shape = tuple(shape + remaining_axes)
            x = jax.device_put(x, sharding.reshape(final_shape))
        else:
            x = jnp.asarray(x)
        return x

    def enqueue(n):
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree_util.tree_map(_prefetch, data))

    enqueue(size)
    while queue:
        yield queue.popleft()
        enqueue(1)
