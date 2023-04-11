import collections
import itertools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from ase.neighborlist import PrimitiveNeighborList
from jax_md.partition import NeighborFn
from tqdm import trange

log = logging.getLogger(__name__)


@jax.jit
def extract_nl(neighbors, position):
    # vmapped neighborlist probably only useful for larger structures
    neighbors = neighbors.update(position)
    return neighbors


def dataset_neighborlist(
    neighbor_fn: NeighborFn,
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
    positions = [jnp.asarray(pos) for pos in positions]
    neighbors = neighbor_fn.allocate(positions[0])
    idx_list = []
    offset_list = []
    last_n_atoms = n_atoms[0]
    neighbors_dict = {
        "neighbor_fn_0": {"neighbors": neighbors, "box": box[0], "n_atoms": n_atoms[0]}
    }

    pbar_update_freq = 10
    with trange(
        len(positions),
        desc="Precomputing NL",
        ncols=100,
        disable=disable_pbar,
        leave=True,
    ) as nl_pbar:
        for i, position in enumerate(positions):
            if np.all(box[i] < 1e-6):
                if n_atoms[i] != last_n_atoms:
                    neighbors = neighbor_fn.allocate(position)
                    last_n_atoms = n_atoms[i]

                neighbors = extract_nl(neighbors, position)

                if neighbors.did_buffer_overflow:
                    log.info("Neighbor list overflowed, reallocating.")
                    neighbors = neighbor_fn.allocate(position)

                neighbor_idxs = neighbors.idx
                n_neighbors = neighbor_idxs.shape[1]
                offsets = jnp.full([n_neighbors, 3], 0)

            elif np.all(box[i] > 2 * r_max):
                reallocate = True
                for neighbor_vals in neighbors_dict.values():
                    if (
                        np.all(box[i] == neighbor_vals["box"])
                        and n_atoms[i] == neighbor_vals["n_atoms"]
                    ):
                        neighbors = extract_nl(neighbor_vals["neighbors"], position)
                        reallocate = False
                if reallocate:
                    neighbors = neighbor_fn.allocate(position, box=box[i])
                    neighbors_dict[f"neighbor_fn_{i}"] = {
                        "neighbors": neighbors,
                        "box": box[i],
                        "n_atoms": n_atoms[i],
                    }
                if neighbors.did_buffer_overflow:
                    log.info("Neighbor list overflowed, reallocating.")
                    neighbors = neighbor_fn.allocate(position, box=box[i])

                neighbor_idxs = neighbors.idx
                n_neighbors = neighbor_idxs.shape[1]
                offsets = jnp.full([n_neighbors, 3], 0)

            else:
                cell = [
                    [box[i][0], 0.0, 0.0],
                    [0.0, box[i][1], 0.0],
                    [0.0, 0.0, box[i][2]],
                ]
                ase_neighbor_fn = PrimitiveNeighborList(
                    jnp.full(n_atoms[i], r_max / 2),
                    skin=0.0,
                    self_interaction=False,
                    bothways=True,
                )  # dict comparison possible like in jax_nl
                ase_neighbor_fn.update(
                    pbc=[True, True, True], cell=cell, coordinates=atoms_list[i].positions
                )
                idxs_i = []
                idxs_j = []
                offsets = []

                for atom_idx in range(n_atoms[i]):
                    idx, offset = ase_neighbor_fn.get_neighbors(atom_idx)
                    idxs_i.extend([atom_idx] * len(idx))
                    idxs_j.extend(idx)
                    offsets.extend(offset)
                neighbor_idxs = jnp.array([idxs_i, idxs_j])
                offsets = jnp.array(offset)

            idx_list.append(neighbor_idxs)
            offset_list.append(offsets)
            if i % pbar_update_freq == 0:
                nl_pbar.update(pbar_update_freq)
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
