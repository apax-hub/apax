# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from enum import Enum, IntEnum
from functools import partial
from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp
from jax import eval_shape, jit, lax, vmap
from jax.core import ShapedArray

from apax.utils.jax_md_reduced import dataclasses, space, util

Array = util.Array
PyTree = Any
f32 = util.f32
f64 = util.f64

i32 = util.i32
i64 = util.i64

Box = space.Box
DisplacementOrMetricFn = space.DisplacementOrMetricFn
MetricFn = space.MetricFn
MaskFn = Callable[[Array], Array]


def _cell_size(box, minimum_cell_size) -> Array:
    cells_per_side = jnp.floor(box / minimum_cell_size)
    return box / cells_per_side


def _fractional_cell_size(box, cutoff):
    if jnp.isscalar(box) or box.ndim == 0:
        return cutoff / box
    elif box.ndim == 1:
        return cutoff / jnp.min(box)
    elif box.ndim == 2:
        if box.shape[0] == 1:
            return 1 / jnp.floor(box[0, 0] / cutoff)
        elif box.shape[0] == 2:
            xx = box[0, 0]
            yy = box[1, 1]
            xy = box[0, 1] / yy

            nx = xx / jnp.sqrt(1 + xy**2)
            ny = yy

            nmin = jnp.floor(jnp.min(jnp.array([nx, ny])) / cutoff)
            nmin = jnp.where(nmin == 0, 1, nmin)
            return 1 / nmin
        elif box.shape[0] == 3:
            xx = box[0, 0]
            yy = box[1, 1]
            zz = box[2, 2]
            xy = box[0, 1] / yy
            xz = box[0, 2] / zz
            yz = box[1, 2] / zz

            nx = xx / jnp.sqrt(1 + xy**2 + (xy * yz - xz) ** 2)
            ny = yy / jnp.sqrt(1 + yz**2)
            nz = zz

            nmin = jnp.floor(jnp.min(jnp.array([nx, ny, nz])) / cutoff)
            nmin = jnp.where(nmin == 0, 1, nmin)
            return 1 / nmin
        else:
            raise ValueError(
                f"Expected box to be either 1-, 2-, or 3-dimensional found {box.shape[0]}"
            )
    else:
        raise ValueError(
            "Expected box to be either a scalar, a vector, or a "
            f"matrix. Found {type(box)}."
        )


def _displacement_or_metric_to_metric_sq(
    displacement_or_metric: DisplacementOrMetricFn,
) -> MetricFn:
    """Checks whether or not a displacement or metric was provided."""
    for dim in range(1, 4):
        try:
            R = ShapedArray((dim,), f32)
            dR_or_dr = eval_shape(displacement_or_metric, R, R, t=0)
            if len(dR_or_dr.shape) == 0:
                return (
                    lambda Ra, Rb, **kwargs: displacement_or_metric(Ra, Rb, **kwargs) ** 2
                )
            else:
                return lambda Ra, Rb, **kwargs: space.square_distance(
                    displacement_or_metric(Ra, Rb, **kwargs)
                )
        except TypeError:
            continue
        except ValueError:
            continue
    raise ValueError(
        "Canonicalize displacement not implemented for spatial dimension largerthan 4."
    )


# Neighbor Lists


@dataclasses.dataclass
class CellList:
    """Stores the spatial partition of a system into a cell list.

    See :meth:`cell_list` for details on the construction / specification.
    Cell list buffers all have a common shape, S, where
      * `S = [cell_count_x, cell_count_y, cell_capacity]`
      * `S = [cell_count_x, cell_count_y, cell_count_z, cell_capacity]`
    in two- and three-dimensions respectively. It is assumed that each cell has
    the same capacity.

    Attributes:
      position_buffer: An ndarray of floating point positions with shape
        `S + [spatial_dimension]`.
      id_buffer: An ndarray of int32 particle ids of shape `S`. Note that empty
        slots are specified by `id = N` where `N` is the number of particles in
        the system.
      named_buffer: A dictionary of ndarrays of shape `S + [...]`. This contains
        side data placed into the cell list.
      did_buffer_overflow: A boolean specifying whether or not the cell list
        exceeded the maximum allocated capacity.
      cell_capacity: An integer specifying the maximum capacity of each cell in
        the cell list.
      update_fn: A function that updates the cell list at a fixed capacity.
    """

    position_buffer: Array
    id_buffer: Array
    named_buffer: Dict[str, Array]

    did_buffer_overflow: Array

    cell_capacity: int = dataclasses.static_field()
    cell_size: float = dataclasses.static_field()

    update_fn: Callable[..., "CellList"] = dataclasses.static_field()

    def update(self, position: Array, **kwargs) -> "CellList":
        cl_data = (self.cell_capacity, self.did_buffer_overflow, self.update_fn)
        return self.update_fn(position, cl_data, **kwargs)

    @property
    def kwarg_buffers(self):
        logging.warning(
            "kwarg_buffers renamed to named_buffer. The name "
            "kwarg_buffers will be deprecated."
        )
        return self.named_buffer


class PartitionErrorCode(IntEnum):
    """An enum specifying different error codes.

    Attributes:
      NONE: Means that no error was encountered during simulation.
      NEIGHBOR_LIST_OVERFLOW: Indicates that the neighbor list was not large
        enough to contain all of the particles. This should indicate that it is
        necessary to allocate a new neighbor list.
      CELL_LIST_OVERFLOW: Indicates that the cell list was not large enough to
        contain all of the particles. This should indicate that it is necessary
        to allocate a new cell list.
      CELL_SIZE_TOO_SMALL: Indicates that the size of cells in a cell list was
        not large enough to properly capture particle interactions. This
        indicates that it is necessary to allocate a new cell list with larger
        cells.
      MALFORMED_BOX: Indicates that a box matrix was not properly upper
        triangular.
    """

    NONE = 0
    NEIGHBOR_LIST_OVERFLOW = 1 << 0
    CELL_LIST_OVERFLOW = 1 << 1
    CELL_SIZE_TOO_SMALL = 1 << 2
    MALFORMED_BOX = 1 << 3


PEC = PartitionErrorCode


@dataclasses.dataclass
class PartitionError:
    """A struct containing error codes while building / updating neighbor lists.

    Attributes:
      code: An array storing the error code. See `PartitionErrorCode` for
        details.
    """

    code: Array

    def update(self, bit: bytes, pred: Array) -> Array:
        """Possibly adds an error based on a predicate."""
        zero = jnp.zeros((), jnp.uint8)
        bit = jnp.array(bit, dtype=jnp.uint8)
        return PartitionError(self.code | jnp.where(pred, bit, zero))


class NeighborListFormat(Enum):
    """An enum listing the different neighbor list formats.

    Attributes:
      Dense: A dense neighbor list where the ids are a square matrix
        of shape `(N, max_neighbors_per_atom)`. Here the capacity of the neighbor
        list must scale with the highest connectivity neighbor.
      Sparse: A sparse neighbor list where the ids are a rectangular
        matrix of shape `(2, max_neighbors)` specifying the start / end particle
        of each neighbor pair.
      OrderedSparse: A sparse neighbor list whose format is the same as `Sparse`
        where only bonds with i < j are included.
    """

    Dense = 0
    Sparse = 1
    OrderedSparse = 2


@dataclasses.dataclass
class NeighborList:
    """A struct containing the state of a Neighbor List.

    Attributes:
      idx: For an N particle system this is an `[N, max_occupancy]` array of
        integers such that `idx[i, j]` is the j-th neighbor of particle i.
      reference_position: The positions of particles when the neighbor list was
        constructed. This is used to decide whether the neighbor list ought to be
        updated.
      error: An error code that is used to identify errors that occurred during
        neighbor list construction. See `PartitionError` and `PartitionErrorCode`
        for details.
      cell_list_capacity: An optional integer specifying the capacity of the cell
        list used as an intermediate step in the creation of the neighbor list.
      max_occupancy: A static integer specifying the maximum size of the
        neighbor list. Changing this will invoke a recompilation.
      format: A NeighborListFormat enum specifying the format of the neighbor
        list.
      cell_size: A float specifying the current minimum size of the cells used
        in cell list construction.
      cell_list_fn: The function used to construct the cell list.
      update_fn: A static python function used to update the neighbor list.
    """

    idx: Array
    reference_position: Array
    error: PartitionError
    cell_list_capacity: Optional[int] = dataclasses.static_field()
    max_occupancy: int = dataclasses.static_field()

    format: NeighborListFormat = dataclasses.static_field()
    cell_size: Optional[float] = dataclasses.static_field()
    cell_list_fn: Callable[[Array, CellList], CellList] = dataclasses.static_field()
    update_fn: Callable[[Array, "NeighborList"], "NeighborList"] = (
        dataclasses.static_field()
    )

    def update(self, position: Array, **kwargs) -> "NeighborList":
        return self.update_fn(position, self, **kwargs)

    @property
    def did_buffer_overflow(self) -> bool:
        return self.error.code & (PEC.NEIGHBOR_LIST_OVERFLOW | PEC.CELL_LIST_OVERFLOW)

    @property
    def cell_size_too_small(self) -> bool:
        return self.error.code & PEC.CELL_SIZE_TOO_SMALL

    @property
    def malformed_box(self) -> bool:
        return self.error.code & PEC.MALFORMED_BOX

    def __str__(self) -> str:
        """Produces a string representation of the error code."""
        if not jnp.any(self.code):
            return ""

        if jnp.any(self.code & PEC.NEIGHBOR_LIST_OVERFLOW):
            return "Partition Error: Neighbor list buffer overflow."

        if jnp.any(self.code & PEC.CELL_LIST_OVERFLOW):
            return "Partition Error: Cell list buffer overflow"

        if jnp.any(self.code & PEC.CELL_SIZE_TOO_SMALL):
            return "Partition Error: Cell size too small"

        if jnp.any(self.code & PEC.MALFORMED_BOX):
            return "Partition Error: Incorrect box format. Expecting upper triangular."

        raise ValueError(f"Unexpected Error Code {self.code}.")

    __repr__ = __str__


@dataclasses.dataclass
class NeighborList:
    """A struct containing the state of a Neighbor List.

    Attributes:
      idx: For an N particle system this is an `[N, max_occupancy]` array of
        integers such that `idx[i, j]` is the j-th neighbor of particle i.
      reference_position: The positions of particles when the neighbor list was
        constructed. This is used to decide whether the neighbor list ought to be
        updated.
      error: An error code that is used to identify errors that occurred during
        neighbor list construction. See `PartitionError` and `PartitionErrorCode`
        for details.
      cell_list_capacity: An optional integer specifying the capacity of the cell
        list used as an intermediate step in the creation of the neighbor list.
      max_occupancy: A static integer specifying the maximum size of the
        neighbor list. Changing this will invoke a recompilation.
      format: A NeighborListFormat enum specifying the format of the neighbor
        list.
      cell_size: A float specifying the current minimum size of the cells used
        in cell list construction.
      cell_list_fn: The function used to construct the cell list.
      update_fn: A static python function used to update the neighbor list.
    """

    idx: Array
    reference_position: Array
    error: PartitionError
    cell_list_capacity: Optional[int] = dataclasses.static_field()
    max_occupancy: int = dataclasses.static_field()

    format: NeighborListFormat = dataclasses.static_field()
    cell_size: Optional[float] = dataclasses.static_field()
    cell_list_fn: Callable[[Array, CellList], CellList] = dataclasses.static_field()
    update_fn: Callable[[Array, "NeighborList"], "NeighborList"] = (
        dataclasses.static_field()
    )

    def update(self, position: Array, **kwargs) -> "NeighborList":
        return self.update_fn(position, self, **kwargs)

    @property
    def did_buffer_overflow(self) -> bool:
        return self.error.code & (PEC.NEIGHBOR_LIST_OVERFLOW | PEC.CELL_LIST_OVERFLOW)

    @property
    def cell_size_too_small(self) -> bool:
        return self.error.code & PEC.CELL_SIZE_TOO_SMALL

    @property
    def malformed_box(self) -> bool:
        return self.error.code & PEC.MALFORMED_BOX


def is_sparse(fmt: NeighborListFormat) -> bool:
    return fmt is NeighborListFormat.Sparse or fmt is NeighborListFormat.OrderedSparse


def is_format_valid(fmt: NeighborListFormat):
    if fmt not in list(NeighborListFormat):
        raise ValueError(
            f"Neighbor list format must be a member of NeighborListFormat found {fmt}."
        )


def is_box_valid(box: Array) -> bool:
    if jnp.isscalar(box) or box.ndim == 0 or box.ndim == 1:
        return True
    if box.ndim == 2:
        return jnp.triu(box) == box
    return False


@dataclasses.dataclass
class NeighborListFns:
    """A struct containing functions to allocate and update neighbor lists.

    Attributes:
      allocate: A function to allocate a new neighbor list. This function cannot
        be compiled, since it uses the values of positions to infer the shapes.
      update: A function to update a neighbor list given a new set of positions
        and a previously allocated neighbor list.
    """

    allocate: Callable[..., NeighborList] = dataclasses.static_field()
    update: Callable[[Array, NeighborList], NeighborList] = dataclasses.static_field()

    def __call__(
        self,
        position: Array,
        neighbors: Optional[NeighborList] = None,
        extra_capacity: int = 0,
        **kwargs,
    ) -> NeighborList:
        """A function for backward compatibility with previous neighbor lists.

        Args:
          position: An `(N, dim)` array of particle positions.
          neighbors: An optional neighbor list object. If it is provided then
            the function updates the neighbor list, otherwise it allocates a new
            neighbor list.
          extra_capacity: Extra capacity to add if allocating the neighbor list.
        Returns:
          A neighbor list object.
        """
        logging.warning(
            "Using a deprecated code path to create / update neighbor "
            "lists. It will be removed in a later version of JAX MD. "
            "Using `neighbor_fn.allocate` and `neighbor_fn.update` "
            "is preferred."
        )
        if neighbors is None:
            return self.allocate(position, extra_capacity, **kwargs)
        return self.update(position, neighbors, **kwargs)

    def __iter__(self):
        return iter((self.allocate, self.update))


NeighborFn = Callable[[Array, Optional[NeighborList], Optional[int]], NeighborList]


# def cell_list(box_size: Box,
#               minimum_cell_size: float,
#               buffer_size_multiplier: float = 1.25
#               ):
#   r"""Returns a function that partitions point data spatially.

#   Given a set of points :math:`\{x_i \in R^d\}` with associated data
#   :math:`\{k_i \in R^m\}` it is often useful to partition the points / data
#   spatially. A simple partitioning that can be implemented efficiently within
#   XLA is a dense partition into a uniform grid called a cell list.

#   Since XLA requires that shapes be statically specified inside of a JIT block,
#   the cell list code can operate in two modes: allocation and update.

#   Allocation creates a new cell list that uses a set of input positions to
#   estimate the capacity of the cell list. This capacity can be adjusted by
#   setting the `buffer_size_multiplier` or setting the `extra_capacity`.
#   Allocation cannot be JIT.

#   Updating takes a previously allocated cell list and places a new set of
#   particles in the cells. Updating cannot resize the cell list and is therefore
#   compatible with JIT. However, if the configuration has changed substantially
#   it is possible that the existing cell list won't be large enough to
#   accommodate all of the particles. In this case the `did_buffer_overflow` bit
#   will be set to True.

#   Args:
#     box_size: A float or an ndarray of shape `[spatial_dimension]` specifying
#       the size of the system. Note, this code is written for the case where the
#       boundaries are periodic. If this is not the case, then the current code
#       will be slightly less efficient.
#     minimum_cell_size: A float specifying the minimum side length of each cell.
#       Cells are enlarged so that they exactly fill the box.
#     buffer_size_multiplier: A floating point multiplier that multiplies the
#       estimated cell capacity to allow for fluctuations in the maximum cell
#       occupancy.
#   Returns:
#     A `CellListFns` object that contains two methods, one to allocate the cell
#     list and one to update the cell list. The update function can be called
#     with either a cell list from which the capacity can be inferred or with
#     an explicit integer denoting the capacity. Note that an existing cell list
#     can also be updated by calling `cell_list.update(position)`.
#   """

#   return None #CellListFns(allocate_fn, update_fn)  # pytype: disable=wrong-arg-count


def neighbor_list(
    displacement_or_metric: DisplacementOrMetricFn,
    box: Box,
    r_cutoff: float,
    dr_threshold: float = 0.0,
    capacity_multiplier: float = 1.25,
    disable_cell_list: bool = False,
    mask_self: bool = True,
    custom_mask_function: Optional[MaskFn] = None,
    fractional_coordinates: bool = False,
    format: NeighborListFormat = NeighborListFormat.Dense,
    **static_kwargs,
) -> NeighborFn:
    """Returns a function that builds a list neighbors for collections of points.

    Neighbor lists must balance the need to be jit compatible with the fact that
    under a jit the maximum number of neighbors cannot change (owing to static
    shape requirements). To deal with this, our `neighbor_list` returns a
    `NeighborListFns` object that contains two functions: 1)
    `neighbor_fn.allocate` create a new neighbor list and 2) `neighbor_fn.update`
    updates an existing neighbor list. Neighbor lists themselves additionally
    have a convenience `update` member function.

    Note that allocation of a new neighbor list cannot be jit compiled since it
    uses the positions to infer the maximum number of neighbors (along with
    additional space specified by the `capacity_multiplier`). Updating the
    neighbor list can be jit compiled; if the neighbor list capacity is not
    sufficient to store all the neighbors, the `did_buffer_overflow` bit
    will be set to `True` and a new neighbor list will need to be reallocated.

    Here is a typical example of a simulation loop with neighbor lists:

    .. code-block:: python

       init_fn, apply_fn = simulate.nve(energy_fn, shift, 1e-3)
       exact_init_fn, exact_apply_fn = simulate.nve(exact_energy_fn, shift, 1e-3)

       nbrs = neighbor_fn.allocate(R)
       state = init_fn(random.PRNGKey(0), R, neighbor_idx=nbrs.idx)

       def body_fn(i, state):
         state, nbrs = state
         nbrs = nbrs.update(state.position)
         state = apply_fn(state, neighbor_idx=nbrs.idx)
         return state, nbrs

       step = 0
       for _ in range(20):
         new_state, nbrs = lax.fori_loop(0, 100, body_fn, (state, nbrs))
         if nbrs.did_buffer_overflow:
           nbrs = neighbor_fn.allocate(state.position)
         else:
           state = new_state
           step += 1

    Args:
      displacement: A function `d(R_a, R_b)` that computes the displacement
        between pairs of points.
      box: Either a float specifying the size of the box, an array of
        shape `[spatial_dim]` specifying the box size for a cubic box in each
        spatial dimension, or a matrix of shape `[spatial_dim, spatial_dim]` that
        is _upper triangular_ and specifies the lattice vectors of the box.
      r_cutoff: A scalar specifying the neighborhood radius.
      dr_threshold: A scalar specifying the maximum distance particles can move
        before rebuilding the neighbor list.
      capacity_multiplier: A floating point scalar specifying the fractional
        increase in maximum neighborhood occupancy we allocate compared with the
        maximum in the example positions.
      disable_cell_list: An optional boolean. If set to `True` then the neighbor
        list is constructed using only distances. This can be useful for
        debugging but should generally be left as `False`.
      mask_self: An optional boolean. Determines whether points can consider
        themselves to be their own neighbors.
      custom_mask_function: An optional function. Takes the neighbor array
        and masks selected elements. Note: The input array to the function is
        `(n_particles, m)` where the index of particle 1 is in index in the first
        dimension of the array, the index of particle 2 is given by the value in
        the array
      fractional_coordinates: An optional boolean. Specifies whether positions
        will be supplied in fractional coordinates in the unit cube, :math:`[0, 1]^d`.
        If this is set to True then the `box_size` will be set to `1.0` and the
        cell size used in the cell list will be set to `cutoff / box_size`.
      format: The format of the neighbor list; see the :meth:`NeighborListFormat` enum
        for details about the different choices for formats. Defaults to `Dense`.
      **static_kwargs: kwargs that get threaded through the calculation of
        example positions.
    Returns:
      A NeighborListFns object that contains a method to allocate a new neighbor
      list and a method to update an existing neighbor list.
    """
    is_format_valid(format)
    box = lax.stop_gradient(box)
    r_cutoff = lax.stop_gradient(r_cutoff)
    dr_threshold = lax.stop_gradient(dr_threshold)
    box = f32(box)

    cutoff = r_cutoff + dr_threshold
    cutoff_sq = cutoff**2
    threshold_sq = (dr_threshold / f32(2)) ** 2
    metric_sq = _displacement_or_metric_to_metric_sq(displacement_or_metric)

    @jit
    def candidate_fn(position: Array) -> Array:
        candidates = jnp.arange(position.shape[0], dtype=i32)
        return jnp.broadcast_to(
            candidates[None, :], (position.shape[0], position.shape[0])
        )

    # @jit
    # def cell_list_candidate_fn(cl: CellList, position: Array) -> Array:
    #     N, dim = position.shape

    #     idx = cl.id_buffer

    #     cell_idx = [idx]

    #     for dindex in _neighboring_cells(dim):
    #         if onp.all(dindex == 0):
    #             continue
    #         cell_idx += [_shift_array(idx, dindex)]

    #     cell_idx = jnp.concatenate(cell_idx, axis=-2)
    #     cell_idx = cell_idx[..., jnp.newaxis, :, :]
    #     cell_idx = jnp.broadcast_to(cell_idx, idx.shape[:-1] + cell_idx.shape[-2:])

    #     def copy_values_from_cell(value, cell_value, cell_id):
    #         scatter_indices = jnp.reshape(cell_id, (-1,))
    #         cell_value = jnp.reshape(cell_value, (-1,) + cell_value.shape[-2:])
    #         return value.at[scatter_indices].set(cell_value)

    #     neighbor_idx = jnp.zeros((N + 1,) + cell_idx.shape[-2:], i32)
    #     neighbor_idx = copy_values_from_cell(neighbor_idx, cell_idx, idx)
    #     return neighbor_idx[:-1, :, 0]

    @jit
    def mask_self_fn(idx: Array) -> Array:
        self_mask = idx == jnp.reshape(
            jnp.arange(idx.shape[0], dtype=i32), (idx.shape[0], 1)
        )
        return jnp.where(self_mask, idx.shape[0], idx)

    @jit
    def prune_neighbor_list_dense(position: Array, idx: Array, **kwargs) -> Array:
        d = partial(metric_sq, **kwargs)
        d = space.map_neighbor(d)

        N = position.shape[0]
        neigh_position = position[idx]
        dR = d(position, neigh_position)

        mask = (dR < cutoff_sq) & (idx < N)
        out_idx = N * jnp.ones(idx.shape, i32)

        cumsum = jnp.cumsum(mask, axis=1)
        index = jnp.where(mask, cumsum - 1, idx.shape[1] - 1)
        p_index = jnp.arange(idx.shape[0])[:, None]
        out_idx = out_idx.at[p_index, index].set(idx)
        max_occupancy = jnp.max(cumsum[:, -1])

        return out_idx, max_occupancy

    @jit
    def prune_neighbor_list_sparse(position: Array, idx: Array, **kwargs) -> Array:
        d = partial(metric_sq, **kwargs)
        d = space.map_bond(d)

        N = position.shape[0]
        # explicit i32 type
        sender_idx = jnp.broadcast_to(jnp.arange(N, dtype=i32)[:, None], idx.shape)

        sender_idx = jnp.reshape(sender_idx, (-1,))
        receiver_idx = jnp.reshape(idx, (-1,))
        dR = d(position[sender_idx], position[receiver_idx])

        mask = (dR < cutoff_sq) & (receiver_idx < N)
        if format is NeighborListFormat.OrderedSparse:
            mask = mask & (receiver_idx < sender_idx)

        out_idx = N * jnp.ones(receiver_idx.shape, i32)

        cumsum = jnp.cumsum(mask)
        index = jnp.where(mask, cumsum - 1, len(receiver_idx) - 1)
        receiver_idx = out_idx.at[index].set(receiver_idx)
        sender_idx = out_idx.at[index].set(sender_idx)
        max_occupancy = cumsum[-1]

        return jnp.stack((receiver_idx, sender_idx)), max_occupancy

    def neighbor_list_fn(
        position: Array, neighbors=None, extra_capacity: int = 0, **kwargs
    ) -> NeighborList:
        def neighbor_fn(position_and_error, max_occupancy=None):
            position, err = position_and_error
            N = position.shape[0]

            cl_fn = None
            cl = None
            # cell_size = None
            # if not disable_cell_list:
            #     if neighbors is None:
            #         _box = kwargs.get("box", box)
            #         cell_size = cutoff
            #         if fractional_coordinates:
            #             err = err.update(PEC.MALFORMED_BOX, is_box_valid(_box))
            #             cell_size = _fractional_cell_size(_box, cutoff)
            #             _box = 1.0
            #         if jnp.all(cell_size < _box / 3.0):
            #             cl_fn = cell_list(_box, cell_size, capacity_multiplier)
            #             cl = cl_fn.allocate(position, extra_capacity=extra_capacity)
            #     else:
            #         cell_size = neighbors.cell_size
            #         cl_fn = neighbors.cell_list_fn
            #         if cl_fn is not None:
            #             cl = cl_fn.update(position, neighbors.cell_list_capacity)

            if cl is None:
                cl_capacity = None
                idx = candidate_fn(position)
            else:
                raise NotImplementedError(
                    "Cell lists not implemented in apax' reduced jaxmd"
                )
                # err = err.update(PEC.CELL_LIST_OVERFLOW, cl.did_buffer_overflow)
                # idx = cell_list_candidate_fn(cl, position)
                # cl_capacity = cl.cell_capacity

            if mask_self:
                idx = mask_self_fn(idx)
            if custom_mask_function is not None:
                idx = custom_mask_function(idx)

            if is_sparse(format):
                idx, occupancy = prune_neighbor_list_sparse(position, idx, **kwargs)
            else:
                idx, occupancy = prune_neighbor_list_dense(position, idx, **kwargs)

            if max_occupancy is None:
                _extra_capacity = (
                    extra_capacity if not is_sparse(format) else N * extra_capacity
                )
                max_occupancy = int(occupancy * capacity_multiplier + _extra_capacity)
                if max_occupancy > idx.shape[-1]:
                    max_occupancy = idx.shape[-1]
                if not is_sparse(format):
                    capacity_limit = N - 1 if mask_self else N
                elif format is NeighborListFormat.Sparse:
                    capacity_limit = N * (N - 1) if mask_self else N**2
                else:
                    capacity_limit = N * (N - 1) // 2
                if max_occupancy > capacity_limit:
                    max_occupancy = capacity_limit
            idx = idx[:, :max_occupancy]
            update_fn = neighbor_list_fn if neighbors is None else neighbors.update_fn
            return NeighborList(
                idx,
                position,
                err.update(PEC.NEIGHBOR_LIST_OVERFLOW, occupancy > max_occupancy),
                cl_capacity,
                max_occupancy,
                format,
                0,  # cell_size
                cl_fn,
                update_fn,
            )  # pytype: disable=wrong-arg-count

        nbrs = neighbors
        if nbrs is None:
            return neighbor_fn((position, PartitionError(jnp.zeros((), jnp.uint8))))

        neighbor_fn = partial(neighbor_fn, max_occupancy=nbrs.max_occupancy)

        # If the box has been updated, then check that fractional coordinates are
        # enabled and that the cell list has big enough cells.
        if "box" in kwargs:
            if not fractional_coordinates:
                raise ValueError(
                    "Neighbor list cannot accept a box keyword argument "
                    "if fractional_coordinates is not enabled."
                )

            # Added conditional
            if not disable_cell_list:
                # `cell_size` is really the minimum cell size.
                cur_cell_size = _cell_size(1.0, nbrs.cell_size)
                new_cell_size = _cell_size(
                    1.0, _fractional_cell_size(kwargs["box"], cutoff)
                )
                err = nbrs.error.update(
                    PEC.CELL_SIZE_TOO_SMALL, new_cell_size > cur_cell_size
                )
                err = err.update(PEC.MALFORMED_BOX, is_box_valid(kwargs["box"]))
                nbrs = dataclasses.replace(nbrs, error=err)

        d = partial(metric_sq, **kwargs)
        d = vmap(d)
        return lax.cond(
            jnp.any(d(position, nbrs.reference_position) > threshold_sq),
            (position, nbrs.error),
            neighbor_fn,
            nbrs,
            lambda x: x,
        )

    def allocate_fn(position: Array, extra_capacity: int = 0, **kwargs):
        return neighbor_list_fn(position, extra_capacity=extra_capacity, **kwargs)

    def update_fn(position: Array, neighbors, **kwargs):
        return neighbor_list_fn(position, neighbors, **kwargs)

    return NeighborListFns(allocate_fn, update_fn)  # pytype: disable=wrong-arg-count


Dense = NeighborListFormat.Dense
Sparse = NeighborListFormat.Sparse
OrderedSparse = NeighborListFormat.OrderedSparse
