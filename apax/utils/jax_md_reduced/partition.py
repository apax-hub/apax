from functools import partial
from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as onp
from jax import jit, lax, vmap
from jax_md import dataclasses, space, util
from jax_md.partition import (
    PEC,
    CellList,
    NeighborFn,
    NeighborList,
    NeighborListFns,
    NeighborListFormat,
    PartitionError,
    _cell_size,
    _displacement_or_metric_to_metric_sq,
    _fractional_cell_size,
    _neighboring_cells,
    _shift_array,
    cell_list,
    is_box_valid,
    is_format_valid,
    is_sparse,
)

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
    **static_kwargs
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

    @jit
    def cell_list_candidate_fn(cl: CellList, position: Array) -> Array:
        N, dim = position.shape

        idx = cl.id_buffer

        cell_idx = [idx]

        for dindex in _neighboring_cells(dim):
            if onp.all(dindex == 0):
                continue
            cell_idx += [_shift_array(idx, dindex)]

        cell_idx = jnp.concatenate(cell_idx, axis=-2)
        cell_idx = cell_idx[..., jnp.newaxis, :, :]
        cell_idx = jnp.broadcast_to(cell_idx, idx.shape[:-1] + cell_idx.shape[-2:])

        def copy_values_from_cell(value, cell_value, cell_id):
            scatter_indices = jnp.reshape(cell_id, (-1,))
            cell_value = jnp.reshape(cell_value, (-1,) + cell_value.shape[-2:])
            return value.at[scatter_indices].set(cell_value)

        neighbor_idx = jnp.zeros((N + 1,) + cell_idx.shape[-2:], i32)
        neighbor_idx = copy_values_from_cell(neighbor_idx, cell_idx, idx)
        return neighbor_idx[:-1, :, 0]

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
            cell_size = None
            if not disable_cell_list:
                if neighbors is None:
                    _box = kwargs.get("box", box)
                    cell_size = cutoff
                    if fractional_coordinates:
                        err = err.update(PEC.MALFORMED_BOX, is_box_valid(_box))
                        cell_size = _fractional_cell_size(_box, cutoff)
                        _box = 1.0
                    if jnp.all(cell_size < _box / 3.0):
                        cl_fn = cell_list(_box, cell_size, capacity_multiplier)
                        cl = cl_fn.allocate(position, extra_capacity=extra_capacity)
                else:
                    cell_size = neighbors.cell_size
                    cl_fn = neighbors.cell_list_fn
                    if cl_fn is not None:
                        cl = cl_fn.update(position, neighbors.cell_list_capacity)

            if cl is None:
                cl_capacity = None
                idx = candidate_fn(position)
            else:
                err = err.update(PEC.CELL_LIST_OVERFLOW, cl.did_buffer_overflow)
                idx = cell_list_candidate_fn(cl, position)
                cl_capacity = cl.cell_capacity

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
                cell_size,
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
