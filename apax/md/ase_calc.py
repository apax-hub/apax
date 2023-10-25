from functools import partial
from pathlib import Path
from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from flax.traverse_util import flatten_dict
from jax_md import partition, quantity, space
from matscipy.neighbours import neighbour_list

from apax.model import ModelBuilder
from apax.train.checkpoints import restore_parameters
from apax.utils import jax_md_reduced


def maybe_vmap(apply, params, Z):
    flat_params = flatten_dict(params)
    shapes = [v.shape[0] for v in flat_params.values()]
    is_ensemble = shapes == shapes[::-1]

    if is_ensemble:
        apply = jax.vmap(apply, in_axes=(0, None, None, None, None, None))

    # Maybe the partial mapping should happen at the very end of initialize
    # That way other functions can mae use of the parameter shape information
    energy_fn = partial(apply, params)
    return energy_fn


def build_energy_neighbor_fns(atoms, config, params, dr_threshold, neigbor_from_jax):
    r_max = config.model.r_max
    atomic_numbers = jnp.asarray(atoms.numbers)
    box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
    neigbor_from_jax = neighbor_calculable_with_jax(box, r_max)
    box = box.T
    displacement_fn = None
    neighbor_fn = None

    if neigbor_from_jax:
        if np.all(box < 1e-6):
            displacement_fn, _ = space.free()
        else:
            displacement_fn, _ = space.periodic_general(box, fractional_coordinates=True)

        neighbor_fn = jax_md_reduced.partition.neighbor_list(
            displacement_fn,
            box,
            config.model.r_max,
            dr_threshold,
            fractional_coordinates=True,
            disable_cell_list=True,
            format=partition.Sparse,
        )

    Z = jnp.asarray(atomic_numbers)
    n_species = 119  # int(np.max(Z) + 1)
    builder = ModelBuilder(config.model.get_dict(), n_species=n_species)

    model = builder.build_energy_derivative_model(
        apply_mask=True, init_box=np.array(box), inference_disp_fn=displacement_fn
    )

    energy_fn = maybe_vmap(model.apply, params, Z)

    return energy_fn, neighbor_fn


def process_stress(results, box):
    V = quantity.volume(3, box)
    results = {
        # We should properly check whether CP2K uses the ASE cell convention
        # for tetragonal strain, it doesn't matter whether we transpose or not
        k: val.T / V if k.startswith("stress") else val
        for k, val in results.items()
    }
    return results


def make_ensemble(model):
    def ensemble(positions, Z, idx, box, offsets):
        results = model(positions, Z, idx, box, offsets)
        uncertainty = {k + "_uncertainty": jnp.std(v, axis=0) for k, v in results.items()}
        results = {k: jnp.mean(v, axis=0) for k, v in results.items()}
        results.update(uncertainty)

        return results

    return ensemble


class ASECalculator(Calculator):
    """
    ASE Calculator for apax models.
    """

    implemented_properties = [
        "energy",
        "forces",
    ]

    def __init__(
        self,
        model_dir: Union[Path, list[Path]],
        dr_threshold: float = 0.5,
        transformations: Callable = [],
        padding_factor: float = 1.5,
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.dr_threshold = dr_threshold
        self.is_ensemble = False
        self.transformations = transformations
        self.n_models = 1 if isinstance(model_dir, (Path, str)) else len(model_dir)

        self.model_config, self.params = restore_parameters(model_dir)
        self.padding_factor = padding_factor

        if self.model_config.model.calc_stress:
            self.implemented_properties.append("stress")

        if self.n_models > 1:
            uncertainty_kws = [
                prop + "_uncertainty" for prop in self.implemented_properties
            ]
            self.implemented_properties += uncertainty_kws
            self.is_ensemble = True

        self.step = None
        self.neighbor_fn = None
        self.neighbors = None
        self.offsets = None

    def initialize(self, atoms):
        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
        self.r_max = self.model_config.model.r_max
        self.neigbor_from_jax = neighbor_calculable_with_jax(box, self.r_max)
        model, neighbor_fn = build_energy_neighbor_fns(
            atoms,
            self.model_config,
            self.params,
            self.dr_threshold,
            self.neigbor_from_jax,
        )

        if self.is_ensemble:
            model = make_ensemble(model)

        for transformation in self.transformations:
            model = transformation.apply(model, self.n_models)

        self.step = get_step_fn(model, atoms, self.neigbor_from_jax)
        self.neighbor_fn = neighbor_fn

    def set_neighbours_and_offsets(self, atoms, box):
        idxs_i, idxs_j, offsets = neighbour_list("ijS", atoms, self.r_max)

        if len(idxs_i) > self.padded_length:
            print("neighbor list overflowed, reallocating.")
            self.padded_length = int(len(idxs_i) * self.padding_factor)
            self.initialize(atoms)

        zeros_to_add = self.padded_length - len(idxs_i)

        self.neighbors = np.array([idxs_i, idxs_j], dtype=np.int32)
        self.neighbors = np.pad(self.neighbors, ((0, 0), (0, zeros_to_add)), "constant")

        offsets = np.matmul(offsets, box)
        self.offsets = np.pad(offsets, ((0, zeros_to_add), (0, 0)), "constant")

    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)

        # setup model and neighbours
        if self.step is None:
            self.initialize(atoms)

            if self.neigbor_from_jax:
                self.neighbors = self.neighbor_fn.allocate(positions)
            else:
                idxs_i = neighbour_list("i", atoms, self.r_max)
                self.padded_length = int(len(idxs_i) * self.padding_factor)

        elif "numbers" in system_changes:
            self.initialize(atoms)

            if self.neigbor_from_jax:
                self.neighbors = self.neighbor_fn.allocate(positions)

        elif "cell" in system_changes:
            neigbor_from_jax = neighbor_calculable_with_jax(box, self.r_max)
            if self.neigbor_from_jax != neigbor_from_jax:
                self.initialize(atoms)

        # predict
        if self.neigbor_from_jax:
            results, self.neighbors = self.step(positions, self.neighbors, box)

            if self.neighbors.did_buffer_overflow:
                print("neighbor list overflowed, reallocating.")
                self.initialize(atoms)
                self.neighbors = self.neighbor_fn.allocate(positions)

                results, self.neighbors = self.step(positions, self.neighbors, box)

        else:
            self.set_neighbours_and_offsets(atoms, box)
            positions = np.array(space.transform(np.linalg.inv(box), atoms.positions))

            results = self.step(positions, self.neighbors, box, self.offsets)

        self.results = {k: np.array(v, dtype=np.float64) for k, v in results.items()}
        self.results["energy"] = self.results["energy"].item()


def neighbor_calculable_with_jax(box, r_max):
    if np.all(box < 1e-6):
        return True
    else:
        # all lettice vector combinations to calculate all three plane distances
        a_vec_list = [box[0], box[0], box[1]]
        b_vec_list = [box[1], box[2], box[2]]
        c_vec_list = [box[2], box[1], box[0]]

        height = []
        for i in range(3):
            normvec = np.cross(a_vec_list[i], b_vec_list[i])
            projection = (
                c_vec_list[i]
                - np.sum(normvec * c_vec_list[i]) / np.sum(normvec**2) * normvec
            )
            height.append(np.linalg.norm(c_vec_list[i] - projection))

        if np.min(height) / 2 > r_max:
            return True
        else:
            return False


def get_step_fn(model, atoms, neigbor_from_jax):
    Z = jnp.asarray(atoms.numbers)
    if neigbor_from_jax:

        @jax.jit
        def step_fn(positions, neighbor, box):
            if np.any(atoms.get_cell().lengths() > 1e-6):
                box = box.T
                inv_box = jnp.linalg.inv(box)
                positions = space.transform(inv_box, positions)
                neighbor = neighbor.update(positions, box=box)
            else:
                neighbor = neighbor.update(positions)

            offsets = jnp.full([neighbor.idx.shape[1], 3], 0)
            results = model(positions, Z, neighbor.idx, box, offsets)

            if "stress" in results.keys():
                results = process_stress(results, box)

            return results, neighbor

    else:

        @jax.jit
        def step_fn(positions, neighbor, box, offsets):
            results = model(positions, Z, neighbor, box, offsets)

            if "stress" in results.keys():
                results = process_stress(results, box)

            return results

    return step_fn
