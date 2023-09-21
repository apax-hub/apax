from functools import partial
from pathlib import Path
from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from flax.core.frozen_dict import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax_md import partition, quantity, space

from apax.config import parse_config
from apax.model import ModelBuilder
from apax.train.eval import load_params
from apax.utils import jax_md_reduced


def stack_parameters(param_list):
    flat_param_list = []
    for params in param_list:
        params = unfreeze(params)
        flat_params = flatten_dict(params)
        flat_param_list.append(flat_params)

    stacked_flat_params = flat_params
    for p in flat_param_list[0].keys():
        stacked_flat_params[p] = jnp.stack(
            [flat_param[p] for flat_param in flat_param_list]
        )

    stacked_params = unflatten_dict(stacked_flat_params)
    stack_params = freeze(stacked_params)
    return stack_params


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


def build_energy_neighbor_fns(atoms, config, params, dr_threshold):
    atomic_numbers = jnp.asarray(atoms.numbers)
    box = jnp.asarray(atoms.get_cell().array, dtype=jnp.float32)
    box = box.T

    if np.all(box < 1e-6):
        displacement_fn, _ = space.free()
    else:
        displacement_fn, _ = space.periodic_general(box, fractional_coordinates=True)

    Z = jnp.asarray(atomic_numbers)
    n_species = 119  # int(np.max(Z) + 1)
    builder = ModelBuilder(config.model.get_dict(), n_species=n_species)
    model = builder.build_energy_derivative_model(
        apply_mask=True, init_box=np.array(box), inference_disp_fn=displacement_fn
    )
    energy_fn = maybe_vmap(model.apply, params, Z)
    neighbor_fn = jax_md_reduced.partition.neighbor_list(
        displacement_fn,
        box,
        config.model.r_max,
        dr_threshold,
        fractional_coordinates=True,
        disable_cell_list=True,
        format=partition.Sparse,
    )
    return energy_fn, neighbor_fn


def process_stress(results, box):
    V = quantity.volume(3, box)
    results = {
        # We should properly check whether CP2K uses the ASE cell convention
        # for tetragonal strain, it doesn't matter whether we transpose or not
        k: (val.T / V if k.startswith("stress") else val)
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
    DOES NOT SUPPORT CUTOFFS LARGER THAN MIN(BOX SIZE / 2)!
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
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.dr_threshold = dr_threshold
        self.is_ensemble = False
        self.transformations = transformations
        self.n_models = 1

        if isinstance(model_dir, Path) or isinstance(model_dir, str):
            self.params = self.restore_parameters(model_dir)
            if self.model_config.model.calc_stress:
                self.implemented_properties.extend(
                    [
                        "stress",
                    ]
                )
        elif isinstance(model_dir, list):
            self.n_models = len(model_dir)
            params = []
            for path in model_dir:
                params.append(self.restore_parameters(path))

            stacked_params = stack_parameters(params)
            self.params = stacked_params
            self.is_ensemble = True
            self.implemented_properties.extend(
                [
                    "energy_uncertainty",
                    "forces_uncertainty",
                ]
            )

            if self.model_config.model.calc_stress:
                self.implemented_properties.extend(
                    [
                        "stress",
                        "stress_uncertainty",
                    ]
                )
        else:
            raise NotImplementedError(
                "Please provide either a path or list of paths to trained models"
            )

        self.step = None
        self.neighbor_fn = None
        self.neighbors = None

    def restore_parameters(self, model_dir):
        self.model_config = parse_config(Path(model_dir) / "config.yaml")
        ckpt_dir = self.model_config.data.model_version_path()
        return load_params(ckpt_dir)

    def initialize(self, atoms):
        model, neighbor_fn = build_energy_neighbor_fns(
            atoms,
            self.model_config,
            self.params,
            self.dr_threshold,
        )

        if self.is_ensemble:
            model = make_ensemble(model)

        for transformation in self.transformations:
            model = transformation.apply(model, self.n_models)

        Z = jnp.asarray(atoms.numbers)

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

        self.step = step_fn
        self.neighbor_fn = neighbor_fn

    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
        if self.step is None or "numbers" in system_changes:
            self.initialize(atoms)
            self.neighbors = self.neighbor_fn.allocate(positions)

        results, self.neighbors = self.step(positions, self.neighbors, box)

        if self.neighbors.did_buffer_overflow:
            print("neighbor list overflowed, reallocating.")
            self.initialize(atoms)
            self.neighbors = self.neighbor_fn.allocate(positions)
            results, self.neighbors = self.step(positions, self.neighbors, box)

        self.results = {k: np.array(v, dtype=np.float64) for k, v in results.items()}
        self.results["energy"] = self.results["energy"].item()
