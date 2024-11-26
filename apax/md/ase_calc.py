from functools import partial
from pathlib import Path
from typing import Callable, Union

import ase
import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from flax.core.frozen_dict import freeze, unfreeze
from tqdm import trange
from vesin import NeighborList

from apax.data.input_pipeline import (
    CachedInMemoryDataset,
)
from apax.train.checkpoints import check_for_ensemble, restore_parameters
from apax.utils.jax_md_reduced import partition, quantity, space


def maybe_vmap(apply, params):
    n_models = check_for_ensemble(params)

    if n_models > 1:
        apply = jax.vmap(apply, in_axes=(0, None, None, None, None, None))

    # Maybe the partial mapping should happen at the very end of initialize
    # That way other functions can mae use of the parameter shape information
    energy_fn = partial(apply, params)
    return energy_fn


def build_energy_neighbor_fns(atoms, config, params, dr_threshold, neigbor_from_jax):
    r_max = config.model.basis.r_max
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

        neighbor_fn = partition.neighbor_list(
            displacement_fn,
            box,
            config.model.basis.r_max,
            dr_threshold,
            fractional_coordinates=True,
            disable_cell_list=True,
            format=partition.Sparse,
        )

    n_species = 119  # int(np.max(Z) + 1)
    Builder = config.model.get_builder()
    builder = Builder(config.model.model_dump(), n_species=n_species)

    model = builder.build_energy_derivative_model(
        apply_mask=True, init_box=np.array(box), inference_disp_fn=displacement_fn
    )

    energy_fn = maybe_vmap(model.apply, params)
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
        ensemble = {k + "_ensemble": v for k, v in results.items()}
        results = {k: jnp.mean(v, axis=0) for k, v in results.items()}
        if "forces_ensemble" in ensemble.keys():
            ensemble["forces_ensemble"] = jnp.transpose(
                ensemble["forces_ensemble"], (1, 2, 0)
            )
        if "forces_ensemble" in ensemble.keys():
            ensemble["stress_ensemble"] = jnp.transpose(
                ensemble["forces_ensemble"], (1, 2, 0)
            )
        results.update(uncertainty)
        results.update(ensemble)

        return results

    return ensemble


def unpack_results(results, inputs):
    n_structures = len(results["energy"])
    unpacked_results = []
    for i in range(n_structures):
        single_results = jax.tree_map(lambda x: x[i], results)
        single_results["energy"] = single_results["energy"].item()
        for k, v in single_results.items():
            if "forces" in k:
                single_results[k] = v[: inputs["n_atoms"][i]]
        unpacked_results.append(single_results)
    return unpacked_results


class ASECalculator(Calculator):
    """
    ASE Calculator for apax models.
    Always implements energy and force predictions.
    Stress predictions and corresponding uncertainties are added to
    `implemented_properties` based on whether the stress flag is set
    in the model config and whether a model ensemble is loaded.

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
        **kwargs,
    ):
        """
        Parameters
        ----------
        model_dir:
            Path to a model directory of the form `.../directory/experiment`
            (see Config docs for details).
            If a list of model paths is provided, they will be ensembled.
        dr_threshold:
            Neighborlist skin for the JaxMD neighborlist.
        transformations:
            Function transformations applied on top of the EnergyDerivativeModel.
            Transfomrations are implemented under `apax.md.transformations`.
        padding_factor:
            Multiple of the fallback vesin's amount of neighbors.
            This NL will be padded to `len(neighbors) * padding_factor`
            on NL initialization.
        """
        Calculator.__init__(self, **kwargs)
        self.dr_threshold = dr_threshold
        self.transformations = transformations

        self.model_config, self.params = restore_parameters(model_dir)
        self.n_models = check_for_ensemble(self.params)
        self.padding_factor = padding_factor
        self.padded_length = 0

        if self.model_config.model.calc_stress:
            self.implemented_properties.append("stress")

        if self.n_models > 1:
            uncertainty_kws = [
                prop + "_uncertainty" for prop in self.implemented_properties
            ]
            self.implemented_properties += uncertainty_kws

        self.step = None
        self.neighbor_fn = None
        self.neighbors = None
        self.offsets = None
        self.model = None

    def initialize(self, atoms):
        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
        self.r_max = self.model_config.model.basis.r_max
        self.neigbor_from_jax = neighbor_calculable_with_jax(box, self.r_max)
        model, neighbor_fn = build_energy_neighbor_fns(
            atoms,
            self.model_config,
            self.params,
            self.dr_threshold,
            self.neigbor_from_jax,
        )

        if self.n_models > 1:
            model = make_ensemble(model)

        for transformation in self.transformations:
            model = transformation.apply(model)

        self.model = model
        self.step = get_step_fn(model, atoms, self.neigbor_from_jax)
        self.neighbor_fn = neighbor_fn

        if self.neigbor_from_jax:
            positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
            if np.any(atoms.get_cell().lengths() > 1e-6):
                box = atoms.cell.array.T
                inv_box = jnp.linalg.inv(box)
                positions = space.transform(inv_box, positions)  # frac coords
                self.neighbors = self.neighbor_fn.allocate(positions, box=box)
            else:
                self.neighbors = self.neighbor_fn.allocate(positions)
        else:
            calculator = NeighborList(cutoff=self.r_max, full_list=True)
            idxs_i, _, _ = calculator.compute(
                points=atoms.positions,
                box=atoms.cell.array,
                periodic=np.any(atoms.pbc),
                quantities="ijS",
            )
            self.padded_length = int(len(idxs_i) * self.padding_factor)

    def set_neighbours_and_offsets(self, atoms, box):
        calculator = NeighborList(cutoff=self.r_max, full_list=True)
        idxs_i, idxs_j, offsets = calculator.compute(
            points=atoms.positions,
            box=atoms.cell.array,
            periodic=np.any(atoms.pbc),
            quantities="ijS",
        )
        if len(idxs_i) > self.padded_length:
            print("neighbor list overflowed, extending.")
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
                results, self.neighbors = self.step(positions, self.neighbors, box)

        else:
            self.set_neighbours_and_offsets(atoms, box)
            positions = np.array(space.transform(np.linalg.inv(box), atoms.positions))

            results = self.step(positions, self.neighbors, box, self.offsets)

        self.results = {k: np.array(v, dtype=np.float64) for k, v in results.items()}
        self.results["energy"] = self.results["energy"].item()

    def batch_eval(
        self, atoms_list: list[ase.Atoms], batch_size: int = 64, silent: bool = False
    ) -> list[ase.Atoms]:
        """Evaluate the model on a list of Atoms. This is preferable to assigning
        the calculator to each Atoms instance for 2 reasons:
        1. Processing can be abtched, which is advantageous for larger datasets.
        2. Inputs are padded so no recompilation is triggered when evaluating
        differently sized systems.

        Parameters
        ----------
        atoms_list :
            List of Atoms to be evaluated.
        batch_size:
            Processing batch size. Does not affect results,
            only speed and memory requirements.
        silent:
            Whether or not to suppress progress bars.

        Returns
        -------
        evaluated_atoms_list:
            List of Atoms with labels predicted by the model.
        """

        init_box = atoms_list[0].cell.array

        dataset = CachedInMemoryDataset(
            atoms_list,
            self.model_config.model.basis.r_max,
            batch_size,
            n_epochs=1,
            ignore_labels=True,
        )

        evaluated_atoms_list = []
        n_data = dataset.n_data
        ds = dataset.batch()

        Builder = self.model_config.model.get_builder()
        builder = Builder(self.model_config.model.model_dump())
        model = builder.build_energy_derivative_model(
            apply_mask=True,
            init_box=init_box,
        ).apply

        if self.n_models > 1:
            model = jax.vmap(model, in_axes=(0, None, None, None, None, None))

        model = partial(model, self.params)

        if self.n_models > 1:
            model = make_ensemble(model)

        for transformation in self.transformations:
            model = transformation.apply(model)

        model = jax.vmap(model, in_axes=(0, 0, 0, 0, 0))
        model = jax.jit(model)

        pbar = trange(
            n_data, desc="Evaluating data", ncols=100, leave=False, disable=silent
        )
        for i, inputs in enumerate(ds):
            results = model(
                inputs["positions"],
                inputs["numbers"],
                inputs["idx"],
                inputs["box"],
                inputs["offsets"],
            )
            unpadded_results = unpack_results(results, inputs)

            # for the last batch, the number of structures may be less
            # than the batch_size,  which is why we check this explicitly
            num_strucutres_in_batch = results["energy"].shape[0]
            for j in range(num_strucutres_in_batch):
                atoms = atoms_list[i * batch_size + j].copy()
                atoms.calc = SinglePointCalculator(atoms=atoms, **unpadded_results[j])
                evaluated_atoms_list.append(atoms)
            pbar.update(batch_size)
        pbar.close()
        dataset.cleanup()
        return evaluated_atoms_list

    @property
    def ll_weights(self):
        dense_layers = list(
            self.params["params"]["energy_model"]["atomistic_model"]["readout"].keys()
        )
        llweights = self.params["params"]["energy_model"]["atomistic_model"]["readout"][
            dense_layers[-1]
        ]["w"]
        return np.asarray(llweights)

    def set_ll_weights(self, new_weights):
        params = unfreeze(self.params)
        dense_layers = list(
            params["params"]["energy_model"]["atomistic_model"]["readout"].keys()
        )
        params["params"]["energy_model"]["atomistic_model"]["readout"][dense_layers[-1]][
            "w"
        ] = jnp.asarray(new_weights, dtype=jnp.float32)
        self.params = freeze(params)
        self.step = None


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
