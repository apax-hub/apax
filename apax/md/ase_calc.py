from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import ase
import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes, all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from jax import tree_util
from tqdm import tqdm, trange
from vesin import NeighborList

from apax.config.train_config import Config
from apax.data.input_pipeline import CachedInMemoryDataset, OTFInMemoryDataset
from apax.md.function_transformations import ProcessStress
from apax.train.checkpoints import (
    canonicalize_energy_model_parameters,
    check_for_ensemble,
    restore_parameters,
)
from apax.utils.jax_md_reduced import partition, space


def maybe_vmap(apply: Callable, params: FrozenDict) -> Callable:
    n_models = check_for_ensemble(params)

    if n_models > 1:
        apply = jax.vmap(apply, in_axes=(0, None, None, None, None, None))

    # Maybe the partial mapping should happen at the very end of initialize
    # That way other functions can mae use of the parameter shape information
    energy_fn = partial(apply, params)
    return energy_fn


def build_energy_neighbor_fns(
    atoms: ase.Atoms,
    config: Config,
    params: FrozenDict,
    dr_threshold: float,
    neigbor_from_jax: bool,
) -> Tuple[Callable, Callable]:
    r_max = config.model.basis.r_max
    box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
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


def make_ensemble(model: Callable) -> Callable:
    def ensemble(
        positions: jnp.ndarray,
        Z: jnp.ndarray,
        idx: jnp.ndarray,
        box: jnp.ndarray,
        offsets: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
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


def unpack_results(
    results: Dict[str, np.ndarray], inputs: Dict[str, np.ndarray]
) -> List[Dict[str, Any]]:
    n_structures = len(results["energy"])
    unpacked_results = []
    for i in range(n_structures):
        single_results = tree_util.tree_map(lambda x: x[i], results)
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

    implemented_properties: List[str] = [
        "energy",
        "forces",
    ]

    def __init__(
        self,
        model_dir: Union[Path, List[Path]],
        dr_threshold: float = 0.5,
        transformations: List[Callable] = [],
        padding_factor: float = 1.5,
        **kwargs,
    ) -> None:
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

        for head in self.model_config.model.property_heads:
            self.implemented_properties.append(head.name)

        self.n_models = check_for_ensemble(self.params)
        self.padding_factor = padding_factor
        self.padded_length = 0

        self._update_implemented_properties()

        self.step: Union[Callable, None] = None
        self.neighbor_fn: Union[Callable, None] = None
        self.neighbors: Union[partition.NeighborList, None] = None
        self.offsets: Union[jnp.ndarray, None] = None
        self.model: Union[Callable, None] = None

    def _update_implemented_properties(self):
        """
        Dynamically determines the properties this model produces based on its config,
        including custom heads, stress, and ensemble uncertainties.
        Updates ASE's global all_properties to ensure compatibility.
        """
        props = {"energy", "forces"}

        if self.model_config.model.calc_stress:
            props.add("stress")

        for head in self.model_config.model.property_heads:
            props.add(head.name)

        # Handle ensembles
        ensemble_config = self.model_config.model.ensemble
        if ensemble_config:
            ensembled_props = set()
            if ensemble_config.kind == "full":
                # For a full ensemble, all base properties are ensembled
                ensembled_props.update(props)

            elif ensemble_config.kind == "shallow":
                # For a shallow ensemble, only energy and potentially forces are ensembled
                ensembled_props.add("energy")
                if ensemble_config.force_variance:
                    ensembled_props.add("forces")

            for p in ensembled_props:
                props.add(f"{p}_uncertainty")
                props.add(f"{p}_ensemble")

        # Handle individual property head ensembles
        for head in self.model_config.model.property_heads:
            if head.n_shallow_members > 0:
                props.add(f"{head.name}_uncertainty")
                props.add(f"{head.name}_ensemble")

        # Finalize and update global list
        self.implemented_properties = sorted(props)
        for p in self.implemented_properties:
            if p not in all_properties:
                all_properties.append(p)

    def initialize(self, atoms: ase.Atoms) -> None:
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

        if "stress" in self.implemented_properties:
            model = ProcessStress().apply(model)

        for transformation in self.transformations:
            model = transformation.apply(model)

        self.model = model
        self.step = get_step_fn(
            model,
            jnp.asarray(atoms.numbers),
            bool(np.any(atoms.cell.array > 1e-6)),
            self.neigbor_from_jax,
        )
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

            (idxs_i,) = calculator.compute(
                points=atoms.positions,
                box=atoms.cell.array,
                periodic=bool(np.any(atoms.pbc)),
                quantities="i",
            )
            self.padded_length = int(len(idxs_i) * self.padding_factor)

    def get_descriptors(
        self,
        frames: List[ase.Atoms],
        processing_batch_size: int = 1,
        only_use_n_layers: Union[int, None] = None,
        should_average: bool = True,
    ) -> np.ndarray:
        """Compute the descriptors for a list of Atoms.

        Parameters
        ----------
        frames : list[ase.Atoms]
            List of Atoms to compute descriptors for.
        processing_batch_size : int, default = 1
            Batch size for processing the frames. This does not affect the results,
            only the speed and memory requirements.
        only_use_n_layers: int | None, default = None
            If specified, only the first `only_use_n_layers` layers of the feature model will
            be used to compute the descriptors. If None, all layers will be used.
        should_average: bool, default = True
            Whether to average the descriptors over the atomic species.

        Returns
        -------
        np.ndarray
            Array of computed descriptors for each frame with shape (n_frames, n_descriptors)
            or (n_frames, n_atoms, n_descriptors) if ``should_average=False``.
        """
        params = canonicalize_energy_model_parameters(self.params)

        dataset = OTFInMemoryDataset(
            frames,
            cutoff=self.model_config.model.basis.r_max,
            bs=processing_batch_size,
            n_epochs=1,
            ignore_labels=True,
            pos_unit=self.model_config.data.pos_unit,
            energy_unit=self.model_config.data.energy_unit,
        )

        _, init_box = dataset.init_input()

        Builder = self.model_config.model.get_builder()
        builder = Builder(self.model_config.model.model_dump(), n_species=119)

        feature_model = builder.build_feature_model(
            apply_mask=True,
            init_box=init_box,
            only_use_n_layers=only_use_n_layers,
            should_average=should_average,
        )

        feature_fn = feature_model.apply

        batched_feature_fn = jax.vmap(feature_fn, in_axes=(None, 0, 0, 0, 0, 0))

        jitted_fn = jax.jit(batched_feature_fn)

        results = []
        for batch in tqdm(dataset.batch(), ncols=100, total=len(frames)):
            data = jitted_fn(
                params,
                batch["positions"],
                batch["numbers"],
                batch["idx"],
                batch["box"],
                batch["offsets"],
            )
            results.append(data)
        return np.concatenate(results, axis=0)

    def set_neighbours_and_offsets(self, atoms: ase.Atoms, box: np.ndarray) -> None:
        calculator = NeighborList(cutoff=self.r_max, full_list=True)
        idxs_i, idxs_j, offsets = calculator.compute(
            points=atoms.positions,
            box=atoms.cell.array,
            periodic=bool(np.any(atoms.pbc)),
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

    def calculate(
        self,
        atoms: ase.Atoms,
        properties: List[str] = ["energy"],
        system_changes=all_changes,
    ) -> None:
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
        self,
        atoms_list: List[ase.Atoms],
        batch_size: int = 64,
        silent: bool = False,
    ) -> List[ase.Atoms]:
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

        if "stress" in self.implemented_properties:
            model = ProcessStress().apply(model)

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
    def ll_weights(self) -> np.ndarray:
        dense_layers = list(self.params["params"]["energy_model"]["readout"].keys())
        llweights = self.params["params"]["energy_model"]["readout"][dense_layers[-1]][
            "w"
        ]
        return np.asarray(llweights)

    def set_ll_weights(self, new_weights: np.ndarray) -> None:
        params = unfreeze(self.params)
        dense_layers = list(params["params"]["energy_model"]["readout"].keys())
        params["params"]["energy_model"]["readout"][dense_layers[-1]]["w"] = jnp.asarray(
            new_weights, dtype=jnp.float32
        )
        self.params = freeze(params)
        self.step = None


def neighbor_calculable_with_jax(box: np.ndarray, r_max: float) -> bool:
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


def get_step_fn(
    model: Callable, Z: jnp.ndarray, atoms_is_periodic: bool, neigbor_from_jax: bool
) -> Callable:
    if neigbor_from_jax:

        @jax.jit
        def step_fn(
            positions: jnp.ndarray, neighbor: partition.NeighborList, box: jnp.ndarray
        ) -> Tuple[Dict, partition.NeighborList]:
            if atoms_is_periodic:
                box = box.T
                inv_box = jnp.linalg.inv(box)
                positions = space.transform(inv_box, positions)
                neighbor = neighbor.update(positions, box=box)
            else:
                neighbor = neighbor.update(positions)

            offsets = jnp.full([neighbor.idx.shape[1], 3], 0)
            results = model(positions, Z, neighbor.idx, box, offsets)
            return results, neighbor

    else:

        @jax.jit
        def step_fn(
            positions: jnp.ndarray,
            neighbor: jnp.ndarray,
            box: jnp.ndarray,
            offsets: jnp.ndarray,
        ) -> Dict:
            results = model(positions, Z, neighbor, box, offsets)
            return results

    return step_fn
