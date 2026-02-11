from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np
from ase import Atoms
from openmm.app import Element, Simulation, Topology
from openmm.openmm import Integrator, PythonForce, State, System
from openmm.unit import angstrom, ev, item

from apax.md.ase_calc import (build_energy_neighbor_fns, check_for_ensemble,
                              get_step_fn, make_ensemble,
                              neighbor_calculable_with_jax)
from apax.train.checkpoints import restore_parameters
from apax.utils.convert import openmm_unit_dict
from apax.utils.jax_md_reduced import space


def create_topology_from_ase_atoms(atoms: Atoms) -> System:
    topology = Topology()
    for i, atom in enumerate(atoms):
        chain = topology.addChain()
        residue = topology.addResidue(f"atom-{i}", chain)
        element = Element.getByAtomicNumber(atom.number)
        topology.addAtom(f"atom-{i}", element, residue)

    return topology


def create_system(atoms: Atoms) -> System:
    system = System()
    for atom in atoms:
        system.addParticle(atom.mass)
    return system


def create_simulation(atoms: Atoms, system: System, integrator: Integrator) -> Simulation:
    topology = create_topology_from_ase_atoms(atoms)
    simulation = Simulation(topology, system, integrator)

    simulation.context.setPositions(atoms.positions * angstrom)

    if np.any(atoms.cell.array > 1e-6):
        a, b, c = (
            atoms.cell[0] * angstrom,
            atoms.cell[1] * angstrom,
            atoms.cell[2] * angstrom,
        )
        simulation.context.setPeriodicBoxVectors(a, b, c)
    else:
        # If the atoms instance is nonperiodic, the Simulation (and resulting States)
        # will still have getPeriodicBoxVectors set to the default value of the System
        # (which is [
        #     [20, 0,  0],
        #     [0,  20, 0],
        #     [0,  0,  20],
        # ] in Angstrom
        # by default in OpenMM from my testing)
        pass

    return simulation


class OpenMMInterface:
    def __init__(
        self,
        model_dir: str | Path,
        dr_threshold: float = 0.5,
        padding_factor: float = 1.5,
        transformations: list[Callable] = [],
    ):
        self.model_config, self.params = restore_parameters(model_dir)

        # self.n_models = check_for_ensemble(self.params)
        # self.padding_factor = padding_factor
        # self.transformations = transformations
        # self.padded_length = 0

        self.dr_threshold = dr_threshold

        self.step = None
        self.neighbor_fn = None
        self.neighbors = None
        self.offsets = None
        self.model = None

    def _initialize(self, atoms: Atoms) -> None:
        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
        self.r_max = self.model_config.model.basis.r_max
        neighbor_from_jax = neighbor_calculable_with_jax(box, self.r_max)
        model, self.neighbor_fn = build_energy_neighbor_fns(
            atoms,
            self.model_config,
            self.params,
            self.dr_threshold,
            neighbor_from_jax,
        )

        # if self.n_models > 1:
        #     model = make_ensemble(model)
        #
        # if "stress" in self.implemented_properties:
        #     model = ProcessStress().apply(model)
        #
        # for transformation in self.transformations:
        #     model = transformation.apply(model)

        self.step = get_step_fn(model, atoms, neighbor_from_jax)

        # if self.neigbor_from_jax:
        #     positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
        #     if np.any(atoms.get_cell().lengths() > 1e-6):
        #         box = atoms.cell.array.T
        #         inv_box = jnp.linalg.inv(box)
        #         positions = space.transform(inv_box, positions)  # frac coords
        #         self.neighbors = self.neighbor_fn.allocate(positions, box=box)
        #     else:
        #         self.neighbors = self.neighbor_fn.allocate(positions)
        # else:
        #     calculator = NeighborList(cutoff=self.r_max, full_list=True)
        #     idxs_i, _, _ = calculator.compute(
        #         points=atoms.positions,
        #         box=atoms.cell.array,
        #         periodic=bool(np.any(atoms.pbc)),
        #         quantities="ijS",
        #     )
        #     self.padded_length = int(len(idxs_i) * self.padding_factor)

    # def get_neighbours_and_offsets(
    #     self,
    #     positions: np.ndarray,
    #     box: np.ndarray,
    #     periodic: bool,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     calculator = NeighborList(cutoff=self.r_max, full_list=True)
    #     idxs_i, idxs_j, offsets = calculator.compute(
    #         points=positions,
    #         box=box,
    #         periodic=np.any(np.linalg.norm(box, axis=0) > 1e-6),
    #         quantities="ijS",
    #     )
    #     if len(idxs_i) > self.padded_length:
    #         raise ValueError(
    #             f"neighbor list overflowed, reallocating not implemented yet for {self.__class__.__name__}."
    #         )
    #         # self.initialize(atoms)

    #     zeros_to_add = self.padded_length - len(idxs_i)

    #     neighbors = np.array([idxs_i, idxs_j], dtype=np.int32)
    #     neighbors = np.pad(neighbors, ((0, 0), (0, zeros_to_add)), "constant")

    #     offsets = np.matmul(offsets, box)
    #     offsets = np.pad(offsets, ((0, zeros_to_add), (0, 0)), "constant")
    #     return neighbors, offsets

    def get_python_force_fn(
        self, atoms: Atoms
    ) -> Callable[State, tuple[float, np.ndarray]]:
        self._initialize(atoms)

        _energy_unit_scaling = ev / item
        _pos_unit_scaling = angstrom
        _force_unit_scaling = _energy_unit_scaling / _pos_unit_scaling

        # def inline_initialize(positions: np.ndarray, box: np.ndarray):
        #     neigbor_from_jax = neighbor_calculable_with_jax(box, self.r_max)
        #     model, neighbor_fn = build_energy_neighbor_fns(
        #         atoms,
        #         self.model_config,
        #         self.params,
        #         self.dr_threshold,
        #         neigbor_from_jax,
        #     )

        #     # if self.n_models > 1:
        #     #     model = make_ensemble(model)

        #     # if "stress" in self.implemented_properties:
        #     #     model = ProcessStress().apply(model)

        #     # for transformation in self.transformations:
        #     #     model = transformation.apply(model)

        #     step = get_step_fn(self.model, atoms, neigbor_from_jax)
        #     self.neighbor_fn = neighbor_fn

        #     # if self.neigbor_from_jax:
        #     #     if np.any(atoms.get_cell().lengths() > 1e-6):
        #     #         box = atoms.cell.array.T
        #     #         inv_box = jnp.linalg.inv(box)
        #     #         positions = space.transform(inv_box, pos)  # frac coords
        #     #         self.neighbors = self.neighbor_fn.allocate(positions, box=box)
        #     #     else:
        #     #         self.neighbors = self.neighbor_fn.allocate(positions)
        #     # else:
        #     #     calculator = NeighborList(cutoff=self.r_max, full_list=True)
        #     #     idxs_i, _, _ = calculator.compute(
        #     #         points=atoms.positions,
        #     #         box=atoms.cell.array,
        #     #         periodic=bool(np.any(atoms.pbc)),
        #     #         quantities="ijS",
        #     #     )
        #     #     self.padded_length = int(len(idxs_i) * self.padding_factor)

        #     return step

        def compute_periodic(state: State) -> tuple[float, np.ndarray]:
            pos = jnp.asarray(
                state.getPositions(asNumpy=True).value_in_unit(_pos_unit_scaling),
                dtype=jnp.float64,
            )
            box = jnp.asarray(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                    _pos_unit_scaling
                ),
                dtype=jnp.float64,
            )

            neighbor_from_jax = neighbor_calculable_with_jax(box, self.r_max)
            if not neighbor_from_jax:
                raise ValueError(
                    f"The box is not big enough for the neighbors to be calculated with jax. This is not implemented for {self.__class__.__name__}"
                )
                # prev_neigbor_from_jax != neigbor_from_jax:
                # step = inline_initialize(pos, box)
                # prev_neigbor_from_jax = neigbor_from_jax

            # predict
            if neighbor_from_jax:
                inv_box = jnp.linalg.inv(box)
                pos_frac = space.transform(inv_box, pos)  # frac coords
                neighbors = self.neighbor_fn.allocate(pos_frac, box=box)

                if neighbors.did_buffer_overflow:
                    raise ValueError(
                        f"neighbor list overflowed, reallocating not implemented yet for {self.__class__.__name__}."
                    )
                    # step = inline_initialize(pos, box)
                    # results, _ = step(pos, self.neighbors, box)

            # else:
            #     neighbors, offsets = self.get_neighbours_and_offsets(atoms, box)
            #     pos = np.array(space.transform(np.linalg.inv(box), pos))

            #     results = step(pos, neighbors, box, offsets)

            results, _ = self.step(pos, neighbors, box)

            # Return energy and forces in OpenMM units, i.e. kJ/mol and kJ/(mol nm)
            # It is important that the force unit scaling happens outside of
            # np.asarray, because otherwise the force unit scaling is lost
            return results["energy"] * _energy_unit_scaling, np.asarray(
                results["forces"]
            ) * _force_unit_scaling

        def compute_nonperiodic(state: State) -> tuple[float, np.ndarray]:
            pos = jnp.asarray(
                state.getPositions(asNumpy=True).value_in_unit(_pos_unit_scaling),
                dtype=jnp.float64,
            )

            neighbors = self.neighbor_fn.allocate(pos)

            if neighbors.did_buffer_overflow:
                raise ValueError(
                    f"neighbor list overflowed, reallocating not implemented yet for {self.__class__.__name__}."
                )

            results, _ = self.step(pos, neighbors, jnp.full((3, 3), 0.0))

            # Return energy and forces in OpenMM units, i.e. kJ/mol and kJ/(mol nm)
            # It is important that the force unit scaling happens outside of
            # np.asarray, because otherwise the force unit scaling is lost
            return results["energy"] * _energy_unit_scaling, np.asarray(
                results["forces"]
            ) * _force_unit_scaling

        if np.any(atoms.cell.array > 1e-6):
            return compute_periodic
        else:
            return compute_nonperiodic


def get_PythonForce_from_Apax(model_dir: str | Path, atoms: Atoms) -> PythonForce:
    atoms_is_periodic = np.any(atoms.cell.array) > 1e-6

    interface = OpenMMInterface(model_dir)
    compute_fn = interface.get_python_force_fn(atoms)
    python_force = PythonForce(compute_fn)
    python_force.setUsesPeriodicBoundaryConditions(atoms_is_periodic)
    return python_force
