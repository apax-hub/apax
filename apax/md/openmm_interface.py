import logging
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np
from ase import Atoms
from openmm.app import Element, Simulation, Topology
from openmm.openmm import CMMotionRemover, Integrator, PythonForce, State, System
from openmm.unit import angstrom, ev, item
from openmm.vec3 import Vec3
from vesin import NeighborList

from apax.md.ase_calc import (
    build_energy_neighbor_fns,
    check_for_ensemble,
    get_step_fn,
    make_ensemble,
    neighbor_calculable_with_jax,
)
from apax.train.checkpoints import restore_parameters
from apax.utils.jax_md_reduced import space

log = logging.getLogger(__name__)


def create_topology_from_ase_atoms(atoms: Atoms) -> Topology:
    """Create an OpenMM topology from an Atoms instance.

    Args:
        atoms (Atoms):

    Returns:
        toplogy (Topology):
    """
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("structure", chain)
    for i, atom in enumerate(atoms):
        element = Element.getByAtomicNumber(atom.number)
        topology.addAtom(f"atom-{i}", element, residue)

    if np.any(atoms.cell.array > 1e-6):
        a, b, c = (
            atoms.cell[0] * angstrom,
            atoms.cell[1] * angstrom,
            atoms.cell[2] * angstrom,
        )
        a = Vec3(a[0], a[1], a[2])
        b = Vec3(b[0], b[1], b[2])
        c = Vec3(c[0], c[1], c[2])

        topology.setPeriodicBoxVectors(atoms.cell.array * angstrom)

    return topology


def create_system(atoms: Atoms, removeCMMotion: bool = True) -> System:
    """Create an OpenMM System from an Atoms instance.

    Args:
        atoms (Atoms):
        removeCMMotion (bool): whether to keep the center-of-mass fixed during
            the simulation. Default = True

    Returns:
        system (System):
    """
    system = System()
    for atom in atoms:
        system.addParticle(atom.mass)

    if removeCMMotion:
        system.addForce(CMMotionRemover(1))

    if np.any(atoms.cell.array > 1e-6):
        a, b, c = (
            atoms.cell[0] * angstrom,
            atoms.cell[1] * angstrom,
            atoms.cell[2] * angstrom,
        )
        system.setDefaultPeriodicBoxVectors(a, b, c)

    return system


def create_simulation(atoms: Atoms, system: System, integrator: Integrator) -> Simulation:
    """Create an OpenMM Simulation.

    Args:
        atoms (Atoms):
        system (System):
        integrator (Integrator):

    Returns:
        simulation (Simulation):
    """
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

    return simulation


_dummy_box = jnp.full((3, 3), 0.0, dtype=jnp.float64)


class OpenMMInterface:
    def __init__(
        self,
        model_dir: str | Path,
        dr_threshold: float = 0.5,
        padding_factor: float = 1.5,
        transformations: list[Callable] = [],
    ):
        self.model_config, self.params = restore_parameters(model_dir)
        self.r_max = self.model_config.model.basis.r_max

        self.n_models = check_for_ensemble(self.params)

        self.dr_threshold = dr_threshold
        self.transformations = transformations
        self.padding_factor = padding_factor

        self.step = None
        self.neighbor_fn = None
        self.neighbors = None
        self._atoms = None

    def _initialize(self, atoms: Atoms) -> None:
        if self._atoms is not None:
            raise ValueError(
                f"This PythonForceClass instance was already initialized with atoms {atoms}, and cannot be reinitialized with different atoms. Please create a new instance"
            )
        self._atoms = atoms
        self._atoms_is_periodic = bool(np.any(atoms.pbc))

        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
        self.neigbor_from_jax = neighbor_calculable_with_jax(box, self.r_max)

        model, self.neighbor_fn = build_energy_neighbor_fns(
            self._atoms,
            self.model_config,
            self.params,
            self.dr_threshold,
            self.neigbor_from_jax,
        )

        if self.n_models > 1:
            model = make_ensemble(model)

        for transformation in self.transformations:
            model = transformation.apply(model)

        self.step = get_step_fn(model, atoms, self.neigbor_from_jax)

        positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
        self.previous_cell = box
        self.allocate_neighbors(positions, box)

    def allocate_neighbors(self, positions: jnp.ndarray, box: jnp.ndarray) -> None:
        if self.neigbor_from_jax:
            if np.any(box > 1e-6):
                inv_box = jnp.linalg.inv(box.T)
                positions = space.transform(inv_box, positions)  # frac coords
                self.neighbors = self.neighbor_fn.allocate(positions, box=box.T)
            else:
                self.neighbors = self.neighbor_fn.allocate(positions)
        else:
            calculator = NeighborList(cutoff=self.r_max, full_list=True)
            (idxs_i,) = calculator.compute(
                points=positions,
                box=box,
                periodic=self._atoms_is_periodic,
                quantities="i",
            )
            self.padded_length = int(len(idxs_i) * self.padding_factor)

    def set_neighbors_and_offsets(self, positions: jnp.ndarray, box: jnp.ndarray) -> None:
        if self.neigbor_from_jax:
            log.warning(
                "Setting neighbors and offsets for a PythonForceClass that has self.neigbor_from_jax = True. Unnecessary?"
            )
        calculator = NeighborList(cutoff=self.r_max, full_list=True)
        idxs_i, idxs_j, offsets = calculator.compute(
            points=positions,
            box=box,
            periodic=self._atoms_is_periodic,
            quantities="ijS",
        )
        if len(idxs_i) > self.padded_length:
            log.warning("Neighbor list overflowed, extending.")
            self.allocate_neighbors(positions, box)

        zeros_to_add = self.padded_length - len(idxs_i)

        self.neighbors = np.array([idxs_i, idxs_j], dtype=np.int32)
        self.neighbors = np.pad(self.neighbors, ((0, 0), (0, zeros_to_add)), "constant")

        offsets = np.matmul(offsets, box)
        self.offsets = np.pad(offsets, ((0, zeros_to_add), (0, 0)), "constant")

    def get_python_force_fn(self, atoms: Atoms) -> Callable:
        self._initialize(atoms)

        if np.any(atoms.cell.array > 1e-6):
            if not self._atoms_is_periodic:
                # This can happen if it is not set to True by the user. Then ASECalculator and
                # this will give varying results, because in calculator.compute, we assume
                # that periodic is always True (which makes sense for
                log.warning(
                    f"Atoms instance has non-zero cell {atoms.cell}, but its pbc attribute is False. Not using periodic boundary conditions to calculate neighbor lists."
                )
            return self._python_force_fn_periodic
        else:
            return self._python_force_fn

    def _python_force_fn_periodic(self, state: State) -> tuple[float, np.ndarray]:
        pos = jnp.asarray(
            state.getPositions(asNumpy=True).value_in_unit(angstrom), dtype=jnp.float64
        )
        box = jnp.asarray(
            state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(angstrom),
            dtype=jnp.float64,
        )

        if jnp.any(box != self.previous_cell):
            log.debug(
                "Cell changed, rechecking if neighbor list can be calculated with Jax"
            )
            neigbor_from_jax = neighbor_calculable_with_jax(box, self.r_max)
            if neigbor_from_jax != self.neigbor_from_jax:
                raise NotImplementedError("Need to re-get step fn. Not implemented yet.")
                self.neigbor_from_jax = neigbor_from_jax
                self.allocate_neighbors(pos, box)
            self.previous_cell = box

        if self.neigbor_from_jax:
            results, self.neighbors = self.step(pos, self.neighbors, box)
            if self.neighbors.did_buffer_overflow:
                log.debug("Neighbor list overflowed, reallocating")
                inv_box = jnp.linalg.inv(box.T)
                frac_pos = space.transform(inv_box, pos)  # frac coords
                self.neighbors = self.neighbor_fn.allocate(frac_pos, box=box.T)
            results, self.neighbors = self.step(pos, self.neighbors, box)
        else:
            self.set_neighbors_and_offsets(pos, box)
            pos = np.array(space.transform(np.linalg.inv(box), pos))

            results = self.step(pos, self.neighbors, box, self.offsets)

        return results["energy"] * ev / item, np.asarray(results["forces"]) * ev / (
            item * angstrom
        )

    def _python_force_fn(self, state: State) -> tuple[float, np.ndarray]:
        pos = jnp.asarray(
            state.getPositions(asNumpy=True).value_in_unit(angstrom), dtype=jnp.float64
        )

        results, self.neighbors = self.step(pos, self.neighbors, _dummy_box)
        if self.neighbors.did_buffer_overflow:
            self.neighbors = self.neighbor_fn.allocate(pos)
        results, self.neighbors = self.step(pos, self.neighbors, _dummy_box)

        return results["energy"] * ev / item, np.asarray(results["forces"]) * ev / (
            item * angstrom
        )


def get_PythonForce_from_Apax(model_dir: str | Path, atoms: Atoms) -> PythonForce:
    """Get a PythonForce instance that can be used for OpenMM simulations from
    a trained Apax model and the Atoms instance that it will be used for.

    Args:
        model_dir (str | Path):
        atoms (Atoms):

    Returns:
        python_force (PythonForce):
    """
    log.debug(f"Creating PythonForce instance with atoms {atoms}")

    atoms_is_periodic = bool(np.any(atoms.pbc))

    interface = OpenMMInterface(model_dir)
    compute_fn = interface.get_python_force_fn(atoms)
    python_force = PythonForce(compute_fn)
    python_force.setUsesPeriodicBoundaryConditions(atoms_is_periodic)
    log.debug(f"Set PythonForce.usesPeriodicBoundaryConditions to {atoms_is_periodic}")
    return python_force
