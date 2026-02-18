import numpy as np
import pytest
from ase import Atoms
from openmm.openmm import LangevinMiddleIntegrator, PythonForce, State
from openmm.unit import (
    angstrom,
    dalton,
    femtosecond,
    kelvin,
    kilojoules_per_mole,
    nanometer,
    picosecond,
)

from apax.md.openmm_interface import create_simulation, create_system


def compute(state: State) -> tuple[float, np.ndarray]:
    pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
    k = state.getParameters()["k"]
    energy = k * np.sum(pos * pos)
    force = -0.5 * k * pos
    return energy * kilojoules_per_mole, force * kilojoules_per_mole / nanometer


atoms_list = [
    (Atoms(symbols="H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[0, 0, 0]), False),
    (Atoms(symbols="H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10]), True),
]


@pytest.mark.parametrize("atoms, periodic_force", atoms_list)
def test_create_openmm_simulation(atoms: Atoms, periodic_force: bool):
    force = PythonForce(compute, {"k": 2.5})
    force.setUsesPeriodicBoundaryConditions(periodic_force)

    system = create_system(atoms)
    assert system.getNumParticles() == len(atoms)

    masses = atoms.get_masses()
    for i, atom in enumerate(atoms):
        assert system.getParticleMass(i).value_in_unit(dalton) == masses[i]

    _ = system.addForce(force)

    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.5 * femtosecond
    )

    simulation = create_simulation(atoms, system, integrator)

    state = simulation.context.getState(forces=True, positions=True, energy=True)

    state_box_vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(angstrom)
    state_pos = state.getPositions(asNumpy=True).value_in_unit(angstrom)

    if np.any(atoms.pbc):
        assert np.all(state_box_vectors == atoms.cell.array)

    assert np.all(state_pos == atoms.positions)
