import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gmnn_jax.data.statistics import energy_per_element


def test_energy_per_element():
    pos = [0.0, 0.0, 0.0]
    atoms1 = Atoms("H2O", positions=[pos, pos, pos])
    atoms2 = Atoms("CH4", positions=[pos, pos, pos, pos, pos])
    atoms3 = Atoms("N2", positions=[pos, pos])

    dummy_energies = np.array([0, 20.0, 0, 0, 0, 0, 50.0, 100.0, 70.0])

    atoms_list = [atoms1, atoms2, atoms3]
    energies = []
    for atoms in atoms_list:
        energy = np.sum(dummy_energies[atoms.numbers])
        energies.append(energy)
        atoms.calc = SinglePointCalculator(atoms, energy=energy)

    elemental_energies_mean, elemental_energies_std = energy_per_element(atoms_list, 0.0)
    regression_energies = []
    for atoms in atoms_list:
        energy = np.sum(elemental_energies_mean[atoms.numbers])
        regression_energies.append(energy)

    assert np.allclose(energies, regression_energies, atol=1e-4)
