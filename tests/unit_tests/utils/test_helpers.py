import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_masses

from apax.utils.helpers import (
    get_ase_mass,
    get_masses,
    get_updated_atomic_masses,
    update_nested_dictionary,
)


def test_update_nested_dictionary():
    d1 = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    d2 = {"b": {"c": 4, "f": 5}}
    expected_dict = {"a": 1, "b": {"c": 4, "d": {"e": 3}, "f": 5}}
    updated_dict = update_nested_dictionary(d1, d2)
    assert updated_dict == expected_dict


get_ase_mass_data = [("He", 4.0), (10, 10), ("H", 1.0)]


@pytest.mark.parametrize("symbol_or_mass, expected_mass", get_ase_mass_data)
def test_get_ase_mass(symbol_or_mass, expected_mass):
    mass = get_ase_mass(symbol_or_mass)
    assert np.isclose(mass, expected_mass, atol=1e-2)


get_updated_atomic_masses_data = [({}, atomic_masses)]


@pytest.mark.parametrize(
    "custom_mass_dictionary, expected_atomic_masses", get_updated_atomic_masses_data
)
def test_get_updated_atomic_masses(custom_mass_dictionary, expected_atomic_masses):
    updated_atomic_masses = get_updated_atomic_masses(custom_mass_dictionary)
    assert np.allclose(updated_atomic_masses, expected_atomic_masses)
    pass


get_masses_data = [
    ({"H": 2, "O": 0.5}, Atoms("OH2"), np.array([0.5, 2, 2])),
    ({"H": "O", "O": 0.5}, Atoms("OH2"), np.array([0.5, 16, 16])),
    ({}, Atoms("OH2"), np.array([16, 1, 1])),
]


@pytest.mark.parametrize(
    "custom_mass_dictionary, atoms, expected_masses", get_masses_data
)
def test_get_masses(custom_mass_dictionary, atoms, expected_masses):
    masses = get_masses(custom_mass_dictionary, atoms)
    assert np.allclose(masses, expected_masses, atol=1e-2)
