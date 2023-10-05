import os
import urllib
import zipfile
from typing import List

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from apax.utils.random import seed_py_np_tf


@pytest.fixture(autouse=True)
def set_radom_seeds():
    seed_py_np_tf()


def create_cell(a: float, lattice: str) -> np.ndarray:
    free_cell = (0.0, 0.0, 0.0)
    sc_cell = (a, a, a)
    fcc_cell = [(0, a, a), (a, 0, a), (a, a, 0)]
    bcc_cell = [(-a, a, a), (a, -a, a), (a, a, -a)]

    cells = {
        "free": free_cell,
        "sc": sc_cell,
        "fcc": fcc_cell,
        "bcc": bcc_cell,
    }

    return cells[lattice]


@pytest.fixture()
def example_atoms(num_data: int, pbc: bool, calc_results: List[str]) -> Atoms:
    atoms_list = []

    for _ in range(num_data):
        num_atoms = np.random.randint(10, 15)
        numbers = np.random.randint(1, 119, size=num_atoms)
        cell_const = np.random.uniform(low=10.0, high=12.0)
        positions = np.random.uniform(low=0.0, high=cell_const, size=(num_atoms, 3))

        additional_data = {}
        additional_data["pbc"] = pbc
        # lattice = random.choice(["free", "sc", "fcc", "bcc"])
        # at the moment we can only work with cubic cells
        lattice = "sc"
        if pbc:
            additional_data["cell"] = create_cell(cell_const, lattice)
        else:
            additional_data["cell"] = [0.0, 0.0, 0.0]

        result_shapes = {
            "energy": (np.random.rand() - 5.0) * 10_000,
            "forces": np.random.uniform(low=-1.0, high=1.0, size=(num_atoms, 3)),
            # "stress": np.random.uniform(low=-1.0, high=1.0, size=(3, 3)),
            # "dipole": np.random.randn(3),
            # "charge": np.random.randint(-3, 4),
            # "ma_tensors": np.random.uniform(low=-1.0, high=1.0, size=(3, 3)),
        }

        atoms = Atoms(numbers=numbers, positions=positions, **additional_data)
        if calc_results:
            results = {}
            for key in calc_results:
                results[key] = result_shapes[key]

            atoms.calc = SinglePointCalculator(atoms, **results)
        atoms_list.append(atoms)

    return atoms_list


@pytest.fixture(scope="session")
def get_tmp_path(tmp_path_factory):
    test_path = tmp_path_factory.mktemp("apax_tests")
    return test_path


@pytest.fixture(scope="session")
def get_md22_stachyose(get_tmp_path):
    url = "http://www.quantum-machine.org/gdml/repo/static/md22_stachyose.zip"
    data_path = get_tmp_path / "data"
    file_path = data_path / "md22_stachyose.zip"

    os.makedirs(data_path, exist_ok=True)
    urllib.request.urlretrieve(url, file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    file_path = modify_xyz_file(
        file_path.with_suffix(".xyz"), target_string="Energy", replacement_string="energy"
    )

    return file_path


def modify_xyz_file(file_path, target_string, replacement_string):
    new_file_path = file_path.with_name(file_path.stem + "_mod" + file_path.suffix)

    with open(file_path, "r") as input_file, open(new_file_path, "w") as output_file:
        for line in input_file:
            # Replace all occurrences of the target string with the replacement string
            modified_line = line.replace(target_string, replacement_string)
            output_file.write(modified_line)
    return new_file_path
