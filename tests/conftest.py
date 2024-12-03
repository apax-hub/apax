import os
from typing import List

import jax
import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator

from apax.config.train_config import Config
from apax.train.run import run
from apax.utils.datasets import download_md22_stachyose
from apax.utils.helpers import mod_config
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


@pytest.fixture()
def example_dataset(num_data: int) -> Atoms:
    atoms_list = []

    p2 = np.random.uniform(low=1.0, high=1.5, size=(num_data,))
    for i in range(num_data):
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, p2[i]]])

        additional_data = {}
        additional_data["cell"] = [0.0, 0.0, 0.0]

        atoms = Atoms("H2", positions=positions, **additional_data)
        atoms.calc = EMT()
        atoms.get_potential_energy()
        atoms.get_forces()

        atoms_list.append(atoms)

    return atoms_list


@pytest.fixture()
def get_tmp_path(tmp_path_factory):
    test_path = tmp_path_factory.mktemp("apax_tests")
    return test_path


@pytest.fixture(scope="session")
def tmp_data_path(tmp_path_factory):
    test_path = tmp_path_factory.mktemp("data")
    return test_path


@pytest.fixture(scope="session")
def get_md22_stachyose(tmp_data_path):
    file_path = download_md22_stachyose(tmp_data_path)
    return file_path


@pytest.fixture()
def get_sample_input():
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    atomic_numbers = np.array([1, 1, 8])
    box = np.diag(np.zeros(3))
    offsets = np.full([3, 3], 0)
    idx = np.array([[1, 2, 0, 2, 0, 1], [0, 0, 1, 1, 2, 2]])

    inputs = (positions, atomic_numbers, idx, box, offsets)
    return inputs, box


def initialize_model(model_config, sample_input):
    Builder = model_config.model.get_builder()
    builder = Builder(model_config.model.model_dump())
    model = builder.build_energy_derivative_model()
    rng_key = jax.random.PRNGKey(model_config.seed)
    params = model.init(rng_key, *sample_input)

    return model, params


def load_and_dump_config(config_path, dump_path):
    with open(config_path.as_posix(), "r") as stream:
        model_config_dict = yaml.safe_load(stream)

    model_config_dict["data"]["directory"] = dump_path.as_posix()

    model_config = Config.model_validate(model_config_dict)
    os.makedirs(model_config.data.model_version_path, exist_ok=True)
    model_config.dump_config(model_config.data.model_version_path)
    return model_config


def load_config_and_run_training(config_path, updated_config):
    config_dict = mod_config(config_path, updated_config)
    run(config_dict)
    return config_dict
