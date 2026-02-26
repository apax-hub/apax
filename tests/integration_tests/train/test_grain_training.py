import yaml
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from pathlib import Path
from apax.train.run import run

def create_dummy_data(tmp_path):
    atoms_list = []
    for _ in range(10):
        positions = np.random.rand(3, 3)
        numbers = np.array([1, 6, 8])
        cell = np.eye(3) * 10.0
        atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
        energy = np.random.rand()
        forces = np.random.rand(3, 3)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms_list.append(atoms)
    
    from ase.io import write
    data_path = tmp_path / "data.extxyz"
    write(str(data_path), atoms_list)
    return data_path

def test_grain_training_integration(tmp_path):
    data_path = create_dummy_data(tmp_path)
    
    config = {
        "n_epochs": 2,
        "seed": 42,
        "data": {
            "directory": str(tmp_path),
            "experiment": "test_grain",
            "data_path": str(data_path),
            "n_train": 5,
            "n_valid": 5,
            "batch_size": 2,
            "valid_batch_size": 2,
            "dataset": {
                "processing": "grain",
                "num_workers": 0,
            }
        },
        "model": {
            "name": "gmnn",
            "basis": {"name": "gaussian", "r_max": 3.0}
        },
        "loss": [
            {"name": "energy", "weight": 1.0},
            {"name": "forces", "weight": 10.0}
        ],
        "optimizer": {
            "name": "adam",
            "emb_lr": 0.01,
            "nn_lr": 0.01,
        }
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    # Run training
    run(config_path, log_level="info")
    
    # Check if model was saved
    assert (tmp_path / "test_grain" / "best").exists()
    assert (tmp_path / "test_grain" / "latest").exists()
