import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from apax.data.grain_pipeline import SoADataSource, NeighborListTransform, ApaxGrainDataLoader

def create_dummy_atoms(num_samples=12, num_atoms=3):
    atoms_list = []
    for _ in range(num_samples):
        positions = np.random.rand(num_atoms, 3)
        numbers = np.random.randint(1, 10, num_atoms)
        cell = np.eye(3) * 10.0
        atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
        energy = np.random.rand()
        forces = np.random.rand(num_atoms, 3)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms_list.append(atoms)
    return atoms_list

def test_apax_grain_dataloader():
    num_samples = 12
    batch_size = 4
    atoms_list = create_dummy_atoms(num_samples)
    
    loader = ApaxGrainDataLoader(
        atoms_list,
        cutoff=2.0,
        bs=batch_size,
        n_epochs=1,
        max_nbrs=10,
        pre_shuffle=False,
        num_workers=0,
    )
    
    batches = list(loader)
    assert len(batches) == num_samples // batch_size
    
    inputs, labels = batches[0]
    assert labels["energy"].shape == (batch_size,)
    assert "idx" in inputs
    assert "offsets" in inputs

def test_apax_grain_dataloader_shuffle():
    num_samples = 12
    batch_size = 4
    atoms_list = create_dummy_atoms(num_samples)
    
    loader = ApaxGrainDataLoader(
        atoms_list,
        cutoff=2.0,
        bs=batch_size,
        n_epochs=1,
        max_nbrs=10,
        pre_shuffle=True,
    )
    
    batches = list(loader.shuffle_and_batch())
    all_energies = np.concatenate([b[1]["energy"] for b in batches])
    # Very high probability that they are not in order
    expected_energies = np.array([a.calc.results["energy"] for a in atoms_list])
    assert not np.allclose(all_energies, expected_energies[:12])

def test_apax_grain_dataloader_ragged():
    # Data with different number of atoms
    atoms_list = [
        Atoms(numbers=[1, 1, 1], positions=np.random.rand(3, 3), cell=np.eye(3)*10, pbc=True),
        Atoms(numbers=[1]*5, positions=np.random.rand(5, 3), cell=np.eye(3)*10, pbc=True),
        Atoms(numbers=[1]*4, positions=np.random.rand(4, 3), cell=np.eye(3)*10, pbc=True),
        Atoms(numbers=[1]*6, positions=np.random.rand(6, 3), cell=np.eye(3)*10, pbc=True),
    ]
    for a in atoms_list:
        a.calc = SinglePointCalculator(a, energy=0.0, forces=np.zeros((len(a), 3)))
    
    loader = ApaxGrainDataLoader(
        atoms_list,
        bs=2,
        n_epochs=1,
        cutoff=2.0,
        bucket_boundaries=[4, 10],
        pre_shuffle=False,
    )
    
    batches = list(loader)
    assert len(batches) == 2
    
    # Check that one batch has max_atoms=4 and other has max_atoms=6
    shapes = sorted([b[0]["numbers"].shape[1] for b in batches])
    assert shapes == [4, 6]

def test_soa_datasource():
    data = {
        "positions": np.random.rand(10, 3, 3),
        "numbers": np.random.randint(1, 10, (10, 3)),
        "energy": np.random.rand(10),
    }
    
    source = SoADataSource(data)
    assert len(source) == 10
    
    sample = source[0]
    assert isinstance(sample, dict)
    assert np.allclose(sample["positions"], data["positions"][0])

def test_soa_datasource_slicing():
    data = {
        "energy": np.arange(10),
    }
    source = SoADataSource(data)
    sliced = source[2:5]
    assert len(sliced["energy"]) == 3
    assert np.all(sliced["energy"] == np.array([2, 3, 4]))

def test_nl_transform():
    sample = {
        "positions": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "box": np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
    }
    cutoff = 2.0
    transform = NeighborListTransform(cutoff)
    transformed_sample = transform.map(sample)
    assert "idx" in transformed_sample
    assert "offsets" in transformed_sample
