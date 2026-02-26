import importlib.metadata
import os
import warnings

import jax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
jax.config.update("jax_enable_x64", True)

warnings.filterwarnings("ignore", message=".*os.fork()*")


import time
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from apax.data.input_pipeline import CachedInMemoryDataset, prefetch_to_single_device
from apax.data.grain_pipeline import ApaxGrainDataLoader
from apax.utils.convert import atoms_to_inputs, atoms_to_labels

def create_dummy_data(num_samples=1000, num_atoms=10):
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

def benchmark_legacy(atoms_list, batch_size, cutoff, num_epochs=10):
    start_init = time.time()
    ds = CachedInMemoryDataset(
        atoms_list,
        cutoff=cutoff,
        bs=batch_size,
        n_epochs=num_epochs,
    )
    iterator = ds.shuffle_and_batch()
    init_time = time.time() - start_init
    
    start_run = time.time()
    num_batches = 0
    for batch in iterator:
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), batch)
        num_batches += 1
    run_time = time.time() - start_run

    ds.cleanup()
    
    return init_time, run_time, num_batches

def benchmark_grain(atoms_list, batch_size, cutoff, num_epochs=10):
    start_init = time.time()
    # Convert atoms to SoA
    inputs = atoms_to_inputs(atoms_list)
    labels = atoms_to_labels(atoms_list)
    
    max_atoms = max(inputs["n_atoms"])
    # Padding inputs
    padded_data = {
        "n_atoms": inputs["n_atoms"],
        "box": inputs["box"],
        "energy": labels["energy"],
    }
    
    num_samples = len(atoms_list)
    padded_data["numbers"] = np.zeros((num_samples, max_atoms), dtype=np.int32)
    padded_data["positions"] = np.zeros((num_samples, max_atoms, 3), dtype=np.float64)
    padded_data["forces"] = np.zeros((num_samples, max_atoms, 3), dtype=np.float64)
    
    for i, a in enumerate(atoms_list):
        n = len(a)
        padded_data["numbers"][i, :n] = a.numbers
        padded_data["positions"][i, :n] = inputs["positions"][i]
        padded_data["forces"][i, :n] = labels["forces"][i]

    loader = ApaxGrainDataLoader(
        padded_data,
        batch_size=batch_size,
        cutoff=cutoff,
        max_nbrs=4000, # typical default
        num_epochs=num_epochs,
        shuffle=True,
        num_workers=4, # Enable workers for better performance
        worker_buffer_size=4, # Increase buffer
    )
    # Wrap in prefetch_to_single_device
    prefetched_loader = prefetch_to_single_device(iter(loader), 2)
    
    init_time = time.time() - start_init
    
    start_run = time.time()
    num_batches = 0
    for batch in prefetched_loader:
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), batch)
        num_batches += 1
    
    run_time = time.time() - start_run
    
    return init_time, run_time, num_batches

if __name__ == "__main__":
    num_samples = 500
    batch_size = 2
    cutoff = 5.0
    num_atoms = 250
    
    print(f"Generating {num_samples} dummy samples...")
    atoms_list = create_dummy_data(num_samples=num_samples, num_atoms=num_atoms)
    
    print("\nBenchmarking Legacy (TensorFlow-based) Pipeline...")
    legacy_init, legacy_run, legacy_batches = benchmark_legacy(atoms_list, batch_size, cutoff)
    print(f"Initialization: {legacy_init:.4f}s")
    print(f"Execution (10 epoch): {legacy_run:.4f}s")
    print(f"Throughput: {num_samples * 10 / legacy_run:.2f} samples/s")
    
    print("\nBenchmarking Grain-based Pipeline...")
    grain_init, grain_run, grain_batches = benchmark_grain(atoms_list, batch_size, cutoff)
    print(f"Initialization: {grain_init:.4f}s")
    print(f"Execution (10 epoch): {grain_run:.4f}s")
    print(f"Throughput: {num_samples * 10 / grain_run:.2f} samples/s")
