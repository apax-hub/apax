import numpy as np
import pytest
from apax.data.grain_pipeline import SoADataSource, NeighborListTransform, ApaxGrainDataLoader

def test_apax_grain_dataloader():
    num_samples = 12
    batch_size = 4
    data = {
        "positions": np.random.rand(num_samples, 3, 3),
        "box": np.random.rand(num_samples, 3, 3),
        "energy": np.random.rand(num_samples),
    }
    
    loader = ApaxGrainDataLoader(
        data,
        batch_size=batch_size,
        cutoff=2.0,
        max_nbrs=10,
        shuffle=False,
        num_workers=0, # Use 0 for deterministic unit tests if possible, or 1
    )
    
    batches = list(loader)
    assert len(batches) == num_samples // batch_size
    
    first_batch = batches[0]
    assert first_batch["energy"].shape == (batch_size,)
    assert "idx" in first_batch
    assert "offsets" in first_batch

def test_apax_grain_dataloader_shuffle():
    num_samples = 12
    batch_size = 4
    data = {
        "energy": np.arange(num_samples),
        "positions": np.random.rand(num_samples, 3, 3),
        "box": np.random.rand(num_samples, 3, 3),
    }
    
    loader = ApaxGrainDataLoader(
        data,
        batch_size=batch_size,
        cutoff=2.0,
        max_nbrs=10,
        shuffle=True,
    )
    
    batches = list(loader)
    all_energies = np.concatenate([b["energy"] for b in batches])
    assert not np.all(all_energies == np.arange(num_samples))

def test_nl_transform_padding_scenarios():
    sample = {
        "positions": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "box": np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
    }
    cutoff = 2.0
    
    # Case 1: Padding needed
    transform = NeighborListTransform(cutoff, max_nbrs=10)
    transformed = transform.map(sample.copy())
    assert transformed["idx"].shape == (2, 10)
    
    # Case 2: Truncation needed (unlikely in real use but covered in code)
    transform = NeighborListTransform(cutoff, max_nbrs=1)
    transformed = transform.map(sample.copy())
    assert transformed["idx"].shape == (2, 1)
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
    assert np.allclose(sample["numbers"], data["numbers"][0])
    assert np.allclose(sample["energy"], data["energy"][0])
    
    sample_last = source[9]
    assert np.allclose(sample_last["energy"], data["energy"][9])

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
    assert transformed_sample["idx"].shape[0] == 2
    assert transformed_sample["idx"].shape[1] > 0
