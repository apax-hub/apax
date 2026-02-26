import numpy as np
import pytest
from apax.data.grain_pipeline import SoADataSource, NeighborListTransform

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
