import numpy as np
import grain.python as grain

class SoADataSource(grain.RandomAccessDataSource):
    """
    A Grain DataSource that operates on a Structure of Arrays (SoA).
    """
    def __init__(self, data: dict):
        self._data = data
        # Assume all arrays have the same first dimension (num_samples)
        self._num_samples = len(next(iter(data.values())))

    def __len__(self):
        return self._num_samples

    def __getitem__(self, index):
        if isinstance(index, slice):
            return {k: v[index] for k, v in self._data.items()}
        return {k: v[index] for k, v in self._data.items()}
