import numpy as np
import grain.python as grain
from apax.data.preprocessing import compute_nl

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

class NeighborListTransform(grain.MapTransform):
    """
    A Grain MapTransform that computes the neighbor list for a sample and pads it.
    """
    def __init__(self, cutoff: float, max_nbrs: int = None):
        self.cutoff = cutoff
        self.max_nbrs = max_nbrs

    def map(self, sample: dict):
        positions = sample["positions"]
        box = sample["box"]
        idx, offsets = compute_nl(positions, box, self.cutoff)
        
        if self.max_nbrs:
            n_nbrs = idx.shape[1]
            if n_nbrs > self.max_nbrs:
                # Should we raise an error or just truncate? 
                # Apax usually expects max_nbrs to be sufficient.
                idx = idx[:, :self.max_nbrs]
                offsets = offsets[:self.max_nbrs]
            else:
                zeros_to_add = self.max_nbrs - n_nbrs
                idx = np.pad(idx, ((0, 0), (0, zeros_to_add)), "constant").astype(np.int32)
                offsets = np.pad(offsets, ((0, zeros_to_add), (0, 0)), "constant")
        
        sample["idx"] = idx
        sample["offsets"] = offsets
        return sample

class ApaxGrainDataLoader:
    """
    A high-level wrapper for the Grain DataLoader.
    """
    def __init__(
        self,
        data: dict,
        batch_size: int,
        cutoff: float,
        max_nbrs: int = None,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        self.data_source = SoADataSource(data)
        
        # Sampler
        if shuffle:
            self.sampler = grain.IndexSampler(
                num_records=len(self.data_source),
                num_epochs=1,
                shard_options=grain.NoSharding(),
                shuffle=True,
                seed=42,
            )
        else:
            self.sampler = grain.IndexSampler(
                num_records=len(self.data_source),
                num_epochs=1,
                shard_options=grain.NoSharding(),
                shuffle=False,
            )
            
        # Operations
        ops = [
            NeighborListTransform(cutoff, max_nbrs),
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ]
        
        self.loader = grain.DataLoader(
            data_source=self.data_source,
            sampler=self.sampler,
            operations=ops,
            worker_count=num_workers,
        )

    def __iter__(self):
        return iter(self.loader)
