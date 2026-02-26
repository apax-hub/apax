import numpy as np
import grain.python as grain
from apax.data.preprocessing import compute_nl

class SoADataSource(grain.RandomAccessDataSource):
    """
    A Grain DataSource that operates on a Structure of Arrays (SoA).
    """
    def __init__(self, data: dict):
        self._data = data
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
    A high-level wrapper for the Grain DataLoader using threaded prefetching.
    """
    def __init__(
        self,
        data: dict,
        batch_size: int,
        cutoff: float,
        max_nbrs: int = None,
        num_epochs: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        worker_buffer_size: int = 1,
    ):
        # 1. Create MapDataset
        ds = grain.MapDataset.source(SoADataSource(data))
        
        # 2. Shuffle and Repeat
        if shuffle:
            ds = ds.shuffle(seed=42)
        if num_epochs > 1:
            ds = ds.repeat(num_epochs)
        
        # 3. Transform
        ds = ds.map(NeighborListTransform(cutoff, max_nbrs))
        
        # 4. Optimized Threaded Conversion to IterDataset
        # This uses C++ threads (if available) or background Python threads to parallelize mapping.
        # It's often faster for small-to-medium tasks than full multiprocess prefetching.
        self.loader = ds.to_iter_dataset(
            grain.ReadOptions(
                num_threads=max(1, num_workers),
                prefetch_buffer_size=batch_size * worker_buffer_size
            )
        )
        
        # 5. Batching
        self.loader = self.loader.batch(batch_size, drop_remainder=True)
        
        # 6. Final Thread Prefetch
        self.loader = grain.experimental.ThreadPrefetchIterDataset(
            self.loader, 
            prefetch_buffer_size=worker_buffer_size
        )

    def __iter__(self):
        return iter(self.loader)
