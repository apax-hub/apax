import numpy as np
import grain.python as grain
from apax.data.preprocessing import compute_nl

class SoADataSource(grain.RandomAccessDataSource):
    """
    A Grain DataSource that operates on a Structure of Arrays (SoA).
    Supports both pre-padded arrays and lists of ragged arrays.
    """
    def __init__(self, data: dict):
        self._data = data
        first_val = next(iter(data.values()))
        self._num_samples = len(first_val)

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

class PaddingTransform(grain.MapTransform):
    """
    Pads positions, numbers, and forces to a fixed number of atoms.
    """
    def __init__(self, max_atoms: int):
        self.max_atoms = max_atoms

    def map(self, sample: dict):
        n_atoms = len(sample["numbers"])
        pad_width = self.max_atoms - n_atoms
        
        if pad_width > 0:
            sample["positions"] = np.pad(sample["positions"], ((0, pad_width), (0, 0)))
            sample["numbers"] = np.pad(sample["numbers"], (0, pad_width)).astype(np.int32)
            if "forces" in sample:
                sample["forces"] = np.pad(sample["forces"], ((0, pad_width), (0, 0)))
        
        sample["n_atoms"] = n_atoms
        return sample

class ApaxGrainDataLoader:
    """
    A high-level wrapper for the Grain DataLoader with support for fixed and bucketed padding.
    """
    def __init__(
        self,
        data: dict,
        batch_size: int,
        cutoff: float,
        max_nbrs: int = None,
        bucket_boundaries: list[int] = None,
        num_epochs: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        worker_buffer_size: int = 1,
    ):
        self.cutoff = cutoff
        if bucket_boundaries is None:
            # Fixed padding approach
            max_atoms = np.max([len(x) for x in data["numbers"]])
            if max_nbrs is None:
                max_nbrs = self._find_max_nbrs(data)
            self.loader = self._create_dataset(
                data, batch_size, cutoff, max_atoms, max_nbrs, 
                num_epochs, shuffle, num_workers, worker_buffer_size
            )
        else:
            # Bucketed padding approach
            buckets = self._partition_data(data, bucket_boundaries)
            bucket_datasets = []
            for _, b_data in buckets.items():
                if len(b_data["numbers"]) >= batch_size:
                    # For each bucket, use the actual max atoms/neighbors in THAT bucket
                    b_max_atoms = np.max([len(x) for x in b_data["numbers"]])
                    b_max_nbrs = max_nbrs if max_nbrs else self._find_max_nbrs(b_data)
                    
                    ds = self._create_dataset(
                        b_data, batch_size, cutoff, b_max_atoms, b_max_nbrs,
                        num_epochs, shuffle, num_workers=0, worker_buffer_size=1
                    )
                    bucket_datasets.append(ds)
            
            if not bucket_datasets:
                raise ValueError("No buckets found with enough samples for batch_size.")

            # Interleave buckets
            self.loader = grain.experimental.InterleaveIterDataset(
                bucket_datasets, 
                cycle_length=len(bucket_datasets)
            )
            
            if num_workers > 0:
                self.loader = self.loader.mp_prefetch(
                    grain.MultiprocessingOptions(num_workers=num_workers, per_worker_buffer_size=worker_buffer_size)
                )

    def _find_max_nbrs(self, data):
        max_nbrs = 0
        for pos, box in zip(data["positions"], data["box"]):
            idx, _ = compute_nl(pos, box, self.cutoff)
            max_nbrs = max(max_nbrs, idx.shape[1])
        return max_nbrs

    def _partition_data(self, data, boundaries):
        boundaries = sorted(boundaries) + [float('inf')]
        buckets = {b: {k: [] for k in data.keys()} for b in boundaries}
        
        for i in range(len(data["numbers"])):
            n = len(data["numbers"][i])
            for b in boundaries:
                if n <= b:
                    for k in data.keys():
                        buckets[b][k].append(data[k][i])
                    break
        
        return {b: d for b, d in buckets.items() if len(d["numbers"]) > 0}

    def _create_dataset(
        self, data, batch_size, cutoff, max_atoms, max_nbrs,
        num_epochs, shuffle, num_workers, worker_buffer_size
    ):
        ds = grain.MapDataset.source(SoADataSource(data))
        if shuffle:
            ds = ds.shuffle(seed=42)
        if num_epochs > 1:
            ds = ds.repeat(num_epochs)
        
        ds = ds.map(PaddingTransform(max_atoms))
        ds = ds.map(NeighborListTransform(cutoff, max_nbrs))
        
        it_ds = ds.to_iter_dataset(
            grain.ReadOptions(
                num_threads=max(1, num_workers),
                prefetch_buffer_size=batch_size * worker_buffer_size
            )
        )
        it_ds = it_ds.batch(batch_size, drop_remainder=True)
        
        return grain.experimental.ThreadPrefetchIterDataset(it_ds, prefetch_buffer_size=worker_buffer_size)

    def __iter__(self):
        return iter(self.loader)
