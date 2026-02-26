import grain.python as grain
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from apax.data.preprocessing import compute_nl, prefetch_to_single_device
from apax.utils.convert import atoms_to_inputs, atoms_to_labels, unit_dict

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

class InputLabelSplitTransform(grain.MapTransform):
    """
    Splits the sample dictionary into (inputs, labels) tuples.
    """
    def map(self, sample: dict):
        input_keys = ["positions", "numbers", "box", "idx", "offsets", "n_atoms"]
        inputs = {k: sample[k] for k in input_keys if k in sample}
        labels = {k: v for k, v in sample.items() if k not in input_keys}
        return (inputs, labels)

class ApaxGrainDataLoader:
    """
    A high-level wrapper for the Grain DataLoader with support for fixed and bucketed padding.

    Args:
        atoms_list: List of ASE Atoms objects.
        cutoff: Cutoff radius for the neighbor list.
        bs: Batch size.
        n_epochs: Number of epochs to repeat the data.
        pos_unit: Unit for positions (default: "Ang").
        energy_unit: Unit for energy (default: "eV").
        pre_shuffle: Whether to shuffle the data before batching (default: False).
        additional_properties: List of additional properties to extract.
        num_workers: Number of multiprocessing workers for prefetching.
        worker_buffer_size: Prefetch buffer size per worker.
        bucket_boundaries: List of atom counts defining bucket sizes for ragged data.
        max_nbrs: Maximum number of neighbors for padding. If None, it is computed from the data.
    """
    def __init__(
        self,
        atoms_list,
        cutoff: float,
        bs: int,
        n_epochs: int,
        pos_unit: str = "Ang",
        energy_unit: str = "eV",
        pre_shuffle: bool = False,
        additional_properties: list = [],
        num_workers: int = 0,
        worker_buffer_size: int = 1,
        bucket_boundaries: list[int] = None,
        max_nbrs: int = None,
    ):
        self.cutoff = cutoff
        self.batch_size = bs
        self.n_epochs = n_epochs
        self.num_workers = num_workers
        self.worker_buffer_size = worker_buffer_size
        self.bucket_boundaries = bucket_boundaries
        self.max_nbrs = max_nbrs
        
        # Prepare raw data
        self._inputs = atoms_to_inputs(atoms_list, pos_unit)
        self._labels = atoms_to_labels(atoms_list, pos_unit, energy_unit, additional_properties)
        self.data = {**self._inputs, **self._labels}
        self.n_data = len(atoms_list)
        self.sample_atoms = atoms_list[0]

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    def steps_per_epoch(self) -> int:
        return self.n_data // self.batch_size

    def init_input(self):
        """Returns first batch of inputs and labels to init the model."""
        positions = self.sample_atoms.positions * unit_dict["Ang"] # default
        box = self.sample_atoms.cell.array * unit_dict["Ang"]
        idx, offsets = compute_nl(positions, box, self.cutoff)
        inputs = (
            positions,
            self.sample_atoms.numbers,
            idx,
            box,
            offsets,
        )
        inputs = tree_util.tree_map(lambda x: jnp.array(x), inputs)
        return inputs, np.array(box)

    def shuffle_and_batch(self, mesh=None):
        return self._get_iterator(shuffle=True, mesh=mesh)

    def batch(self, mesh=None):
        return self._get_iterator(shuffle=False, mesh=mesh)

    def _get_iterator(self, shuffle=True, mesh=None):
        if self.bucket_boundaries is None:
            max_atoms = np.max([len(x) for x in self.data["numbers"]])
            m_nbrs = self.max_nbrs if self.max_nbrs else self._find_max_nbrs(self.data)
            loader = self._create_dataset(
                self.data, self.batch_size, self.cutoff, max_atoms, m_nbrs, 
                self.n_epochs, shuffle, self.num_workers, self.worker_buffer_size
            )
        else:
            buckets = self._partition_data(self.data, self.bucket_boundaries)
            bucket_datasets = []
            for _, b_data in buckets.items():
                if len(b_data["numbers"]) >= self.batch_size:
                    b_max_atoms = np.max([len(x) for x in b_data["numbers"]])
                    b_max_nbrs = self.max_nbrs if self.max_nbrs else self._find_max_nbrs(b_data)
                    ds = self._create_dataset(
                        b_data, self.batch_size, self.cutoff, b_max_atoms, b_max_nbrs,
                        self.n_epochs, shuffle, num_workers=0, worker_buffer_size=1
                    )
                    bucket_datasets.append(ds)
            
            loader = grain.experimental.InterleaveIterDataset(
                bucket_datasets, cycle_length=len(bucket_datasets)
            )
            if self.num_workers > 0:
                loader = loader.mp_prefetch(
                    grain.MultiprocessingOptions(
                        num_workers=self.num_workers, 
                        per_worker_buffer_size=self.worker_buffer_size
                    )
                )

        if mesh:
            data_sharding = NamedSharding(mesh, P("data"))
        else:
            data_sharding = None
            
        return prefetch_to_single_device(iter(loader), 2, data_sharding)

    def _find_max_nbrs(self, data):
        max_nbrs = 0
        # Use a small sample to find max_nbrs if data is large? 
        # For now, consistent with legacy, we check all.
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
        ds = ds.map(InputLabelSplitTransform())
        
        it_ds = ds.to_iter_dataset(
            grain.ReadOptions(
                num_threads=max(1, num_workers),
                prefetch_buffer_size=batch_size * worker_buffer_size
            )
        )
        it_ds = it_ds.batch(batch_size, drop_remainder=True)
        return grain.experimental.ThreadPrefetchIterDataset(it_ds, prefetch_buffer_size=worker_buffer_size)

    def cleanup(self):
        pass

    def __iter__(self):
        return self.batch()
