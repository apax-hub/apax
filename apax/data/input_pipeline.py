import logging
import multiprocessing
import time
import uuid
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from random import shuffle
from threading import Event
from typing import Dict, Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax import tree_util
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from apax.data.preprocessing import compute_nl, prefetch_to_single_device
from apax.utils.convert import (
    atoms_to_inputs,
    atoms_to_labels,
    transpose_dict_of_lists,
    unit_dict,
)

log = logging.getLogger(__name__)


def pad_nl(idx, offsets, max_neighbors):
    """
    Pad the neighbor list arrays to the maximal number of neighbors occurring.

    Parameters
    ----------
    idx : np.ndarray
        Neighbor indices array.
    offsets : np.ndarray
        Offset array.
    max_neighbors : int
        Maximum number of neighbors.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing padded neighbor indices array and offsets array.
    """
    zeros_to_add = max_neighbors - idx.shape[1]
    idx = np.pad(idx, ((0, 0), (0, zeros_to_add)), "constant").astype(np.int16)
    offsets = np.pad(offsets, ((0, zeros_to_add), (0, 0)), "constant")
    return idx, offsets


def find_largest_system(inputs, r_max) -> tuple[int]:
    """
    Finds the maximal number of atoms and neighbors.

    Parameters
    ----------
    inputs : dict
        Dictionary containing input data.
    r_max : float
        Maximum interaction radius.

    Returns
    -------
    Tuple[int]
        Tuple containing the maximum number of atoms and neighbors.
    """
    positions, boxes = inputs["positions"], inputs["box"]
    max_atoms = np.max(inputs["n_atoms"])

    max_nbrs = 0
    for pos, box in zip(positions, boxes):
        neighbor_idxs, _ = compute_nl(pos, box, r_max)
        n_neighbors = neighbor_idxs.shape[1]
        max_nbrs = max(max_nbrs, n_neighbors)

    return max_atoms, max_nbrs


class InMemoryDataset:
    """Baseclass for all datasets which store data in memory."""

    def __init__(
        self,
        atoms_list,
        cutoff,
        bs,
        n_epochs,
        pos_unit: str = "Ang",
        energy_unit: str = "eV",
        additional_properties: list[tuple] = [],
        pre_shuffle=False,
        shuffle_buffer_size=1000,
        ignore_labels=False,
        cache_path=".",
    ) -> None:
        self.n_epochs = n_epochs
        self.cutoff = cutoff
        self.buffer_size = shuffle_buffer_size
        self.n_data = len(atoms_list)
        self.batch_size = self.validate_batch_size(bs)
        self.pos_unit = pos_unit
        self.additional_properties = additional_properties

        if pre_shuffle:
            shuffle(atoms_list)
        self.sample_atoms = atoms_list[0]
        self.inputs = atoms_to_inputs(atoms_list, pos_unit)

        max_atoms, max_nbrs = find_largest_system(self.inputs, self.cutoff)
        self.max_atoms = max_atoms
        self.max_nbrs = max_nbrs
        if atoms_list[0].calc and not ignore_labels:
            self.labels = atoms_to_labels(
                atoms_list, pos_unit, energy_unit, additional_properties
            )
        else:
            self.labels = None

        self.count = 0
        self.buffer = deque()
        self.file = Path(cache_path) / str(uuid.uuid4())

        self.enqueue(min(self.buffer_size, self.n_data))

    def steps_per_epoch(self) -> int:
        """Returns the number of steps per epoch dependent on the number of data and the
        batch size. Steps per epoch are calculated in a way that all epochs have the same
        number of steps, and all batches have the same length. To do so, some training
        data are dropped in each epoch.
        """
        return self.n_data // self.batch_size

    def validate_batch_size(self, batch_size: int) -> int:
        if batch_size > self.n_data:
            msg = (
                f"requested batch size {batch_size} is larger than the number of data"
                f" points {self.n_data}. Setting batch size = {self.n_data}"
            )
            log.warning(msg)
            batch_size = self.n_data
        return batch_size

    def prepare_data(self, i):
        inputs = {k: v[i] for k, v in self.inputs.items()}
        idx, offsets = compute_nl(inputs["positions"], inputs["box"], self.cutoff)
        inputs["idx"], inputs["offsets"] = pad_nl(idx, offsets, self.max_nbrs)

        zeros_to_add = self.max_atoms - inputs["numbers"].shape[0]
        inputs["positions"] = np.pad(
            inputs["positions"], ((0, zeros_to_add), (0, 0)), "constant"
        )
        inputs["numbers"] = np.pad(
            inputs["numbers"], (0, zeros_to_add), "constant"
        ).astype(np.int16)

        if not self.labels:
            return inputs

        labels = {k: v[i] for k, v in self.labels.items()}
        if "forces" in labels:
            labels["forces"] = np.pad(
                labels["forces"], ((0, zeros_to_add), (0, 0)), "constant"
            )

        for prop in self.additional_properties:
            name, shape = prop
            pad_width = []
            for s in shape:
                if s == "natoms":
                    pad_width.append((0, zeros_to_add))
                else:
                    pad_width.append((0, 0))

            if any(p[1] > 0 for p in pad_width):
                labels[name] = np.pad(labels[name], pad_width, "constant")

        inputs = {k: tf.constant(v) for k, v in inputs.items()}
        labels = {k: tf.constant(v) for k, v in labels.items()}

        return (inputs, labels)

    def enqueue(self, num_elements):
        for _ in range(num_elements):
            data = self.prepare_data(self.count)
            self.buffer.append(data)
            self.count += 1

    def make_signature(self) -> tf.TensorSpec:
        input_signature = {}
        input_signature["n_atoms"] = tf.TensorSpec((), dtype=tf.int16, name="n_atoms")
        input_signature["numbers"] = tf.TensorSpec(
            (self.max_atoms,), dtype=tf.int16, name="numbers"
        )
        input_signature["positions"] = tf.TensorSpec(
            (self.max_atoms, 3), dtype=tf.float64, name="positions"
        )
        input_signature["box"] = tf.TensorSpec((3, 3), dtype=tf.float64, name="box")
        input_signature["idx"] = tf.TensorSpec(
            (2, self.max_nbrs), dtype=tf.int16, name="idx"
        )
        input_signature["offsets"] = tf.TensorSpec(
            (self.max_nbrs, 3), dtype=tf.float64, name="offsets"
        )

        if not self.labels:
            return input_signature

        label_signature = {}
        if "energy" in self.labels.keys():
            label_signature["energy"] = tf.TensorSpec((), dtype=tf.float64, name="energy")
        if "forces" in self.labels.keys():
            label_signature["forces"] = tf.TensorSpec(
                (self.max_atoms, 3), dtype=tf.float64, name="forces"
            )
        if "stress" in self.labels.keys():
            label_signature["stress"] = tf.TensorSpec(
                (3, 3), dtype=tf.float64, name="stress"
            )

        for prop in self.additional_properties:
            name, shape = prop
            actual_shape = [self.max_atoms if s == "natoms" else s for s in shape]

            sig = tf.TensorSpec(tuple(actual_shape), dtype=tf.float64, name=name)
            label_signature[name] = sig
        signature = (input_signature, label_signature)
        return signature

    def init_input(self) -> tuple[Dict[str, jnp.ndarray], np.ndarray]:
        """Returns first batch of inputs and labels to init the model."""
        positions = self.sample_atoms.positions * unit_dict[self.pos_unit]
        box = self.sample_atoms.cell.array * unit_dict[self.pos_unit]
        # For an input sample, it does not matter whether pos is fractional or cartesian
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

    def __iter__(self):
        raise NotImplementedError

    def shuffle_and_batch(self, mesh=None):
        raise NotImplementedError

    def batch(self, mesh=None) -> Iterator[jax.Array]:
        raise NotImplementedError

    def cleanup(self):
        pass


class CachedInMemoryDataset(InMemoryDataset):
    """Dataset which pads everything (atoms, neighbors)
    to the largest system in the dataset.
    The NL is computed on the fly during the first epoch and stored to disk using
    tf.data's cache.
    Most performant option for datasets with samples of very similar size.
    """

    def __iter__(self):
        while self.count < self.n_data or len(self.buffer) > 0:
            yield self.buffer.popleft()

            space = self.buffer_size - len(self.buffer)
            if self.count + space > self.n_data:
                space = self.n_data - self.count
            self.enqueue(space)

    def shuffle_and_batch(self, mesh=None):
        """Shuffles and batches the inputs/labels. This function prepares the
        inputs and labels for the whole training and prefetches the data.

        Returns
        -------
        ds :
            Iterator that returns inputs and labels of one batch in each step.
        """
        ds = (
            tf.data.Dataset.from_generator(
                lambda: self, output_signature=self.make_signature()
            )
            .cache(self.file.as_posix())
            .repeat(self.n_epochs)
        )

        ds = ds.shuffle(
            buffer_size=self.buffer_size, reshuffle_each_iteration=True
        ).batch(batch_size=self.batch_size)
        if mesh:
            data_sharding = NamedSharding(mesh, P("data"))
        else:
            data_sharding = None
        ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2, data_sharding)
        return ds

    def batch(self, mesh=None) -> Iterator[jax.Array]:
        ds = (
            tf.data.Dataset.from_generator(
                lambda: self, output_signature=self.make_signature()
            )
            .cache(self.file.as_posix())
            .repeat(self.n_epochs)
        )
        ds = ds.batch(batch_size=self.batch_size)
        if mesh:
            data_sharding = NamedSharding(mesh, P("data"))
        else:
            data_sharding = None
        ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2, data_sharding)
        return ds

    def cleanup(self):
        for p in self.file.parent.glob(f"{self.file.name}.data*"):
            p.unlink()

        index_file = self.file.parent / f"{self.file.name}.index"
        index_file.unlink()


class OTFInMemoryDataset(InMemoryDataset):
    """Dataset which pads everything (atoms, neighbors)
    to the largest system in the dataset.
    The NL is computed on the fly and fed into a tf.data generator.
    Mostly for internal purposes.
    """

    def __iter__(self):
        outer_count = 0
        max_iter = self.n_data * self.n_epochs
        while outer_count < max_iter:
            yield self.buffer.popleft()

            space = self.buffer_size - len(self.buffer)
            if self.count + space > self.n_data:
                space = self.n_data - self.count

            if self.count >= self.n_data:
                self.count = 0
            self.enqueue(space)
            outer_count += 1

    def shuffle_and_batch(self, mesh=None):
        """Shuffles and batches the inputs/labels. This function prepares the
        inputs and labels for the whole training and prefetches the data.

        Returns
        -------
        ds :
            Iterator that returns inputs and labels of one batch in each step.
        """
        ds = tf.data.Dataset.from_generator(
            lambda: self, output_signature=self.make_signature()
        )

        ds = ds.shuffle(
            buffer_size=self.buffer_size, reshuffle_each_iteration=True
        ).batch(batch_size=self.batch_size)
        if mesh:
            data_sharding = NamedSharding(mesh, P("data"))
        else:
            data_sharding = None
        ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2, data_sharding)
        return ds

    def batch(self, mesh=None) -> Iterator[jax.Array]:
        ds = tf.data.Dataset.from_generator(
            lambda: self, output_signature=self.make_signature()
        )
        ds = ds.batch(batch_size=self.batch_size)
        if mesh:
            data_sharding = NamedSharding(mesh, P("data"))
        else:
            data_sharding = None
        ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2, data_sharding)
        return ds


def next_power_of_two(x):
    return 1 << (int(x) - 1).bit_length()


def round_up_to_multiple(value, multiple):
    """
    Rounds up the given integer `value` to the next multiple of `multiple`.

    Parameters:
    - value (int): The integer to round up.
    - multiple (int): The multiple to round up to.

    Returns:
    - int: The rounded-up value.
    """
    return int(np.ceil(value / multiple) * multiple)


class BatchProcessor:
    def __init__(
        self,
        cutoff,
        atom_padding: int,
        nl_padding: int,
        forces=True,
        stress=False,
        additional_properties=[],
    ) -> None:
        self.cutoff = cutoff
        self.atom_padding = atom_padding
        self.nl_padding = nl_padding

        self.forces = forces
        self.stress = stress
        self.additional_properties = additional_properties

    def __call__(self, samples: list[dict], shape: Optional[tuple[int, int]] = None):
        """`shape` = (atoms, pairs) to pad to (size buckets); otherwise pad to the batch
        maximum rounded up to `atom_padding` / `nl_padding`."""
        n_samples = len(samples)
        if shape is None:
            max_atoms = np.max([inp[0]["n_atoms"] for inp in samples])
            max_atoms = round_up_to_multiple(max_atoms, self.atom_padding)
        else:
            max_atoms = shape[0]

        inputs = {
            "numbers": np.zeros((n_samples, max_atoms), dtype=np.int16),
            "n_atoms": np.zeros(n_samples, dtype=np.int16),
            "positions": np.zeros((n_samples, max_atoms, 3), dtype=np.float64),
            "box": np.zeros((n_samples, 3, 3), dtype=np.float32),
        }

        labels = {
            "energy": np.zeros(n_samples, dtype=np.float64),
        }
        for prop in self.additional_properties:
            name, shape = prop
            if shape[0] == "natoms":
                shape = [max_atoms] + shape[1:]
            shape = [n_samples] + shape
            labels[name] = np.zeros(shape, dtype=np.float64)
        if self.forces:
            labels["forces"] = np.zeros((n_samples, max_atoms, 3), dtype=np.float64)
        if self.stress:
            labels["stress"] = np.zeros((n_samples, 3, 3), dtype=np.float64)

        idxs = []
        offsets = []
        for i, (inp, lab) in enumerate(samples):
            inputs["numbers"][i, : inp["n_atoms"]] = inp["numbers"]
            inputs["n_atoms"][i] = inp["n_atoms"]
            inputs["positions"][i, : inp["n_atoms"]] = inp["positions"]
            inputs["box"][i] = inp["box"]

            idx, offset = compute_nl(inp["positions"], inp["box"], self.cutoff)
            idxs.append(idx)
            offsets.append(offset)

            labels["energy"][i] = lab["energy"]
            if self.forces:
                labels["forces"][i, : inp["n_atoms"]] = lab["forces"]
            if self.stress:
                labels["stress"][i] = lab["stress"]

            for prop in self.additional_properties:
                name, shape = prop
                if shape[0] == "natoms":
                    labels[name][i, : inp["n_atoms"]] = lab[name]
                else:
                    labels[name][i] = lab[name]

        if shape is None:
            max_nbrs = np.max([idx.shape[1] for idx in idxs])
            max_nbrs = round_up_to_multiple(max_nbrs, self.nl_padding)
        else:
            max_nbrs = shape[1]

        inputs["idx"] = np.zeros((n_samples, 2, max_nbrs), dtype=np.int16)
        inputs["offsets"] = np.zeros((n_samples, max_nbrs, 3), dtype=np.float64)

        for i, (idx, offset) in enumerate(zip(idxs, offsets)):
            inputs["idx"][i, :, : idx.shape[1]] = idx
            inputs["offsets"][i, : offset.shape[0], :] = offset

        return inputs, labels


def size_buckets(n_atoms, n_pairs, ratio, atom_quantile=0.99):
    """Geometric pair-count buckets with one static (atoms, pairs) padding shape each.

    Pairs pad to the bucket's upper edge. Atoms pad to the `atom_quantile` of the bucket,
    made non-decreasing over buckets; structures with more atoms move to the next bucket
    that fits them. Returns (bucket index per structure, list of (atoms, pairs) shapes).
    """
    n_atoms = np.asarray(n_atoms)
    n_pairs = np.maximum(np.asarray(n_pairs), 1)
    edges = [1]
    while edges[-1] < n_pairs.max():
        edges.append(max(edges[-1] + 1, int(np.ceil(edges[-1] * ratio))))
    edges = np.array(edges)
    bucket = np.searchsorted(edges, n_pairs)
    atom_pad = np.zeros(len(edges), dtype=int)
    for k in np.unique(bucket):
        atom_pad[k] = int(np.ceil(np.quantile(n_atoms[bucket == k], atom_quantile)))
    atom_pad = np.maximum.accumulate(atom_pad)
    for k in range(len(edges) - 1):
        bucket[(bucket == k) & (n_atoms > atom_pad[k])] = k + 1
    atom_pad[-1] = max(atom_pad[-1], n_atoms[bucket == len(edges) - 1].max(initial=0))
    shapes = [(int(a), int(p)) for a, p in zip(atom_pad, edges)]
    return bucket, shapes


class PerBatchPaddedDataset(InMemoryDataset):
    """Dataset with padding that leverages multiprocessing and optimized buffering.

    Per-atom and per-neighbor arrays are padded to the next multiple of a user specified integer.
    This limits the compute wasted due to padding at the (negligible) cost of some recompilations.
    Since the padding occurs on a per-batch basis, it is the most performant option for datasets with significantly differently sized systems (e.g. MaterialsProject, SPICE).

    Further, the neighborlist is computed on-the-fly in parallel on a side thread.
    Does not use tf.data.

    Attributes
    ----------
    num_workers : int
        Number of processes to use for preprocessing batches.
    atom_padding : int
        Pad extensive arrays (positions, etc.) to next multiple of this integer.
    nl_padding : int
        Pad neighborlist arrays to next multiple of this integer.
    bucket_ratio : float, optional
        If set, batch structures of similar size together: geometric neighbour-pair
        buckets of this width ratio, each padded to one fixed shape (see `size_buckets`).
    """

    def __init__(
        self,
        atoms_list,
        cutoff,
        bs,
        n_epochs,
        num_workers: Optional[int] = None,
        atom_padding: int = 10,
        nl_padding: int = 2000,
        pos_unit: str = "Ang",
        energy_unit: str = "eV",
        additional_properties=[],
        pre_shuffle=False,
        bucket_ratio: Optional[float] = None,
    ) -> None:
        self.cutoff = cutoff
        self.n_epochs = n_epochs
        self.n_data = len(atoms_list)
        self.batch_size = self.validate_batch_size(bs)
        self.pos_unit = pos_unit
        self.additional_properties = additional_properties

        if num_workers:
            self.num_workers = num_workers
        else:
            self.num_workers = multiprocessing.cpu_count() - 1

        self.sample_atoms = atoms_list[0]

        # Transform atoms into inputs and labels
        self.inputs = atoms_to_inputs(atoms_list, pos_unit)
        self.labels = atoms_to_labels(
            atoms_list, pos_unit, energy_unit, additional_properties
        )
        label_keys = self.labels.keys()

        self.data = list(
            zip(
                transpose_dict_of_lists(self.inputs), transpose_dict_of_lists(self.labels)
            )
        )

        forces = "forces" in label_keys
        stress = "stress" in label_keys
        self.prepare_batch = BatchProcessor(
            cutoff, atom_padding, nl_padding, forces, stress, additional_properties
        )

        self.bucket_shapes = None
        if bucket_ratio is not None:
            n_pairs = [
                compute_nl(inp["positions"], inp["box"], cutoff)[0].shape[1]
                for inp, _ in self.data
            ]
            n_atoms = [inp["n_atoms"] for inp, _ in self.data]
            buckets, self.bucket_shapes = size_buckets(n_atoms, n_pairs, bucket_ratio)
            self.data = [(inp, lab, int(b)) for (inp, lab), b in zip(self.data, buckets)]
            used = sorted(set(buckets.tolist()))
            log.info(
                f"Size buckets: {len(used)} shapes {[self.bucket_shapes[k] for k in used]}"
            )
        self.epoch_batches = None

        self.count = 0

        self.max_count = self.n_epochs * self.steps_per_epoch()

        self.buffer_size = min(600, self.steps_per_epoch())
        self.buffer = Queue(maxsize=self.buffer_size)

        self.process_pool = ProcessPoolExecutor(self.num_workers)
        self.thread_pool = ThreadPoolExecutor(1)  # Single thread for buffering batches
        self.epoch_finished = False
        self.enqueue_future = None
        self.needs_data = Event()

    def enqueue_batches(self):
        """Function to enqueue batches on a side thread."""
        while self.count < self.steps_per_epoch() * self.n_epochs:
            self.needs_data.wait()
            if self.epoch_finished:
                break
            num_batches = min(
                self.buffer_size - self.buffer.qsize(),
                self.steps_per_epoch() - self.count,
            )
            if num_batches > 0:
                self.enqueue(num_batches)
            self.needs_data.clear()  # Reset event

    def bucket_batches(self):
        """One epoch of size-bucketed batches: full batches within each bucket; the
        leftovers of all buckets, sorted by size, chunked and padded to the shape of their
        largest member's bucket. Batch order is shuffled across buckets."""
        bs = self.batch_size
        by_bucket = {}
        for inp, lab, b in self.data:  # self.data is already shuffled
            by_bucket.setdefault(b, []).append((inp, lab))
        batches, leftovers = [], []
        for b, items in by_bucket.items():
            n_full = len(items) // bs * bs
            batches += [(items[i : i + bs], b) for i in range(0, n_full, bs)]
            leftovers += [(it, b) for it in items[n_full:]]
        leftovers.sort(key=lambda x: x[1])
        for i in range(0, len(leftovers) // bs * bs, bs):
            chunk = leftovers[i : i + bs]
            batches.append(([it for it, _ in chunk], max(b for _, b in chunk)))
        shuffle(batches)
        return [(items, self.bucket_shapes[b]) for items, b in batches]

    def enqueue(self, num_batches):
        if self.bucket_shapes is not None:
            chunks = self.epoch_batches[self.count : self.count + num_batches]
        else:
            start = self.count * self.batch_size
            chunks = [
                (
                    self.data[
                        start + self.batch_size * i : start + self.batch_size * (i + 1)
                    ],
                    None,
                )
                for i in range(num_batches)
            ]

        # Using submit and as_completed for faster batch retrieval
        futures = [
            self.process_pool.submit(self.prepare_batch, chunk, shape)
            for chunk, shape in chunks
        ]
        for future in as_completed(futures):
            batch = future.result()
            self.buffer.put(batch)

        self.count += num_batches

    def __iter__(self):
        for n in range(self.n_epochs):
            self.count = 0
            self.buffer.queue.clear()  # Reset buffer
            self.epoch_finished = False

            if self.should_shuffle:
                shuffle(self.data)
            if self.bucket_shapes is not None:
                self.epoch_batches = self.bucket_batches()

            # Start pre-filling the buffer
            self.enqueue_future = self.thread_pool.submit(self.enqueue_batches)

            for i in range(self.steps_per_epoch()):
                if self.buffer.qsize() < (self.buffer_size * 0.75):
                    self.needs_data.set()  # Trigger buffer refill
                while self.buffer.empty():
                    time.sleep(0.001)
                yield self.buffer.get()

            self.epoch_finished = True
            self.needs_data.set()
            self.enqueue_future.result()

    def shuffle_and_batch(self, mesh):
        self.should_shuffle = True
        if mesh:
            data_sharding = NamedSharding(mesh, P("data"))
        else:
            data_sharding = None
        ds = prefetch_to_single_device(iter(self), 2, data_sharding)
        return ds

    def batch(self, mesh) -> Iterator[jax.Array]:
        self.should_shuffle = False
        if mesh:
            data_sharding = NamedSharding(mesh, P("data"))
        else:
            data_sharding = None
        ds = prefetch_to_single_device(iter(self), 2, data_sharding)
        return ds

    def make_signature(self) -> None:
        pass

    def cleanup(self):
        self.epoch_finished = True
        self.needs_data.set()
        if self.enqueue_future is not None:  # None if the dataset was never iterated
            self.enqueue_future.result()
        self.needs_data.clear()
        self.thread_pool.shutdown(wait=True, cancel_futures=True)
        self.process_pool.shutdown(wait=True, cancel_futures=True)
        self.buffer.queue.clear()


dataset_dict = {
    "cached": CachedInMemoryDataset,
    "otf": OTFInMemoryDataset,
    "pbp": PerBatchPaddedDataset,
}
