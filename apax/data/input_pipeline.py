import logging
import uuid
from collections import deque
from pathlib import Path
from random import shuffle
from typing import Dict, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from apax.data.preprocessing import compute_nl, prefetch_to_single_device
from apax.utils.convert import atoms_to_inputs, atoms_to_labels

log = logging.getLogger(__name__)


def pad_nl(idx, offsets, max_neighbors):
    zeros_to_add = max_neighbors - idx.shape[1]
    idx = np.pad(idx, ((0, 0), (0, zeros_to_add)), "constant").astype(np.int16)
    offsets = np.pad(offsets, ((0, zeros_to_add), (0, 0)), "constant")
    return idx, offsets


def find_largest_system(inputs: dict[str, np.ndarray], r_max) -> tuple[int]:
    max_atoms = np.max(inputs["n_atoms"])

    max_nbrs = 0
    for position, box in zip(inputs["positions"], inputs["box"]):
        neighbor_idxs, _ = compute_nl(position, box, r_max)
        n_neighbors = neighbor_idxs.shape[1]
        max_nbrs = max(max_nbrs, n_neighbors)

    return max_atoms, max_nbrs


class InMemoryDataset:
    def __init__(
        self,
        atoms,
        cutoff,
        bs,
        n_epochs,
        buffer_size=1000,
        n_jit_steps=1,
        pre_shuffle=False,
        ignore_labels=False,
        cache_path=".",
    ) -> None:
        if pre_shuffle:
            shuffle(atoms)
        self.sample_atoms = atoms[0]
        self.inputs = atoms_to_inputs(atoms)

        self.n_epochs = n_epochs
        self.buffer_size = buffer_size

        max_atoms, max_nbrs = find_largest_system(self.inputs, cutoff)
        self.max_atoms = max_atoms
        self.max_nbrs = max_nbrs

        if atoms[0].calc and not ignore_labels:
            self.labels = atoms_to_labels(atoms)
        else:
            self.labels = None

        self.n_data = len(atoms)
        self.count = 0
        self.cutoff = cutoff
        self.buffer = deque()
        self.batch_size = self.validate_batch_size(bs)
        self.n_jit_steps = n_jit_steps
        self.file = Path(cache_path) / str(uuid.uuid4())

        self.enqueue(min(self.buffer_size, self.n_data))

    def steps_per_epoch(self) -> int:
        """Returns the number of steps per epoch dependent on the number of data and the
        batch size. Steps per epoch are calculated in a way that all epochs have the same
        number of steps, and all batches have the same length. To do so, some training
        data are dropped in each epoch.
        """
        return self.n_data // self.batch_size // self.n_jit_steps

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
        inputs["n_atoms"] = np.pad(
            inputs["n_atoms"], (0, zeros_to_add), "constant"
        ).astype(np.int16)

        if not self.labels:
            return inputs

        labels = {k: v[i] for k, v in self.labels.items()}
        if "forces" in labels:
            labels["forces"] = np.pad(
                labels["forces"], ((0, zeros_to_add), (0, 0)), "constant"
            )

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
        signature = (input_signature, label_signature)
        return signature

    def init_input(self) -> Dict[str, np.ndarray]:
        """Returns first batch of inputs and labels to init the model."""
        positions = self.sample_atoms.positions
        box = self.sample_atoms.cell.array
        idx, offsets = compute_nl(positions, box, self.cutoff)
        inputs = (
            positions,
            self.sample_atoms.numbers,
            idx,
            box,
            offsets,
        )

        inputs = jax.tree_map(lambda x: jnp.array(x), inputs)
        return inputs, np.array(box)

    def __iter__(self):
        raise NotImplementedError

    def shuffle_and_batch(self):
        raise NotImplementedError

    def batch(self) -> Iterator[jax.Array]:
        raise NotImplementedError

    def cleanup(self):
        pass


class CachedInMemoryDataset(InMemoryDataset):
    def __iter__(self):
        while self.count < self.n_data or len(self.buffer) > 0:
            yield self.buffer.popleft()

            space = self.buffer_size - len(self.buffer)
            if self.count + space > self.n_data:
                space = self.n_data - self.count
            self.enqueue(space)

    def shuffle_and_batch(self):
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
        if self.n_jit_steps > 1:
            ds = ds.batch(batch_size=self.n_jit_steps)
        ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2)
        return ds

    def batch(self) -> Iterator[jax.Array]:
        ds = (
            tf.data.Dataset.from_generator(
                lambda: self, output_signature=self.make_signature()
            )
            .cache(self.file.as_posix())
            .repeat(self.n_epochs)
        )
        ds = ds.batch(batch_size=self.batch_size)
        ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2)
        return ds

    def cleanup(self):
        for p in self.file.parent.glob(f"{self.file.name}.data*"):
            p.unlink()

        index_file = self.file.parent / f"{self.file.name}.index"
        index_file.unlink()


class OTFInMemoryDataset(InMemoryDataset):
    def __iter__(self):
        epoch = 0
        while epoch < self.n_epochs or len(self.buffer) > 0:
            yield self.buffer.popleft()

            space = self.buffer_size - len(self.buffer)
            if self.count + space > self.n_data:
                space = self.n_data - self.count

            if self.count >= self.n_data and epoch < self.n_epochs:
                epoch += 1
                self.count = 0
            self.enqueue(space)

    def shuffle_and_batch(self):
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
        if self.n_jit_steps > 1:
            ds = ds.batch(batch_size=self.n_jit_steps)
        ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2)
        return ds

    def batch(self) -> Iterator[jax.Array]:
        ds = tf.data.Dataset.from_generator(
            lambda: self, output_signature=self.make_signature()
        )
        ds = ds.batch(batch_size=self.batch_size)
        ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2)
        return ds


dataset_dict = {
    "cached": CachedInMemoryDataset,
    "otf": OTFInMemoryDataset,
}
