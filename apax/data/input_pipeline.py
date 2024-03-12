import logging
from typing import Dict, Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from apax.data.preprocessing import dataset_neighborlist, prefetch_to_single_device
from apax.utils.convert import atoms_to_inputs

log = logging.getLogger(__name__)


def find_largest_system(inputs: dict[str, np.ndarray]) -> tuple[int]:
    max_atoms = np.max(inputs["fixed"]["n_atoms"])
    nbr_shapes = [idx.shape[1] for idx in inputs["fixed"]["idx"]] # REMOVE
    max_nbrs = np.max(nbr_shapes)
    return max_atoms, max_nbrs


class PadToSpecificSize:
    def __init__(self, max_atoms: int, max_nbrs: int) -> None:
        """Function is padding all input and label dicts that values are of type ragged
        to largest element in the batch. Afterward, the distinction between ragged
        and fixed inputs/labels is not needed and all inputs/labels are updated to
        one list.

        Parameters
        ----------
        max_atoms: Number of atoms that atom-wise inputs will be padded to.
        max_nbrs: Number of neighbors that neighborlists will be padded to.
        """

        self.max_atoms = max_atoms
        self.max_nbrs = max_nbrs

    def __call__(self, inputs: dict, labels: dict = None) -> tuple[dict, dict]:
        """
        Arguments
        ---------

        r_inputs :
            Inputs of ragged shape.
        f_inputs :
            Inputs of fixed shape.
        r_labels :
            Labels of ragged shape. Trainable system properties.
        f_labels :
            Labels of fixed shape. Trainable system properties.

        Returns
        -------
        inputs:
            Contains all inputs and all entries are uniformly shaped.
        labels:
            Contains all labels and all entries are uniformly shaped.
        """
        r_inputs = inputs["ragged"]
        f_inputs = inputs["fixed"]
        for key, val in r_inputs.items():
            if self.max_atoms is None:
                r_inputs[key] = val.to_tensor()
            # elif key == "idx":
            #     shape = r_inputs[key].shape
            #     padded_shape = [shape[0], shape[1], self.max_nbrs]  # batch, ij, nbrs
            # elif key == "offsets":
            #     shape = r_inputs[key].shape
            #     padded_shape = [shape[0], self.max_nbrs, 3]  # batch, ij, nbrs # KILL
            elif key == "numbers":
                shape = r_inputs[key].shape
                padded_shape = [shape[0], self.max_atoms]  # batch, atoms
            else:
                shape = r_inputs[key].shape
                padded_shape = [shape[0], self.max_atoms, shape[2]]  # batch, atoms, 3
            r_inputs[key] = val.to_tensor(shape=padded_shape)

        new_inputs = r_inputs.copy()
        new_inputs.update(f_inputs)

        if labels:
            r_labels = labels["ragged"]
            f_labels = labels["fixed"]
            for key, val in r_labels.items():
                if self.max_atoms is None:
                    r_labels[key] = val.to_tensor()
                else:
                    shape = r_labels[key].shape
                    padded_shape = [shape[0], self.max_atoms, shape[2]]
                    r_labels[key] = val.to_tensor(default_value=0.0, shape=padded_shape)

            new_labels = r_labels.copy()
            new_labels.update(f_labels)

            return new_inputs, new_labels
        else:
            return new_inputs


def pad_neighborlist(idxs, offsets, max_neighbors):
    new_idxs = []
    new_offsets = []

    for idx, offset in zip(idxs, offsets):
        zeros_to_add = max_neighbors - idx.shape[1]
        new_idx = np.pad(idx, ((0, 0), (0, zeros_to_add)), "constant").astype(np.int16)
        new_offset = np.pad(offset, ((0, zeros_to_add), (0, 0)), "constant").astype(np.int16)
        new_idxs.append(new_idx)
        new_offsets.append(new_offset)

    return new_idxs, new_offsets


def process_inputs(
    atoms_list: list,
    r_max: float,
    disable_pbar=False,
    pos_unit: str = "Ang",
) -> dict:
    inputs = atoms_to_inputs(atoms_list, pos_unit) # find largest input
    idx, offsets, max_neighbors = dataset_neighborlist(
        inputs["ragged"]["positions"],
        inputs["fixed"]["box"],
        r_max=r_max,
        disable_pbar=disable_pbar,
    )

    idx, offsets = pad_neighborlist(idx, offsets, max_neighbors)

    inputs["fixed"]["idx"] = idx
    inputs["fixed"]["offsets"] = offsets
    return inputs


def dataset_from_dicts(
    inputs: Dict[str, np.ndarray], labels: Optional[Dict[str, np.ndarray]] = None
) -> tf.data.Dataset:
    # tf.RaggedTensors should be created from `tf.ragged.stack`
    # instead of `tf.ragged.constant` for performance reasons.
    # See https://github.com/tensorflow/tensorflow/issues/47853
    for key, val in inputs["ragged"].items():
        inputs["ragged"][key] = tf.ragged.stack(val)
    for key, val in inputs["fixed"].items():
        inputs["fixed"][key] = tf.constant(val)

    if labels:
        for key, val in labels["ragged"].items():
            labels["ragged"][key] = tf.ragged.stack(val)
        for key, val in labels["fixed"].items():
            labels["fixed"][key] = tf.constant(val)

        tensors = (inputs, labels)
    else:
        tensors = inputs

    ds = tf.data.Dataset.from_tensor_slices(tensors)

    return ds

from apax.utils.convert import atoms_to_inputs
class AtomisticDataset:
    """Class processes inputs/labels and makes them accessible for training."""

    def __init__(
        self,
        inputs,
        n_epoch: int,
        labels=None,
        buffer_size: int = 1000,
    ) -> None:
        """Processes inputs/labels and makes them accessible for training.

        Parameters
        ----------
        cutoff :
            Radial cutoff in angstrom for the neighbor list.
        n_epoch :
            Number of epochs
        batch_size :
            Number of strictures in one batch.
        atoms_list :
            List of all structures. Entries are ASE atoms objects.
        buffer_size : optional
            The number of structures that are shuffled for choosing the batches. Should be
            significantly larger than the batch size. It is recommended to use the default
            value.
        """
        self.n_epoch = n_epoch
        self.batch_size = None
        self.n_jit_steps = 1
        self.buffer_size = buffer_size

        max_atoms, max_nbrs = find_largest_system(inputs)
        self.max_atoms = max_atoms
        self.max_nbrs = max_nbrs

        self.n_data = len(inputs["fixed"]["n_atoms"])

        if labels:
            self.ds = dataset_from_dicts(inputs, labels)
        else:
            self.ds = dataset_from_dicts(inputs)

    def set_batch_size(self, batch_size: int):
        self.batch_size = self.validate_batch_size(batch_size)

    def batch_multiple_steps(self, n_steps: int):
        self.n_jit_steps = n_steps

    def _check_batch_size(self):
        if self.batch_size is None:
            raise ValueError("Dataset Batch Size has not been set yet")

    def validate_batch_size(self, batch_size: int) -> int:
        if batch_size > self.n_data:
            msg = (
                f"requested batch size {batch_size} is larger than the number of data"
                f" points {self.n_data}. Setting batch size = {self.n_data}"
            )
            print("Warning: " + msg)
            log.warning(msg)
            batch_size = self.n_data
        return batch_size

    def steps_per_epoch(self) -> int:
        """Returns the number of steps per epoch dependent on the number of data and the
        batch size. Steps per epoch are calculated in a way that all epochs have the same
        number of steps, and all batches have the same length. To do so, some training
        data are dropped in each epoch.
        """
        return self.n_data // self.batch_size // self.n_jit_steps

    def init_input(self) -> Dict[str, np.ndarray]:
        """Returns first batch of inputs and labels to init the model."""
        inputs = next(
            self.ds.batch(1)
            .map(PadToSpecificSize(self.max_atoms, self.max_nbrs))
            .take(1)
            .as_numpy_iterator()
        )
        if isinstance(inputs, tuple):
            inputs = inputs[0]  # remove labels

        inputs = jax.tree_map(lambda x: jnp.array(x[0]), inputs)
        init_box = np.array(inputs["box"])
        inputs = (
            inputs["positions"],
            inputs["numbers"],
            inputs["idx"],
            init_box,
            inputs["offsets"],
        )
        return inputs, init_box

    def shuffle_and_batch(self) -> Iterator[jax.Array]:
        """Shuffles, batches, and pads the inputs/labels. This function prepares the
        inputs and labels for the whole training and prefetches the data.

        Returns
        -------
        shuffled_ds :
            Iterator that returns inputs and labels of one batch in each step.
        """
        self._check_batch_size()
        #should we shuffle before or after repeat??
        ds = (
            self.ds
            .shuffle(buffer_size=self.buffer_size)
            .repeat(self.n_epoch)
            .batch(batch_size=self.batch_size)
            .map(PadToSpecificSize(self.max_atoms, self.max_nbrs))
        )

        if self.n_jit_steps > 1:
            ds = ds.batch(batch_size=self.n_jit_steps)

        ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2)
        return ds

    def batch(self) -> Iterator[jax.Array]:
        self._check_batch_size()
        ds = self.ds.batch(batch_size=self.batch_size).map(
            PadToSpecificSize(self.max_atoms, self.max_nbrs)
        )

        ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2)
        return ds




import numpy as np
from collections import deque
from random import shuffle
import tensorflow as tf
from apax.data.preprocessing import compute_nl, prefetch_to_single_device
from apax.utils.convert import atoms_to_inputs, atoms_to_labels

def pad_nl(idx, offsets, max_neighbors):
    zeros_to_add = max_neighbors - idx.shape[1]
    idx = np.pad(idx, ((0, 0), (0, zeros_to_add)), "constant").astype(np.int16)
    offsets = np.pad(offsets, ((0, zeros_to_add), (0, 0)), "constant")
    return idx, offsets


def find_largest_system2(inputs: dict[str, np.ndarray], r_max) -> tuple[int]:
    max_atoms = np.max(inputs["n_atoms"])

    max_nbrs = 0
    for position, box in zip(inputs["positions"], inputs["box"]):
        neighbor_idxs, _ = compute_nl(position, box, r_max)
        n_neighbors = neighbor_idxs.shape[1]
        max_nbrs = max(max_nbrs, n_neighbors)

    return max_atoms, max_nbrs

class Dataset:
    def __init__(self, atoms, cutoff, bs, n_jit_steps= 1, name="train", pre_shuffle=False) -> None:
        if pre_shuffle:
            shuffle(atoms)
        self.sample_atoms = atoms[0]
        inputs = atoms_to_inputs(atoms)
        finputs = {k: v for k,v in inputs["fixed"].items()}
        finputs.update({k: v for k,v in inputs["ragged"].items()})
        self.inputs = finputs

        max_atoms, max_nbrs = find_largest_system2(self.inputs, cutoff)
        self.max_atoms = max_atoms
        self.max_nbrs = max_nbrs

        labels = atoms_to_labels(atoms)
        flabels = {k: v for k,v in labels["fixed"].items()}
        flabels.update({k: v for k,v in labels["ragged"].items()})
        self.labels = flabels

        self.n_data = len(atoms)
        self.count=0
        self.cutoff = cutoff
        self.buffer = deque()
        self.batch_size = self.validate_batch_size(bs)
        self.n_jit_steps = n_jit_steps
        self.name = name

        self.buffer_size = 10

        self.enqueue(self.buffer_size)
    
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
            print("Warning: " + msg)
            log.warning(msg)
            batch_size = self.n_data
        return batch_size

    def prepare_item(self, i):
        inputs = {k:v[i] for k,v in self.inputs.items()}
        labels = {k:v[i] for k,v in self.labels.items()}
        idx, offsets = compute_nl(inputs["positions"], inputs["box"], self.cutoff)
        inputs["idx"], inputs["offsets"] = pad_nl(idx, offsets, self.max_nbrs)

        zeros_to_add = self.max_atoms - inputs["numbers"].shape[0]
        inputs["positions"] = np.pad(inputs["positions"], ((0, zeros_to_add), (0, 0)), "constant")
        inputs["numbers"] = np.pad(inputs["numbers"], (0, zeros_to_add), "constant").astype(np.int16)
        inputs["n_atoms"] = np.pad(inputs["n_atoms"], (0, zeros_to_add), "constant").astype(np.int16)
        if "forces" in labels:
            labels["forces"] = np.pad(labels["forces"], ((0, zeros_to_add), (0, 0)), "constant")

        inputs = {k:tf.constant(v) for k,v in inputs.items()}
        labels = {k:tf.constant(v) for k,v in labels.items()}
        return (inputs, labels)

    def enqueue(self, num_elements):
        for _ in range(num_elements):
            data = self.prepare_item(self.count)
            self.buffer.append(data)
            self.count += 1
        
    def __iter__(self):
        while self.count < self.n_data or len(self.buffer) > 0:
            yield self.buffer.popleft()
            space = self.buffer_size - len(self.buffer)
            if self.count + space > self.n_data:
                space = self.n_data - self.count
            self.enqueue(space)

    def make_signature(self) -> tf.TensorSpec:
        input_singature = {}
        input_singature["n_atoms"] = tf.TensorSpec((), dtype=tf.int16, name="n_atoms")
        input_singature["numbers"] = tf.TensorSpec((self.max_atoms,), dtype=tf.int16, name="numbers")
        input_singature["positions"] = tf.TensorSpec((self.max_atoms, 3), dtype=tf.float64, name="positions")
        input_singature["box"] = tf.TensorSpec((3, 3), dtype=tf.float64, name="box")
        input_singature["idx"] = tf.TensorSpec((2, self.max_nbrs), dtype=tf.int16, name="idx")
        input_singature["offsets"] = tf.TensorSpec((self.max_nbrs, 3), dtype=tf.float64, name="offsets")

        label_signature = {}
        label_signature
        if "energy" in self.labels.keys():
            label_signature["energy"] = tf.TensorSpec((), dtype=tf.float64, name="energy")
        if "forces" in self.labels.keys():
            label_signature["forces"] = tf.TensorSpec((self.max_atoms, 3), dtype=tf.float64, name="forces")
        if "stress" in self.labels.keys():
            label_signature["stress"] = tf.TensorSpec((3, 3), dtype=tf.float64, name="stress")
        signature = (input_singature, label_signature)
        return signature
    
    def init_input(self) -> Dict[str, np.ndarray]:
        """Returns first batch of inputs and labels to init the model."""
        positions = self.sample_atoms.positions
        box = self.sample_atoms.cell.array
        idx, offsets = compute_nl(positions,box, self.cutoff)
        inputs = (
            positions,
            self.sample_atoms.numbers,
            idx,
            box,
            offsets,
        )

        inputs = jax.tree_map(lambda x: jnp.array(x), inputs)
        return inputs, np.array(box)
    
    def shuffle_and_batch(self):
        gen = lambda: self
        ds = tf.data.Dataset.from_generator(gen, output_signature=self.make_signature())
        
        ds = (
            ds
            .cache(self.name)
            .repeat(10)
            .shuffle(buffer_size=100, reshuffle_each_iteration=True)
            .batch(batch_size=self.batch_size)
        )
        if self.n_jit_steps > 1:
            ds = ds.batch(batch_size=self.n_jit_steps)
        ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2)
        return ds
    
    def batch(self) -> Iterator[jax.Array]:
        gen = lambda: self
        ds = tf.data.Dataset.from_generator(gen, output_signature=self.make_signature())
        ds = (ds
            .cache(self.name)
            .repeat(10)
            .batch(batch_size=self.batch_size)
        )
        ds = prefetch_to_single_device(ds.as_numpy_iterator(), 2)
        return ds