import logging
from typing import Dict, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from apax.data.preprocessing import dataset_neighborlist, prefetch_to_single_device
from apax.utils.convert import atoms_to_arrays

log = logging.getLogger(__name__)


def find_largest_system(inputs: dict[str, np.ndarray]) -> tuple[int]:
    max_atoms = np.max(inputs["fixed"]["n_atoms"])
    nbr_shapes = [idx.shape[1] for idx in inputs["ragged"]["idx"]]
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

    def __call__(
        self, r_inputs: dict, f_inputs: dict, r_labels: dict, f_labels: dict
    ) -> tuple[dict, dict]:
        """
        Arguments
        ---------

        r_inputs :
            Inputs of ragged shape. Untrainable system-determining properties.
        f_inputs :
            Inputs of fixed shape. Untrainable system-determining properties.
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
        for key, val in r_inputs.items():
            if self.max_atoms is None:
                r_inputs[key] = val.to_tensor()
            elif key == "idx":
                shape = r_inputs[key].shape
                padded_shape = [shape[0], shape[1], self.max_nbrs]  # batch, ij, nbrs
            elif key == "offsets":
                shape = r_inputs[key].shape
                padded_shape = [shape[0], self.max_nbrs, 3]  # batch, ij, nbrs
            elif key == "numbers":
                shape = r_inputs[key].shape
                padded_shape = [shape[0], self.max_atoms]  # batch, atoms
            else:
                shape = r_inputs[key].shape
                padded_shape = [shape[0], self.max_atoms, shape[2]]  # batch, atoms, 3
            r_inputs[key] = val.to_tensor(shape=padded_shape)

        for key, val in r_labels.items():
            if self.max_atoms is None:
                r_labels[key] = val.to_tensor()
            else:
                padded_shape = [shape[0], self.max_atoms, shape[2]]
                r_labels[key] = val.to_tensor(default_value=0.0, shape=padded_shape)

        inputs = r_inputs.copy()
        inputs.update(f_inputs)

        labels = r_labels.copy()
        labels.update(f_labels)

        return inputs, labels


def create_dict_dataset(
    atoms_list: list,
    r_max: float,
    external_labels: dict = {},
    disable_pbar=False,
    pos_unit: str = "Ang",
    energy_unit: str = "eV",
) -> tuple[dict]:
    inputs, labels = atoms_to_arrays(atoms_list, pos_unit, energy_unit)

    if external_labels:
        for shape, label in external_labels.items():
            labels[shape].update(label)

    idx, offsets = dataset_neighborlist(
        inputs["ragged"]["positions"],
        box=inputs["fixed"]["box"],
        r_max=r_max,
        atoms_list=atoms_list,
        disable_pbar=disable_pbar,
    )

    inputs["ragged"]["idx"] = idx
    inputs["ragged"]["offsets"] = offsets
    return inputs, labels


def dataset_from_dicts(
    inputs: Dict[str, np.ndarray], labels: Dict[str, np.ndarray]
) -> tf.data.Dataset:
    # tf.RaggedTensors should be created from `tf.ragged.stack`
    # instead of `tf.ragged.constant` for performance reasons.
    # See https://github.com/tensorflow/tensorflow/issues/47853
    for key, val in inputs["ragged"].items():
        inputs["ragged"][key] = tf.ragged.stack(val)
    for key, val in inputs["fixed"].items():
        inputs["fixed"][key] = tf.constant(val)

    for key, val in labels["ragged"].items():
        labels["ragged"][key] = tf.ragged.stack(val)
    for key, val in labels["fixed"].items():
        labels["fixed"][key] = tf.constant(val)

    ds = tf.data.Dataset.from_tensor_slices((
        inputs["ragged"],
        inputs["fixed"],
        labels["ragged"],
        labels["fixed"],
    ))
    return ds


class AtomisticDataset:
    """Class processes inputs/labels and makes them accessible for training."""

    def __init__(
        self,
        inputs,
        labels,
        n_epoch: int,
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

        self.ds = dataset_from_dicts(inputs, labels)

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
        inputs, _ = next(
            self.ds.batch(1)
            .map(PadToSpecificSize(self.max_atoms, self.max_nbrs))
            .take(1)
            .as_numpy_iterator()
        )

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
        ds = (
            self.ds.shuffle(buffer_size=self.buffer_size)
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
