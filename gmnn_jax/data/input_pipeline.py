import logging

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax_md import partition, space

from gmnn_jax.data.preprocessing import dataset_neighborlist, prefetch_to_single_device
from gmnn_jax.utils.convert import convert_atoms_to_arrays

log = logging.getLogger(__name__)


def initialize_nbr_displacement_fns(atoms, cutoff):
    default_box = 100

    box = jnp.asarray(atoms.get_cell().lengths())

    if np.all(box < 1e-6):
        displacement_fn, _ = space.free()
        box = default_box
    else:
        displacement_fn, _ = space.periodic(box)

    neighbor_fn = partition.neighbor_list(
        displacement_or_metric=displacement_fn,
        box=box,
        r_cutoff=cutoff,
        format=partition.Sparse,
    )

    return displacement_fn, neighbor_fn


class PadToSpecificSize:
    def __init__(self, max_atoms=None, max_nbrs=None) -> None:
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
    neighbor_fn,
    external_labels: dict = {},
    disable_pbar=False,
    pos_unit: str = "Ang",
    energy_unit: str = "eV",
) -> None:
    inputs, labels = convert_atoms_to_arrays(atoms_list, pos_unit, energy_unit)

    if external_labels:
        for shape, label in external_labels.items():
            labels[shape].update(label)

    idx = dataset_neighborlist(
        neighbor_fn,
        inputs["ragged"]["positions"],
        inputs["fixed"]["n_atoms"],
        disable_pbar=disable_pbar,
    )
    inputs["ragged"]["idx"] = [np.array(i) for i in idx]
    # TODO: to construct the nbr mask, I need to add n_nbrs to the inputs
    return inputs, labels


class TFPipeline:
    """Class processes inputs/labels and makes them accessible for training."""

    def __init__(
        self,
        inputs,
        labels,
        n_epoch: int,
        batch_size: int,
        max_atoms=None,
        max_nbrs=None,
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
        self.batch_size = batch_size
        self.max_atoms = max_atoms
        self.max_nbrs = max_nbrs
        self.buffer_size = buffer_size

        self.n_data = len(inputs["fixed"]["n_atoms"])

        if batch_size > self.n_data:
            raise ValueError("batch size is larger than the number of data points!")

        for key, val in inputs["ragged"].items():
            inputs["ragged"][key] = tf.ragged.constant(val)
        for key, val in inputs["fixed"].items():
            inputs["fixed"][key] = tf.constant(val)

        for key, val in labels["ragged"].items():
            labels["ragged"][key] = tf.ragged.constant(val)
        for key, val in labels["fixed"].items():
            labels["fixed"][key] = tf.constant(val)

        self.ds = tf.data.Dataset.from_tensor_slices(
            (
                inputs["ragged"],
                inputs["fixed"],
                labels["ragged"],
                labels["fixed"],
            )
        )

    def steps_per_epoch(self) -> int:
        """Returns the number of steps per epoch dependent on the number of data and the
        batch size. Steps per epoch are calculated in a way that all epochs have the same
        number of steps, and all batches have the same length. To do so some training
        data are dropped in each epoch.
        """
        return self.n_data // self.batch_size

    def init_input(self):
        """Returns first batch of inputs and labels to init the model."""
        inputs, _ = next(
            self.ds.batch(1)
            .map(PadToSpecificSize(self.max_atoms, self.max_nbrs))
            .take(1)
            .as_numpy_iterator()
        )
        return inputs

    def shuffle_and_batch(self):
        """Shuffles, batches, and pads the inputs/labels. This function prepares the
        inputs and labels for the whole training and prefetches the data.

        Returns
        -------
        shuffled_ds :
            Iterator that returns inputs and labels of one batch in each step.
        """
        shuffled_ds = (
            self.ds.shuffle(buffer_size=self.buffer_size)
            .repeat(self.n_epoch)
            .batch(batch_size=self.batch_size)
            .map(PadToSpecificSize(self.max_atoms, self.max_nbrs))
        )

        shuffled_ds = prefetch_to_single_device(shuffled_ds.as_numpy_iterator(), 2)
        return shuffled_ds
