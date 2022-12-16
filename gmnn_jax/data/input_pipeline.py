import logging

import numpy as np
import tensorflow as tf
from jax_md import partition, space

from gmnn_jax.data.preprocessing import dataset_neighborlist, prefetch_to_single_device
from gmnn_jax.utils.convert import convert_atoms_to_arrays

log = logging.getLogger(__name__)


def pad_to_largest_element(
    r_inputs: dict, f_inputs: dict, r_labels: dict, f_labels: dict
) -> tuple[dict, dict]:
    """Function is padding all input and label dicts that values are of type ragged
        to largest element in the batch. Afterward, the distinction between ragged
        and fixed inputs/labels is not needed and all inputs/labels are updated to
        one list.
    Parameters
    ----------
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
        r_inputs[key] = val.to_tensor()

    for key, val in r_labels.items():
        r_labels[key] = val.to_tensor()

    inputs = r_inputs.copy()
    inputs.update(f_inputs)

    labels = r_labels.copy()
    labels.update(f_labels)

    return inputs, labels


class InputPipeline:
    """Class processes inputs/labels and makes them accessible for training."""

    def __init__(
        self,
        cutoff: float,
        n_epoch: int,
        batch_size: int,
        atoms_list: list,
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
        self.buffer_size = buffer_size

        inputs, labels = convert_atoms_to_arrays(atoms_list)

        self.n_data = len(inputs["fixed"]["n_atoms"])

        cubic_box_size = 100

        nl_format = partition.Sparse

        if "cell" in inputs["fixed"]:
            cubic_box_size = inputs["fixed"]["cell"][0][0]
            displacement_fn, _ = space.periodic(cubic_box_size)
        else:
            displacement_fn, _ = space.free()
        self.displacement_fn = displacement_fn

        neighbor_fn = partition.neighbor_list(
            displacement_or_metric=self.displacement_fn,
            box=cubic_box_size,
            r_cutoff=cutoff,
            format=nl_format,
        )

        idx = dataset_neighborlist(
            neighbor_fn,
            inputs["ragged"]["positions"],
            inputs["fixed"]["n_atoms"],
        )

        inputs["ragged"]["idx"] = []
        for i in idx:
            inputs["ragged"]["idx"].append(np.array(i))

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
        input = next(
            self.ds.batch(1).map(pad_to_largest_element).take(1).as_numpy_iterator()
        )
        return input

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
            .map(pad_to_largest_element)
        )

        shuffled_ds = prefetch_to_single_device(shuffled_ds.as_numpy_iterator(), 2)
        return shuffled_ds
