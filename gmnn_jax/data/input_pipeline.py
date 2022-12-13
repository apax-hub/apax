import logging
from typing import Type

import numpy as np
import tensorflow as tf
from jax_md import partition, space

from gmnn_jax.data.preprocessing import dataset_neighborlist
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
    """Processes all inputs and labels and prepares them for the training cycle.
    Parameters
    ----------
    cutoff :
        Radial cutoff in angstrom for the neighbor list.
    batch_size :
        Number of strictures in one batch.
    atoms_list :
        List of all structures. Entries are ASE atoms objects.
    buffer_size : optional
        The number of structures that are shuffled for choosing the batches. Should be
        significantly larger than the batch size. It is recommended to use the default
        value.
    """

    def __init__(
        self,
        cutoff: float,
        atoms_list: list,
        batch_size: int,
        buffer_size: int = 1000,
    ) -> Type[tf.data.Dataset]:
        """_summary_

        Parameters
        ----------
        cutoff : _type_
            _description_
        atoms_list : _type_
            _description_
        batch_size : _type_
            _description_
        buffer_size : _type_, optional
            _description_, by default 1000
        """
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        inputs, labels = convert_atoms_to_arrays(atoms_list)
        cubic_box_size = 100

        nl_format = partition.Sparse
        if "cell" in inputs["fixed"]:
            cubic_box_size = inputs["fixed"]["cell"][0][0]
            self.displacement_fn, _ = space.periodic(cubic_box_size)
        else:
            self.displacement_fn, _ = space.free()

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

    def get_displacement_fn(self):
        return self.displacement_fn

    def __call__(self):
        """Inputs and Labels are shuffled, padded (to the largest element in the batch),
        and returned.

        Returns
        -------
        ds :
            A dataset that includes all data prepared for training e.g. split into
            batches and padded. The dataset contains tf.Tensors.
        """
        shuffled_ds = (
            self.ds.shuffle(buffer_size=self.buffer_size)
            .batch(batch_size=self.batch_size)
            .map(pad_to_largest_element)
        )
        return shuffled_ds
