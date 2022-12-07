import logging
import os
from typing import Type

import numpy as np

import tensorflow as tf
from ase.io import read
from jax_md import partition, space

from gmnn_jax.data.preprocessing import dataset_neighborlist
from gmnn_jax.utils.convert import convert_atoms_to_arrays

log = logging.getLogger(__name__)


def pad_to_largest_element(
    r_inputs: dict, f_inputs: dict, r_labels: dict, f_labels: dict
) -> tuple[dict, dict]:
    """Function is padding all input and label dicts that values are of type ragged
        to largest element in the batch. Afterward, the distinction between ragged and fixed
        inputs/labels is not needed and all inputs/labels are updated to one list.

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


def input_pipeline(
    cutoff: float,
    batch_size: int,
    data_path: str = None,
    atoms_list: str = None,
    buffer_size: int = 1000,
) -> Type[tf.data.Dataset]:
    """Processes all inputs and labels and prepares them for the training cycle.
    Inputs and Labels are padded to the largest element in the batch.

    Parameters
    ----------
    cutoff :
        Radial cutoff in angstrom for the neighbor list
    batch_size :
        Number of strictures in one batch
    data_path :
        Path to the ASE readable file that includes all structures. By default None.
        IMPORTANT: Eighter data_path ore atoms_list have to be defined if both are
        defined atoms_list is primarily.
    atoms_list :
        List of all structures. Entries are ASE atoms objects. By default None.
        IMPORTANT: Eighter data_path ore atoms_list have to be defined
        if both are defined atoms_list is primarily.
    buffer_size : optional
        The number of structures that are shuffled for choosing the batches. Should be
        significantly larger than the batch size. It is recommended to use the default
        value.

    Returns
    -------
    ds :
        A dataset that includes all data prepared for training e.g. split into
        batches and padded. The dataset contains tf.Tensors.

    Raises
    ------
    ValueError
        Raises if no inputs and labels are defined.
    """

    if data_path is not None and atoms_list is None:
        try:
            atoms_list = read(data_path, index=":")
            # TODO read non-ASE labels from file
            log.info("Get data from file")
        except IOError:
            log.error(f"data_path ({data_path}) is not leading to file")

    elif type(atoms_list) == list:
        log.info("Get atoms directly as ASE atoms objects")
        # TODO get non-ASE labels from dict
    else:
        raise ValueError(
            "Input data and labels are missing eigther define a data_path or an"
            " atoms_list (ASE atoms objects)"
        )

    inputs, labels = convert_atoms_to_arrays(atoms_list)
    cubic_box_size = 100

    nl_format = partition.Sparse
    if "cell" in inputs["fixed"]:
        cubic_box_size = inputs["fixed"]["cell"][0][0]
        displacement_fn, _ = space.periodic(cubic_box_size)
    else:
        displacement_fn, _ = space.free()

    neighbor_fn = partition.neighbor_list(
        displacement_or_metric=displacement_fn,
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

    ds = tf.data.Dataset.from_tensor_slices(
        (
            inputs["ragged"],
            inputs["fixed"],
            labels["ragged"],
            labels["fixed"],
        )
    )

    ds = (
        ds.shuffle(buffer_size=buffer_size)
        .batch(batch_size=batch_size)
        .map(pad_to_largest_element)
    )

    return ds
