import logging
from pathlib import Path

import numpy as np
from ase.io import read

log = logging.getLogger(__name__)


def load_data(data_path):
    """Non ASE compatibel parameters have to be saved in an exta file that has the same
        name as the datapath but with the extention '_labels.npz'.
        example for the npz-file:
                dipole = np.random.rand(3, 1)
                charge = np.random.rand(3, 2)
                mat = np.random.rand(3, 1)
                shape = ['ragged', 'ragged', 'fixed']

                np.savez(
                    "data_path_labels.npz",
                    dipole=dipole,
                    charge=charge,
                    mat=mat,
                    shape=shape
                )

        shape has to be in the same order than the parameters
    Parameters
    ----------
    data_path :
        Path to the ASE readable file that includes all structures.

    Returns
    -------
    atoms_list
        List of all structures where entries are ASE atoms objects.
    """
    external_labels = {}

    try:
        log.info(f"Loading data from {data_path}")
        atoms_list = read(data_path, index=":")
    except IOError:
        log.error(f"data path ({data_path}) does not exist.")

    label_path = Path(data_path)
    label_path = label_path.with_stem(label_path.stem + "_label").with_suffix(".npz")

    if label_path.is_file():
        log.info(f"Loading non ASE labels from {label_path.as_posix()}")

        label_dict = np.load(label_path.as_posix(), allow_pickle=True)

        unique_shape = np.unique(label_dict["shape"])
        for shape in unique_shape:
            external_labels.update({shape: {}})

        i = 0
        for key, val in label_dict.items():
            if key != "shape":
                external_labels[label_dict["shape"][i]].update({key: val})
                i += 1

    return atoms_list, external_labels


def split_list(data_list, external_labels, length1, length2):
    """Schuffles and splits a list in two resulting lists
    of the length length1 and length2.

    Parameters
    ----------
    data_list :
        A list.
    length1 :
        Length of the first resulting list.
    length2 :
        Length of the second resulting list.

    Returns
    -------
    splitted_list1
        List of random structures from atoms_list of the length length1.
    splitted_list2
        List of random structures from atoms_list of the length length2.
    """
    sp_label_dict1, sp_label_dict2 = ({}, {})
    if external_labels:
        idx = np.arrange(len(data_list))
        np.random.shuffle(idx)
        idx1 = idx[:length1]
        idx2 = idx[length1 : length1 + length2]

        sp_data_list1 = [data_list[i] for i in idx1]
        sp_data_list2 = [data_list[i] for i in idx2]

        for shape, labels in external_labels.items():
            sp_label_dict1.update({shape: {}})
            sp_label_dict2.update({shape: {}})
            for label, vals in labels.items():
                if len(data_list) == len(vals):
                    sp_label_dict1[shape].update({label: vals[idx1]})
                    sp_label_dict2[shape].update({label: vals[idx2]})
                else:
                    raise ValueError(
                        "number of external labels is not metching the number of data"
                        f" (strucktures) {len(data_list)} != {len(vals)}."
                    )
    else:
        np.random.shuffle(data_list)
        sp_data_list1 = data_list[:length1]
        sp_data_list2 = data_list[length1 : length1 + length2]

    return sp_data_list1, sp_data_list2, sp_label_dict1, sp_label_dict2
