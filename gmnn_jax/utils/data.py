import logging
from os.path import splitext

import numpy as np
from ase.io import read

log = logging.getLogger(__name__)


def load_data(data_path):
    """Non ASE compatibel parameters have to be saved in an exta file that has the same
        name as the datapath but with the extention '_labels.npz'.
        example for the npz-file:
                dipole = np.random.rand(3, 1)
                charge = np.random.rand(3, 2)
                mat = np.random.rand(4)
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
    label_dict = {}

    try:
        log.info(f"Loading data from {data_path}")
        atoms_list = read(data_path, index=":")
    except IOError:
        log.error(f"data_path ({data_path}) is not leading to file")

    system_name = splitext("data_path")[0]
    label_path = f"{system_name}_labels.npz"

    if label_path.is_file():
        log.info(f"Loading non ASE labels from {label_path}")

        dict = np.load(label_path, allow_pickle=True)

        unique_shape = np.unique(dict["shape"])
        for shape in unique_shape:
            label_dict.update({shape: {}})

        i = 0
        for key, val in dict.items():
            if key != "shape":
                label_dict[dict["shape"][i]].update({key: val})
                i += 1

    return atoms_list, label_dict


def split_list(data_list, label_dict, length1, length2):
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

    # TODO shuffle label_dict values the same way
    if label_dict is not None:
        raise NotImplementedError("External labels can not be shuffled at the moment")
    else:
        np.random.shuffle(data_list)
        splitted_list1 = data_list[:length1]
        splitted_list2 = data_list[length1 : length1 + length2]
        return splitted_list1, splitted_list2
