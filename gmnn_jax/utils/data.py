import logging

import numpy as np
from ase.io import read

log = logging.getLogger(__name__)


def load_data(data_path):
    """_summary_

    Parameters
    ----------
    data_path :
        Path to the ASE readable file that includes all structures.

    Returns
    -------
    atoms_list
        List of all structures where entries are ASE atoms objects.
    """
    # TODO get non-ASE labels from dict
    try:
        atoms_list = read(data_path, index=":")
        # TODO read non-ASE labels from file
        log.info("Get data from file")
    except IOError:
        log.error(f"data_path ({data_path}) is not leading to file")

    return atoms_list


def split_list(list, length1, length2):
    """Schuffles and splits a list in two resulting lists
    of the length length1 and length2.

    Parameters
    ----------
    list :
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
    np.random.shuffle(list)
    splitted_list1 = list[:length1]
    splitted_list2 = list[length1:length2]

    return splitted_list1, splitted_list2
