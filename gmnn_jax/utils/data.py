import csv
import logging
from os.path import splitext

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
    atoms_list = None
    label_dict = None
    try:
        log.info("Loading data from file")
        atoms_list = read(data_path, index=":")

        system_name = splitext("data_path")[0]
        label_path = f"{system_name}_labels.csv"

        if label_path.is_file():
            log.info("Loading non ASE labels from file")
            label_dict = {
                "ragged": {},
                "fixed": {},
            }
            with open(label_path, "r") as file:
                reader = csv.DictReader(file)
                for label in reader:
                    label_dict[label["shape"]].update(
                        {label["name"]: np.array(label["value"])}
                    )

    except IOError:
        log.error(f"data_path ({data_path}) is not leading to file")

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
