import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.io import read
from ase.units import Ang, Bohr, Hartree, eV, kcal, kJ, mol
from jax_md import space

log = logging.getLogger(__name__)


def make_minimal_input():
    R, Z, idx = (
        jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32),
        jnp.array([6, 8]),
        jnp.array([[1], [0]]),
    )
    box = np.array([0.0, 0.0, 0.0])
    offsets = np.array([0.0, 0.0, 0.0])
    return R, Z, idx, box, offsets


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


def prune_dict(data_dict):
    pruned = {key: val for key, val in data_dict.items() if len(val) != 0}
    return pruned


def convert_atoms_to_arrays(
    atoms_list: list[Atoms],
    pos_unit: str = "Ang",
    energy_unit: str = "eV",
) -> tuple[dict[str, dict[str, list]], dict[str, dict[str, list]]]:
    """Converts an list of ASE atoms to two dicts where all inputs and labels
    are sorted by there shape (ragged/fixed), and proberty. Units are
    adjusted if ASE compatible and provided in the inputpipeline.


    Parameters
    ----------
    atoms_list :
        List of all structures. Enties are ASE atoms objects.

    Returns
    -------
    inputs :
        Inputs are untrainable system-determining properties.
    labels :
        Labels are trainable system properties.
    """
    inputs = {
        "ragged": {
            "positions": [],
            "numbers": [],
        },
        "fixed": {
            "n_atoms": [],
            "box": [],
        },
    }

    labels = {
        "ragged": {
            "forces": [],
        },
        "fixed": {
            "energy": [],
            "stress": [],
        },
    }
    DTYPE = np.float64

    unit_dict = {
        "Ang": Ang,
        "Bohr": Bohr,
        "eV": eV,
        "kcal/mol": kcal / mol,
        "Hartree": Hartree,
        "kJ/mol": kJ / mol,
    }
    box = np.array(atoms_list[0].cell.lengths())
    pbc = np.all(box > 1e-6)

    for atoms in atoms_list:
        box = np.diagonal(atoms.cell * unit_dict[pos_unit]).astype(DTYPE)
        inputs["fixed"]["box"].append(box)

        if pbc != np.all(box > 1e-6):
            raise ValueError(
                "Apax does not support dataset periodic and non periodic structures"
            )

        if np.all(box < 1e-6):
            inputs["ragged"]["positions"].append(
                (atoms.positions * unit_dict[pos_unit]).astype(DTYPE)
            )
        else:
            inv_box = np.divide(1, box, where=box != 0)
            inputs["ragged"]["positions"].append(
                np.array(
                    space.transform(
                        inv_box, (atoms.positions * unit_dict[pos_unit]).astype(DTYPE)
                    )
                )
            )

        inputs["ragged"]["numbers"].append(atoms.numbers)
        inputs["fixed"]["n_atoms"].append(len(atoms))

        for key, val in atoms.calc.results.items():
            if key == "forces":
                labels["ragged"][key].append(
                    val * unit_dict[energy_unit] / unit_dict[pos_unit]
                )
            elif key == "energy":
                labels["fixed"][key].append(val * unit_dict[energy_unit])
            elif key == "stress":
                stress = (
                    atoms.get_stress(voigt=False)
                    * unit_dict[energy_unit]
                    / (unit_dict[pos_unit] ** 3)
                    * atoms.cell.volume
                )
                labels["fixed"][key].append(stress)

    inputs["fixed"] = prune_dict(inputs["fixed"])
    labels["fixed"] = prune_dict(labels["fixed"])
    inputs["ragged"] = prune_dict(inputs["ragged"])
    labels["ragged"] = prune_dict(labels["ragged"])
    return inputs, labels


def split_idxs(atoms_list, n_train, n_valid):
    idxs = np.arange(len(atoms_list))
    np.random.shuffle(idxs)
    train_idxs = idxs[:n_train]
    val_idxs = idxs[n_train : n_train + n_valid]

    return train_idxs, val_idxs


def split_atoms(atoms_list, train_idxs, val_idxs=None):
    """Shuffles and splits a list in two resulting lists
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
    train_atoms_list = [atoms_list[i] for i in train_idxs]

    if val_idxs is not None:
        val_atoms_list = [atoms_list[i] for i in val_idxs]
    else:
        val_atoms_list = []

    return train_atoms_list, val_atoms_list


def split_label(external_labels, train_idxs, val_idxs=None):
    train_label_dict, val_label_dict = ({}, {})

    if val_idxs is not None:
        for shape, labels in external_labels.items():
            train_label_dict.update({shape: {}})
            val_label_dict.update({shape: {}})
            for label, vals in labels.items():
                train_label_dict[shape].update({label: vals[train_idxs]})
                val_label_dict[shape].update({label: vals[val_idxs]})
    else:
        for shape, labels in external_labels.items():
            train_label_dict.update({shape: {}})
            for label, vals in labels.items():
                train_label_dict[shape].update({label: vals[train_idxs]})

    return train_label_dict, val_label_dict
