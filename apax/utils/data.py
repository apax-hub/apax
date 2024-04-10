import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from ase.io import read

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
    """Non ASE compatible parameters have to be saved in an exta file that has the same
    name as the datapath but with the extension `_labels.npz`.

    Example
    -------
    example for the npz-file::


        dipole = np.random.rand(3, 1)
        charge = np.random.rand(3, 2)
        mat = np.random.rand(3, 1)
        shape = ['ragged', 'ragged', 'fixed']

        np.savez(
            "data_path_labels.npz",
            dipole=dipole,
            charge=charge,
            mat=mat,
            shape=shape,
        )

    shape has to be in the same order than the parameters

    Parameters
    ----------
    data_path:
        Path to the ASE readable file that includes all structures.

    Returns
    -------
    list
        List of all structures where entries are ASE atoms objects.

    """
    # TODO external labels can be included via hdf5 files only? this would clean up a lot

    try:
        log.info(f"Loading data from {data_path}")
        atoms_list = read(data_path, index=":")
    except IOError:
        msg = f"data path ({data_path}) does not exist."
        log.error(msg)
        raise FileNotFoundError(msg)

    label_path = Path(data_path)
    label_path = label_path.with_stem(label_path.stem + "_label").with_suffix(".npz")

    if label_path.is_file():
        log.info(f"Loading non ASE labels from {label_path.as_posix()}")
        label_dict = np.load(label_path.as_posix(), allow_pickle=True)

        # check if len atoms == len labels
        for key, val in label_dict.items():
            for a, v in zip(atoms_list, val):
                a.calc.results[key] = v

    return atoms_list


def split_idxs(atoms_list, n_train, n_valid):
    idxs = np.arange(len(atoms_list))
    np.random.shuffle(idxs)
    train_idxs = idxs[:n_train]
    val_idxs = idxs[n_train : n_train + n_valid]

    return train_idxs, val_idxs


def split_atoms(atoms_list, train_idxs, val_idxs=None):
    """
    Split the list of atoms into training and validation sets (validation is optional).

    Parameters
    ----------
    atoms_list : list[ase.Atoms]
        List of atoms.
    train_idxs : list[int]
        List of indices for the training set.
    val_idxs : list[int], optional
        List of indices for the validation set.

    Returns
    -------
    Tuple[list, list]
        Tuple containing lists of atoms for training and validation sets.
    """
    train_atoms_list = [atoms_list[i] for i in train_idxs]

    if val_idxs is not None:
        val_atoms_list = [atoms_list[i] for i in val_idxs]
    else:
        val_atoms_list = []

    return train_atoms_list, val_atoms_list