import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import znh5md
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
    """

    Parameters
    ----------
    data_path:
        Path to the ASE readable file that includes all structures.

    Returns
    -------
    list
        List of all structures where entries are ASE atoms objects.

    """
    data_path = Path(data_path)

    if not data_path.is_file():
        msg = f"data path ({data_path}) does not exist."
        log.error(msg)
        raise FileNotFoundError(msg)

    if data_path.suffix in [".h5", ".h5md", ".hdf5"]:
        atoms_list = znh5md.IO(data_path)[:]
    else:
        atoms_list = read(data_path.as_posix(), index=":")

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
