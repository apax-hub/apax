import logging

import numpy as np

from apax.utils.data import load_data, split_atoms, split_idxs

log = logging.getLogger(__name__)


def load_data_files(data_config):
    log.info("Running Input Pipeline")
    if data_config.data_path is not None:
        log.info(f"Read data file {data_config.data_path}")
        atoms_list = load_data(data_config.data_path)

        train_idxs, val_idxs = split_idxs(
            atoms_list, data_config.n_train, data_config.n_valid
        )
        train_atoms_list, val_atoms_list = split_atoms(atoms_list, train_idxs, val_idxs)

        np.savez(
            data_config.model_version_path / "train_val_idxs",
            train_idxs=train_idxs,
            val_idxs=val_idxs,
        )

    elif data_config.train_data_path and data_config.val_data_path is not None:
        log.info(f"Read training data file {data_config.train_data_path}")
        log.info(f"Read validation data file {data_config.val_data_path}")
        train_atoms_list = load_data(data_config.train_data_path)
        val_atoms_list = load_data(data_config.val_data_path)
    else:
        raise ValueError("input data path/paths not defined")

    return train_atoms_list, val_atoms_list
