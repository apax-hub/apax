import logging

import numpy as np

from apax.data.input_pipeline import AtomisticDataset, process_inputs
from apax.data.statistics import compute_scale_shift_parameters
from apax.utils.convert import atoms_to_labels
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


def initialize_dataset(
    config,
    atoms_list,
    read_labels: bool = True,
    calc_stats: bool = True,
):
    if calc_stats and not read_labels:
        raise ValueError(
            "Cannot calculate scale/shift parameters without reading labels."
        )
    inputs = process_inputs(
        atoms_list,
        r_max=config.model.r_max,
        disable_pbar=config.progress_bar.disable_nl_pbar,
        pos_unit=config.data.pos_unit,
    )
    labels = atoms_to_labels(
        atoms_list,
        additional_properties_info=config.data.additional_properties_info,
        read_labels=read_labels,
        pos_unit=config.data.pos_unit,
        energy_unit=config.data.energy_unit,
    )

    if calc_stats:
        ds_stats = compute_scale_shift_parameters(
            inputs,
            labels,
            config.data.shift_method,
            config.data.scale_method,
            config.data.shift_options,
            config.data.scale_options,
        )

    dataset = AtomisticDataset(
        inputs,
        config.n_epochs,
        labels=labels,
        buffer_size=config.data.shuffle_buffer_size,
    )

    if calc_stats:
        return dataset, ds_stats
    else:
        return dataset
