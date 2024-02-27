import dataclasses
import logging
from typing import Optional

import numpy as np
from ase import Atoms

from apax.data.input_pipeline import AtomisticDataset, process_inputs, process_labels
from apax.data.statistics import compute_scale_shift_parameters
from apax.utils.data import load_data, split_atoms, split_idxs, split_label

log = logging.getLogger(__name__)


@dataclasses.dataclass
class RawDataset:
    atoms_list: list[Atoms]
    additional_labels: Optional[dict] = None


def load_data_files(data_config):
    log.info("Running Input Pipeline")
    if data_config.data_path is not None:
        log.info(f"Read data file {data_config.data_path}")
        atoms_list, label_dict = load_data(data_config.data_path)

        train_idxs, val_idxs = split_idxs(
            atoms_list, data_config.n_train, data_config.n_valid
        )
        train_atoms_list, val_atoms_list = split_atoms(atoms_list, train_idxs, val_idxs)
        train_label_dict, val_label_dict = split_label(label_dict, train_idxs, val_idxs)

        np.savez(
            data_config.model_version_path / "train_val_idxs",
            train_idxs=train_idxs,
            val_idxs=val_idxs,
        )

    elif data_config.train_data_path and data_config.val_data_path is not None:
        log.info(f"Read training data file {data_config.train_data_path}")
        log.info(f"Read validation data file {data_config.val_data_path}")
        train_atoms_list, train_label_dict = load_data(data_config.train_data_path)
        val_atoms_list, val_label_dict = load_data(data_config.val_data_path)
    else:
        raise ValueError("input data path/paths not defined")

    train_raw_ds = RawDataset(
        atoms_list=train_atoms_list, additional_labels=train_label_dict
    )
    val_raw_ds = RawDataset(atoms_list=val_atoms_list, additional_labels=val_label_dict)

    return train_raw_ds, val_raw_ds


def initialize_dataset(config, raw_ds, calc_stats: bool = True, read_labels: bool =True):
    inputs = process_inputs(
        raw_ds.atoms_list,
        r_max=config.model.r_max,
        disable_pbar=config.progress_bar.disable_nl_pbar,
        pos_unit=config.data.pos_unit,
    )
    if read_labels:
        labels = process_labels(
            raw_ds.atoms_list,
            external_labels=raw_ds.additional_labels,
            pos_unit=config.data.pos_unit,
            energy_unit=config.data.energy_unit,
        )
    else:
        labels = None


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
