from collections import deque
from pathlib import Path
from random import shuffle
from typing import SupportsIndex, Any as T
import uuid

from apax.data.input_pipeline import find_largest_system, pad_nl
from apax.data.preprocessing import compute_nl
from apax.utils.convert import atoms_to_inputs, atoms_to_labels
import numpy as np


class RandomAccessDataSource():
    """Interface for datasources where storage supports efficient random access."""

    def __init__(
        self,
        atoms_list,
        cutoff,
        bs,
        n_epochs,
        pos_unit: str = "Ang",
        energy_unit: str = "eV",
        additional_properties: list[tuple] = [],
        pre_shuffle=False,
        shuffle_buffer_size=1000,
        ignore_labels=False,
    ) -> None:
        self.n_epochs = n_epochs
        self.cutoff = cutoff
        self.buffer_size = shuffle_buffer_size
        self.n_data = len(atoms_list)
        self.batch_size = self.validate_batch_size(bs)
        self.pos_unit = pos_unit
        self.additional_properties = additional_properties

        if pre_shuffle:
            shuffle(atoms_list)
        self.sample_atoms = atoms_list[0]
        self.inputs = atoms_to_inputs(atoms_list, pos_unit)

        max_atoms, max_nbrs = find_largest_system(self.inputs, self.cutoff)
        self.max_atoms = max_atoms
        self.max_nbrs = max_nbrs
        if atoms_list[0].calc and not ignore_labels:
            self.labels = atoms_to_labels(
                atoms_list, pos_unit, energy_unit, additional_properties
            )
        else:
            self.labels = None

    def __len__(self) -> int:
        """Number of records in the dataset."""
        return self.n_data

    def __getitem__(self, record_key: SupportsIndex) -> T:
        """Retrieves record for the given record_key."""

        inputs = {k: v[record_key] for k, v in self.inputs.items()}
        idx, offsets = compute_nl(inputs["positions"], inputs["box"], self.cutoff)
        inputs["idx"], inputs["offsets"] = idx, offsets

        if not self.labels:
            return inputs

        labels = {k: v[record_key] for k, v in self.labels.items()}

        return (inputs, labels)
  


