from functools import partial
from pathlib import Path
from typing import Callable, Union

import ase
import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from matscipy.neighbours import neighbour_list
import torch
from tqdm import trange

from apax.data.input_pipeline import OTFInMemoryDataset
from apax.model import ModelBuilder
from apax.train.checkpoints import check_for_ensemble, restore_parameters
from apax.utils.jax_md_reduced import partition, quantity, space


class TorchASECalculator(Calculator):
    """
    ASE Calculator for apax models.
    """

    implemented_properties = [
        "energy",
        "forces",
    ]

    def __init__(
        self,
        model_path: Union[Path, list[Path]],
        dr_threshold: float = 0.5,
        transformations: Callable = [],
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.dr_threshold = dr_threshold

        self.model = model_path#torch.jit.load(model_path)
        print(self.model)
        self.r_max = 5.0

        self.step = None
        self.neighbor_fn = None
        self.neighbors = None
        self.offsets = None

    def set_neighbours_and_offsets(self, atoms, box):
        idxs_i, idxs_j, offsets = neighbour_list("ijS", positions=atoms.positions, pbc=[False, False, False], cutoff=self.r_max)

        self.neighbors = np.array([idxs_i, idxs_j], dtype=np.int32)
        self.offsets = np.zeros_like(self.neighbors) #np.matmul(offsets, box)

    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = atoms.positions
        box = atoms.cell.array

        # predict
        self.set_neighbours_and_offsets(atoms, box)
        # positions = np.array(space.transform(np.linalg.inv(box), atoms.positions))
        inputt = (
            torch.from_numpy(positions),
            torch.from_numpy(atoms.numbers),
            torch.from_numpy(np.asarray(self.neighbors, dtype=np.int64)),
            torch.from_numpy(np.asarray(box, dtype=np.float64)),
            torch.from_numpy(np.asarray(self.offsets, dtype=np.float64)),
        )

        results = self.model(*inputt)

        self.results = {k: np.array(v.detach().numpy(), dtype=np.float64) for k, v in results.items()}
        self.results["energy"] = self.results["energy"].item()




