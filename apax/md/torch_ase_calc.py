from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from matscipy.neighbours import neighbour_list


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
        self.skin = dr_threshold

        self.model = torch.jit.load(model_path)
        self.r_max = (
            self.model.energy_model.atomistic_model.descriptor.radial_fn.basis_fn.r_max
        )

        self.step = None
        self.neighbor_fn = None
        self.neighbors = None
        self.offsets = None
        self.pos0 = 0
        self.Z = [0, 0]
        self.pbc = False

    def set_neighbours_and_offsets(self, atoms, box):
        condition = (
            np.any(self.pbc != atoms.pbc)
            or len(self.Z) != len(atoms.numbers)
            or np.max(np.sum(((self.pos0 - atoms.positions) ** 2), axis=1))
            > self.skin**2 / 4.0
        )
        if condition:
            idxs_i, idxs_j, offsets = neighbour_list(
                "ijS", positions=atoms.positions, pbc=atoms.pbc, cutoff=self.r_max
            )

            self.neighbors = np.array([idxs_i, idxs_j], dtype=np.int32)
            self.offsets = np.matmul(offsets, box)  # np.zeros_like(self.neighbors) #
            self.pos0 = atoms.positions
            self.Z = atoms.numbers
            self.pbc = atoms.pbc

    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = atoms.positions
        box = atoms.cell.array
        if np.any(atoms.pbc):
            positions = atoms.positions @ np.linalg.inv(box)

        # predict
        self.set_neighbours_and_offsets(atoms, box)

        inputt = (
            torch.from_numpy(positions),
            torch.from_numpy(atoms.numbers),
            torch.from_numpy(np.asarray(self.neighbors, dtype=np.int64)),
            torch.from_numpy(np.asarray(box, dtype=np.float64)),
            torch.from_numpy(np.asarray(self.offsets, dtype=np.float64)),
        )

        results = self.model(*inputt)

        self.results = {
            k: np.array(v.detach().numpy(), dtype=np.float64) for k, v in results.items()
        }
        self.results["energy"] = self.results["energy"].item()
