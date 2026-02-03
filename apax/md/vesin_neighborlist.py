from typing import Tuple

import numpy as np
from ase import Atoms
from vesin import NeighborList


class VesinNeighborListWrapper:
    def __init__(self, cutoff: float, skin: float, padding_factor: float = 1.2):
        self.cutoff = cutoff + skin
        self.skin = skin
        self.padding_factor = padding_factor
        self._last_positions = None
        self._last_cell = None
        self._nl_data = None
        self._padded_length = 0
        self._vesin_nl = None

    def update(self, atoms: Atoms) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = atoms.positions
        cell = atoms.cell.array
        periodic = bool(np.any(atoms.pbc))

        recompute = False
        if self._last_positions is None or self._last_cell is None:
            recompute = True
        else:
            # Check for position change
            max_sq_disp = ((self._last_positions - positions) ** 2).sum(axis=1).max()
            if max_sq_disp > self.skin**2 / 4.0:
                recompute = True
            
            # Check for cell change
            if not np.allclose(cell, self._last_cell):
                recompute = True

        if recompute:
            self._vesin_nl = NeighborList(cutoff=self.cutoff, full_list=True)
            idxs_i, idxs_j, offsets = self._vesin_nl.compute(
                points=positions,
                box=cell,
                periodic=periodic,
                quantities="ijS",
            )

            current_len = len(idxs_i)
            if current_len > self._padded_length:
                print("Vesin neighbor list overflowed, extending padding.")
                self._padded_length = int(current_len * self.padding_factor)
                if self._padded_length == 0:
                    self._padded_length = int(1 * self.padding_factor)

            zeros_to_add = self._padded_length - current_len

            padded_idxs_i = np.pad(idxs_i, (0, zeros_to_add), "constant").astype(np.int32)
            padded_idxs_j = np.pad(idxs_j, (0, zeros_to_add), "constant").astype(np.int32)
            padded_offsets = np.pad(offsets, ((0, zeros_to_add), (0, 0)), "constant")
            
            padded_offsets = np.matmul(padded_offsets, cell)


            self._nl_data = (padded_idxs_i, padded_idxs_j, padded_offsets)
            self._last_positions = positions.copy()
            self._last_cell = cell.copy()

        return self._nl_data