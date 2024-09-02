import logging
from pathlib import Path

import h5py
import numpy as np
import znh5md
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from apax.md.sim_utils import System
from apax.utils.helpers import APAX_PROPERTIES
from apax.utils.jax_md_reduced import space

log = logging.getLogger(__name__)


class TrajHandler:
    def __init__(
        self,
        system: System,
        sampling_rate: int,
        buffer_size: int,
        traj_path: Path,
        time_step: float = 0.5,
        properties: list[str] = APAX_PROPERTIES,
    ) -> None:
        self.atomic_numbers = system.atomic_numbers
        self.box = system.box
        self.fractional = np.any(self.box > 1e-6)
        self.sampling_rate = sampling_rate
        self.traj_path = traj_path
        self.time_step = time_step
        self.properties = properties

    def step(self, state_and_energy, transform=None):
        pass

    def write(self, x=None, transform=None):
        pass

    def close(self):
        pass

    def reset_buffer(self):
        pass

    def atoms_from_state(self, state, predictions, nbr_kwargs):
        if "box" in nbr_kwargs.keys():
            box = nbr_kwargs["box"]
        else:
            box = self.box

        if self.fractional:
            positions = space.transform(box, state.position)
        else:
            positions = state.position

        positions = np.asarray(positions)
        momenta = np.asarray(state.momentum)
        forces = np.asarray(state.force)

        atoms = Atoms(self.atomic_numbers, positions, momenta=momenta, cell=box)
        atoms.cell = atoms.cell.T
        atoms.pbc = np.diag(atoms.cell.array) > 1e-6
        predictions = {k: np.array(v) for k, v in predictions.items()}
        predictions["energy"] = predictions["energy"].item()
        predictions = {k: v for k, v in predictions.items() if k in self.properties}
        atoms.calc = SinglePointCalculator(atoms, **predictions)
        return atoms


class H5TrajHandler(TrajHandler):
    def __init__(
        self,
        system: System,
        sampling_rate: int,
        buffer_size: int,
        traj_path: Path,
        time_step: float = 0.5,
        properties: list[str] = [],
    ) -> None:
        self.atomic_numbers = system.atomic_numbers
        self.box = system.box
        self.fractional = np.any(self.box > 1e-6)
        self.sampling_rate = sampling_rate
        self.traj_path = traj_path
        self.time_step = time_step
        self.properties = properties
        self.db = znh5md.IO(
            self.traj_path, timestep=self.time_step, store="time", save_units=False
        )

        self.step_counter = 0
        self.buffer = []
        self.buffer_size = buffer_size

    def reset_buffer(self):
        self.buffer = []

    def step(self, state, transform=None):
        state, predictions, nbr_kwargs = state

        if self.step_counter % self.sampling_rate == 0:
            new_atoms = self.atoms_from_state(state, predictions, nbr_kwargs)
            self.buffer.append(new_atoms)
        self.step_counter += 1

        if len(self.buffer) >= self.buffer_size:
            self.write()

    def write(self, x=None, transform=None):
        if len(self.buffer) > 0:
            self.db.extend(self.buffer)
            self.reset_buffer()


class DSTruncator:
    def __init__(self, length):
        self.length = length
        self.node_names = []

    def __call__(self, name, node):
        if isinstance(node, h5py.Dataset):
            if len(node.shape) > 1 or name.endswith("energy/value"):
                self.node_names.append(name)

    def truncate(self, ds):
        for name in self.node_names:
            shape = tuple([None] + list(ds[name].shape[1:]))
            truncated_data = ds[name][: self.length]
            del ds[name]
            ds.create_dataset(name, maxshape=shape, data=truncated_data, chunks=True)


def truncate_trajectory_to_checkpoint(traj_path, length):
    truncator = DSTruncator(length=length)
    with h5py.File(traj_path, "r+") as ds:
        ds.visititems(truncator)
        truncator.truncate(ds)
