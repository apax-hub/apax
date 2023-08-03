import numpy as np
import znh5md
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from jax_md.space import transform

class TrajHandler:
    def step(self, state_and_energy, transform):
        pass

    def write(self, x=None, transform=None):
        pass

    def close(self):
        pass

    def reset_buffer(self):
        pass

    def atoms_from_state(self, state, energy, nbr_kwargs):


        if "box" in nbr_kwargs.keys():
            box = nbr_kwargs["box"]
        else:
            box = self.box

        if self.fractional:
            positions = transform(box, state.position)
        else:
            positions = state.position

        positions = np.asarray(positions)
        momenta = np.asarray(state.momentum)
        forces = np.asarray(state.force)

        atoms = Atoms(self.atomic_numbers, positions, momenta=momenta, cell=box)
        atoms.cell = atoms.cell.T
        atoms.pbc = np.diag(atoms.cell.array) > 1e-7
        atoms.calc = SinglePointCalculator(atoms, energy=float(energy), forces=forces)
        return atoms


class H5TrajHandler(TrajHandler):
    def __init__(self, system, sampling_rate, traj_path) -> None:
        self.atomic_numbers = system.atomic_numbers
        self.box = system.box
        self.fractional = np.any(self.box < 1e-6)
        self.sampling_rate = sampling_rate
        self.traj_path = traj_path
        self.db = znh5md.io.DataWriter(self.traj_path)
        self.db.initialize_database_groups()

        self.sampling_counter = 1
        self.buffer = []

    def reset_buffer(self):
        self.buffer = []

    def step(self, state, transform):
        state, energy, nbr_kwargs = state

        if self.sampling_counter < self.sampling_rate:
            self.sampling_counter += 1
        else:
            new_atoms = self.atoms_from_state(state, energy, nbr_kwargs)
            self.buffer.append(new_atoms)
            self.sampling_counter = 1

    def write(self, x=None, transform=None):
        if len(self.buffer) > 0:
            reader = znh5md.io.AtomsReader(
                self.buffer,
                step=1,
                time=self.sampling_rate,
            )
            self.db.add(reader)
            self.reset_buffer()
