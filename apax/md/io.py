import numpy as np
import znh5md
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


class TrajHandler:
    def step(self, state_and_energy, transform):
        pass

    def write(self, x=None, transform=None):
        pass

    def close(self):
        pass

    def reset_buffer(self):
        pass

    def atoms_from_state(self, state, energy):
        positions = np.asarray(state.position)
        momenta = np.asarray(state.momentum)
        forces = np.asarray(state.force)

        atoms = Atoms(self.atomic_numbers, positions, momenta=momenta, cell=self.box)
        atoms.calc = SinglePointCalculator(atoms, energy=float(energy), forces=forces)
        return atoms


class H5TrajHandler(TrajHandler):
    def __init__(self, R, atomic_numbers, box, sampling_rate, traj_path) -> None:
        self.atomic_numbers = atomic_numbers
        self.box = box
        self.sampling_rate = sampling_rate
        self.traj_path = traj_path
        self.db = znh5md.io.DataWriter(self.traj_path)
        self.db.initialize_database_groups()

        self.sampling_counter = 1
        self.buffer = []

    def reset_buffer(self):
        self.buffer = []

    def step(self, state_and_energy, transform):
        state, energy = state_and_energy

        if self.sampling_counter < self.sampling_rate:
            self.sampling_counter += 1
        else:
            new_atoms = self.atoms_from_state(state, energy)
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
