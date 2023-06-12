from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import TrajectoryWriter
import znh5md
import numpy as np



class TrajHandler:
    def step(self, state_and_energy, transform):
        pass

    def write(self):
        pass

    def close(self):
        pass

    def reset_buffer(self):
        pass

    def atoms_from_state(self, state, energy):
        positions = np.asarray(state.position)
        momenta = np.asarray(state.momentum)
        forces = np.asarray(state.force)

        check1 = np.any(positions == None)
        check2 = np.any(momenta == None)
        check3 = np.any(forces == None)
        if check1 or check2 or check3:
            print(check1, check2, check3)
            quit()
        # print(forces[0])
        atoms = Atoms(
            self.atomic_numbers, positions, cell=self.box # , momenta=momenta
        )
        atoms.calc = SinglePointCalculator(
            atoms, energy=float(energy), forces=forces
        )
        return atoms


class ASETrajHandler(TrajHandler):
    def __init__(self, R, atomic_numbers, box, traj_path) -> None:
        self.atomic_numbers = atomic_numbers
        self.box = box
        self.traj = TrajectoryWriter(traj_path, mode="w")
        new_atoms = Atoms(atomic_numbers, R, cell=box)
        self.traj.write(new_atoms)

        self.sampling_rate=10
        self.sampling_counter = 0

    def step(self, state_and_energy, transform):
        state, energy = state_and_energy
        
        if self.sampling_counter < self.sampling_rate:
            self.sampling_counter += 1
        else:
            new_atoms = self.atoms_from_state(state, energy)
            print(new_atoms)
            # self.traj.write(new_atoms)
            self.sampling_counter = 0

    def write(self):
        pass

    def close(self):
        self.traj.close()


class H5TrajHandler(TrajHandler):
    def __init__(self, R, atomic_numbers, box, traj_path) -> None:
        self.atomic_numbers = atomic_numbers
        self.box = box
        self.traj_path = traj_path
        self.db = znh5md.io.DataWriter(self.traj_path)
        self.db.initialize_database_groups()

        self.sampling_rate=5
        self.sampling_counter = 0
        self.buffer_size = 100

        new_atoms = Atoms(atomic_numbers, R, cell=box)
        new_atoms.calc = SinglePointCalculator(
            new_atoms, energy=0.0, forces=np.zeros_like(new_atoms.positions)
        )
        self.buffer = [] #[new_atoms]

    def reset_buffer(self):
        self.buffer = []

    def step(self, state_and_energy, transform):
        state, energy = state_and_energy
        
        if self.sampling_counter < self.sampling_rate:
            self.sampling_counter += 1
        else:
            new_atoms = self.atoms_from_state(state, energy)
            self.buffer.append(new_atoms)
            self.sampling_counter = 0


    def write(self, x=None, transform=None):
        if len(self.buffer) > 0:
            reader = znh5md.io.AtomsReader(
                self.buffer,
                frames_per_chunk=self.buffer_size,
                step=1,
                time=self.sampling_rate,
            )
            self.db.add(reader)
            self.buffer=[]
