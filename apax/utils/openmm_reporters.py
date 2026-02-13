import logging
from pathlib import Path
from typing import Any

from ase.units import fs as ase_fs

try:
    from openmm.app import Simulation
    from openmm.openmm import State
    from openmm.unit import angstrom, ev, femtosecond, item

    _openmm_imported = True
except ImportError:
    _openmm_imported = False

_float_width = 18

log = logging.getLogger(__name__)


class XYZReporter:
    """Convenient way to write an xyz file during an OpenMM simulation

    See https://docs.openmm.org/latest/userguide/application/04_advanced_sim_examples.html#extracting-and-reporting-forces-and-other-data
    """

    def __init__(
        self,
        file: str | Path,
        reportInterval: int,
        elements: list[str],
        enforcePeriodicBox: bool | None = None,
        atomSubset: list[int] | None = None,
        includeVelocities: bool = False,
        includeForces: bool = False,
        flushEvery: int = 1,
        append: bool = False,
    ):
        """
        Parameters
        ----------
        file (str | Path): file to write to
        reportInterval (int): Write a report every reportInterval steps
        elements (list[str]): Elemental symbols of the simulated system.
        enforcePeriodicBox (bool | None): Whether to wrap atomic coordinates
            to within PBC. If None, decide automatically. See
            https://docs.openmm.org/7.3.0/api-python/generated/simtk.openmm.app.pdbreporter.PDBxReporter.html
        atomSubset (list[int] | None): if a list of integers, only write the
            atoms indeces in the list. Default = None.
        includeVelocities (bool): Whether to also write the velocities in
            ase units. Default = False
        includeForces (bool): Whether to also write the forces in
            eV / Angstrom. Default = False
        flushEvery (int): flush every flushEvery reports. Default = 1
        append (bool): if True, append to the end of the file instead of
            overwriting it. Default = False
        """
        if not _openmm_imported:
            raise ImportError(f"XYZReporter requires OpenMM to be installed")

        if append:
            self._out = open(file, "a")
        else:
            self._out = open(file, "w")

        self._reportInterval = reportInterval
        self._elements = elements
        self._enforcePeriodicBox = enforcePeriodicBox
        if atomSubset is None:
            atomSubset = list(range(len(elements)))
        self._atomSubset = atomSubset
        self._n_atoms_string = "".join([str(len(self._atomSubset)), "\n"])

        self._flush_every = flushEvery
        self._flush_counter = 1

        self._includeForces = includeForces
        self._includeVelocities = includeVelocities

        self._include_list = ["positions"]
        if self._includeForces:
            self._include_list.append("forces")
        if self._includeVelocities:
            self._include_list.append("velocities")

        # Write header here
        self._properties_string = "Properties=species:S:1:pos:R:3"
        if self._includeVelocities:
            self._properties_string = ":".join([self._properties_string, "vel:R:3"])
        if self._includeForces:
            self._properties_string = ":".join([self._properties_string, "forces:R:3"])
        self._properties_string = "".join([self._properties_string, " Time="])

        self._is_pbc = None

    def __del__(self):
        if hasattr(self._out, "flush") and callable(self._out.flush):
            self._out.flush()
        self._out.close()

    def describeNextReport(self, simulation: Simulation) -> dict[str, Any]:
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return {
            "steps": steps,
            "periodic": self._enforcePeriodicBox,
            "include": self._include_list,
        }

    def report(self, simulation: Simulation, state: State) -> None:
        time = state.getTime().value_in_unit(femtosecond)
        positions = state.getPositions().value_in_unit(angstrom)

        if self._includeForces:
            forces = state.getForces().value_in_unit(ev / (angstrom * item))
        if self._includeVelocities:
            velocities = (
                state.getVelocities(asNumpy=True).value_in_unit(angstrom / femtosecond)
                * ase_fs
            )

        if self._is_pbc is None:
            self._is_pbc = simulation.system.usesPeriodicBoundaryConditions()

        if self._is_pbc:
            lattice = state.getPeriodicBoxVectors().value_in_unit(angstrom)
            lattice_string = f'Lattice="{lattice[0][0]} {lattice[0][1]} {lattice[0][2]} {lattice[1][0]} {lattice[1][1]} {lattice[1][2]} {lattice[2][0]} {lattice[2][1]} {lattice[2][2]}" '
            properties_string = "".join([lattice_string, self._properties_string])
        else:
            properties_string = self._properties_string

        self._out.write(self._n_atoms_string)
        self._out.write("".join([properties_string, f"{time:.2f} # fs\n"]))

        for atom_idx in self._atomSubset:
            atom_line = f" {self._elements[atom_idx]:<10}{positions[atom_idx][0]:< {_float_width}f}{positions[atom_idx][1]:< {_float_width}f}{positions[atom_idx][2]:< {_float_width}f}"
            if self._includeVelocities:
                atom_line = "".join(
                    [
                        atom_line,
                        f"{velocities[atom_idx][0]:< {_float_width}f}{velocities[atom_idx][1]:< {_float_width}f}{velocities[atom_idx][2]:< {_float_width}f}",
                    ]
                )
            if self._includeForces:
                atom_line = "".join(
                    [
                        atom_line,
                        f"{forces[atom_idx][0]:< {_float_width}f}{forces[atom_idx][1]:< {_float_width}f}{forces[atom_idx][2]:< {_float_width}f}",
                    ]
                )
            self._out.write("".join([atom_line, "\n"]))

        if self._flush_counter != self._flush_every:
            self._flush_counter += 1
            return

        self._flush_counter = 1
        if hasattr(self._out, "flush") and callable(self._out.flush):
            self._out.flush()
