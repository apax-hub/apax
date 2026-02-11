from pathlib import Path
from typing import Any

from ase.units import fs as ase_fs
from openmm.app import Simulation
from openmm.openmm import State
from openmm.unit import angstrom, ev, femtosecond, item


class XYZReporter:
    def __init__(
        self,
        file: str | Path,
        reportInterval: int,
        elements: list[str],
        enforcePeriodicBox: bool | None = None,
        atomSubset: list[int] | None = None,
        includeVelocities: bool = False,
        includeForces: bool = False,
    ):
        self._out = open(file, "w")
        self._reportInterval = reportInterval
        self._elements = elements
        self._enforcePeriodicBox = enforcePeriodicBox
        if atomSubset is None:
            atomSubset = list(range(len(elements)))
        self._atomSubset = atomSubset
        self._n_atoms_string = "".join([str(len(self._atomSubset)), "\n"])

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
        lattice = state.getPeriodicBoxVectors().value_in_unit(angstrom)
        positions = state.getPositions().value_in_unit(angstrom)

        if self._includeForces:
            forces = state.getForces().value_in_unit(ev / (angstrom * item))
        if self._includeVelocities:
            velocities = (
                state.getVelocities(asNumpy=True).value_in_unit(angstrom / femtosecond)
                * ase_fs
            )

        self._out.write(self._n_atoms_string)

        lattice_string = f'Lattice="{lattice[0][0]} {lattice[0][1]} {lattice[0][2]} {lattice[1][0]} {lattice[1][1]} {lattice[1][2]} {lattice[2][0]} {lattice[2][1]} {lattice[2][2]}" '
        self._out.write(
            "".join([lattice_string, self._properties_string, f"{time:.2f} # fs\n"])
        )

        for atom_idx in self._atomSubset:
            atom_line = f"{self._elements[atom_idx]:<10}{positions[atom_idx][0]:<16f}{positions[atom_idx][1]:<16f}{positions[atom_idx][2]:<16f}"
            if self._includeVelocities:
                atom_line = "".join(
                    [
                        atom_line,
                        f"{velocities[atom_idx][0]:<16f}{velocities[atom_idx][2]:<16f}{velocities[atom_idx][2]:<16f}",
                    ]
                )
            if self._includeForces:
                atom_line = "".join(
                    [
                        atom_line,
                        f"{forces[atom_idx][0]:<16f}{forces[atom_idx][1]:<16f}{forces[atom_idx][2]:<16f}",
                    ]
                )
            self._out.write("".join([atom_line, "\n"]))

        if hasattr(self._out, "flush") and callable(self._out.flush):
            self._out.flush()
