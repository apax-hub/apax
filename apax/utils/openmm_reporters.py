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
    ):
        self._out = open(file, "w")
        self._reportInterval = reportInterval
        self._elements = elements
        self._n_atoms_string = "".join([str(len(elements)), "\n"])
        self._enforcePeriodicBox = enforcePeriodicBox

        # Write header here
        self.properties_string = "Properties=species:S:1:pos:R:3:vel:R:3:forces:R:3 Time="

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation: Simulation) -> dict[str, Any]:
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return {
            "steps": steps,
            "periodic": self._enforcePeriodicBox,
            "include": ["positions", "forces", "velocities"],
        }

    def report(self, simulation: Simulation, state: State) -> None:
        time = state.getTime().value_in_unit(femtosecond)
        lattice = state.getPeriodicBoxVectors().value_in_unit(angstrom)
        positions = state.getPositions().value_in_unit(angstrom)
        forces = state.getForces().value_in_unit(ev / (angstrom * item))
        velocities = (
            state.getVelocities(asNumpy=True).value_in_unit(angstrom / femtosecond)
            * ase_fs
        )

        self._out.write(self._n_atoms_string)

        lattice_string = f'Lattice="{lattice[0][0]} {lattice[0][1]} {lattice[0][2]} {lattice[1][0]} {lattice[1][1]} {lattice[1][2]} {lattice[2][0]} {lattice[2][1]} {lattice[2][2]}" '
        self._out.write(
            "".join([lattice_string, self.properties_string, f"{time:.2f} # fs\n"])
        )

        for i, element in enumerate(self._elements):
            self._out.write(
                f"{element:<10}{positions[i][0]:<16f}{positions[i][1]:<16f}{positions[i][2]:<16f}{velocities[i][0]:<16f}{velocities[i][2]:<16f}{velocities[i][2]:<16f}{forces[i][0]:<16f}{forces[i][1]:<16f}{forces[i][2]:<16f}\n"
            )

        if hasattr(self._out, "flush") and callable(self._out.flush):
            self._out.flush()
