from typing import Tuple

import numpy as np
import uncertainty_toolbox as uct
from ase import Atoms

from apax.md import ASECalculator


def compute_calibration_factors(
    calc: ASECalculator,
    atoms_list: list[Atoms],
    batch_size: int = 32,
    criterion: str = "ma_cal",
    shared_factor=False,
    optimizer_bounds: Tuple[float, float] = (1e-2, 1e2),
) -> Tuple[float, float]:
    """Computes global recalibration factors. These can be supplied to the ASEcalculator as part of the GlobalCalibration transformation.

    Parameters
    ----------
    calc: ASECalculator
        Model to be calibrated
    atoms_list: list[Atoms]
        Calibration dataset
    batch_size: int, default = 32
        Processing batch size. Choose the largest allowed by your VRAM.
    criterion: str, default = "ma_cal
        Calibration criterion. See uncertainty_toolbox for more details.
    shared_factor: bool, default = False
        Whether or not to calibrate energies and forces separately.
    optimizer_bounds: Tuple[float, float], default = (1e-2, 1e2)
        Search value bounds.

    Returns
    -------
    e_factor: float
        Global energy calibration factor.
    f_factor: float
        Global forces calibration factor.
    """

    num_atoms = np.array([len(a) for a in atoms_list])
    Etrue = np.array([a.get_potential_energy() for a in atoms_list]) / num_atoms
    Ftrue = np.reshape([a.get_forces() for a in atoms_list], (-1,))
    new_atoms = calc.batch_eval(atoms_list, batch_size=batch_size)

    Epred = np.array([a.get_potential_energy() for a in new_atoms]) / num_atoms
    Fpred = np.reshape([a.get_forces() for a in new_atoms], (-1,))

    Estd = np.array([a.calc.results["energy_uncertainty"] for a in new_atoms]) / num_atoms
    Fstd = np.reshape([a.calc.results["forces_uncertainty"] for a in new_atoms], (-1,))

    e_factor = uct.optimize_recalibration_ratio(
        Epred, Estd, Etrue, criterion, optimizer_bounds=optimizer_bounds
    )

    if shared_factor:
        f_factor = e_factor
    else:
        f_factor = uct.optimize_recalibration_ratio(
            Fpred, Fstd, Ftrue, criterion, optimizer_bounds=optimizer_bounds
        )

    return e_factor, f_factor
