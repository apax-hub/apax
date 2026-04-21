"""Parity vs torch-mace ``MACECalculator`` (ASE wrapper). Opt-in only.

Requires:
    uv sync --group mace-convert --extra mace

All tests in this module skip when ``torch`` or ``mace`` are not installed.
"""
import numpy as np
import pytest


pytestmark = pytest.mark.mace_parity


@pytest.fixture(params=["medium-mpa-0", "medium"])
def foundation_name(request):
    return request.param


@pytest.fixture
def ase_water():
    pytest.importorskip("ase")
    from ase import Atoms

    return Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        pbc=False,
    )


@pytest.fixture
def ase_periodic_sio2():
    pytest.importorskip("ase")
    from ase import Atoms

    return Atoms(
        symbols=["Si", "O", "O"],
        positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )


def _torch_energy_forces(name, atoms):
    """Run the upstream MACECalculator and return (energy, forces).

    Parameters
    ----------
    name : str
        Canonical MACE foundation model name.
    atoms : ase.Atoms
        System to evaluate.

    Returns
    -------
    energy : float
        Potential energy in eV.
    forces : np.ndarray, shape (n_atoms, 3)
        Forces in eV/Angstrom.
    """
    pytest.importorskip("ase")
    from mace.calculators.foundations_models import mace_mp

    calc = mace_mp(name, default_dtype="float64", device="cpu")
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    return float(e), np.asarray(f)


def _apax_energy_forces(apax_dir, atoms):
    """Run apax's MaceFoundationEnergyModel + derivative via apax ASE calc.

    Parameters
    ----------
    apax_dir : Path
        Path to the converted apax-native ``.apax`` directory.
    atoms : ase.Atoms
        System to evaluate.

    Returns
    -------
    energy : float
        Potential energy in eV.
    forces : np.ndarray, shape (n_atoms, 3)
        Forces in eV/Angstrom.
    """
    from apax.md.ase_calc import ASECalculator

    calc = ASECalculator(apax_dir)  # apax-native; no torch
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    return float(e), np.asarray(f)


def test_energy_force_parity_water(tmp_path, foundation_name, ase_water):
    """Molecular parity for a 3-atom system."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion

    # 1. convert foundation model to apax-native dir
    dst = tmp_path / f"{foundation_name}.apax"
    run_conversion(foundation_name, dst, head="mp", family="mace_mp")

    # 2. torch reference (via mace_mp ASE calculator)
    e_torch, f_torch = _torch_energy_forces(foundation_name, ase_water.copy())

    # 3. apax prediction
    e_apax, f_apax = _apax_energy_forces(dst, ase_water.copy())

    # 4. parity
    np.testing.assert_allclose(e_apax, e_torch, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(f_apax, f_torch, rtol=1e-4, atol=1e-5)


def test_energy_force_parity_periodic(tmp_path, foundation_name, ase_periodic_sio2):
    """Periodic-box parity: validates neighbor lists + PBC offsets."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion

    dst = tmp_path / f"{foundation_name}.apax"
    run_conversion(foundation_name, dst, head="mp", family="mace_mp")

    e_torch, f_torch = _torch_energy_forces(foundation_name, ase_periodic_sio2.copy())
    e_apax, f_apax = _apax_energy_forces(dst, ase_periodic_sio2.copy())

    np.testing.assert_allclose(e_apax, e_torch, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(f_apax, f_torch, rtol=1e-4, atol=1e-5)


def test_force_consistency_via_finite_difference(tmp_path, foundation_name, ase_water):
    """Independent of torch parity: apax autodiff forces match numerical grad."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion
    from apax.md.ase_calc import ASECalculator

    dst = tmp_path / f"{foundation_name}.apax"
    run_conversion(foundation_name, dst, head="mp", family="mace_mp")

    calc = ASECalculator(dst)
    atoms = ase_water.copy()
    atoms.calc = calc

    f_analytic = atoms.get_forces()
    h = 1e-4
    f_numeric = np.zeros_like(f_analytic)
    for i in range(len(atoms)):
        for d in range(3):
            a = atoms.copy()
            a.positions[i, d] += h
            a.calc = calc
            ep = a.get_potential_energy()
            a = atoms.copy()
            a.positions[i, d] -= h
            a.calc = calc
            em = a.get_potential_energy()
            f_numeric[i, d] = -(ep - em) / (2 * h)

    np.testing.assert_allclose(f_analytic, f_numeric, atol=1e-3)


def test_stress_parity_periodic(tmp_path, foundation_name, ase_periodic_sio2):
    """Validate stress via autodiff matches the upstream torch stress."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion
    from apax.md.ase_calc import ASECalculator
    from mace.calculators.foundations_models import mace_mp

    dst = tmp_path / f"{foundation_name}.apax"
    run_conversion(foundation_name, dst, head="mp", family="mace_mp")

    # torch stress
    torch_calc = mace_mp(foundation_name, default_dtype="float64", device="cpu")
    a = ase_periodic_sio2.copy()
    a.calc = torch_calc
    s_torch = a.get_stress(voigt=False)

    # apax stress — requires calc_stress=True in the apax config
    apax_calc = ASECalculator(dst, calc_stress=True)
    a = ase_periodic_sio2.copy()
    a.calc = apax_calc
    s_apax = a.get_stress(voigt=False)

    np.testing.assert_allclose(s_apax, s_torch, rtol=1e-4, atol=1e-6)
