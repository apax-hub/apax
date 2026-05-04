"""Parity vs torch-mace MACECalculator on the same ase.Atoms.

Gated by mace_parity. Requires:
    uv sync --group mace-convert --extra mace
"""
import numpy as np
import pytest

pytestmark = pytest.mark.mace_parity


@pytest.fixture(params=["small"])
def foundation_name(request):
    """Return canonical MACE foundation model name to evaluate.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request providing the parametrized name.

    Returns
    -------
    str
        Canonical MACE foundation model name.
    """
    return request.param


@pytest.fixture
def ase_water():
    """Return a simple non-periodic water molecule.

    Returns
    -------
    ase.Atoms
        Water molecule with O at origin and two H at chemically reasonable
        positions; no PBC.
    """
    from ase import Atoms
    return Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        pbc=False,
    )


def _torch_energy_forces(name, atoms):
    """Return reference (energy, forces) using torch-mace ``mace_mp``.

    Parameters
    ----------
    name : str
        Foundation model name.
    atoms : ase.Atoms
        System to evaluate.

    Returns
    -------
    energy : float
        Potential energy in eV.
    forces : np.ndarray, shape (n_atoms, 3)
        Forces in eV/Angstrom.
    """
    from mace.calculators.foundations_models import mace_mp
    calc = mace_mp(name, default_dtype="float64", device="cpu")
    atoms.calc = calc
    return float(atoms.get_potential_energy()), np.asarray(atoms.get_forces())


def _apax_energy_forces(apax_dir, atoms):
    """Return apax-native (energy, forces) using the converted model.

    Parameters
    ----------
    apax_dir : pathlib.Path
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
    calc = ASECalculator(apax_dir)
    atoms.calc = calc
    return float(atoms.get_potential_energy()), np.asarray(atoms.get_forces())


def test_energy_force_parity_water(tmp_path, foundation_name, ase_water):
    """End-to-end: convert -> ASECalculator(converted_dir) -> match torch."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion

    dst = tmp_path / f"{foundation_name}.apax"
    run_conversion(foundation_name, dst, head="default", family="mace_mp")

    e_torch, f_torch = _torch_energy_forces(foundation_name, ase_water.copy())
    e_apax,  f_apax  = _apax_energy_forces(dst, ase_water.copy())

    print(f"\nE_torch = {e_torch:.6f}, E_apax = {e_apax:.6f}, diff = {e_apax - e_torch:.3e}")
    print(f"F_torch:\n{f_torch}\nF_apax:\n{f_apax}\nF_diff:\n{f_apax - f_torch}")

    np.testing.assert_allclose(e_apax, e_torch, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(f_apax, f_torch, rtol=1e-3, atol=1e-4)


def test_force_consistency_via_finite_difference(tmp_path, foundation_name, ase_water):
    """Independent of torch: apax autodiff forces match numerical grad."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion
    from apax.md.ase_calc import ASECalculator

    dst = tmp_path / f"{foundation_name}.apax"
    run_conversion(foundation_name, dst, head="default", family="mace_mp")

    calc = ASECalculator(dst)
    atoms = ase_water.copy()
    atoms.calc = calc

    f_analytic = atoms.get_forces()
    h = 1e-4
    f_numeric = np.zeros_like(f_analytic)
    for i in range(len(atoms)):
        for d in range(3):
            a = atoms.copy(); a.positions[i, d] += h; a.calc = calc
            ep = a.get_potential_energy()
            a = atoms.copy(); a.positions[i, d] -= h; a.calc = calc
            em = a.get_potential_energy()
            f_numeric[i, d] = -(ep - em) / (2 * h)

    np.testing.assert_allclose(f_analytic, f_numeric, atol=1e-3)
