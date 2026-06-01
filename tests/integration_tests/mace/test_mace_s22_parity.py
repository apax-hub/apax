"""s22 single-point parity vs torch-mace MACECalculator on every system.

Locks in the Phase 1 parity fix. Gated by the ``mace_parity`` mark; gated by
fixture skip if the local foundation ``.model`` and converted apax dir are
not present.

Tolerances:
- energy: atol = 1e-5 eV  (foundation models are float64; this is a few ULPs).
- forces: atol = 1e-4 eV/Ang.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.mace_parity

_LOCAL_APAX_DIR = Path("tmp/mace-mpa-0-medium")
_LOCAL_MODEL = Path("tmp/mace-mpa-0-medium.model")


@pytest.fixture(scope="module")
def converted_pair():
    """Return ``(apax_dir, model_path)`` if both exist, else skip the test.

    The converted directory and the source ``.model`` must be present locally
    (``tmp/`` is gitignored). CI gating to a different fixture is left to a
    separate ticket.

    Returns
    -------
    tuple of (Path, Path)
        ``(apax_dir, model_path)``.
    """
    if not _LOCAL_APAX_DIR.exists() or not _LOCAL_MODEL.exists():
        pytest.skip(
            f"local foundation files missing: {_LOCAL_APAX_DIR} and "
            f"{_LOCAL_MODEL}; this test requires a converted MPA-0 medium model."
        )
    return _LOCAL_APAX_DIR, _LOCAL_MODEL


@pytest.fixture(scope="module")
def torch_calc(converted_pair):
    """Build the reference MACECalculator once per module.

    Parameters
    ----------
    converted_pair : tuple of (Path, Path)
        ``(apax_dir, model_path)`` from the ``converted_pair`` fixture.

    Returns
    -------
    mace.calculators.mace.MACECalculator
        Reference calculator instance.
    """
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from mace.calculators.mace import MACECalculator

    _, model_path = converted_pair
    return MACECalculator(
        model_paths=str(model_path),
        default_dtype="float64",
        device="cpu",
    )


@pytest.fixture(scope="module")
def apax_calc(converted_pair):
    """Build the apax ASECalculator once per module.

    Parameters
    ----------
    converted_pair : tuple of (Path, Path)
        ``(apax_dir, model_path)`` from the ``converted_pair`` fixture.

    Returns
    -------
    apax.md.ase_calc.ASECalculator
        apax-native calculator instance.
    """
    from apax.md.ase_calc import ASECalculator

    apax_dir, _ = converted_pair
    return ASECalculator(apax_dir)


@pytest.mark.parametrize("idx", list(range(22)))
def test_s22_energy_force_parity(idx, torch_calc, apax_calc):
    """Each s22 dimer: apax energy/forces match torch-mace within atol.

    Parameters
    ----------
    idx : int
        Index into ``ase.collections.s22``.
    torch_calc : mace.calculators.mace.MACECalculator
        Reference calculator (module-scoped fixture).
    apax_calc : apax.md.ase_calc.ASECalculator
        apax calculator (module-scoped fixture).
    """
    from ase.collections import s22

    atoms_t = list(s22)[idx].copy()
    atoms_t.calc = torch_calc
    e_t = float(atoms_t.get_potential_energy())
    f_t = np.asarray(atoms_t.get_forces())

    atoms_a = list(s22)[idx].copy()
    atoms_a.calc = apax_calc
    e_a = float(atoms_a.get_potential_energy())
    f_a = np.asarray(atoms_a.get_forces())

    np.testing.assert_allclose(
        e_a,
        e_t,
        atol=1e-5,
        err_msg=f"system idx={idx} ({atoms_t.get_chemical_formula()}): "
        f"E_apax={e_a:.10f}, E_torch={e_t:.10f}, diff={e_a - e_t:.3e}",
    )
    np.testing.assert_allclose(
        f_a,
        f_t,
        atol=1e-4,
        err_msg=f"system idx={idx}: max |Δf| = {np.abs(f_a - f_t).max():.3e}",
    )
