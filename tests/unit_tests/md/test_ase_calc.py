from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp

from apax.config.model_config import (
    FullEnsembleConfig,
    GMNNConfig,
    PropertyHead,
    ShallowEnsembleConfig,
)
from apax.config.train_config import Config
from apax.md.ase_calc import ASECalculator, make_ensemble


def get_mock_config(
    calc_stress, calc_hessian=False, ensemble_config=None, property_heads=None
):
    """Creates a mock Config object for testing."""
    model_config = GMNNConfig(
        calc_stress=calc_stress,
        calc_hessian=calc_hessian,
        ensemble=ensemble_config,
        property_heads=property_heads or [],
    )
    config = MagicMock(spec=Config)
    config.model = model_config
    return config


def test_no_ensemble():
    mock_config = get_mock_config(calc_stress=False)

    with patch("apax.md.ase_calc.restore_parameters") as mock_restore:
        mock_restore.return_value = (mock_config, None)
        with patch("apax.md.ase_calc.check_for_ensemble", return_value=1):
            calc = ASECalculator(model_dir=Path("dummy"))

    # Hessian should not be present by default anymore
    expected_properties = ["energy", "forces"]
    assert sorted(calc.implemented_properties) == sorted(expected_properties)


def test_no_ensemble_with_hessian_override():
    mock_config = get_mock_config(calc_stress=False, calc_hessian=False)

    with patch("apax.md.ase_calc.restore_parameters") as mock_restore:
        mock_restore.return_value = (mock_config, None)
        with patch("apax.md.ase_calc.check_for_ensemble", return_value=1):
            # Explicitly enable Hessian override
            calc = ASECalculator(model_dir=Path("dummy"), calc_hessian=True)

    expected_properties = ["energy", "forces", "hessian"]
    assert sorted(calc.implemented_properties) == sorted(expected_properties)


def test_no_ensemble_with_hessian_disabled_override():
    # Hessian enabled in training config
    mock_config = get_mock_config(calc_stress=False, calc_hessian=True)

    with patch("apax.md.ase_calc.restore_parameters") as mock_restore:
        mock_restore.return_value = (mock_config, None)
        with patch("apax.md.ase_calc.check_for_ensemble", return_value=1):
            # Hessian should be disabled by default now, but let's be explicit
            calc = ASECalculator(model_dir=Path("dummy"), calc_hessian=False)

    expected_properties = ["energy", "forces"]
    assert sorted(calc.implemented_properties) == sorted(expected_properties)


def test_full_ensemble():
    ensemble_config = FullEnsembleConfig(kind="full", n_members=2)
    mock_config = get_mock_config(calc_stress=True, ensemble_config=ensemble_config)

    with patch("apax.md.ase_calc.restore_parameters") as mock_restore:
        mock_restore.return_value = (mock_config, None)
        with patch("apax.md.ase_calc.check_for_ensemble", return_value=2):
            calc = ASECalculator(model_dir=Path("dummy"))

    expected_properties = [
        "energy",
        "forces",
        "stress",
        "energy_uncertainty",
        "energy_ensemble",
        "forces_uncertainty",
        "forces_ensemble",
        "stress_uncertainty",
        "stress_ensemble",
    ]
    assert sorted(calc.implemented_properties) == sorted(expected_properties)


def test_shallow_ensemble_no_force_variance():
    ensemble_config = ShallowEnsembleConfig(
        kind="shallow", n_members=2, force_variance=False
    )
    mock_config = get_mock_config(calc_stress=True, ensemble_config=ensemble_config)

    with patch("apax.md.ase_calc.restore_parameters") as mock_restore:
        mock_restore.return_value = (mock_config, None)
        with patch("apax.md.ase_calc.check_for_ensemble", return_value=2):
            calc = ASECalculator(model_dir=Path("dummy"))

    expected_properties = [
        "energy",
        "forces",
        "stress",
        "energy_uncertainty",
        "energy_ensemble",
    ]
    assert sorted(calc.implemented_properties) == sorted(expected_properties)


def test_shallow_ensemble_with_force_variance():
    ensemble_config = ShallowEnsembleConfig(
        kind="shallow", n_members=2, force_variance=True
    )
    mock_config = get_mock_config(calc_stress=True, ensemble_config=ensemble_config)

    with patch("apax.md.ase_calc.restore_parameters") as mock_restore:
        mock_restore.return_value = (mock_config, None)
        with patch("apax.md.ase_calc.check_for_ensemble", return_value=2):
            calc = ASECalculator(model_dir=Path("dummy"))

    expected_properties = [
        "energy",
        "forces",
        "stress",
        "energy_uncertainty",
        "energy_ensemble",
        "forces_uncertainty",
        "forces_ensemble",
    ]
    assert sorted(calc.implemented_properties) == sorted(expected_properties)


def test_property_head_ensemble():
    property_heads = [PropertyHead(name="dipole", n_shallow_members=2)]
    mock_config = get_mock_config(calc_stress=False, property_heads=property_heads)

    with patch("apax.md.ase_calc.restore_parameters") as mock_restore:
        mock_restore.return_value = (mock_config, None)
        with patch("apax.md.ase_calc.check_for_ensemble", return_value=1):
            calc = ASECalculator(model_dir=Path("dummy"))

    expected_properties = [
        "energy",
        "forces",
        "dipole",
        "dipole_uncertainty",
        "dipole_ensemble",
    ]
    assert sorted(calc.implemented_properties) == sorted(expected_properties)


def test_shallow_and_property_head_ensemble():
    ensemble_config = ShallowEnsembleConfig(
        kind="shallow", n_members=2, force_variance=True
    )
    property_heads = [
        PropertyHead(name="dipole", n_shallow_members=2),
        PropertyHead(name="quadrupole", n_shallow_members=0),
    ]
    mock_config = get_mock_config(
        calc_stress=False,
        ensemble_config=ensemble_config,
        property_heads=property_heads,
    )

    with patch("apax.md.ase_calc.restore_parameters") as mock_restore:
        mock_restore.return_value = (mock_config, None)
        with patch("apax.md.ase_calc.check_for_ensemble", return_value=2):
            calc = ASECalculator(model_dir=Path("dummy"))

    expected_properties = [
        "energy",
        "forces",
        "dipole",
        "quadrupole",
        "energy_uncertainty",
        "energy_ensemble",
        "forces_uncertainty",
        "forces_ensemble",
        "dipole_uncertainty",
        "dipole_ensemble",
    ]
    assert sorted(calc.implemented_properties) == sorted(expected_properties)


def test_make_ensemble_uses_sample_std():
    """make_ensemble (deep ensemble) must report the Bessel-corrected
    sample std 1/(N-1), consistent with the shallow ensemble and
    PropertyHead paths -- not the 1/N population std."""
    energy_ens = jnp.asarray([1.0, 2.0, 4.0, 7.0])  # N = 4 stacked models
    forces_ens = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[1.0, 0.0, 2.0], [0.0, 2.0, 1.0]],
            [[2.0, 1.0, 0.0], [3.0, 0.0, 2.0]],
            [[0.0, 3.0, 1.0], [1.0, 1.0, 4.0]],
        ]
    )

    def stacked_model(positions, Z, idx, box, offsets):
        return {"energy": energy_ens, "forces": forces_ens}

    ensemble_fn = make_ensemble(stacked_model)
    results = ensemble_fn(None, None, None, None, None)

    expected_energy_unc = jnp.std(energy_ens, axis=0, ddof=1)
    population_energy_unc = jnp.std(energy_ens, axis=0)  # ddof=0

    # Reported uncertainty must equal the sample std ...
    assert jnp.allclose(results["energy_uncertainty"], expected_energy_unc)
    # ... and must NOT equal the population std (guards against regression).
    assert not jnp.allclose(results["energy_uncertainty"], population_energy_unc)

    expected_forces_unc = jnp.std(forces_ens, axis=0, ddof=1)
    assert jnp.allclose(results["forces_uncertainty"], expected_forces_unc)

    # Mean must be unchanged by the estimator switch.
    assert jnp.allclose(results["energy"], jnp.mean(energy_ens, axis=0))
