"""Stress and energy consistency across periodic cell shapes and sizes.

Training path: explicit image offsets from the host neighbour list (compute_nl).
Inference path: ASECalculator, which uses the jax_md neighbour list + periodic_general displacement when the
cell is thick enough (min height / 2 > r_max) and vesin otherwise.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.build import bulk

from apax.config.model_config import GMNNConfig
from apax.config.train_config import Config
from apax.data.preprocessing import compute_nl
from apax.md.ase_calc import ASECalculator

R_MAX = 5.0


def _deformed(F):
    def make(rep):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(rep)
        atoms.set_cell(atoms.cell.array @ np.asarray(F).T, scale_atoms=True)
        return atoms

    return make


SHAPES = {
    "cubic": _deformed(np.eye(3)),
    "tetragonal": _deformed(np.diag([1.0, 1.0, 1.08])),
    "orthorhombic": _deformed(np.diag([1.0, 1.1, 0.93])),
    "monoclinic": _deformed([[1, 0, 0.25], [0, 1, 0], [0, 0, 1]]),
    "triclinic": _deformed([[1, 0.15, 0.2], [0.05, 1, 0.1], [0.12, 0.07, 1]]),
    "skewed": _deformed([[1, 0.5, 0.4], [0, 1, 0.45], [0, 0, 1]]),
    "hexagonal": lambda rep: bulk("Cu", "hcp", a=2.55, c=4.17).repeat((rep, rep, max(1, rep - 1))),
    "rhombohedral": lambda rep: bulk("Cu", "fcc", a=3.6).repeat(rep),  # primitive fcc, 60 degree angles
}
# repeats giving a thin cell (min height < 2 r_max) and a thick one (>= 2 r_max)
SIZES = {"hexagonal": (2, 5), "rhombohedral": (3, 6), "skewed": (2, 4)}
CASES = [(s, size, rep) for s in SHAPES for size, rep in zip(("thin", "thick"), SIZES.get(s, (2, 3)))]


def _min_height(cell):
    vol = abs(np.linalg.det(cell))
    a, b, c = cell
    return min(vol / np.linalg.norm(np.cross(x, y)) for x, y in ((b, c), (c, a), (a, b)))


def _atoms(shape, rep):
    atoms = SHAPES[shape](rep)
    atoms.rattle(0.05, seed=1)
    return atoms


def _model_config():
    return GMNNConfig(
        basis={"name": "bessel", "n_basis": 8, "r_max": R_MAX},
        nn=[16, 16],
        calc_stress=True,
        descriptor_dtype="fp64",
        readout_dtype="fp64",
    )


def _offset_inputs(atoms):
    """Inputs as the training pipeline builds them (box = cell.T, fractional positions)."""
    box = np.asarray(atoms.cell.array).T
    frac = atoms.get_scaled_positions()
    idx, offsets = compute_nl(frac, box, R_MAX)
    return frac, atoms.numbers, idx, box, offsets


def _params(cfg, atoms):
    builder = cfg.get_builder()(cfg.model_dump())
    inp = _offset_inputs(atoms)
    model = builder.build_energy_derivative_model(init_box=inp[3])
    return builder, model.init(jax.random.PRNGKey(0), *inp)


def _offset_path(builder, params, atoms):
    inp = _offset_inputs(atoms)
    out = builder.build_energy_derivative_model(init_box=inp[3]).apply(params, *inp)
    return float(out["energy"]), np.asarray(out["stress"]) / atoms.get_volume()


def _calculator(cfg, params):
    config = MagicMock(spec=Config)
    config.model = cfg
    with patch("apax.md.ase_calc.restore_parameters", return_value=(config, params)):
        with patch("apax.md.ase_calc.check_for_ensemble", return_value=1):
            return ASECalculator(model_dir=Path("dummy"), calc_stress=True)


def _fd_stress(energy, atoms, h=1e-5):
    """Central differences of E under a homogeneous cell strain, symmetrised, divided by volume."""
    cell, frac, fd = atoms.cell.array.copy(), atoms.get_scaled_positions(), np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            e = []
            for s in (1, -1):
                eps = np.eye(3)
                eps[i, j] += s * h
                b = atoms.copy()
                b.set_cell(cell @ eps.T, scale_atoms=False)
                b.set_scaled_positions(frac)
                e.append(energy(b))
            fd[i, j] = (e[0] - e[1]) / (2 * h)
    return 0.5 * (fd + fd.T) / atoms.get_volume()


@pytest.mark.parametrize("shape,size,rep", CASES, ids=[f"{s}-{z}" for s, z, _ in CASES])
def test_training_path_stress_matches_finite_differences(shape, size, rep):
    atoms = _atoms(shape, rep)
    assert (_min_height(atoms.cell.array) < 2 * R_MAX) == (size == "thin")
    with jax.enable_x64(True):
        builder, params = _params(_model_config(), atoms)
        _, stress = _offset_path(builder, params, atoms)
        fd = _fd_stress(lambda a: _offset_path(builder, params, a)[0], atoms)
    assert np.abs(fd).max() > 1e-7  # nontrivial stress
    np.testing.assert_allclose(stress, fd, rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize("shape,size,rep", CASES, ids=[f"{s}-{z}" for s, z, _ in CASES])
def test_ase_stress_matches_finite_differences(shape, size, rep):
    atoms = _atoms(shape, rep)
    with jax.enable_x64(True):
        cfg = _model_config()
        _, params = _params(cfg, atoms)
        calc = _calculator(cfg, params)

        def energy(a):
            a = a.copy()
            a.calc = calc
            return a.get_potential_energy()

        b = atoms.copy()
        b.calc = calc
        stress = b.get_stress(voigt=False)
        fd = _fd_stress(energy, atoms)
    assert np.abs(fd).max() > 1e-7
    np.testing.assert_allclose(stress, fd, rtol=1e-5, atol=1e-10)


_VERY_SKEWED = _deformed([[1, 1.0, 0.8], [0, 1, 0.9], [0, 0, 1]])
_SKEWED = _deformed([[1, 0.5, 0.4], [0, 1, 0.45], [0, 0, 1]])
THICK_CELLS = [
    pytest.param(_deformed(np.eye(3)), 3, id="cubic"),
    pytest.param(_deformed([[1, 0.15, 0.2], [0.05, 1, 0.1], [0.12, 0.07, 1]]), 3, id="triclinic"),
    pytest.param(_SKEWED, 4, id="skewed"),
    pytest.param(_deformed([[1, 0, 0], [0.9, 1, 0], [0.7, 0.8, 1]]), 5, id="lower-triangular"),
    pytest.param(SHAPES["hexagonal"], 5, id="hexagonal"),
    pytest.param(SHAPES["rhombohedral"], 6, id="rhombohedral"),
    pytest.param(_VERY_SKEWED, 5, id="very-skewed"),
    pytest.param(_VERY_SKEWED, 6, id="very-skewed-large"),
]


def _thick_atoms(make, rep):
    atoms = make(rep)
    atoms.rattle(0.05, seed=1)
    assert _min_height(atoms.cell.array) > 2 * R_MAX  # ASECalculator takes the jax_md path
    return atoms


@pytest.mark.parametrize("make,rep", THICK_CELLS)
def test_jax_md_neighbor_list_finds_all_pairs(make, rep):
    """jax_md periodic_general (fractional minimum image) finds the same pairs as explicit images."""
    from jax_md import partition, space

    atoms = _thick_atoms(make, rep)
    with jax.enable_x64(True):
        box = jnp.asarray(atoms.cell.array.T)
        frac = jnp.asarray(atoms.get_scaled_positions())
        disp, _ = space.periodic_general(box, fractional_coordinates=True)
        nbr = partition.neighbor_list(
            disp, box, R_MAX, 0.5, fractional_coordinates=True, format=partition.Sparse
        ).allocate(frac)
        i, j = np.asarray(nbr.idx)
        keep = i < len(atoms)
        dr = jax.vmap(disp)(frac[j[keep]], frac[i[keep]])
        n_jax = int((jnp.linalg.norm(dr, axis=1) < R_MAX).sum())
    idx, _ = compute_nl(atoms.get_scaled_positions(), atoms.cell.array.T, R_MAX)
    assert n_jax == idx.shape[1]


@pytest.mark.parametrize("make,rep", THICK_CELLS)
def test_ase_energy_matches_explicit_images(make, rep):
    atoms = _thick_atoms(make, rep)
    with jax.enable_x64(True):
        cfg = _model_config()
        builder, params = _params(cfg, atoms)
        e_offsets, _ = _offset_path(builder, params, atoms)
        b = atoms.copy()
        b.calc = _calculator(cfg, params)
        e_ase = b.get_potential_energy()
    np.testing.assert_allclose(e_ase, e_offsets, rtol=1e-10, atol=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason="ASECalculator keeps the neighbour list of the first cell when only the cell changes "
    "(same atoms, jax_md path both times) and misses pairs",
)
def test_ase_calculator_reused_across_cells():
    first = _SKEWED(5)  # 500 atoms
    first.rattle(0.05, seed=1)
    target = _thick_atoms(_VERY_SKEWED, 5)  # also 500 atoms, different cell
    with jax.enable_x64(True):
        cfg = _model_config()
        builder, params = _params(cfg, target)
        e_offsets, _ = _offset_path(builder, params, target)
        calc = _calculator(cfg, params)
        first.calc = calc
        first.get_potential_energy()
        b = target.copy()
        b.calc = calc
        e_reused = b.get_potential_energy()
    np.testing.assert_allclose(e_reused, e_offsets, rtol=1e-10, atol=1e-12)
