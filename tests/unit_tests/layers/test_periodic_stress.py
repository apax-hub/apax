import jax
import numpy as np
from ase.build import bulk

from apax.config.model_config import GMNNConfig
from apax.data.preprocessing import compute_nl
from apax.utils.convert import atoms_to_inputs

R_MAX = 5.0


def _inputs(atoms):
    inp = atoms_to_inputs([atoms])
    R, box = np.asarray(inp["positions"][0]), np.asarray(inp["box"][0])
    idx, offsets = compute_nl(R, box, R_MAX)
    return R, np.asarray(inp["numbers"][0]), idx, box, offsets


def test_small_cell_self_images_and_stress():
    # rocksalt primitive cell: lattice vectors (~4 A) < r_max, so atoms see their own images
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    atoms.rattle(0.05, seed=0)
    with jax.enable_x64(True):
        cfg = GMNNConfig(
            basis={"name": "bessel", "n_basis": 8, "r_max": R_MAX},
            nn=[16, 16],
            calc_stress=True,
            descriptor_dtype="fp64",
            readout_dtype="fp64",
        )
        builder = cfg.get_builder()(cfg.model_dump())
        R, Z, idx, box, off = _inputs(atoms)
        model = builder.build_energy_derivative_model(init_box=box)
        params = model.init(jax.random.PRNGKey(0), R, Z, idx, box, off)
        eparams = {"params": params["params"]["energy_model"]}

        def energy(a):
            inp = _inputs(a)
            return float(builder.build_energy_model(init_box=inp[3]).apply(eparams, *inp)[0])

        # energy per atom must not depend on the choice of cell
        sc = atoms.repeat(3)
        np.testing.assert_allclose(energy(atoms) / len(atoms), energy(sc) / len(sc), atol=1e-10)

        # stress * volume must be dE/dstrain (central finite differences of a real cell strain)
        stress = np.asarray(model.apply(params, R, Z, idx, box, off)["stress"])
        h, fd = 1e-5, np.zeros((3, 3))
        cell, frac = atoms.cell.array.copy(), atoms.get_scaled_positions()
        for i in range(3):
            for j in range(3):
                e = []
                for s in (1, -1):
                    eps = np.zeros((3, 3))
                    eps[i, j] = s * h
                    b = atoms.copy()
                    b.set_cell(cell @ (np.eye(3) + eps).T, scale_atoms=False)
                    b.set_scaled_positions(frac)
                    e.append(energy(b))
                fd[i, j] = (e[0] - e[1]) / (2 * h)
        np.testing.assert_allclose(stress, 0.5 * (fd + fd.T), atol=1e-8)
        assert np.abs(fd).max() > 1e-4  # nontrivial stress
