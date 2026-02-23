import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import yaml
from ase import Atoms
from ase.vibrations import Vibrations

from apax.config import Config
from apax.md.ase_calc import ASECalculator

TEST_PATH = pathlib.Path(__file__).parent.resolve()

def test_ase_hessian(get_tmp_path):
    model_confg_path = TEST_PATH / "config.yaml"

    with open(model_confg_path.as_posix(), "r") as stream:
        model_config_dict = yaml.safe_load(stream)

    model_config_dict["data"]["directory"] = get_tmp_path.as_posix()
    model_config = Config.model_validate(model_config_dict)
    os.makedirs(model_config.data.model_version_path)
    model_config.dump_config(model_config.data.model_version_path)

    # Simple water molecule for vibrations
    # roughly equilibrium geometry for a dummy model
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.7, 0.5],
        [0.0, -0.7, 0.5]
    ], dtype=np.float64)
    atomic_numbers = np.array([8, 1, 1])
    box = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    atoms = Atoms(atomic_numbers, positions, cell=box)

    # Initialize model and save checkpoint
    n_species = 119
    Builder = model_config.model.get_builder()
    builder = Builder(model_config.model.model_dump(), n_species=n_species)
    model = builder.build_energy_derivative_model(apply_mask=False)

    rng_key = jax.random.PRNGKey(model_config.seed)

    # Minimal input for init
    R = jnp.asarray(positions, dtype=jnp.float64)
    Z = jnp.asarray(atomic_numbers)
    idx = jnp.array([[1, 2, 0, 2, 0, 1], [0, 0, 1, 1, 2, 2]])
    offsets = jnp.zeros((6, 3), dtype=jnp.float64)
    box_jax = jnp.zeros((3, 3), dtype=jnp.float64)

    params = model.init(rng_key, R, Z, idx, box_jax, offsets)

    ckpt = {"model": {"params": params}, "epoch": 0}
    options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=1)
    with ocp.CheckpointManager(
        model_config.data.best_model_path, options=options
    ) as mngr:
        mngr.save(0, args=ocp.args.StandardSave(ckpt))

    # Initialize ASE Calculator
    calc = ASECalculator(model_config.data.model_version_path)
    atoms.calc = calc

    # 1. Test direct hessian calculation via calculator
    hessian = calc.get_hessian(atoms)
    assert hessian.shape == (9, 9)
    assert not np.allclose(hessian, 0.0)

    # 2. Test compatibility with Vibrations module
    vib_dir = get_tmp_path / "vib"
    vib = Vibrations(atoms, name=str(vib_dir), nfree=4)

    analytical_hessian = calc.get_hessian(atoms)

    # Numerical Hessian from Vibrations (finite difference of forces)
    vib.run()
    numerical_hessian = vib.get_vibrations().get_hessian().reshape(9, 9)

    # Compare analytical vs numerical
    # tolerance is higher because finite difference is less accurate than analytical
    assert np.allclose(analytical_hessian, numerical_hessian, atol=1e-2)
