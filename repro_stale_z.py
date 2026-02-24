import os
import pathlib
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import yaml
from ase import Atoms
from apax.config import Config
from apax.md.ase_calc import ASECalculator

def test_repro():
    model_config_dict = {
        "n_epochs": 1,
        "data": {
            "directory": "tmp_repro",
            "experiment": "repro",
            "n_train": 1,
            "n_valid": 1,
            "batch_size": 1,
        },
        "model": {
            "name": "gmnn",
            "basis": {"r_max": 3.0, "n_basis": 10},
            "n_radial": 5,
            "nn": [16, 16],
            "calc_stress": False,
            "calc_hessian": False,
            "descriptor_dtype": "fp64",
            "readout_dtype": "fp32",
            "scale_shift_dtype": "fp64",
        },
        "metrics": [{"name": "energy"}],
        "loss": [{"name": "energy", "weight": 1.0}],
        "seed": 42,
    }
    
    get_tmp_path = pathlib.Path("tmp_repro")
    get_tmp_path.mkdir(exist_ok=True)
    
    model_config = Config.model_validate(model_config_dict)
    model_dir = model_config.data.model_version_path
    os.makedirs(model_dir, exist_ok=True)
    model_config.dump_config(model_dir)

    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.7, 0.5], [0.0, -0.7, 0.5]], dtype=np.float64)
    atomic_numbers = np.array([8, 1, 1])
    atoms = Atoms(atomic_numbers, positions)

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
    options = ocp.CheckpointManagerOptions(max_to_keep=1)
    with ocp.CheckpointManager(model_config.data.best_model_path, options=options) as mngr:
        mngr.save(0, args=ocp.args.StandardSave(ckpt))

    calc = ASECalculator(model_dir, calc_hessian=False)
    atoms.calc = calc

    print("Initial Hessian calculation...")
    h1 = calc.get_hessian(atoms)
    
    # Change species
    print("Changing species...")
    atoms.numbers = np.array([7, 1, 1])
    
    print("Second Hessian calculation...")
    # This should trigger initialize() and set hessian_step = None
    # Then _initialize_hessian(atoms) should be called with new Z
    h2 = calc.get_hessian(atoms)
    
    print(f"Hessian 1 shape: {h1.shape}")
    print(f"Hessian 2 shape: {h2.shape}")
    
    print("Success!")

if __name__ == "__main__":
    test_repro()
