import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from ase import Atoms
from ase.io import read, write
from flax.training import checkpoints
from jax_md import partition, space

from gmnn_jax.config import Config, MDConfig
from gmnn_jax.md import run_md
from gmnn_jax.md.ase_calc import ASECalculator
from gmnn_jax.model.gmnn import get_training_model

TEST_PATH = pathlib.Path(__file__).parent.resolve()


def test_run_md(get_tmp_path):
    model_confg_path = TEST_PATH / "config.yaml"
    md_confg_path = TEST_PATH / "md_config.yaml"

    with open(model_confg_path.as_posix(), "r") as stream:
        model_config_dict = yaml.safe_load(stream)
    with open(md_confg_path.as_posix(), "r") as stream:
        md_config_dict = yaml.safe_load(stream)

    model_config_dict["data"]["model_path"] = get_tmp_path.as_posix()
    md_config_dict["sim_dir"] = get_tmp_path.as_posix()
    md_config_dict["initial_structure"] = get_tmp_path.as_posix() + "/atoms.extxyz"

    model_config = Config.parse_obj(model_config_dict)
    md_config = MDConfig.parse_obj(md_config_dict)

    cell_size = 10.0
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    atomic_numbers = np.array([1, 1, 8])
    cell = np.diag([cell_size] * 3)
    atoms = Atoms(atomic_numbers, positions, cell=cell)
    write(md_config.initial_structure, atoms)

    n_atoms = 3
    n_species = int(np.max(atomic_numbers) + 1)

    displacement_fn, _ = space.periodic(cell_size)

    neighbor_fn = partition.neighbor_list(
        displacement_or_metric=displacement_fn,
        box=10.0,
        r_cutoff=model_config.model.r_max,
        format=partition.Sparse,
    )
    neighbors = neighbor_fn.allocate(positions)

    model_init, _ = get_training_model(
        n_atoms=n_atoms,
        n_species=n_species,
        displacement_fn=displacement_fn,
        **model_config.model.dict()
    )
    rng_key = jax.random.PRNGKey(model_config.seed)
    params = model_init(
        rng_key, jnp.asarray(positions), jnp.asarray(atomic_numbers), neighbors.idx
    )
    ckpt = {"model": {"params": params}, "epoch": 0}
    best_dir = os.path.join(
        model_config.data.model_path, model_config.data.model_name, "best"
    )
    checkpoints.save_checkpoint(
        ckpt_dir=best_dir,
        target=ckpt,
        step=0,
        overwrite=True,
    )

    run_md(model_config_dict, md_config_dict)

    traj = read(md_config.sim_dir + "/" + md_config.traj_name, index=":")
    n_outer = int(md_config.n_steps // md_config.n_inner)
    assert len(traj) == n_outer + 1


def test_ase_calc(get_tmp_path):
    model_confg_path = TEST_PATH / "config.yaml"
    initial_structure_path = get_tmp_path / "atoms.extxyz"

    with open(model_confg_path.as_posix(), "r") as stream:
        model_config_dict = yaml.safe_load(stream)

    model_config_dict["data"]["model_path"] = get_tmp_path.as_posix()

    model_config = Config.parse_obj(model_config_dict)
    model_config.dump_config(model_config_dict["data"]["model_path"])

    cell_size = 10.0
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    atomic_numbers = np.array([1, 1, 8])
    cell = np.diag([cell_size] * 3)
    atoms = Atoms(atomic_numbers, positions, cell=cell)
    write(initial_structure_path.as_posix(), atoms)

    n_atoms = 3
    n_species = int(np.max(atomic_numbers) + 1)

    displacement_fn, _ = space.periodic(cell_size)

    neighbor_fn = partition.neighbor_list(
        displacement_or_metric=displacement_fn,
        box=10.0,
        r_cutoff=model_config.model.r_max,
        format=partition.Sparse,
    )
    neighbors = neighbor_fn.allocate(positions)

    model_init, _ = get_training_model(
        n_atoms=n_atoms,
        n_species=n_species,
        displacement_fn=displacement_fn,
        **model_config.model.dict()
    )
    rng_key = jax.random.PRNGKey(model_config.seed)
    params = model_init(
        rng_key, jnp.asarray(positions), jnp.asarray(atomic_numbers), neighbors.idx
    )
    ckpt = {"model": {"params": params}, "epoch": 0}
    best_dir = os.path.join(
        model_config.data.model_path, model_config.data.model_name, "best"
    )
    checkpoints.save_checkpoint(
        ckpt_dir=best_dir,
        target=ckpt,
        step=0,
        overwrite=True,
    )

    atoms = read(initial_structure_path.as_posix())
    calc = ASECalculator(model_config_dict["data"]["model_path"])

    atoms.calc = calc
    E = atoms.get_potential_energy()
    F = atoms.get_forces()

    assert E != 0
    assert F.shape == (3, 3)
