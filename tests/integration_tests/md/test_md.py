import os
import pathlib
import uuid

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
import znh5md
from ase import Atoms
from ase.io import read, write
from flax.training import checkpoints

from apax.config import Config, MDConfig
from apax.md import run_md
from apax.md.ase_calc import ASECalculator
from apax.utils import jax_md_reduced
from tests.conftest import load_config_and_run_training

TEST_PATH = pathlib.Path(__file__).parent.resolve()


def test_run_md(get_tmp_path):
    model_confg_path = TEST_PATH / "config.yaml"
    md_confg_path = TEST_PATH / "md_config.yaml"

    with open(model_confg_path.as_posix(), "r") as stream:
        model_config_dict = yaml.safe_load(stream)
    with open(md_confg_path.as_posix(), "r") as stream:
        md_config_dict = yaml.safe_load(stream)

    model_config_dict["data"]["directory"] = get_tmp_path.as_posix()
    md_config_dict["sim_dir"] = get_tmp_path.as_posix()
    md_config_dict["initial_structure"] = get_tmp_path.as_posix() + "/atoms.extxyz"

    model_config = Config.model_validate(model_config_dict)
    os.makedirs(model_config.data.model_version_path)
    model_config.dump_config(model_config.data.model_version_path)
    md_config = MDConfig.model_validate(md_config_dict)

    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    atomic_numbers = np.array([1, 2, 2])
    box = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    offsets = jnp.full([3, 3], 0)
    atoms = Atoms(atomic_numbers, positions, cell=box)
    write(md_config.initial_structure, atoms)

    n_species = 119  # int(np.max(atomic_numbers) + 1)

    displacement_fn, _ = jax_md_reduced.space.free()

    neighbor_fn = jax_md_reduced.partition.neighbor_list(
        displacement_or_metric=displacement_fn,
        box=box,
        r_cutoff=model_config.model.basis.r_max,
        format=jax_md_reduced.partition.Sparse,
        fractional_coordinates=False,
    )
    neighbors = neighbor_fn.allocate(positions)

    Builder = model_config.model.get_builder()
    builder = Builder(model_config.model.model_dump(), n_species=n_species)
    model = builder.build_energy_derivative_model(
        apply_mask=False, inference_disp_fn=displacement_fn
    )
    rng_key = jax.random.PRNGKey(model_config.seed)
    params = model.init(
        rng_key,
        positions,
        atomic_numbers,
        neighbors.idx,
        box,
        offsets,
    )

    ckpt = {"model": {"params": params}, "epoch": 0}
    checkpoints.save_checkpoint(
        ckpt_dir=model_config.data.best_model_path,
        target=ckpt,
        step=0,
        overwrite=True,
    )

    run_md(model_config_dict, md_config_dict)

    traj = znh5md.IO(md_config.sim_dir + "/" + md_config.traj_name)[:]
    assert len(traj) == 5  # num steps

    # checkpoint loading
    md_config_dict["duration"] = 0.5

    run_md(model_config_dict, md_config_dict)

    traj = znh5md.IO(md_config.sim_dir + "/" + md_config.traj_name)[:]
    assert len(traj) == 10  # num steps


def test_ase_calc(get_tmp_path):
    model_confg_path = TEST_PATH / "config.yaml"
    initial_structure_path = get_tmp_path / "atoms.extxyz"

    with open(model_confg_path.as_posix(), "r") as stream:
        model_config_dict = yaml.safe_load(stream)

    model_config_dict["data"]["directory"] = get_tmp_path.as_posix()

    model_config = Config.model_validate(model_config_dict)
    os.makedirs(model_config.data.model_version_path, exist_ok=True)
    model_config.dump_config(model_config.data.model_version_path)

    cell_size = 10.0
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    atomic_numbers = np.array([1, 1, 8])
    box = np.diag([cell_size] * 3)
    offsets = jnp.full([3, 3], 0)
    atoms = Atoms(atomic_numbers, positions, cell=box)
    write(initial_structure_path.as_posix(), atoms)

    displacement_fn, _ = jax_md_reduced.space.periodic_general(
        cell_size, fractional_coordinates=False
    )

    neighbor_fn = jax_md_reduced.partition.neighbor_list(
        displacement_or_metric=displacement_fn,
        box=box,
        r_cutoff=model_config.model.basis.r_max,
        format=jax_md_reduced.partition.Sparse,
        fractional_coordinates=False,
    )
    neighbors = neighbor_fn.allocate(positions)

    Builder = model_config.model.get_builder()
    builder = Builder(model_config.model.model_dump())
    model = builder.build_energy_derivative_model(inference_disp_fn=displacement_fn)
    rng_key = jax.random.PRNGKey(model_config.seed)
    params = model.init(
        rng_key,
        jnp.asarray(positions, dtype=jnp.float32),
        jnp.asarray(atomic_numbers),
        neighbors.idx,
        box,
        offsets=offsets,
    )
    ckpt = {"model": {"params": params}, "epoch": 0}

    checkpoints.save_checkpoint(
        ckpt_dir=model_config.data.best_model_path,
        target=ckpt,
        step=0,
        overwrite=True,
    )

    atoms = read(initial_structure_path.as_posix())
    calc = ASECalculator(
        [model_config.data.model_version_path, model_config.data.model_version_path]
    )

    atoms.calc = calc
    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    S = atoms.get_stress()

    assert E != 0
    assert F.shape == (3, 3)
    assert S.shape == (6,)

    assert "energy_uncertainty" in atoms.calc.results.keys()
    assert "forces_uncertainty" in atoms.calc.results.keys()
    assert "stress_uncertainty" in atoms.calc.results.keys()


@pytest.mark.parametrize("num_data", (30,))
def test_jaxmd_schedule_and_thresold(get_tmp_path, example_dataset):
    model_confg_path = TEST_PATH / "config.yaml"
    working_dir = get_tmp_path / str(uuid.uuid4())
    data_path = get_tmp_path / "ds.extxyz"

    write(data_path, example_dataset)

    data_config_mods = {
        "data": {
            "directory": working_dir.as_posix(),
            "experiment": "model",
            "data_path": data_path.as_posix(),
        },
    }
    model_config_dict = load_config_and_run_training(model_confg_path, data_config_mods)

    md_confg_path = TEST_PATH / "md_config_threshold.yaml"

    with open(md_confg_path.as_posix(), "r") as stream:
        md_config_dict = yaml.safe_load(stream)
    md_config_dict["sim_dir"] = get_tmp_path.as_posix()
    md_config_dict["initial_structure"] = get_tmp_path.as_posix() + "/ds.extxyz"
    md_config = MDConfig.model_validate(md_config_dict)

    model_config = Config.model_validate(model_config_dict)

    run_md(model_config, md_config)

    traj = znh5md.IO(md_config.sim_dir + "/" + md_config.traj_name)[:]
    assert len(traj) < 1000  # num steps

    results_keys = list(traj[0].calc.results.keys())

    assert "energy" in results_keys
    assert "forces" in results_keys
    assert "energy_uncertainty" in results_keys
    assert "forces_ensemble" in results_keys

    assert "energy_ensemble" not in results_keys
    assert "forces_uncertainty" not in results_keys
