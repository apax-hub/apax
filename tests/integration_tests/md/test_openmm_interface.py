import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import yaml
from ase import Atoms
from ase.io import read, write
from openmm.app import PDBReporter
from openmm.openmm import LangevinMiddleIntegrator, PythonForce
from openmm.unit import femtosecond, kelvin, picosecond

from apax.config import Config
from apax.md.openmm_interface import (OpenMMInterface, create_simulation,
                                      create_system)
from apax.utils import jax_md_reduced
from apax.utils.openmm_reporters import XYZReporter

TEST_PATH = pathlib.Path(__file__).parent.resolve()


def test_openmm_interface(get_tmp_path):
    model_confg_path = TEST_PATH / "config.yaml"
    initial_structure_path = get_tmp_path / "atoms.extxyz"

    with open(model_confg_path.as_posix(), "r") as stream:
        model_config_dict = yaml.safe_load(stream)

    model_config_dict["data"]["directory"] = get_tmp_path.as_posix()

    model_config = Config.model_validate(model_config_dict)
    os.makedirs(model_config.data.model_version_path, exist_ok=True)
    model_config.dump_config(model_config.data.model_version_path)

    cell_size = 20.0
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

    options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=1)
    mngr = ocp.CheckpointManager(model_config.data.best_model_path, options=options)
    mngr.save(0, args=ocp.args.StandardSave(ckpt))
    mngr.wait_until_finished()

    atoms = read(initial_structure_path.as_posix())
    interface = OpenMMInterface(model_config.data.model_version_path)
    force = PythonForce(interface.get_python_force_fn(atoms))

    force.setUsesPeriodicBoundaryConditions(True)

    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.5 * femtosecond
    )
    system = create_system(atoms)
    system.addForce(force)
    simulation = create_simulation(atoms, system, integrator)
    simulation.context.setVelocitiesToTemperature(300 * kelvin, 1)

    pdb_path = (get_tmp_path / "test.pdb").as_posix()
    md_steps = 2000
    pdb_write_interval = 500

    simulation.reporters.append(PDBReporter(pdb_path, pdb_write_interval))

    simulation.step(md_steps)

    output_trajectory = read(pdb_path, index=":")

    assert np.all(output_trajectory[0].numbers == atoms.numbers)
    assert np.all(output_trajectory[0].cell == atoms.cell)

    assert len(output_trajectory) == md_steps // pdb_write_interval


def test_xyz_reporter_openmm_interface_periodic(get_tmp_path):
    model_confg_path = TEST_PATH / "config.yaml"
    initial_structure_path = get_tmp_path / "atoms.extxyz"

    with open(model_confg_path.as_posix(), "r") as stream:
        model_config_dict = yaml.safe_load(stream)

    model_config_dict["data"]["directory"] = get_tmp_path.as_posix()

    model_config = Config.model_validate(model_config_dict)
    os.makedirs(model_config.data.model_version_path, exist_ok=True)
    model_config.dump_config(model_config.data.model_version_path)

    cell_size = 20.0
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

    options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=1)
    mngr = ocp.CheckpointManager(model_config.data.best_model_path, options=options)
    mngr.save(0, args=ocp.args.StandardSave(ckpt))
    mngr.wait_until_finished()

    atoms = read(initial_structure_path.as_posix())
    interface = OpenMMInterface(model_config.data.model_version_path)
    force = PythonForce(interface.get_python_force_fn(atoms))
    force.setUsesPeriodicBoundaryConditions(True)

    # Run the simulation with an unreasonably short timestep to
    # ensure that we got the position units correctly.
    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.0001 * femtosecond
    )
    system = create_system(atoms)
    system.addForce(force)
    simulation = create_simulation(atoms, system, integrator)
    simulation.context.setVelocitiesToTemperature(300 * kelvin, 1)

    xyz_path = (get_tmp_path / "test.xyz").as_posix()
    xyz_subset_path = (get_tmp_path / "test_subset.xyz").as_posix()
    md_steps = 3
    xyz_write_interval = 1

    simulation.reporters.append(
        XYZReporter(
            xyz_path,
            xyz_write_interval,
            atoms.symbols,
            enforcePeriodicBox=False,
            includeForces=True,
            includeVelocities=True,
        )
    )

    subset_indices = [0, 1]
    simulation.reporters.append(
        XYZReporter(
            xyz_subset_path,
            xyz_write_interval,
            atoms.symbols,
            enforcePeriodicBox=False,
            includeForces=True,
            includeVelocities=False,
            atomSubset=subset_indices,
        )
    )

    simulation.step(md_steps)

    with open(xyz_path, "r") as file:
        lines = file.readlines()

    assert len(lines) == md_steps // xyz_write_interval * (2 + len(atoms))
    assert int(lines[0].strip()) == len(atoms)

    assert "pos:R:3" in lines[1]
    assert "vel:R:3" in lines[1]
    assert "forces:R:3" in lines[1]

    with open(xyz_path, "r") as file:
        lines = file.readlines()
    for line in lines:
        print(line, end="")

    output_trajectory = read(xyz_path, index=":")

    assert np.all(output_trajectory[0].numbers == atoms.numbers)
    assert np.allclose(atoms.positions, output_trajectory[0].positions, atol=1e-4)
    assert np.all(output_trajectory[0].cell == atoms.cell)

    assert len(output_trajectory) == md_steps // xyz_write_interval

    with open(xyz_subset_path, "r") as file:
        lines_subset = file.readlines()

    subset_atoms = atoms[0]
    assert len(lines_subset) == md_steps // xyz_write_interval * (2 + len(subset_indices))

    assert int(lines_subset[0].strip()) == len(subset_indices)

    assert "Lattice" in lines_subset[1]
    assert "pos:R:3" in lines_subset[1]
    assert "forces:R:3" in lines_subset[1]
    assert "vel:R:3" not in lines_subset[1]

    assert len(lines_subset[2].split()) == 1 + 3 + 3  # spec + pos + forces

    output_trajectory_subset = read(xyz_subset_path, index=":")

    assert np.all(output_trajectory_subset[0].numbers == atoms.numbers[subset_indices])
    assert np.allclose(
        atoms.positions[subset_indices], output_trajectory_subset[0].positions, atol=1e-4
    )
    assert np.all(
        output_trajectory[0].positions[subset_indices]
        == output_trajectory_subset[0].positions,
    )
    assert np.all(
        output_trajectory[0].get_forces()[subset_indices]
        == output_trajectory_subset[0].get_forces()
    )


def test_xyz_reporter_openmm_interface_nonperiodic(get_tmp_path):
    model_confg_path = TEST_PATH / "config.yaml"
    initial_structure_path = get_tmp_path / "atoms.extxyz"

    with open(model_confg_path.as_posix(), "r") as stream:
        model_config_dict = yaml.safe_load(stream)

    model_config_dict["data"]["directory"] = get_tmp_path.as_posix()

    model_config = Config.model_validate(model_config_dict)
    os.makedirs(model_config.data.model_version_path, exist_ok=True)
    model_config.dump_config(model_config.data.model_version_path)

    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    atomic_numbers = np.array([1, 1, 8])
    box = np.diag([0, 0, 0])
    offsets = jnp.full([3, 3], 0)
    atoms = Atoms(atomic_numbers, positions, cell=box)
    write(initial_structure_path.as_posix(), atoms)

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

    options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=1)
    mngr = ocp.CheckpointManager(model_config.data.best_model_path, options=options)
    mngr.save(0, args=ocp.args.StandardSave(ckpt))
    mngr.wait_until_finished()

    atoms = read(initial_structure_path.as_posix())
    interface = OpenMMInterface(model_config.data.model_version_path)
    force = PythonForce(interface.get_python_force_fn(atoms))
    force.setUsesPeriodicBoundaryConditions(False)

    # Run the simulation with an unreasonably short timestep to
    # ensure that we got the position units correctly.
    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.0001 * femtosecond
    )
    system = create_system(atoms)
    system.addForce(force)
    simulation = create_simulation(atoms, system, integrator)
    simulation.context.setVelocitiesToTemperature(300 * kelvin, 1)

    xyz_path = (get_tmp_path / "test.xyz").as_posix()
    md_steps = 3
    xyz_write_interval = 1

    simulation.reporters.append(
        XYZReporter(
            xyz_path,
            xyz_write_interval,
            atoms.symbols,
            enforcePeriodicBox=False,
            includeForces=True,
            includeVelocities=True,
        )
    )

    simulation.step(md_steps)

    with open(xyz_path, "r") as file:
        lines = file.readlines()

    assert len(lines) == md_steps // xyz_write_interval * (2 + len(atoms))
    assert int(lines[0].strip()) == len(atoms)

    assert "Lattice" not in lines[1]
    assert "pos:R:3" in lines[1]
    assert "vel:R:3" in lines[1]
    assert "forces:R:3" in lines[1]

    output_trajectory = read(xyz_path, index=":")

    assert np.all(output_trajectory[0].numbers == atoms.numbers)
    assert np.allclose(atoms.positions, output_trajectory[0].positions, atol=1e-4)
    assert np.all(output_trajectory[0].cell == atoms.cell)

    assert len(output_trajectory) == md_steps // xyz_write_interval
