import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import yaml
from ase import Atoms
from ase.io import read, write
from ase.units import kJ, mol
from openmm.openmm import LangevinMiddleIntegrator, PythonForce
from openmm.unit import angstrom, ev, femtosecond, item, kelvin, picosecond

from apax.config import Config
from apax.md.ase_calc import ASECalculator
from apax.md.openmm_interface import (OpenMMInterface, create_simulation,
                                      create_system)
from apax.utils import jax_md_reduced

TEST_PATH = pathlib.Path(__file__).parent.resolve()


def test_openmm_interface_periodic(get_tmp_path):
    model_confg_path = TEST_PATH / "../../integration_tests/md/config.yaml"
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

    ase_calc = ASECalculator(model_config.data.model_version_path)
    atoms.calc = ase_calc
    ase_forces = atoms.get_forces()
    ase_energy = atoms.get_potential_energy()

    interface = OpenMMInterface(model_config.data.model_version_path)
    force = PythonForce(interface.get_python_force_fn(atoms))
    force.setUsesPeriodicBoundaryConditions(True)

    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.5 * femtosecond
    )
    system = create_system(atoms)
    _ = system.addForce(force)

    simulation = create_simulation(atoms, system, integrator)
    state = simulation.context.getState(forces=True, positions=True, energy=True)

    interface_energy = state.getPotentialEnergy().value_in_unit(ev / item)
    interface_forces = state.getForces(asNumpy=True).value_in_unit(ev / (item * angstrom))
    interface_positions = state.getPositions(asNumpy=True).value_in_unit(angstrom)

    assert np.all(atoms.positions == interface_positions)
    assert np.allclose(ase_energy, interface_energy)
    assert np.allclose(ase_forces, interface_forces)


def test_openmm_interface_nonperiodic(get_tmp_path):
    model_confg_path = TEST_PATH / "../../integration_tests/md/config.yaml"
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
    offsets = jnp.full([3, 3], 0)
    box = np.array([0, 0, 0] * 3)
    atoms = Atoms(atomic_numbers, positions)
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

    ase_calc = ASECalculator(model_config.data.model_version_path)
    atoms.calc = ase_calc
    ase_forces = atoms.get_forces()
    ase_energy = atoms.get_potential_energy()

    interface = OpenMMInterface(model_config.data.model_version_path)
    force = PythonForce(interface.get_python_force_fn(atoms))
    force.setUsesPeriodicBoundaryConditions(False)

    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.5 * femtosecond
    )

    system = create_system(atoms)
    _ = system.addForce(force)

    assert not system.usesPeriodicBoundaryConditions()

    simulation = create_simulation(atoms, system, integrator)
    state = simulation.context.getState(forces=True, positions=True, energy=True)

    interface_energy = state.getPotentialEnergy().value_in_unit(ev / item)
    interface_forces = state.getForces(asNumpy=True).value_in_unit(ev / (item * angstrom))
    interface_positions = state.getPositions(asNumpy=True).value_in_unit(angstrom)

    assert np.all(atoms.positions == interface_positions)
    assert np.allclose(ase_energy, interface_energy)
    assert np.allclose(ase_forces, interface_forces)


def test_openmm_interface_different_unit(get_tmp_path):
    model_confg_path = TEST_PATH / "../../integration_tests/md/config.yaml"
    initial_structure_path = get_tmp_path / "atoms.extxyz"

    with open(model_confg_path.as_posix(), "r") as stream:
        model_config_dict = yaml.safe_load(stream)

    model_config_dict["data"]["directory"] = get_tmp_path.as_posix()
    model_config_dict["data"]["energy_unit"] = "kJ/mol"

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
    offsets = jnp.full([3, 3], 0)
    box = np.array([0, 0, 0] * 3)
    atoms = Atoms(atomic_numbers, positions)
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

    ase_calc = ASECalculator(model_config.data.model_version_path)
    atoms.calc = ase_calc
    ase_forces = atoms.get_forces()
    ase_energy = atoms.get_potential_energy()

    interface = OpenMMInterface(model_config.data.model_version_path)
    force = PythonForce(interface.get_python_force_fn(atoms))
    force.setUsesPeriodicBoundaryConditions(False)

    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.5 * femtosecond
    )

    system = create_system(atoms)
    _ = system.addForce(force)

    assert not system.usesPeriodicBoundaryConditions()

    simulation = create_simulation(atoms, system, integrator)
    state = simulation.context.getState(forces=True, positions=True, energy=True)

    interface_energy = state.getPotentialEnergy().value_in_unit(ev / item)
    interface_forces = state.getForces(asNumpy=True).value_in_unit(ev / (item * angstrom))
    interface_positions = state.getPositions(asNumpy=True).value_in_unit(angstrom)

    assert np.all(atoms.positions == interface_positions)
    assert np.allclose(ase_energy, interface_energy)
    assert np.allclose(ase_forces, interface_forces)
