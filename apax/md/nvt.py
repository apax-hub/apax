import dataclasses
import logging
import os
import time
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from ase import units
from ase.io import read
from jax.experimental.host_callback import barrier_wait, id_tap
from jax_md import partition, quantity, simulate, space
from jax_md.space import transform
from tqdm import trange

from apax.config import Config, MDConfig, parse_config
from apax.md.io import H5TrajHandler
from apax.md.md_checkpoint import load_md_state
from apax.model import ModelBuilder
from apax.train.checkpoints import (
    canonicalize_energy_model_parameters,
    restore_parameters,
)
from apax.utils import jax_md_reduced

log = logging.getLogger(__name__)


def create_energy_fn(model, params, numbers, box, n_models):
    def ensemble(params, R, Z, neighbor, box, offsets):
        vmodel = jax.vmap(model, (0, None, None, None, None, None), 0)
        energies = vmodel(params, R, Z, neighbor, box, offsets)
        energy = jnp.mean(energies)

        return energy

    if n_models > 1:
        energy_fn = ensemble
    else:
        energy_fn = model

    energy_fn = partial(
        energy_fn,
        params,
        Z=numbers,
        offsets=jnp.array([0.0, 0.0, 0.0]),
    )

    return energy_fn


def heights_of_box_sides(box):
    heights = []

    for i in range(len(box)):
        for j in range(i + 1, len(box)):
            area = np.linalg.norm(np.cross(box[i], box[j]))
            height = area / np.linalg.norm(box[i])
            heights.append(height)
            height = area / np.linalg.norm(box[j])
            heights.append(height)

    return np.array(heights)


@dataclasses.dataclass
class System:
    atomic_numbers: jnp.array
    masses: jnp.array
    positions: jnp.array
    box: jnp.array
    momenta: Optional[jnp.array]

    @classmethod
    def from_atoms(cls, atoms):
        atomic_numbers = jnp.asarray(atoms.numbers, dtype=jnp.int32)
        masses = jnp.asarray(atoms.get_masses(), dtype=jnp.float64)
        momenta = atoms.get_momenta()

        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
        box = box.T
        positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
        if np.any(box > 1e-6):
            positions = transform(jnp.linalg.inv(box), positions)

        system = cls(
            atomic_numbers=atomic_numbers,
            masses=masses,
            positions=positions,
            box=box,
            momenta=momenta,
        )

        return system


@dataclasses.dataclass
class SimulationFunctions:
    energy_fn: Callable
    shift_fn: Callable
    neighbor_fn: Callable


def nbr_update_options_default(state):
    return {}


def nbr_update_options_npt(state):
    box = simulate.npt_box(state)
    return {"box": box}


def get_ensemble(ensemble, sim_fns):
    energy, shift = sim_fns.energy_fn, sim_fns.shift_fn

    dt = ensemble.dt * units.fs
    nbr_options = nbr_update_options_default

    if ensemble.name == "nve":
        init_fn, apply_fn = simulate.nve(energy, shift, dt)
    elif ensemble.name == "nvt":
        kT = units.kB * ensemble.temperature
        thermostat_chain = dict(ensemble.thermostat_chain)
        thermostat_chain["tau"] *= dt

        init_fn, apply_fn = simulate.nvt_nose_hoover(energy, shift, dt, kT)

    elif ensemble.name == "npt":
        kT = units.kB * ensemble.temperature
        pressure = ensemble.pressure * units.bar
        thermostat_chain = dict(ensemble.thermostat_chain)
        barostat_chain = dict(ensemble.barostat_chain)
        thermostat_chain["tau"] *= dt
        barostat_chain["tau"] *= dt

        init_fn, apply_fn = simulate.npt_nose_hoover(
            energy,
            shift,
            dt,
            pressure,
            kT,
            thermostat_kwargs=thermostat_chain,
            barostat_kwargs=barostat_chain,
        )
        nbr_options = nbr_update_options_npt
    else:
        raise NotImplementedError(
            "Only the NVE and Nose Hoover NVT/NPT thermostats are currently interfaced."
        )

    return init_fn, apply_fn, nbr_options


def run_nvt(
    system: System,
    sim_fns,
    ensemble,
    n_steps: int,
    n_inner: int,
    sampling_rate: int,
    extra_capacity: int,
    rng_key: int,
    load_momenta: bool = False,
    restart: bool = True,
    sim_dir: str = ".",
    traj_name: str = "nvt.h5",
    disable_pbar: bool = False,
):
    """
    Performs NVT MD.

    Parameters
    ----------
    ensemble:
        Thermodynamic ensemble.
    temperature:
        Temperature of the system in K.
    n_steps:
        Total time steps.
    n_inner:
        JIT compiled inner loop. Also determines atoms buffer size.
    sampling_rate:
        Trajectory dumping interval.
    extra_capacity:
        Extra capacity for the neighborlist.
    rng_key:
        RNG key used to initialize the simulation.
    restart:
        Whether a checkpoint should be loaded. No implemented yet.
    sim_dir:
        Directory where the trajectory and (soon) simulation checkpoints will be saved.
    traj_name:
        File name of the ASE trajectory.
    """
    step = 0
    checkpoint_interval = 5_000_000  # TODO will be supplied in the future
    energy_fn = sim_fns.energy_fn
    neighbor_fn = sim_fns.neighbor_fn

    log.info("initializing simulation")
    init_fn, apply_fn, nbr_options = get_ensemble(ensemble, sim_fns)

    neighbor = sim_fns.neighbor_fn.allocate(
        system.positions, extra_capacity=extra_capacity
    )

    restart = False  # TODO needs to be implemented
    if restart:
        log.info("loading previous md state")
        state, step = load_md_state(sim_dir)
    else:
        state = init_fn(
            rng_key,
            system.positions,
            box=system.box,
            mass=system.masses,
            neighbor=neighbor,
        )

        if load_momenta:
            log.info("loading momenta from starting configuration")
            state = state.set(momentum=system.momenta)

    traj_path = os.path.join(sim_dir, traj_name)
    traj_handler = H5TrajHandler(system, sampling_rate, traj_path)

    n_outer = int(np.ceil(n_steps / n_inner))
    pbar_update_freq = int(np.ceil(500 / n_inner))
    pbar_increment = n_inner * pbar_update_freq

    # TODO capability to restart md.
    # May require serializing the state instead of ASE Atoms trajectory + conversion
    # Maybe we can use flax checkpoints for that?
    # -> can't serialize NHState and chain for some reason?
    @jax.jit
    def sim(state, neighbor):  # TODO make more modular
        def body_fn(i, state):
            state, neighbor = state
            # TODO neighbor update kword factory f(state) -> {}
            if isinstance(state, simulate.NPTNoseHooverState):
                box = state.box
            else:
                system.box
            nbr_kwargs = nbr_options(state)
            current_energy = energy_fn(R=state.position, neighbor=neighbor, box=box)
            state = apply_fn(state, neighbor=neighbor)

            nbr_kwargs = nbr_options(state)
            neighbor = neighbor.update(state.position, **nbr_kwargs)


            id_tap(traj_handler.step, (state, current_energy, nbr_kwargs))
            return state, neighbor

        id_tap(traj_handler.write, None)

        state, neighbor = jax.lax.fori_loop(0, n_inner, body_fn, (state, neighbor))
        current_temperature = (
            quantity.temperature(velocity=state.velocity, mass=state.mass) / units.kB
        )
        return state, neighbor, current_temperature

    start = time.time()
    sim_time = n_outer * ensemble.dt  # * units.fs
    log.info("running nvt for %.1f fs", sim_time)
    sim_pbar = trange(
        0, n_steps, desc="Simulation", ncols=100, disable=disable_pbar, leave=True
    )
    while step < n_outer:
        new_state, neighbor, current_temperature = sim(state, neighbor)

        if neighbor.did_buffer_overflow:
            log.info("step %d: neighbor list overflowed, reallocating.", step)
            traj_handler.reset_buffer()
            neighbor = neighbor_fn.allocate(
                state.position
            )  # TODO check that this actually works
        else:
            state = new_state
            step += 1

            if np.any(np.isnan(state.position)) or np.any(np.isnan(state.velocity)):
                raise ValueError(
                    f"NaN encountered, simulation aborted after {step} steps."
                )

            if step % checkpoint_interval == 0:
                log.info("saving checkpoint at step: %d", step)
                log.info("checkpoints not yet implemented")

            if step % pbar_update_freq == 0:
                sim_pbar.set_postfix(T=f"{(current_temperature):.1f} K")  # set string
                sim_pbar.update(pbar_increment)
    sim_pbar.close()

    barrier_wait()
    traj_handler.write()
    traj_handler.close()
    end = time.time()
    elapsed_time = end - start
    log.info("simulation finished after elapsed time: %.2f s", elapsed_time)


def md_setup(model_config: Config, md_config: MDConfig):
    """
    Sets up the energy and neighborlist functions for an MD simulation,
    loads the initial structure.

    Parameters
    ----------
    model_config:
        Configuration of the model used as an interatomic potential.
    md_config:
        configuration of the MD simulation.

    Returns
    -------
    R:
        Initial positions in Angstrom.
    atomic_numbers:
        Atomic numbers of the system.
    masses:
        Atomic masses in ASE units.
    box:
        Side length of the cubic box.
    energy_fn:
        Interatomic potential.
    neighbor_fn:
        Neighborlist function.
    shift_fn:
        Shift function for the integrator.
    """
    os.makedirs(md_config.sim_dir, exist_ok=True)

    log.info("reading structure")
    atoms = read(md_config.initial_structure)
    system = System.from_atoms(atoms)

    r_max = model_config.model.r_max
    log.info("initializing model")
    if np.all(system.box < 1e-6):
        displacement_fn, shift_fn = space.free()
    else:
        heights = heights_of_box_sides(system.box)

        if np.any(atoms.cell.lengths() / 2 < r_max):
            log.error(
                "cutoff is larger than box/2 in at least",
                f"one cell vector direction {atoms.cell.lengths()/2} < {r_max}",
                "can not calculate the correct neighbors",
            )
        if np.any(heights / 2 < r_max):
            log.error(
                "cutoff is larger than box/2 in at least",
                f"one cell vector direction {heights/2} < {r_max}",
                "can not calculate the correct neighbors",
            )
        displacement_fn, shift_fn = space.periodic_general(
            system.box, fractional_coordinates=True
        )

    n_species = 119  # int(np.max(Z) + 1)
    # TODO large forces at init, maybe displacement fn is incorrect?
    builder = ModelBuilder(model_config.model.get_dict(), n_species=n_species)
    model = builder.build_energy_model(
        apply_mask=True, init_box=np.array(system.box), inference_disp_fn=displacement_fn
    )
    neighbor_fn = jax_md_reduced.partition.neighbor_list(
        displacement_fn,
        system.box,
        r_max,
        md_config.dr_threshold,
        fractional_coordinates=True,
        format=partition.Sparse,
        disable_cell_list=True,
    )

    _, params = restore_parameters(model_config.data.model_version_path())
    params = canonicalize_energy_model_parameters(params)
    energy_fn = create_energy_fn(
        model.apply, params, system.atomic_numbers, system.box, model_config.n_models
    )
    sim_fns = SimulationFunctions(energy_fn, shift_fn, neighbor_fn)
    return system, sim_fns


def run_md(
    model_config: Config, md_config: MDConfig, log_file="md.log", log_level="error"
):
    """
    Utiliy function to start NVT molecualr dynamics simulations from
    a previously trained model.

    Parameters
    ----------
    model_config:
        Configuration of the model used as an interatomic potential.
    md_config:
        configuration of the MD simulation.
    """
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logging.basicConfig(filename=log_file, level=log_levels[log_level])

    log.info("loading configs for md")
    model_config = parse_config(model_config)
    md_config = parse_config(md_config, mode="md")

    system, sim_fns = md_setup(model_config, md_config)
    n_steps = int(np.ceil(md_config.duration / md_config.ensemble.dt))

    run_nvt(
        system,
        sim_fns,
        md_config.ensemble,
        n_steps=n_steps,
        n_inner=md_config.n_inner,
        sampling_rate=md_config.sampling_rate,
        extra_capacity=md_config.extra_capacity,
        load_momenta=md_config.load_momenta,
        rng_key=jax.random.PRNGKey(md_config.seed),
        restart=md_config.restart,
        sim_dir=md_config.sim_dir,
        traj_name=md_config.traj_name,
    )
