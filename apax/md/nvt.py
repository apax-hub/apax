import logging
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from ase import units
from ase.io import read
from flax.training import checkpoints
from jax.experimental.host_callback import barrier_wait, id_tap
from jax_md import partition, quantity, simulate, space
from jax_md.util import Array
from tqdm import trange

from apax.config import Config, MDConfig
from apax.md.io import H5TrajHandler
from apax.md.md_checkpoint import load_md_state, look_for_checkpoints
from apax.model import ModelBuilder

log = logging.getLogger(__name__)


def run_nvt(
    R: Array,
    atomic_numbers: Array,
    masses: Array,
    box: np.array,
    energy_fn,
    neighbor_fn,
    shift_fn,
    dt: float,
    temperature: float,
    n_steps: int,
    n_inner: int,
    sampling_rate: int,
    extra_capacity: int,
    rng_key: int,
    restart: bool = True,
    sim_dir: str = ".",
    traj_name: str = "nvt.h5",
    disable_pbar: bool = False,
):
    """
    Performs NVT MD.

    Parameters
    ----------
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
    dt:
        Time step in fs.
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
    dt = dt * units.fs
    kT = units.kB * temperature
    step = 0
    checkpoint_interval = 5_000_000  # TODO will be supplied in the future

    log.info("initializing simulation")
    neighbor = neighbor_fn.allocate(R, extra_capacity=extra_capacity)

    init_fn, apply_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)
    restart = False  # TODO needs to be implemented
    if restart:
        log.info("looking for checkpoints")
        ckpts_exist = look_for_checkpoints(sim_dir)
        if ckpts_exist:
            log.info("loading previous md state")
            state, step = load_md_state(sim_dir)
        else:
            state = init_fn(rng_key, R, masses, neighbor=neighbor)
    else:
        state = init_fn(rng_key, R, masses, neighbor=neighbor)

    traj_path = os.path.join(sim_dir, traj_name)
    traj_handler = H5TrajHandler(R, atomic_numbers, box, sampling_rate, traj_path)

    n_outer = int(np.ceil(n_steps / n_inner))
    pbar_update_freq = int(np.ceil(500 / n_inner))  # TODO add to config
    pbar_increment = n_inner * pbar_update_freq

    # TODO capability to restart md.
    # May require serializing the state instead of ASE Atoms trajectory + conversion
    # Maybe we can use flax checkpoints for that?
    # -> can't serialize NHState and chain for some reason?
    @jax.jit
    def sim(state, neighbor):
        def body_fn(i, state):
            state, neighbor, current_energy = state
            neighbor = neighbor.update(state.position)
            state = apply_fn(state, neighbor=neighbor)
            current_energy = energy_fn(R=state.position, neighbor=neighbor)

            id_tap(traj_handler.step, (state, current_energy))

            return state, neighbor, current_energy

        id_tap(traj_handler.write, None)

        state, neighbor, current_energy = jax.lax.fori_loop(
            0, n_inner, body_fn, (state, neighbor, 0.0)
        )
        current_temperature = quantity.temperature(
            velocity=state.velocity, mass=state.mass
        )
        return state, neighbor, current_temperature, current_energy

    start = time.time()
    sim_time = n_outer * dt
    log.info("running nvt for %.1f fs", sim_time)
    with trange(
        0, n_steps, desc="Simulation", ncols=100, disable=disable_pbar, leave=True
    ) as sim_pbar:
        while step < n_outer:
            new_state, neighbor, current_temperature, current_energy = sim(
                state, neighbor
            )

            if neighbor.did_buffer_overflow:
                log.info("step %d: neighbor list overflowed, reallocating.", step)
                traj_handler.reset_buffer()
                neighbor = neighbor_fn.allocate(state.position)
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
                    sim_pbar.set_postfix(T=f"{(current_temperature / units.kB):.1f} K")
                    sim_pbar.update(pbar_increment)

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
    log.info("reading structure")
    atoms = read(md_config.initial_structure)

    R = jnp.asarray(atoms.positions, dtype=jnp.float64)
    atomic_numbers = jnp.asarray(atoms.numbers, dtype=jnp.int32)
    masses = jnp.asarray(atoms.get_masses(), dtype=jnp.float64)
    box = jnp.asarray(atoms.get_cell().lengths(), dtype=jnp.float64)

    log.info("initializing model")
    if np.all(box < 1e-6):
        displacement_fn, shift_fn = space.free()
    else:
        displacement_fn, shift_fn = space.periodic_general(
            box, fractional_coordinates=False
        )

    Z = jnp.asarray(atomic_numbers)
    n_species = 119  # int(np.max(Z) + 1)
    builder = ModelBuilder(model_config.model.get_dict(), n_species=n_species)
    model = builder.build_energy_model(
        displacement_fn=displacement_fn, apply_mask=True, init_box=np.array(box)
    )
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box,
        model_config.model.r_max,
        md_config.dr_threshold,
        fractional_coordinates=False,
        format=partition.Sparse,
        disable_cell_list=True,
    )

    os.makedirs(md_config.sim_dir, exist_ok=True)

    log.info("loading model parameters")
    best_dir = os.path.join(
        model_config.data.model_path, model_config.data.model_name, "best"
    )
    raw_restored = checkpoints.restore_checkpoint(best_dir, target=None, step=None)
    params = jax.tree_map(jnp.asarray, raw_restored["model"]["params"])

    energy_fn = partial(
        model.apply, params, Z=Z, box=box, offsets=jnp.array([0.0, 0.0, 0.0])
    )

    return R, atomic_numbers, masses, box, energy_fn, neighbor_fn, shift_fn


def run_md(
    model_config: Config, md_config: MDConfig, log_file="md.log", log_level="error"
):
    """
    Utiliy function to start NVT molecualr dynamics simulations from
    a previousy trained model.

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
    if isinstance(model_config, (str, os.PathLike)):
        with open(model_config, "r") as stream:
            model_config = yaml.safe_load(stream)

    if isinstance(md_config, (str, os.PathLike)):
        with open(md_config, "r") as stream:
            md_config = yaml.safe_load(stream)

    model_config = Config.parse_obj(model_config)
    md_config = MDConfig.parse_obj(md_config)

    rng_key = jax.random.PRNGKey(md_config.seed)
    md_init_rng_key, rng_key = jax.random.split(rng_key, 2)

    R, atomic_numbers, masses, box, energy_fn, neighbor_fn, shift_fn = md_setup(
        model_config, md_config
    )
    n_steps = int(np.ceil(md_config.duration / md_config.dt))

    run_nvt(
        R=R,
        atomic_numbers=atomic_numbers,
        masses=masses,
        box=box,
        energy_fn=energy_fn,
        neighbor_fn=neighbor_fn,
        shift_fn=shift_fn,
        dt=md_config.dt,
        temperature=md_config.temperature,
        n_steps=n_steps,
        n_inner=md_config.n_inner,
        sampling_rate=md_config.sampling_rate,
        extra_capacity=md_config.extra_capacity,
        rng_key=md_init_rng_key,
        restart=md_config.restart,
        sim_dir=md_config.sim_dir,
        traj_name=md_config.traj_name,
    )
