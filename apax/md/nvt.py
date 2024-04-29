import logging
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ase import units
from ase.io import read
from flax.training import checkpoints
from jax.experimental.host_callback import barrier_wait, id_tap
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from apax.config import Config, MDConfig, parse_config
from apax.md.io import H5TrajHandler, TrajHandler, truncate_trajectory_to_checkpoint
from apax.md.md_checkpoint import load_md_state
from apax.md.sim_utils import SimulationFunctions, System
from apax.model import ModelBuilder
from apax.train.checkpoints import (
    canonicalize_energy_model_parameters,
    restore_parameters,
)
from apax.train.run import setup_logging
from apax.utils.jax_md_reduced import partition, quantity, simulate, space

log = logging.getLogger(__name__)


def create_energy_fn(model, params, numbers, n_models):
    def ensemble(params, R, Z, neighbor, box, offsets, perturbation=None):
        vmodel = jax.vmap(model, (0, None, None, None, None, None, None), 0)
        energies = vmodel(params, R, Z, neighbor, box, offsets, perturbation)
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


def handle_checkpoints(state, step, system, load_momenta, ckpt_dir, should_load_ckpt):
    if load_momenta and not should_load_ckpt:
        log.info("loading momenta from starting configuration")
        state = state.set(momentum=system.momenta)

    elif should_load_ckpt:
        state, step = load_md_state(state, ckpt_dir)
    return state, step


def run_nvt(
    system: System,
    sim_fns,
    ensemble,
    sim_dir: Path,
    n_steps: int,
    n_inner: int,
    extra_capacity: int,
    rng_key: int,
    load_momenta: bool = False,
    restart: bool = True,
    checkpoint_interval: int = 50_000,
    traj_handler: TrajHandler = TrajHandler(),
    disable_pbar: bool = False,
):
    """
    Performs NVT MD.

    Parameters
    ----------
    ensemble :
        Thermodynamic ensemble.
    n_steps : int
        Total time steps.
    n_inner : int
        JIT compiled inner loop. Also determines atoms buffer size.
    extra_capacity : int
        Extra capacity for the neighborlist.
    rng_key : int
        RNG key used to initialize the simulation.
    restart : bool, default = True
        Whether a checkpoint should be loaded. No implemented yet.
    checkpoint_interval : int, default = 50_000
        Number of time steps between saving
        full simulation state checkpoints.
    sim_dir : Path
        Directory where the trajectory and simulation checkpoints will be saved.
    """
    energy_fn = sim_fns.energy_fn
    neighbor_fn = sim_fns.neighbor_fn
    ckpt_dir = sim_dir / "ckpts"
    ckpt_dir.mkdir(exist_ok=True)

    log.info("initializing simulation")
    init_fn, apply_fn, nbr_options = get_ensemble(ensemble, sim_fns)

    neighbor = sim_fns.neighbor_fn.allocate(
        system.positions, extra_capacity=extra_capacity
    )

    state = init_fn(
        rng_key,
        system.positions,
        box=system.box,
        mass=system.masses,
        neighbor=neighbor,
    )

    step = 0
    ckpts_exist = any([True for p in ckpt_dir.rglob("*") if "checkpoint" in p.stem])
    should_load_ckpt = restart and ckpts_exist
    state, step = handle_checkpoints(
        state, step, system, load_momenta, ckpt_dir, should_load_ckpt
    )
    if should_load_ckpt:
        length = step * n_inner
        truncate_trajectory_to_checkpoint(traj_handler.traj_path, length)

    async_manager = checkpoints.AsyncManager()

    n_outer = int(np.ceil(n_steps / n_inner))
    pbar_update_freq = int(np.ceil(500 / n_inner))
    pbar_increment = n_inner * pbar_update_freq

    @jax.jit
    def sim(state, neighbor):  # TODO make more modular
        def body_fn(i, state):
            state, neighbor = state
            if isinstance(state, simulate.NPTNoseHooverState):
                box = state.box
                apply_fn_kwargs = {}
            else:
                box = system.box
                apply_fn_kwargs = {"box": box}

            current_energy = energy_fn(R=state.position, neighbor=neighbor, box=box)
            nbr_kwargs = nbr_options(state)
            state = apply_fn(state, neighbor=neighbor, **apply_fn_kwargs)

            nbr_kwargs = nbr_options(state)
            neighbor = neighbor.update(state.position, **nbr_kwargs)

            id_tap(traj_handler.step, (state, current_energy, nbr_kwargs))
            return state, neighbor

        state, neighbor = jax.lax.fori_loop(0, n_inner, body_fn, (state, neighbor))
        current_temperature = (
            quantity.temperature(velocity=state.velocity, mass=state.mass) / units.kB
        )
        return state, neighbor, current_temperature

    start = time.time()
    total_sim_time = n_steps * ensemble.dt / 1000
    log.info("running simulation for %.1f ps", total_sim_time)
    initial_time = step * n_inner
    sim_pbar = trange(
        initial_time,
        n_steps,
        initial=initial_time,
        total=n_steps,
        desc="Simulation",
        ncols=100,
        disable=disable_pbar,
        leave=True,
    )
    while step < n_outer:
        new_state, neighbor, current_temperature = sim(state, neighbor)

        if neighbor.did_buffer_overflow:
            with logging_redirect_tqdm():
                log.warn("step %d: neighbor list overflowed, reallocating.", step)
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
                with logging_redirect_tqdm():
                    current_sim_time = step * n_inner * ensemble.dt / 1000
                    log.info(
                        f"saving checkpoint at {current_sim_time:.1f} ps - step: {step}"
                    )
                ckpt = {"state": state, "step": step}
                checkpoints.save_checkpoint(
                    ckpt_dir=ckpt_dir.resolve(),
                    target=ckpt,
                    step=step,
                    overwrite=True,
                    keep=2,
                    async_manager=async_manager,
                )

            if step % pbar_update_freq == 0:
                sim_pbar.set_postfix(T=f"{(current_temperature):.1f} K")  # set string
                sim_pbar.update(pbar_increment)

    # In case of mismatch update freq and n_steps, we can set it to 100% manually
    sim_pbar.update(n_steps - sim_pbar.n)
    sim_pbar.close()

    barrier_wait()
    ckpt = {"state": state, "step": step}
    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir.resolve(),
        target=ckpt,
        step=step,
        overwrite=True,
        keep=2,
        async_manager=async_manager,
    )
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
    model_config : Config
        Configuration of the model used as an interatomic potential.
    md_config : MDConfig
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
    system = System.from_atoms(atoms)

    r_max = model_config.model.r_max
    log.info("initializing model")
    if np.all(system.box < 1e-6):
        frac_coords = False
        displacement_fn, shift_fn = space.free()
    else:
        frac_coords = True
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
            system.box, fractional_coordinates=frac_coords
        )

    builder = ModelBuilder(model_config.model.get_dict())
    model = builder.build_energy_model(
        apply_mask=True, init_box=np.array(system.box), inference_disp_fn=displacement_fn
    )
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        system.box,
        r_max,
        md_config.dr_threshold,
        fractional_coordinates=frac_coords,
        format=partition.Sparse,
        disable_cell_list=True,
    )

    _, params = restore_parameters(model_config.data.model_version_path)
    params = canonicalize_energy_model_parameters(params)
    energy_fn = create_energy_fn(
        model.apply, params, system.atomic_numbers, model_config.n_models
    )
    sim_fns = SimulationFunctions(energy_fn, shift_fn, neighbor_fn)
    return system, sim_fns


def run_md(model_config: Config, md_config: MDConfig, log_level="error"):
    """
    Utiliy function to start NVT molecualr dynamics simulations from
    a previously trained model.

    Parameters
    ----------
    model_config : Config
        Configuration of the model used as an interatomic potential.
    md_config : MDConfig
        configuration of the MD simulation.
    """

    model_config = parse_config(model_config)
    md_config = parse_config(md_config, mode="md")

    sim_dir = Path(md_config.sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)
    log_file = sim_dir / "md.log"
    setup_logging(log_file, log_level)
    traj_path = sim_dir / md_config.traj_name

    system, sim_fns = md_setup(model_config, md_config)
    n_steps = int(np.ceil(md_config.duration / md_config.ensemble.dt))

    traj_handler = H5TrajHandler(
        system,
        md_config.sampling_rate,
        md_config.buffer_size,
        traj_path,
        md_config.ensemble.dt,
    )
    # TODO implement correct chunking

    run_nvt(
        system,
        sim_fns,
        md_config.ensemble,
        n_steps=n_steps,
        n_inner=md_config.n_inner,
        extra_capacity=md_config.extra_capacity,
        load_momenta=md_config.load_momenta,
        rng_key=jax.random.PRNGKey(md_config.seed),
        restart=md_config.restart,
        checkpoint_interval=md_config.checkpoint_interval,
        sim_dir=sim_dir,
        traj_handler=traj_handler,
    )
