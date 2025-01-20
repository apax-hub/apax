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
from jax.experimental import io_callback
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from apax.config import Config, MDConfig, parse_config
from apax.config.md_config import Integrator
from apax.md.ase_calc import make_ensemble, maybe_vmap
from apax.md.constraints import Constraint, ConstraintBase
from apax.md.dynamics_checks import DynamicsCheckBase, DynamicsChecks
from apax.md.io import H5TrajHandler, TrajHandler, truncate_trajectory_to_checkpoint
from apax.md.md_checkpoint import load_md_state
from apax.md.sim_utils import SimulationFunctions, System
from apax.train.checkpoints import (
    canonicalize_energy_model_parameters,
    restore_parameters,
)
from apax.train.run import setup_logging
from apax.utils.jax_md_reduced import partition, quantity, simulate, space
from apax.utils.transform import make_energy_only_model

log = logging.getLogger(__name__)


def create_energy_fn(model, params, numbers, n_models, shallow=False):
    def full_ensemble(params, R, Z, neighbor, box, offsets, perturbation=None):
        vmodel = jax.vmap(model, (0, None, None, None, None, None, None), 0)
        energies, _ = vmodel(params, R, Z, neighbor, box, offsets, perturbation)
        energy = jnp.mean(energies)
        return energy

    def shallow_ensemble(params, R, Z, neighbor, box, offsets, perturbation=None):
        energies, _ = model(params, R, Z, neighbor, box, offsets, perturbation)
        energy = jnp.mean(energies)
        return energy

    if n_models > 1:
        if shallow:
            energy_fn = shallow_ensemble
        else:
            energy_fn = full_ensemble
    else:
        energy_fn = make_energy_only_model(model)

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


def get_ensemble(ensemble: Integrator, sim_fns):
    energy, shift = sim_fns.energy_fn, sim_fns.shift_fn

    dt = ensemble.dt * units.fs
    nbr_options = nbr_update_options_default

    kT = ensemble.temperature_schedule.get_schedule()
    if ensemble.name == "nve":
        init_fn, apply_fn = simulate.nve(energy, shift, kT(0), dt)
    elif ensemble.name == "nvt":
        thermostat_chain = dict(ensemble.thermostat_chain)
        thermostat_chain["tau"] *= dt

        init_fn, apply_fn = simulate.nvt_nose_hoover(energy, shift, dt, kT(0))

    elif ensemble.name == "npt":
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
            kT(0),
            thermostat_kwargs=thermostat_chain,
            barostat_kwargs=barostat_chain,
        )
        nbr_options = nbr_update_options_npt
    else:
        raise NotImplementedError(
            "Only the NVE and Nose Hoover NVT/NPT thermostats are currently interfaced."
        )

    return init_fn, apply_fn, kT, nbr_options


def handle_checkpoints(state, step, system, load_momenta, ckpt_dir, should_load_ckpt):
    if load_momenta and not should_load_ckpt:
        log.info("loading momenta from starting configuration")
        state = state.set(momentum=system.momenta)

    elif should_load_ckpt:
        state, step = load_md_state(state, ckpt_dir)
    return state, step


def create_evaluation_functions(aux_fn, positions, Z, neighbor, box, dynamics_checks):
    offsets = jnp.zeros((neighbor.idx.shape[1], 3))

    def on_eval(positions, neighbor, box):
        predictions = aux_fn(positions, Z, neighbor, box, offsets)
        all_checks_passed = True

        for check in dynamics_checks:
            check_passed = check.check(predictions)
            all_checks_passed = all_checks_passed & check_passed
        return predictions, all_checks_passed

    predictions = aux_fn(positions, Z, neighbor, box, offsets)
    dummpy_preds = jax.tree_map(lambda x: jnp.zeros_like(x), predictions)

    def no_eval(positions, neighbor, box):
        predictions = dummpy_preds
        all_checks_passed = True
        return predictions, all_checks_passed

    return on_eval, no_eval


def create_constraint_function(constraints: list[ConstraintBase], state):
    all_constraints = [c.create(state) for c in constraints]

    def apply_constraints(state):
        for constraint in all_constraints:
            state = constraint(state)

        return state

    return apply_constraints


def run_sim(
    system: System,
    sim_fns: SimulationFunctions,
    ensemble,
    sim_dir: Path,
    n_steps: int,
    n_inner: int,
    extra_capacity: int,
    rng_key: int,
    traj_handler: TrajHandler,
    load_momenta: bool = False,
    restart: bool = True,
    checkpoint_interval: int = 50_000,
    dynamics_checks: list[DynamicsCheckBase] = [],
    constraints: list[ConstraintBase] = [],
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
    neighbor_fn = sim_fns.neighbor_fn
    ckpt_dir = sim_dir / "ckpts"
    ckpt_dir.mkdir(exist_ok=True)

    log.info("initializing simulation")
    init_fn, apply_fn, kT, nbr_options = get_ensemble(ensemble, sim_fns)

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
    ckpts_exist = any(True for p in ckpt_dir.rglob("*") if "checkpoint" in p.stem)
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

    sampling_rate = traj_handler.sampling_rate
    on_eval, no_eval = create_evaluation_functions(
        sim_fns.auxiliary_fn,
        state.position,
        system.atomic_numbers,
        neighbor,
        system.box,
        dynamics_checks,
    )

    apply_constraints = create_constraint_function(
        constraints,
        state,
    )

    @jax.jit
    def sim(state, outer_step, neighbor):  # TODO make more modular
        def body_fn(i, state):
            state, outer_step, neighbor, all_checks_passed = state
            step = i + outer_step * n_inner

            apply_fn_kwargs = {}
            if isinstance(state, simulate.NPTNoseHooverState):
                box = state.box
            else:
                box = system.box
                apply_fn_kwargs = {"box": box}

            apply_fn_kwargs["kT"] = kT(step)  # Get current Temperature

            state = apply_fn(state, neighbor=neighbor, **apply_fn_kwargs)

            state = apply_constraints(state)

            nbr_kwargs = nbr_options(state)
            neighbor = neighbor.update(state.position, **nbr_kwargs)

            condition = step % sampling_rate == 0
            predictions, check_passed = jax.lax.cond(
                condition, on_eval, no_eval, state.position, neighbor, box
            )

            all_checks_passed = all_checks_passed & check_passed

            # maybe move this to on_eval
            io_callback(traj_handler.step, None, (state, predictions, nbr_kwargs))
            return state, outer_step, neighbor, all_checks_passed

        all_checks_passed = True
        state, outer_step, neighbor, all_checks_passed = jax.lax.fori_loop(
            0, n_inner, body_fn, (state, outer_step, neighbor, all_checks_passed)
        )
        current_temperature = (
            quantity.temperature(velocity=state.velocity, mass=state.mass) / units.kB
        )

        return state, neighbor, current_temperature, all_checks_passed

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
        new_state, neighbor, current_temperature, all_checks_passed = sim(
            state, step, neighbor
        )

        if np.any(np.isnan(state.position)) or np.any(np.isnan(state.velocity)):
            raise ValueError(
                f"NaN encountered, simulation aborted after {step + 1} steps."
            )

        if not all_checks_passed:
            with logging_redirect_tqdm():
                log.critical(f"One or more dynamics checks failed at step: {step + 1}")
            break

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

    ps_per_s = total_sim_time / elapsed_time
    nanosec_per_day = ps_per_s / 1e3 * 60 * 60 * 24

    sec_per_step = elapsed_time / n_steps
    n_atoms = system.positions.shape[0]
    musec_per_step_per_atom = sec_per_step * 1e6 / n_atoms

    log.info("simulation finished after: %.2f s", elapsed_time)
    log.info(
        "performance summary: %.2f ns/day, %.2f mu s/step/atom",
        nanosec_per_day,
        musec_per_step_per_atom,
    )


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

    r_max = model_config.model.basis.r_max
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
                f"one cell vector direction {atoms.cell.lengths() / 2} < {r_max}",
                "can not calculate the correct neighbors",
            )
        if np.any(heights / 2 < r_max):
            log.error(
                "cutoff is larger than box/2 in at least",
                f"one cell vector direction {heights / 2} < {r_max}",
                "can not calculate the correct neighbors",
            )
        displacement_fn, shift_fn = space.periodic_general(
            system.box, fractional_coordinates=frac_coords
        )

    Builder = model_config.model.get_builder()
    builder = Builder(model_config.model.model_dump())
    energy_model = builder.build_energy_model(
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

    _, gradient_model_params = restore_parameters(model_config.data.model_version_path)
    params = canonicalize_energy_model_parameters(gradient_model_params)

    n_models = 1
    shallow = False
    if (
        "ensemble" in model_config.model.model_dump().keys()
        and model_config.model.ensemble is not None
        and model_config.model.ensemble.n_members > 1
    ):
        n_models = model_config.model.ensemble.n_members
        if model_config.model.ensemble.kind == "shallow":
            shallow = True

    energy_fn = create_energy_fn(
        energy_model.apply, params, system.atomic_numbers, n_models, shallow
    )

    auxiliary_fn = builder.build_energy_derivative_model(
        apply_mask=True, init_box=np.array(system.box), inference_disp_fn=displacement_fn
    ).apply

    if n_models > 1 and not shallow:
        auxiliary_fn = maybe_vmap(auxiliary_fn, gradient_model_params)
        auxiliary_fn = make_ensemble(auxiliary_fn)
    else:
        auxiliary_fn = partial(
            auxiliary_fn,
            gradient_model_params,
        )

    sim_fns = SimulationFunctions(energy_fn, auxiliary_fn, shift_fn, neighbor_fn)
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

    dynamics_checks = []
    if md_config.dynamics_checks:
        check_list = [
            DynamicsChecks(check.model_dump()) for check in md_config.dynamics_checks
        ]
        dynamics_checks.extend(check_list)

    constraints = []
    if md_config.constraints:
        constraint_list = [Constraint(c.model_dump()) for c in md_config.constraints]
        constraints.extend(constraint_list)

    n_steps = int(np.ceil(md_config.duration / md_config.ensemble.dt))

    traj_handler = H5TrajHandler(
        system,
        md_config.sampling_rate,
        md_config.buffer_size,
        traj_path,
        md_config.ensemble.dt,
        properties=md_config.properties,
    )
    # TODO implement correct chunking

    run_sim(
        system,
        sim_fns,
        md_config.ensemble,
        n_steps=n_steps,
        n_inner=md_config.n_inner,
        extra_capacity=md_config.extra_capacity,
        load_momenta=md_config.load_momenta,
        traj_handler=traj_handler,
        rng_key=jax.random.PRNGKey(md_config.seed),
        restart=md_config.restart,
        checkpoint_interval=md_config.checkpoint_interval,
        sim_dir=sim_dir,
        dynamics_checks=dynamics_checks,
    )
