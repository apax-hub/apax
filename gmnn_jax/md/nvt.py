import logging
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import yaml
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from flax.training import checkpoints
from jax_md import simulate, space

from gmnn_jax.config import Config, MDConfig
from gmnn_jax.model.gmnn import get_md_model

log = logging.getLogger(__name__)


def run_nvt(
    R,
    atomic_numbers,
    masses,
    box,
    energy_fn,
    neighbor_fn,
    shift_fn,
    dt,
    T,
    n_steps,
    n_inner,
    extra_capacity,
    rng_key,
    traj_name="nvt.traj",
):
    sim_time = dt * n_steps
    K_B = 8.617e-5
    dt = dt * units.fs
    kT = K_B * T

    log.info("initializing simulation")
    neighbor = neighbor_fn.allocate(R, extra_capacity=extra_capacity)
    init_fn, apply_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)
    apply_fn = jax.jit(apply_fn)
    state = init_fn(rng_key, R, masses, neighbor=neighbor)
    # TODO capability to restart md.
    # May require serializing the state instead of ASE Atoms trajectory + conversion
    # Maybe we can use flax checkpoints for that?

    @jax.jit
    def sim(state, neighbor):
        def body_fn(i, state):
            state, neighbor = state
            neighbor = neighbor.update(state.position)
            state = apply_fn(state, neighbor=neighbor)
            return state, neighbor

        return jax.lax.fori_loop(0, n_inner, body_fn, (state, neighbor))

    traj = TrajectoryWriter(traj_name, mode="w")
    start = time.time()
    step = 0
    n_outer = int(n_steps // n_inner)

    log.info("running nvt for %.1f fs", sim_time)
    while step < n_outer:
        new_state, neighbor = sim(state, neighbor)
        if neighbor.did_buffer_overflow:
            log.info("step %d: neighbor list overflowed, reallocating.", step)
            neighbor = neighbor_fn.allocate(state.position)
        else:
            state = new_state
            step += 1
            new_atoms = Atoms(atomic_numbers, state.position, cell=box)
            new_atoms.calc = SinglePointCalculator(new_atoms, forces=state.force)
            traj.write(new_atoms)
    traj.close()

    end = time.time()
    elapsed_time = end - start
    log.info("simulation finished after elapsed time: %.2f s", elapsed_time)


def run_md(model_config, md_config):
    log.info("loading configs for md")
    if isinstance(model_config, str):
        with open(model_config, "r") as stream:
            model_config = yaml.safe_load(stream)

    if isinstance(md_config, str):
        with open(md_config, "r") as stream:
            md_config = yaml.safe_load(stream)

    model_config = Config.parse_obj(model_config)
    md_config = MDConfig.parse_obj(md_config)

    rng_key = jax.random.PRNGKey(md_config.seed)

    log.info("reading structure")
    atoms = read(md_config.initial_structure)

    R = jnp.asarray(atoms.positions)
    atomic_numbers = jnp.asarray(atoms.numbers)
    masses = jnp.asarray(atoms.get_masses())
    box = jnp.asarray(atoms.get_cell().lengths())

    log.info("initializing model")
    displacement_fn, shift_fn = space.periodic(box)

    neighbor_fn, _, model = get_md_model(
        atomic_numbers=atomic_numbers,
        displacement_fn=displacement_fn,
        displacement=displacement_fn,
        box_size=box,
        dr_threshold=md_config.dr_threshold,
        **model_config.model.dict()
    )

    os.makedirs(md_config.sim_dir, exist_ok=True)

    log.info("loading model parameters")
    raw_restored = checkpoints.restore_checkpoint(
        md_config.ckpt_dir, target=None, step=None
    )
    params = jax.tree_map(jnp.asarray, raw_restored["model"]["params"])
    energy_fn = partial(model, params)

    md_init_rng_key, rng_key = jax.random.split(rng_key, 2)
    traj_path = os.path.join(md_config.sim_dir, md_config.traj_name)
    run_nvt(
        R=R,
        atomic_numbers=atomic_numbers,
        masses=masses,
        box=box,
        energy_fn=energy_fn,
        neighbor_fn=neighbor_fn,
        shift_fn=shift_fn,
        dt=md_config.dt,
        T=md_config.T,
        n_steps=md_config.n_steps,
        n_inner=md_config.n_inner,
        extra_capacity=md_config.extra_capacity,
        rng_key=md_init_rng_key,
        traj_name=traj_path,
    )
