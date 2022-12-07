import time

import jax
import jax.numpy as jnp
from jax_md import simulate, partition, space
from ase import units
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import TrajectoryWriter


def run_nvt(R, atomic_numbers, masses, cell, energy_fn, neighbor_fn, shift_fn, dt, T, n_steps, n_inner, rng_key, traj_name="nvt.traj"):
    nl_format = partition.Sparse

    K_B = 8.617e-5
    dt = dt * units.fs
    kT = K_B * T
    # masses = jnp.array(atoms.get_masses())

    init_fn, apply_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)
    apply_fn = jax.jit(apply_fn)
    state = init_fn(jax.random.PRNGKey(0), R, masses, neighbor=neighbor)
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

    # positions = []
    # forces = []
    traj = TrajectoryWriter(traj_name, mode="w")
    start = time.time()
    step = 0
    while step < n_steps // n_inner:
        new_state, neighbor = sim(state, neighbor)
        if neighbor.did_buffer_overflow:
            print('Neighbor list overflowed, reallocating.')
            neighbor = neighbor_fn.allocate(state.position)
        else:
            state = new_state
            # positions += [state.position]
            # forces += [state.force]
            step += 1
            new_atoms = Atoms(atomic_numbers, state.position, cell=cell)
            new_atoms.calc = SinglePointCalculator(new_atoms, forces=state.forces)
            traj.write(new_atoms)
    traj.close()

    
    end = time.time()
    # print(f"elapsed time: {end - start}") #
    return end - start