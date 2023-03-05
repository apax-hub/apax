import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from ase.calculators.calculator import Calculator, all_changes
from flax.training import checkpoints
from jax_md import partition, space

from apax.config.train_config import Config
from apax.md.md_checkpoint import look_for_checkpoints
from apax.model import ModelBuilder, get_md_model


def build_energy_neighbor_fns(atoms, config, params, dr_threshold, use_flax=True):
    atomic_numbers = jnp.asarray(atoms.numbers)
    box = jnp.asarray(atoms.get_cell().lengths(), dtype=jnp.float32)

    if np.all(box < 1e-6):
        displacement_fn, _ = space.free()
    else:
        displacement_fn, _ = space.periodic_general(box, fractional_coordinates=False)

    if use_flax:
        Z = jnp.asarray(atomic_numbers)
        n_species = int(np.max(Z) + 1)
        builder = ModelBuilder(config.model.get_dict(), n_species=n_species)
        model = builder.build_energy_model(
            displacement_fn=displacement_fn, apply_mask=False, init_box=np.array(box)
        )
        energy_fn = partial(model.apply, params, Z=Z, box=box)
        neighbor_fn = partition.neighbor_list(
            displacement_fn,
            box,
            config.model.r_max,
            dr_threshold,
            fractional_coordinates=False,
            format=partition.Sparse,
        )
    else:
        neighbor_fn, model = get_md_model(
            atomic_numbers=atomic_numbers,
            displacement_fn=displacement_fn,
            displacement=displacement_fn,
            box=box,
            dr_threshold=dr_threshold,
            **config.model.get_dict(),
        )
        energy_fn = partial(model.apply, params)
    return energy_fn, neighbor_fn


class ASECalculator(Calculator):
    """
    DOES NOT SUPPORT CHANING PARTICLE NUMBERS DURING THE SIMULATION!
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self, model_dir: Path, dr_threshold: float = 0.5, use_flax=True, **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.dr_threshold = dr_threshold
        self.use_flax = use_flax

        model_config = Path(model_dir) / "config.yaml"
        with open(model_config, "r") as stream:
            model_config = yaml.safe_load(stream)

        self.model_config = Config.parse_obj(model_config)

        ckpt_dir = os.path.join(
            self.model_config.data.model_path, self.model_config.data.model_name, "best"
        )

        ckpt_exists = look_for_checkpoints(ckpt_dir)
        assert ckpt_exists
        raw_restored = checkpoints.restore_checkpoint(ckpt_dir, target=None, step=None)
        self.params = jax.tree_map(jnp.asarray, raw_restored["model"]["params"])

        self.step = None
        self.neighbor_fn = None
        self.neighbors = None

    def initialize(self, atoms):
        energy_fn, neighbor_fn = build_energy_neighbor_fns(
            atoms,
            self.model_config,
            self.params,
            self.dr_threshold,
            use_flax=self.use_flax,
        )

        @jax.jit
        def body_fn(positions, neighbor):
            neighbor = neighbor.update(positions)
            energy, neg_forces = jax.value_and_grad(energy_fn)(
                positions, neighbor=neighbor
            )
            forces = -neg_forces
            return energy, forces, neighbor

        self.step = body_fn
        self.neighbor_fn = neighbor_fn

    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        positions = jnp.asarray(atoms.positions, dtype=jnp.float32)

        if self.step is None or "numbers" in system_changes or "box" in system_changes:
            self.initialize(atoms)

            self.neighbors = self.neighbor_fn.allocate(positions)
            energy, forces, self.neighbors = self.step(positions, self.neighbors)

        energy, forces, self.neighbors = self.step(positions, self.neighbors)

        if self.neighbors.did_buffer_overflow:
            print("neighbor list overflowed, reallocating.")
            self.neighbors = self.neighbor_fn.allocate(positions)
            energy, forces, self.neighbors = self.step(positions, self.neighbors)

        self.results = {
            "energy": np.array(energy, dtype=np.float64).item(),
            "forces": np.array(forces, dtype=np.float64),
        }