from functools import partial
import os

import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from flax.training import checkpoints
from jax_md import space
import yaml
from gmnn_jax.config.train_config import Config

from gmnn_jax.md.md_checkpoint import look_for_checkpoints
from gmnn_jax.model.gmnn import get_md_model


def build_energy_neighbor_fns(atoms, config, params, dr_threshold):
    atomic_numbers = jnp.asarray(atoms.numbers)
    box = jnp.asarray(atoms.get_cell().lengths())

    displacement_fn, _ = space.periodic(box)

    neighbor_fn, _, model = get_md_model(
        atomic_numbers=atomic_numbers,
        displacement_fn=displacement_fn,
        displacement=displacement_fn,
        box_size=box,
        dr_threshold=dr_threshold,
        **config.model.dict(),
    )
    energy_fn = partial(model, params)
    return energy_fn, neighbor_fn


class ASECalculator(Calculator):
    """
    DOES NOT SUPPORT CHANING PARTICLE NUMBERS DURING THE SIMULATION!
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, model_config, dr_threshold=0.5, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.dr_threshold = dr_threshold

        # model_config = os.path.join(model_dir,"config.yaml")

        if isinstance(model_config, str):
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
            atoms, self.model_config, self.params, self.dr_threshold
        )

        @jax.jit
        def body_fn(positions, neighbor):
            neighbor = neighbor.update(positions)
            energy, neg_forces = jax.value_and_grad(energy_fn)(positions, neighbor)
            forces = -neg_forces
            return energy, forces, neighbor

        self.step = body_fn
        self.neighbor_fn = neighbor_fn

    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if self.step is None:
            self.initialize(atoms)
            self.neighbors = self.neighbor_fn.allocate(atoms.positions)
            energy, forces, self.neighbors = self.step(atoms.positions, self.neighbors)

        energy, forces, self.neighbors = self.step(atoms.positions, self.neighbors)

        if self.neighbors.did_buffer_overflow:
            print("neighbor list overflowed, reallocating.")
            self.neighbors = self.neighbor_fn.allocate(atoms.positions)
            energy, forces, self.neighbors = self.step(atoms.positions, self.neighbors)

        self.results = {
            "energy": np.array(energy, dtype=np.float64).item(),
            "forces": np.array(forces, dtype=np.float64),
        }
