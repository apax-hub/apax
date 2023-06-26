from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from jax_md import partition, space

from apax.config.train_config import parse_train_config
from apax.md.md_checkpoint import look_for_checkpoints
from apax.model import ModelBuilder
from apax.train.eval import load_params


def build_energy_neighbor_fns(atoms, config, params, dr_threshold):
    atomic_numbers = jnp.asarray(atoms.numbers)
    box = jnp.asarray(atoms.get_cell().lengths(), dtype=jnp.float32)

    if np.all(box < 1e-6):
        displacement_fn, _ = space.free()
    else:
        displacement_fn, _ = space.periodic_general(box, fractional_coordinates=True)

    Z = jnp.asarray(atomic_numbers)
    n_species = 119  # int(np.max(Z) + 1)
    builder = ModelBuilder(config.model.get_dict(), n_species=n_species)
    model = builder.build_energy_derivative_model(
        displacement_fn=displacement_fn, apply_mask=True, init_box=np.array(box)
    )
    energy_fn = partial(model.apply, params, Z=Z)
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box,
        config.model.r_max,
        dr_threshold,
        fractional_coordinates=True,
        disable_cell_list=True,
        format=partition.Sparse,
    )
    return energy_fn, neighbor_fn


class ASECalculator(Calculator):
    """
    ASE Calculator for APAX models.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model_dir: Path, dr_threshold: float = 0.5, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.dr_threshold = dr_threshold

        self.model_config = parse_train_config(Path(model_dir) / "config.yaml")
        ckpt_dir = (
            Path(self.model_config.data.model_path) / self.model_config.data.model_name
        )
        ckpt_exists = look_for_checkpoints(ckpt_dir / "best")
        if not ckpt_exists:
            raise FileNotFoundError(f"No checkpoint found at {ckpt_dir}")
        self.params = load_params(ckpt_dir)

        self.step = None
        self.neighbor_fn = None
        self.neighbors = None

    def initialize(self, atoms):
        model, neighbor_fn = build_energy_neighbor_fns(
            atoms,
            self.model_config,
            self.params,
            self.dr_threshold,
        )

        @jax.jit
        def step_fn(positions, neighbor, box):
            if np.any(atoms.get_cell().lengths() > 1e-6):
                inv_box = jnp.linalg.inv(box)
                positions = space.transform(inv_box, positions)
                neighbor = neighbor.update(positions, box=box)
            else:
                neighbor = neighbor.update(positions)
            offsets = jnp.full([neighbor.idx.shape[1], 3], 0)
            results = model(positions, neighbor=neighbor, box=box, offsets=offsets)
            return results, neighbor

        self.step = step_fn
        self.neighbor_fn = neighbor_fn

    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
        if self.step is None or "numbers" in system_changes:
            self.initialize(atoms)
            self.neighbors = self.neighbor_fn.allocate(positions)

        results, self.neighbors = self.step(positions, self.neighbors, box)

        if self.neighbors.did_buffer_overflow:
            print("neighbor list overflowed, reallocating.")
            self.neighbors = self.neighbor_fn.allocate(positions)
            results, self.neighbors = self.step(positions, self.neighbors, box)

        self.results = {k: np.array(v, dtype=np.float64) for k, v in results.items()}
        self.results["energy"] = self.results["energy"].item()
