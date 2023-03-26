from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from flax.training import checkpoints
from jax.experimental import jax2tf
from jax_md import partition, space

from apax.md.md_checkpoint import look_for_checkpoints
from apax.model.builder import ModelBuilder


def deploy_to_savedmodel(atoms, model_config, deployed_path):
    model_dir = Path(model_config.data.model_path)
    ckpt_dir = model_dir / model_config.data.model_name / "best"

    R = jnp.asarray(atoms.positions)
    Z = jnp.asarray(atoms.numbers)
    n_species = int(np.max(Z) + 1)
    n_atoms = len(atoms)
    box = jnp.asarray(atoms.get_cell().lengths(), dtype=jnp.float32)

    if np.all(box < 1e-6):
        displacement_fn, _ = space.free()
        box = 100
    else:
        displacement_fn, _ = space.periodic_general(box, fractional_coordinates=False)

    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box,
        model_config.model.r_max,
        0.5,
        fractional_coordinates=False,
        format=partition.Sparse,
    )
    neighbors = neighbor_fn.allocate(R)
    idx = neighbors.idx

    ckpt_exists = look_for_checkpoints(ckpt_dir)
    assert ckpt_exists
    raw_restored = checkpoints.restore_checkpoint(ckpt_dir, target=None, step=None)
    params = jax.tree_map(jnp.asarray, raw_restored["model"]["params"])

    builder = ModelBuilder(model_config.model.get_dict(), n_species=n_species)
    model = builder.build_energy_force_model(
        displacement_fn=displacement_fn, apply_mask=False, init_box=np.zeros(3)
    )

    apply_fn = partial(model.apply, params, box=np.zeros(3))
    f_tf = jax2tf.convert(
        apply_fn,
        polymorphic_shapes=[f"({n_atoms},3)", f"({n_atoms},)", f"(2,{idx.shape[1]})"],
        enable_xla=True,
    )

    model_function = tf.function(f_tf, autograph=False)
    model_concrete = model_function.get_concrete_function(
        tf.TensorSpec([n_atoms, 3], tf.float64),
        tf.TensorSpec(
            [
                n_atoms,
            ],
            tf.int32,
        ),
        tf.TensorSpec([2, idx.shape[1]], tf.int32),
    )

    model_tf = tf.Module()
    model_tf.f = model_concrete

    tf.saved_model.save(model_tf, deployed_path)
