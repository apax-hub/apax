import jax.numpy as jnp
import tensorflow as tf
from ase.io import read
from jax_md import partition, space

export_dir = "deployed_model"

model = tf.saved_model.load(export_dir, tags=None, options=None)

atoms = read("etoh.traj")

displacement_fn, _ = space.free()
neighbor_fn = partition.neighbor_list(
    displacement_fn,
    100,
    6.0,
    0.5,
    fractional_coordinates=False,
    format=partition.Sparse,
)
R_jnp = jnp.asarray(atoms.positions)
neighbors = neighbor_fn.allocate(R_jnp)
idx = neighbors.idx


R_tf = tf.convert_to_tensor(atoms.positions)
Z = tf.convert_to_tensor(atoms.numbers, dtype=tf.int32)
idx = tf.convert_to_tensor(idx)

results = model.f(R_tf, Z, idx)
print(results)
print(model.f)
