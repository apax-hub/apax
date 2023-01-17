import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from ase.io import read
from jax_md import partition, space

from gmnn_jax.data.preprocessing import dataset_neighborlist
from gmnn_jax.utils.convert import convert_atoms_to_arrays

atoms_list = read("raw_data/buoh.traj", index=":")

inputs, labels = convert_atoms_to_arrays(atoms_list)

# TODO box size and space type are data dependent,
# they need to be recomputed in the ds_nl function
nl_format = partition.Sparse
displacement_fn, _ = space.free()
neighbor_fn = partition.neighbor_list(displacement_fn, 100.0, 6.0, format=nl_format)

idx = dataset_neighborlist(neighbor_fn, inputs["positions"])

inputs["idx"] = idx
ds = tf.data.Dataset.from_tensor_slices((inputs, labels))

sample_inputs, _ = next(ds.take(1).as_numpy_iterator())

ds = ds.shuffle(32 * 4).batch(32)
