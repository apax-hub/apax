import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
import tensorflow as tf
import numpy as np
import time

n = 1000
min_atoms = 100
max_atoms = 300

positions = []
number_of_atoms = []
padded_number_of_atoms = []

for i in range(n):
    n_atoms = np.random.randint(min_atoms, max_atoms)
    number_of_atoms.append(n_atoms)

    n_atoms = int(math.ceil(n_atoms / 10) * 10) # pad to the next multiple of ten
    padded_number_of_atoms.append(n_atoms)

    pos = np.random.normal(size=(n_atoms,3))
    positions.append(pos)


n_sizes = len(set(number_of_atoms))
n_sizes_padded = len(set(padded_number_of_atoms))
print(n_sizes) # ~200
print(n_sizes_padded) # ~21

positions = tf.ragged.constant(positions)
ds = tf.data.Dataset.from_tensor_slices(positions)

def pad_to_largest_element(data):
    # pads ragged tensor to regular tensor with size of the largest element in batch
    data = data.to_tensor()
    return data

ds = ds.shuffle(32*3).batch(32).map(pad_to_largest_element)

shapes = []

epochs = 1000
start = time.time()
for epoch in range(epochs):
    for data in ds:
        shapes.append(data.shape[1])


end = time.time()
print(f"duration {(end-start) / epoch}")

num_shapes = len(set(shapes))
print(num_shapes) # ~12
# worst case scenario: as many recompilations as there are molecules in the ds,
# that differ by a size of 10 atoms.
# Here it would be the same as n_sizes padded (~21)