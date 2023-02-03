import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import time

import numpy as np
import tensorflow as tf

n = 5
min_atoms = 2
max_atoms = 15

positions = []
number_of_atoms = []
padded_number_of_atoms = []

for i in range(n):
    n_atoms = np.random.randint(min_atoms, max_atoms)
    number_of_atoms.append(n_atoms)

    # n_atoms = int(math.ceil(n_atoms / 10) * 10)  # pad to the next multiple of ten
    # padded_number_of_atoms.append(n_atoms)

    pos = np.random.normal(size=(n_atoms, 3))
    positions.append(pos)


max_atoms = max(number_of_atoms)
print("max_atoms", max_atoms)


n_sizes = len(set(number_of_atoms))
# n_sizes_padded = len(set(padded_number_of_atoms))
print("n_sizes", n_sizes)  # ~200
# print(n_sizes_padded)  # ~21

pos = tf.ragged.constant(positions)
padded_positions = tf.ragged.constant(positions)

data = {"pos": pos, "padded_pos": padded_positions}

ds = tf.data.Dataset.from_tensor_slices(data)

def pad_to_largest_element(data):
    # pads ragged tensor to regular tensor with size of the largest element in batch
    data = data.to_tensor()
    return data


class PadToMaxElement:
    def __init__(self, n_max) -> None:
        self.n_max = n_max
    def __call__(self, data):
        # pads ragged tensor to regular tensor with size of the largest element in batch
        for key, val in data.items():
            if key == "padded_pos":
                shape = data[key].shape
                data[key] = data[key].to_tensor(default_value=0.0, shape=[shape[0], self.n_max, shape[-1]])
            else:
                data[key] = data[key].to_tensor()
        return data


ds = ds.shuffle(32 * 3).batch(1).map(PadToMaxElement(20))#.map(pad_to_largest_element)

shapes = []

epochs = 2
start = time.time()
for epoch in range(epochs):
    for data in ds:
        # shapes.append(data.shape[1])
        print(data["pos"].shape)
        print(data["padded_pos"].shape)
        print()


# end = time.time()
# print(f"duration {(end-start) / epoch}")

# num_shapes = len(set(shapes))
# print(num_shapes)  # ~12
# worst case scenario: as many recompilations as there are molecules in the ds,
# that differ by a size of 10 atoms.
# Here it would be the same as n_sizes padded (~21)
