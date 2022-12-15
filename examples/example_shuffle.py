import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from gmnn_jax.data.input_pipeline import InputPipeline

num_data = 9
atoms_list = []
batch_size = 2
n_epoch = 11
for _ in range(num_data):
    num_atoms = np.random.randint(4, 6)
    numbers = np.random.randint(1, 119, size=num_atoms)
    cell_const = np.random.uniform(low=10.0, high=12.0)
    positions = np.random.uniform(low=0.0, high=cell_const, size=(num_atoms, 3))

    additional_data = {}

    result_shapes = {
        "energy": (np.random.rand() - 5.0) * 10_000,
        "forces": np.random.uniform(low=-1.0, high=1.0, size=(num_atoms, 3)),
    }

    atoms = Atoms(numbers=numbers, positions=positions, **additional_data)
    results = {}
    for key in ["energy", "forces"]:
        results[key] = result_shapes[key]

    atoms.calc = SinglePointCalculator(atoms, **results)
    atoms_list.append(atoms)

ds = InputPipeline(cutoff=6.0, batch_size=batch_size, atoms_list=atoms_list, n_epoch=n_epoch)
batch_ds = ds.shuffle_and_batch()

step_per_epoch = ds.steps_per_epoch()
to_drop = num_data - (step_per_epoch * batch_size)

for epoch in range(n_epoch):
    epoch += 1
    for batch_idx in range(step_per_epoch):
        data = next(batch_ds)
        inputs, labels = data
        print(inputs["numbers"])
    print(epoch)
