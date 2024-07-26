import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from jax import vmap

from apax.data.preprocessing import compute_nl
from apax.layers.distances import disp_fn
from apax.utils.convert import atoms_to_inputs, atoms_to_labels
from apax.utils.data import split_atoms, split_idxs
from apax.utils.random import seed_py_np_tf

# TODO RE-ENABLE LATER
# @pytest.mark.parametrize(
#     "num_data, pbc, calc_results, external_labels",
#     (
#         [5, False, ["energy"], None],
#         [5, False, ["energy", "forces"], None],
#         [5, True, ["energy", "forces"], None],
#         [
#             5,
#             True,
#             ["energy", "forces"],
#             [{
#                 "name": "ma_tensors",
#                 "values": np.random.uniform(low=-1.0, high=1.0, size=(5, 3, 3)),
#             }],
#         ],
#     ),
# )
# def test_input_pipeline(example_atoms, calc_results, num_data, external_labels):
#     batch_size = 2
#     r_max = 6.0

#     if external_labels:
#         for label in external_labels:
#             for a, v in zip(example_atoms, label["values"]):
#                 a.calc.results[label["name"]] = v

#     ds = InMemoryDataset(
#         example_atoms,
#         r_max,
#         batch_size,
#         1,
#         buffer_size=1000,
#     )
#     assert ds.steps_per_epoch() == num_data // batch_size

#     ds = ds.shuffle_and_batch()

#     sample_inputs, sample_labels = next(ds)

#     assert "box" in sample_inputs
#     assert len(sample_inputs["box"]) == batch_size
#     assert len(sample_inputs["box"][0]) == 3

#     assert "numbers" in sample_inputs
#     for i in range(batch_size):
#         assert len(sample_inputs["numbers"][i]) == max(sample_inputs["n_atoms"])

#     assert "idx" in sample_inputs
#     assert len(sample_inputs["idx"][0]) == len(sample_inputs["idx"][1])

#     assert "positions" in sample_inputs
#     assert len(sample_inputs["positions"][0][0]) == 3
#     for i in range(batch_size):
#         assert len(sample_inputs["positions"][i]) == max(sample_inputs["n_atoms"])

#     assert "n_atoms" in sample_inputs
#     assert len(sample_inputs["n_atoms"]) == batch_size

#     assert "energy" in sample_labels
#     assert len(sample_labels["energy"]) == batch_size

#     if "forces" in calc_results:
#         assert "forces" in sample_labels
#         assert len(sample_labels["forces"][0][0]) == 3
#         for i in range(batch_size):
#             assert len(sample_labels["forces"][i]) == max(sample_inputs["n_atoms"])

#     if external_labels:
#         assert "ma_tensors" in sample_labels
#         assert len(sample_labels["ma_tensors"]) == batch_size

#     sample_inputs2, _ = next(ds)
#     assert (sample_inputs["positions"][0][0] != sample_inputs2["positions"][0][0]).all()


@pytest.mark.parametrize(
    "num_data, pbc, calc_results",
    (
        [10, False, ["energy", "forces"]],
        [10, True, ["energy", "forces"]],
    ),
)
def test_split_data(example_atoms):
    seed_py_np_tf(1)
    train_idxs1, val_idxs1 = split_idxs(example_atoms, 4, 2)
    train_idxs2, val_idxs2 = split_idxs(example_atoms, 4, 2)
    assert np.all(train_idxs1 != train_idxs2) and np.all(val_idxs1 != val_idxs2)

    train_atoms1, val_atoms1 = split_atoms(example_atoms, train_idxs1, val_idxs1)
    train_atoms2, val_atoms2 = split_atoms(example_atoms, train_idxs2, val_idxs2)
    assert np.all(train_atoms1[0].get_positions() != train_atoms2[0].get_positions())
    assert np.all(val_atoms1[0].get_positions() != val_atoms2[0].get_positions())

    seed_py_np_tf(1)
    train_idxs2, val_idxs2 = split_idxs(example_atoms, 4, 2)
    assert np.all(train_idxs1 == train_idxs2) and np.all(val_idxs1 == val_idxs2)

    train_atoms2, val_atoms2 = split_atoms(example_atoms, train_idxs2, val_idxs2)
    assert np.all(train_atoms1[0].get_positions() == train_atoms2[0].get_positions())
    assert np.all(val_atoms1[0].get_positions() == val_atoms2[0].get_positions())


@pytest.mark.parametrize(
    "num_data, pbc, calc_results",
    (
        [5, False, ["energy", "forces"]],
        [5, True, ["energy", "forces"]],
    ),
)
def test_convert_atoms_to_arrays(example_atoms, pbc):
    inputs = atoms_to_inputs(example_atoms)
    labels = atoms_to_labels(example_atoms)

    assert "positions" in inputs
    assert len(inputs["positions"]) == len(example_atoms)

    assert "numbers" in inputs
    assert len(inputs["numbers"]) == len(example_atoms)

    assert "box" in inputs
    assert len(inputs["box"]) == len(example_atoms)
    if not pbc:
        assert np.all(inputs["box"][0] < 1e-6)

    assert "n_atoms" in inputs
    assert len(inputs["n_atoms"]) == len(example_atoms)

    assert "energy" in labels
    assert len(labels["energy"]) == len(example_atoms)

    assert "forces" in labels
    assert len(labels["forces"]) == len(example_atoms)


@pytest.mark.parametrize(
    "pbc, calc_results, cell",
    (
        [
            True,
            ["energy"],
            np.array([[1.8, 0.1, 0.0], [0.0, 2.5, 0.1], [0.1, 0.0, 2.5]]),
        ],
        [
            True,
            ["energy"],
            np.array([[1.8, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 2.5]]),
        ],
        [
            True,
            ["energy"],
            np.array([[1.5, 0.0, 0.5], [0.0, 2.5, 0.0], [0.0, 0.5, 1.5]]),
        ],
    ),
)
def test_neighbors_and_displacements(pbc, calc_results, cell):
    r_max = 2.0

    numbers = np.array([1, 1])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    additional_data = {}
    additional_data["pbc"] = pbc
    additional_data["cell"] = cell

    result_shapes = {
        "energy": (np.random.rand() - 5.0) * 10_000,
    }

    atoms = Atoms(numbers=numbers, positions=positions, **additional_data)
    if calc_results:
        results = {}
        for key in calc_results:
            results[key] = result_shapes[key]
        atoms.calc = SinglePointCalculator(atoms, **results)

    inputs = atoms_to_inputs([atoms])
    box = np.asarray(inputs["box"][0])
    idx, offsets = compute_nl(inputs["positions"][0], box, r_max)

    Ri = positions[idx[0]]
    Rj = positions[idx[1]] + offsets
    matscipy_dr_vec = Rj - Ri
    matscipy_dr_vec = np.asarray(matscipy_dr_vec)

    positions = np.asarray(inputs["positions"][0])
    Ri = positions[idx[0]]
    Rj = positions[idx[1]]
    displacement = vmap(disp_fn, (0, 0, None, None), 0)
    apax_dr_vec = displacement(Rj, Ri, None, box)
    apax_dr_vec += offsets
    apax_dr_vec = np.asarray(apax_dr_vec)

    matscipy_dist = np.linalg.norm(matscipy_dr_vec, axis=1)
    apax_dist = np.linalg.norm(apax_dr_vec, axis=1)

    assert np.all(matscipy_dr_vec - apax_dr_vec < 10e-7)
    assert np.all(matscipy_dist - apax_dist < 10e-7)
