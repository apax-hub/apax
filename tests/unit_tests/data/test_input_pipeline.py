import numpy as np
import pytest
import tensorflow as tf

from apax.data.input_pipeline import PadToSpecificSize, TFPipeline, create_dict_dataset
from apax.train.run import find_largest_system
from apax.utils.convert import atoms_to_arrays
from apax.utils.data import split_atoms, split_idxs
from apax.utils.random import seed_py_np_tf
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from apax.layers.descriptor.gaussian_moment_descriptor import disp_fn
from jax import vmap


@pytest.mark.parametrize(
    "num_data, pbc, calc_results, external_labels",
    (
        [5, False, ["energy"], None],
        [5, False, ["energy", "forces"], None],
        [5, True, ["energy", "forces"], None],
        [
            5,
            True,
            ["energy", "forces"],
            {
                "fixed": {
                    "ma_tensors": np.random.uniform(low=-1.0, high=1.0, size=(5, 3, 3))
                }
            },
        ],
    ),
)
def test_input_pipeline(example_atoms, calc_results, num_data, external_labels):
    batch_size = 2
    r_max = 6.0

    inputs, labels = create_dict_dataset(
        example_atoms,
        r_max,
        external_labels,
        disable_pbar=True,
    )
    max_atoms, max_nbrs = find_largest_system([inputs])

    ds = TFPipeline(
        inputs,
        labels,
        1,
        batch_size,
        max_atoms=max_atoms,
        max_nbrs=max_nbrs,
        buffer_size=1000,
    )
    assert ds.steps_per_epoch() == num_data // batch_size

    ds = ds.shuffle_and_batch()

    sample_inputs, sample_labels = next(ds)

    assert "box" in sample_inputs
    assert len(sample_inputs["box"]) == batch_size
    assert len(sample_inputs["box"][0]) == 3

    assert "numbers" in sample_inputs
    for i in range(batch_size):
        assert len(sample_inputs["numbers"][i]) == max(sample_inputs["n_atoms"])

    assert "idx" in sample_inputs
    assert len(sample_inputs["idx"][0]) == len(sample_inputs["idx"][1])

    assert "positions" in sample_inputs
    assert len(sample_inputs["positions"][0][0]) == 3
    for i in range(batch_size):
        assert len(sample_inputs["positions"][i]) == max(sample_inputs["n_atoms"])

    assert "n_atoms" in sample_inputs
    assert len(sample_inputs["n_atoms"]) == batch_size

    assert "energy" in sample_labels
    assert len(sample_labels["energy"]) == batch_size

    if "forces" in calc_results:
        assert "forces" in sample_labels
        assert len(sample_labels["forces"][0][0]) == 3
        for i in range(batch_size):
            assert len(sample_labels["forces"][i]) == max(sample_inputs["n_atoms"])

    if external_labels:
        assert "ma_tensors" in sample_labels
        assert len(sample_labels["ma_tensors"]) == batch_size

    sample_inputs2, _ = next(ds)
    assert (sample_inputs["positions"][0][0] != sample_inputs2["positions"][0][0]).all()


def test_pad_to_specific_size():
    idx_1 = [[1, 4, 3], [3, 1, 4]]
    idx_2 = [[5, 4, 2, 3, 1], [1, 2, 3, 4, 5]]
    r_inp = {"idx": tf.ragged.constant([idx_1, idx_2])}
    p_inp = {"n_atoms": tf.constant([3, 5])}
    f_1 = [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    f_2 = [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]
    r_lab = {"forces": tf.ragged.constant([f_1, f_2])}
    p_lab = {"energy": tf.constant([103.3, 98.4])}

    max_atoms = 5
    max_nbrs = 6

    padding_fn = PadToSpecificSize(max_atoms=max_atoms, max_nbrs=max_nbrs)

    inputs, labels = padding_fn(r_inp, p_inp, r_lab, p_lab)

    assert "idx" in inputs
    assert inputs["idx"].shape == [2, 2, 6]

    assert "n_atoms" in inputs

    assert "forces" in labels
    assert labels["forces"].shape == [2, 5, 3]

    assert "energy" in labels


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
    inputs, labels = atoms_to_arrays(example_atoms)

    assert "fixed" in inputs
    assert "ragged" in inputs
    assert "fixed" or "ragged" in labels

    assert "positions" in inputs["ragged"]
    assert len(inputs["ragged"]["positions"]) == len(example_atoms)

    assert "numbers" in inputs["ragged"]
    assert len(inputs["ragged"]["numbers"]) == len(example_atoms)

    assert "box" in inputs["fixed"]
    assert len(inputs["fixed"]["box"]) == len(example_atoms)
    if not pbc:
        assert np.all(inputs["fixed"]["box"][0] < 1e-6)

    assert "n_atoms" in inputs["fixed"]
    assert len(inputs["fixed"]["n_atoms"]) == len(example_atoms)

    assert "energy" in labels["fixed"]
    assert len(labels["fixed"]["energy"]) == len(example_atoms)

    assert "forces" in labels["ragged"]
    assert len(labels["ragged"]["forces"]) == len(example_atoms)


# TODO auf distanzen testen da nurnoch ase neighbor verwendet werden
@pytest.mark.parametrize(
    "num_data, pbc, calc_results, external_labels",
    ([3, True, ["energy"], None],),
)
def test_neighbors_and_displacements(pbc, calc_results, num_data, external_labels):
    r_max = 2.0

    numbers = np.array([1, 1])
    positions = np.array([[0.0, 0.0, 0.0], [1., 0.0, 1.0]])

    additional_data = {}
    additional_data["pbc"] = pbc
    additional_data["cell"] = np.array([[1.8, 0.1, 0.], [0.0, 2.5, 0.1], [0.1, 0.0, 2.5]])

    result_shapes = {
        "energy": (np.random.rand() - 5.0) * 10_000,
    }

    atoms = Atoms(numbers=numbers, positions=positions, **additional_data)
    if calc_results:
        results = {}
        for key in calc_results:
            results[key] = result_shapes[key]
        atoms.calc = SinglePointCalculator(atoms, **results)

    inputs, _ = create_dict_dataset(
        [atoms],
        r_max,
        external_labels,
        disable_pbar=True,
    )

    idx = np.asarray(inputs["ragged"]["idx"])[0]
    offsets =  np.asarray(inputs["ragged"]["offsets"][0])
    box =  np.asarray(inputs["fixed"]["box"][0])

    Ri = positions[idx[0]]
    Rj = positions[idx[1]] + offsets
    matscipy_dr_vec = Ri - Rj
    matscipy_dr_vec = np.asarray(matscipy_dr_vec)

    positions =  np.asarray(inputs["ragged"]["positions"][0])
    Ri = positions[idx[0]]
    Rj = positions[idx[1]]
    displacement = vmap(disp_fn, (0, 0, None, None), 0)
    apax_dr_vec = displacement(Ri, Rj, None, box)
    apax_dr_vec -= offsets
    apax_dr_vec = np.asarray(apax_dr_vec)

    matscipy_dist = np.linalg.norm(matscipy_dr_vec, axis=1)
    apax_dist = np.linalg.norm(apax_dr_vec, axis=1)

    print(matscipy_dist)
    print(apax_dist)

    assert np.all(matscipy_dr_vec - apax_dr_vec < 10e-8)
    assert np.all(matscipy_dist - apax_dist < 10e-8)