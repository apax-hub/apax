import numpy as np
import pytest
import tensorflow as tf

from gmnn_jax.data.input_pipeline import (
    PadToSpecificSize,
    TFPipeline,
    create_dict_dataset,
    initialize_nbr_displacement_fns,
)
from gmnn_jax.train.run import find_largest_system
from gmnn_jax.utils.data import split_atoms
from gmnn_jax.utils.random import seed_py_np_tf


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
def test_input_pipeline(example_atoms, pbc, calc_results, num_data, external_labels):
    batch_size = 2
    r_max = 6.0

    _, neighbor_fn = initialize_nbr_displacement_fns(example_atoms[0], r_max)

    inputs, labels = create_dict_dataset(
        example_atoms,
        neighbor_fn,
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

    if pbc:
        assert "cell" in sample_inputs
        assert len(sample_inputs["cell"]) == batch_size
        assert len(sample_inputs["cell"][0]) == 3
    else:
        assert "cell" not in sample_inputs

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
    train_atoms1, val_atoms1, train_idxs1, val_idxs1 = split_atoms(example_atoms, 4, 2)
    train_atoms2, val_atoms2, train_idxs2, val_idxs2 = split_atoms(example_atoms, 4, 2)
    assert np.all(train_idxs1 != train_idxs2) and np.all(val_idxs1 != val_idxs2)
    assert np.all(train_atoms1[0].get_positions() != train_atoms2[0].get_positions())
    assert np.all(val_atoms1[0].get_positions() != val_atoms2[0].get_positions())

    seed_py_np_tf(1)
    train_atoms2, val_atoms2, train_idxs2, val_idxs2 = split_atoms(example_atoms, 4, 2)
    assert np.all(train_idxs1 == train_idxs2) and np.all(val_idxs1 == val_idxs2)
    assert np.all(train_atoms1[0].get_positions() == train_atoms2[0].get_positions())
    assert np.all(val_atoms1[0].get_positions() == val_atoms2[0].get_positions())
