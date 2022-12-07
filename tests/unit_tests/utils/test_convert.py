import pytest

from gmnn_jax.utils.convert import convert_atoms_to_arrays


@pytest.mark.parametrize(
    "num_data, pbc, calc_results",
    (
        [5, False, ["energy", "forces"]],
        [5, True, ["energy", "forces"]],
    ),
)
def test_convert_atoms_to_arrays(example_atoms, pbc):
    inputs, labels = convert_atoms_to_arrays(example_atoms)

    print(inputs, labels)

    assert "fixed" in inputs
    assert "ragged" in inputs
    assert "fixed" or "ragged" in labels

    assert "positions" in inputs["ragged"]
    assert len(inputs["ragged"]["positions"]) == len(example_atoms)

    assert "numbers" in inputs["ragged"]
    assert len(inputs["ragged"]["numbers"]) == len(example_atoms)

    if pbc:
        assert "cell" in inputs["fixed"]
        assert len(inputs["fixed"]["cell"]) == len(example_atoms)
    else:
        assert "cell" not in inputs["fixed"]

    assert "n_atoms" in inputs["fixed"]
    assert len(inputs["fixed"]["n_atoms"]) == len(example_atoms)

    assert "energy" in labels["fixed"]
    assert len(labels["fixed"]["energy"]) == len(example_atoms)

    assert "forces" in labels["ragged"]
    assert len(labels["ragged"]["forces"]) == len(example_atoms)
