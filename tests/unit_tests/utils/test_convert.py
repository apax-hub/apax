import numpy as np
import jax.numpy as jnp
import pytest
from apax.utils.convert import prune_dict, str_to_dtype, transpose_dict_of_lists



def test_transpose_dict_of_lists():
    b = np.arange(8).reshape((4, 2))
    a = [0, 1, 2, 3]
    inputs = {
        "a": a,
        "b": b,
    }

    out = transpose_dict_of_lists(inputs)
    assert len(out) == len(a)

    for ii, entry in enumerate(out):
        assert "a" in entry.keys()
        assert "b" in entry.keys()
        assert entry["a"] == a[ii]
        assert np.all(entry["b"] == b[ii])


def test_str_to_dtype():
    assert str_to_dtype("fp32") == jnp.float32
    assert str_to_dtype("fp64") == jnp.float64
    with pytest.raises(KeyError):
        str_to_dtype("unknown")


def test_prune_dict():
    d = {"a": [1, 2], "b": [], "c": [3]}
    pruned_d = prune_dict(d)
    assert "a" in pruned_d
    assert "b" not in pruned_d
    assert "c" in pruned_d
