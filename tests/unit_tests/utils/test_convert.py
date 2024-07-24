import numpy as np

from apax.utils.convert import transpose_dict_of_lists


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
