import numpy as np

from apax.train.eval import get_test_idxs


def test_get_test_idxs():
    atoms_list = list(range(10))
    used_idxs = np.array([1, 3, 5, 7, 9])
    test_idxs = get_test_idxs(atoms_list=atoms_list, used_idxs=used_idxs, n_test=2)
    assert len(test_idxs) == 2
    assert not any(item in used_idxs for item in test_idxs)

    test_idxs = get_test_idxs(atoms_list=atoms_list, used_idxs=used_idxs, n_test=-1)
    assert len(test_idxs) == 5
    assert not any(item in used_idxs for item in test_idxs)

    test_idxs = np.append(test_idxs, 7)
    assert any(item in used_idxs for item in test_idxs)
