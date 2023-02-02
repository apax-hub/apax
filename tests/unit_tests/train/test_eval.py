# TODO will be fixed in other branch by Nico
# import numpy as np

# from gmnn_jax.train.eval import get_test_idxs


# def test_get_test_idxs():
#     atoms_list = list(range(10))
#     used_idxs = np.array([1, 3, 5, 7, 9])
#     n_test = 2
#     test_idxs = get_test_idxs(atoms_list=atoms_list, used_idxs=used_idxs, n_test=n_test)
#     assert len(test_idxs) == 2
#     assert not any(item in used_idxs for item in test_idxs)

#     n_test = -1
#     test_idxs = get_test_idxs(atoms_list=atoms_list, used_idxs=used_idxs, n_test=n_test)
#     assert len(test_idxs) == 5
#     assert not any(item in used_idxs for item in test_idxs)

#     test_idxs = np.append(test_idxs, 7)
#     assert any(item in used_idxs for item in test_idxs)
