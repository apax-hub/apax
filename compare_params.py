from apax.train.checkpoints import restore_parameters
from jax import tree_util
from flax.traverse_util import flatten_dict, unflatten_dict
import numpy as np
c1, p1 = restore_parameters("test_tl/test0")
c2, p2 = restore_parameters("test_tl/test1")

flat_before = flatten_dict(p1, sep="/")
flat_after = flatten_dict(p2, sep="/")

layer = "dense_0"

for path, p_before in flat_before.items():
    p_after = flat_after[path]
    same = np.allclose(p_before, p_after)
    should_be_same = layer in path

    if should_be_same and not same:
        raise ValueError("parameters should be the unchanged but deviation was found")
    elif not should_be_same and same:
        raise ValueError("parameters should not be the unchanged but are equal")

    # print(path)

# print(p1)
# print(tree_util.tree_map(lambda x: (x.shape), p1))
