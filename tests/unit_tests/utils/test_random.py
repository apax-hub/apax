import random

import numpy as np
import tensorflow as tf

from apax.utils.random import seed_py_np_tf


def test_seed_py_np_tf():
    seed = 42

    # First run
    seed_py_np_tf(seed)
    py_rand1 = random.random()
    np_rand1 = np.random.rand()
    tf_rand1 = tf.random.uniform([1]).numpy()[0]

    # Second run with same seed
    seed_py_np_tf(seed)
    py_rand2 = random.random()
    np_rand2 = np.random.rand()
    tf_rand2 = tf.random.uniform([1]).numpy()[0]

    assert py_rand1 == py_rand2
    assert np_rand1 == np_rand2
    assert tf_rand1 == tf_rand2
