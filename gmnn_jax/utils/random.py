import random

import numpy as np
import tensorflow as tf


def seed_py_np_tf(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
