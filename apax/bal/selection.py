from typing import Optional

import numpy as np

from apax.bal.kernel import KernelMatrix


def max_dist_selection(matrix: KernelMatrix, batch_size: Optional[int] = None):
    """
    Iteratively selects samples from the pool which are
    most distant from all previously selected samples.
    \\argmax_{S \\in \\mathbb{X}_{rem}} \\min_{S' \\in \\mathbb{X}_{sel} } d(S, S')

    https://arxiv.org/pdf/2203.09410.pdf
    https://doi.org/10.1039/D2DD00034B

    Attributes
    ----------
    matrix: KernelMatrix
        Kernel used to compare structures.
    batch_size: int
        Number of new data points to be selected.
    """
    n_train = matrix.n_train

    min_squared_distances = matrix.diagonal
    min_squared_distances[:n_train] = -np.inf

    n_pool = len(min_squared_distances[n_train:])
    end = n_pool

    if batch_size:
        end = batch_size

    # Use max norm for first point
    new_idx = np.argmax(min_squared_distances)
    selected_idxs = list(range(n_train)) + [new_idx]
    distances = [np.max(min_squared_distances)]

    for _ in range(1, end):
        squared_distances = matrix.score(new_idx)

        squared_distances[selected_idxs] = -np.inf
        min_squared_distances = np.minimum(min_squared_distances, squared_distances)

        new_idx = np.argmax(min_squared_distances)
        max_dist = np.max(min_squared_distances)
        selected_idxs.append(new_idx)
        distances.append(max_dist)

    # shift by number of train datapoints
    selected_idxs = np.array(selected_idxs[n_train:]) - n_train
    distances = np.array(distances)
    return selected_idxs, distances
