import numpy as np

from apax.bal.kernel import KernelMatrix


def max_dist_selection(matrix: KernelMatrix, batch_size: int):
    """
    \\argmax_{S \\in \\mathbb{X}_{rem}} \\min_{S' \\in \\mathbb{X}_{sel} } d(S, S')
    """
    n_train = matrix.n_train

    min_squared_distances = matrix.diagonal
    min_squared_distances[:n_train] = -np.inf

    # Use max norm for first point
    new_idx = np.argmax(min_squared_distances)
    selected_idxs = list(range(n_train)) + [new_idx]

    for _ in range(1, batch_size):
        squared_distances = matrix.score(new_idx)

        squared_distances[selected_idxs] = -np.inf
        min_squared_distances = np.minimum(min_squared_distances, squared_distances)

        new_idx = np.argmax(min_squared_distances)
        selected_idxs.append(new_idx)

    return (
        np.array(selected_idxs[n_train:]) - n_train
    )  # shift by number of train datapoints
