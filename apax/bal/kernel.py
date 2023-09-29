import einops
import numpy as np


class KernelMatrix:
    """
    Matrix representation of a kernel defined by a feature map g
    K_{ij} = \\sum_{k} g_{ik} g_{jk}
    """

    def __init__(self, g: np.ndarray, n_train: int):
        self.num_columns = g.shape[0]
        self.g = g
        self.diagonal = einops.einsum(g, g, "s feature, s feature -> s")
        self.n_train = n_train

    def compute_column(self, idx: int) -> np.ndarray:
        return einops.einsum(self.g, self.g[idx, :], "s feature, feature -> s")

    def score(self, idx: int) -> np.ndarray:
        """Computes the distance of sample i from all other samples j as
        K_{ii} + K_{jj} - 2 K_{ij}
        """
        return self.diagonal[idx] + self.diagonal - 2 * self.compute_column(idx)
