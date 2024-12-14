import logging
import typing
from pathlib import Path

import ase.io
import numpy as np
import zntrack.utils
from matplotlib import pyplot as plt

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

from apax.bal import kernel_selection
from apax.nodes.model import ApaxBase

log = logging.getLogger(__name__)


def plot_with_uncertainty(value, ylabel: str, xlabel: str, x=None, **kwargs) -> tuple:
    """Parameters
    ----------
    value: data of shape (n, m) where n is the number of ensembles.
    x: optional x values of shape (m,)

    Returns
    -------

    """
    if isinstance(value, dict):
        data = value
    else:
        data = {
            "mean": np.mean(value, axis=0),
            "std": np.std(value, axis=0),
            "max": np.max(value, axis=0),
            "min": np.min(value, axis=0),
        }

    fig, ax = plt.subplots(**kwargs)
    if x is None:
        x = np.arange(len(data["mean"]))
    ax.fill_between(
        x,
        data["mean"] + data["std"],
        data["mean"] - data["std"],
        facecolor="lightblue",
    )
    if "max" in data:
        ax.plot(x, data["max"], linestyle="--", color="darkcyan")
    if "min" in data:
        ax.plot(x, data["min"], linestyle="--", color="darkcyan")
    ax.plot(x, data["mean"], color="black")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return fig, ax, data


class BatchKernelSelection(zntrack.Node):
    """Interface to the batch active learning methods implemented in apax.
    Check the apax documentation for a list and explanation of implemented properties.

    Attributes
    ----------
    models: Union[Apax, List[Apax]]
        One or more Apax models to construct a feature map from.
    base_feature_map: dict
        Name and parameters for the feature map transformation.
    selection_method: str
        Name of the selection method to be used. Choose from:
        ["max_dist", ]
    n_configurations: int
        Number of samples to be selected.
    min_distance_threshold: float
        Minimal allowed distance between selected points before selection stops.
        Selection stops either if n samples were selected or if the distance is smaller
        than specified by this criterion.
        Default is 0 so this criterion will not apply.
    rank_all: bool
        If true, ranks all pool samples. Great for visualization but might slow things
        down if the pool is really large. If false, only the most informative
        n_configurations are ranked.
    processing_batch_size: int
        Number of samples to be processed in parallel.
        Does not affect the result, just the speed of computing features.
    """

    data: list[ase.Atoms] = zntrack.deps()
    train_data: list[ase.Atoms] = zntrack.deps()

    selected_ids: list[int] = zntrack.outs(independent=True)

    models: typing.List[ApaxBase] = zntrack.deps()
    base_feature_map: dict = zntrack.params(
        default_factory=lambda: {"name": "ll_grad", "layer_name": "dense_2"}
    )
    selection_method: str = zntrack.params("max_dist")
    n_configurations: int = zntrack.params()
    min_distance_threshold: int = zntrack.params(0.0)
    processing_batch_size: int = zntrack.params(64)
    rank_all: bool = zntrack.params(True)

    img_selection: Path = zntrack.outs_path(zntrack.nwd / "selection.png")
    img_distances: Path = zntrack.outs_path(zntrack.nwd / "distances.png")
    img_features: Path = zntrack.outs_path(zntrack.nwd / "features.png")

    def get_data(self) -> list[ase.Atoms]:
        """Get the atoms data to process."""
        if self.data is not None:
            return self.data
        else:
            raise ValueError("No data given.")

    def run(self):
        """ZnTrack Node Run method."""

        log.debug(f"Selecting from {len(self.data)} configurations.")
        self.selected_ids = self.select_atoms(self.data)

    @property
    def frames(self) -> list[ase.Atoms]:
        """Get a list of the selected atoms objects."""
        return [atoms for i, atoms in enumerate(self.data) if i in self.selected_ids]

    @property
    def excluded_frames(self) -> list[ase.Atoms]:
        """Get a list of the atoms objects that were not selected."""
        return [atoms for i, atoms in enumerate(self.data) if i not in self.selected_ids]

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        if isinstance(self.models, list):
            param_files = [m.parameter["data"]["directory"] for m in self.models]
        else:
            param_files = self.models.parameter["data"]["directory"]

        if self.rank_all:
            selection_batch_size = None
        else:
            selection_batch_size = self.n_configurations

        ranking, distances, features_train, features_pool = kernel_selection(
            param_files,
            self.train_data,
            atoms_lst,
            self.base_feature_map,
            self.selection_method,
            selection_batch_size=selection_batch_size,
            processing_batch_size=self.processing_batch_size,
        )

        features_pool = features_pool[ranking]
        distance_mask = distances > self.min_distance_threshold

        mask = distance_mask
        if self.rank_all:
            # if we rank all, we need to slice the indices to only
            # select the first n
            numbers = np.arange(len(distances))
            number_mask = numbers < self.n_configurations
            mask = mask & number_mask

        ranking = ranking[mask]
        features_selection = features_pool[mask]
        features_remaining = features_pool[~mask]

        false_indices = np.where(~mask)[0]
        if len(false_indices) > 0:
            last_selected = false_indices[0] - 1
        else:
            last_selected = -1

        self._get_distances_plot(distances, last_selected)
        self._get_pca_plot(features_train, features_selection, features_remaining)
        self._get_selection_plot(atoms_lst, ranking)
        return [int(x) for x in ranking]

    def _get_selection_plot(
        self, atoms_lst: typing.List[ase.Atoms], indices: typing.List[int]
    ):
        energies = np.array([atoms.calc.results["energy"] for atoms in atoms_lst])

        if "energy_uncertainty" in atoms_lst[0].calc.results.keys():
            uncertainty = np.array(
                [atoms.calc.results["energy_uncertainty"] for atoms in atoms_lst]
            )
            fig, ax, _ = plot_with_uncertainty(
                {"mean": energies, "std": uncertainty},
                ylabel="energy",
                xlabel="configuration",
            )
        else:
            fig, ax = plt.subplots()
            ax.plot(energies, label="energy")
            ax.set_ylabel("energy")
            ax.set_xlabel("configuration")

        ax.plot(indices, energies[indices], "x", color="red")

        fig.savefig(self.img_selection, bbox_inches="tight", dpi=240)

    def _get_distances_plot(self, distances: np.ndarray, last_selected: int):
        fig, ax = plt.subplots()
        ax.semilogy(distances, label="sq. distances")

        ax.axvline(last_selected, ls="--", color="gray", label="last selected")

        ax.set_ylabel("Squared distance")
        ax.set_xlabel("Configuration")
        ax.legend()
        fig.savefig(self.img_distances, bbox_inches="tight", dpi=240)

    def _get_pca_plot(
        self, g_train: np.ndarray, g_selection: np.ndarray, g_remaining: np.ndarray
    ):
        if PCA:
            all_features = [g_train, g_selection]
            if len(g_remaining) > 0:
                all_features.append(g_remaining)
            g_full = np.concatenate(all_features, axis=0)
            pca = PCA(n_components=2)
            pca.fit(g_full)

            g_train_2d = pca.transform(g_train)
            g_selection_2d = pca.transform(g_selection)
            if len(g_remaining) > 0:
                g_pool_2d = pca.transform(g_remaining)
        else:
            F = g_train.shape[1]
            W = np.random.randn(F, 2) / np.sqrt(F)
            g_train_2d = g_train @ W
            g_selection_2d = g_selection @ W
            if len(g_remaining) > 0:
                g_pool_2d = g_remaining @ W

        fig, ax = plt.subplots()

        if len(g_remaining) > 0:
            ax.scatter(
                *g_pool_2d.T,
                color="gray",
                marker="o",
                s=5,
                alpha=0.6,
                label="Remaining data",
            )

        ax.scatter(
            *g_train_2d.T, color="C0", marker="^", s=5, alpha=0.6, label="Train data"
        )
        ax.scatter(
            *g_selection_2d.T,
            color="red",
            marker="*",
            s=5,
            alpha=0.6,
            label="Selected data",
        )

        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.legend()

        fig.savefig(self.img_features, bbox_inches="tight", dpi=240)
