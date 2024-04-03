import ase
import matplotlib.pyplot as plt
import numpy as np
import znflow
import zntrack

from apax.bal import kernel_selection

from .utils import get_flat_data_from_dict, plot_with_uncertainty


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
    processing_batch_size: int
        Number of samples to be processed in parallel.
        Does not affect the result, just the speed of computing features.
    """

    models: list = zntrack.deps()
    base_feature_map: dict = zntrack.params({"name": "ll_grad", "layer_name": "dense_2"})
    selection_method: str = zntrack.params("max_dist")
    n_configurations: str = zntrack.params()
    processing_batch_size: str = zntrack.meta.Text(64)
    img_selection = zntrack.outs_path(zntrack.nwd / "selection.png")

    train_data: list[ase.Atoms] = zntrack.deps()

    def _post_init_(self):
        if self.train_data is not None and not isinstance(self.train_data, dict):
            try:
                self.train_data = znflow.combine(
                    self.train_data, attribute="atoms", return_dict_attr="name"
                )
            except TypeError:
                self.train_data = znflow.combine(self.train_data, attribute="atoms")

        if self.data is not None and not isinstance(self.data, dict):
            try:
                self.data = znflow.combine(
                    self.data, attribute="atoms", return_dict_attr="name"
                )
            except TypeError:
                self.data = znflow.combine(self.data, attribute="atoms")

    def select_atoms(self, atoms_lst: list[ase.Atoms]) -> list[int]:
        if isinstance(self.models, list):
            param_files = [m._parameter["data"]["directory"] for m in self.models]
        else:
            param_files = self.models._parameter["data"]["directory"]

        if isinstance(self.train_data, dict):
            self.train_data = get_flat_data_from_dict(self.train_data)

        selected = kernel_selection(
            param_files,
            self.train_data,
            atoms_lst,
            self.base_feature_map,
            self.selection_method,
            selection_batch_size=self.n_configurations,
            processing_batch_size=self.processing_batch_size,
        )
        self._get_plot(atoms_lst, selected)

        return list(selected)

    def _get_plot(self, atoms_lst: list[ase.Atoms], indices: list[int]):
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

        fig.savefig(self.img_selection, bbox_inches="tight")
