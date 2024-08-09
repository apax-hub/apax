import logging
import typing

import ase.io
import numpy as np
import zntrack.utils
from ipsuite.analysis.ensemble import plot_with_uncertainty
from ipsuite.configuration_selection.base import BatchConfigurationSelection
from matplotlib import pyplot as plt

from apax.bal import kernel_selection
from apax.nodes.model import ApaxBase

log = logging.getLogger(__name__)


class BatchKernelSelection(BatchConfigurationSelection):
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

    _module_ = "apax.nodes"

    models: typing.List[ApaxBase] = zntrack.deps()
    base_feature_map: dict = zntrack.params({"name": "ll_grad", "layer_name": "dense_2"})
    selection_method: str = zntrack.params("max_dist")
    n_configurations: str = zntrack.params()
    processing_batch_size: str = zntrack.meta.Text(64)
    img_selection = zntrack.outs_path(zntrack.nwd / "selection.png")

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        if isinstance(self.models, list):
            param_files = [m._parameter["data"]["directory"] for m in self.models]
        else:
            param_files = self.models._parameter["data"]["directory"]

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

    def _get_plot(self, atoms_lst: typing.List[ase.Atoms], indices: typing.List[int]):
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
