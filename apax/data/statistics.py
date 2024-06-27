import dataclasses
import logging

import numpy as np

log = logging.getLogger(__name__)


@dataclasses.dataclass
class DatasetStats:
    elemental_shift: np.array = None
    elemental_scale: float = None
    n_species: int = 119


class PerElementRegressionShift:
    name = "per_element_regression_shift"
    parameters = ["energy_regularisation"]
    dtypes = [float]

    @staticmethod
    def compute(inputs, labels, shift_options) -> np.ndarray:
        log.info("Computing per element energy regression.")

        lambd = shift_options["energy_regularisation"]
        energies = labels["energy"]
        numbers = inputs["numbers"]
        system_sizes = inputs["n_atoms"]

        energies = np.array(energies)
        system_sizes = np.array(system_sizes)

        ds_energy = np.sum(energies)
        n_atoms_total = np.sum(system_sizes)

        mean_energy = ds_energy / n_atoms_total
        n_species = 119  # for simplicity, we assume any element could be in the dataset
        X = np.zeros(shape=(energies.shape[0], n_species))
        y = np.zeros(energies.shape[0])

        for i in range(energies.shape[0]):
            Z_counts = np.zeros(n_species)
            for z in set(numbers[i]):
                Z_counts[z] = np.count_nonzero(numbers[i] == z)
            X[i] = Z_counts

            E_sub_mean = energies[i] - mean_energy * numbers[i].shape[0]
            y[i] = E_sub_mean
        XTX = X.T @ X
        reg_term = lambd * np.eye(XTX.shape[0])

        result = np.linalg.lstsq(XTX + reg_term, X.T @ y, rcond=-1)
        elemental_energies_shift = result[0]
        elemental_energies_shift += mean_energy

        return elemental_energies_shift


class IsolatedAtomEnergyShift:
    name = "isolated_atom_energy_shift"
    parameters = ["E0s"]
    dtypes = [dict[int, float]]

    @staticmethod
    def compute(inputs, labels, shift_options):
        shift_options = shift_options["E0s"]
        n_species = 119
        elemental_energies_shift = np.zeros(n_species)
        for k, v in shift_options.items():
            elemental_energies_shift[k] = v

        return elemental_energies_shift


class MeanEnergyShift:
    name = "mean_atom_energy_shift"
    parameters = []
    dtypes = []

    @staticmethod
    def compute(inputs, labels, shift_options):
        energies = labels["energy"]
        system_sizes = inputs["n_atoms"]

        energies = np.array(energies)
        system_sizes = np.array(system_sizes)

        ds_energy = np.sum(energies)
        n_atoms_total = np.sum(system_sizes)

        mean_energy = ds_energy / n_atoms_total
        return mean_energy


class MeanEnergyRMSScale:
    name = "mean_energy_rms_scale"
    parameters = []
    dtypes = []

    @staticmethod
    def compute(inputs, labels, scale_options):
        # log.info("Computing per element energy regression.")
        energies = labels["energy"]
        numbers = inputs["numbers"]
        system_sizes = inputs["n_atoms"]

        energies = np.array(energies)
        system_sizes = np.array(system_sizes)

        ds_energy = np.sum(energies)
        n_atoms_total = np.sum(system_sizes)

        mean_energy = ds_energy / n_atoms_total

        mean_err_sse = 0.0

        for i in range(energies.shape[0]):
            E_sub_mean = energies[i] - mean_energy * numbers[i].shape[0]
            mean_err_sse += E_sub_mean**2 / numbers[i].shape[0]

        energy_std = np.sqrt(mean_err_sse / n_atoms_total)
        return energy_std


class PerElementForceRMSScale:
    name = "per_element_force_rms_scale"
    parameters = []
    dtypes = []

    @staticmethod
    def compute(inputs, labels, scale_options):
        n_species = 119

        forces = np.concatenate(labels["forces"], axis=0)
        numbers = np.concatenate(inputs["numbers"], axis=0)

        elements = np.unique(numbers)

        element_scale = np.ones(n_species)

        for element in elements:
            element_forces = forces[numbers == element]
            force_rms = np.sqrt(np.mean(np.linalg.norm(element_forces, 2, axis=1)))
            element_scale[element] = force_rms

        return element_scale


class GlobalCustomScale:
    name = "global_custom_scale"
    parameters = ["factor"]
    dtypes = [float]

    @staticmethod
    def compute(inputs, labels, scale_options):
        element_scale = scale_options["factor"]
        return element_scale


class PerElementCustomScale:
    name = "per_element_custom_scale"
    parameters = ["factors"]
    dtypes = [dict[int, float]]

    @staticmethod
    def compute(inputs, labels, scale_options):
        n_species = 119
        element_scale = np.ones(n_species)
        for k, v in scale_options["factors"].items():
            element_scale[k] = v

        return element_scale


shift_method_list = [PerElementRegressionShift, IsolatedAtomEnergyShift, MeanEnergyShift]
scale_method_list = [
    MeanEnergyRMSScale,
    PerElementForceRMSScale,
    GlobalCustomScale,
    PerElementCustomScale,
]


def compute_scale_shift_parameters(
    inputs, labels, shift_method, scale_method, shift_options, scale_options
):
    shift_methods = {method.name: method for method in shift_method_list}
    scale_methods = {method.name: method for method in scale_method_list}

    if shift_method not in shift_methods.keys():
        raise KeyError(
            f"The shift method '{shift_method}' is not among the implemented methods."
            f" Choose from {shift_method.keys()}"
        )
    if scale_method not in scale_methods.keys():
        raise KeyError(
            f"The scale method '{scale_method}' is not among the implemented methods."
            f" Choose from {scale_method.keys()}"
        )

    shift_method = shift_methods[shift_method]
    scale_method = scale_methods[scale_method]

    shift_parameters = shift_method.compute(inputs, labels, shift_options)
    scale_parameters = scale_method.compute(inputs, labels, scale_options)

    ds_stats = DatasetStats(shift_parameters, scale_parameters)
    return ds_stats
