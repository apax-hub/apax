import dataclasses
import logging

import numpy as np

log = logging.getLogger(__name__)


@dataclasses.dataclass
class DatasetStats:
    elemental_shift: np.array = None
    elemental_scale: float = None
    n_atoms: int = 0
    n_species: int = 0
    displacement_fn = None


def energy_per_element(atoms_list, lambd=1.0):
    log.info("Computing per element energy regression.")
    energies = [atoms.get_potential_energy() for atoms in atoms_list]
    numbers = [atoms.numbers for atoms in atoms_list]
    system_sizes = [num.shape[0] for num in numbers]

    energies = np.array(energies)
    system_sizes = np.array(system_sizes)

    ds_energy = np.sum(energies)
    n_atoms_total = np.sum(system_sizes)

    mean_energy = ds_energy / n_atoms_total
    n_species = max([max(n) for n in numbers]) + 1

    mean_err_sse = 0.0
    X = np.zeros(shape=(energies.shape[0], n_species))
    y = np.zeros(energies.shape[0])

    for i in range(energies.shape[0]):
        Z_counts = np.zeros(n_species)
        for z in set(numbers[i]):
            Z_counts[z] = np.count_nonzero(numbers[i] == z)
        X[i] = Z_counts

        E_sub_mean = energies[i] - mean_energy * numbers[i].shape[0]
        y[i] = E_sub_mean
        mean_err_sse += E_sub_mean**2 / numbers[i].shape[0]
    XTX = X.T @ X
    reg_term = lambd * np.eye(XTX.shape[0])

    result = np.linalg.lstsq(XTX + reg_term, X.T @ y, rcond=-1)
    elemental_energies_mean = result[0]
    elemental_energies_mean += mean_energy

    elemental_energies_std = np.sqrt(mean_err_sse / n_atoms_total)

    ds_stats = DatasetStats(
        elemental_energies_mean, elemental_energies_std, np.max(system_sizes), n_species
    )
    return ds_stats
