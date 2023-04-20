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


def per_element_regression(atoms_list, scale_shift_options):
    if "energy_regularization" not in scale_shift_options:
        raise KeyError("Per element regression requires the 'energy_regularization' key")
    lambd=scale_shift_options["energy_regularization"]

    key_words = scale_shift_options.keys()
    key_words.remove("energy_regularization")
    if len(key_words) > 0:
        raise KeyError(f"Per element regression received unknown arguments: {key_words}")


    log.info("Computing per element energy regression.")
    energies = [atoms.get_potential_energy() for atoms in atoms_list]
    numbers = [atoms.numbers for atoms in atoms_list]
    system_sizes = [num.shape[0] for num in numbers]

    energies = np.array(energies)
    system_sizes = np.array(system_sizes)

    ds_energy = np.sum(energies)
    n_atoms_total = np.sum(system_sizes)

    mean_energy = ds_energy / n_atoms_total
    n_species = 119  # max([max(n) for n in numbers]) + 1

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

    ds_stats = DatasetStats(elemental_energies_mean, elemental_energies_std, 0, n_species)
    return ds_stats


def isolated_atom_energies(train_atoms_list, E0s):
    n_species = 119

    elemental_energies_shift = np.zeros(n_species)
    for k,v in E0s.items():
        elemental_energies_shift[k] = v

    elemental_energies_scale = np.zeros(n_species)
    ds_stats = DatasetStats(elemental_energies_shift, elemental_energies_scale, 0, n_species)
    return ds_stats