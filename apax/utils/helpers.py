import collections
import csv
import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

# default whitelist of properties for jaxMD
APAX_PROPERTIES = [
    "energy",
    "forces",
    "stress",
    "forces_uncertainty",
    "energy_uncertainty",
    "stress_uncertainty",
    "energy_ensemble",
    "forces_ensemble",
    "stress_ensemble",
    "energy_unbiased",
    "forces_unbiased",
    "charge",
    "charges",
]


log = logging.getLogger(__name__)


def mod_config(
    config_path: Union[str, Path], updated_config: dict[str, Any]
) -> dict[str, Any]:
    """Update a configuration in a YAML file.

    Args:
        config_path (Union[str, Path]): path to YAML file containing old
            configuration
        updated_config (dict[str, Any]): dictionary with new key-value pairs

    Returns:
        config_dict (dict[str, Any]): dictionary of updated configuration
    """

    with open(config_path.as_posix(), "r") as stream:
        config_dict = yaml.safe_load(stream)

    for key, new_value in updated_config.items():
        if key in config_dict.keys():
            if isinstance(config_dict[key], dict):
                config_dict[key].update(new_value)
            else:
                config_dict[key] = new_value
        else:
            config_dict[key] = new_value
    return config_dict


def load_csv_metrics(path: Union[str, Path]) -> dict[str, list[float]]:
    """Load metrics from during training.

    Args:
        path (Union[str, Path]): path to csv file

    Returns:
        data_dict (dict[str, list[float]]): dictionary with a key for each
            metric and values of each metric during training.
    """

    data_dict = {}

    with open(path, "r") as file:
        reader = csv.reader(file)

        # Extract the headers (keys) from the first row
        try:
            headers = next(reader)
        except StopIteration as e:
            raise RuntimeError(
                f"Could not load csv metrics from {path}, file is empty"
            ) from e

        # Initialize empty lists for each key
        for header in headers:
            data_dict[header] = []

        # Read the rest of the rows and append values to the corresponding key
        for row in reader:
            for idx, value in enumerate(row):
                key = headers[idx]
                data_dict[key].append(float(value))

    return data_dict


def update_nested_dictionary(dct: dict, other: dict) -> dict:
    """Update a nested dictionary with new key-value pairs.

    Args:
        dct (dict): dictionary to update
        other (dict): dictionary with new key-value pairs

    Returns:
        dct (dct): Updated dictionary
    """
    # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    for k, v in other.items():
        if isinstance(v, collections.abc.Mapping):
            dct[k] = update_nested_dictionary(dct.get(k, {}), v)
        else:
            dct[k] = v
    return dct


def get_ase_mass(symbol_or_mass: str | int | float) -> float:
    """Get the mass in ASE, or just return the mass.

    Simply returns the value if `symbol_or_mass` is an integer or float,
    and otherwise returns the mass that element has in ASE.

    Args:
        symbol_or_mass (str | int | float): The symbol, or mass.

    Returns:
        float: Mass of element.

    Raises:
        ValueError: If `symbol_or_mass` is an integer or float that is less
            than or equal to 0.
        TypeError: If `symbol_or_mass` is not a string, integer or float.

    """
    if isinstance(symbol_or_mass, str):
        if symbol_or_mass not in atomic_numbers:
            msg = f"Unknown element symbol '{symbol_or_mass}'."
            raise ValueError(msg)
        atomic_number = atomic_numbers[symbol_or_mass]
        return atomic_masses[atomic_number]
    elif isinstance(symbol_or_mass, int | float):
        if symbol_or_mass <= 0:
            msg = f"The provided mass is less than 0 ({symbol_or_mass})."
            raise ValueError(msg)
        return float(symbol_or_mass)
    else:
        raise TypeError(
            f"Expected a string or number for mass, got {type(symbol_or_mass)}."
        )


def get_updated_atomic_masses(
    custom_mass_dictionary: dict[str, float | str],
) -> np.ndarray:
    """Get an updated list of custom atomic masses.

    Args:
        custom_mass_dictionary (dict[str, float | str]): Dictionary of custom
            atomic masses, with the keys being the elements, and the values
            either being numbers to indicate its mass, or strings to indicate that
            this the element indicated by the key has the same mass as the element in
            the value.

    Returns:
        atomic_masses_cpy (np.ndarray): Array with the updated atomic masses.

    """
    atomic_masses_cpy = atomic_masses.copy()
    for element, symbol_or_mass in custom_mass_dictionary.items():
        element_mass = get_ase_mass(symbol_or_mass)
        atomic_number = atomic_numbers[element]
        log.info(
            f"Setting mass of element {element} (atomic number {atomic_number}) to {symbol_or_mass}"
        )
        atomic_masses_cpy[atomic_number] = element_mass

    return atomic_masses_cpy


def get_masses(
    custom_mass_dictionary: dict[str, float | str], atoms: Atoms
) -> np.ndarray:
    """Get the masses of all atoms in `atoms`.

    Args:
        custom_mass_dictionary (dict[str, float | str]): Dictionary of custom
            atomic masses, with the keys being the elements, and the values
            either being numbers to indicate its mass, or strings to indicate that
            this the element indicated by the key has the same mass as the element in
            the value.
        atoms (Atoms): Atoms to get the masses for.

    Returns:
        np.ndarray: Masses of all atoms in `atoms`.

    """
    atomic_masses = get_updated_atomic_masses(custom_mass_dictionary)

    masses = []
    for atom in atoms:
        masses.append(atomic_masses[atom.number])

    return np.array(masses)
