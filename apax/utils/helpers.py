import collections
import csv

import yaml

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
]


def setup_ase():
    """Add uncertainty keys to ASE all properties.
    from https://github.com/zincware/IPSuite/blob/main/ipsuite/utils/helpers.py#L10
    """
    from ase.calculators.calculator import all_properties

    for val in APAX_PROPERTIES:
        if val not in all_properties:
            all_properties.append(val)


def mod_config(config_path, updated_config):
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


def load_csv_metrics(path):
    data_dict = {}

    with open(path, "r") as file:
        reader = csv.reader(file)

        # Extract the headers (keys) from the first row
        headers = next(reader)

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
    # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    for k, v in other.items():
        if isinstance(v, collections.abc.Mapping):
            dct[k] = update_nested_dictionary(dct.get(k, {}), v)
        else:
            dct[k] = v
    return dct
