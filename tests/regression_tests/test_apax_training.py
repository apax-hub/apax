import os
import pathlib
import urllib
import zipfile
import uuid

import numpy as np
import pandas as pd
import pytest
import yaml

from apax.train.run import run
from apax.data.statistics import scale_method_list, shift_method_list

TEST_PATH = pathlib.Path(__file__).parent.resolve()

def load_config_and_run_training(
    config_path, **config_kwargs
):
    with open(config_path.as_posix(), "r") as stream:
        config_dict = yaml.safe_load(stream)

    for pydentic_model_key, config_mods in config_kwargs.items():
        for h_param_key, value in config_mods.items():
            config_dict[pydentic_model_key][h_param_key] = value

    run(config_dict)
    

def load_csv(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  # Skip the header row

    with open(filename, 'r') as file:
        header = file.readline().strip().split(',')

    data_dict = {header[i]: data[:, i].tolist() for i in range(len(header))}

    return data_dict

    
@pytest.mark.slow
def test_regression_model_training(get_md22_stachyose, get_tmp_path):
    config_path = TEST_PATH / "apax_config.yaml"
    working_dir = get_tmp_path / str(uuid.uuid4())
    file_path = get_md22_stachyose

    data_config_mods = {
        "directory": working_dir.as_posix(),
        "data_path": file_path.as_posix(),
        "energy_unit": "kcal/mol",
    }

    load_config_and_run_training(config_path, data=data_config_mods)

    current_metrics = load_csv(working_dir / "test/log.csv")

    comparison_metrics = {
        "val_energy_mae": 0.24696787788040334,
        "val_forces_mae": 0.09672525137916232,
        "val_forces_mse": 0.017160819058234304,
        "val_loss": 0.45499257304743396,
    }
    for key in comparison_metrics.keys():
        print(np.array(current_metrics[key])[-1])

    for key in comparison_metrics.keys():
        assert (
            abs((np.array(current_metrics[key])[-1] / comparison_metrics[key]) - 1) < 1e-3
        )