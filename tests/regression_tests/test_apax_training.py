import os
import pathlib
import urllib
import zipfile

import numpy as np
import pandas as pd
import pytest
import yaml

from apax.train.run import run

TEST_PATH = pathlib.Path(__file__).parent.resolve()
MD22_STACHYOSE_URL = "http://www.quantum-machine.org/gdml/repo/static/md22_stachyose.zip"


def download_and_extract_data(data_path, filename, url, file_format):
    file_path = data_path / filename

    os.makedirs(data_path, exist_ok=True)
    urllib.request.urlretrieve(url, file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    return file_path.with_suffix(f".{file_format}")


def modify_xyz_file(file_path, target_string, replacement_string):
    new_file_path = file_path.with_name(file_path.stem + "_mod" + file_path.suffix)

    with open(file_path, "r") as input_file, open(new_file_path, "w") as output_file:
        for line in input_file:
            # Replace all occurrences of the target string with the replacement string
            modified_line = line.replace(target_string, replacement_string)
            output_file.write(modified_line)
    return new_file_path


def load_config_and_run_training(
    config_path, file_path, working_dir, energy_unit="eV", pos_unit="Ang"
):
    with open(config_path.as_posix(), "r") as stream:
        config_dict = yaml.safe_load(stream)

    config_dict["data"]["directory"] = working_dir.as_posix()
    config_dict["data"]["data_path"] = file_path.as_posix()
    config_dict["data"]["energy_unit"] = energy_unit
    config_dict["data"]["pos_unit"] = pos_unit

    run(config_dict)

    return


@pytest.mark.slow
def test_regression_model_training(get_tmp_path):
    config_path = TEST_PATH / "apax_config.yaml"
    working_dir = get_tmp_path
    data_path = working_dir / "data"
    filename = "md22_stachyose.zip"

    file_path = download_and_extract_data(data_path, filename, MD22_STACHYOSE_URL, "xyz")

    file_path = modify_xyz_file(
        file_path, target_string="Energy", replacement_string="energy"
    )

    load_config_and_run_training(config_path, file_path, working_dir, "kcal/mol")

    current_metrics = pd.read_csv(working_dir / "test/log.csv")

    comparison_metrics = {
        "val_energy_mae": 0.2048215700433502,
        "val_forces_mae": 0.054957914591049,
        "val_forces_mse": 0.0056583952479869,
        "val_loss": 0.1395589689994847,
    }

    for key in comparison_metrics.keys():
        assert (
            abs((np.array(current_metrics[key])[-1] / comparison_metrics[key]) - 1) < 1e-4
        )
