import pytest
import pathlib
import yaml
from apax.train.run import run
import urllib
import zipfile
import os

TEST_PATH = pathlib.Path(__file__).parent.resolve()
MD22_STACHYOSE_URL = "http://www.quantum-machine.org/gdml/repo/static/md22_stachyose.zip"

@pytest.mark.slow
def test_regression_model_training(get_tmp_path):
    temp_path = get_tmp_path

    #load dataset and safe it in temp_dir
    data_path = temp_path / "data"
    os.makedirs(data_path, exist_ok=True)
    urllib.request.urlretrieve(MD22_STACHYOSE_URL, data_path / "md22_stachyose.zip")

    with zipfile.ZipFile(data_path / "md22_stachyose.zip", "r") as zip_ref:
        zip_ref.extractall(data_path)

    input_file_path = (data_path / "md22_stachyose.xyz").as_posix()
    output_file_path = (data_path / "md22_stachyose_mod.xyz").as_posix()
    target_string = 'Energy'
    replacement_string = 'energy'
    
    replace_string_in_file(input_file_path, output_file_path, target_string, replacement_string)

    #read and adjust config
    confg_path = TEST_PATH / "apax_config.yaml"
    with open(confg_path.as_posix(), "r") as stream:
        config_dict = yaml.safe_load(stream)

    config_dict["data"]["directory"] = temp_path.as_posix()
    config_dict["data"]["data_path"] = output_file_path
    config_dict["data"]["energy_unit"] = "kcal/mol"

    run(config_dict)

    
    assert False

def replace_string_in_file(input_file_path, output_file_path, target_string, replacement_string):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            # Replace all occurrences of the target string with the replacement string
            modified_line = line.replace(target_string, replacement_string)
            output_file.write(modified_line)