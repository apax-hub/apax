import os
import urllib
import zipfile


def download_md22_stachyose(data_path):
    url = "http://www.quantum-machine.org/gdml/repo/static/md22_stachyose.zip"
    file_path = data_path / "md22_stachyose.zip"

    os.makedirs(data_path, exist_ok=True)
    urllib.request.urlretrieve(url, file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    file_path = modify_xyz_file(
        file_path.with_suffix(".xyz"), target_string="Energy", replacement_string="energy"
    )

    return file_path


def download_benzene_DFT(data_path):
    url = "http://www.quantum-machine.org/gdml/data/xyz/benzene2018_dft.zip"
    file_path = data_path / "benzene2018_dft.zip"

    os.makedirs(data_path, exist_ok=True)
    urllib.request.urlretrieve(url, file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    new_file_path = data_path / "benzene.xyz"
    os.remove(file_path)

    return new_file_path


def download_etoh_ccsdt(data_path):
    url = "http://www.quantum-machine.org/gdml/data/xyz/ethanol_ccsd_t.zip"
    file_path = data_path / "ethanol_ccsd_t.zip"

    os.makedirs(data_path, exist_ok=True)
    urllib.request.urlretrieve(url, file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    test_file_path = data_path / "ethanol_ccsd_t-test.xyz"
    train_file_path = data_path / "ethanol_ccsd_t-train.xyz"
    os.remove(file_path)

    return train_file_path, test_file_path


def download_md22_benzene_CCSDT(data_path):
    url = "http://www.quantum-machine.org/gdml/data/xyz/benzene_ccsd_t.zip"
    file_path = data_path / "benzene_ccsdt.zip"

    os.makedirs(data_path, exist_ok=True)
    urllib.request.urlretrieve(url, file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    train_file_path = data_path / "benzene_ccsd_t-train.xyz"
    test_file_path = data_path / "benzene_ccsd_t-test.xyz"
    os.remove(file_path)

    return train_file_path, test_file_path


def modify_xyz_file(file_path, target_string, replacement_string):
    new_file_path = file_path.with_name(file_path.stem + "_mod" + file_path.suffix)

    with open(file_path, "r") as input_file, open(new_file_path, "w") as output_file:
        for line in input_file:
            # Replace all occurrences of the target string with the replacement string
            modified_line = line.replace(target_string, replacement_string)
            output_file.write(modified_line)
    return new_file_path


def mod_md_datasets(file_path):
    new_file_path = file_path.with_name(file_path.stem + "_mod" + file_path.suffix)
    with open(file_path, "r") as input_file, open(new_file_path, "w") as output_file:
        for line in input_file:
            if line.startswith("-"):
                modified_line = f"Properties=species:S:1:pos:R:3:forces:R:3 energy={line}"
            else:
                modified_line = line
            output_file.write(modified_line)

    os.remove(file_path)

    return new_file_path
