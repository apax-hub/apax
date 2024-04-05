import os
import pathlib

import yaml
import zntrack

from apax.nodes.model import Apax, ApaxEnsemble
from apax.nodes.utils import AddData

CONFIG_PATH = pathlib.Path(__file__).parent / "example.yaml"


def save_config_with_seed(path: str, seed: int = 1) -> None:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["seed"] = seed

    with open(path, "w") as f:
        yaml.dump(config, f)


def test_n_train_model(tmp_path, get_md22_stachyose):
    os.chdir(tmp_path)
    save_config_with_seed(tmp_path / "example.yaml")
    proj = zntrack.Project()
    with proj:
        data = AddData(file=get_md22_stachyose)
        model = Apax(data=data.atoms, validation_data=data.atoms, config="example.yaml")

    proj.run()

    model = model.from_rev()
    data = data.from_rev()

    atoms = data.atoms[0]
    atoms.calc = model.get_calculator()

    assert atoms.get_potential_energy() < 0


def test_n_train_2_model(tmp_path, get_md22_stachyose):
    os.chdir(tmp_path)

    save_config_with_seed(tmp_path / "example.yaml")
    save_config_with_seed(tmp_path / "example2.yaml", seed=2)

    proj = zntrack.Project(automatic_node_names=True)
    with proj:
        data = AddData(file=get_md22_stachyose)
        model1 = Apax(data=data.atoms, validation_data=data.atoms, config="example.yaml")
        model2 = Apax(data=data.atoms, validation_data=data.atoms, config="example2.yaml")
        ensemble = ApaxEnsemble(models=[model1, model2])

    proj.run()

    model = ensemble.from_rev()
    data = data.from_rev()

    atoms = data.atoms[0]
    atoms.calc = model.get_calculator()

    assert atoms.get_potential_energy() < 0
    assert atoms.calc.results["energy_uncertainty"] > 0
