import os
import pathlib
import sys
import subprocess

try:
    import ipsuite as ips
except ImportError:
    pass

import numpy as np
import pytest
import yaml
import zntrack

import apax.nodes
from apax.nodes.model import Apax, ApaxEnsemble
from apax.nodes.utils import AddData

CONFIG_PATH = pathlib.Path(__file__).parent / "example.yaml"
TEST_PATH = pathlib.Path(__file__).parent.resolve()


def save_config_with_seed(path: str, seed: int = 1) -> None:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["seed"] = seed

    with open(path, "w") as f:
        yaml.dump(config, f)


def test_n_train_model(tmp_path, get_md22_stachyose):
    os.chdir(tmp_path)
    subprocess.run(["git", "init"], check=True)
    subprocess.run(["dvc", "init"], check=True)
    save_config_with_seed(tmp_path / "example.yaml")
    proj = zntrack.Project()
    with proj:
        data = AddData(file=get_md22_stachyose)
        model = Apax(data=data.atoms, validation_data=data.atoms, config="example.yaml")

    proj.repro()

    model = model.from_rev()
    data = data.from_rev()

    atoms = data.atoms[0]
    atoms.calc = model.get_calculator()

    assert atoms.get_potential_energy() < 0


@pytest.mark.skipif("ipsuite" not in sys.modules, reason="requires new ipsuite release")
def test_n_train_2_model(tmp_path, get_md22_stachyose):
    os.chdir(tmp_path)
    subprocess.run(["git", "init"], check=True)
    subprocess.run(["dvc", "init"], check=True)

    save_config_with_seed(tmp_path / "example.yaml")
    save_config_with_seed(tmp_path / "example2.yaml", seed=2)

    proj = zntrack.Project(automatic_node_names=True)
    thermostat = ips.calculators.LangevinThermostat(
        time_step=1.0, temperature=100.0, friction=0.01
    )
    with proj:
        data = AddData(file=get_md22_stachyose)
        model1 = Apax(data=data.atoms, validation_data=data.atoms, config="example.yaml")
        model2 = Apax(data=data.atoms, validation_data=data.atoms, config="example2.yaml")
        ensemble = ApaxEnsemble(models=[model1, model2])
        md = ips.calculators.ASEMD(
            data=data.atoms,
            model=ensemble,
            thermostat=thermostat,
            steps=20,
            sampling_rate=1,
        )

        uncertainty_selection = ips.configuration_selection.ThresholdSelection(
            data=md, n_configurations=1, threshold=0.0001
        )

        selection_batch_size = 3
        kernel_selection = apax.nodes.BatchKernelSelection(
            data=md.atoms,
            train_data=data.atoms,
            models=[model1, model2],
            n_configurations=selection_batch_size,
            processing_batch_size=4,
        )

        prediction = ips.analysis.Prediction(data=kernel_selection.atoms, model=ensemble)
        analysis = ips.analysis.PredictionMetrics(
            x=kernel_selection.atoms, y=prediction.atoms
        )

    proj.repro()

    model = ensemble.from_rev()
    data = data.from_rev()

    atoms = data.atoms[0]
    atoms.calc = model.get_calculator()

    assert atoms.get_potential_energy() < 0

    uncertainty_selection.load()
    kernel_selection.load()
    md.load()

    uncertainties = [x.calc.results["energy_uncertainty"] for x in md.atoms]
    assert [md.atoms[np.argmax(uncertainties)]] == uncertainty_selection.atoms

    assert len(kernel_selection.atoms) == selection_batch_size
