import pathlib

import pytest
import yaml
from typer.testing import CliRunner

from apax.cli.apax_app import app, template_app, validate_app

TEST_PATH = pathlib.Path(__file__).parent.resolve()

runner = CliRunner()


def test_cli_validate(get_tmp_path):
    # This test also checks whether the templates we provide are valid
    pytest.MonkeyPatch().chdir(get_tmp_path)
    result = runner.invoke(app, ["-h"])
    assert result.exit_code == 0
    assert "Commands" in result.stdout

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0

    result = runner.invoke(template_app, "train")
    assert result.exit_code == 0
    result = runner.invoke(template_app, "train")
    assert result.exit_code == 1

    result = runner.invoke(template_app, ["train", "--full"])
    assert result.exit_code == 0
    result = runner.invoke(template_app, ["train", "--full"])
    assert result.exit_code == 1

    result = runner.invoke(template_app, "md")
    assert result.exit_code == 0
    result = runner.invoke(template_app, "md")
    assert result.exit_code == 1

    # Expect the unedited templates to fail
    result = runner.invoke(validate_app, ["train", "config.yaml"])
    assert result.exit_code == 1

    result = runner.invoke(validate_app, ["train", "config_full.yaml"])
    assert result.exit_code == 1

    result = runner.invoke(validate_app, ["md", "md_config.yaml"])
    assert result.exit_code == 1

    # Load and fix the train config, then try again
    with open("config_full.yaml", "r") as stream:
        model_config_dict = yaml.safe_load(stream)

    model_config_dict["n_epochs"] = 100
    model_config_dict["data"]["data_path"] = "dataset.extxyz"

    with open("config_full_fixed.yaml", "w") as conf:
        yaml.dump(model_config_dict, conf, default_flow_style=False)

    result = runner.invoke(validate_app, ["train", "config_full_fixed.yaml"])
    assert result.exit_code == 0
    assert "Success!" in result.stdout

    result = runner.invoke(app, ["visualize", "config_full_fixed.yaml"])
    assert result.exit_code == 0
    assert "Total Parameters" in result.stdout

    # same for md
    with open("md_config.yaml", "r") as stream:
        md_config_dict = yaml.safe_load(stream)

    md_config_dict["initial_structure"] = "initial_structure.extxyz"
    md_config_dict["duration"] = 10
    md_config_dict["ensemble"]["temperature_schedule"]["T0"] = 10

    with open("md_config_fixed.yaml", "w") as conf:
        yaml.dump(md_config_dict, conf, default_flow_style=False)

    result = runner.invoke(validate_app, ["md", "md_config_fixed.yaml"])
    assert result.exit_code == 0
    assert "Success!" in result.stdout
