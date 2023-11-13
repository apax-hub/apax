import pathlib
import pytest
from typer.testing import CliRunner
import yaml

from apax.cli.apax_app import app, validate_app, template_app


TEST_PATH = pathlib.Path(__file__).parent.resolve()

runner = CliRunner()


def test_cli_basic(get_tmp_path):
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
    # TODO to the same for md
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
