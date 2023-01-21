import importlib.metadata
from pathlib import Path

import typer
import yaml
from rich import print

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})
validate_app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Validate training or MD config files.",
)
app.add_typer(validate_app, name="validate")


@app.command()
def train(
    train_config_path: Path = typer.Argument(
        ..., help="Training configuration YAML file."
    ),
    log_level: str = typer.Option("error", help="Sets the training logging level."),
    log_file: str = typer.Option("train.log", help="Specifies the name of the log file"),
):
    """
    Starts the training of a GMNN model with parameters provided by a configuration file.
    """
    import tensorflow as tf

    tf.config.experimental.set_visible_devices([], "GPU")

    from jax.config import config

    config.update("jax_enable_x64", True)

    from gmnn_jax.train.run import run

    run(train_config_path, log_file, log_level)


@app.command()
def md(
    train_config_path: Path = typer.Argument(
        ..., help="Configuration YAML file that was used to train a model."
    ),
    md_config_path: Path = typer.Argument(..., help="MD configuration YAML file."),
    log_level: str = typer.Option("error", help="Sets the training logging level."),
    log_file: str = typer.Option("train.log", help="Specifies the name of the log file"),
):
    """
    Starts performing a molecular dynamics simulation (currently only NHC thermostat)
    with paramters provided by a configuration file.
    """
    from gmnn_jax.md import run_md

    run_md(train_config_path, md_config_path, log_file, log_level)


@app.command()
def docs():
    """
    Opens the documentation website in your browser.
    """
    print("Opening gmnn-jax's docs at https://github.com/GM-NN/gmnn-jax")
    typer.launch("https://github.com/GM-NN/gmnn-jax")


@validate_app.command("train")
def validate_train_config(
    config_path: Path = typer.Argument(
        ..., help="Configuration YAML file to be validated."
    ),
):
    """
    Validates a training configuration file.

    Parameters
    ----------
    config_path: Path to the training configruation file.
    """
    from gmnn_jax.config import Config

    with open(config_path, "r") as stream:
        user_config = yaml.safe_load(stream)

    _ = Config.parse_obj(user_config)

    print("Success!")
    print(f"{config_path} is a valid training config.")


@validate_app.command("md")
def validate_md_config(
    config_path: Path = typer.Argument(
        ..., help="Configuration YAML file to be validated."
    ),
):
    """
    Validates a molecular dynamics configuration file.

    Parameters
    ----------
    config_path: Path to the molecular dynamics  configruation file.
    """
    from gmnn_jax.config import MDConfig

    with open(config_path, "r") as stream:
        user_config = yaml.safe_load(stream)

    _ = MDConfig.parse_obj(user_config)

    print("Success!")
    print(f"{config_path} is a valid MD config.")


def version_callback(value: bool) -> None:
    """Get the installed gmnn-jax version."""
    if value:
        print(f"gmnn-jax {importlib.metadata.version('gmnn-jax')}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-V", callback=version_callback, is_eager=True
    ),
):
    # Taken from https://github.com/zincware/dask4dvc/blob/main/dask4dvc/cli/main.py
    _ = version


if __name__ == "__main__":
    app()
