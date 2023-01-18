import importlib.metadata
import logging
from pathlib import Path

import typer
from rich import print

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})


@app.command()
def train(
    train_config_path: Path = typer.Argument(
        ..., help="Training configuration YAML file."
    ),
    log_level: str = typer.Option("off", help="Sets the training logging level."),
    log_file: str = typer.Option("train.log", help="Specifies the name of the log file"),
):
    """
    Starts the training of a GMNN model with parameters provided by a configuration file.
    """

    if log_level != "off":
        log_levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        logging.basicConfig(filename=log_file, level=log_levels[log_level])

    import tensorflow as tf

    tf.config.experimental.set_visible_devices([], "GPU")
    from jax.config import config

    config.update("jax_enable_x64", True)
    from gmnn_jax.train.run import run

    run(train_config_path)


@app.command()
def md(
    train_config_path: Path = typer.Argument(
        ..., help="Configuration YAML file that was used to train a model."
    ),
    md_config_path: Path = typer.Argument(..., help="MD configuration YAML file."),
    log_level: str = typer.Option("off", help="Sets the training logging level."),
    log_file: str = typer.Option("train.log", help="Specifies the name of the log file"),
):
    """
    Starts performing a molecular dynamics simulation (currently only NHC thermostat)
    with paramters provided by a configuration file.
    """
    if log_level != "off":
        log_levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        logging.basicConfig(filename=log_file, level=log_levels[log_level])
    from gmnn_jax.md import run_md

    run_md(train_config_path, md_config_path)


@app.command()
def docs():
    """
    Opens the documentation website in your browser.
    """
    print("Opening gmnn-jax's docs at https://github.com/GM-NN/gmnn-jax")
    typer.launch("https://github.com/GM-NN/gmnn-jax")


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
