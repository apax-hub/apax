import typer
from pathlib import Path
import importlib.metadata
import logging
from rich import print


app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})


@app.command()
def train(
    train_config_path: Path,
    log_level: int = typer.Option(3, help="Sets the training logging level."),
    log_file: str = typer.Option("train.log", help="Specifies the name of the log file"),
):
    """
    Starts the training of a GMNN model with parameters provided by a configuration file.
    """
    print("md_config_path", train_config_path)
    print("log_level", log_level)


@app.command()
def md(
    md_config_path: Path,
    log_level: int = typer.Option(3, help="Sets the training logging level."),
    log_file: str = typer.Option("train.log", help="Specifies the name of the log file"),
):
    """
    Starts performing a molecular dynamics simulation (currently only NHC thermostat)
    with paramters provided by a configuration file.
    """
    print("md_config_path", md_config_path)


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
