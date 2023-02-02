import importlib.metadata
import importlib.resources as pkg_resources
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console

from gmnn_jax.cli import templates

console = Console(highlight=False)
app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})
validate_app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Validate training or MD config files.",
)
template_app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Create configuration file templates.",
)
app.add_typer(validate_app, name="validate")
app.add_typer(template_app, name="template")


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
    from gmnn_jax.train.run import run

    run(train_config_path, log_file, log_level)


@app.command()
def md(
    train_config_path: Path = typer.Argument(
        ..., help="Configuration YAML file that was used to train a model."
    ),
    md_config_path: Path = typer.Argument(..., help="MD configuration YAML file."),
    log_level: str = typer.Option("error", help="Sets the training logging level."),
    log_file: str = typer.Option("md.log", help="Specifies the name of the log file"),
):
    """
    Starts performing a molecular dynamics simulation (currently only NHC thermostat)
    with paramters provided by a configuration file.
    """
    from gmnn_jax.md import run_md

    run_md(train_config_path, md_config_path, log_file, log_level)


@app.command()
def eval(
    train_config_path: Path = typer.Argument(
        ..., help="Configuration YAML file that was used to train a model."
    ),
    n_data: int = typer.Option(..., help="Number of test structures"),
):
    """
    Starts performing the evaluation of the test dataset
    with parameters provided by a configuration file.
    """
    from gmnn_jax.train.eval import eval_model

    eval_model(train_config_path, n_data)


@app.command()
def docs():
    """
    Opens the documentation website in your browser.
    """
    console.print("Opening gmnn-jax's docs at https://github.com/GM-NN/gmnn-jax")
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
    config_path: Path to the training configuration file.
    """
    from gmnn_jax.config import Config

    with open(config_path, "r") as stream:
        user_config = yaml.safe_load(stream)

    try:
        _ = Config.parse_obj(user_config)
    except ValidationError as e:
        print(e)
        console.print("Configuration Invalid!", style="red3")
        raise typer.Exit(code=1)
    else:
        console.print("Success!", style="green3")
        console.print(f"{config_path} is a valid training config.")


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
    config_path: Path to the molecular dynamics configuration file.
    """
    from gmnn_jax.config import MDConfig

    with open(config_path, "r") as stream:
        user_config = yaml.safe_load(stream)

    try:
        _ = MDConfig.parse_obj(user_config)
    except ValidationError as e:
        print(e)
        console.print("Configuration Invalid!", style="red3")
        raise typer.Exit(code=1)
    else:
        console.print("Success!", style="green3")
        console.print(f"{config_path} is a valid MD config.")


@app.command("visualize")
def visualize_model(
    config_path: Path = typer.Argument(
        ...,
        help=(
            "Training configuration file to be visualized. A CO molecule is taken as"
            " sample input."
        ),
    )
):
    """
    Visualize a model based on a configuration file.
    A CO molecule is taken as sample input (influences number of atoms,
    number of species is set to 10).

    Parameters
    ----------
    config_path: Path to the training configuration file.
    """
    from jax_md.partition import space

    from gmnn_jax.config import Config
    from gmnn_jax.model import get_training_model
    from gmnn_jax.utils.data import make_minimal_input
    from gmnn_jax.visualize import model_tabular

    with open(config_path, "r") as stream:
        user_config = yaml.safe_load(stream)

    try:
        config = Config.parse_obj(user_config)
    except ValidationError as e:
        print(e)
        console.print("Configuration Invalid!", style="red3")
        raise typer.Exit(code=1)

    displacement_fn, _ = space.free()
    R, Z, idx = make_minimal_input()

    gmnn = get_training_model(
        n_atoms=2,
        n_species=10,
        displacement_fn=displacement_fn,
        **config.model.get_dict(),
    )
    model_tabular(gmnn, R, Z, idx)


@template_app.command("train")
def template_train_config(
    full: bool = typer.Option(False, help="Use all input options."),
):
    """
    Creates a training input template in the current working directory.
    """
    if full:
        template_file = "train_config_full.yaml"
        config_path = "config_full.yaml"
    else:
        template_file = "train_config_minimal.yaml"
        config_path = "config.yaml"

    template_content = pkg_resources.read_text(templates, template_file)

    if Path(config_path).is_file():
        console.print("There is already a config file in the working directory.")
    else:
        with open(config_path, "w") as config:
            config.write(template_content)


@template_app.command("md")
def template_md_config():
    """
    Creates a training input template in the current working directory.
    """

    template_file = "md_config_minimal.yaml"
    config_path = "md_config.yaml"

    template_content = pkg_resources.read_text(templates, template_file)

    if Path(config_path).is_file():
        console.print("There is already a config file in the working directory.")
    else:
        with open(config_path, "w") as config:
            config.write(template_content)


logo = """
[bold white]  /######  /##      /## /##   /## /##   /##[bold turquoise2]            /#####  /######  /##   /##
[bold white] /##__  ##| ###    /###| ### | ##| ### | ##[bold turquoise2]           |__  ## /##__  ##| ##  / ##
[bold white]| ##  \__/| ####  /####| ####| ##| ####| ##[bold turquoise2]              | ##| ##  \ ##|  ##/ ##/
[bold white]| ## /####| ## ##/## ##| ## ## ##| ## ## ##[bold turquoise2] /######      | ##| ######## \  ####/ 
[bold white]| ##|_  ##| ##  ###| ##| ##  ####| ##  ####[bold turquoise2]|______/ /##  | ##| ##__  ##  >##  ## 
[bold white]| ##  \ ##| ##\  # | ##| ##\  ###| ##\  ###[bold turquoise2]        | ##  | ##| ##  | ## /##/\  ##
[bold white]|  ######/| ## \/  | ##| ## \  ##| ## \  ##[bold turquoise2]        |  ######/| ##  | ##| ##  \ ##
[bold white] \______/ |__/     |__/|__/  \__/|__/  \__/[bold turquoise2]         \______/ |__/  |__/|__/  |__/
"""  # noqa: E501, W605, W291, E261, E303


def version_callback(value: bool) -> None:
    """Get the installed gmnn-jax version."""
    if value:
        console.print(logo)
        console.print(f"gmnn-jax {importlib.metadata.version('gmnn-jax')}")
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
