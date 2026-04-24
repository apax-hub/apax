import importlib.resources as pkg_resources
import json
import sys
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console

from apax.cli import templates

console = Console(highlight=False)

_TYPER_OPTS = dict(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app = typer.Typer(**_TYPER_OPTS)
validate_app = typer.Typer(**_TYPER_OPTS, help="Validate training or MD config files.")
template_app = typer.Typer(**_TYPER_OPTS, help="Create configuration file templates.")
schema_app = typer.Typer(**_TYPER_OPTS, help="Generate JSON schemata for train/md configs.")
app.add_typer(validate_app, name="validate")
app.add_typer(template_app, name="template")
app.add_typer(schema_app, name="schema")


@app.command()
def train(
    train_config_path: Path = typer.Argument(
        ..., help="Training configuration YAML file."
    ),
    log_level: str = typer.Option("info", help="Sets the training logging level."),
):
    """
    Starts the training of a model with parameters provided by a configuration file.
    """
    from apax.train.run import run

    run(train_config_path, log_level)


@app.command()
def md(
    train_config_path: Path = typer.Argument(
        ..., help="Configuration YAML file that was used to train a model."
    ),
    md_config_path: Path = typer.Argument(..., help="MD configuration YAML file."),
    log_level: str = typer.Option("info", help="Sets the training logging level."),
):
    """
    Starts performing a molecular dynamics simulation (currently only NHC thermostat)
    with parameters provided by a configuration file.
    """
    from apax.md import run_md

    run_md(train_config_path, md_config_path, log_level)


@app.command()
def eval(
    train_config_path: Path = typer.Argument(
        ..., help="Configuration YAML file that was used to train a model."
    ),
    n_data: int = typer.Option(
        -1,
        help=(
            "Number of test structures. (All structures are selected by not specifying"
            " it) Gets ignored if test_data_path is specified"
        ),
    ),
):
    """
    Starts performing the evaluation of the test dataset
    with parameters provided by a configuration file.
    """
    from apax.train.eval import eval_model

    eval_model(train_config_path, n_data)


@app.command()
def docs():
    """Opens the documentation website in your browser."""
    console.print("Opening apax's docs at https://apax.readthedocs.io/en/latest/")
    typer.launch("https://apax.readthedocs.io/en/latest/")


# --- Schema commands ---


def _handle_schema(config_cls, section, keywords, flat):
    """Shared logic for schema train/md commands."""
    if flat:
        from apax.config.flat_schema import print_flat

        print_flat(config_cls)
        return
    from apax.config.schema_navigation import filter_schema, print_keywords

    schema = config_cls.model_json_schema()
    if keywords:
        error = print_keywords(schema, section)
        if error:
            console.print(error, style="red3")
            raise typer.Exit(code=1)
        return
    if section:
        node, error = filter_schema(schema, section)
        if error:
            console.print(error, style="red3")
            raise typer.Exit(code=1)
        schema = node
    print(json.dumps(schema, indent=2))


@schema_app.command("train")
def schema_train(
    section: str = typer.Argument(
        None,
        help=(
            "Dotted path to filter (e.g. 'model', 'model.gmnn',"
            " 'model.gmnn.basis', 'optimizer'). Omit for the full schema."
        ),
    ),
    keywords: bool = typer.Option(
        False, "--keywords",
        help="Only list navigable subsection names at the given path.",
    ),
    flat: bool = typer.Option(
        False, "--flat",
        help="List all parameters as a flat table with path, type, default, and description.",
    ),
):
    """Print the training config JSON schema to stdout."""
    from apax.config import Config

    _handle_schema(Config, section, keywords, flat)


@schema_app.command("md")
def schema_md(
    section: str = typer.Argument(
        None,
        help=(
            "Dotted path to filter (e.g. 'ensemble', 'ensemble.nvt',"
            " 'constraints'). Omit for the full schema."
        ),
    ),
    keywords: bool = typer.Option(
        False, "--keywords",
        help="Only list navigable subsection names at the given path.",
    ),
    flat: bool = typer.Option(
        False, "--flat",
        help="List all parameters as a flat table with path, type, default, and description.",
    ),
):
    """Print the MD config JSON schema to stdout."""
    from apax.config import MDConfig

    _handle_schema(MDConfig, section, keywords, flat)


@schema_app.command("vscode")
def schema_vscode():
    """Generate JSON schema files under .vscode/ for YAML autocompletion."""
    console.print("Generating JSON schema")
    from apax.config import Config, MDConfig

    vscode_path = Path(".vscode")
    vscode_path.mkdir(exist_ok=True)

    settings_path = vscode_path / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    settings.setdefault("yaml.schemas", {})

    for name, cls, globs in [
        ("apaxtrain", Config, ["train*.yaml"]),
        ("apaxmd", MDConfig, ["md*.yaml"]),
    ]:
        schema_path = vscode_path / f"{name}.schema.json"
        settings["yaml.schemas"][schema_path.resolve().as_posix()] = globs
        schema_path.write_text(json.dumps(cls.model_json_schema(), indent=2))

    settings_path.write_text(json.dumps(settings, indent=2))


# --- Validation commands ---


def _format_errors(e):
    """Format pydantic ValidationError into readable lines."""
    parts = []
    for error in e.errors():
        loc = ".".join(str(x) for x in error["loc"])
        msg = f"{loc}\n  {error['msg']}\n  input_type: {type(error['input']).__name__}"
        if type(error["input"]) not in (dict, list):
            msg += f"\n  input: {error['input']}"
        parts.append(msg + "\n")
    return "".join(parts)


def _validate_config(config_cls, config_path, user_config, label):
    """Validate a config dict and print result."""
    try:
        config_cls.model_validate(user_config)
    except ValidationError as e:
        print(f"{e.error_count()} validation errors for config")
        print(_format_errors(e))
        console.print("Configuration Invalid!", style="red3")
        raise typer.Exit(code=1)
    console.print("Success!", style="green3")
    console.print(f"{config_path} is a valid {label} config.")


_TEMPLATE_DATA_DEFAULTS = {
    "directory": "/tmp/apax_validate",
    "experiment": "placeholder",
    "data_path": "/tmp/apax_validate/placeholder.extxyz",
}


@validate_app.command("train")
def validate_train_config(
    config_path: Path = typer.Argument(
        ..., help="Configuration YAML file to be validated."
    ),
    template: bool = typer.Option(
        False,
        "--template",
        help=(
            "Validate as a zntrack/ipsuite template. Fills in dummy values for"
            " directory, experiment, and data_path so configs that leave these"
            " to be set at runtime can still be validated."
        ),
    ),
):
    """Validates a training configuration file."""
    from apax.config import Config

    with open(config_path, "r") as stream:
        user_config = yaml.safe_load(stream)

    if template:
        data = user_config.setdefault("data", {})
        for key, default in _TEMPLATE_DATA_DEFAULTS.items():
            if key not in data or data[key] is None:
                data[key] = default

    _validate_config(Config, config_path, user_config, "training")


@validate_app.command("md")
def validate_md_config(
    config_path: Path = typer.Argument(
        ..., help="Configuration YAML file to be validated."
    ),
):
    """Validates a molecular dynamics configuration file."""
    from apax.config import MDConfig

    with open(config_path, "r") as stream:
        user_config = yaml.safe_load(stream)

    _validate_config(MDConfig, config_path, user_config, "MD")


# --- Other commands ---


@app.command("visualize")
def visualize_model(
    config_path: Path = typer.Argument(
        ...,
        help=(
            "Training configuration file to be visualized. A CO molecule is taken as"
            " sample input."
        ),
    ),
):
    """
    Visualize a model based on a configuration file.
    A CO molecule is taken as sample input (influences number of atoms,
    number of species is set to 10).
    """
    import jax

    from apax.config import Config
    from apax.utils.data import make_minimal_input

    with open(config_path, "r") as stream:
        user_config = yaml.safe_load(stream)

    try:
        config = Config.model_validate(user_config)
    except ValidationError as e:
        print(e)
        console.print("Configuration Invalid!", style="red3")
        raise typer.Exit(code=1)

    R, Z, idx, box, offsets = make_minimal_input()
    Builder = config.model.get_builder()
    builder = Builder(config.model.model_dump(), n_species=10)
    model = builder.build_energy_model()
    print(model.tabulate(jax.random.PRNGKey(0), R, Z, idx, box, offsets))


def _write_template(template_file, config_path):
    """Write a template file, exiting if one already exists."""
    if Path(config_path).is_file():
        console.print("There is already a config file in the working directory.")
        sys.exit(1)
    content = pkg_resources.read_text(templates, template_file)
    with open(config_path, "w") as f:
        f.write(content)


@template_app.command("train")
def template_train_config(
    full: bool = typer.Option(False, help="Use all input options."),
):
    """Creates a training input template in the current working directory."""
    if full:
        _write_template("train_config_full.yaml", "config_full.yaml")
    else:
        _write_template("train_config_minimal.yaml", "config.yaml")


@template_app.command("md")
def template_md_config():
    """Creates an MD input template in the current working directory."""
    _write_template("md_config_minimal.yaml", "md_config.yaml")


def version_callback(value: bool) -> None:
    """Get the installed apax version."""
    if value:
        from apax import __version__

        console.print(f"apax {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-V", callback=version_callback, is_eager=True
    ),
):
    _ = version
