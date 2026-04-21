"""CLI subcommand: convert torch-mace foundation models into apax-native directories.

Notes
-----
``torch`` and ``mace-torch`` are imported lazily inside ``convert_mace``. If
either is missing, the command exits with a typer error and a hint to install
the ``mace-convert`` dependency group.
"""
from __future__ import annotations

from pathlib import Path

import typer


def convert_mace(
    source: str = typer.Argument(
        ...,
        help=(
            "Canonical MACE foundation name (e.g. 'medium-mpa-0', 'medium', "
            "'large') or path to a local torch .model file."
        ),
    ),
    dst: Path = typer.Argument(..., help="Output apax-native directory"),
    head: str = typer.Option("mp", help="Which head to select for multi-head models"),
    family: str = typer.Option(
        "mace_mp",
        help="Foundation-model family: 'mace_mp' covers MPA-0 and MP-0/0b/0b2/0b3. "
             "Others (mace_off, mace_anicc) are deferred.",
    ),
) -> None:
    """Convert a torch-mace foundation model into an apax-native directory.

    Parameters
    ----------
    source
        Canonical MACE foundation name, or a filesystem path to a local
        ``.model`` file produced by torch-mace.
    dst
        Output directory. Will be created.
    head
        For multi-head foundation models (e.g. MPA), the head to retain.
    family
        Foundation-family resolver. Initial scope: ``"mace_mp"``.
    """
    try:
        import torch  # noqa: F401
        import mace   # noqa: F401
    except ImportError as e:
        raise typer.BadParameter(
            "Converting MACE foundation models requires torch and mace-torch. "
            "Install them with `uv sync --group mace-convert --extra mace`. "
            f"Missing module: {e.name}"
        ) from None

    from apax.transfer_learning.mace_foundation import run_conversion

    run_conversion(source, dst, head=head, family=family)
