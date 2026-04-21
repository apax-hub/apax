"""MACE foundation model loading and conversion.

Functions
---------
run_conversion
    Entry point for ``apax convert-mace``. Accepts a canonical MACE model
    name or a path to a local ``.model`` file.
load_mace_foundation
    Runtime loader for apax-native converted directories.
"""
from __future__ import annotations

from pathlib import Path


def run_conversion(
    source, dst: Path, *, head: str = "mp", family: str = "mace_mp"
) -> None:
    """Convert a torch-mace checkpoint (lazy torch import). Filled in by P3.2."""
    raise NotImplementedError("Filled in by P3.2")


def load_mace_foundation(source):
    """Load a converted apax-native MACE directory. Filled in by P3.3."""
    raise NotImplementedError("Filled in by P3.3")
