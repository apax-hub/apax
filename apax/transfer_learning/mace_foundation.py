"""MACE foundation model loading and conversion utilities."""
from __future__ import annotations

from pathlib import Path


def run_conversion(
    source: str | Path, dst: Path, *, head: str = "mp", family: str = "mace_mp"
) -> None:
    """Convert a torch-mace checkpoint (lazy torch import). Filled in by P3.2."""
    raise NotImplementedError("Filled in by P3.2")


def load_mace_foundation(source: str | Path):
    """Load a converted apax-native MACE directory. Filled in by P3.3."""
    raise NotImplementedError("Filled in by P3.3")
