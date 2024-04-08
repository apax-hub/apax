import functools
import logging

import ase.io
import zntrack


class AddData(zntrack.Node):
    file: str = zntrack.deps_path()

    def run(self):
        pass

    @functools.cached_property
    def atoms(self) -> list[ase.Atoms]:
        data = []
        for atoms in ase.io.iread(self.file):
            data.append(atoms)
            if len(data) == 50:
                return data


def check_duplicate_keys(dict_a: dict, dict_b: dict, log: logging.Logger) -> None:
    """Check if a key of dict_a is present in dict_b and then log a warning."""
    for key in dict_a:
        if key in dict_b:
            log.warning(
                f"Found <{key}> in given config file. Please be aware that <{key}>"
                " will be overwritten by MLSuite!"
            )
