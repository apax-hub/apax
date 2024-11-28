import logging

import ase.io
import zntrack


class AddData(zntrack.Node):
    file: str = zntrack.deps_path()

    def run(self):
        pass

    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.file, "r") as f:
            return list(ase.io.iread(f))


def check_duplicate_keys(dict_a: dict, dict_b: dict, log: logging.Logger) -> None:
    """Check if a key of dict_a is present in dict_b and then log a warning."""
    for key in dict_a:
        if key in dict_b:
            log.warning(
                f"Found <{key}> in given config file. Please be aware that <{key}>"
                " will be overwritten by MLSuite!"
            )
