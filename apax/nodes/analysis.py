import logging
import pathlib

import ase.io
import znflow
import zntrack.utils
from ipsuite import fields

from .model import Apax

log = logging.getLogger(__name__)


class ProcessAtoms(zntrack.Node):
    """Protocol for objects that process atoms.

    Attributes
    ----------
    data: list[ase.Atoms]
        The atoms data to process. This must be an input to the Node
    data_file: str | None
        The path to the file containing the atoms data. This is an
        alternative to 'data' and can be used to load the data from
        a file. If both are given, 'data' is used. Set 'data' to None
        if you want to use 'data_file'.
    atoms: list[ase.Atoms]
        The processed atoms data. This is an output of the Node.
        It does not have to be 'field.Atoms' but can also be e.g. a 'property'.
    """

    data: list[ase.Atoms] = zntrack.deps()
    data_file: str = zntrack.dvc.deps(None)
    atoms: list[ase.Atoms] = fields.Atoms()

    def _post_init_(self):
        if self.data is not None:
            self.data = znflow.combine(self.data, attribute="atoms")

    def update_data(self):
        """Update the data attribute."""
        if self.data is None:
            self.data = self.get_data()

    def get_data(self) -> list[ase.Atoms]:
        """Get the atoms data to process."""
        if self.data is not None:
            return self.data
        elif self.data_file is not None:
            try:
                with self.state.fs.open(pathlib.Path(self.data_file).as_posix()) as f:
                    return list(ase.io.iread(f))
            except FileNotFoundError:
                # File can not be opened with DVCFileSystem, try normal open
                return list(ase.io.iread(self.data_file))
        else:
            raise ValueError("No data given.")


class ApaxBatchPrediction(ProcessAtoms):
    """Create and Save the predictions from model on atoms.

    Attributes
    ----------
    model: Apax
        The Apax model node that implements the 'predict' method
    data: list[Atoms
        Atoms to predict properties for
    batch_size: int
        Number of structures to process at once.
        Descriptor fp64 recommended for large batch sizes

    predictions: list[Atoms] the atoms that have the predicted properties from model
    """

    model: Apax = zntrack.deps()
    batch_size: int = zntrack.params(1)

    def run(self):
        self.atoms = []
        calc = self.model.get_calculator()
        data = self.get_data()
        self.atoms = calc.batch_eval(data, self.batch_size)
