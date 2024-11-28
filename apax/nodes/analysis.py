import logging
import pathlib

import ase
import h5py
import znh5md
import zntrack

from apax.nodes.model import Apax

log = logging.getLogger(__name__)


class ApaxBatchPrediction(zntrack.Node):
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

    data: list[ase.Atoms] = zntrack.deps()

    model: Apax = zntrack.deps()
    batch_size: int = zntrack.params(1)
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.h5")

    def run(self):
        calc = self.model.get_calculator()
        frames = calc.batch_eval(self.data, self.batch_size)
        znh5md.write(self.frames_path, frames)

    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f, "r") as h5:
                return znh5md.IO(file_handle=h5)[:]
