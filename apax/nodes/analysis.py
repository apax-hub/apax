import logging

import zntrack.utils
from ipsuite import base

from .model import Apax

log = logging.getLogger(__name__)


class ApaxBatchPrediction(base.ProcessAtoms):
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

    _module_ = "apax.nodes"

    model: Apax = zntrack.deps()
    batch_size: int = zntrack.params(1)

    def run(self):
        self.atoms = []
        calc = self.model.get_calculator()
        data = self.get_data()
        self.atoms = calc.batch_eval(data, self.batch_size)
