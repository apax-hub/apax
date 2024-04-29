import functools
import logging
import pathlib
import typing

import ase.io
import h5py
import yaml
import znh5md
import zntrack.utils

from apax.md.nvt import run_md

from .model import Apax
from .utils import check_duplicate_keys

log = logging.getLogger(__name__)


class ApaxJaxMD(zntrack.Node):
    """Class to run a more performant JaxMD simulation with a apax Model.

    Parameters
    ----------
    data: list[ase.Atoms]
        MD starting structure
    data_id: int, default = -1
        index of the configuration from the data list to use
    model: ApaxModel
        model to use for the simulation
    repeat: float
        number of repeats
    config: str
        path to the MD simulation parameter file
    """

    data: list[ase.Atoms] = zntrack.deps()
    data_id: int = zntrack.params(-1)

    model: Apax = zntrack.deps()
    repeat = zntrack.params(None)

    config: str = zntrack.params_path(None)

    sim_dir: pathlib.Path = zntrack.outs_path(zntrack.nwd / "md")
    init_struc_dir: pathlib.Path = zntrack.outs_path(
        zntrack.nwd / "initial_structure.extxyz"
    )

    _parameter: dict = None

    def _post_load_(self) -> None:
        self._handle_parameter_file()

    def _handle_parameter_file(self):
        with self.state.use_tmp_path():
            self._parameter = yaml.safe_load(pathlib.Path(self.config).read_text())

        custom_parameters = {
            "sim_dir": self.sim_dir.as_posix(),
            "initial_structure": self.init_struc_dir.as_posix(),
        }
        check_duplicate_keys(custom_parameters, self._parameter, log)
        self._parameter.update(custom_parameters)

    def run(self):
        """Primary method to run which executes all steps of the model training"""

        atoms = self.data[self.data_id]
        if self.repeat is not None:
            atoms = atoms.repeat(self.repeat)
        ase.io.write(self.init_struc_dir.as_posix(), atoms)

        run_md(self.model._parameter, self._parameter)

    @functools.cached_property
    def atoms(self) -> typing.List[ase.Atoms]:
        def file_handle(filename):
            file = self.state.fs.open(filename, "rb")
            return h5py.File(file)

        return znh5md.ASEH5MD(
            self.sim_dir / "md.h5",
            format_handler=functools.partial(
                znh5md.FormatHandler, file_handle=file_handle
            ),
        ).get_atoms_list()
