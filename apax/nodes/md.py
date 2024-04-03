import functools
import logging
import pathlib
import typing

import ase.io
import h5py
import yaml
import znh5md
import zntrack.utils
from zntrack import dvc, zn

from .model import Apax
from .utils import check_duplicate_keys

log = logging.getLogger(__name__)


class ApaxJaxMD(zntrack.Node):
    """Class to run a more performant JaxMD simulation with a apax Model.

    Attributes
    ----------
    model: ApaxModel
        model to use for the simulation
    repeat: float
        number of repeats
    md_parameter: dict
        parameter for the MD simulation
    md_parameter_file: str
        path to the MD simulation parameter file
    """

    model: Apax = zntrack.deps()
    repeat = zn.params(None)

    md_parameter: dict = zn.params(None)
    md_parameter_file: str = dvc.params(None)

    sim_dir: pathlib.Path = dvc.outs(zntrack.nwd / "md")
    init_struc_dir: pathlib.Path = dvc.outs(zntrack.nwd / "initial_structure.extxyz")

    # def post_init(self):
    #     if not self.state.loaded:
    #         dict_supplied = self.md_parameter is None
    #         file_supplied = self.md_parameter_file is None

    #         if dict_supplied and file_supplied:
    #             raise TypeError("Please specify either an input dict or a file")
    #         elif not dict_supplied and not file_supplied:
    #             raise TypeError(
    #                 "Can not train apax model without a parameter dict or file"
    #             )
    #         else:
    #             log.info(
    #                 "Please keep track of the parameter file with git, just like the"
    #                 " params.yaml"
    #             )

    #     if not isinstance(self.model, Apax):
    #         raise TypeError(
    #             "Performing simulations with JaxMD requires a apax model Node"
    #         )

    #     self.data = utils.helpers.get_deps_if_node(self.data, "atoms")

    def _post_load_(self) -> None:
        self._handle_parameter_file()

    def _handle_parameter_file(self):
        if self.md_parameter_file:
            md_parameter_file_content = pathlib.Path(self.md_parameter_file).read_text()
            self.md_parameter = yaml.safe_load(md_parameter_file_content)

        custom_parameters = {
            "sim_dir": self.sim_dir.as_posix(),
            "initial_structure": self.init_struc_dir.as_posix(),
        }
        check_duplicate_keys(custom_parameters, self.md_parameter, log)
        self.md_parameter.update(custom_parameters)

    def run(self):
        """Primary method to run which executes all steps of the model training"""
        from apax.md.nvt import run_md

        # self._handle_parameter_file()
        atoms = self.get_data()
        if self.repeat is not None:
            atoms = atoms.repeat(self.repeat)
        ase.io.write(self.init_struc_dir.as_posix(), atoms)

        self.model._handle_parameter_file()
        run_md(self.model._parameter, self.md_parameter)

    @functools.cached_property
    def atoms(self) -> typing.List[ase.Atoms]:
        # filename should be changeable
        def file_handle(filename):
            file = self.state.fs.open(filename, "rb")
            return h5py.File(file)

        return znh5md.ASEH5MD(
            self.sim_dir / "md.h5",
            format_handler=functools.partial(
                znh5md.FormatHandler, file_handle=file_handle
            ),
        ).get_atoms_list()
