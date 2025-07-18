import logging
import os
import pathlib
import typing

import ase
import h5py
import jax
import yaml
import znh5md
import zntrack

from apax.md.simulate import run_md
from apax.nodes.model import ApaxBase
from apax.nodes.utils import check_duplicate_keys

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
    repeat: None|int|tuple[int, int, int]
        number of repeats
    config: str
        path to the MD simulation parameter file
    """

    data: list[ase.Atoms] = zntrack.deps()
    data_id: int = zntrack.params(-1)

    model: ApaxBase = zntrack.deps()
    repeat: None | int | tuple[int, int, int] = zntrack.params(None)

    config: str = zntrack.params_path(None)

    sim_dir: pathlib.Path = zntrack.outs_path(zntrack.nwd / "md")
    init_struc_dir: pathlib.Path = zntrack.outs_path(zntrack.nwd / "initial_structure.h5")

    @property
    def parameter(self) -> dict:
        with self.state.fs.open(self.config, "r") as f:
            parameter = yaml.safe_load(f)

        custom_parameters = {
            "sim_dir": self.sim_dir.as_posix(),
            "initial_structure": self.init_struc_dir.as_posix(),
        }
        check_duplicate_keys(custom_parameters, parameter, log)
        parameter.update(custom_parameters)

        return parameter

    def _write_initial_structure(self):
        atoms = self.data[self.data_id]
        if self.repeat is not None:
            atoms = atoms.repeat(self.repeat)
        db = znh5md.IO(self.init_struc_dir.as_posix())
        db.extend([atoms])

    def run(self):
        """Primary method to run which executes all steps of the model training"""
        if not self.state.restarted:
            self._write_initial_structure()

        run_md(self.model.parameter, self.parameter, log_level="info")

    def map(self):
        os.environ["JAX_COMPILATION_CACHE_DIR"] = '"/tmp/jax_cache"'

        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )

        from jax.experimental.compilation_cache import compilation_cache as cc

        cc.set_cache_dir("/tmp/jax_cache")

        """Primary method to run which executes all steps of the model training"""
        for id in range(len(self.data)):
            self.data_id = id
            if not self.state.restarted:
                self._write_initial_structure()

            run_md(self.model.parameter, self.parameter, log_level="info")

    @property
    def frames(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.sim_dir / "md.h5", "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]
