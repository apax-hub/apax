import os

# from types import UnionType
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


class NHCOptions(BaseModel, extra="forbid"):
    chain_length: PositiveInt = 3
    chain_steps: PositiveInt = 2
    sy_steps: PositiveInt = 3
    tau: PositiveFloat = 100


class Integrator(BaseModel, extra="forbid"):
    dt: PositiveFloat = 0.5  # fs


class NVEOptions(Integrator, extra="forbid"):
    name: Literal["nve"]


class NVTOptions(Integrator, extra="forbid"):
    name: Literal["nvt"]
    temperature: PositiveFloat = 298.15  # K
    thermostat_chain: NHCOptions = NHCOptions()


class NPTOptions(NVTOptions, extra="forbid"):
    name: Literal["npt"]
    pressure: PositiveFloat = 1.01325  # bar
    barostat_chain: NHCOptions = NHCOptions(tau=1000)


class MDConfig(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration for a NHC molecular dynamics simulation.

    Parameters:
    seed: Random seed for momentum initialization.
    temperature: Temperature of the simulation in Kelvin.
    dt: Time step in fs.
    duration: Total simulation time in fs.
    n_inner: Number of compiled simulation steps (i.e. number of iterations of the
        `jax.lax.fori_loop` loop). Also determines atoms buffer size.
    sampling_rate:
        Trajectory dumping interval.
    dr_threshold: Skin of the neighborlist.
    extra_capacity: JaxMD allocates a maximal number of neighbors.
        This argument lets you add additional capacity to avoid recompilation.
        The default is usually fine.
    intitial_structure: Path to the starting structure of the simulation.
    sim_dir: Directory where simulation file will be stored.
    traj_name: Name of the trajectory file.
    restart: Whether the simulation should restart from the latest configuration
        in `traj_name`.
    disable_pbar: Disables the MD progressbar.
    """

    seed: int = 1

    # https://docs.pydantic.dev/latest/usage/types/unions/#discriminated-unions-aka-tagged-unions
    ensemble: Union[NVEOptions, NVTOptions, NPTOptions] = Field(
        NVTOptions(name="nvt"), discriminator="name"
    )

    duration: PositiveFloat
    n_inner: PositiveInt = 100
    sampling_rate: PositiveInt = 10
    dr_threshold: PositiveFloat = 0.5
    extra_capacity: PositiveInt = 0

    initial_structure: str
    load_momenta: bool = False
    sim_dir: str = "."
    traj_name: str = "md.h5"
    restart: bool = True
    disable_pbar: bool = False

    def dump_config(self):
        """
        Writes the current config file to the MD directory.
        """
        with open(os.path.join(self.sim_dir, "md_config.yaml"), "w") as conf:
            yaml.dump(self.model_dump(), conf, default_flow_style=False)
