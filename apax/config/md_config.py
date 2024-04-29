import os

# from types import UnionType
from typing import Literal, Union

import yaml
from pydantic import BaseModel, Field, NonNegativeInt, PositiveFloat, PositiveInt


class NHCOptions(BaseModel, extra="forbid"):
    """
    Options for Nose-Hoover chain thermostat.

    Parameters
    ----------
    chain_length : PositiveInt, default = 3
        Number of thermostats in the chain.
    chain_steps : PositiveInt, default = 2
        Number of steps per chain.
    sy_steps : PositiveInt, default = 3
        Number of steps for Suzuki-Yoshida integration.
    tau : PositiveFloat, default = 100
        Relaxation time parameter.
    """

    chain_length: PositiveInt = 3
    chain_steps: PositiveInt = 2
    sy_steps: PositiveInt = 3
    tau: PositiveFloat = 100


class Integrator(BaseModel, extra="forbid"):
    """
    Molecular dynamics integrator options.

    Parameters
    ----------
    dt : PositiveFloat, default = 0.5
        Time step size in femtoseconds (fs).
    """

    dt: PositiveFloat = 0.5  # fs


class NVEOptions(Integrator, extra="forbid"):
    """
    Options for NVE ensemble simulations.

    Attributes
    ----------
    name : Literal["nve"]
        Name of the ensemble.
    """

    name: Literal["nve"]


class NVTOptions(Integrator, extra="forbid"):
    """
    Options for NVT ensemble simulations.

    Parameters
    ----------
    name : Literal["nvt"]
        Name of the ensemble.
    temperature : PositiveFloat, default = 298.15
        Temperature in Kelvin (K).
    thermostat_chain : NHCOptions, default = NHCOptions()
        Thermostat chain options.
    """

    name: Literal["nvt"]
    temperature: PositiveFloat = 298.15  # K
    thermostat_chain: NHCOptions = NHCOptions()


class NPTOptions(NVTOptions, extra="forbid"):
    """
    Options for NPT ensemble simulations.

    Parameters
    ----------
    name : Literal["npt"]
        Name of the ensemble.
    pressure : PositiveFloat, default = 1.01325
        Pressure in bar.
    barostat_chain : NHCOptions, default = NHCOptions(tau=1000)
        Barostat chain options.
    """

    name: Literal["npt"]
    pressure: PositiveFloat = 1.01325  # bar
    barostat_chain: NHCOptions = NHCOptions(tau=1000)


class MDConfig(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration for a NHC molecular dynamics simulation.
    Full config :ref:`here <md_config>`:

    Parameters
    ----------
    seed : int, default = 1
        | Random seed for momentum initialization.
    temperature : float, default = 298.15
        | Temperature of the simulation in Kelvin.
    dt : float, default = 0.5
        | Time step in fs.
    duration : float, required
        | Total simulation time in fs.
    n_inner : int, default = 100
        | Number of compiled simulation steps (i.e. number of iterations of the
        | `jax.lax.fori_loop` loop). Also determines atoms buffer size.
    sampling_rate : int, default = 10
        | Interval between saving frames.
    buffer_size : int, default = 100
        | Number of collected frames to be dumped at once.
    dr_threshold : float, default = 0.5
        | Skin of the neighborlist.
    extra_capacity : int, default = 0
        | JaxMD allocates a maximal number of neighbors. This argument lets you add
        | additional capacity to avoid recompilation. The default is usually fine.
    initial_structure : str, required
        | Path to the starting structure of the simulation.
    sim_dir : str, default = "."
        | Directory where simulation file will be stored.
    traj_name : str, default = "md.h5"
        | Name of the trajectory file.
    restart : bool, default = True
        | Whether the simulation should restart from the latest configuration in
        | `traj_name`.
    checkpoint_interval : int, default = 50_000
        | Number of time steps between saving full simulation state checkpoints.
        | These will be loaded with the `restart` option.
    disable_pbar : bool, False
        | Disables the MD progressbar.
    """

    seed: int = 1

    # https://docs.pydantic.dev/latest/usage/types/unions/#discriminated-unions-aka-tagged-unions
    ensemble: Union[NVEOptions, NVTOptions, NPTOptions] = Field(
        NVTOptions(name="nvt"), discriminator="name"
    )

    duration: PositiveFloat
    n_inner: PositiveInt = 100
    sampling_rate: PositiveInt = 10
    buffer_size: PositiveInt = 100
    dr_threshold: PositiveFloat = 0.5
    extra_capacity: NonNegativeInt = 0

    initial_structure: str
    load_momenta: bool = False
    sim_dir: str = "."
    traj_name: str = "md.h5"
    restart: bool = True
    checkpoint_interval: int = 50_000
    disable_pbar: bool = False

    def dump_config(self):
        """
        Writes the current config file to the MD directory.

        """
        with open(os.path.join(self.sim_dir, "md_config.yaml"), "w") as conf:
            yaml.dump(self.model_dump(), conf, default_flow_style=False)
