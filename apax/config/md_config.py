import os
from typing import Literal, Union

import yaml
from pydantic import BaseModel, Field, NonNegativeInt, PositiveFloat, PositiveInt
from typing_extensions import Annotated

from apax.utils.helpers import APAX_PROPERTIES


class ConstantTempSchedule(BaseModel, extra="forbid"):
    """Constant temperature schedule.

    Attributes
    ----------
    name: str
        Identifier of the temperature schedule.
    T0 : PositiveFloat, default = 298.15
        Initial temperature in Kelvin (K).
    """

    name: Literal["constant"] = "constant"
    T0: PositiveFloat = 298.15  # K

    def get_schedule(self):
        from apax.md.schedules import ConstantTSchedule

        return ConstantTSchedule(self.T0)


class PiecewiseLinearTempSchedule(ConstantTempSchedule, extra="forbid"):
    """Piecewise linear temperature schedule.
    Temperature is linearly interpolated between T0 and the supplied
    values at the specified time steps.

    Attributes
    ----------
    temperatures: list[PositiveFloat]
        List of temperatures to interpolate between.
    durations: list[PositiveInt]
        Duration (in time steps) of the interpolation between two
        subsequent values of `temperatures`.
    """

    name: Literal["piecewise"] = "piecewise"
    temperatures: list[PositiveFloat]
    durations: list[PositiveInt]

    def get_schedule(self):
        from apax.md.schedules import PieceWiseLinearTSchedule

        schedule = PieceWiseLinearTSchedule(
            self.T0,
            self.temperatures,
            self.durations,
        )
        return schedule


class OscillatingRampTempSchedule(ConstantTempSchedule, extra="forbid"):
    """Combination of a linear interpolation between T0 and Tend and a temperature oscillation.
    Mostly for sampling purposes.

    Attributes
    ----------
    Tend: PositiveFloat
        Final temperature in Kelvin.
    amplitude: PositiveFloat
        Amplitude of temperature oscilaltions.
    num_oscillations: PositiveInt
        Number of oscillations to occur during the simulation.
    total_steps: PositiveInt
        Total steps of the schedule. Afterwards, Tend will be kept.
    """

    name: Literal["oscillating_ramp"] = "oscillating_ramp"
    Tend: PositiveFloat
    amplitude: PositiveFloat
    num_oscillations: PositiveInt
    total_steps: PositiveInt

    def get_schedule(self):
        from apax.md.schedules import OscillatingRampTSchedule

        schedule = OscillatingRampTSchedule(
            self.T0,
            self.Tend,
            self.amplitude,
            self.num_oscillations,
            self.total_steps,
        )
        return schedule


TemperatureSchedule = Union[
    ConstantTempSchedule, PiecewiseLinearTempSchedule, OscillatingRampTempSchedule
]


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
    name : str
        Name of the ensemble.
    dt: PositiveFloat, default = 0.5
        Time step in femto seconds.
    temperature_schedule: TemperatureSchedule
        Temperature schedule to use throughout the simulation.
        For NVE, it is only used for velocity initialization and
        disregarded at subsequent steps.
    """

    name: str
    dt: PositiveFloat = 0.5  # fs
    temperature_schedule: TemperatureSchedule = Field(
        ConstantTempSchedule(name="constant", T0=298.15), discriminator="name"
    )


class NVEOptions(Integrator, extra="forbid"):
    """
    Options for NVE ensemble simulations.

    Attributes
    ----------
    name : Literal["nve"]
        Name of the ensemble.
    """

    name: Literal["nve"]


class NVTOptions(NVEOptions, extra="forbid"):
    """
    Options for NVT ensemble simulations.

    Parameters
    ----------
    name : Literal["nvt"]
        Name of the ensemble.
    thermostat_chain : NHCOptions, default = NHCOptions()
        Thermostat chain options.
    """

    name: Literal["nvt"]
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


class EnergyUncertaintyCheck(BaseModel, extra="forbid"):
    name: Literal["energy_uncertainty"] = "energy_uncertainty"
    threshold: PositiveFloat
    per_atom: bool = True


class ForcesUncertaintyCheck(BaseModel, extra="forbid"):
    name: Literal["forces_uncertainty"] = "forces_uncertainty"
    threshold: PositiveFloat


DynamicsCheck = Annotated[
    Union[EnergyUncertaintyCheck, ForcesUncertaintyCheck], Field(discriminator="name")
]


class FixAtomsConstraint(BaseModel, extra="forbid"):
    name: Literal["fixatoms"] = "fixatoms"
    indices: list[int]


Constraint = Annotated[Union[FixAtomsConstraint], Field(discriminator="name")]


class MDConfig(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration for a NHC molecular dynamics simulation.
    Full config :ref:`here <md_config>`:

    Parameters
    ----------
    seed : int, default = 1
        | Random seed for momentum initialization.
    ensemle :
        | Options for integrating the EoM and which ensemble to use.
    dt : float, default = 0.5
        | Time step in fs.
    duration : float, required
        | Total simulation time in fs.
    n_inner : int, default = 500
        | Number of compiled simulation steps (i.e. number of iterations of the
        | `jax.lax.fori_loop` loop). Also determines atoms buffer size.
    sampling_rate : int, default = 10
        | Interval between saving frames.
    buffer_size : int, default = 2500
        | Number of collected frames to be dumped at once.
    dr_threshold : float, default = 0.5
        | Skin of the neighborlist.
    extra_capacity : int, default = 0
        | JaxMD allocates a maximal number of neighbors. This argument lets you add
        | additional capacity to avoid recompilation. The default is usually fine.

    dynamics_checks: list[DynamicsCheck]
        | List of termination criteria. Currently energy and force uncertainty
        | are available
    properties: list[str]
        | Whitelist of properties to be saved in the trajectory.
        | This does not effect what the model will calculate, e.g..
        | an ensemble will still calculate uncertainties.
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
    n_inner: PositiveInt = 500
    sampling_rate: PositiveInt = 10
    buffer_size: PositiveInt = 2500
    dr_threshold: PositiveFloat = 0.5
    extra_capacity: NonNegativeInt = 0

    dynamics_checks: list[DynamicsCheck] = []
    constraints: list[Constraint] = []

    properties: list[str] = APAX_PROPERTIES

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
