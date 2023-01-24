import os

import yaml
from pydantic import BaseModel, Extra, PositiveFloat, PositiveInt


class MDConfig(BaseModel, frozen=True, extra=Extra.forbid):
    """
    Configuration for a NHC molecular dynamics simulation.

    Parameters:
    seed: Random seed for momentum initialization.
    temperature: Temperature of the simulation in Kelvin.
    dt: Time step in fs.
    n_steps: Total number of simulation steps. Will be replaced with
        simulation time in the future.
    n_inner: Number of compiled simulation steps (i.e. number of iterations of the
        `jax.lax.fori_loop` loop). Currently also determines sampling interval.
    dr_threshold: Skin of the neighborlist.
    extra_capacity: JaxMD allocates a maximal number of neighbors.
        This argument lets you add additional capacity to avoid recompilation.
        The default is usually fine.
    enable_fp64: Enables the JAX fp64 configuration.
    intitial_structure: Path to the starting structure of the simulation.
    sim_dir: Directory where simulation file will be stored.
    traj_name: Name of the trajectory file.
    restart: Whether the simulation should restart from the latest configuration
        in `traj_name`.
    disable_pbar: Disables the MD progressbar.
    """

    seed: int = 1

    temperature: PositiveFloat
    dt: PositiveFloat = 0.5
    n_steps: PositiveInt
    n_inner: PositiveInt = 4
    dr_threshold: PositiveFloat = 0.5
    extra_capacity: PositiveInt = 0
    enable_fp64: bool = True

    initial_structure: str
    sim_dir: str = "."
    traj_name: str = "md.traj"
    restart: bool = True
    disable_pbar: bool = False

    def dump_config(self):
        """
        Writes the current config file to the MD directory.
        """
        with open(os.path.join(self.sim_dir, "md_config.yaml"), "w") as conf:
            yaml.dump(self.dict(), conf, default_flow_style=False)
