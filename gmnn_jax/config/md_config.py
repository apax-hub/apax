import os

import yaml
from pydantic import BaseModel, Extra, PositiveFloat, PositiveInt


class MDConfig(BaseModel, frozen=True, extra=Extra.forbid):
    seed: int = 1

    temperature: PositiveFloat
    dt: PositiveFloat = 0.5
    n_steps: PositiveInt
    n_inner: PositiveInt = 4
    dr_threshold: PositiveFloat = 0.5
    extra_capacity: PositiveInt = 0

    initial_structure: str
    sim_dir: str = "."
    traj_name: str = "md.traj"
    restart: bool = True
    disable_pbar: bool = False

    def dump_config(self):
        with open(os.path.join(self.sim_dir, "md_config.yaml"), "w") as conf:
            yaml.dump(self.dict(), conf, default_flow_style=False)
