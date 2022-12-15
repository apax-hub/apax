import pathlib
from gmnn_jax.md import run_md
from gmnn_jax.config import Config, MDConfig
from ase.io import read, write
import yaml
from jax_md import partition, space
import jax.numpy as jnp
from flax.training import checkpoints
import jax
import numpy as np
from ase import Atoms

from gmnn_jax.model.gmnn import get_training_model

TEST_PATH = pathlib.Path(__file__).parent.resolve()

def test_run_md():
    model_confg_path = TEST_PATH / "config.yaml"
    md_confg_path = TEST_PATH / "md_config.yaml"
    
    with open(model_confg_path.as_posix(), "r") as stream:
        model_config = yaml.safe_load(stream)    
    with open(md_confg_path.as_posix(), "r") as stream:
        md_config = yaml.safe_load(stream)

    model_config = Config.parse_obj(model_config)
    md_config = MDConfig.parse_obj(md_config)

    cell_size = 10.
    positions = np.array([
        [1.0,0.0,0.0],
        [0.0,1.0,0.0],
        [0.0,0.0,1.0],
    ])
    atomic_numbers = np.array([1,1,8])
    cell = np.diag([cell_size]*3)
    atoms = Atoms(atomic_numbers, positions, cell=cell)
    write("atoms.extxyz", atoms)

    n_atoms = 3
    n_species = int(np.max(atomic_numbers) + 1)

    displacement_fn, _ = space.periodic(cell_size)

    neighbor_fn = partition.neighbor_list(
        displacement_or_metric=displacement_fn,
        box=10.,
        r_cutoff=model_config.model.r_max,
        format=partition.Sparse,
    )
    neighbors = neighbor_fn.allocate(positions)

    model_init, _ = get_training_model(
        n_atoms=n_atoms,
        n_species=n_species,
        displacement_fn=displacement_fn,
        **model_config.model.dict()
    )
    rng_key = jax.random.PRNGKey(model_config.seed)
    params = model_init(rng_key, positions, atomic_numbers, neighbors.idx)
    ckpt = {"model": {"params": params}, "epoch": 0}
    checkpoints.save_checkpoint(
        ckpt_dir=model_config.data.model_path,
        target=ckpt,
        step=0,
        overwrite=True,
    )

    run_md(model_confg_path.as_posix(), md_confg_path.as_posix())

    traj = read(md_config.sim_dir + "/" + md_config.traj_name, index=":")
    n_outer = int(md_config.n_steps // md_config.n_inner)
    assert len(traj) == n_outer
