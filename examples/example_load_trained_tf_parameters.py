from jax.config import config
config.update("jax_enable_x64", True)
import haiku as hk
from ase import Atoms
import numpy as np
from typing import List
from jax_md import partition
from jax_md import partition, space
from jax_md.util import high_precision_sum
import jax
import jax.numpy as jnp
from ase.io import read

from gmnn_jax.utils.weight_transfer import transfer_parameters
from gmnn_jax.model import GMNN


def convert_ase_to_data_dict(atoms_list: List[Atoms]):
    """Converts a list of ASE Atoms objects into a dictionary containing numpy arrays.
    All arrays containing per-atom quantities (e.g. positions, forces) are zero-padded
    to have the same size as the largest structure in the list.
    """
    n_data = len(atoms_list)
    n_maxat = max([len(atoms) for atoms in atoms_list])

    key_map = {"energy": "energy", "forces": "forces", "stress": "stress"}

    data = {
        "numbers": np.zeros((n_data, n_maxat), dtype=np.int64),  # atomic numbers
        "n_atoms": np.zeros((n_data,), dtype=np.int64),  # number of atoms
        "positions": np.zeros(
            (n_data, n_maxat, 3), dtype=np.float64
        ),  # Cartesian coordinates
        "energy": np.zeros((n_data,), dtype=np.float64),  # total energies
        "forces": np.zeros((n_data, n_maxat, 3), dtype=np.float64),  # atomic forces
        "cell": np.zeros((n_data, 3, 3), dtype=np.float64),  # periodic cell
        "stress": np.zeros((n_data, 3, 3), dtype=np.float64),
        "charge": np.zeros((n_data,), dtype=np.float64),
        "dipole": np.zeros((n_data, 6), dtype=np.float64),
        "mat": np.zeros((n_data, 6), dtype=np.float64),
    }

    for ii, atoms in enumerate(atoms_list):
        data["numbers"][ii, : len(atoms)] = atoms.get_atomic_numbers()
        data["n_atoms"][ii] = len(atoms)
        data["positions"][ii, : len(atoms), :] = atoms.get_positions()
        data["cell"][ii] = atoms.get_cell()

        if atoms.calc is not None:
            results = {key_map.get(k, k): v for k, v in atoms.calc.results.items()}
            for k, v in results.items():
                if k == "forces":
                    data["forces"][ii, : len(atoms), :] = v
                else:
                    data[k][ii] = v

    pruned_data = {k: v for k, v in data.items() if np.any(np.abs(v) > 1e-6)}

    return pruned_data


def get_model(
    atomic_numbers,
    units,
    displacement,
    box_size: float=10.0,
    cutoff_distance=6.0,
    n_basis=7,
    n_radial=5,
    dr_threshold=0.5,
    nl_format: partition.NeighborListFormat=partition.Sparse,
    **neighbor_kwargs
    ):
    neighbor_fn = partition.neighbor_list(
        displacement,
        box_size,
        cutoff_distance,
        dr_threshold,
        fractional_coordinates=False,
        format=nl_format,
        **neighbor_kwargs
        )

    n_atoms = atomic_numbers.shape[0]
    Z = jnp.asarray(atomic_numbers)
    n_species = 9#10#jnp.max(Z)
    
    @hk.without_apply_rng
    @hk.transform
    def model(R, neighbor):
        gmnn = GMNN(
            units,
            displacement,
            n_atoms=n_atoms,
            n_basis=n_basis,
            n_radial=n_radial,
            n_species=n_species,
        )
        out = gmnn(R, Z, neighbor)
        return high_precision_sum(out) # jnp.sum(out)

    return neighbor_fn, model.init, model.apply


trained_params = np.load("./etoh_model_params.npz")

# atoms = read("raw_data/ds.extxyz")
atoms = read("ethanol.traj")
# print(len(idx_i))
data = convert_ase_to_data_dict([atoms])
data = {k: jnp.asarray(v[0]) for k,v in data.items()}

# box_size = data["cell"][0,0]
box_size = None
r_cutoff = 6.0
nl_format = partition.Sparse

displacement_fn, shift_fn = space.free()

neighbor_fn, model_init, model = get_model(
    atomic_numbers=data["numbers"],
    units=[512,512],
    displacement=displacement_fn,
    box_size=box_size,
    cutoff_distance=r_cutoff,
    dr_threshold=0.0,
    )
neighbor = neighbor_fn.allocate(data["positions"], extra_capacity=0)

rng_key = jax.random.PRNGKey(42)
params = model_init(rng=rng_key, R=data["positions"], neighbor=neighbor)

transfered_params = transfer_parameters(params, trained_params)

result = model(params=transfered_params, R=data["positions"], neighbor=neighbor) # , data["numbers"]
print(result)

F = jax.grad(model, argnums=1)(transfered_params, data["positions"], neighbor)
print(F)