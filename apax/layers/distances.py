import jax.numpy as jnp
import numpy as np
from jax import vmap

from apax.utils.jax_md_reduced import partition, space


def canonicalize_neighbors(neighbor):
    return neighbor.idx if isinstance(neighbor, partition.NeighborList) else neighbor


def disp_fn(ri, rj, perturbation, box):
    dR = space.pairwise_displacement(ri, rj)
    dR = space.transform(box, dR)

    if perturbation is not None:
        dR = dR + space.raw_transform(perturbation, dR)
        # https://github.com/mir-group/nequip/blob/c56f48fcc9b4018a84e1ed28f762fadd5bc763f1/nequip/nn/_grad_output.py#L267
        # https://github.com/sirmarcel/glp/blob/main/glp/calculators/utils.py
        # other codes do R = R + strain, not dR
        # can be implemented for efficiency
    return dR


def get_disp_fn(displacement):
    def disp_fn(ri, rj, perturbation, box):
        return displacement(ri, rj, perturbation, box=box)

    return disp_fn


def make_distance_fn(init_box, inference_disp_fn=None):
    """Model which post processes the output of an atomistic model and
    adds empirical energy terms.
    """

    if np.all(init_box < 1e-6):
        # gas phase training and predicting
        displacement_fn = space.free()[0]
        displacement = space.map_bond(displacement_fn)
    elif inference_disp_fn is None:
        # for training on periodic systems
        displacement = vmap(disp_fn, (0, 0, None, None), 0)
    else:
        mappable_displacement_fn = get_disp_fn(inference_disp_fn)
        displacement = vmap(mappable_displacement_fn, (0, 0, None, None), 0)

    def compute_distances(R, neighbor, box, offsets, perturbation=None):
        # Distances
        idx = canonicalize_neighbors(neighbor)
        idx_i, idx_j = idx[0], idx[1]

        # R shape n_atoms x 3
        R = R.astype(jnp.float64)
        Ri = R[idx_i]
        Rj = R[idx_j]

        # dr_vec shape: neighbors x 3
        if np.all(init_box < 1e-6):
            # reverse conventnion to match TF
            # distance vector for gas phase training and predicting
            dr_vec = displacement(Rj, Ri)
        else:
            # distance vector for training on periodic systems
            dr_vec = displacement(Rj, Ri, perturbation, box)
            dr_vec += offsets
        return dr_vec, idx

    return compute_distances
