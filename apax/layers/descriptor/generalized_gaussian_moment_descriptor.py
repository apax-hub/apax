from typing import Any

import einops
import flax.linen as nn
import jax.numpy as jnp
from jax import vmap
import jax

from apax.layers.descriptor.basis_functions import RadialFunction
from apax.layers.descriptor.moments import geometric_moments
from apax.layers.descriptor.triangular_indices import tril_2d_indices, tril_3d_indices
from apax.layers.masking import mask_by_neighbor
from apax.utils.convert import str_to_dtype
from apax.utils.jax_md_reduced import space
from itertools import combinations_with_replacement
import string

def generate_fully_connected_gmd_contractions(Lmax=3, Kmax=3):
    """
    Generate fully connected δ-only contractions for moments[1..Lmax] up to multiplicity Kmax.
    Returns a dict: {ells_canon: [(inputs_tuple, output_subscript)]}
    """
    contractions = {}

    # Include 2-body scalar from moments[0]
    # contractions[(0,)] = [(("ar",), "ar")]

    radial_letters = list(string.ascii_lowercase[17:])  # r,s,t,...
    ang_letters_pool = list(string.ascii_lowercase[8:])  # i,j,k,...

    seen_multisets = set()

    # --- Helper: recursive backtracking for cross-tensor pairings ---
    def find_pairings(slots, current_pairs, tensor_graph):
        """
        slots: list of remaining angular slots (tensor_index, slot_index)
        current_pairs: list of assigned pairs
        tensor_graph: dict tensor_index -> set of connected tensor indices
        Returns list of valid fully connected pairings
        """
        if not slots:
            # check connectivity
            tensors = list(tensor_graph.keys())
            if len(tensors) <= 1:
                return [list(current_pairs)]
            # simple DFS for connectivity
            visited = set()
            stack = [tensors[0]]
            while stack:
                t = stack.pop()
                if t in visited:
                    continue
                visited.add(t)
                stack.extend(tensor_graph.get(t, []))
            if len(visited) == len(tensors):
                return [list(current_pairs)]
            return []

        first = slots[0]
        pairings_found = []
        for j in range(1, len(slots)):
            second = slots[j]
            if first[0] == second[0]:
                continue  # cross-tensor only
            # update tensor graph
            tg = {k: set(v) for k,v in tensor_graph.items()}
            tg.setdefault(first[0], set()).add(second[0])
            tg.setdefault(second[0], set()).add(first[0])
            # recurse
            rest = slots[1:j] + slots[j+1:]
            pairings_found.extend(find_pairings(rest, current_pairs + [(first, second)], tg))
        return pairings_found

    # --- Main loop: generate candidate multisets ---
    for k in range(2, Kmax+1):
        for ells in combinations_with_replacement(range(1, Lmax+1), k):  # only ell>=1
            ells_canon = tuple(sorted(ells, reverse=True))
            if ells_canon in seen_multisets:
                continue
            seen_multisets.add(ells_canon)

            total_slots = sum(ells_canon)
            if total_slots % 2 != 0:
                continue  # cannot pair fully

            # build flat angular slots
            slots = []
            for t, ell in enumerate(ells_canon):
                slots.extend([(t, i) for i in range(ell)])

            # find at least one fully connected pairing
            pairings = find_pairings(slots, [], {})
            if not pairings:
                continue  # invalid multiset

            chosen_pairing = pairings[0]  # pick first canonical pairing

            # assign angular letters
            ang_letter_map = {}
            for idx, ((t0,p0),(t1,p1)) in enumerate(chosen_pairing):
                letter = ang_letters_pool[idx]
                ang_letter_map[(t0,p0)] = letter
                ang_letter_map[(t1,p1)] = letter

            # build einsum input subscripts
            inputs = []
            for tidx, ell in enumerate(ells_canon):
                R = radial_letters[tidx]
                angs = "".join(ang_letter_map[(tidx, s)] for s in range(ell))
                sub = "a" + R + angs
                inputs.append(sub)

            output = "a" + "".join(radial_letters[:len(ells_canon)])

            einsum_str = ", ".join(inputs) + " -> " + output
            contractions[ells_canon] = einsum_str

    return contractions


def generalized_geometric_moments(radial_function, dn, idx_i, n_atoms=None, Lmax=3):
    # dn shape: neighbors x 3
    # radial_function shape: n_neighbors x n_radial

    # s = spatial dim = 3
    # xyz = einops.repeat(dn, "nbrs s -> nbrs 1 s")
    # xyz2 = einops.repeat(dn, "nbrs s -> nbrs 1 1 s")
    # xyz3 = einops.repeat(dn, "nbrs s -> nbrs 1 1 1 s")

    # # shape: n_neighbors x n_radial x (3)^(moment_number)
    # # s_i = spatial = 3
    # zero_moment = radial_function
    # first_moment = einops.repeat(zero_moment, "n r -> n r 1") * xyz
    # second_moment = einops.repeat(first_moment, "n r s1 -> n r s1 1") * xyz2
    # third_moment = einops.repeat(second_moment, "n r s1 s2 -> n r s1 s2 1") * xyz3

    # # shape: n_atoms x n_radial x (3)^(moment_number)
    # zero_moment = jax.ops.segment_sum(zero_moment, idx_i, n_atoms)
    # first_moment = jax.ops.segment_sum(first_moment, idx_i, n_atoms)
    # second_moment = jax.ops.segment_sum(second_moment, idx_i, n_atoms)
    # third_moment = jax.ops.segment_sum(third_moment, idx_i, n_atoms)
    # moments = [zero_moment, first_moment, second_moment, third_moment]

    xyx_l_1 = dn
    moment = radial_function
    zero_moment = jax.ops.segment_sum(radial_function, idx_i, n_atoms)
    moments= [zero_moment]
    for l in range(Lmax):
        xyz_l = xyx_l_1[...,None, :]
        new_moment = moment[...,None] * xyz_l

        summed_moment = jax.ops.segment_sum(new_moment, idx_i, n_atoms)
        moments.append(summed_moment)
        xyx_l_1 = xyz_l
        moment = new_moment


    return moments


class GeneralizedGaussianMomentDescriptor(nn.Module):
    radial_fn: nn.Module = RadialFunction()
    Lmax: int = 3
    Kmax: int = 3
    dtype: Any = jnp.float32
    apply_mask: bool = True

    def setup(self):
        self.r_max = self.radial_fn.r_max
        self.n_radial = self.radial_fn._n_radial

        self.distance = vmap(space.distance, 0, 0)

    def __call__(self, dr_vec, Z, neighbor_idxs):
        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)
        # Z shape n_atoms
        n_atoms = Z.shape[0]

        idx_i, idx_j = neighbor_idxs[0], neighbor_idxs[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        # dr shape: neighbors
        dr = self.distance(dr_vec)

        # TODO: maybe try jnp where
        dr_repeated = einops.repeat(dr + 1e-5, "neighbors -> neighbors 1")
        # normalized distance vectors, shape neighbors x 3
        dn = dr_vec / dr_repeated

        radial_function = self.radial_fn(dr, Z_i, Z_j)
        if self.apply_mask:
            radial_function = mask_by_neighbor(radial_function, neighbor_idxs)

        contraction_paths = generate_fully_connected_gmd_contractions(Lmax=self.Lmax, Kmax=self.Kmax)

        moments = generalized_geometric_moments(radial_function, dn, idx_j, n_atoms, self.Lmax)

        gaussian_moments = [moments[0]]
        for ranks, einsum_str in contraction_paths.items():
            # print("ranks:", ranks, "einsum:", einsum_str)
            contr = jnp.einsum(einsum_str, *[moments[l] for l in ranks])
            contr = jnp.reshape(contr, [n_atoms, -1])
            gaussian_moments.append(contr)

        # gaussian_moments shape: n_atoms x n_features
        gaussian_moments = jnp.concatenate(gaussian_moments, axis=-1)
        assert gaussian_moments.dtype == dtype
        return gaussian_moments
