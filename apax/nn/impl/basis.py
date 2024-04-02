import einops
from apax import ops

def gaussian_basis_impl(dr, shifts, betta, rad_norm):
    dr = einops.repeat(dr, "neighbors -> neighbors 1")
    # 1 x n_basis, neighbors x 1 -> neighbors x n_basis
    distances = shifts - dr

    # shape: neighbors x n_basis
    basis = ops.exp(-betta * (distances**2))
    basis = rad_norm * basis
    return basis


def cosine_cutoff(dr, r_max):
    # shape: neighbors
    dr_clipped = ops.clip(dr, a_max=r_max)
    cos_cutoff = 0.5 * (ops.cos(np.pi * dr_clipped / r_max) + 1.0)
    cutoff = einops.repeat(cos_cutoff, "neighbors -> neighbors 1")
    return cutoff


def radial_basis_impl(basis, Z_i, Z_j, embeddings, embed_norm):
    if embeddings is None:
        radial_function = basis
    else:
        # coeffs shape: n_neighbors x n_radialx n_basis
        # reverse convention to match original
        species_pair_coeffs = embeddings[Z_j, Z_i, ...]
        species_pair_coeffs = embed_norm * species_pair_coeffs

        # radial shape: neighbors x n_radial
        radial_function = einops.einsum(
            species_pair_coeffs, basis, "nbrs radial basis, nbrs basis -> nbrs radial"
        )
    return radial_function