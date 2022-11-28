import einops
import jax


def geometric_moments(radial_function, dn, idx_i, n_atoms=None):
    # dn shape: neighbors x 3
    # radial_function shape: n_neighbors x n_radial

    # s = spatial dim = 3
    xyz = einops.repeat(dn, "nbrs s -> nbrs 1 s")
    xyz2 = einops.repeat(dn, "nbrs s -> nbrs 1 1 s")
    xyz3 = einops.repeat(dn, "nbrs s -> nbrs 1 1 1 s")

    # shape: n_neighbors x n_radial x (3)^(moment_number)
    # s_i = spatial = 3
    zero_moment = radial_function
    first_moment = einops.repeat(zero_moment, "n r -> n r 1") * xyz
    second_moment = einops.repeat(first_moment, "n r s1 -> n r s1 1") * xyz2
    third_moment = einops.repeat(second_moment, "n r s1 s2 -> n r s1 s2 1") * xyz3

    # shape: n_atoms x n_radial x (3)^(moment_number)
    zero_moment = jax.ops.segment_sum(zero_moment, idx_i, n_atoms)
    first_moment = jax.ops.segment_sum(first_moment, idx_i, n_atoms)
    second_moment = jax.ops.segment_sum(second_moment, idx_i, n_atoms)
    third_moment = jax.ops.segment_sum(third_moment, idx_i, n_atoms)

    moments = [zero_moment, first_moment, second_moment, third_moment]

    return moments
