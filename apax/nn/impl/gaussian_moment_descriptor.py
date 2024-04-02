import einops

from apax import ops


def gaussian_moment_impl(moments, triang_idxs_2d, triang_idxs_3d, n_contr):
    
    contr_0 = moments[0]
    contr_1 = ops.einsum("ari, asi -> rsa", moments[1], moments[1])
    contr_2 = ops.einsum("arij, asij -> rsa", moments[2], moments[2])
    contr_3 = ops.einsum("arijk, asijk -> rsa", moments[3], moments[3])
    contr_4 = ops.einsum("arij, asik, atjk -> rsta", moments[2], moments[2], moments[2])
    contr_5 = ops.einsum("ari, asj, atij -> rsta", moments[1], moments[1], moments[2])
    contr_6 = ops.einsum("arijk, asijl, atkl -> rsta", moments[3], moments[3], moments[2])
    contr_7 = ops.einsum("arijk, asij, atk -> rsta", moments[3], moments[2], moments[1])

    # n_symm01_features = triang_idxs_2d.shape[0] * n_radial
    tril_2_i, tril_2_j = triang_idxs_2d[:, 0], triang_idxs_2d[:, 1]
    tril_3_i, tril_3_j, tril_3_k = (
        triang_idxs_3d[:, 0],
        triang_idxs_3d[:, 1],
        triang_idxs_3d[:, 2],
    )

    contr_1 = contr_1[tril_2_i, tril_2_j]
    contr_2 = contr_2[tril_2_i, tril_2_j]
    contr_3 = contr_3[tril_2_i, tril_2_j]
    contr_4 = contr_4[tril_3_i, tril_3_j, tril_3_k]
    contr_5 = contr_5[tril_2_i, tril_2_j]
    contr_6 = contr_6[tril_2_i, tril_2_j]

    contr_1 = einops.rearrange(contr_1, "features atoms -> atoms features")
    contr_2 = einops.rearrange(contr_2, "features atoms -> atoms features")
    contr_3 = einops.rearrange(contr_3, "features atoms -> atoms features")
    contr_4 = einops.rearrange(contr_4, "features atoms -> atoms features")
    contr_5 = einops.rearrange(contr_5, "f1 f2 atoms -> atoms (f1 f2)")
    contr_6 = einops.rearrange(contr_6, "f1 f2 atoms -> atoms (f1 f2)")
    contr_7 = einops.rearrange(contr_7, "f1 f2 f3 atoms -> atoms (f1 f2 f3)")

    gaussian_moments = [
        contr_0,
        contr_1,
        contr_2,
        contr_3,
        contr_4,
        contr_5,
        contr_6,
        contr_7,
    ]

    # gaussian_moments shape: n_atoms x n_features
    gaussian_moments = ops.concatenate(gaussian_moments[: n_contr], axis=-1)
    return gaussian_moments
