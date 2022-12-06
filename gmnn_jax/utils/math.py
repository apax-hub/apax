import jax.numpy as jnp

def normed_dotp(F_0, F_pred):
    F_0_norm = jnp.linalg.norm(F_0, ord=2, axis=2, keepdims=True)
    F_p_norm = jnp.linalg.norm(F_pred, ord=2, axis=2, keepdims=True)

    F_0_n = F_0 / F_0_norm
    F_p_n = F_pred / F_p_norm

    dotp = jnp.einsum("bai, bai -> ba", F_0_n, F_p_n)
    return dotp