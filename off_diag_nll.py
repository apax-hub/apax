import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)


def off_diag_nll(label, prediction, name, parameters={}):
    """Computes the gaussian NLL loss given
    means, targets and standard deviations (uncertainty estimate)
    """
    label = label[name]
    means = prediction[name]
    sigmas = prediction[name + "_uncertainty"]

    eps = 1e-6
    sigmas = jnp.clip(sigmas, min=eps)
    variances = jnp.pow(sigmas, 2)

    x1 = jnp.log(variances)
    x2 = ((means - label) ** 2) / variances
    nll = 0.5 * (x1 + x2)

    # Only consider off-diagonal elements of the covariance matrix
    off_diag_nll = nll - jnp.diag(nll)

    return off_diag_nll


def inv_and_det_3x3(Sigma):
    a00 = Sigma[..., 0, 0]
    a01 = Sigma[..., 0, 1]
    a02 = Sigma[..., 0, 2]
    a10 = Sigma[..., 1, 0] # Sym: a01
    a11 = Sigma[..., 1, 1]
    a12 = Sigma[..., 1, 2]
    a20 = Sigma[..., 2, 0] # Sym: a02
    a21 = Sigma[..., 2, 1] # Sym: a12
    a22 = Sigma[..., 2, 2]

    # 3. Analytical Determinant (Rule of Sarrus)
    det = (a00 * (a11 * a22 - a12 * a21) -
            a01 * (a10 * a22 - a12 * a20) +
            a02 * (a10 * a21 - a11 * a20))
    

    invDet = 1.0 / det
    inv00 = (a11 * a22 - a12 * a21) * invDet
    inv01 = (a02 * a21 - a01 * a22) * invDet
    inv02 = (a01 * a12 - a02 * a11) * invDet

    inv10 = (a12 * a20 - a10 * a22) * invDet
    inv11 = (a00 * a22 - a02 * a20) * invDet
    inv12 = (a10 * a02 - a00 * a12) * invDet


    inv12 = (a02 * a10 - a00 * a12) * invDet
    inv20 = (a10 * a21 - a11 * a20) * invDet # Same as inv02 if symmetric
    inv21 = (a20 * a01 - a00 * a21) * invDet # Same as inv12 if symmetric
    inv22 = (a00 * a11 - a01 * a10) * invDet

    # Reconstruct Inverse Matrix (N, 3, 3)
    # Stack is faster than assignment
    row0 = jnp.stack([inv00, inv01, inv02], axis=-1)
    row1 = jnp.stack([inv10, inv11, inv12], axis=-1)
    row2 = jnp.stack([inv20, inv21, inv22], axis=-1)
    Sigma_inv = jnp.stack([row0, row1, row2], axis=-2)
    return Sigma_inv, det

forces_y = np.random.rand(10, 3) * 5  # Example forces
forces_x = forces_y[..., None] + np.random.rand(10, 3, 16) * 0.1  # Example predictions

fmean = jnp.mean(forces_x, axis=2)  # Mean across the 16 predictions
# fstd = jnp.std(forces_x, axis=2)  # Std across the
diff = forces_y - fmean  # Deviation of mean prediction from the target

deviations = forces_x - fmean[..., None]  # Deviation of each prediction from the mean

K = deviations.shape[2]  # Number of members
Sigma = jnp.einsum('bijk,bilk->bijl', deviations, deviations) / (K - 1)  # Sample covariance matrix

Sigma = Sigma + jnp.eye(3)[None, None, ...] * 1e-2  # Add small value to diagonal for numerical stability

Sigma_inv, det = inv_and_det_3x3(Sigma)

det = jnp.maximum(det, 1e-12)
log_det = jnp.log(det)

# 5. Mahalanobis Distance: diff.T @ Sigma_inv @ diff
# Einsum: (N, 3) * (N, 3, 3) * (N, 3) -> (N,)
# z = Sigma_inv @ diff
z = jnp.einsum('...ij, ...j -> ...i', Sigma_inv, diff)
mahalanobis = jnp.sum(diff * z, axis=-1)

nll = 0.5 * (mahalanobis + log_det + 3 * jnp.log(2 * jnp.pi))

print("log_det shape:", nll)  # Should be (10, 3)
