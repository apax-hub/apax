import jax
import jax.numpy as jnp
import numpy as np

from apax.layers.masking import mask_by_atom, mask_by_neighbor


def make_atomic_prediction(n_atoms, n_total):
    n_padding = n_total - n_atoms

    Z = jnp.concatenate([np.ones(n_atoms), np.zeros(n_padding)])
    atomic_prediction = jnp.ones((n_total, 1))

    return atomic_prediction, Z


def make_neighbor_prediction(n_nbrs, n_total):
    n_padding = n_total - n_nbrs
    radial_func = np.ones((n_nbrs, 3))
    radial_func_padded = np.concatenate([radial_func, np.full((n_padding, 3), 20)])

    idx_i = np.arange(start=0, stop=n_nbrs)
    idx_j = np.arange(start=0, stop=n_nbrs)[::-1]
    idx = np.stack([idx_i, idx_j])

    idx_padding = np.zeros((2, n_padding))
    idx_padded = np.concatenate([idx, idx_padding], axis=1)

    return radial_func, radial_func_padded, idx_padded


def test_mask_by_atom():
    n_atoms = np.array([4, 6, 10])
    n_total = 10

    Zs = []
    preds = []
    for n in n_atoms:
        prediction, Z = make_atomic_prediction(n, n_total)
        preds.append(prediction)
        Zs.append(Z)

    preds = jnp.stack(preds, axis=0)
    Zs = jnp.stack(Zs, axis=0)
    assert np.all((np.sum(preds, axis=2) - n_total) < 1e-6)

    batched_mask_fn = jax.vmap(mask_by_atom, 0, 0)
    masked_preds = batched_mask_fn(preds[0][None, ...], Zs[0][None, ...])

    assert np.all((np.sum(masked_preds, axis=(1, 2)) - n_atoms) < 1e-6)


def test_mask_by_neighbor():
    n_nbrs = np.array([4, 6, 10])
    n_total = 10

    rfs = []
    rfs_padded = []
    idxs = []
    for n in n_nbrs:
        radial_func, radial_func_padded, idx_padded = make_neighbor_prediction(n, n_total)
        rfs.append(radial_func)
        rfs_padded.append(radial_func_padded)
        idxs.append(idx_padded)

    rfs_padded = jnp.stack(rfs_padded)
    idxs = jnp.stack(idxs)

    for rf, n in zip(rfs, n_nbrs):
        assert (jnp.sum(rf) - n * rf.shape[1]) < 1e-6

    batched_mask_fn = jax.vmap(mask_by_neighbor)

    masked_rfs = batched_mask_fn(rfs_padded, idxs)

    assert np.all(
        (jnp.sum(masked_rfs, axis=(1, 2)) - n_nbrs * masked_rfs.shape[2]) < 1e-6
    )
