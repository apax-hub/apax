"""MaceRadialEmbedding: radial-only contract + injected basis_fn."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apax.layers.descriptor.basis_functions import (
    MaceBesselBasis,
    MaceRadialEmbedding,
)


@pytest.fixture
def tiny_edges():
    rng = np.random.default_rng(0)
    dr_vec = jnp.asarray(rng.normal(size=(6, 3))).astype(jnp.float32)
    Z = jnp.array([1, 8, 1, 6], dtype=jnp.int32)
    idx = jnp.array(
        [[0, 0, 1, 1, 2, 3], [1, 2, 0, 3, 0, 2]],
        dtype=jnp.int32,
    )
    return dr_vec, Z, idx


def test_radial_embedding_returns_only_radial(tiny_edges):
    dr_vec, Z, idx = tiny_edges
    basis = MaceBesselBasis(n_basis=4, r_max=5.0, dtype=jnp.float32)
    embed = MaceRadialEmbedding(
        basis_fn=basis,
        num_polynomial_cutoff=5,
        r_max=5.0,
        distance_transform=None,
    )
    params = embed.init(jax.random.PRNGKey(0), dr_vec, Z, idx)
    out = embed.apply(params, dr_vec, Z, idx)
    # Single tensor, not a (radial, sph) pair.
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (dr_vec.shape[0], 4)
    assert out.dtype == dr_vec.dtype


def test_radial_embedding_has_no_max_ell_or_n_basis_attrs():
    """The submodule no longer carries max_ell / num_bessel / n_basis fields."""
    fields = MaceRadialEmbedding.__dataclass_fields__  # flax dataclass fields
    assert "max_ell" not in fields
    assert "num_bessel" not in fields
    assert "n_basis" not in fields
    assert "basis_fn" in fields


def test_radial_embedding_with_distance_transform(tiny_edges):
    """Cutoff sees original r; basis sees transformed r."""
    from apax.layers.descriptor.basis_functions import AgnesiTransform

    dr_vec, Z, idx = tiny_edges
    basis = MaceBesselBasis(n_basis=4, r_max=5.0, dtype=jnp.float32)
    dt = AgnesiTransform(trainable=False)
    embed = MaceRadialEmbedding(
        basis_fn=basis,
        num_polynomial_cutoff=5,
        r_max=5.0,
        distance_transform=dt,
    )
    params = embed.init(jax.random.PRNGKey(0), dr_vec, Z, idx)
    out = embed.apply(params, dr_vec, Z, idx)
    assert out.shape == (dr_vec.shape[0], 4)
    assert jnp.all(jnp.isfinite(out))
