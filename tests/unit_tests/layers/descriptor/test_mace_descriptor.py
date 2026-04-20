"""Shape & contract tests for MaceRepresentation.

These tests use random weights and a skeleton forward pass; correctness
against upstream MACE is validated in the parity tests (gated).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apax.layers.descriptor.mace import MaceRepresentation


@pytest.fixture
def tiny_system():
    n_atoms = 4
    n_neighbors = 8
    rng = np.random.default_rng(0)
    dr_vec = jnp.asarray(rng.normal(size=(n_neighbors, 3))).astype(jnp.float32)
    Z = jnp.asarray([1, 8, 1, 6], dtype=jnp.int32)
    idx = jnp.asarray(
        [[0, 0, 1, 1, 2, 2, 3, 3],
         [1, 2, 0, 3, 0, 3, 1, 2]], dtype=jnp.int32,
    )
    return dr_vec, Z, idx, n_atoms


def test_mace_representation_contract(tiny_system):
    dr_vec, Z, idx, n_atoms = tiny_system
    model = MaceRepresentation(
        r_max=5.0,
        num_bessel=8,
        max_ell=2,
        hidden_irreps="16x0e + 16x1o",
        num_interactions=2,
    )
    params = model.init(jax.random.PRNGKey(0), dr_vec, Z, idx)
    out = model.apply(params, dr_vec, Z, idx)
    assert out.ndim == 2
    assert out.shape[0] == n_atoms
    assert out.dtype == jnp.float32
    assert jnp.isfinite(out).all()


def test_mace_representation_is_jittable(tiny_system):
    dr_vec, Z, idx, n_atoms = tiny_system
    model = MaceRepresentation(hidden_irreps="8x0e")
    params = model.init(jax.random.PRNGKey(0), dr_vec, Z, idx)
    jitted = jax.jit(model.apply)
    out = jitted(params, dr_vec, Z, idx)
    assert out.shape[0] == n_atoms
