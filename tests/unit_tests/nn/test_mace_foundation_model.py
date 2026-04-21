"""Smoke tests for MaceFoundationEnergyModel."""
import jax
import jax.numpy as jnp

from apax.nn.mace_foundation_model import MaceFoundationEnergyModel


def test_mace_foundation_energy_model_random_init():
    m = MaceFoundationEnergyModel(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        hidden_irreps="16x0e",
        num_interactions=2,
        correlation=2,
        interaction_cls="RealAgnosticResidual",
        num_elements=3,
        atomic_energies=[0.0, -1.0, -2.0],
        atomic_numbers=[1, 6, 8],
        avg_num_neighbors=3.0,
        MLP_irreps="4x0e",
    )
    dr = jnp.zeros((4, 3)).at[0].set(jnp.array([0.5, 0.0, 0.0]))
    Z = jnp.array([1, 6, 8, 1], dtype=jnp.int32)
    idx = jnp.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=jnp.int32)
    params = m.init(jax.random.PRNGKey(0), dr, Z, idx)
    e = m.apply(params, dr, Z, idx)
    assert e.shape == (4,)
    assert jnp.isfinite(e).all()
