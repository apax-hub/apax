import jax
import jax.numpy as jnp

from apax.layers.descriptor.basis_functions import GaussianBasis, RadialFunction


def test_gaussian_basis():
    n_basis = 5
    key = jax.random.PRNGKey(0)
    basis = GaussianBasis(n_basis=n_basis)

    dr = jnp.array([0.5, 1.0, 2.0, 3.0])  # n_neighbors

    params = basis.init(key, dr)
    assert len(params.keys()) == 0
    result = basis.apply(params, dr)
    assert result.shape == (4, n_basis)  # n_neighbors x n_basis


def test_radial_function():
    key = jax.random.PRNGKey(0)
    n_species = 119  # 3
    n_basis = 5
    n_radial = 2

    dr = jnp.array([0.5, 1.0, 2.0, 3.0], dtype=jnp.float32)  # n_neighbors
    Z_i = jnp.array([1, 2, 1, 2])
    Z_j = jnp.array([2, 1, 2, 1])
    # cutoff = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

    radial_fn = RadialFunction(
        n_species=n_species, n_radial=n_radial, basis_fn=GaussianBasis(n_basis)
    )

    params = radial_fn.init(key, dr, Z_i, Z_j)
    result = radial_fn.apply(params, dr, Z_i, Z_j)

    assert params["params"]["atomic_type_embedding"].shape == (
        n_species,
        n_species,
        n_radial,
        n_basis,
    )
    assert result.shape == (4, n_radial)  # n_neighbors x n_radial
