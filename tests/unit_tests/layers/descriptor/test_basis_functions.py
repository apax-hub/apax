import jax
import jax.numpy as jnp

from apax.layers.descriptor.basis_functions import (
    AgnesiTransform,
    GaussianBasis,
    RadialFunction,
)


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


def test_polynomial_cutoff_zero_at_rmax():
    import jax.numpy as jnp

    from apax.layers.descriptor.basis_functions import PolynomialCutoff

    cutoff = PolynomialCutoff(p=5, r_max=5.0)
    r = jnp.array([0.0, 2.5, 4.999, 5.0, 5.1])
    f = cutoff(r)
    assert f.shape == r.shape
    assert jnp.isclose(f[0], 1.0, atol=1e-5)  # value at 0 ≈ 1
    assert f[3] == 0.0  # exactly 0 at r_max
    assert f[4] == 0.0  # 0 beyond r_max
    assert f[2] > 0.0  # positive just below


def test_polynomial_cutoff_monotone_decreasing():
    import jax.numpy as jnp

    from apax.layers.descriptor.basis_functions import PolynomialCutoff

    cutoff = PolynomialCutoff(p=5, r_max=5.0)
    r = jnp.linspace(0.0, 5.0, 50)
    f = cutoff(r)
    diffs = jnp.diff(f)
    assert (diffs <= 1e-6).all()  # never increases


def test_agnesi_transform_shape_and_finite():
    key = jax.random.PRNGKey(0)
    n_edges = 8
    r = jax.random.uniform(key, (n_edges,), minval=0.5, maxval=5.0)
    Z = jnp.array([1, 8, 1, 6], dtype=jnp.int32)
    idx = jnp.array(
        [
            [0, 1, 2, 3, 0, 2, 1, 3],
            [1, 0, 3, 2, 2, 0, 3, 1],
        ],
        dtype=jnp.int32,
    )

    transform = AgnesiTransform()
    variables = transform.init(key, r, Z, idx)
    out = transform.apply(variables, r, Z, idx)

    assert out.shape == r.shape
    assert jnp.all(jnp.isfinite(out))


def test_agnesi_transform_pair_symmetric():
    key = jax.random.PRNGKey(1)
    r = jnp.array([0.8, 1.2, 1.7, 2.5], dtype=jnp.float64)
    Z = jnp.array([1, 8, 6, 14], dtype=jnp.int32)
    idx = jnp.array(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ],
        dtype=jnp.int32,
    )
    idx_swapped = jnp.stack([idx[1], idx[0]], axis=0)

    transform = AgnesiTransform()
    variables = transform.init(key, r, Z, idx)
    out_a = transform.apply(variables, r, Z, idx)
    out_b = transform.apply(variables, r, Z, idx_swapped)

    assert jnp.allclose(out_a, out_b, rtol=1e-12, atol=1e-12)


def test_agnesi_transform_param_collections_buffers():
    key = jax.random.PRNGKey(2)
    r = jnp.array([1.0, 2.0], dtype=jnp.float64)
    Z = jnp.array([1, 8], dtype=jnp.int32)
    idx = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)

    transform = AgnesiTransform(trainable=False)
    variables = transform.init(key, r, Z, idx)

    assert "params" not in variables or len(variables["params"]) == 0
    assert "buffers" in variables
    buffer_leaves = variables["buffers"]
    assert set(buffer_leaves.keys()) == {"a", "q", "p", "covalent_radii"}


def test_agnesi_transform_param_collections_trainable():
    key = jax.random.PRNGKey(3)
    r = jnp.array([1.0, 2.0], dtype=jnp.float64)
    Z = jnp.array([1, 8], dtype=jnp.int32)
    idx = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)

    transform = AgnesiTransform(trainable=True)
    variables = transform.init(key, r, Z, idx)

    assert "params" in variables
    assert set(variables["params"].keys()) == {"a", "q", "p"}
    assert "buffers" in variables
    assert set(variables["buffers"].keys()) == {"covalent_radii"}
