"""Unit tests for apax.layers.descriptor.mace_blocks.

Blocks tested here use random weights; parity against upstream torch-mace is
validated elsewhere under @pytest.mark.mace_parity.
"""

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np

from apax.layers.descriptor.mace_blocks import assemble_edge_features


def test_assemble_edge_features_shapes():
    dr = jnp.asarray(np.random.default_rng(0).normal(size=(8, 3))).astype(jnp.float32)
    radial, sph = assemble_edge_features(
        dr,
        r_max=5.0,
        num_bessel=8,
        num_poly_cutoff=5,
        max_ell=2,
    )
    # Radial: (n_edges, num_bessel)
    assert radial.shape == (8, 8)
    # Spherical harmonics: 0e + 1o + 2e = 1 + 3 + 5 = 9 real components
    assert sph.array.shape == (8, 9)


def test_assemble_edge_features_cutoff_zeroes_far_edges():
    """Edges beyond r_max get zero radial features (cutoff envelope)."""
    dr = jnp.asarray([[0.0, 0.0, 10.0]] * 4).astype(jnp.float32)  # |r| = 10 > r_max=5
    radial, _ = assemble_edge_features(
        dr,
        r_max=5.0,
        num_bessel=4,
        num_poly_cutoff=5,
        max_ell=1,
    )
    assert jnp.allclose(radial, 0.0, atol=1e-6)


def test_assemble_edge_features_dtype_preserved():
    dr = jnp.asarray(np.random.default_rng(0).normal(size=(4, 3))).astype(jnp.float64)
    radial, sph = assemble_edge_features(
        dr,
        r_max=5.0,
        num_bessel=4,
        num_poly_cutoff=5,
        max_ell=1,
    )
    assert radial.dtype == jnp.float64
    assert sph.array.dtype == jnp.float64


from apax.layers.descriptor.mace_blocks import LinearNodeEmbedding


def test_linear_node_embedding_scalar_output():
    n_atoms = 5
    num_elements = 10
    hidden_irreps = "16x0e"
    Z = jnp.array([0, 3, 5, 0, 9], dtype=jnp.int32)
    emb = LinearNodeEmbedding(num_elements=num_elements, irreps_out=hidden_irreps)
    params = emb.init(jax.random.PRNGKey(0), Z)
    out = emb.apply(params, Z)
    # IrrepsArray output; scalar-only irreps means (n_atoms, 16)
    assert out.array.shape == (n_atoms, 16)
    assert str(out.irreps) == "16x0e"


from apax.layers.descriptor.mace_blocks import InteractionBlock


def test_interaction_block_shape_and_finite():
    n_atoms, n_edges = 5, 12
    hidden = "16x0e + 16x1o"
    sph_irreps = "1x0e + 1x1o + 1x2e"  # max_ell=2

    node_feats = e3nn.IrrepsArray(
        e3nn.Irreps(hidden),
        jnp.asarray(np.random.default_rng(0).normal(size=(n_atoms, 64))),
    )
    sph_array = jnp.asarray(np.random.default_rng(1).normal(size=(n_edges, 9)))
    edge_attrs = e3nn.IrrepsArray(e3nn.Irreps(sph_irreps), sph_array)
    edge_feats = jnp.asarray(np.random.default_rng(2).normal(size=(n_edges, 8)))
    i = jnp.asarray(np.random.default_rng(3).integers(0, n_atoms, size=n_edges))
    j = jnp.asarray(np.random.default_rng(4).integers(0, n_atoms, size=n_edges))

    block = InteractionBlock(irreps_out=hidden, interaction_cls="RealAgnosticResidual")
    params = block.init(jax.random.PRNGKey(0), node_feats, edge_attrs, edge_feats, i, j)
    out = block.apply(params, node_feats, edge_attrs, edge_feats, i, j)
    assert out.array.shape == (n_atoms, e3nn.Irreps(hidden).dim)
    assert jnp.isfinite(out.array).all()


from apax.layers.descriptor.mace_blocks import ProductBlock


def test_product_block_shape_and_finite():
    n_atoms = 5
    hidden = "16x0e + 16x1o"
    node_feats = e3nn.IrrepsArray(
        e3nn.Irreps(hidden),
        jnp.asarray(np.random.default_rng(0).normal(size=(n_atoms, 64))),
    )
    Z = jnp.array([0, 1, 2, 3, 1], dtype=jnp.int32)
    block = ProductBlock(
        hidden_irreps=hidden,
        correlation=3,
        num_elements=10,
    )
    params = block.init(jax.random.PRNGKey(0), node_feats, Z)
    out = block.apply(params, node_feats, Z)
    assert out.array.shape == (n_atoms, e3nn.Irreps(hidden).dim)
    assert jnp.isfinite(out.array).all()


def test_product_block_weight_param_shape():
    """Pin the weight param shape — the P3 torch→linen converter targets this."""
    hidden = "16x0e + 16x1o"
    num_elements = 10
    correlation = 3
    node_feats = e3nn.IrrepsArray(
        e3nn.Irreps(hidden),
        jnp.asarray(np.random.default_rng(0).normal(size=(3, 64))),
    )
    Z = jnp.array([0, 1, 2], dtype=jnp.int32)
    block = ProductBlock(
        hidden_irreps=hidden, correlation=correlation, num_elements=num_elements,
    )
    params = block.init(jax.random.PRNGKey(0), node_feats, Z)
    weight = params["params"]["weight"]
    # (num_elements, weight_basis_dim, mul) — mul=16 from "16x0e + 16x1o"
    assert weight.shape[0] == num_elements
    assert weight.shape[2] == 16
    # weight_basis_dim depends on correlation and irreps; just check rank + positivity
    assert weight.ndim == 3 and weight.shape[1] > 0


def test_product_block_z_changes_output():
    """Changing Z must change the output — guards against Z-gather being a silent no-op."""
    hidden = "16x0e + 16x1o"
    node_feats = e3nn.IrrepsArray(
        e3nn.Irreps(hidden),
        jnp.asarray(np.random.default_rng(0).normal(size=(4, 64))),
    )
    block = ProductBlock(hidden_irreps=hidden, correlation=3, num_elements=10)
    Z_a = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    Z_b = jnp.array([5, 6, 7, 8], dtype=jnp.int32)
    params = block.init(jax.random.PRNGKey(0), node_feats, Z_a)
    out_a = block.apply(params, node_feats, Z_a)
    out_b = block.apply(params, node_feats, Z_b)
    assert not jnp.allclose(out_a.array, out_b.array)


from apax.layers.descriptor.mace_blocks import (
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    ScaleShift,
)


def test_linear_readout_block_scalar_output():
    n_atoms = 4
    single_feat = e3nn.IrrepsArray("16x0e", jnp.ones((16,)))
    block = LinearReadoutBlock(n_out=1)
    params = block.init(jax.random.PRNGKey(0), single_feat)
    batched_feats = e3nn.IrrepsArray(
        "16x0e",
        jnp.asarray(np.random.default_rng(0).normal(size=(n_atoms, 16))),
    )
    e = jax.vmap(lambda x: block.apply(params, x))(batched_feats)
    assert e.shape == (n_atoms, 1)
    assert jnp.isfinite(e).all()


def test_nonlinear_readout_block_scalar_output():
    n_atoms = 4
    single_feat = e3nn.IrrepsArray("32x0e", jnp.ones((32,)))
    block = NonLinearReadoutBlock(MLP_irreps="16x0e", n_out=1)
    params = block.init(jax.random.PRNGKey(0), single_feat)
    batched_feats = e3nn.IrrepsArray(
        "32x0e",
        jnp.asarray(np.random.default_rng(0).normal(size=(n_atoms, 32))),
    )
    e = jax.vmap(lambda x: block.apply(params, x))(batched_feats)
    assert e.shape == (n_atoms, 1)
    assert jnp.isfinite(e).all()


def test_scale_shift_block_affine():
    x = jnp.array([1.0, 2.0, 3.0])
    block = ScaleShift(scale_init=2.0, shift_init=-1.0)
    params = block.init(jax.random.PRNGKey(0), x)
    y = block.apply(params, x)
    assert jnp.allclose(y, jnp.array([1.0, 3.0, 5.0]))


def test_linear_readout_block_scalar_out():
    """LinearReadoutBlock returns (1,) on a single-atom input."""
    block = LinearReadoutBlock(n_out=1)
    feat = e3nn.IrrepsArray("16x0e", jnp.ones((16,)))
    params = block.init(jax.random.PRNGKey(0), feat)
    out = block.apply(params, feat)
    assert out.shape == (1,)


def test_linear_readout_block_ensemble_out():
    """LinearReadoutBlock returns (n_out,) when n_out>1 (shallow ensemble)."""
    block = LinearReadoutBlock(n_out=4)
    feat = e3nn.IrrepsArray("16x0e", jnp.ones((16,)))
    params = block.init(jax.random.PRNGKey(0), feat)
    out = block.apply(params, feat)
    assert out.shape == (4,)


def test_nonlinear_readout_block_scalar_out():
    """NonLinearReadoutBlock returns (1,) with a SiLU-gated hidden."""
    block = NonLinearReadoutBlock(MLP_irreps="8x0e", n_out=1)
    feat = e3nn.IrrepsArray("16x0e", jnp.ones((16,)))
    params = block.init(jax.random.PRNGKey(0), feat)
    out = block.apply(params, feat)
    assert out.shape == (1,)


def test_nonlinear_readout_block_vmap_over_atoms():
    """vmap over atoms produces (n_atoms, 1) without breaking the per-atom contract."""
    block = NonLinearReadoutBlock(MLP_irreps="8x0e", n_out=1)
    single = e3nn.IrrepsArray("16x0e", jnp.ones((16,)))
    params = block.init(jax.random.PRNGKey(0), single)
    feats_batched = e3nn.IrrepsArray("16x0e", jnp.ones((5, 16)))
    out = jax.vmap(lambda x: block.apply(params, x))(feats_batched)
    assert out.shape == (5, 1)
