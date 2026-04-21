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
