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
    node_feats_irreps = "16x0e"
    node_attrs_irreps = "10x0e"
    sph_irreps = "1x0e + 1x1o + 1x2e"  # max_ell=2
    target_irreps = "16x0e + 16x1o + 16x2e"
    hidden_irreps = "16x0e"

    node_feats = e3nn.IrrepsArray(
        e3nn.Irreps(node_feats_irreps),
        jnp.asarray(
            np.random.default_rng(0).normal(
                size=(n_atoms, e3nn.Irreps(node_feats_irreps).dim)
            )
        ),
    )
    sph_array = jnp.asarray(np.random.default_rng(1).normal(size=(n_edges, 9)))
    edge_attrs = e3nn.IrrepsArray(e3nn.Irreps(sph_irreps), sph_array)
    edge_feats = jnp.asarray(np.random.default_rng(2).normal(size=(n_edges, 8)))
    node_attrs = e3nn.IrrepsArray(
        node_attrs_irreps,
        jax.nn.one_hot(jnp.arange(n_atoms) % 10, 10),
    )
    i = jnp.asarray(np.random.default_rng(3).integers(0, n_atoms, size=n_edges))
    j = jnp.asarray(np.random.default_rng(4).integers(0, n_atoms, size=n_edges))

    block = InteractionBlock(
        node_feats_irreps=node_feats_irreps,
        node_attrs_irreps=node_attrs_irreps,
        edge_attrs_irreps=sph_irreps,
        target_irreps=target_irreps,
        hidden_irreps=hidden_irreps,
        layer_idx=0,
    )
    params = block.init(
        jax.random.PRNGKey(0),
        node_feats, edge_attrs, edge_feats, node_attrs, i, j,
    )
    message, sc = block.apply(
        params, node_feats, edge_attrs, edge_feats, node_attrs, i, j
    )
    assert message.array.shape == (n_atoms, e3nn.Irreps(target_irreps).dim)
    assert sc.array.shape == (n_atoms, e3nn.Irreps(hidden_irreps).dim)
    assert jnp.isfinite(message.array).all()
    assert jnp.isfinite(sc.array).all()


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
        node_feats_irreps=hidden,
        target_irreps=hidden,
        correlation=3,
        num_elements=10,
        use_sc=False,
        layer_idx=0,
    )
    params = block.init(jax.random.PRNGKey(0), node_feats, None, Z)
    out = block.apply(params, node_feats, None, Z)
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
        node_feats_irreps=hidden,
        target_irreps=hidden,
        correlation=correlation,
        num_elements=num_elements,
        use_sc=False,
        layer_idx=0,
    )
    params = block.init(jax.random.PRNGKey(0), node_feats, None, Z)
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
    block = ProductBlock(
        node_feats_irreps=hidden,
        target_irreps=hidden,
        correlation=3,
        num_elements=10,
        use_sc=False,
        layer_idx=0,
    )
    Z_a = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    Z_b = jnp.array([5, 6, 7, 8], dtype=jnp.int32)
    params = block.init(jax.random.PRNGKey(0), node_feats, None, Z_a)
    out_a = block.apply(params, node_feats, None, Z_a)
    out_b = block.apply(params, node_feats, None, Z_b)
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


def test_interaction_block_emits_target_irreps_and_skip():
    """InteractionBlock returns (message in target_irreps, sc in hidden_irreps)."""
    n_atoms, n_edges = 4, 6
    node_feats_irreps = "8x0e"
    node_attrs_irreps = "5x0e"
    edge_attrs_irreps = "1x0e + 1x1o + 1x2e"      # max_ell=2
    target_irreps = "8x0e + 8x1o + 8x2e"           # interaction_irreps
    hidden_irreps = "8x0e"

    block = InteractionBlock(
        node_feats_irreps=node_feats_irreps,
        node_attrs_irreps=node_attrs_irreps,
        edge_attrs_irreps=edge_attrs_irreps,
        target_irreps=target_irreps,
        hidden_irreps=hidden_irreps,
        layer_idx=0,
    )
    rng = jax.random.PRNGKey(0)
    node_feats = e3nn.IrrepsArray(
        node_feats_irreps,
        jnp.ones((n_atoms, e3nn.Irreps(node_feats_irreps).dim)),
    )
    sph = e3nn.IrrepsArray(
        edge_attrs_irreps,
        jnp.ones((n_edges, e3nn.Irreps(edge_attrs_irreps).dim)),
    )
    radial = jnp.ones((n_edges, 4))
    Z_one_hot = e3nn.IrrepsArray(
        node_attrs_irreps,
        jax.nn.one_hot(jnp.arange(n_atoms) % 5, 5),
    )
    receivers = jnp.array([0, 1, 2, 3, 0, 1])
    senders = jnp.array([1, 2, 3, 0, 2, 3])

    params = block.init(rng, node_feats, sph, radial, Z_one_hot, receivers, senders)
    message, sc = block.apply(
        params, node_feats, sph, radial, Z_one_hot, receivers, senders
    )

    assert isinstance(message, e3nn.IrrepsArray)
    assert isinstance(sc, e3nn.IrrepsArray)
    assert e3nn.Irreps(message.irreps) == e3nn.Irreps(target_irreps)
    assert e3nn.Irreps(sc.irreps) == e3nn.Irreps(hidden_irreps)
    assert message.array.shape[0] == n_atoms
    assert sc.array.shape[0] == n_atoms


def test_product_block_emits_target_irreps_with_skip():
    """ProductBlock symmetric-contracts node_feats_irreps -> target_irreps + sc."""
    n_atoms = 3
    node_feats_irreps = "8x0e + 8x1o + 8x2e"
    target_irreps = "8x0e"

    block = ProductBlock(
        node_feats_irreps=node_feats_irreps,
        target_irreps=target_irreps,
        correlation=2,
        num_elements=5,
        use_sc=True,
        layer_idx=0,
    )
    node_feats = e3nn.IrrepsArray(
        node_feats_irreps,
        jnp.ones((n_atoms, e3nn.Irreps(node_feats_irreps).dim)),
    )
    sc = e3nn.IrrepsArray(
        target_irreps,
        jnp.ones((n_atoms, e3nn.Irreps(target_irreps).dim)),
    )
    Z = jnp.array([1, 2, 3])

    params = block.init(jax.random.PRNGKey(0), node_feats, sc, Z)
    out = block.apply(params, node_feats, sc, Z)
    assert isinstance(out, e3nn.IrrepsArray)
    assert e3nn.Irreps(out.irreps) == e3nn.Irreps(target_irreps)
    assert out.array.shape == (n_atoms, e3nn.Irreps(target_irreps).dim)


def test_tp_out_irreps_with_instructions_basic():
    """Helper computes the simplified intersection of in1 ⊗ in2 with target."""
    import e3nn_jax as e3nn

    from apax.layers.descriptor.mace_blocks import tp_out_irreps_with_instructions

    irreps_in1 = e3nn.Irreps("8x0e")
    irreps_in2 = e3nn.Irreps("1x0e + 1x1o + 1x2e + 1x3o")
    target = e3nn.Irreps("8x0e + 8x1o + 8x2e + 8x3o")
    irreps_mid, instructions = tp_out_irreps_with_instructions(
        irreps_in1, irreps_in2, target,
    )
    assert e3nn.Irreps(irreps_mid).dim > 0
    assert all(isinstance(i, tuple) for i in instructions)


from apax.layers.descriptor.mace_blocks import (
    _INTERACTION_BLOCK_CLS,
    InteractionBlockDensity,
    InteractionBlockDensityResidual,
    InteractionBlockResidual,
)


def _make_density_inputs(seed: int = 0):
    """Build a small set of inputs reusable by Density / DensityResidual tests."""
    n_atoms, n_edges = 5, 12
    node_feats_irreps = "16x0e"
    node_attrs_irreps = "10x0e"
    sph_irreps = "1x0e + 1x1o + 1x2e"
    target_irreps = "16x0e + 16x1o + 16x2e"
    hidden_irreps = "16x0e"

    rng = np.random.default_rng(seed)
    node_feats = e3nn.IrrepsArray(
        node_feats_irreps,
        jnp.asarray(rng.normal(size=(n_atoms, e3nn.Irreps(node_feats_irreps).dim))),
    )
    sph = e3nn.IrrepsArray(
        sph_irreps,
        jnp.asarray(rng.normal(size=(n_edges, e3nn.Irreps(sph_irreps).dim))),
    )
    radial = jnp.asarray(rng.normal(size=(n_edges, 8)))
    Z_one_hot = e3nn.IrrepsArray(
        node_attrs_irreps, jax.nn.one_hot(jnp.arange(n_atoms) % 10, 10),
    )
    receivers = jnp.asarray(rng.integers(0, n_atoms, size=n_edges))
    senders = jnp.asarray(rng.integers(0, n_atoms, size=n_edges))
    return {
        "node_feats_irreps": node_feats_irreps,
        "node_attrs_irreps": node_attrs_irreps,
        "sph_irreps": sph_irreps,
        "target_irreps": target_irreps,
        "hidden_irreps": hidden_irreps,
        "node_feats": node_feats,
        "sph": sph,
        "radial": radial,
        "Z_one_hot": Z_one_hot,
        "receivers": receivers,
        "senders": senders,
        "n_atoms": n_atoms,
    }


def test_interaction_block_density_shape_and_finite():
    """Density variant returns ``(message, None)`` with message in target_irreps."""
    ctx = _make_density_inputs(seed=0)
    block = InteractionBlockDensity(
        node_feats_irreps=ctx["node_feats_irreps"],
        node_attrs_irreps=ctx["node_attrs_irreps"],
        edge_attrs_irreps=ctx["sph_irreps"],
        target_irreps=ctx["target_irreps"],
        hidden_irreps=ctx["hidden_irreps"],  # unused
        layer_idx=0,
    )
    params = block.init(
        jax.random.PRNGKey(0),
        ctx["node_feats"], ctx["sph"], ctx["radial"], ctx["Z_one_hot"],
        ctx["receivers"], ctx["senders"],
    )
    message, sc = block.apply(
        params,
        ctx["node_feats"], ctx["sph"], ctx["radial"], ctx["Z_one_hot"],
        ctx["receivers"], ctx["senders"],
    )
    assert sc is None
    assert isinstance(message, e3nn.IrrepsArray)
    assert e3nn.Irreps(message.irreps) == e3nn.Irreps(ctx["target_irreps"])
    assert message.array.shape == (ctx["n_atoms"], e3nn.Irreps(ctx["target_irreps"]).dim)
    assert jnp.isfinite(message.array).all()


def test_interaction_block_density_residual_shape_and_finite():
    """DensityResidual returns ``(message in target, sc in hidden)``."""
    ctx = _make_density_inputs(seed=1)
    block = InteractionBlockDensityResidual(
        node_feats_irreps=ctx["node_feats_irreps"],
        node_attrs_irreps=ctx["node_attrs_irreps"],
        edge_attrs_irreps=ctx["sph_irreps"],
        target_irreps=ctx["target_irreps"],
        hidden_irreps=ctx["hidden_irreps"],
        layer_idx=0,
    )
    params = block.init(
        jax.random.PRNGKey(0),
        ctx["node_feats"], ctx["sph"], ctx["radial"], ctx["Z_one_hot"],
        ctx["receivers"], ctx["senders"],
    )
    message, sc = block.apply(
        params,
        ctx["node_feats"], ctx["sph"], ctx["radial"], ctx["Z_one_hot"],
        ctx["receivers"], ctx["senders"],
    )
    assert sc is not None
    assert e3nn.Irreps(message.irreps) == e3nn.Irreps(ctx["target_irreps"])
    assert e3nn.Irreps(sc.irreps) == e3nn.Irreps(ctx["hidden_irreps"])
    assert message.array.shape[0] == ctx["n_atoms"]
    assert sc.array.shape[0] == ctx["n_atoms"]
    assert jnp.isfinite(message.array).all() and jnp.isfinite(sc.array).all()


def test_interaction_scaffold_param_names_match_torch():
    """Residual param tree carries the torch-aligned slot names.

    Acts as a regression guard for converter mapping: the slots
    ``linear_up``, ``radial_mlp``, ``linear``, ``skip_tp`` are the keys the
    torch->apax mapper writes into.
    """
    ctx = _make_density_inputs(seed=2)
    block = InteractionBlockResidual(
        node_feats_irreps=ctx["node_feats_irreps"],
        node_attrs_irreps=ctx["node_attrs_irreps"],
        edge_attrs_irreps=ctx["sph_irreps"],
        target_irreps=ctx["target_irreps"],
        hidden_irreps=ctx["hidden_irreps"],
        layer_idx=0,
    )
    params = block.init(
        jax.random.PRNGKey(0),
        ctx["node_feats"], ctx["sph"], ctx["radial"], ctx["Z_one_hot"],
        ctx["receivers"], ctx["senders"],
    )
    keys = set(params["params"].keys())
    assert keys == {"linear_up", "radial_mlp", "linear", "skip_tp"}


def test_interaction_block_density_param_tree_has_density_fn():
    """Density variant adds a ``density_fn`` slot for the per-edge gate weight."""
    ctx = _make_density_inputs(seed=3)
    block = InteractionBlockDensity(
        node_feats_irreps=ctx["node_feats_irreps"],
        node_attrs_irreps=ctx["node_attrs_irreps"],
        edge_attrs_irreps=ctx["sph_irreps"],
        target_irreps=ctx["target_irreps"],
        hidden_irreps=ctx["hidden_irreps"],
        layer_idx=0,
    )
    params = block.init(
        jax.random.PRNGKey(0),
        ctx["node_feats"], ctx["sph"], ctx["radial"], ctx["Z_one_hot"],
        ctx["receivers"], ctx["senders"],
    )
    keys = set(params["params"].keys())
    assert keys == {"linear_up", "radial_mlp", "linear", "skip_tp", "density_fn"}
    # density_fn is a single Dense(num_bessel -> 1) layer.
    density_kernel = params["params"]["density_fn"]["Dense_0"]["kernel"]
    assert density_kernel.shape == (ctx["radial"].shape[-1], 1)


def test_interaction_block_dispatch_table_complete():
    """``_INTERACTION_BLOCK_CLS`` covers every Literal value in MaceModelConfig."""
    assert _INTERACTION_BLOCK_CLS["RealAgnosticResidual"] is InteractionBlockResidual
    assert _INTERACTION_BLOCK_CLS["RealAgnosticDensity"] is InteractionBlockDensity
    assert (
        _INTERACTION_BLOCK_CLS["RealAgnosticDensityResidual"]
        is InteractionBlockDensityResidual
    )


def test_nonlinear_readout_block_applies_silu_normalization():
    """Gate is SiLU * silu_normalization (torch-mace normalize2mom constant)."""
    import e3nn_jax as e3nn
    import jax
    import jax.numpy as jnp
    import numpy as np

    from apax.layers.descriptor.mace_blocks import NonLinearReadoutBlock

    # Build a block where we can inspect the linear_2 output directly.
    block = NonLinearReadoutBlock(MLP_irreps="4x0e", n_out=1)
    feat = e3nn.IrrepsArray("8x0e", jnp.ones((8,)))
    params = block.init(jax.random.PRNGKey(0), feat)
    out_default = block.apply(params, feat)

    # With silu_normalization=1.0, the post-gate activation is plain SiLU.
    block_ref = NonLinearReadoutBlock(MLP_irreps="4x0e", n_out=1,
                                       silu_normalization=1.0)
    out_ref = block_ref.apply(params, feat)

    # Default differs from unnormalised — the constant is actually wired in.
    assert not np.allclose(out_default, out_ref)
    # And the default constant is the documented value.
    assert block.silu_normalization == 1.6791767923989418
