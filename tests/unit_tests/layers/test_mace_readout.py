"""MaceReadout — per-layer sum readout for the EnergyModel readout slot."""
import jax
import jax.numpy as jnp
import numpy as np

from apax.layers.readout import MaceReadout


def test_mace_readout_single_atom_shape():
    """Per-atom call returns (1,) by default."""
    num_interactions = 2
    hidden_dim = 16
    readout = MaceReadout(
        num_interactions=num_interactions,
        hidden_dim=hidden_dim,
        MLP_irreps="8x0e",
    )
    x = jnp.ones((num_interactions * hidden_dim,))
    params = readout.init(jax.random.PRNGKey(0), x)
    out = readout.apply(params, x)
    assert out.shape == (1,)


def test_mace_readout_shallow_ensemble_shape():
    """n_shallow_ensemble>0 produces an (n_members,) output per atom."""
    n_members = 4
    readout = MaceReadout(
        num_interactions=2,
        hidden_dim=16,
        MLP_irreps="8x0e",
        n_shallow_ensemble=n_members,
    )
    x = jnp.ones((2 * 16,))
    params = readout.init(jax.random.PRNGKey(0), x)
    out = readout.apply(params, x)
    assert out.shape == (n_members,)


def test_mace_readout_vmapped_over_atoms():
    """Vmap returns (n_atoms, n_out) — matches EnergyModel call site."""
    num_interactions = 2
    hidden_dim = 16
    n_atoms = 5
    readout = MaceReadout(num_interactions=num_interactions, hidden_dim=hidden_dim)
    x_single = jnp.ones((num_interactions * hidden_dim,))
    params = readout.init(jax.random.PRNGKey(0), x_single)
    g = jnp.ones((n_atoms, num_interactions * hidden_dim))
    batched = jax.vmap(lambda xi: readout.apply(params, xi))(g)
    assert batched.shape == (n_atoms, 1)


def test_mace_readout_uses_one_linear_then_one_nonlinear():
    """First N-1 readouts are linear; last is non-linear (Linear+Linear)."""
    num_interactions = 2
    hidden_dim = 16
    readout = MaceReadout(
        num_interactions=num_interactions,
        hidden_dim=hidden_dim,
        MLP_irreps="8x0e",
    )
    x = jnp.ones((num_interactions * hidden_dim,))
    params = readout.init(jax.random.PRNGKey(0), x)
    flat_keys = ["/".join(str(k) for k in path)
                 for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]]
    # readout_0 is the LinearReadoutBlock — one Linear sub-module.
    assert any("readout_0" in k and "linear" in k for k in flat_keys)
    # readout_1 is the NonLinearReadoutBlock — linear_1 + linear_2 sub-modules.
    assert any("readout_1" in k and "linear_1" in k for k in flat_keys)
    assert any("readout_1" in k and "linear_2" in k for k in flat_keys)


def test_mace_readout_outputs_finite_with_random_input():
    """Sanity: random input with random params returns finite scalars."""
    rng = np.random.default_rng(0)
    num_interactions = 3
    hidden_dim = 8
    readout = MaceReadout(num_interactions=num_interactions, hidden_dim=hidden_dim)
    x = jnp.asarray(rng.normal(size=(num_interactions * hidden_dim,)))
    params = readout.init(jax.random.PRNGKey(1), x)
    out = readout.apply(params, x)
    assert out.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(out)))
