"""Dimer parity for :class:`MaceZBLPairRepulsion` vs torch-mace ``ZBLBasis``.

Gated by the ``mace_parity`` marker; requires::

    uv sync --extra mace-convert

Validates the faithful port at machine precision: for a single dimer at a
sweep of separations, the apax energy must match the torch reference to
``rtol=1e-12, atol=1e-12``.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.mace_parity


@pytest.mark.parametrize(
    "Z_pair",
    [(1, 1), (1, 8), (8, 8), (14, 8)],
    ids=["H-H", "H-O", "O-O", "Si-O"],
)
def test_mace_zbl_pair_repulsion_matches_torch_on_dimer(Z_pair):
    """Per-pair ZBL energy matches torch-mace ``ZBLBasis`` exactly."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")

    import jax.numpy as jnp
    import jax.random as jr
    import torch
    from mace.modules.radial import ZBLBasis

    from apax.layers.empirical import MaceZBLPairRepulsion

    Z_a, Z_b = Z_pair
    drs = np.linspace(0.5, 2.5, 21)  # range covering the polynomial cutoff regime

    # Reference: torch ZBLBasis. Its ``forward`` consumes ``(x, node_attrs,
    # edge_index, atomic_numbers)`` and returns per-atom V_ZBL; we sum across
    # atoms to get the total scalar.
    torch.set_default_dtype(torch.float64)
    zbl_torch = ZBLBasis(p=6, trainable=False)
    # ``atomic_numbers`` is a 1D table; ``node_attrs`` is a one-hot row over
    # that table. For a 2-atom dimer we set up the table explicitly.
    atomic_numbers = torch.tensor([Z_a, Z_b], dtype=torch.int64)
    node_attrs = torch.eye(2, dtype=torch.float64)
    # Edge index: (sender, receiver) pairs. For a symmetric dimer we want
    # both directions so the pairwise sum matches apax (which already
    # iterates each ordered pair).
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)

    # Apax fixture: same two atoms, dr_vec along x-axis. ``idx[0]=receiver,
    # idx[1]=sender``; we mirror the torch convention (both directions).
    apax_zbl = MaceZBLPairRepulsion(p=6, apply_mask=False, dtype="fp64")
    Z_jnp = jnp.asarray([Z_a, Z_b], dtype=jnp.int32)
    idx = jnp.asarray([[1, 0], [0, 1]], dtype=jnp.int32)
    params = apax_zbl.init(
        jr.PRNGKey(0),
        jnp.zeros((2, 3)),  # R (unused except for shape)
        jnp.zeros((2, 3)),  # dr_vec
        Z_jnp,
        idx,
        jnp.zeros((3, 3)),  # box
        {},                 # properties
    )

    for r in drs:
        # Torch reference. ``lengths`` is passed in with shape (n_edges, 1)
        # by upstream MACE (``torch.norm(..., keepdim=True)``), so internal
        # broadcasting against per-edge ``Z_u`` (also (n_edges, 1)) is
        # element-wise. A 1D ``(n_edges,)`` would broadcast to (n, n).
        x = torch.tensor([[r], [r]], dtype=torch.float64)
        e_torch = float(
            zbl_torch.forward(x, node_attrs, edge_index, atomic_numbers).sum()
        )

        # Apax: dr_vec along +x for both directed edges (sign doesn't matter,
        # only |r|), drives the per-edge dr = r used inside the module.
        dr_vec = jnp.asarray([[r, 0.0, 0.0], [-r, 0.0, 0.0]], dtype=jnp.float64)
        e_apax = float(
            apax_zbl.apply(
                params,
                jnp.zeros((2, 3)),
                dr_vec,
                Z_jnp,
                idx,
                jnp.zeros((3, 3)),
                {},
            )
        )
        assert np.isclose(e_apax, e_torch, rtol=1e-12, atol=1e-12), (
            f"ZBL mismatch at r={r:.3f}, Z=({Z_a},{Z_b}): "
            f"apax={e_apax:.12e} torch={e_torch:.12e} "
            f"diff={e_apax - e_torch:.3e}"
        )


def test_mace_zbl_pair_repulsion_param_tree():
    """Without ``trainable=True`` all five floats live in the ``buffers`` collection."""
    import jax.numpy as jnp
    import jax.random as jr

    from apax.layers.empirical import MaceZBLPairRepulsion

    block = MaceZBLPairRepulsion(p=6, apply_mask=False, trainable=False, dtype="fp64")
    init = block.init(
        jr.PRNGKey(0),
        jnp.zeros((2, 3)),
        jnp.zeros((2, 3)),
        jnp.array([1, 1], dtype=jnp.int32),
        jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
        jnp.zeros((3, 3)),
        {},
    )
    assert "buffers" in init
    buffers = init["buffers"]
    assert set(buffers.keys()) == {"c", "a_exp", "a_prefactor", "covalent_radii"}
    assert buffers["c"].shape == (4,)
    assert buffers["covalent_radii"].shape == (119,)
    # No trainable params under "params".
    assert init.get("params", {}) == {}


def test_mace_zbl_pair_repulsion_trainable_param_tree():
    """``trainable=True`` lifts ``a_exp`` and ``a_prefactor`` into ``params``."""
    import jax.numpy as jnp
    import jax.random as jr

    from apax.layers.empirical import MaceZBLPairRepulsion

    block = MaceZBLPairRepulsion(p=6, apply_mask=False, trainable=True, dtype="fp64")
    init = block.init(
        jr.PRNGKey(0),
        jnp.zeros((2, 3)),
        jnp.zeros((2, 3)),
        jnp.array([1, 1], dtype=jnp.int32),
        jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
        jnp.zeros((3, 3)),
        {},
    )
    assert set(init["params"].keys()) == {"a_exp", "a_prefactor"}
    assert set(init["buffers"].keys()) == {"c", "covalent_radii"}
