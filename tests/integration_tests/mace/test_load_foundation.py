"""load_mace_foundation runtime loader tests. No torch dependency."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization


def test_load_mace_foundation_from_dir(tmp_path):
    """Fabricated apax dir round-trips through the loader."""
    cfg = {
        "name": "mace",
        "r_max": 5.0,
        "num_bessel": 4,
        "num_polynomial_cutoff": 5,
        "max_ell": 1,
        "hidden_irreps": "8x0e",
        "num_interactions": 1,
        "correlation": 2,
        "interaction_cls": "RealAgnosticResidual",
        "num_elements": 5,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "metadata.json").write_text("{}")

    from apax.layers.descriptor.mace import MaceRepresentation

    model = MaceRepresentation(
        r_max=cfg["r_max"],
        num_bessel=cfg["num_bessel"],
        num_polynomial_cutoff=cfg["num_polynomial_cutoff"],
        max_ell=cfg["max_ell"],
        hidden_irreps=cfg["hidden_irreps"],
        num_interactions=cfg["num_interactions"],
        correlation=cfg["correlation"],
        interaction_cls=cfg["interaction_cls"],
        num_elements=cfg["num_elements"],
    )
    dr_vec = jnp.zeros((4, 3))
    Z = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    idx = jnp.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), dr_vec, Z, idx)
    (tmp_path / "params.msgpack").write_bytes(serialization.to_bytes(params))

    from apax.transfer_learning.mace_foundation import load_mace_foundation
    loaded_params, loaded_cfg = load_mace_foundation(tmp_path)

    assert loaded_cfg.name == "mace"
    assert loaded_cfg.r_max == 5.0
    # Param structure matches
    assert jax.tree_util.tree_structure(loaded_params) == jax.tree_util.tree_structure(params)
    # Values round-trip bitwise
    ref_leaves = jax.tree_util.tree_leaves(params)
    out_leaves = jax.tree_util.tree_leaves(loaded_params)
    for a, b in zip(ref_leaves, out_leaves):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_load_mace_foundation_missing_dir(tmp_path):
    """Loader errors clearly when path doesn't exist."""
    from apax.transfer_learning.mace_foundation import load_mace_foundation
    with pytest.raises((FileNotFoundError, ValueError)):
        load_mace_foundation(tmp_path / "nonexistent")
