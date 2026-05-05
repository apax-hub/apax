"""MaceBuilder — descriptor + readout composition tests (nested schema)."""

import jax
import jax.numpy as jnp
import numpy as np

from apax.config.model_config import MaceModelConfig
from apax.nn.builder import MaceBuilder


def _minimal_cfg(**overrides):
    """Build a small MaceModelConfig dict with placeholder data fields."""
    cfg = MaceModelConfig(
        basis={"name": "bessel", "variant": "standard", "n_basis": 4, "r_max": 5.0},
        radial_embedding={"num_polynomial_cutoff": 5, "distance_transform": None},
        descriptor={
            "max_ell": 1,
            "hidden_irreps": "8x0e",
            "correlation": 2,
            "interactions": [
                {"name": "RealAgnosticResidual"},
                {"name": "RealAgnosticResidual"},
            ],
            "avg_num_neighbors": 1.0,
            "use_cueq": False,
        },
        readout={"kind": "mace", "MLP_irreps": "16x0e"},
        descriptor_dtype="fp32",
        readout_dtype="fp32",
        scale_shift_dtype="fp64",
    )
    cfg = cfg.model_copy(update=overrides)
    return cfg.model_dump()


def test_mace_builder_uses_mace_readout_by_default():
    from apax.layers.readout import MaceReadout

    builder = MaceBuilder(_minimal_cfg(), n_species=5)
    readout = builder.build_readout(builder.config)
    assert isinstance(readout, MaceReadout)
    assert readout.num_interactions == 2
    assert readout.hidden_dim == 8
    assert readout.MLP_irreps == "16x0e"


def test_mace_builder_standard_readout_falls_back():
    from apax.layers.readout import AtomisticReadout

    cfg = _minimal_cfg(readout={"kind": "standard", "MLP_irreps": "16x0e"})
    builder = MaceBuilder(cfg, n_species=5)
    readout = builder.build_readout(builder.config)
    assert isinstance(readout, AtomisticReadout)


def test_mace_builder_shallow_ensemble_plumbs_n_members():
    from apax.layers.readout import MaceReadout

    cfg = _minimal_cfg()
    cfg["ensemble"] = {
        "kind": "shallow",
        "n_members": 4,
        "force_variance": True,
        "chunk_size": None,
    }
    builder = MaceBuilder(cfg, n_species=5)
    readout = builder.build_readout(builder.config)
    assert isinstance(readout, MaceReadout)
    assert readout.n_shallow_ensemble == 4


def test_mace_builder_feature_fn_uses_atomistic_readout():
    from apax.layers.readout import AtomisticReadout

    builder = MaceBuilder(_minimal_cfg(), n_species=5)
    readout = builder.build_readout(builder.config, is_feature_fn=True)
    assert isinstance(readout, AtomisticReadout)


def test_mace_builder_end_to_end_energy_derivative_model():
    cfg = _minimal_cfg()
    builder = MaceBuilder(cfg, n_species=5)
    model = builder.build_energy_derivative_model()

    n_atoms = 3
    R = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    Z = jnp.array([1, 2, 3], dtype=jnp.int32)
    neighbor = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32).T
    box = jnp.zeros((3,))
    offsets = jnp.zeros((neighbor.shape[1], 3))

    params = model.init(jax.random.PRNGKey(0), R, Z, neighbor, box, offsets)
    out = model.apply(params, R, Z, neighbor, box, offsets)
    assert "energy" in out
    assert "forces" in out
    assert out["forces"].shape == (n_atoms, 3)
    assert np.all(np.isfinite(np.asarray(out["forces"])))
