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
        },
        readout={"MLP_irreps": "16x0e"},
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
    assert readout.MLP_irreps == "16x0e"


def test_mace_builder_energy_readout_dispatch_is_structural_not_identity():
    """The energy-head readout is selected structurally (presence of the nested
    'readout' config), not by `head_config is self.config` object identity, so an
    equal-but-not-identical config dict still routes to the energy branch."""
    from apax.layers.readout import MaceReadout

    builder = MaceBuilder(_minimal_cfg(), n_species=5)
    # shallow copy: same content, different object. The old identity check
    # (`head_config is self.config`) would misroute this to the property branch
    # and raise KeyError on the missing top-level 'MLP_irreps'.
    cfg_copy = dict(builder.config)
    readout = builder.build_readout(cfg_copy)
    assert isinstance(readout, MaceReadout)
    assert readout.MLP_irreps == cfg_copy["readout"]["MLP_irreps"]


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


def test_mace_builder_feature_fn_returns_none():
    """The MACE feature path must use the raw (trained) descriptor features,
    not a fresh AtomisticReadout whose params the converter/training never fill.
    ``FeatureModel`` skips the readout when it is falsy, so ``None`` yields the
    descriptor output directly."""
    builder = MaceBuilder(_minimal_cfg(), n_species=5)
    readout = builder.build_readout(builder.config, is_feature_fn=True)
    assert readout is None


def test_mace_builder_feature_fn_partial_layers_raises():
    """Partial-layer feature extraction is not yet supported for MACE and must
    fail loudly rather than silently returning wrong features."""
    import pytest

    builder = MaceBuilder(_minimal_cfg(), n_species=5)
    with pytest.raises(NotImplementedError):
        builder.build_readout(builder.config, is_feature_fn=True, only_use_n_layers=1)


def test_mace_build_feature_model_returns_descriptor_features():
    """build_feature_model must produce the per-atom descriptor features
    (n_atoms, num_interactions * hidden_dim) with no extra readout params."""
    builder = MaceBuilder(_minimal_cfg(), n_species=5)
    model = builder.build_feature_model(should_average=False)

    n_atoms = 3
    R = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    Z = jnp.array([1, 2, 3], dtype=jnp.int32)
    neighbor = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32).T
    box = jnp.zeros((3,))
    offsets = jnp.zeros((neighbor.shape[1], 3))

    params = model.init(jax.random.PRNGKey(0), R, Z, neighbor, box, offsets)
    features = model.apply(params, R, Z, neighbor, box, offsets)

    # 2 interactions * hidden_dim 8 (8x0e) = 16 scalar features per atom
    assert features.shape == (n_atoms, 16)
    assert np.all(np.isfinite(np.asarray(features)))
    # readout=None -> the feature model carries only the descriptor params
    assert "readout" not in params["params"]
    assert "representation" in params["params"]


def test_mace_builder_uses_mace_bessel_basis():
    """MACE owns the 'standard' bessel variant; the builder returns the
    torch-mace-faithful MaceBesselBasis."""
    from apax.layers.descriptor.basis_functions import MaceBesselBasis

    builder = MaceBuilder(_minimal_cfg(), n_species=5)
    basis = builder.build_basis_function()
    assert isinstance(basis, MaceBesselBasis)


def test_base_builder_does_not_handle_mace_bessel_variant():
    """The shared base builder must not know about MACE-specific bases; the
    'standard' variant is rejected so the leak stays inside MaceBuilder."""
    import pytest

    from apax.nn.builder import ModelBuilder

    cfg = {
        "basis": {"name": "bessel", "variant": "standard", "n_basis": 4, "r_max": 5.0},
        "descriptor_dtype": "fp32",
    }
    with pytest.raises(ValueError):
        ModelBuilder(cfg, n_species=5).build_basis_function()


def test_converter_threads_builder_default_n_species():
    """The foundation converter uses the builder's canonical element-table size,
    not a private duplicate constant."""
    from apax.nn import builder
    from apax.transfer_learning import mace_foundation

    assert mace_foundation.DEFAULT_N_SPECIES == builder.DEFAULT_N_SPECIES == 119


def test_mace_hidden_irreps_without_scalars_raises():
    """MACE node features require a 0e (scalar) component; a hidden_irreps with
    no scalars must raise a clear ValueError rather than fail obscurely."""
    import pytest

    cfg = _minimal_cfg()
    cfg["descriptor"]["hidden_irreps"] = "8x1o"  # no 0e
    builder = MaceBuilder(cfg, n_species=5)
    model = builder.build_energy_derivative_model()

    R = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    Z = jnp.array([1, 2, 3], dtype=jnp.int32)
    neighbor = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32).T
    box = jnp.zeros((3,))
    offsets = jnp.zeros((neighbor.shape[1], 3))

    with pytest.raises(ValueError):
        model.init(jax.random.PRNGKey(0), R, Z, neighbor, box, offsets)


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
