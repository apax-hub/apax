"""MaceBuilder wiring — verifies the builder produces a runnable EnergyModel."""
import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def mace_config_dict():
    """Minimal MACE config dict, matching what ModelBuilder expects.

    All keys from BaseModelConfig plus MaceModelConfig. Values kept tiny
    so init and forward pass are fast.
    """
    return {
        "name": "mace",
        "r_max": 5.0,
        "num_bessel": 4,
        "num_polynomial_cutoff": 5,
        "max_ell": 1,
        "hidden_irreps": "8x0e",
        "num_interactions": 1,
        "correlation": 2,
        "interaction_cls": "RealAgnosticResidual",
        "use_cueq": False,
        "pretrained": None,
        "freeze_backbone": False,
        "unfreeze_backbone_epoch": None,
        # BaseModelConfig fields
        "descriptor_dtype": "fp32",
        "readout_dtype": "fp32",
        "scale_shift_dtype": "fp64",
        "activation_fn": "silu",
        "nn": [16, 16],
        "b_init": "zeros",
        "w_init": "lecun",
        "use_ntk": False,
        "basis": {"name": "bessel", "n_basis": 4, "r_max": 5.0},
        "ensemble": None,
        "property_heads": [],
        "empirical_corrections": [],
        "calc_stress": False,
    }


def test_mace_builder_builds_energy_model(mace_config_dict):
    from apax.nn.builder import MaceBuilder

    builder = MaceBuilder(mace_config_dict, n_species=10)
    model = builder.build_energy_model()
    assert model.representation is not None
    assert model.readout is not None
    assert model.scale_shift is not None


def test_mace_builder_get_builder_dispatch(mace_config_dict):
    """MaceModelConfig.get_builder() must return MaceBuilder."""
    from apax.config.model_config import MaceModelConfig
    from apax.nn.builder import MaceBuilder

    # Only keep MaceModelConfig fields for this pydantic instantiation
    mace_only = {
        k: v for k, v in mace_config_dict.items()
        if k in MaceModelConfig.model_fields
    }
    cfg = MaceModelConfig(**mace_only)
    assert cfg.get_builder() is MaceBuilder


def test_mace_builder_derivative_model_runs(mace_config_dict):
    from apax.nn.builder import MaceBuilder

    builder = MaceBuilder(mace_config_dict, n_species=10)
    model = builder.build_energy_derivative_model()
    # Forward init on a tiny system
    n_atoms = 4
    R = jnp.zeros((n_atoms, 3))
    Z = jnp.array([1, 8, 1, 6], dtype=jnp.int32)
    idx = jnp.array(
        [[0, 0, 0, 1, 1, 2, 2, 3, 3, 3],
         [1, 2, 3, 0, 2, 0, 3, 0, 1, 2]],
        dtype=jnp.int32,
    )
    # EnergyDerivativeModel.__call__ signature: (R, Z, neighbor, box, offsets)
    params = model.init(
        jax.random.PRNGKey(0),
        R, Z, idx, jnp.zeros(3), jnp.zeros((idx.shape[1], 3)),
    )
    out = model.apply(
        params,
        R, Z, idx, jnp.zeros(3), jnp.zeros((idx.shape[1], 3)),
    )
    # EnergyDerivativeModel returns a dict with energy, forces
    assert out is not None
