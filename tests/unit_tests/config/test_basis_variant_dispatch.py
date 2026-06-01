"""BesselBasisConfig.variant + ModelBuilder.build_basis_function dispatch."""

import jax
import jax.numpy as jnp
import pytest

from apax.config.model_config import BesselBasisConfig
from apax.layers.descriptor.basis_functions import BesselBasis, MaceBesselBasis
from apax.nn.builder import MaceBuilder, ModelBuilder


def _basis_cfg(variant: str, n_basis: int, r_max: float) -> dict:
    return {
        "basis": {
            "name": "bessel",
            "variant": variant,
            "n_basis": n_basis,
            "r_max": r_max,
        },
        "descriptor_dtype": "fp32",
    }


def _builder_with_basis(variant: str, n_basis: int = 4, r_max: float = 5.0):
    # The base builder handles the apax-native "kocer" bessel; the MACE-specific
    # "standard" variant is owned by MaceBuilder.build_basis_function.
    cfg = _basis_cfg(variant, n_basis, r_max)
    if variant == "standard":
        return MaceBuilder(cfg, n_species=5)
    return ModelBuilder(cfg, n_species=5)


def test_bessel_basis_config_default_variant_is_kocer():
    cfg = BesselBasisConfig()
    assert cfg.variant == "kocer"


def test_bessel_basis_config_accepts_kocer_and_standard():
    assert BesselBasisConfig(variant="kocer").variant == "kocer"
    assert BesselBasisConfig(variant="standard").variant == "standard"


def test_bessel_basis_config_rejects_unknown_variant():
    with pytest.raises(Exception, match="variant"):
        BesselBasisConfig(variant="garbage")


def test_build_basis_function_dispatches_kocer_to_BesselBasis():
    builder = _builder_with_basis("kocer", n_basis=4, r_max=5.0)
    basis = builder.build_basis_function()
    assert isinstance(basis, BesselBasis)
    assert basis.n_basis == 4
    assert basis.r_max == 5.0


def test_build_basis_function_dispatches_standard_to_MaceBesselBasis():
    builder = _builder_with_basis("standard", n_basis=8, r_max=6.0)
    basis = builder.build_basis_function()
    assert isinstance(basis, MaceBesselBasis)
    assert basis.n_basis == 8
    assert basis.r_max == 6.0


def test_build_basis_function_kocer_and_standard_are_numerically_distinct():
    """Kocer (apax legacy) and standard (torch-mace) bessels differ at any r."""
    kocer = _builder_with_basis("kocer", n_basis=4, r_max=5.0).build_basis_function()
    standard = _builder_with_basis(
        "standard", n_basis=4, r_max=5.0
    ).build_basis_function()

    r = jnp.array([1.5])
    k_params = kocer.init(jax.random.PRNGKey(0), r)
    s_params = standard.init(jax.random.PRNGKey(0), r)
    k_out = kocer.apply(k_params, r)
    s_out = standard.apply(s_params, r)
    assert k_out.shape == s_out.shape == (1, 4)
    # Different formulas at the same r should not coincide.
    assert not jnp.allclose(k_out, s_out, atol=1e-6)
