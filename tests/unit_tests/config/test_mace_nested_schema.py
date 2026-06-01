"""MaceModelConfig nested schema — round-trip + rejection tests."""

import pytest
import yaml

from apax.config.model_config import (
    MaceDescriptorConfig,
    MaceModelConfig,
    MaceRadialEmbeddingConfig,
    MaceReadoutConfig,
    RealAgnosticResidualConfig,
)


def test_mace_model_config_default_basis_overridden_to_standard():
    """MACE basis defaults to (variant=standard, n_basis=8, r_max=5.0)."""
    cfg = MaceModelConfig()
    assert cfg.basis.name == "bessel"
    assert cfg.basis.variant == "standard"
    assert cfg.basis.n_basis == 8
    assert cfg.basis.r_max == 5.0


def test_mace_model_config_default_subgroups_present():
    cfg = MaceModelConfig()
    assert isinstance(cfg.radial_embedding, MaceRadialEmbeddingConfig)
    assert isinstance(cfg.descriptor, MaceDescriptorConfig)
    assert isinstance(cfg.readout, MaceReadoutConfig)
    assert cfg.descriptor.max_ell == 3
    assert cfg.descriptor.hidden_irreps == "128x0e + 128x1o"
    assert cfg.descriptor.correlation == 3
    assert cfg.readout.MLP_irreps == "16x0e"


def test_mace_model_config_default_interactions_two_residual():
    cfg = MaceModelConfig()
    assert len(cfg.descriptor.interactions) == 2
    for inter in cfg.descriptor.interactions:
        assert isinstance(inter, RealAgnosticResidualConfig)
        assert inter.name == "RealAgnosticResidual"


def test_mace_model_config_yaml_roundtrip_with_three_variants():
    yml = """
name: mace
basis:
  name: bessel
  variant: standard
  n_basis: 10
  r_max: 6.0
radial_embedding:
  num_polynomial_cutoff: 5
  distance_transform: null
descriptor:
  max_ell: 3
  hidden_irreps: 128x0e + 128x1o
  correlation: 3
  interactions:
    - name: RealAgnosticResidual
    - name: RealAgnosticDensity
    - name: RealAgnosticDensityResidual
  avg_num_neighbors: 62.0
readout:
  MLP_irreps: 16x0e
"""
    raw = yaml.safe_load(yml)
    cfg = MaceModelConfig.model_validate(raw)
    assert [i.name for i in cfg.descriptor.interactions] == [
        "RealAgnosticResidual",
        "RealAgnosticDensity",
        "RealAgnosticDensityResidual",
    ]


def test_mace_model_config_rejects_bare_string_interaction():
    """Schema requires the ``name:`` wrap; bare strings are not accepted."""
    with pytest.raises(Exception):
        MaceModelConfig(
            descriptor={"interactions": ["RealAgnosticResidual"]},
        )


def test_mace_model_config_rejects_unknown_variant_name():
    with pytest.raises(Exception, match="name"):
        MaceModelConfig(
            descriptor={"interactions": [{"name": "NotARealVariant"}]},
        )


def test_distance_transform_is_discriminated_union():
    """distance_transform is a name-discriminated union (consistent with the
    sibling InteractionConfig), and the discriminator surfaces in the schema."""
    from apax.config.model_config import AgnesiTransformConfig

    cfg = MaceRadialEmbeddingConfig(
        num_polynomial_cutoff=5,
        distance_transform={"name": "agnesi", "a": 2.0},
    )
    assert isinstance(cfg.distance_transform, AgnesiTransformConfig)
    assert cfg.distance_transform.a == 2.0

    schema = MaceRadialEmbeddingConfig.model_json_schema()
    assert "discriminator" in str(schema)

    with pytest.raises(Exception, match="name"):
        MaceRadialEmbeddingConfig(
            num_polynomial_cutoff=5,
            distance_transform={"name": "not-a-transform"},
        )


def test_mace_model_config_rejects_empty_interactions_list():
    with pytest.raises(Exception):
        MaceModelConfig(descriptor={"interactions": []})


def test_mace_model_config_has_no_removed_flat_fields():
    fields = MaceModelConfig.model_fields
    for removed in (
        "r_max",
        "num_bessel",
        "num_polynomial_cutoff",
        "max_ell",
        "hidden_irreps",
        "num_interactions",
        "correlation",
        "interaction_cls",
        "use_cueq",
        "readout_kind",
        "MLP_irreps",
        "avg_num_neighbors",
        "distance_transform",
    ):
        assert removed not in fields, f"{removed!r} must move to a sub-config"
    for kept in ("basis", "radial_embedding", "descriptor", "readout"):
        assert kept in fields, f"{kept!r} must be a top-level group"


def test_existing_models_default_to_kocer_variant():
    """GMNN / EquivMP / So3krates inherit ``variant=kocer`` from BaseModelConfig."""
    from apax.config.model_config import (
        EquivMPConfig,
        GMNNConfig,
        So3kratesConfig,
    )

    for ModelCls in (GMNNConfig, EquivMPConfig, So3kratesConfig):
        cfg = ModelCls()
        assert cfg.basis.name == "bessel"
        assert cfg.basis.variant == "kocer"
