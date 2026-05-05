"""MaceModelConfig pydantic schema — nested-shape smoke tests.

The detailed round-trip + rejection coverage lives in test_mace_nested_schema.py.
This file keeps the original spirit (defaults, removed legacy fields) under the
nested shape.
"""

import pytest

from apax.config.model_config import MaceModelConfig


def test_mace_model_config_defaults():
    cfg = MaceModelConfig()
    assert cfg.name == "mace"
    assert cfg.readout.kind == "mace"
    assert cfg.readout.MLP_irreps == "16x0e"
    assert cfg.descriptor.hidden_irreps == "128x0e + 128x1o"
    assert len(cfg.descriptor.interactions) == 2


def test_mace_model_config_readout_kind_literal():
    cfg = MaceModelConfig(readout={"kind": "standard"})
    assert cfg.readout.kind == "standard"
    with pytest.raises(Exception, match="kind"):
        MaceModelConfig(readout={"kind": "garbage"})


def test_mace_model_config_has_no_removed_fields():
    """Regression: pretrained / freeze_backbone / num_elements must be gone."""
    fields = MaceModelConfig.model_fields
    assert "pretrained" not in fields
    assert "freeze_backbone" not in fields
    assert "unfreeze_backbone_epoch" not in fields
    assert "num_elements" not in fields


def test_property_head_default_kind_is_standard():
    from apax.config.model_config import PropertyHead

    head = PropertyHead(name="charges")
    assert head.kind == "standard"
    assert head.MLP_irreps == "16x0e"


def test_property_head_kind_mace_accepted():
    from apax.config.model_config import PropertyHead

    head = PropertyHead(name="charges", kind="mace", MLP_irreps="32x0e")
    assert head.kind == "mace"
    assert head.MLP_irreps == "32x0e"


def test_property_head_kind_invalid_rejected():
    import pydantic

    from apax.config.model_config import PropertyHead

    with pytest.raises(pydantic.ValidationError):
        PropertyHead(name="charges", kind="bogus")
