"""MaceModelConfig pydantic schema — nested-shape smoke tests.

The detailed round-trip + rejection coverage lives in test_mace_nested_schema.py.
"""

import pytest

from apax.config.model_config import MaceModelConfig


def test_mace_model_config_defaults():
    cfg = MaceModelConfig()
    assert cfg.name == "mace"
    assert cfg.readout.MLP_irreps == "16x0e"
    assert cfg.descriptor.hidden_irreps == "128x0e + 128x1o"
    assert len(cfg.descriptor.interactions) == 2


def test_mace_model_config_has_no_removed_fields():
    """Regression: pretrained / freeze_backbone / num_elements must be gone."""
    fields = MaceModelConfig.model_fields
    assert "pretrained" not in fields
    assert "freeze_backbone" not in fields
    assert "unfreeze_backbone_epoch" not in fields
    assert "num_elements" not in fields


def test_mace_readout_config_rejects_unknown_field():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        MaceModelConfig(readout={"kind": "mace"})


def test_property_head_defaults():
    from apax.config.model_config import PropertyHead

    head = PropertyHead(name="charges")
    assert head.MLP_irreps == "16x0e"


def test_property_head_rejects_unknown_field():
    import pydantic

    from apax.config.model_config import PropertyHead

    with pytest.raises(pydantic.ValidationError):
        PropertyHead(name="charges", kind="mace")
