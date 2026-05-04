"""MaceModelConfig pydantic schema — smoke tests."""
import pytest
from apax.config.model_config import MaceModelConfig


def test_mace_model_config_defaults():
    cfg = MaceModelConfig()
    assert cfg.name == "mace"
    assert cfg.readout_kind == "mace"
    assert cfg.MLP_irreps == "16x0e"
    assert cfg.hidden_irreps == "128x0e + 128x1o"
    assert cfg.num_interactions == 2


def test_mace_model_config_readout_kind_literal():
    cfg = MaceModelConfig(readout_kind="standard")
    assert cfg.readout_kind == "standard"
    with pytest.raises(Exception, match="readout_kind"):
        MaceModelConfig(readout_kind="garbage")


def test_mace_model_config_has_no_removed_fields():
    """Regression: pretrained / freeze_backbone / num_elements must be gone."""
    fields = MaceModelConfig.model_fields
    assert "pretrained" not in fields
    assert "freeze_backbone" not in fields
    assert "unfreeze_backbone_epoch" not in fields
    assert "num_elements" not in fields


def test_mace_model_config_interaction_cls_only_residual():
    """Schema enforces the only currently-implemented interaction variant.

    Pydantic narrows the field rather than relying on a runtime guard inside
    ``InteractionBlock``, so unsupported variants (Density, non-residual) get
    rejected at config-validation time with a clear message — not 30s later
    inside the JIT-traced forward pass.
    """
    cfg = MaceModelConfig(interaction_cls="RealAgnosticResidual")
    assert cfg.interaction_cls == "RealAgnosticResidual"
    for unsupported in (
        "RealAgnostic",
        "RealAgnosticDensity",
        "RealAgnosticDensityResidual",
        "garbage",
    ):
        with pytest.raises(Exception, match="interaction_cls"):
            MaceModelConfig(interaction_cls=unsupported)
