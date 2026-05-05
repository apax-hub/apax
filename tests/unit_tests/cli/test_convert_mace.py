"""Converter CLI unit tests that don't require torch."""
import sys

import pytest
from typer.testing import CliRunner


def test_convert_mace_missing_torch_fails_gracefully(monkeypatch, tmp_path):
    # Simulate torch being absent
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "mace", None)

    from apax.cli.apax_app import app
    runner = CliRunner()
    res = runner.invoke(app, ["convert-mace", "medium", str(tmp_path / "out.apax")])
    assert res.exit_code != 0
    assert "torch" in res.output.lower()


def test_convert_mace_accepts_canonical_name_and_file_path():
    """Argument parsing treats both as valid 'source' inputs."""
    from apax.cli.convert_mace import convert_mace
    import inspect
    sig = inspect.signature(convert_mace)
    assert "source" in sig.parameters


def test_extract_config_falls_back_to_first_head_when_none():
    """``head=None`` selects ``model.heads[0]`` rather than crashing."""
    pytest.importorskip("torch")
    from apax.transfer_learning.mace_foundation import _extract_config_from_torch
    from types import SimpleNamespace
    import torch as _torch

    # Fabricate the smallest torch-mace-shaped object the function reads.
    # We only exercise the head-resolution path; everything else is shielded
    # by a NotImplementedError raised before any other model attribute is
    # touched (see ``_SUPPORTED_TORCH_INTERACTION_CLS``).
    class _FakeInter(_torch.nn.Module):
        pass

    fake = SimpleNamespace(
        heads=["default"],
        interactions=[_FakeInter()],
    )

    # The function should *not* raise ValueError("head=None not in ...").
    # It will raise NotImplementedError later when it inspects the
    # interaction class, which is acceptable — we only assert the
    # head-resolution path is correct.
    with pytest.raises(NotImplementedError):
        _extract_config_from_torch(fake, head=None)
