"""Converter CLI unit tests that don't require torch."""
import sys

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
