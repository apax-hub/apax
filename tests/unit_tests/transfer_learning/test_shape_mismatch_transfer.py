"""Shape-mismatch transfer learning — error on mismatch, skip via reset_layers.

Covers the new contract added by the (a) follow-up to PR #558:
``black_list_param_transfer`` must raise ``TransferLearningShapeMismatchError``
when a source leaf cannot be written into the target without changing shape,
and ``reset_layers`` entries may be either the legacy ``p[-2]`` suffix or the
full ``/``-joined leaf path.
"""
import re

import numpy as np
import pytest

from apax.transfer_learning import (
    TransferLearningShapeMismatchError,
    black_list_param_transfer,
)


def _params(shapes: dict) -> dict:
    """Build a 2-level nested params pytree from ``{leaf_path: shape}``.

    ``leaf_path`` is a ``/``-joined string. Trailing component is the leaf
    name; everything before is nested dict keys.
    """
    out = {}
    for path, shape in shapes.items():
        keys = path.split("/")
        cursor = out
        for k in keys[:-1]:
            cursor = cursor.setdefault(k, {})
        cursor[keys[-1]] = np.zeros(shape)
    return out


def test_matching_shapes_transfer_succeeds():
    src = _params({"params/dense/kernel": (4, 8), "params/dense/bias": (8,)})
    tgt = _params({"params/dense/kernel": (4, 8), "params/dense/bias": (8,)})
    src["params"]["dense"]["kernel"][:] = 1.0
    src["params"]["dense"]["bias"][:] = 2.0

    out = black_list_param_transfer(src, tgt, [])

    assert np.all(np.asarray(out["params"]["dense"]["kernel"]) == 1.0)
    assert np.all(np.asarray(out["params"]["dense"]["bias"]) == 2.0)


def test_mismatched_shapes_raise_with_actionable_message():
    src = _params({"params/readout/readout_0/linear/kernel": (128, 1)})
    tgt = _params({"params/readout/readout_0/linear/kernel": (128, 8)})

    with pytest.raises(TransferLearningShapeMismatchError) as excinfo:
        black_list_param_transfer(src, tgt, [])

    msg = str(excinfo.value)
    assert "params/readout/readout_0/linear/kernel" in msg
    assert "(128, 1)" in msg
    assert "(128, 8)" in msg
    assert "reset_layers:" in msg
    assert re.search(
        r"-\s*params/readout/readout_0/linear/kernel", msg
    ), f"missing yaml-ready bullet in error message:\n{msg}"


def test_full_leaf_path_in_reset_layers_skips_transfer():
    src = _params({"params/readout/readout_0/linear/kernel": (128, 1)})
    tgt = _params({"params/readout/readout_0/linear/kernel": (128, 8)})
    tgt["params"]["readout"]["readout_0"]["linear"]["kernel"][:] = 99.0

    out = black_list_param_transfer(
        src, tgt, ["params/readout/readout_0/linear/kernel"]
    )

    leaf = np.asarray(out["params"]["readout"]["readout_0"]["linear"]["kernel"])
    assert leaf.shape == (128, 8)
    assert np.all(leaf == 99.0)


def test_legacy_suffix_in_reset_layers_skips_transfer():
    src = _params({"params/dense/kernel": (4, 8), "params/basis/emb": (3,)})
    tgt = _params({"params/dense/kernel": (4, 8), "params/basis/emb": (3,)})
    src["params"]["dense"]["kernel"][:] = 1.0
    src["params"]["basis"]["emb"][:] = 1.0
    tgt["params"]["basis"]["emb"][:] = 99.0

    out = black_list_param_transfer(src, tgt, ["basis"])

    assert np.all(np.asarray(out["params"]["dense"]["kernel"]) == 1.0)
    assert np.all(np.asarray(out["params"]["basis"]["emb"]) == 99.0)


def test_multiple_mismatches_collected_in_one_error():
    src = _params({
        "params/a/kernel": (4, 1),
        "params/b/kernel": (8, 1),
        "params/c/kernel": (16, 1),
    })
    tgt = _params({
        "params/a/kernel": (4, 4),
        "params/b/kernel": (8, 4),
        "params/c/kernel": (16, 4),
    })

    with pytest.raises(TransferLearningShapeMismatchError) as excinfo:
        black_list_param_transfer(src, tgt, [])

    msg = str(excinfo.value)
    for path in ("params/a/kernel", "params/b/kernel", "params/c/kernel"):
        assert path in msg, f"missing {path} in:\n{msg}"


def test_full_path_skip_does_not_trigger_legacy_match():
    """Full path skip is matched independently; ensure no double-skip false positive."""
    src = _params({"params/foo/bar": (3,)})
    tgt = _params({"params/foo/bar": (3,)})
    src["params"]["foo"]["bar"][:] = 7.0

    out = black_list_param_transfer(src, tgt, ["params/foo/bar"])

    assert np.all(np.asarray(out["params"]["foo"]["bar"]) == 0.0)
