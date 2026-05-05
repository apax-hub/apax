# MACE Transfer-Learning Shape Mismatch + Property-Head Kind — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close three follow-ups to the MACE foundation integration: (a) make `black_list_param_transfer` raise an actionable error on shape mismatch instead of silently broadcasting; (b)+(c) add a `PropertyHead.kind` discriminator so MACE property heads route through `MaceReadout` and cross-model misuse errors loudly; (d) update the fine-tune template comment to describe the run-once-paste-`reset_layers` workflow.

**Architecture:** Three independent edits that share a single review surface. (a) `black_list_param_transfer` collects shape mismatches into a new `TransferLearningShapeMismatchError` exception and extends `reset_layers` to additionally accept full `/`-joined leaf paths (legacy `p[-2]` suffix still works). (b)+(c) `PropertyHead` gains `kind: Literal["standard", "mace"] = "standard"` and `MLP_irreps: str = "16x0e"`; `MaceBuilder.build_readout` and `ModelBuilder.build_readout` discriminate energy head vs property head via identity check `head_config is self.config` and dispatch. (d) Template preamble comment rewrite — no code changes.

**Tech Stack:** Python 3.11+, `uv` for envs, pydantic v2, flax linen, e3nn-jax, pytest.

---

## File Structure

| File | Role |
|---|---|
| `apax/transfer_learning/parameter_transfer.py` | Add `TransferLearningShapeMismatchError`; rewrite `black_list_param_transfer` to do shape checks and accept full-path entries in `reset_layers`. |
| `apax/transfer_learning/__init__.py` | Re-export `TransferLearningShapeMismatchError`. |
| `apax/config/model_config.py` | Add `kind: Literal["standard", "mace"] = "standard"` and `MLP_irreps: str = "16x0e"` fields to `PropertyHead`. |
| `apax/nn/builder.py` | `ModelBuilder.build_readout`: guard `kind="mace"` on non-MACE models with a loud error. `MaceBuilder.build_readout`: discriminate energy head vs property head via `head_config is self.config`; dispatch property heads on `head_config["kind"]`. |
| `apax/cli/templates/mace_finetune_minimal.yaml` | Preamble comment rewrite — no code changes. |
| `tests/unit_tests/transfer_learning/test_shape_mismatch_transfer.py` (new) | Unit tests for (a). |
| `tests/unit_tests/nn/test_property_heads.py` (new) | Unit tests for (b)+(c). |
| `tests/integration_tests/mace/test_mace_finetune.py` | Add two parametrized cases for the foundation→ensemble shape-mismatch workflow. |

---

## Task 1: Add `TransferLearningShapeMismatchError` and shape-checked transfer

**Files:**
- Modify: `apax/transfer_learning/parameter_transfer.py`
- Modify: `apax/transfer_learning/__init__.py`
- Test: `tests/unit_tests/transfer_learning/test_shape_mismatch_transfer.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit_tests/transfer_learning/test_shape_mismatch_transfer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit_tests/transfer_learning/test_shape_mismatch_transfer.py -v`
Expected: ImportError on `TransferLearningShapeMismatchError` (not yet defined).

- [ ] **Step 3: Implement the new exception and rewrite `black_list_param_transfer`**

Replace the whole body of `apax/transfer_learning/parameter_transfer.py` with:

```python
import logging
from typing import Union

from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict

from apax.config.train_config import TransferLearningConfig
from apax.train.checkpoints import load_params

log = logging.getLogger(__name__)


class TransferLearningShapeMismatchError(ValueError):
    """Raised when source and target params disagree on a leaf's shape.

    The message names every offending leaf with both shapes and emits a
    ready-to-paste ``reset_layers:`` YAML snippet.
    """


def _format_mismatch_error(mismatches: list[tuple[tuple, tuple, tuple]]) -> str:
    """Build the actionable error message body.

    Parameters
    ----------
    mismatches
        List of ``(leaf_path_tuple, source_shape, target_shape)`` triples.
    """
    lines = [
        f"Transfer learning shape mismatch on {len(mismatches)} parameter(s):",
        "",
    ]
    for path, src_shape, tgt_shape in mismatches:
        joined = "/".join(map(str, path))
        lines.append(f"  {joined}")
        lines.append(f"    source: {tuple(src_shape)}   target: {tuple(tgt_shape)}")
    lines.append("")
    lines.append(
        "Add the following to your config to re-initialize these slots from "
        "the model's default initialization:"
    )
    lines.append("")
    lines.append("  transfer_learning:")
    lines.append("    reset_layers:")
    for path, _src, _tgt in mismatches:
        lines.append(f"      - {'/'.join(map(str, path))}")
    return "\n".join(lines)


def black_list_param_transfer(
    source_params: Union[FrozenDict, dict],
    target_params: Union[FrozenDict, dict],
    param_black_list: list[str],
) -> FrozenDict:
    """Transfer parameters from one pytree to another with shape checking.

    A leaf at path ``p`` is written iff ``shape(source[p]) == shape(target[p])``
    AND the leaf is not blacklisted. A leaf is blacklisted if either:

    - ``p[-2]`` matches an entry in ``param_black_list`` (legacy suffix form), or
    - the ``/``-joined full path matches an entry (new full-path form).

    Any leaf that is reachable in both trees but has mismatched shape and is
    NOT blacklisted is collected and reported via
    :class:`TransferLearningShapeMismatchError` in a single raise at the end.

    Parameters
    ----------
    source_params
        Source pytree (e.g. converted foundation checkpoint).
    target_params
        Freshly-initialized target pytree.
    param_black_list
        Layer names or full leaf paths to skip during transfer.

    Returns
    -------
    FrozenDict
        ``target_params`` with matching-shape source leaves written in.

    Raises
    ------
    TransferLearningShapeMismatchError
        When any non-blacklisted leaf has a shape mismatch.
    """
    source_params = unfreeze(source_params)
    target_params = unfreeze(target_params)

    flat_source = flatten_dict(source_params)
    flat_target = flatten_dict(target_params)

    mismatches: list[tuple[tuple, tuple, tuple]] = []

    for p, v in flat_source.items():
        full_path = "/".join(map(str, p))
        is_blacklisted = (
            (len(p) >= 2 and p[-2] in param_black_list)
            or full_path in param_black_list
        )
        if is_blacklisted:
            log.info("Skipping (reset_layers): %s", full_path)
            continue

        if p not in flat_target:
            log.info("Skipping (no target slot): %s", full_path)
            continue

        src_shape = tuple(getattr(v, "shape", ()))
        tgt_shape = tuple(getattr(flat_target[p], "shape", ()))
        if src_shape != tgt_shape:
            mismatches.append((p, src_shape, tgt_shape))
            continue

        flat_target[p] = v
        log.info("Transferring parameter: %s", full_path)

    if mismatches:
        raise TransferLearningShapeMismatchError(_format_mismatch_error(mismatches))

    transfered_target = unflatten_dict(flat_target)
    transfered_target = freeze(transfered_target)
    return transfered_target


def transfer_parameters(
    state: TrainState, ckpt_config: TransferLearningConfig
) -> TrainState:
    """Transfer the parameters from the checkpoint to the train state.

    Parameters
    ----------
    state
        Train state with freshly-initialized parameters.
    ckpt_config
        Transfer learning configuration.

    Returns
    -------
    TrainState
        ``state`` with ``params`` updated according to the source checkpoint.
    """
    source_params = load_params(ckpt_config.base_model_checkpoint)
    log.info("Transferring parameters from %s", ckpt_config.base_model_checkpoint)
    params = black_list_param_transfer(
        source_params, state.params, ckpt_config.reset_layers
    )
    state = state.replace(params=params)
    return state
```

- [ ] **Step 4: Re-export the new exception**

Replace the whole body of `apax/transfer_learning/__init__.py` with:

```python
from apax.transfer_learning.parameter_transfer import (
    TransferLearningShapeMismatchError,
    black_list_param_transfer,
    transfer_parameters,
)

__all__ = [
    "TransferLearningShapeMismatchError",
    "black_list_param_transfer",
    "transfer_parameters",
]
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `uv run pytest tests/unit_tests/transfer_learning/test_shape_mismatch_transfer.py -v`
Expected: 6 passing.

- [ ] **Step 6: Run the existing transfer-learning suite to confirm no regressions**

Run: `uv run pytest tests/unit_tests/transfer_learning/ -v`
Expected: all green, including `test_param_transfer` (which uses the legacy `["basis"]` suffix form).

- [ ] **Step 7: Commit**

```bash
git add apax/transfer_learning/parameter_transfer.py \
        apax/transfer_learning/__init__.py \
        tests/unit_tests/transfer_learning/test_shape_mismatch_transfer.py
git commit -m "feat(transfer): raise actionable shape-mismatch error and accept full leaf paths in reset_layers"
```

---

## Task 1.5: Sibling-orphan structural-mismatch detection

> **Why this task exists:** Task 4's integration tests revealed that `e3nn_jax`'s `Linear` layer encodes the input AND output irreps in the parameter dict key string (`w[0,0] 8x0e,1x0e` vs `w[0,0] 8x0e,4x0e`). When `n_shallow_ensemble` widens, the readout's final-layer params live at *different* paths on source vs target — sibling leaves under the same parent, not the same path with different shapes. Task 1's shape-mismatch detection therefore never fires for the canonical foundation→ensemble workflow; the source readout params silently fall into the "no target slot" branch, the target slots stay random-init, training proceeds, and the user has a half-broken transfer with no signal. This task closes that gap.

**Files:**
- Modify: `apax/transfer_learning/parameter_transfer.py`
- Test: `tests/unit_tests/transfer_learning/test_shape_mismatch_transfer.py` (extend)

### Detailed design

`black_list_param_transfer` runs **two** mismatch passes before raising:

1. **Structural mismatch (new):** Group flattened source and target leaves by parent path (`p[:-1]`). For every parent, compute `src_only = src_leaves - tgt_leaves` and `tgt_only = tgt_leaves - src_leaves`, filtering both sets through `reset_layers`. If **both** sets are non-empty for a given parent, that's a structural mismatch.

2. **Path-shape mismatch (existing — unchanged):** For leaves present in both source and target with mismatched shapes, record (path, src_shape, tgt_shape).

`reset_layers` semantic extension: a leaf at path `p` is considered "blacklisted" (excluded from transfer + excluded from both mismatch passes) if **any** of these hold:
- `len(p) >= 2 and p[-2] in reset_layers` (legacy suffix form, unchanged)
- `"/".join(map(str, p)) in reset_layers` (full path, Task 1's extension, unchanged)
- The same full-path match applies to **target** paths during the structural-mismatch pass (new — lets users paste target keys to silence the structural error).

Combined error message:

```
Transfer learning mismatch on N parameter slot(s):

Structural mismatches (source/target diverge under same parent):
  params/energy_model/readout/readout_0/linear/
    source: w[0,0] 8x0e,1x0e   shape=(8, 1)
    target: w[0,0] 8x0e,4x0e   shape=(8, 4)
  params/energy_model/readout/readout_1/linear_2/
    source: w[0,0] 16x0e,1x0e  shape=(16, 1)
    target: w[0,0] 16x0e,4x0e  shape=(16, 4)

Shape mismatches (same path, different shape):
  params/some/path/kernel
    source: (128, 1)   target: (128, 8)

Add the following to your config to mark these slots as freshly initialized:

  transfer_learning:
    reset_layers:
      - params/energy_model/readout/readout_0/linear/w[0,0] 8x0e,4x0e
      - params/energy_model/readout/readout_1/linear_2/w[0,0] 16x0e,4x0e
      - params/some/path/kernel
```

The `reset_layers:` snippet always emits **target** paths for structural mismatches (so the slots stay random-init) and the original path for shape mismatches. Pasting these silences the error on the next run.

If only one category fires, omit the other category's heading. If neither fires, no raise.

### Step 1: Extend the failing tests

Append to `tests/unit_tests/transfer_learning/test_shape_mismatch_transfer.py`:

```python
def test_structural_mismatch_raises_with_target_paths_in_snippet():
    """Source and target have orphan siblings under the same parent → raise."""
    src = _params({"params/readout_0/linear/w 8x0e,1x0e": (8, 1)})
    tgt = _params({"params/readout_0/linear/w 8x0e,4x0e": (8, 4)})

    with pytest.raises(TransferLearningShapeMismatchError) as excinfo:
        black_list_param_transfer(src, tgt, [])

    msg = str(excinfo.value)
    assert "params/readout_0/linear" in msg
    assert "w 8x0e,1x0e" in msg
    assert "w 8x0e,4x0e" in msg
    # Suggested reset_layers entry must be the TARGET path (so the target
    # slot stays random-init when pasted).
    assert re.search(
        r"-\s*params/readout_0/linear/w\s+8x0e,4x0e", msg
    ), f"missing target path in yaml-ready bullet:\n{msg}"


def test_target_path_in_reset_layers_suppresses_structural_mismatch():
    """Pasting the target path into reset_layers silences the structural error."""
    src = _params({"params/readout_0/linear/w 8x0e,1x0e": (8, 1)})
    tgt = _params({"params/readout_0/linear/w 8x0e,4x0e": (8, 4)})
    tgt["params"]["readout_0"]["linear"]["w 8x0e,4x0e"][:] = 99.0

    out = black_list_param_transfer(
        src, tgt, ["params/readout_0/linear/w 8x0e,4x0e"]
    )

    # Target stays random-init (still 99.0); source orphan is dropped silently.
    leaf = np.asarray(out["params"]["readout_0"]["linear"]["w 8x0e,4x0e"])
    assert np.all(leaf == 99.0)


def test_pure_source_only_orphan_does_not_trigger_structural_mismatch():
    """A source leaf with no target counterpart anywhere stays a silent skip.

    Preserves the legitimate refactor / deprecated-param case: the existing
    behavior of silently skipping source-only keys must not regress into a
    spurious structural error.
    """
    src = _params({
        "params/dense/kernel": (4, 8),
        "params/deprecated/old_param": (3,),
    })
    tgt = _params({"params/dense/kernel": (4, 8)})
    src["params"]["dense"]["kernel"][:] = 1.0

    out = black_list_param_transfer(src, tgt, [])

    assert np.all(np.asarray(out["params"]["dense"]["kernel"]) == 1.0)


def test_pure_target_only_orphan_does_not_trigger_structural_mismatch():
    """A target leaf with no source counterpart stays at random init silently.

    The "new slot" case: target adds parameters that didn't exist in the
    source (e.g., a fresh property head). No error; transfer proceeds and
    the new slot stays as initialized.
    """
    src = _params({"params/dense/kernel": (4, 8)})
    tgt = _params({
        "params/dense/kernel": (4, 8),
        "params/new_head/kernel": (8, 1),
    })
    src["params"]["dense"]["kernel"][:] = 1.0
    tgt["params"]["new_head"]["kernel"][:] = 99.0

    out = black_list_param_transfer(src, tgt, [])

    assert np.all(np.asarray(out["params"]["dense"]["kernel"]) == 1.0)
    assert np.all(np.asarray(out["params"]["new_head"]["kernel"]) == 99.0)


def test_combined_structural_and_shape_mismatch_in_one_error():
    """Both mismatch categories fire → one raise listing both."""
    src = _params({
        "params/readout/w 8x0e,1x0e": (8, 1),
        "params/dense/kernel": (4, 1),
    })
    tgt = _params({
        "params/readout/w 8x0e,4x0e": (8, 4),
        "params/dense/kernel": (4, 4),
    })

    with pytest.raises(TransferLearningShapeMismatchError) as excinfo:
        black_list_param_transfer(src, tgt, [])

    msg = str(excinfo.value)
    assert "Structural" in msg or "structural" in msg
    assert "Shape" in msg or "shape" in msg
    assert "params/readout" in msg
    assert "params/dense/kernel" in msg
```

### Step 2: Run tests to verify they fail

Run: `uv run pytest tests/unit_tests/transfer_learning/test_shape_mismatch_transfer.py -v`
Expected: 5 new tests fail (structural-mismatch detection not yet implemented). The 6 existing tests still pass.

### Step 3: Extend `black_list_param_transfer` with the structural-mismatch pass

Replace the body of `apax/transfer_learning/parameter_transfer.py` with the version below. The legacy `_format_mismatch_error` helper is generalized into `_format_combined_error(structural, shape)`; the public signature of `black_list_param_transfer` is unchanged.

```python
import logging
from collections import defaultdict
from typing import Union

from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict

from apax.config.train_config import TransferLearningConfig
from apax.train.checkpoints import load_params

log = logging.getLogger(__name__)


class TransferLearningShapeMismatchError(ValueError):
    """Raised when source and target params disagree on a leaf's shape OR
    when they have orphan leaves under the same parent path.

    The message names every offending leaf with both shapes and emits a
    ready-to-paste ``reset_layers:`` YAML snippet of target paths.
    """


def _is_blacklisted(path: tuple, param_black_list: list) -> bool:
    """Match a leaf path against ``reset_layers`` entries.

    A leaf is blacklisted if either the legacy ``p[-2]`` suffix matches an
    entry, or the full ``/``-joined path matches an entry.
    """
    full_path = "/".join(map(str, path))
    return (
        (len(path) >= 2 and path[-2] in param_black_list)
        or full_path in param_black_list
    )


def _shape_of(leaf) -> tuple:
    """Return ``leaf.shape`` as a tuple, or ``()`` for scalar values."""
    return tuple(getattr(leaf, "shape", ()))


def _format_combined_error(
    structural: list,
    shape: list,
) -> str:
    """Build the actionable error message body.

    Parameters
    ----------
    structural
        List of ``(parent_tuple, src_orphan_paths, tgt_orphan_paths,
        src_shapes, tgt_shapes)`` entries — one per parent path with
        sibling orphans on both sides.
    shape
        List of ``(leaf_path_tuple, source_shape, target_shape)`` triples
        for same-path different-shape mismatches.
    """
    n_total = sum(max(len(s[1]), len(s[2])) for s in structural) + len(shape)
    lines = [f"Transfer learning mismatch on {n_total} parameter slot(s):", ""]

    if structural:
        lines.append("Structural mismatches (source/target diverge under same parent):")
        for parent, src_only, tgt_only, src_shapes, tgt_shapes in structural:
            parent_str = "/".join(map(str, parent)) + "/"
            lines.append(f"  {parent_str}")
            for p in sorted(src_only):
                lines.append(f"    source: {p[-1]}   shape={src_shapes[p]}")
            for p in sorted(tgt_only):
                lines.append(f"    target: {p[-1]}   shape={tgt_shapes[p]}")
        lines.append("")

    if shape:
        lines.append("Shape mismatches (same path, different shape):")
        for path, src_shape, tgt_shape in shape:
            joined = "/".join(map(str, path))
            lines.append(f"  {joined}")
            lines.append(f"    source: {tuple(src_shape)}   target: {tuple(tgt_shape)}")
        lines.append("")

    lines.append(
        "Add the following to your config to mark these slots as freshly "
        "initialized (target paths for structural mismatches; original path "
        "for shape mismatches):"
    )
    lines.append("")
    lines.append("  transfer_learning:")
    lines.append("    reset_layers:")
    for parent, _src_only, tgt_only, _ss, _ts in structural:
        for p in sorted(tgt_only):
            lines.append(f"      - {'/'.join(map(str, p))}")
    for path, _src, _tgt in shape:
        lines.append(f"      - {'/'.join(map(str, path))}")
    return "\n".join(lines)


def black_list_param_transfer(
    source_params: Union[FrozenDict, dict],
    target_params: Union[FrozenDict, dict],
    param_black_list: list,
) -> FrozenDict:
    """Transfer parameters from one pytree to another with shape + structure checking.

    A leaf at path ``p`` is written iff:
    - ``shape(source[p]) == shape(target[p])`` AND
    - the leaf is not blacklisted.

    Two failure categories trigger a single combined raise:

    - **Structural mismatch:** a parent path has source-only leaves AND
      target-only leaves under it (the e3nn ``w 8x0e,1x0e`` vs
      ``w 8x0e,4x0e`` case). Suggested fix: paste the **target** paths into
      ``reset_layers`` so the target slots stay at fresh init.
    - **Shape mismatch:** a leaf path is present in both pytrees but with
      different shapes. Suggested fix: paste the path into ``reset_layers``.

    A leaf is blacklisted if either:

    - ``p[-2]`` matches an entry in ``param_black_list`` (legacy suffix form), or
    - the ``/``-joined full path matches an entry (full-path form).

    Blacklisted leaves are excluded from transfer AND from both mismatch
    passes. For structural mismatches, target-side blacklisting also
    suppresses the parent-level error (so users can paste target paths from
    the error message to silence it on the next run).

    Pure source-only orphans (no leaves under the same parent on the target
    side) and pure target-only orphans are silently allowed. Only the
    sibling-orphan pattern raises.

    Parameters
    ----------
    source_params
        Source pytree (e.g. converted foundation checkpoint).
    target_params
        Freshly-initialized target pytree.
    param_black_list
        Layer names or full leaf paths to skip during transfer / mismatch
        checks. Accepts both source-side and target-side full paths.

    Returns
    -------
    FrozenDict
        ``target_params`` with matching-shape source leaves written in.

    Raises
    ------
    TransferLearningShapeMismatchError
        When any non-blacklisted structural or shape mismatch is detected.
    """
    source_params = unfreeze(source_params)
    target_params = unfreeze(target_params)

    flat_source = flatten_dict(source_params)
    flat_target = flatten_dict(target_params)

    # 1) Group by parent path for structural-mismatch detection
    src_by_parent: dict[tuple, set[tuple]] = defaultdict(set)
    tgt_by_parent: dict[tuple, set[tuple]] = defaultdict(set)
    for p in flat_source:
        src_by_parent[p[:-1]].add(p)
    for p in flat_target:
        tgt_by_parent[p[:-1]].add(p)

    structural: list = []
    for parent in src_by_parent.keys() | tgt_by_parent.keys():
        src_set = src_by_parent.get(parent, set())
        tgt_set = tgt_by_parent.get(parent, set())
        common = src_set & tgt_set
        src_only = {p for p in src_set - common if not _is_blacklisted(p, param_black_list)}
        tgt_only = {p for p in tgt_set - common if not _is_blacklisted(p, param_black_list)}
        if src_only and tgt_only:
            src_shapes = {p: _shape_of(flat_source[p]) for p in src_only}
            tgt_shapes = {p: _shape_of(flat_target[p]) for p in tgt_only}
            structural.append((parent, src_only, tgt_only, src_shapes, tgt_shapes))

    # 2) Same-path shape-mismatch detection + writes
    shape_mismatches: list = []
    for p, v in flat_source.items():
        if _is_blacklisted(p, param_black_list):
            log.info("Skipping (reset_layers): %s", "/".join(map(str, p)))
            continue
        if p not in flat_target:
            log.info("Skipping (no target slot): %s", "/".join(map(str, p)))
            continue
        src_shape = _shape_of(v)
        tgt_shape = _shape_of(flat_target[p])
        if src_shape != tgt_shape:
            shape_mismatches.append((p, src_shape, tgt_shape))
            continue
        flat_target[p] = v
        log.info("Transferring parameter: %s", "/".join(map(str, p)))

    # 3) Combined raise
    if structural or shape_mismatches:
        raise TransferLearningShapeMismatchError(
            _format_combined_error(structural, shape_mismatches)
        )

    transfered_target = unflatten_dict(flat_target)
    transfered_target = freeze(transfered_target)
    return transfered_target


def transfer_parameters(
    state: TrainState, ckpt_config: TransferLearningConfig
) -> TrainState:
    """Transfer the parameters from the checkpoint to the train state.

    Parameters
    ----------
    state
        Train state with freshly-initialized parameters.
    ckpt_config
        Transfer learning configuration.

    Returns
    -------
    TrainState
        ``state`` with ``params`` updated according to the source checkpoint.
    """
    source_params = load_params(ckpt_config.base_model_checkpoint)
    log.info("Transferring parameters from %s", ckpt_config.base_model_checkpoint)
    params = black_list_param_transfer(
        source_params, state.params, ckpt_config.reset_layers
    )
    state = state.replace(params=params)
    return state
```

### Step 4: Update Task 1's existing tests if the message format shifted

Two of the Task 1 tests inspect the error string:

- `test_mismatched_shapes_raise_with_actionable_message`: expects `"Transfer learning shape mismatch on N parameter(s):"` and a YAML bullet for the mismatched path. The new format starts with `"Transfer learning mismatch on N parameter slot(s):"`. **Update** the assertion to `assert "mismatch" in msg.lower()` and the bullet regex stays the same (the `reset_layers:` block still ends with `- params/...`).
- `test_multiple_mismatches_collected_in_one_error`: only asserts that all three paths appear in the message. **No change needed.**

If any other Task 1 assertion is too tight on the exact preamble wording, relax it the same way.

### Step 5: Run all transfer-learning tests

Run: `uv run pytest tests/unit_tests/transfer_learning/ -v`
Expected: 14 passing (6 original Task 1 tests + 5 new Task 1.5 tests + 3 pre-existing tests in the directory).

### Step 6: Commit

```bash
git add apax/transfer_learning/parameter_transfer.py \
        tests/unit_tests/transfer_learning/test_shape_mismatch_transfer.py
git commit -m "feat(transfer): detect sibling-orphan structural mismatches across e3nn-named keys"
```

---

## Task 2: Add `kind` and `MLP_irreps` fields to `PropertyHead`

**Files:**
- Modify: `apax/config/model_config.py:203-238`

- [ ] **Step 1: Write a failing test for the new schema fields**

Append to `tests/unit_tests/config/test_mace_model_config.py` (or create the file if it doesn't exist; check existence first with `ls tests/unit_tests/config/`). Add this test (do not place it inside any existing class):

```python
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
```

If you needed to create the test file, prepend `import pytest` at the top.

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/unit_tests/config/test_mace_model_config.py -v -k property_head`
Expected: FAIL — `kind` field does not exist on `PropertyHead`.

- [ ] **Step 3: Add the new fields**

Edit `apax/config/model_config.py`. Find the `PropertyHead` class (line ~203) and update the docstring + body to:

```python
class PropertyHead(BaseModel, extra="forbid"):
    """
    Configuration for property heads.

    Parameters
    ----------
    name : str
        Name of the property.
    aggregation : str, default = "none"
        Aggregation method for atomic contributions.
    mode : str, default = "l0"
        Rotation order of the property.
    kind : Literal["standard", "mace"], default = "standard"
        Which readout architecture to instantiate. ``"standard"`` selects
        :class:`apax.layers.readout.AtomisticReadout` (the default for
        GMNN/EquivMP/So3krates property heads); ``"mace"`` selects
        :class:`apax.layers.readout.MaceReadout`. MACE users must set this
        to ``"mace"`` explicitly per property head — the builder errors
        otherwise.
    nn : List[PositiveInt], default = [128, 128]
        Number of hidden layers and units in those layers. Used only when
        ``kind="standard"``.
    n_shallow_members : int, default = 0
        Number of shallow ensemble members for this head.
    MLP_irreps : str, default = "16x0e"
        e3nn irreps string for the MaceReadout's intermediate MLP. Used only
        when ``kind="mace"``.
    w_init : Literal["normal", "lecun"], default = "lecun"
        Initialization scheme for the neural network weights.
    b_init : Literal["normal", "zeros"], default = "zeros"
        Initialization scheme for the neural network biases.
    use_ntk : bool, default = False
        Whether or not to use NTK parametrization.
    dtype : Literal["fp32", "fp64"], default = "fp32"
        Data type for property head calculations.
    """

    name: str
    aggregation: str = "none"
    mode: str = "l0"

    kind: Literal["standard", "mace"] = "standard"

    nn: List[PositiveInt] = [128, 128]
    n_shallow_members: int = 0
    MLP_irreps: str = "16x0e"
    w_init: Literal["normal", "lecun"] = "lecun"
    b_init: Literal["normal", "zeros"] = "zeros"
    use_ntk: bool = False
    dtype: Literal["fp32", "fp64"] = "fp32"
```

- [ ] **Step 4: Run the schema test to verify it passes**

Run: `uv run pytest tests/unit_tests/config/test_mace_model_config.py -v -k property_head`
Expected: 3 passing.

- [ ] **Step 5: Run the existing config + builder tests to confirm no regressions**

Run: `uv run pytest tests/unit_tests/config/ tests/unit_tests/nn/ -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add apax/config/model_config.py tests/unit_tests/config/test_mace_model_config.py
git commit -m "feat(config): add PropertyHead.kind discriminator and MLP_irreps for MACE property heads"
```

---

## Task 3: Wire property-head dispatch through `build_readout` and add cross-model guards

**Files:**
- Modify: `apax/nn/builder.py:97-133` (parent `ModelBuilder.build_readout`)
- Modify: `apax/nn/builder.py:381-410` (`MaceBuilder.build_readout`)
- Test: `tests/unit_tests/nn/test_property_heads.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit_tests/nn/test_property_heads.py`:

```python
"""Property-head readout dispatch — kind discriminator + cross-model guards.

Covers the (b)+(c) follow-up to PR #558. The contract:

- ``MaceBuilder.build_readout(head_config)`` where ``head_config`` is a
  property-head config (i.e. not ``self.config``) dispatches on
  ``head_config["kind"]``: ``"mace"`` returns a ``MaceReadout``,
  ``"standard"`` raises with an actionable error.
- ``ModelBuilder.build_readout`` (parent — used by GMNN/EquivMP/So3krates)
  raises when a property head sets ``kind="mace"`` because their descriptors
  do not produce the per-layer-concatenated feature shape ``MaceReadout``
  requires.
- The energy-readout path (``head_config is self.config``) is unaffected.
"""
import pytest

from apax.config.model_config import GMNNConfig, MaceModelConfig
from apax.nn.builder import GMNNBuilder, MaceBuilder


def _mace_cfg(property_heads):
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
            "use_cueq": False,
        },
        readout={"kind": "mace", "MLP_irreps": "16x0e"},
        property_heads=property_heads,
    )
    return cfg.model_dump()


def _gmnn_cfg(property_heads):
    cfg = GMNNConfig(property_heads=property_heads)
    return cfg.model_dump()


def test_mace_property_head_kind_mace_builds_mace_readout():
    from apax.layers.readout import MaceReadout

    cfg = _mace_cfg([{"name": "charges", "kind": "mace"}])
    builder = MaceBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]
    readout = builder.build_readout(head_cfg)

    assert isinstance(readout, MaceReadout)
    assert readout.num_interactions == 2
    assert readout.hidden_dim == 8
    assert readout.MLP_irreps == "16x0e"


def test_mace_property_head_kind_standard_raises():
    cfg = _mace_cfg([{"name": "charges", "kind": "standard"}])
    builder = MaceBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]

    with pytest.raises(ValueError) as excinfo:
        builder.build_readout(head_cfg)

    msg = str(excinfo.value)
    assert "charges" in msg
    assert "kind='mace'" in msg or 'kind="mace"' in msg


def test_mace_property_head_default_kind_is_standard_so_it_raises_on_mace():
    cfg = _mace_cfg([{"name": "charges"}])  # default kind="standard"
    builder = MaceBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]

    with pytest.raises(ValueError) as excinfo:
        builder.build_readout(head_cfg)

    assert "charges" in str(excinfo.value)


def test_gmnn_property_head_default_kind_builds_atomistic_readout():
    from apax.layers.readout import AtomisticReadout

    cfg = _gmnn_cfg([{"name": "charges", "nn": [64, 64]}])
    builder = GMNNBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]
    readout = builder.build_readout(head_cfg)

    assert isinstance(readout, AtomisticReadout)
    assert tuple(readout.units) == (64, 64)


def test_gmnn_property_head_kind_mace_raises():
    cfg = _gmnn_cfg([{"name": "charges", "kind": "mace"}])
    builder = GMNNBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]

    with pytest.raises(ValueError) as excinfo:
        builder.build_readout(head_cfg)

    msg = str(excinfo.value)
    assert "charges" in msg
    assert "MaceReadout" in msg


def test_mace_property_head_mace_readout_n_shallow_members_propagates():
    from apax.layers.readout import MaceReadout

    cfg = _mace_cfg([
        {"name": "charges", "kind": "mace", "n_shallow_members": 4},
    ])
    builder = MaceBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]
    readout = builder.build_readout(head_cfg)

    assert isinstance(readout, MaceReadout)
    assert readout.n_shallow_ensemble == 4


def test_mace_energy_head_unchanged_by_property_head_guards():
    """Identity check ``head_config is self.config`` keeps the energy path intact."""
    from apax.layers.readout import MaceReadout

    cfg = _mace_cfg([])  # no property heads
    builder = MaceBuilder(cfg, n_species=5)
    readout = builder.build_readout(builder.config)

    assert isinstance(readout, MaceReadout)
    assert readout.num_interactions == 2
```

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/unit_tests/nn/test_property_heads.py -v`
Expected: FAIL — the cross-model guards don't exist; the MACE property-head dispatch falls through to the parent `AtomisticReadout` builder.

- [ ] **Step 3: Add the cross-model guard in `ModelBuilder.build_readout`**

In `apax/nn/builder.py`, replace the body of `ModelBuilder.build_readout` (line 97-133) with the version that adds a guard at the top. Locate this block:

```python
    def build_readout(
        self, head_config, is_feature_fn=False, only_use_n_layers: None | int = None
    ):
        has_ensemble = "ensemble" in head_config.keys() and head_config["ensemble"]
```

and replace it with:

```python
    def build_readout(
        self, head_config, is_feature_fn=False, only_use_n_layers: None | int = None
    ):
        is_energy_head = head_config is self.config
        if (
            not is_energy_head
            and isinstance(head_config, dict)
            and head_config.get("kind") == "mace"
        ):
            raise ValueError(
                f"property_head '{head_config.get('name')}' uses kind='mace' "
                f"but the model is {self.config['name']}; MaceReadout requires "
                "the per-layer-concatenated feature shape that only "
                "MaceRepresentation produces."
            )

        has_ensemble = "ensemble" in head_config.keys() and head_config["ensemble"]
```

The rest of the function body (n_shallow_ensemble/dtype/AtomisticReadout construction) is unchanged.

- [ ] **Step 4: Replace `MaceBuilder.build_readout` with the discriminating version**

Locate `MaceBuilder.build_readout` (line ~381-410) and replace its whole body with:

```python
    def build_readout(
        self,
        head_config,
        is_feature_fn: bool = False,
        only_use_n_layers: int | None = None,
    ):
        is_energy_head = head_config is self.config

        if is_energy_head:
            readout_cfg = self.config["readout"]
            if readout_cfg["kind"] != "mace" or is_feature_fn:
                return super().build_readout(
                    head_config, is_feature_fn, only_use_n_layers
                )

            import e3nn_jax as e3nn

            from apax.layers.readout import MaceReadout

            n_shallow_ensemble = 0
            ens = (
                head_config.get("ensemble")
                if isinstance(head_config, dict)
                else None
            )
            if ens and ens.get("kind") == "shallow":
                n_shallow_ensemble = ens["n_members"]

            desc_cfg = self.config["descriptor"]
            hidden_dim = e3nn.Irreps(desc_cfg["hidden_irreps"]).filter("0e").dim
            return MaceReadout(
                num_interactions=len(desc_cfg["interactions"]),
                hidden_dim=hidden_dim,
                MLP_irreps=readout_cfg["MLP_irreps"],
                n_shallow_ensemble=n_shallow_ensemble,
                dtype=self.config["readout_dtype"],
            )

        # Property-head path: dispatch on the per-head ``kind``.
        kind = head_config.get("kind", "standard")
        if kind == "standard":
            raise ValueError(
                f"property_head '{head_config.get('name')}' uses kind='standard' "
                "but the model is MACE; AtomisticReadout doesn't respect MACE's "
                "per-layer body-order structure. Set kind='mace' explicitly, "
                "or change the model."
            )

        # kind == "mace"
        import e3nn_jax as e3nn

        from apax.layers.readout import MaceReadout

        desc_cfg = self.config["descriptor"]
        hidden_dim = e3nn.Irreps(desc_cfg["hidden_irreps"]).filter("0e").dim
        return MaceReadout(
            num_interactions=len(desc_cfg["interactions"]),
            hidden_dim=hidden_dim,
            MLP_irreps=head_config["MLP_irreps"],
            n_shallow_ensemble=head_config["n_shallow_members"],
            dtype=head_config["dtype"],
        )
```

- [ ] **Step 5: Run the property-head tests to verify they pass**

Run: `uv run pytest tests/unit_tests/nn/test_property_heads.py -v`
Expected: 7 passing.

- [ ] **Step 6: Run the broader builder + MACE suites to confirm no regressions**

Run: `uv run pytest tests/unit_tests/nn/ -v`
Expected: all green, including the existing `test_mace_builder.py` which exercises the energy-head identity-check path.

- [ ] **Step 7: Commit**

```bash
git add apax/nn/builder.py tests/unit_tests/nn/test_property_heads.py
git commit -m "feat(nn): route MACE property heads through MaceReadout via kind discriminator"
```

---

## Task 4: Extend the MACE fine-tune integration test for the foundation→ensemble workflow

**Files:**
- Modify: `tests/integration_tests/mace/test_mace_finetune.py`

- [ ] **Step 1: Inspect the existing integration test to understand the fixture**

Read `tests/integration_tests/mace/test_mace_finetune.py:60-167`. The existing test `test_finetune_converted_small_runs_end_to_end` already converts the foundation, builds a config dict, and calls `run(cfg_path, log_level="warning")`. The two new cases extend that pattern: same converter, same dataset, but `n_members=4` (so `(M, 1)` → `(M, 4)` mismatch surfaces) and a different fine-tune config.

- [ ] **Step 2: Add the two new tests**

Append the two new tests to `tests/integration_tests/mace/test_mace_finetune.py` (after `test_finetune_converted_small_runs_end_to_end`). They share most of the fixture setup, so factor a small helper that builds the fine-tune config dict.

Append:

```python
def _mace_finetune_cfg(
    tmp_path,
    converted,
    ds_path,
    *,
    experiment: str,
    n_members: int,
    reset_layers: list,
):
    """Build the inline fine-tune config dict shared by the ensemble tests."""
    return {
        "n_epochs": 1,
        "seed": 1,
        "data": {
            "directory": str(tmp_path),
            "experiment": experiment,
            "data_path": str(ds_path),
            "n_train": 8,
            "n_valid": 4,
            "batch_size": 2,
            "valid_batch_size": 2,
        },
        "model": {
            "name": "mace",
            "basis": {
                "name": "bessel",
                "variant": "standard",
                "n_basis": 10,
                "r_max": 6.0,
            },
            "radial_embedding": {
                "num_polynomial_cutoff": 5,
                "distance_transform": None,
            },
            "descriptor": {
                "max_ell": 3,
                "hidden_irreps": "128x0e",
                "correlation": 3,
                "interactions": [
                    {"name": "RealAgnosticResidual"},
                    {"name": "RealAgnosticResidual"},
                ],
                "avg_num_neighbors": 1.0,
                "use_cueq": False,
            },
            "readout": {"kind": "mace", "MLP_irreps": "16x0e"},
            "ensemble": {
                "kind": "shallow",
                "n_members": n_members,
                "force_variance": True,
            },
            "descriptor_dtype": "fp64",
            "readout_dtype": "fp64",
            "scale_shift_dtype": "fp64",
        },
        "transfer_learning": {
            "base_model_checkpoint": str(converted),
            "reset_layers": reset_layers,
        },
        "loss": [
            {"name": "energy", "loss_type": "crps"},
            {"name": "forces", "loss_type": "crps"},
        ],
        "optimizer": {"nn_lr": 1e-4, "emb_lr": 1e-4},
    }


def test_finetune_foundation_to_ensemble_errors_when_reset_layers_empty(tmp_path):
    """Foundation→shallow-ensemble fine-tune raises a guided shape-mismatch error.

    With ``n_members=4`` the freshly-initialized target's MaceReadout final
    NonLinearReadoutBlock emits 4 scalars, but the converted foundation has
    a (M, 1) slot. Without ``reset_layers`` entries for those slots,
    ``black_list_param_transfer`` must raise ``TransferLearningShapeMismatchError``
    naming at least the readout's final-layer kernel.
    """
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    pytest.importorskip("yaml")
    import yaml

    from apax.train.run import run
    from apax.transfer_learning import TransferLearningShapeMismatchError
    from apax.transfer_learning.mace_foundation import run_conversion

    converted = tmp_path / "converted" / "mace-mp-0-small.apax"
    run_conversion("small", converted, head="default", family="mace_mp")

    ds_path = _tiny_dataset(tmp_path / "ds.extxyz")

    cfg = _mace_finetune_cfg(
        tmp_path,
        converted,
        ds_path,
        experiment="ft_ensemble_no_reset",
        n_members=4,
        reset_layers=[],
    )
    cfg_path = tmp_path / "cfg_no_reset.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    with pytest.raises(TransferLearningShapeMismatchError) as excinfo:
        run(cfg_path, log_level="warning")

    msg = str(excinfo.value)
    # Final readout layer is the only one that widens to (M, n_members).
    assert "readout" in msg
    assert "(4," in msg or ", 4)" in msg, (
        f"expected n_members=4 in target shape; got:\n{msg}"
    )
    assert "reset_layers:" in msg


def test_finetune_foundation_to_ensemble_succeeds_with_suggested_reset_layers(
    tmp_path,
):
    """Pasting the suggested keys into ``reset_layers`` lets training proceed."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    pytest.importorskip("yaml")
    import re

    import yaml

    from apax.train.checkpoints import restore_parameters
    from apax.train.run import run
    from apax.transfer_learning import TransferLearningShapeMismatchError
    from apax.transfer_learning.mace_foundation import run_conversion

    converted = tmp_path / "converted" / "mace-mp-0-small.apax"
    run_conversion("small", converted, head="default", family="mace_mp")

    ds_path = _tiny_dataset(tmp_path / "ds.extxyz")

    # First run: collect the suggested reset_layers from the error message.
    cfg = _mace_finetune_cfg(
        tmp_path,
        converted,
        ds_path,
        experiment="ft_ensemble_collect",
        n_members=4,
        reset_layers=[],
    )
    cfg_path = tmp_path / "cfg_collect.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    with pytest.raises(TransferLearningShapeMismatchError) as excinfo:
        run(cfg_path, log_level="warning")

    suggested = re.findall(
        r"^\s*-\s+(params/[^\s]+)$", str(excinfo.value), flags=re.MULTILINE
    )
    assert suggested, (
        f"error message did not include yaml-ready bullet list; got:\n"
        f"{excinfo.value}"
    )

    # Second run: paste the suggested keys and train one epoch.
    cfg2 = _mace_finetune_cfg(
        tmp_path,
        converted,
        ds_path,
        experiment="ft_ensemble_with_reset",
        n_members=4,
        reset_layers=suggested,
    )
    cfg2_path = tmp_path / "cfg_with_reset.yaml"
    cfg2_path.write_text(yaml.safe_dump(cfg2))

    run(cfg2_path, log_level="warning")

    ft_dir = tmp_path / "ft_ensemble_with_reset"
    assert (ft_dir / "config.yaml").is_file()
    assert (ft_dir / "best").is_dir()

    restored_cfg, restored_params = restore_parameters(ft_dir)
    assert restored_cfg.model.name == "mace"
    assert restored_cfg.model.ensemble is not None
    assert restored_cfg.model.ensemble.n_members == 4

    import jax
    import numpy as np

    leaves = jax.tree_util.tree_leaves(restored_params)
    # At least one leaf has the new (M, 4) ensemble width.
    has_ensemble_axis = any(
        getattr(leaf, "shape", ()) and leaf.shape[-1] == 4 for leaf in leaves
    )
    assert has_ensemble_axis, "no leaf reflects n_members=4 ensemble axis"
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves), (
        "Fine-tuned params contain NaN or inf"
    )
```

- [ ] **Step 3: Run the new integration tests**

Run: `uv run pytest tests/integration_tests/mace/test_mace_finetune.py -v -m mace_parity`
Expected: 3 passing (the existing test plus the two new ones).

If the runner skips by default (no `mace_parity` marker enabled in your environment), use:

`uv run pytest tests/integration_tests/mace/test_mace_finetune.py -v -m "mace_parity or slow" -o addopts=`

Note: these tests require `torch` and `mace` extras. If they're not installed in the current env, run `uv sync --extra mace` first.

- [ ] **Step 4: Commit**

```bash
git add tests/integration_tests/mace/test_mace_finetune.py
git commit -m "test(mace): cover foundation→shallow-ensemble fine-tune shape-mismatch workflow"
```

---

## Task 5: Update the MACE fine-tune template comment

**Files:**
- Modify: `apax/cli/templates/mace_finetune_minimal.yaml:1-21`

- [ ] **Step 1: Replace the preamble comment**

In `apax/cli/templates/mace_finetune_minimal.yaml`, replace lines 1-21 (the entire comment block before `n_epochs: 50`) with:

```yaml
# Minimal fine-tune config for a converted MACE foundation model.
#
# Prerequisites:
#   uv sync --extra mace
#   uv run apax convert-mace small ./mace-mp-0-small.apax
#
# Then:
#   uv run apax train mace_finetune_minimal.yaml
#
# Foundation → shallow-ensemble fine-tune workflow:
#
#   1. Run `apax train mace_finetune_minimal.yaml` once.
#   2. The first run fails with a "shape mismatch" error listing every
#      readout slot that needs to be re-initialized for the new ensemble
#      width. Copy the suggested `reset_layers:` block into this config.
#   3. Re-run; the listed slots stay random-initialized (the classical
#      "drop scalar head, attach N-output head, train" pattern), every
#      other slot transfers from the foundation.
#
# This is the standard apax mechanism for going from a single-output
# foundation to an n_members shallow ensemble. The `model:` and
# `transfer_learning:` blocks below are unchanged across the two runs;
# only `reset_layers` grows.
```

The `model:`, `transfer_learning:`, `loss:`, and `optimizer:` blocks below the comment are unchanged.

- [ ] **Step 2: Verify the YAML still parses**

Run: `uv run python -c "import yaml; yaml.safe_load(open('apax/cli/templates/mace_finetune_minimal.yaml'))"`
Expected: no output (silent success).

- [ ] **Step 3: Commit**

```bash
git add apax/cli/templates/mace_finetune_minimal.yaml
git commit -m "docs(template): describe run-once-paste-reset_layers workflow in mace_finetune_minimal"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run the full unit-test suite**

Run: `uv run pytest tests/unit_tests/ -v`
Expected: all green.

- [ ] **Step 2: Run the gated MACE integration suite (if extras installed)**

Run: `uv run pytest tests/integration_tests/mace/ -v -m "mace_parity or slow" -o addopts=`
Expected: all green, including the three `test_mace_finetune.py` tests.

If the env lacks `torch`/`mace`, document this in the final summary and skip — the unit tests alone exercise the full code path of (a)/(b)/(c).

- [ ] **Step 3: Run lint / pre-commit**

Run: `uvx prek --all-files`
Expected: green (or auto-fixed; if so, stage the fixes and amend the most recent commit IFF auto-fixes are purely cosmetic).

---

## Self-review notes (for the executor)

- **Spec section coverage:**
  - (a) Shape-mismatch transfer learning → Task 1.
  - (b)+(c) MACE property-head support → Tasks 2 + 3.
  - Template wording → Task 5.
  - Tests `test_shape_mismatch_transfer.py` → Task 1.
  - Tests `test_property_heads.py` → Task 3.
  - Integration extension → Task 4.
- **Identity check `head_config is self.config`** is exercised by `test_mace_energy_head_unchanged_by_property_head_guards` (energy path) and the four property-head tests (non-energy path).
- **Backwards compat for `reset_layers`:** `test_legacy_suffix_in_reset_layers_skips_transfer` and the existing `test_param_transfer` pin the legacy `p[-2]` form.
- **Cross-model guard symmetry:** GMNN+`kind=mace` → ValueError (parent guard); MACE+`kind=standard` → ValueError (child guard). Both messages name the head and explain the fix.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-05-mace-transfer-and-property-heads.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
