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


def _path_to_str(path: tuple) -> str:
    """Render a flax-traverse path tuple as a ``/``-joined string."""
    return "/".join(map(str, path))


def _is_blacklisted(path: tuple, param_black_list: list) -> bool:
    """Match a leaf path against ``reset_layers`` entries.

    A leaf is blacklisted if either the legacy ``p[-2]`` suffix matches an
    entry, or the full ``/``-joined path matches an entry.
    """
    return (len(path) >= 2 and path[-2] in param_black_list) or _path_to_str(
        path
    ) in param_black_list


def _shape_of(leaf) -> tuple:
    """Return ``leaf.shape`` as a tuple, or ``()`` for scalar values."""
    return tuple(getattr(leaf, "shape", ()))


def _format_combined_error(
    structural: list,
    shape: list,
) -> str:
    """Build the actionable error message body."""
    n_total = sum(len(s[2]) for s in structural) + len(shape)
    lines = [f"Transfer learning mismatch on {n_total} parameter slot(s):", ""]

    if structural:
        lines.append("Structural mismatches (source/target diverge under same parent):")
        for parent, src_only, tgt_only, src_shapes, tgt_shapes in structural:
            lines.append(f"  {_path_to_str(parent)}/")
            for p in sorted(src_only):
                lines.append(f"    source: {p[-1]}   shape={src_shapes[p]}")
            for p in sorted(tgt_only):
                lines.append(f"    target: {p[-1]}   shape={tgt_shapes[p]}")
        lines.append("")

    if shape:
        lines.append("Shape mismatches (same path, different shape):")
        for path, src_shape, tgt_shape in shape:
            lines.append(f"  {_path_to_str(path)}")
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
    for _parent, _src_only, tgt_only, _ss, _ts in structural:
        for p in sorted(tgt_only):
            lines.append(f"      - {_path_to_str(p)}")
    for path, _src, _tgt in shape:
        lines.append(f"      - {_path_to_str(path)}")
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
    for parent in sorted(src_by_parent.keys() | tgt_by_parent.keys()):
        src_set = src_by_parent.get(parent, set())
        tgt_set = tgt_by_parent.get(parent, set())
        common = src_set & tgt_set
        src_only = {
            p for p in src_set - common if not _is_blacklisted(p, param_black_list)
        }
        tgt_only = {
            p for p in tgt_set - common if not _is_blacklisted(p, param_black_list)
        }
        if src_only and tgt_only:
            src_shapes = {p: _shape_of(flat_source[p]) for p in src_only}
            tgt_shapes = {p: _shape_of(flat_target[p]) for p in tgt_only}
            structural.append((parent, src_only, tgt_only, src_shapes, tgt_shapes))

    # 2) Same-path shape-mismatch detection + writes
    shape_mismatches: list = []
    for p, v in flat_source.items():
        if _is_blacklisted(p, param_black_list):
            log.info("Skipping (reset_layers): %s", _path_to_str(p))
            continue
        if p not in flat_target:
            log.info("Skipping (no target slot): %s", _path_to_str(p))
            continue
        src_shape = _shape_of(v)
        tgt_shape = _shape_of(flat_target[p])
        if src_shape != tgt_shape:
            shape_mismatches.append((p, src_shape, tgt_shape))
            continue
        flat_target[p] = v
        log.info("Transferring parameter: %s", _path_to_str(p))

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
