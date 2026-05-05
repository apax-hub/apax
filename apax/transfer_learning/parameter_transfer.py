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
