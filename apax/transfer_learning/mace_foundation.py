"""MACE foundation model conversion utilities.

``run_conversion`` is called by the ``apax convert-mace`` CLI; everything
else in this module is an internal helper for the torch→apax weight map
and config extraction. Loading a converted model is
``apax.train.checkpoints.restore_parameters`` — the standard apax path.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np
from flax import serialization


def run_conversion(
    source: Union[str, Path],
    dst: Path,
    *,
    head: str = "mp",
    family: str = "mace_mp",
) -> None:
    """Convert a torch-mace foundation model into an apax-native directory.

    Parameters
    ----------
    source : str or Path
        Either a canonical MACE foundation name (e.g. ``"medium-mpa-0"``,
        ``"medium"``, ``"large"``) resolved via ``mace_mp(...)``, or a path to a
        local ``.model`` file.
    dst : Path
        Output directory.
    head : str
        For multi-head foundation models (e.g. MPA), which head to retain.
    family : str
        Which foundation-family resolver to use. Initial scope: ``"mace_mp"``
        (covers MACE-MP-0, 0b, 0b2, 0b3 and MACE-MPA). Others deferred.
    """
    dst = Path(dst)
    torch_model, resolved_path = _load_torch_foundation_model(source, family=family)
    if hasattr(torch_model, "state_dict"):
        state = {k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()}
    else:
        state = torch_model

    cfg = _extract_config_from_torch(torch_model, head=head)
    params_pytree = _map_state_to_pytree(state, cfg, head=head)
    _validate_no_nan(params_pytree)

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "params.msgpack").write_bytes(serialization.to_bytes(params_pytree))
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    meta = {
        "source": str(source),
        "source_resolved_path": str(resolved_path) if resolved_path else None,
        "torch_mace_version": _torch_mace_version(),
        "apax_version": _apax_version(),
        "head_selected": head,
        "family": family,
        "converted_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    if resolved_path and Path(resolved_path).exists():
        meta["source_sha256"] = hashlib.sha256(
            Path(resolved_path).read_bytes()
        ).hexdigest()
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2))


def _load_torch_foundation_model(source, *, family: str):
    """Load a torch MACE model via the upstream foundation loader.

    Goes through ``mace.calculators.foundations_models.mace_mp(return_raw_model=True)``
    so we inherit bundled-local → cache → download logic + ASL license notices.

    Parameters
    ----------
    source : str or Path
        Canonical model name or filesystem path to a ``.model`` file.
    family : str
        Foundation-model family identifier. Only ``"mace_mp"`` is supported.

    Returns
    -------
    torch_model : torch.nn.Module
        Loaded torch module with all parameters on CPU.
    resolved_path : str or None
        Path on disk that the torch model was loaded from (when the resolver
        exposes it). ``None`` if we have only a module-in-memory.

    Raises
    ------
    NotImplementedError
        If ``family`` is not ``"mace_mp"``.
    ValueError
        If ``source`` is not a valid MACE-MP model name.
    """
    if family != "mace_mp":
        raise NotImplementedError(
            f"family={family!r} not yet supported; only 'mace_mp' is in initial scope."
        )

    from mace.calculators.foundations_models import (
        download_mace_mp_checkpoint,
        mace_mp,
        mace_mp_names,
    )

    # Heuristic: treat as a canonical name if it's not an existing file path.
    is_path = isinstance(source, (str, Path)) and Path(source).exists()
    if is_path:
        import torch
        return torch.load(str(source), map_location="cpu"), str(source)

    # Canonical name — validate against the registry for a clearer error
    if source not in mace_mp_names and not str(source).startswith("https:"):
        raise ValueError(
            f"Unknown MACE-MP model name {source!r}. "
            f"Valid names: {', '.join(n for n in mace_mp_names if n)}"
        )

    resolved_path = download_mace_mp_checkpoint(source)
    torch_model = mace_mp(source, return_raw_model=True)
    return torch_model, resolved_path


_INTERACTION_CLS_MAP = {
    "RealAgnosticInteractionBlock": "RealAgnostic",
    "RealAgnosticResidualInteractionBlock": "RealAgnosticResidual",
    "RealAgnosticDensityInteractionBlock": "RealAgnosticDensity",
    "RealAgnosticDensityResidualInteractionBlock": "RealAgnosticDensityResidual",
}


def _extract_config_from_torch(model, head: str) -> dict:
    """Return a dict that matches :class:`MaceModelConfig` schema.

    Reads hyperparameters off a torch ``ScaleShiftMACE`` foundation-model instance.
    For foundation models, many of the hyperparameters are not exposed as
    attributes on the top-level module; they must be discovered by walking the
    submodule tree.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded MACE torch model (typically a ``ScaleShiftMACE``).
    head : str
        Which head to select for multi-head models. Ignored when the model
        only has a single head.

    Returns
    -------
    dict
        Configuration dictionary compatible with ``MaceModelConfig`` plus a few
        extra fields used by the parameter mapper and the full-energy model:

        - ``atomic_energies`` : list[float] of shape ``(num_elements,)``
        - ``atomic_numbers``  : list[int]   of shape ``(num_elements,)``
        - ``scale``, ``shift`` : floats for :class:`ScaleShiftBlock`
        - ``has_zbl`` : whether the source model carries a pair-repulsion tail
        - ``selected_head`` : present only for multi-head foundation models
    """
    # Interaction variant
    iface_cls = type(model.interactions[0]).__name__
    interaction_cls = _INTERACTION_CLS_MAP.get(iface_cls)
    if interaction_cls is None:
        raise ValueError(
            f"Unrecognised torch-mace interaction block {iface_cls!r}; "
            f"known: {sorted(_INTERACTION_CLS_MAP)}"
        )

    # Max spherical harmonic degree: the SH block's output irreps lists all
    # degrees 0..ell_max in order.
    max_ell = int(max(ir.l for _, ir in model.spherical_harmonics.irreps_out))

    # hidden_irreps: the irreps of the final product linear output. For the
    # small MP-0 model this is "128x0e".
    hidden_irreps = str(model.products[0].linear.irreps_out)

    # Correlation: stored on the first contraction.
    correlation = int(model.products[0].symmetric_contractions.contractions[0].correlation)

    # num_elements: number of 0e channels in the node_embedding input (which is
    # one-hot over the model's ``atomic_numbers`` table).
    num_elements = int(model.node_embedding.linear.irreps_in.num_irreps)

    # Bessel basis size & cutoff polynomial order.
    num_bessel = int(model.radial_embedding.bessel_fn.bessel_weights.shape[0])
    num_polynomial_cutoff = int(model.radial_embedding.cutoff_fn.p)

    atomic_energies = (
        model.atomic_energies_fn.atomic_energies.detach().cpu().numpy().tolist()
    )
    atomic_numbers = model.atomic_numbers.detach().cpu().numpy().tolist()

    scale = float(model.scale_shift.scale)
    shift = float(model.scale_shift.shift)

    # ZBL / pair repulsion
    has_zbl = (
        "pair_repulsion_fn" in dict(model.named_children())
        and model.pair_repulsion_fn is not None
    )

    cfg = {
        "name": "mace",
        "r_max": float(model.r_max),
        "num_bessel": num_bessel,
        "num_polynomial_cutoff": num_polynomial_cutoff,
        "max_ell": max_ell,
        "hidden_irreps": hidden_irreps,
        "num_interactions": int(model.num_interactions),
        "correlation": correlation,
        "interaction_cls": interaction_cls,
        "num_elements": num_elements,
        "atomic_energies": atomic_energies,
        "atomic_numbers": atomic_numbers,
        "scale": scale,
        "shift": shift,
        "has_zbl": has_zbl,
    }

    # Multi-head models (MPA): expose ``heads`` as a list of str.
    heads = getattr(model, "heads", None)
    if heads is not None and len(heads) > 1:
        head_names = list(heads)
        if head not in head_names:
            raise ValueError(
                f"head={head!r} not in available heads {head_names}. "
                f"Pass --head <name> from that list."
            )
        cfg["selected_head"] = head
    return cfg


def _map_state_to_pytree(
    state: dict[str, np.ndarray], cfg: dict, head: str
) -> dict:
    """Translate torch parameter names → linen pytree.

    Parameters
    ----------
    state : dict of str to np.ndarray
        Flat ``{name: array}`` mapping produced from ``model.state_dict()``.
    cfg : dict
        Config dict produced by ``_extract_config_from_torch``.
    head : str
        Selected head name for multi-head models.

    Returns
    -------
    dict
        Nested flax linen-compatible parameter pytree ready for
        ``flax.serialization.to_bytes``.

    Notes
    -----
    Parameter path map (canonical list):

    Torch key                                    → apax pytree path
    ------------------------------------------------------------------
    node_embedding.linear.weight                 → params/LinearNodeEmbedding_0/weight
    interactions.<i>.linear_up.weight            → params/MaceRepresentation/.../InteractionBlock_<i>/linear_up/kernel
    interactions.<i>.linear_down.weight          → .../InteractionBlock_<i>/linear_down/kernel
    interactions.<i>.skip_tp.weight              → .../InteractionBlock_<i>/skip_linear/kernel
    interactions.<i>.conv_tp_weights.*           → .../InteractionBlock_<i>/radial_mlp/*
    products.<i>.linear.weight                   → .../ProductBlock_<i>/symmetric_contraction/weight
    readouts.<i>.linear.weight                   → .../Readout_<i>/linear/kernel
    ...

    This map is authoritative — every torch key must land somewhere, and
    the NaN-leaf check (below) will fail if any expected slot is missed.

    To install: ``uv sync --group mace-convert --extra mace``
    then re-run: ``uv run pytest tests/integration_tests/mace/test_convert.py -m mace_parity``

    See plan P3.2 Step 4 in
    ``docs/superpowers/plans/2026-04-20-mace-foundation-model-integration.md``
    for iteration instructions.
    """
    raise NotImplementedError(
        "Filled by a later pass with torch+mace-torch installed; see plan P3.2 Step 4. "
        "Install with: uv sync --group mace-convert --extra mace"
    )


def _extract_norm_consts() -> dict[str, float]:
    """Fetch torch normalize2mom constants for common gates (fail fast).

    Parity relies on reusing the exact normalize2mom constants that torch
    precomputed for its activation wrappers. If they cannot be obtained we raise
    instead of silently recomputing a different value.

    Returns
    -------
    dict of str to float
        Mapping of activation name to normalize2mom constant,
        e.g. ``{"silu": 1.7868..., "swish": 1.7868...}``.

    Raises
    ------
    ImportError
        If torch or e3nn are not available.
    RuntimeError
        If the constant cannot be computed from the loaded torch modules.

    Notes
    -----
    Port of ``_extract_norm_consts`` from
    ``mace-jax/mace_jax/tools/import_from_torch.py``.
    Store the constants at ``params["constants"]["normalize2mom_silu"]`` after
    ``_map_state_to_pytree`` is fully implemented.
    """
    try:
        import torch  # noqa: PLC0415
        from e3nn.math._normalize_activation import (  # noqa: PLC0415
            normalize2mom as torch_norm,
        )
    except Exception as exc:
        raise ImportError(
            "Torch e3nn (and torch) are required to import activation "
            "normalization constants; parity cannot be guaranteed without them."
        ) from exc

    try:
        const = float(torch_norm(torch.nn.functional.silu).cst)
    except Exception as exc:
        raise RuntimeError(
            "Unable to compute normalize2mom constant for torch.nn.functional.silu "
            "during import; parity cannot be guaranteed."
        ) from exc

    return {"silu": const, "swish": const}


def _validate_no_nan(pytree: dict) -> None:
    """Check that no floating-point leaf in a pytree contains NaN.

    Parameters
    ----------
    pytree : dict
        Nested parameter pytree to validate.

    Raises
    ------
    ValueError
        If any floating-point leaf array contains at least one NaN.
    """
    import jax

    bad = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(pytree)[0]:
        if isinstance(leaf, np.ndarray) and np.issubdtype(leaf.dtype, np.floating):
            if np.isnan(leaf).any():
                bad.append("/".join(str(k) for k in path))
    if bad:
        raise ValueError(
            "NaN leaves after conversion:\n  - " + "\n  - ".join(bad)
        )


def _torch_mace_version() -> str:
    """Return the installed mace-torch version string, or ``"unknown"``.

    Returns
    -------
    str
        Version string from ``mace.__version__``, or ``"unknown"`` on failure.
    """
    try:
        import mace
        return mace.__version__
    except Exception:
        return "unknown"


def _apax_version() -> str:
    """Return the installed apax version string, or ``"unknown"``.

    Returns
    -------
    str
        Version string from ``apax.__version__``, or ``"unknown"`` on failure.
    """
    try:
        from apax import __version__
        return __version__
    except Exception:
        return "unknown"


