"""MACE foundation model conversion utilities.

``run_conversion`` is called by the ``apax convert-mace`` CLI. It produces a
directory in apax's standard training-output layout::

    <dst>/
        config.yaml                # full apax Config (validates with Config.model_validate)
        best/                      # orbax checkpoint (load_state-compatible)
        converter_metadata.json    # provenance: source, versions, sha256, head, ...

Loading a converted model is just :func:`apax.train.checkpoints.restore_parameters`
— no special-casing on the consumer side. ``ASECalculator``, ``apax md``, and
the BAL workflow all read this directory verbatim.

The torch→apax weight map and config extraction are internal helpers; they
import torch / mace lazily so the module is safe to import without optional
deps installed.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import flax
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from apax.config.train_config import Config


# Torch interaction class names that apax can convert today, mapped to the
# apax-side ``Literal`` values consumed by ``MaceModelConfig.interaction_cls``.
# ``RealAgnosticInteractionBlock`` (the non-residual non-density variant) is
# still out of scope — adding it is purely additive (one more block class +
# Literal extension). Schema and converter agree on what is implemented.
_SUPPORTED_TORCH_INTERACTION_CLS = {
    "RealAgnosticResidualInteractionBlock": "RealAgnosticResidual",
    "RealAgnosticDensityInteractionBlock": "RealAgnosticDensity",
    "RealAgnosticDensityResidualInteractionBlock": "RealAgnosticDensityResidual",
}

# Number of chemical species in the apax embedding table. Z=0 is reserved for
# padding; physical Z values are scattered into [1, n_species - 1].
_N_SPECIES = 119


def run_conversion(
    source: Union[str, Path],
    dst: Path,
    *,
    head: str | None = None,
    family: str = "mace_mp",
) -> None:
    """Convert a torch-mace foundation model into an apax training-output dir.

    The output layout matches what ``apax train`` produces, so every apax loader
    (:func:`apax.train.checkpoints.restore_parameters`,
    :class:`apax.md.ase_calc.ASECalculator`, ``apax md``, BAL) reads it without
    any special-casing.

    Parameters
    ----------
    source : str or Path
        Either a canonical MACE foundation name (e.g. ``"small"``,
        ``"medium-mpa-0"``, ``"large"``) resolved via ``mace_mp(...)``, or a
        path to a local ``.model`` file.
    dst : Path
        Output directory. Will be created if it does not exist.
    head : str or None, default = None
        For multi-head foundation models (e.g. MPA), which head to retain.
        ``None`` falls back to the first head in ``model.heads`` (or the
        literal ``"default"`` for single-head models).
    family : str, default = "mace_mp"
        Foundation-family resolver. Initial scope: ``"mace_mp"`` (covers
        MACE-MP-0/0b/0b2/0b3 and MACE-MPA). Other families deferred.

    Notes
    -----
    The converter writes::

        <dst>/config.yaml                 # full apax Config dump
        <dst>/best/<orbax checkpoint>/    # state {"model": {"params": ...}, "epoch": 0}
        <dst>/converter_metadata.json     # source/version provenance
    """
    import jax
    import jax.numpy as jnp

    dst = Path(dst)

    # 1. Load torch model and resolve where it came from on disk.
    torch_model, resolved_path = _load_torch_foundation_model(source, family=family)

    # 2. Extract apax-side architecture fields from the torch model.
    mace_cfg_fields = _extract_config_from_torch(torch_model, head=head)
    torch_atomic_numbers = tuple(
        int(z) for z in torch_model.atomic_numbers.detach().cpu().numpy().tolist()
    )

    # 3. Build the full apax Config (training fields use placeholders; users
    #    should never use this YAML to launch training directly, only to
    #    restore parameters).
    full_cfg = _synthesize_full_config(mace_cfg_fields, dst)

    # 4. Build the same model the trainer would build, then init a template
    #    pytree we can scatter torch weights into.
    builder_cls = full_cfg.model.get_builder()
    builder = builder_cls(full_cfg.model.model_dump(), n_species=_N_SPECIES)
    energy_derivative_model = builder.build_energy_derivative_model()

    R_dummy = jnp.zeros((2, 3))
    Z_dummy = jnp.array([1, 1], dtype=jnp.int32)
    neigh_dummy = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    box_dummy = jnp.zeros((3,))
    offsets_dummy = jnp.zeros((neigh_dummy.shape[1], 3))
    # Drop the ``debug`` collection used by the layer-by-layer parity harness
    # (``scripts/mace_layer_parity.py``); it carries vmap tracers during
    # ``init`` that the downstream :func:`np.asarray` mapper can't convert,
    # and is irrelevant to weight conversion.
    params_template = energy_derivative_model.init(
        jax.random.PRNGKey(0),
        R_dummy,
        Z_dummy,
        neigh_dummy,
        box_dummy,
        offsets_dummy,
        mutable=flax.core.DenyList("debug"),
    )

    # 5. Translate torch weights into the template pytree.
    state = {
        k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()
    }
    extra_scalars = {
        "scale": float(torch_model.scale_shift.scale.detach().cpu()),
        "shift": float(torch_model.scale_shift.shift.detach().cpu()),
        "atomic_energies": torch_model.atomic_energies_fn.atomic_energies.detach()
        .cpu()
        .numpy(),
    }
    params = _map_state_to_pytree(
        state,
        params_template,
        torch_atomic_numbers=torch_atomic_numbers,
        extra_scalars=extra_scalars,
        selected_head=head,
        config=full_cfg.model,
        torch_model=torch_model,
    )
    _validate_no_nan(params)

    # 6. Persist as apax-native training output.
    dst.mkdir(parents=True, exist_ok=True)
    full_cfg.dump_config(dst)  # writes <dst>/config.yaml
    _write_orbax_checkpoint(dst / "best", params, epoch=0)

    # 7. Provenance.
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
    (dst / "converter_metadata.json").write_text(json.dumps(meta, indent=2))


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
        Path on disk that the torch model was loaded from when the resolver
        exposes it; ``None`` for module-in-memory loads.

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

    from mace.calculators.foundations_models import (  # noqa: PLC0415
        download_mace_mp_checkpoint,
        mace_mp,
        mace_mp_names,
    )

    # Heuristic: treat as a canonical name if it's not an existing file path.
    is_path = isinstance(source, (str, Path)) and Path(source).exists()
    if is_path:
        import torch  # noqa: PLC0415

        return (
            torch.load(str(source), map_location="cpu", weights_only=False),
            str(source),
        )

    if source not in mace_mp_names and not str(source).startswith("https:"):
        raise ValueError(
            f"Unknown MACE-MP model name {source!r}. "
            f"Valid names: {', '.join(n for n in mace_mp_names if n)}"
        )

    resolved_path = download_mace_mp_checkpoint(source)
    torch_model = mace_mp(
        source, return_raw_model=True, default_dtype="float64", device="cpu"
    )
    return torch_model, resolved_path


def _extract_config_from_torch(model, head: str | None) -> dict:
    """Return a dict that matches :class:`MaceModelConfig` schema.

    Reads hyperparameters off a torch ``ScaleShiftMACE`` foundation-model
    instance. Many of the fields are not exposed as top-level attributes; this
    helper walks the submodule tree to discover them.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded MACE torch model (typically a ``ScaleShiftMACE``).
    head : str or None
        Head selector for multi-head models. ``None`` falls back to the
        first head in ``model.heads`` (or the literal ``"default"`` for
        single-head models). When given, must be a member of
        ``model.heads``.

    Returns
    -------
    dict
        Configuration dictionary compatible with :class:`MaceModelConfig`.

    Raises
    ------
    ValueError
        If ``head`` is not in the torch model's available heads.
    NotImplementedError
        If the interaction block class is not in :data:`_INTERACTION_CLS_MAP`.
    """
    import e3nn  # noqa: PLC0415

    heads = list(getattr(model, "heads", ["default"]))
    if head is None:
        head = heads[0]
    elif head not in heads:
        raise ValueError(
            f"head={head!r} not in available heads {heads}. "
            f"Pass --head <name> from that list."
        )

    per_layer_cls: list[str] = []
    for inter in model.interactions:
        cls_name = type(inter).__name__
        if cls_name not in _SUPPORTED_TORCH_INTERACTION_CLS:
            raise NotImplementedError(
                f"Foundation uses interaction block {cls_name!r}; apax supports "
                f"{sorted(_SUPPORTED_TORCH_INTERACTION_CLS)}. Other variants need "
                "their apax port before they can be converted."
            )
        per_layer_cls.append(_SUPPORTED_TORCH_INTERACTION_CLS[cls_name])
    # Emit a single str when every layer uses the same variant; emit a list
    # otherwise (e.g. mpa-0 / matpes use Density first, DensityResidual last).
    if len(set(per_layer_cls)) == 1:
        interaction_cls: Union[str, list[str]] = per_layer_cls[0]
    else:
        interaction_cls = per_layer_cls

    hidden_irreps = str(model.products[0].linear.irreps_out)

    sph_irreps = e3nn.o3.Irreps(str(model.spherical_harmonics.irreps_out))
    max_ell = max(ir.l for _, ir in sph_irreps)

    sc0 = model.products[0].symmetric_contractions.contractions[0]
    correlation = 1
    while hasattr(sc0, f"U_matrix_{correlation + 1}"):
        correlation += 1

    avg_num_neighbors = float(model.interactions[0].avg_num_neighbors)

    # Detect torch's :class:`mace.modules.radial.ZBLBasis` correction and
    # emit an apax ``MaceZBLPairRepulsion`` config entry for the converter to
    # populate. The correction is detected by the ``pair_repulsion_fn``
    # submodule attribute (present on MACE-MPA-0, MatPES, OMAT foundations).
    empirical_corrections: list[dict] = []
    if hasattr(model, "pair_repulsion_fn"):
        import torch  # noqa: PLC0415
        zbl = model.pair_repulsion_fn
        # ``p`` is a registered buffer (torch tensor); cast to int.
        p_value = int(zbl.p.detach().cpu().item())
        # ``a_exp`` and ``a_prefactor`` may be either buffers (trainable=False)
        # or :class:`nn.Parameter` (trainable=True); detect via instance.
        is_trainable = isinstance(zbl.a_exp, torch.nn.Parameter)
        # Torch-mace folds the ZBL contribution INSIDE ``scale_shift`` (it
        # scales ``readout + ZBL`` by the global ``scale``). apax's
        # ``EnergyModel`` applies corrections AFTER ``scale_shift``, so we
        # replicate the torch behaviour by passing the global ``scale`` in as
        # ``output_scale`` on the apax ZBL module. Buffers (c, a_exp, ...)
        # stay bit-identical to the torch source — no buffer munging.
        # apax's ZBL applies a single output_scale; per-element scales
        # would silently miscompute. Fail loud instead.
        scale_tensor = model.scale_shift.scale.detach().cpu()
        if scale_tensor.numel() > 1 and torch.unique(scale_tensor).numel() > 1:
            raise NotImplementedError(
                "Foundation has per-element scale_shift.scale; apax ZBL "
                "applies a single output_scale only. Supporting per-element "
                "ZBL output_scale would require a change to apax's ZBL "
                "module — please open an issue with the foundation name."
            )
        global_scale = float(scale_tensor.flatten()[0])
        empirical_corrections.append(
            {
                "name": "mace_zbl",
                "p": p_value,
                "trainable": is_trainable,
                "output_scale": global_scale,
            }
        )

    # Detect torch's :class:`mace.modules.radial.AgnesiTransform` and emit a
    # ``DistanceTransformConfig`` entry. The transform is attached to the
    # radial embedding block on foundations like MACE-MPA-0, MatPES,
    # OMAT-r2scan; small / medium do not carry it.
    distance_transform_cfg = None
    if hasattr(model.radial_embedding, "distance_transform"):
        import torch  # noqa: PLC0415
        dt = model.radial_embedding.distance_transform
        cls_name = type(dt).__name__
        if cls_name == "AgnesiTransform":
            is_trainable = isinstance(dt.a, torch.nn.Parameter)
            distance_transform_cfg = {
                "name": "agnesi",
                "a": float(dt.a.detach().cpu()),
                "q": float(dt.q.detach().cpu()),
                "p": float(dt.p.detach().cpu()),
                "trainable": is_trainable,
            }
        else:
            raise NotImplementedError(
                f"Foundation uses distance_transform {cls_name!r}; apax supports "
                f"['AgnesiTransform']. Other variants need their apax port."
            )

    r_max = float(model.r_max)
    num_bessel = int(model.radial_embedding.bessel_fn.bessel_weights.shape[0])

    cfg = {
        "r_max": r_max,
        "num_bessel": num_bessel,
        "num_polynomial_cutoff": int(model.radial_embedding.cutoff_fn.p),
        "max_ell": int(max_ell),
        "hidden_irreps": hidden_irreps,
        "num_interactions": int(model.num_interactions),
        "correlation": int(correlation),
        "interaction_cls": interaction_cls,
        "use_cueq": False,
        "readout_kind": "mace",
        "MLP_irreps": "16x0e",
        "avg_num_neighbors": avg_num_neighbors,
        "empirical_corrections": empirical_corrections,
        "distance_transform": distance_transform_cfg,
        # ``basis`` is consumed by neighbour-list builders across ``md/``,
        # ``bal/``, ``train/`` (they read ``config.model.basis.r_max``, NOT
        # the top-level ``r_max``). Without this entry the BesselBasisConfig
        # default of 5.0 Å is used and the NL truncates pairs in
        # [5.0, model.r_max) Å.
        "basis": {
            "name": "bessel",
            "n_basis": num_bessel,
            "r_max": r_max,
        },
        # Float64 throughout for parity with the torch foundation model loaded
        # with default_dtype="float64".
        "descriptor_dtype": "fp64",
        "readout_dtype": "fp64",
        "scale_shift_dtype": "fp64",
    }
    assert cfg["basis"]["r_max"] == cfg["r_max"], (
        "basis.r_max must mirror top-level r_max"
    )
    assert cfg["basis"]["n_basis"] == cfg["num_bessel"], (
        "basis.n_basis must mirror num_bessel"
    )
    return cfg


def _synthesize_full_config(mace_cfg_fields: dict, dst: Path) -> "Config":
    """Build a valid :class:`apax.config.train_config.Config` around a MaceModelConfig.

    Training-only fields get placeholder values; users should never launch
    training directly from this YAML, only restore parameters from it.

    Parameters
    ----------
    mace_cfg_fields : dict
        Keyword fields for :class:`MaceModelConfig`.
    dst : Path
        Destination directory for the converted model. Used to derive the
        ``data.directory`` and ``data.experiment`` placeholder fields.

    Returns
    -------
    apax.config.train_config.Config
        A validated Config object whose ``dump_config(dst)`` writes
        ``<dst>/config.yaml``.
    """
    from apax.config.model_config import MaceModelConfig  # noqa: PLC0415
    from apax.config.train_config import Config  # noqa: PLC0415

    mace_cfg = MaceModelConfig(**mace_cfg_fields)
    cfg_dict = {
        "n_epochs": 1,
        "data": {
            "directory": str(dst.parent.resolve()),
            "experiment": dst.name,
            "data_path": "placeholder.extxyz",
        },
        "model": mace_cfg.model_dump(),
        "loss": [{"name": "energy"}],
        "optimizer": {},
    }
    return Config.model_validate(cfg_dict)


def _write_orbax_checkpoint(path: Path, params, *, epoch: int) -> None:
    """Write ``{"model": {"params": params}, "epoch": epoch}`` via orbax.

    Matches the schema that :func:`apax.train.checkpoints.load_state` reads.

    Parameters
    ----------
    path : Path
        Directory in which to create the orbax checkpoint manager state.
    params : pytree
        The flax linen parameter pytree to persist.
    epoch : int
        Epoch number to record alongside the parameters.
    """
    import orbax.checkpoint as ocp  # noqa: PLC0415

    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    with ocp.CheckpointManager(path) as mngr:
        mngr.save(
            step=0,
            args=ocp.args.StandardSave(
                {"model": {"params": params}, "epoch": int(epoch)}
            ),
        )
        mngr.wait_until_finished()


def _map_state_to_pytree(
    state: dict[str, np.ndarray],
    template,
    *,
    torch_atomic_numbers: tuple[int, ...],
    extra_scalars: dict,
    selected_head: str,
    config,
    torch_model,
) -> dict:
    """Translate torch ``state_dict`` arrays → a linen pytree matching ``template``.

    The returned pytree mirrors the output of
    :meth:`MaceBuilder.build_energy_derivative_model().init(...)`, so its top
    level is ``{"params": {"energy_model": {...}}}``.

    After the P3.5b architecture extension every torch float parameter has a
    same-numel apax slot, so this mapper performs **direct copies / scatters**
    only — no projections, slices, or per-element averages. The
    symmetric-contraction weights are the only non-trivial reshape; that work
    is delegated to mace-jax's reference adapter so we share its full-CG
    transform.

    Parameters
    ----------
    state : dict of str to np.ndarray
        Flat ``{name: array}`` from ``torch_model.state_dict()``.
    template : pytree
        The freshly-initialised apax pytree whose leaf shapes/dtypes drive the
        scatter. Only structure is consumed; values are overwritten.
    torch_atomic_numbers : tuple of int
        The torch model's ``atomic_numbers`` table, in torch indexing order.
        Used to map per-element rows back to physical Z values.
    extra_scalars : dict
        Extra scalars not in ``state_dict`` but needed for the scale-shift fold:
        ``"scale"`` (global float), ``"shift"`` (global float),
        ``"atomic_energies"`` (1D array indexed by torch element index).
    selected_head : str
        Selected head name. Currently no per-head weights for single-head
        small/medium foundation models.
    config : apax.config.model_config.MaceModelConfig
        The validated model configuration.
    torch_model : torch.nn.Module
        Live torch foundation model. Required by :func:`_map_products` to call
        mace-jax's :func:`_convert_native_weights`, which builds the full-CG
        transform from torch's :class:`SymmetricContraction` directly.

    Returns
    -------
    dict
        New pytree with the same structure as ``template`` populated from
        ``state``. Any leaf left unfilled retains its initial value.
    """
    import jax  # noqa: PLC0415

    out = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), template)
    energy_params = out["params"]["energy_model"]
    rep = energy_params["representation"]

    _map_node_embedding(state, rep, torch_atomic_numbers)
    _map_interactions(state, rep, torch_atomic_numbers, config)
    _map_products(
        state,
        rep,
        torch_atomic_numbers,
        config,
        torch_model=torch_model,
    )
    _map_readouts(
        state,
        energy_params["readout"],
        num_interactions=int(config.num_interactions),
    )
    _map_scale_shift(
        energy_params["scale_shift"],
        global_scale=float(extra_scalars["scale"]),
        global_shift=float(extra_scalars["shift"]),
        atomic_energies=np.asarray(extra_scalars["atomic_energies"]),
        torch_atomic_numbers=torch_atomic_numbers,
    )
    if hasattr(torch_model, "pair_repulsion_fn"):
        # Locate the mace_zbl entry in the configured corrections list. The
        # converter only ever appends a single MaceZBLPairRepulsion today, but
        # honour the position so future configs with multiple corrections still
        # map correctly.
        correction_index = next(
            i for i, c in enumerate(config.empirical_corrections)
            if getattr(c, "name", None) == "mace_zbl"
        )
        _map_pair_repulsion(state, out, correction_index=correction_index)
    return out


# ---------------------------------------------------------------------------
# Per-block mappers
# ---------------------------------------------------------------------------


def _map_node_embedding(
    state: dict[str, np.ndarray],
    rep_params: dict,
    torch_atomic_numbers: tuple[int, ...],
) -> None:
    """Map ``node_embedding.linear.weight`` into ``LinearNodeEmbedding_0/weight``.

    Torch stores the weight as a flat tensor of size ``num_torch_elements * hidden``
    in row-major ``(in, out)`` order (same as ``e3nn.flax.Linear``). The apax
    embedding table is ``(_N_SPECIES, hidden)``; rows for chemical species not
    in the torch table stay at their init value (zero-mean Gaussian).

    Torch's ``e3nn.o3.Linear`` applies a ``path_weight = 1/sqrt(num_elements)``
    factor per forward pass, so the values stored in the torch ``state_dict``
    are scaled by ``sqrt(num_elements_torch)`` relative to what apax expects.
    Apax's :class:`~apax.layers.descriptor.mace_blocks.LinearNodeEmbedding`
    is a plain ``one_hot @ weight`` (no path_weight), so we fold the
    ``1/sqrt(n_torch)`` factor into the copied weights here.

    Parameters
    ----------
    state : dict of str to np.ndarray
        Torch state_dict.
    rep_params : dict
        Representation subtree of the apax pytree (mutated in place).
    torch_atomic_numbers : tuple of int
        Maps each torch row to its physical Z value.
    """
    target = rep_params["LinearNodeEmbedding_0"]["weight"]
    n_torch = len(torch_atomic_numbers)
    hidden = target.shape[1]
    flat = state["node_embedding.linear.weight"]
    if flat.size != n_torch * hidden:
        raise ValueError(
            f"node_embedding.linear.weight has {flat.size} elements, expected "
            f"{n_torch * hidden} (= {n_torch} species × {hidden} hidden)."
        )
    # Fold torch's e3nn.o3.Linear path_weight = 1/sqrt(n_torch) into the
    # copied values; apax's embedding is a plain matmul with no path_weight.
    matrix = flat.reshape(n_torch, hidden) / np.sqrt(n_torch)
    new = np.zeros_like(target)
    for torch_idx, Z in enumerate(torch_atomic_numbers):
        if 0 <= Z < new.shape[0]:
            new[Z] = matrix[torch_idx]
    rep_params["LinearNodeEmbedding_0"]["weight"] = new.astype(target.dtype)


def _map_scale_shift(
    ss_params: dict,
    *,
    global_scale: float,
    global_shift: float,
    atomic_energies: np.ndarray,
    torch_atomic_numbers: tuple[int, ...],
) -> None:
    """Fold ``(global_scale, global_shift, atomic_energies)`` into PerElementScaleShift.

    The torch foundation model emits ``E_i = scale * E_pred_i + shift +
    atomic_energies[Z_i]``. apax's :class:`PerElementScaleShift` does the same
    thing per-element with no separate atomic-energies block, so we collapse:

    - ``scale_per_element[Z] = global_scale`` for every species (foundation
      models only have a single global scale).
    - ``shift_per_element[Z] = global_shift + atomic_energies[torch_idx]`` for
      species in the torch table; ``0`` (default init) elsewhere.

    Parameters
    ----------
    ss_params : dict
        The ``scale_shift`` subtree of the apax pytree (mutated in place).
    global_scale, global_shift : float
        Scalars from ``model.scale_shift``.
    atomic_energies : np.ndarray
        1D atomic-energy vector, length equals ``len(torch_atomic_numbers)``.
    torch_atomic_numbers : tuple of int
        Maps each torch row to its physical Z value.
    """
    n_species = ss_params["scale_per_element"].shape[0]
    scale = np.full((n_species, 1), global_scale, dtype=np.float64)
    shift = np.zeros((n_species, 1), dtype=np.float64)
    # ``atomic_energies`` is 1-D ``(n_species,)`` for some foundations and
    # 2-D ``(n_heads, n_species)`` for others (matpes ships a 2-D table even
    # with a single head). Single-head foundations always use head 0.
    ae = np.asarray(atomic_energies)
    if ae.ndim == 2:
        ae = ae[0]
    for torch_idx, Z in enumerate(torch_atomic_numbers):
        if 0 <= Z < n_species:
            shift[Z, 0] = global_shift + float(ae[torch_idx])
    ss_params["scale_per_element"] = scale.astype(
        ss_params["scale_per_element"].dtype
    )
    ss_params["shift_per_element"] = shift.astype(
        ss_params["shift_per_element"].dtype
    )


def _map_interactions(
    state: dict[str, np.ndarray],
    rep_params: dict,
    torch_atomic_numbers: tuple[int, ...],
    config,
) -> None:
    """Map every ``interactions.k.*`` weight into apax ``InteractionBlock_k``.

    Per-block torch keys (k = 0 .. num_interactions-1)::

        interactions.k.linear_up.weight                (M*M,)
        interactions.k.linear.weight                   (n_paths*M*M,)
        interactions.k.skip_tp.weight                  (M*n_torch*M,)
        interactions.k.conv_tp_weights.layerN.weight   radial-MLP layer N

    All slots are direct copies / scatters now that the apax
    :class:`~apax.layers.descriptor.mace_blocks.InteractionBlock` carries the
    full multi-irrep target plus a per-element skip:

    - ``linear_up`` : single ``Mx0e → Mx0e`` block, flat reshape ``(M, M)``.
    - ``linear``    : torch ``e3nn.o3.Linear`` with one diagonal instruction
      per output irrep (e.g. four ``(128, 128)`` blocks for
      ``128x0e+128x1o+128x2e+128x3o``). Slice the flat torch weight
      block-by-block in the order torch's :meth:`instructions` produces them
      (which matches the e3nn-jax ``w[i,i]`` slot order in
      :class:`e3nn_jax.flax.Linear`). Both implementations apply the same
      ``path_weight = 1/sqrt(M)`` normalisation, so the values are
      bit-for-bit copies.
    - ``skip_tp``   : torch ``FullyConnectedTensorProduct(node_feats × node_attrs
      → hidden)``. apax replaces this with
      ``e3nn.flax.Linear(e3nn.tensor_product(node_feats, Z_one_hot))`` whose
      weight is ``(M_in * n_species, M_out)``. The ``e3nn.tensor_product``
      layout puts ``i*n_species + Z_phys`` in row order, so we scatter
      ``apax_W[i*119 + Z_phys, k_out] = torch_W[i, torch_idx, k_out]`` and
      leave rows for missing species at zero. The two paths share the same
      ``path_weight = 1/sqrt(M_in * n_species)`` normalisation.
    - ``radial_mlp`` Dense_{0..3} : both torch ``FullyConnectedNet`` layers
      and ``e3nn_jax.flax.MultiLayerPerceptron`` store kernels as
      ``(in, out)``. Direct reshape across all four layers — the torch model
      already targets the full-irrep ``n_paths`` (= ``num_irreps`` of
      ``interaction_irreps``) so no slicing is needed.

    Parameters
    ----------
    state : dict of str to np.ndarray
        Torch state_dict.
    rep_params : dict
        Representation subtree of the apax pytree (mutated in place).
    torch_atomic_numbers : tuple of int
        Maps each torch element row to its physical Z value.
    config : MaceModelConfig
        Used for ``num_interactions`` and ``hidden_irreps``.
    """
    import e3nn_jax as e3nn  # noqa: PLC0415

    n_torch = len(torch_atomic_numbers)
    hidden_irreps = e3nn.Irreps(config.hidden_irreps)
    M = hidden_irreps.filter("0e").dim  # scalar channel count (= num_features)

    # Resolve per-layer interaction class (str → broadcast, list → as-is).
    if isinstance(config.interaction_cls, str):
        per_layer_cls = [config.interaction_cls] * int(config.num_interactions)
    else:
        per_layer_cls = list(config.interaction_cls)

    for k in range(int(config.num_interactions)):
        block = rep_params[f"InteractionBlock_{k}"]
        prefix = f"interactions.{k}."

        # 1. linear_up : plain e3nn.o3.Linear in shared irreps.  For scalar-only
        #    layers (small foundation, layer 0 of medium/mpa-0) this is one
        #    (M, M) slot; for multi-irrep layers (layer 1 of medium/mpa-0) it
        #    is one (M, M) slot per shared irrep, in instruction order.
        _scatter_o3_linear_blocks(
            block["linear_up"], state[prefix + "linear_up.weight"], M=M,
        )

        # 2. linear : multi-irrep e3nn.o3.Linear flat → per-irrep apax slots.
        #    torch stores 4 (M, M) blocks back-to-back in the order produced by
        #    its ``instructions`` (which the e3nn-jax slot keys mirror).
        _scatter_o3_linear_blocks(
            block["linear"], state[prefix + "linear.weight"], M=M,
        )

        # 3. skip_tp : torch FCTP → apax Linear(tensor_product(in, attrs)).
        #    One path per output irrep reachable from the input. Torch packs
        #    every path's (mul_in, n_torch, mul_out) block back-to-back in
        #    instruction order, which matches lexicographic sort of apax slot
        #    keys (e.g. ``w[0,0]`` < ``w[1,1]`` < ...).
        _scatter_skip_tp_blocks(
            block["skip_tp"],
            state[prefix + "skip_tp.weight"],
            n_torch=n_torch,
            torch_atomic_numbers=torch_atomic_numbers,
        )

        # 4. radial MLP : direct kernel copy for every layer.
        radial = block["radial_mlp"]
        for layer_idx in range(4):
            target = radial[f"Dense_{layer_idx}"]["kernel"]
            torch_w = state[
                prefix + f"conv_tp_weights.layer{layer_idx}.weight"
            ]
            if torch_w.shape != target.shape:
                raise ValueError(
                    f"radial_mlp Dense_{layer_idx} shape mismatch: "
                    f"torch {torch_w.shape} vs apax {target.shape}"
                )
            radial[f"Dense_{layer_idx}"]["kernel"] = (
                np.asarray(torch_w).astype(target.dtype)
            )

        # 5. density_fn (Density variants only).
        variant = per_layer_cls[k]
        if variant in ("RealAgnosticDensity", "RealAgnosticDensityResidual"):
            target = block["density_fn"]["Dense_0"]["kernel"]
            torch_w = state[prefix + "density_fn.layer0.weight"]
            if torch_w.shape != target.shape:
                raise ValueError(
                    f"density_fn shape mismatch for block {k}: torch "
                    f"{torch_w.shape} vs apax {target.shape}"
                )
            block["density_fn"]["Dense_0"]["kernel"] = (
                np.asarray(torch_w).astype(target.dtype)
            )


def _scatter_o3_linear_blocks(
    block_params: dict,
    flat: np.ndarray,
    *,
    M: int = 0,  # noqa: ARG001  retained for back-compat with prior signature
) -> None:
    """Slice a flat ``e3nn.o3.Linear`` weight into apax slots, size-aware.

    Each diagonal instruction in torch's ``e3nn.o3.Linear`` contributes a
    ``(mul_in, mul_out)`` block to the flat weight in instruction order.
    Sorted apax slot keys (``w[i,i] ...``) match that order, but the per-slot
    shape may vary across slots:

    - Uniform case (``small`` foundation, ``mpa-0`` layer 0 ``linear``): every
      slot is ``(M, M)``.
    - Non-uniform case (``mpa-0`` / ``matpes`` layer 1 ``linear``): multiple
      input paths reach the same output irrep, collapsed by ``simplify()``,
      so an output irrep's slot has shape ``(sum_of_path_muls_in, mul_out)``
      where ``sum_of_path_muls_in`` is generally a small multiple of ``M``.

    This implementation reads each slot's shape from the target and consumes
    ``target.size`` flat elements per slot. The legacy ``M`` parameter is
    retained for callers that still pass it but is not used internally.

    Parameters
    ----------
    block_params : dict
        The ``InteractionBlock_k.linear`` (or analogous) sub-dict (mutated
        in place).
    flat : np.ndarray
        Flat torch ``Linear.weight`` tensor.
    M : int, optional
        Unused — preserved for source-compatibility with prior call sites.
    """
    keys = sorted(block_params.keys())
    offset = 0
    for key in keys:
        target = block_params[key]
        n = target.size
        if offset + n > flat.size:
            raise ValueError(
                f"Linear slot {key!r}: flat weight only has {flat.size - offset} "
                f"elements remaining but slot needs {n}."
            )
        block_params[key] = (
            flat[offset : offset + n].reshape(target.shape).astype(target.dtype)
        )
        offset += n
    if offset != flat.size:
        raise ValueError(
            f"e3nn.o3.Linear flat weight has {flat.size - offset} unconsumed "
            f"elements after filling {len(keys)} slots."
        )


def _scatter_skip_tp_blocks(
    block_params: dict,
    flat: np.ndarray,
    *,
    n_torch: int,
    torch_atomic_numbers: tuple[int, ...],
) -> None:
    """Scatter torch FCTP weight into the apax tensor-product Linear slots.

    Each reachable output irrep gets one slot of shape
    ``(mul_in1 * n_species_apax, mul_out)``. Torch packs every path's
    ``(mul_in1, n_torch, mul_out)`` block back-to-back in instruction order;
    lexicographic sort of apax slot keys (``w[0,0] ...``, ``w[1,1] ...``, ...)
    matches that order.

    Per-path scatter: for each ``(torch_idx, Z)`` in ``torch_atomic_numbers``,
    the apax slot row ``i * n_species_apax + Z`` (matching the
    ``e3nn.tensor_product(node_feats, Z_one_hot)`` layout) takes the torch
    weight scaled by ``sqrt(n_species_apax / n_torch)`` to compensate for the
    different ``path_weight = 1/sqrt(mul_in1 * n_species)`` normalisations
    on either side.

    Parameters
    ----------
    block_params : dict
        The ``skip_tp`` sub-dict of an apax ``InteractionBlock_k`` (mutated
        in place). Each value is a flat ``(mul_in1 * n_species_apax, mul_out)``
        weight slot.
    flat : np.ndarray
        Torch FCTP ``skip_tp.weight``.
    n_torch : int
        Foundation-side species table size (``len(model.atomic_numbers)``).
    torch_atomic_numbers : tuple of int
        Maps each torch row to its physical Z value.
    """
    keys = sorted(block_params.keys())
    if not keys:
        if flat.size != 0:
            raise ValueError(
                f"skip_tp has no slots but torch flat weight has size {flat.size}."
            )
        return

    # Read mul_in1 / mul_out / n_species_apax from the first slot. e3nn-jax
    # creates uniform shapes per slot (one path per output irrep, all with the
    # same mul on input and output sides for these foundation models).
    sample = block_params[keys[0]]
    mul_out = sample.shape[1]
    if sample.shape[0] % n_torch != 0:
        # If torch's species table is smaller than apax's, the apax slot rows
        # are mul_in1 * n_species_apax; back out mul_in1 by dividing by the
        # apax-side species count we infer from the slot shape and n_torch.
        # Heuristic: mul_in1 must be a power-of-two-ish factor that divides
        # both shape[0] and torch's expected per-path size.
        pass  # validated per-path below
    # Path numel: mul_in1 * n_torch * mul_out.
    path_size = flat.size // len(keys)
    expected_total = path_size * len(keys)
    if flat.size != expected_total:
        raise ValueError(
            f"skip_tp flat size {flat.size} not divisible into {len(keys)} "
            f"equal-sized paths."
        )
    if path_size % (n_torch * mul_out) != 0:
        raise ValueError(
            f"skip_tp path size {path_size} not divisible by n_torch*mul_out "
            f"({n_torch}*{mul_out})."
        )
    mul_in1 = path_size // (n_torch * mul_out)
    n_species_apax = sample.shape[0] // mul_in1
    if sample.shape != (mul_in1 * n_species_apax, mul_out):
        raise ValueError(
            f"skip_tp slot shape {sample.shape} not consistent with "
            f"(mul_in1*n_species_apax={mul_in1*n_species_apax}, "
            f"mul_out={mul_out})."
        )

    # path_weight compensation: torch FCTP uses 1/sqrt(mul_in1 * n_torch);
    # apax Linear uses 1/sqrt(mul_in1 * n_species_apax). Scale copied weights
    # by sqrt(n_species_apax / n_torch).
    scale = float(np.sqrt(n_species_apax / n_torch))

    for path_idx, key in enumerate(keys):
        target = block_params[key]
        if target.shape != (mul_in1 * n_species_apax, mul_out):
            raise ValueError(
                f"skip_tp slot {key!r} has shape {target.shape}; expected "
                f"({mul_in1 * n_species_apax}, {mul_out})."
            )
        path_slice = flat[path_idx * path_size : (path_idx + 1) * path_size]
        torch_w_3d = path_slice.reshape(mul_in1, n_torch, mul_out)
        new = np.zeros_like(target)
        for torch_idx, Z in enumerate(torch_atomic_numbers):
            if not (0 <= Z < n_species_apax):
                continue
            for i in range(mul_in1):
                new[i * n_species_apax + Z, :] = torch_w_3d[i, torch_idx, :] * scale
        block_params[key] = new.astype(target.dtype)


def _map_pair_repulsion(
    state: dict[str, np.ndarray],
    out: dict,
    *,
    correction_index: int,
) -> None:
    """Copy torch ``pair_repulsion_fn`` weights into the apax ZBL slot.

    The :class:`MaceZBLPairRepulsion` Linen module registers ``c``,
    ``covalent_radii`` (and, when ``trainable=False``, ``a_exp`` /
    ``a_prefactor``) as variables in the ``buffers`` collection; with
    ``trainable=True`` ``a_exp`` and ``a_prefactor`` live in ``params``
    instead. The resolved pytree path is
    ``{collection}/energy_model/corrections_{correction_index}/<leaf>``
    where ``correction_index`` is the position of the ``mace_zbl`` entry in
    the original ``empirical_corrections`` config list.

    Parameters
    ----------
    state : dict of str to np.ndarray
        Torch ``state_dict`` arrays (``pair_repulsion_fn.*``).
    out : dict
        The full apax pytree (mutated in place). Must contain at least
        ``out["buffers"]["energy_model"][f"corrections_{correction_index}"]``.
    correction_index : int
        Index of the ``mace_zbl`` correction in the empirical_corrections
        list. For foundation models in scope (single ZBL correction) this is
        always ``0``.
    """
    slot_key = f"corrections_{correction_index}"

    # Buffers collection always carries c + covalent_radii; a_exp / a_prefactor
    # live here when trainable=False.
    buf_slot = out["buffers"]["energy_model"][slot_key]
    buf_slot["c"] = np.asarray(state["pair_repulsion_fn.c"]).astype(
        buf_slot["c"].dtype
    )
    buf_slot["covalent_radii"] = np.asarray(
        state["pair_repulsion_fn.covalent_radii"]
    ).astype(buf_slot["covalent_radii"].dtype)

    # a_exp / a_prefactor: present in whichever collection the template
    # initialized them in. Try buffers first (trainable=False), fall back to
    # params (trainable=True).
    for name in ("a_exp", "a_prefactor"):
        torch_arr = np.asarray(state[f"pair_repulsion_fn.{name}"])
        if name in buf_slot:
            buf_slot[name] = torch_arr.astype(buf_slot[name].dtype)
        else:
            par_slot = out["params"]["energy_model"][slot_key]
            par_slot[name] = torch_arr.astype(par_slot[name].dtype)


def _map_distance_transform(
    state: dict[str, np.ndarray],
    out: dict,
) -> None:
    """Copy torch ``radial_embedding.distance_transform`` into the apax slot.

    The :class:`AgnesiTransform` Linen module is attached as a field on
    :class:`MaceRepresentation`, so its leaves land in the pytree at::

        buffers/energy_model/representation/distance_transform/{a,q,p,covalent_radii}

    When ``trainable=True`` the three scalar parameters ``a`` / ``q`` / ``p``
    move from the ``buffers`` collection to ``params``. ``covalent_radii``
    always stays in ``buffers``.

    Parameters
    ----------
    state : dict of str to np.ndarray
        Torch ``state_dict`` arrays. The four expected keys are
        ``radial_embedding.distance_transform.{a,q,p,covalent_radii}``.
    out : dict
        Full apax pytree (mutated in place). Must already carry
        ``out["buffers"]["energy_model"]["representation"]["distance_transform"]``.
    """
    rep_buf = out["buffers"]["energy_model"]["representation"]
    dt_buf = rep_buf["distance_transform"]
    dt_buf["covalent_radii"] = np.asarray(
        state["radial_embedding.distance_transform.covalent_radii"]
    ).astype(dt_buf["covalent_radii"].dtype)
    for name in ("a", "q", "p"):
        torch_arr = np.asarray(state[f"radial_embedding.distance_transform.{name}"])
        if name in dt_buf:
            dt_buf[name] = torch_arr.astype(dt_buf[name].dtype)
        else:
            par_slot = out["params"]["energy_model"]["representation"][
                "distance_transform"
            ]
            par_slot[name] = torch_arr.astype(par_slot[name].dtype)


def _map_products(
    state: dict[str, np.ndarray],
    rep_params: dict,
    torch_atomic_numbers: tuple[int, ...],
    config,
    *,
    torch_model,
) -> None:
    """Map ``products.k.*`` into apax ``ProductBlock_k`` (SC + post-Linear).

    Two slots per layer:

    1. ``ProductBlock_k.weight`` — symmetric-contraction per-element weight
       tensor of shape ``(num_elements_apax, basis_dim, mul)``. The torch
       native module stores its weights per-contraction and per-degree
       (``contractions.{c}.weights_max`` for the top correlation order plus
       ``contractions.{c}.weights.{m}`` for the lower degrees). Reshaping
       these into the cue descriptor's canonical basis requires the
       full-CG transform, which is expensive to derive on the fly. We
       delegate to mace-jax's reference adapter
       :func:`mace_jax.adapters.cuequivariance.symmetric_contraction._convert_native_weights`
       — it's the same code path mace-jax uses for its own torch→jax import.

       The adapter returns a ``(num_elements_torch, basis_dim, mul)`` tensor
       in the cue descriptor's canonical basis, which we then scatter row-wise
       into the apax slot using ``torch_atomic_numbers`` so physical Z values
       index the table. Rows for missing species stay at their init value.

    2. ``ProductBlock_k.linear`` — post-SC ``e3nn.o3.Linear(target_irreps →
       target_irreps)`` whose weight is a single ``(M, M)`` block (the
       small/medium MP-0 ``target_irreps == "128x0e"``). Direct flat reshape;
       see :func:`_scatter_o3_linear_blocks` for the multi-block contract.

    Parameters
    ----------
    state : dict of str to np.ndarray
        Torch state_dict.
    rep_params : dict
        Representation subtree of the apax pytree (mutated in place).
    torch_atomic_numbers : tuple of int
        Maps each torch element row to its physical Z value.
    config : MaceModelConfig
        Used for ``num_interactions``.
    torch_model : torch.nn.Module
        Live torch foundation model. ``products[k].symmetric_contractions`` is
        passed to mace-jax's adapter to handle the full-CG transform.
    """
    import jax.numpy as jnp  # noqa: PLC0415
    from mace_jax.adapters.cuequivariance.symmetric_contraction import (  # noqa: PLC0415
        _convert_native_weights,
    )

    n_torch = len(torch_atomic_numbers)
    for k in range(int(config.num_interactions)):
        block = rep_params[f"ProductBlock_{k}"]
        target = block["weight"]
        n_species_apax, basis_dim, mul = target.shape

        # Build the full-CG-aware conversion through mace-jax's reference
        # adapter; ``target_template`` only carries the desired (basis, mul)
        # shape and dtype, never values.
        torch_sc = torch_model.products[k].symmetric_contractions
        native_template = jnp.zeros(
            (n_torch, basis_dim, mul), dtype=target.dtype
        )
        converted = np.asarray(
            _convert_native_weights(torch_sc, target_template=native_template)
        )
        if converted.shape != (n_torch, basis_dim, mul):
            raise ValueError(
                f"products.{k} SC adapter returned shape {converted.shape}; "
                f"expected ({n_torch}, {basis_dim}, {mul})."
            )

        new = np.zeros_like(target)
        for torch_idx, Z in enumerate(torch_atomic_numbers):
            if 0 <= Z < n_species_apax:
                new[Z] = converted[torch_idx]
        block["weight"] = new.astype(target.dtype)

        # Post-SC Linear: mirrors interactions.linear handling.
        _scatter_o3_linear_blocks(
            block["linear"],
            state[f"products.{k}.linear.weight"],
            M=mul,
        )


def _map_readouts(
    state: dict[str, np.ndarray],
    readout_params: dict,
    *,
    num_interactions: int,
) -> None:
    """Map per-layer readout weights into ``MaceReadout_0/readout_{k}``.

    For a model with ``num_interactions = N``:

    - Layers ``0 .. N-2`` : :class:`LinearReadoutBlock`
      torch ``readouts.k.linear.weight (M,)`` →
      apax ``readout_k/linear/w[0,0] 128x0e,1x0e (M, 1)``.
    - Layer ``N-1`` : :class:`NonLinearReadoutBlock`
      torch ``readouts.{N-1}.linear_1.weight (M*MLP,)`` →
      apax ``readout_{N-1}/linear_1/w[0,0] 128x0e,16x0e (M, MLP)``.
      torch ``readouts.{N-1}.linear_2.weight (MLP,)`` →
      apax ``readout_{N-1}/linear_2/w[0,0] 16x0e,1x0e (MLP, 1)``.

    Parameters
    ----------
    state : dict of str to np.ndarray
        Torch state_dict.
    readout_params : dict
        Readout subtree of the apax pytree (mutated in place). Direct keys are
        the per-layer ``readout_k`` blocks (the :class:`MaceReadout` linen
        module is the readout slot itself, so its child names appear at the
        top level of the readout subtree).
    num_interactions : int
        Total number of interaction layers; equals number of readouts.
    """
    for k in range(num_interactions):
        sub = readout_params[f"readout_{k}"]
        if k < num_interactions - 1:
            torch_w = state[f"readouts.{k}.linear.weight"]
            target = sub["linear"]["w[0,0] 128x0e,1x0e"]
            if torch_w.size != target.size:
                raise ValueError(
                    f"linear readout {k} size {torch_w.size} != apax {target.size}"
                )
            sub["linear"]["w[0,0] 128x0e,1x0e"] = (
                torch_w.reshape(target.shape).astype(target.dtype)
            )
        else:
            t1 = state[f"readouts.{k}.linear_1.weight"]
            t1_target = sub["linear_1"]["w[0,0] 128x0e,16x0e"]
            if t1.size != t1_target.size:
                raise ValueError(
                    f"non-linear readout {k} linear_1 size {t1.size} != "
                    f"apax {t1_target.size}"
                )
            sub["linear_1"]["w[0,0] 128x0e,16x0e"] = (
                t1.reshape(t1_target.shape).astype(t1_target.dtype)
            )
            t2 = state[f"readouts.{k}.linear_2.weight"]
            t2_target = sub["linear_2"]["w[0,0] 16x0e,1x0e"]
            if t2.size != t2_target.size:
                raise ValueError(
                    f"non-linear readout {k} linear_2 size {t2.size} != "
                    f"apax {t2_target.size}"
                )
            sub["linear_2"]["w[0,0] 16x0e,1x0e"] = (
                t2.reshape(t2_target.shape).astype(t2_target.dtype)
            )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _extract_norm_consts() -> dict[str, float]:
    """Fetch torch ``normalize2mom`` constants for common gates (fail fast).

    Parity relies on reusing the exact ``normalize2mom`` constants that torch
    precomputed for its activation wrappers. If they cannot be obtained we raise
    instead of silently recomputing a different value.

    Returns
    -------
    dict of str to float
        Mapping of activation name to ``normalize2mom`` constant,
        e.g. ``{"silu": 1.7868..., "swish": 1.7868...}``.

    Raises
    ------
    ImportError
        If torch or e3nn are not available.
    RuntimeError
        If the constant cannot be computed from the loaded torch modules.
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


def _validate_no_nan(pytree: Any) -> None:
    """Check that no floating-point leaf in a pytree contains NaN.

    Parameters
    ----------
    pytree : pytree
        Nested parameter pytree to validate.

    Raises
    ------
    ValueError
        If any floating-point leaf array contains at least one NaN. The
        message lists every offending leaf path so the next mapping step is
        obvious.
    """
    import jax  # noqa: PLC0415

    bad: list[str] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(pytree)[0]:
        arr = np.asarray(leaf)
        if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
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
        import mace  # noqa: PLC0415

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
        from apax import __version__  # noqa: PLC0415

        return __version__
    except Exception:
        return "unknown"
