"""MACE foundation model conversion utilities.

``run_conversion`` is called by the ``apax convert-mace`` CLI. It writes a
directory in apax's standard training-output layout that loads through
:func:`apax.train.checkpoints.restore_parameters` like any other apax model.

torch / mace are imported lazily so this module is safe to import without
those optional deps installed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from apax.nn.builder import DEFAULT_N_SPECIES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from apax.config.train_config import Config


_TORCH_TO_APAX_INTERACTION = {
    "RealAgnosticResidualInteractionBlock": "RealAgnosticResidual",
    "RealAgnosticDensityInteractionBlock": "RealAgnosticDensity",
    "RealAgnosticDensityResidualInteractionBlock": "RealAgnosticDensityResidual",
}

# Z=0 is reserved for padding; physical Z values are scattered into
# [1, DEFAULT_N_SPECIES - 1]. Sourced from the builder so there is one
# canonical element-table size across apax.


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

    torch_model, resolved_path = _load_torch_foundation_model(source, family=family)
    mace_cfg_fields = _extract_config_from_torch(torch_model, head=head)
    torch_atomic_numbers = tuple(
        int(z) for z in torch_model.atomic_numbers.detach().cpu().numpy().tolist()
    )
    full_cfg = _synthesize_full_config(mace_cfg_fields, dst)

    builder_cls = full_cfg.model.get_builder()
    builder = builder_cls(full_cfg.model.model_dump(), n_species=DEFAULT_N_SPECIES)
    energy_derivative_model = builder.build_energy_derivative_model()

    R_dummy = jnp.zeros((2, 3))
    Z_dummy = jnp.array([1, 1], dtype=jnp.int32)
    neigh_dummy = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    box_dummy = jnp.zeros((3,))
    offsets_dummy = jnp.zeros((neigh_dummy.shape[1], 3))
    params_template = energy_derivative_model.init(
        jax.random.PRNGKey(0),
        R_dummy,
        Z_dummy,
        neigh_dummy,
        box_dummy,
        offsets_dummy,
    )

    state = {k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()}
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

    dst.mkdir(parents=True, exist_ok=True)
    full_cfg.dump_config(dst)
    _write_orbax_checkpoint(dst / "best", params, epoch=0)

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

    interactions: list[dict] = []
    for inter in model.interactions:
        cls_name = type(inter).__name__
        if cls_name not in _TORCH_TO_APAX_INTERACTION:
            raise NotImplementedError(
                f"Foundation uses interaction block {cls_name!r}; apax supports "
                f"{sorted(_TORCH_TO_APAX_INTERACTION)}. In particular the "
                "non-residual first-layer block 'RealAgnosticInteractionBlock' is "
                "intentionally not ported: every MACE foundation in scope "
                "(small/medium/MPA-0) is Residual-first, so the math is unused "
                "(YAGNI). Port the block and add it to _TORCH_TO_APAX_INTERACTION "
                "before converting such a model."
            )
        interactions.append({"name": _TORCH_TO_APAX_INTERACTION[cls_name]})

    hidden_irreps = str(model.products[0].linear.irreps_out)

    sph_irreps = e3nn.o3.Irreps(str(model.spherical_harmonics.irreps_out))
    max_ell = max(ir.l for _, ir in sph_irreps)

    sc0 = model.products[0].symmetric_contractions.contractions[0]
    correlation = 1
    while hasattr(sc0, f"U_matrix_{correlation + 1}"):
        correlation += 1

    avg_num_neighbors = float(model.interactions[0].avg_num_neighbors)

    empirical_corrections: list[dict] = []
    if hasattr(model, "pair_repulsion_fn"):
        import torch  # noqa: PLC0415

        zbl = model.pair_repulsion_fn
        p_value = int(zbl.p.detach().cpu().item())
        is_trainable = isinstance(zbl.a_exp, torch.nn.Parameter)
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
    n_basis = int(model.radial_embedding.bessel_fn.bessel_weights.shape[0])

    cfg = {
        "name": "mace",
        "basis": {
            "name": "bessel",
            "variant": "standard",
            "n_basis": n_basis,
            "r_max": r_max,
        },
        "radial_embedding": {
            "num_polynomial_cutoff": int(model.radial_embedding.cutoff_fn.p),
            "distance_transform": distance_transform_cfg,
        },
        "descriptor": {
            "max_ell": int(max_ell),
            "hidden_irreps": hidden_irreps,
            "correlation": int(correlation),
            "interactions": interactions,
            "avg_num_neighbors": avg_num_neighbors,
        },
        "readout": {"MLP_irreps": "16x0e"},
        "empirical_corrections": empirical_corrections,
        "descriptor_dtype": "fp64",
        "readout_dtype": "fp64",
        "scale_shift_dtype": "fp64",
    }
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
    """Translate torch ``state_dict`` arrays into a linen pytree matching ``template``.

    Every torch float parameter has a same-numel apax slot, so this mapper
    performs direct copies / scatters only. The only non-trivial reshape is
    the symmetric-contraction weight, which is delegated to mace-jax's
    reference adapter to reuse its full-CG transform.

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
        num_interactions=len(config.descriptor.interactions),
    )
    _map_scale_shift(
        energy_params["scale_shift"],
        global_scale=float(extra_scalars["scale"]),
        global_shift=float(extra_scalars["shift"]),
        atomic_energies=np.asarray(extra_scalars["atomic_energies"]),
        torch_atomic_numbers=torch_atomic_numbers,
    )
    if hasattr(torch_model, "pair_repulsion_fn"):
        correction_index, zbl_cfg = next(
            (i, c)
            for i, c in enumerate(config.empirical_corrections)
            if c.name == "mace_zbl"
        )
        _map_pair_repulsion(
            state,
            out,
            correction_index=correction_index,
            trainable=zbl_cfg.trainable,
        )
    if config.radial_embedding.distance_transform is not None:
        _map_distance_transform(
            state,
            out,
            trainable=config.radial_embedding.distance_transform.trainable,
        )
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
    embedding table is ``(DEFAULT_N_SPECIES, hidden)``; rows for chemical species not
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
    emb_block = _require_slots(
        rep_params, "LinearNodeEmbedding_", 1, context="node embedding"
    )[0]
    target = emb_block["weight"]
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
    emb_block["weight"] = new.astype(target.dtype)


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
    ss_params["scale_per_element"] = scale.astype(ss_params["scale_per_element"].dtype)
    ss_params["shift_per_element"] = shift.astype(ss_params["shift_per_element"].dtype)


def _require_slots(params: dict, prefix: str, count: int, *, context: str) -> list:
    """Return the ``count`` submodule dicts named ``f"{prefix}{i}"``.

    The converter maps torch weights into auto-generated Flax module slots
    (``InteractionBlock_k``, ``ProductBlock_k``, ``readout_k``). This helper
    resolves those slots by position and raises an actionable error naming the
    missing slot and the keys that *are* present, instead of letting a bare
    ``KeyError`` surface deep inside a mapping loop when the apax module layout
    drifts.

    Parameters
    ----------
    params : dict
        Parameter sub-tree expected to contain the numbered slots.
    prefix : str
        Slot name prefix, e.g. ``"InteractionBlock_"``.
    count : int
        Number of consecutive slots expected, indexed ``0 .. count - 1``.
    context : str
        Short description of the mapping step, used in the error message.

    Returns
    -------
    list
        The ``count`` slot dicts in index order (the same objects stored in
        ``params``, so in-place mutation by callers still applies).

    Raises
    ------
    KeyError
        If any expected slot is absent from ``params``.
    """
    missing = [f"{prefix}{i}" for i in range(count) if f"{prefix}{i}" not in params]
    if missing:
        present = sorted(k for k in params if k.startswith(prefix))
        raise KeyError(
            f"{context}: expected module slots {missing} in the apax parameter "
            f"tree but found only {present}. The MACE module layout in "
            f"apax/layers/descriptor (auto-generated Flax names) likely changed; "
            f"update the foundation converter mapping to match."
        )
    return [params[f"{prefix}{i}"] for i in range(count)]


def _require_key(mapping: dict, key: str, *, context: str):
    """Return ``mapping[key]`` or raise an actionable error.

    The converter walks fixed-name slots in the apax pytree (e.g.
    ``distance_transform``). This raises a message naming the missing key and the
    keys that are present — pointing at the converter mapping — instead of a bare
    ``KeyError`` when the apax module layout drifts.

    Parameters
    ----------
    mapping : dict
        Parameter sub-tree expected to contain ``key``.
    key : str
        The expected key.
    context : str
        Short description of the mapping step, used in the error message.

    Returns
    -------
    Any
        ``mapping[key]``.

    Raises
    ------
    KeyError
        If ``key`` is absent from ``mapping``.
    """
    if key not in mapping:
        raise KeyError(
            f"{context}: expected key {key!r} in the apax parameter tree but found "
            f"only {sorted(mapping)}. The MACE module layout in "
            f"apax/layers/descriptor (auto-generated Flax names) likely changed; "
            f"update the foundation converter mapping to match."
        )
    return mapping[key]


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
        Used for ``descriptor.interactions`` and ``descriptor.hidden_irreps``.
    """
    n_torch = len(torch_atomic_numbers)
    per_layer_cls = [i.name for i in config.descriptor.interactions]

    blocks = _require_slots(
        rep_params, "InteractionBlock_", len(per_layer_cls), context="interactions"
    )
    for k in range(len(per_layer_cls)):
        block = blocks[k]
        prefix = f"interactions.{k}."

        # 1. linear_up : plain e3nn.o3.Linear in shared irreps.
        _scatter_o3_linear_blocks(
            block["linear_up"],
            state[prefix + "linear_up.weight"],
        )

        # 2. linear : multi-irrep e3nn.o3.Linear flat → per-irrep apax slots.
        _scatter_o3_linear_blocks(
            block["linear"],
            state[prefix + "linear.weight"],
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
            torch_w = state[prefix + f"conv_tp_weights.layer{layer_idx}.weight"]
            if torch_w.shape != target.shape:
                raise ValueError(
                    f"radial_mlp Dense_{layer_idx} shape mismatch: "
                    f"torch {torch_w.shape} vs apax {target.shape}"
                )
            radial[f"Dense_{layer_idx}"]["kernel"] = np.asarray(torch_w).astype(
                target.dtype
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
            block["density_fn"]["Dense_0"]["kernel"] = np.asarray(torch_w).astype(
                target.dtype
            )


def _scatter_o3_linear_blocks(
    block_params: dict,
    flat: np.ndarray,
) -> None:
    """Slice a flat ``e3nn.o3.Linear`` weight into apax slots, size-aware.

    Each diagonal instruction in torch's ``e3nn.o3.Linear`` contributes a
    ``(mul_in, mul_out)`` block to the flat weight in instruction order.
    Sorted apax slot keys (``w[i,i] ...``) match that order; per-slot shapes
    are read from the target so multi-path collapse cases (multiple input
    paths reaching the same output irrep) are handled uniformly.

    Parameters
    ----------
    block_params : dict
        The ``InteractionBlock_k.linear`` (or analogous) sub-dict (mutated
        in place).
    flat : np.ndarray
        Flat torch ``Linear.weight`` tensor.
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
            f"(mul_in1*n_species_apax={mul_in1 * n_species_apax}, "
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
    trainable: bool,
) -> None:
    """Copy torch ``pair_repulsion_fn`` weights into the apax ZBL slot.

    Parameters
    ----------
    state : dict of str to np.ndarray
        Torch ``state_dict`` arrays (``pair_repulsion_fn.*``).
    out : dict
        The full apax pytree (mutated in place).
    correction_index : int
        Index of the ``mace_zbl`` correction in the empirical_corrections list.
    trainable : bool
        Mirrors :class:`MaceZBLPairRepulsion.trainable`. When ``False`` the
        ``a_exp`` / ``a_prefactor`` scalars live in the ``buffers`` collection;
        when ``True`` they live in ``params``.
    """
    slot_key = f"corrections_{correction_index}"
    buf_slot = out["buffers"]["energy_model"][slot_key]
    buf_slot["c"] = np.asarray(state["pair_repulsion_fn.c"]).astype(buf_slot["c"].dtype)
    buf_slot["covalent_radii"] = np.asarray(
        state["pair_repulsion_fn.covalent_radii"]
    ).astype(buf_slot["covalent_radii"].dtype)

    target_slot = out["params"]["energy_model"][slot_key] if trainable else buf_slot
    for name in ("a_exp", "a_prefactor"):
        torch_arr = np.asarray(state[f"pair_repulsion_fn.{name}"])
        target_slot[name] = torch_arr.astype(target_slot[name].dtype)


def _map_distance_transform(
    state: dict[str, np.ndarray],
    out: dict,
    *,
    trainable: bool,
) -> None:
    """Copy torch ``radial_embedding.distance_transform`` into the apax slot.

    The AgnesiTransform is a field of :class:`MaceRadialEmbedding` and is
    called from inside its ``__call__``, so linen materialises the slot at
    ``representation/radial_embedding/distance_transform/...``.

    Parameters
    ----------
    state : dict of str to np.ndarray
        Torch ``state_dict`` arrays.
    out : dict
        Full apax pytree (mutated in place).
    trainable : bool
        Mirrors :class:`AgnesiTransformConfig.trainable`. When ``False`` the
        ``a`` / ``q`` / ``p`` scalars live in ``buffers``; when ``True`` they
        live in ``params``. ``covalent_radii`` always stays in ``buffers``.
    """
    re_buf = out["buffers"]["energy_model"]["representation"]["radial_embedding"]
    dt_buf = _require_key(
        re_buf, "distance_transform", context="distance transform buffer"
    )
    dt_buf["covalent_radii"] = np.asarray(
        state["radial_embedding.distance_transform.covalent_radii"]
    ).astype(dt_buf["covalent_radii"].dtype)

    target_slot = (
        out["params"]["energy_model"]["representation"]["radial_embedding"][
            "distance_transform"
        ]
        if trainable
        else dt_buf
    )
    for name in ("a", "q", "p"):
        torch_arr = np.asarray(state[f"radial_embedding.distance_transform.{name}"])
        target_slot[name] = torch_arr.astype(target_slot[name].dtype)


def _map_products(
    state: dict[str, np.ndarray],
    rep_params: dict,
    torch_atomic_numbers: tuple[int, ...],
    config,
    *,
    torch_model,
) -> None:
    """Map ``products.k.*`` into apax ``ProductBlock_k`` (SC + post-Linear).

    Two slots per layer: ``weight`` (per-element symmetric-contraction tensor
    of shape ``(num_elements_apax, basis_dim, mul)``) and ``linear`` (post-SC
    e3nn Linear). The SC weight requires a non-trivial change-of-basis from
    torch's native layout to the cue descriptor's canonical basis; the work
    is delegated to :func:`apax.transfer_learning.torch_sc_adapter.convert_native_weights`.

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
        Live torch foundation model whose ``products[k].symmetric_contractions``
        is passed to the adapter.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    from apax.transfer_learning.torch_sc_adapter import (  # noqa: PLC0415
        convert_native_weights,
    )

    n_torch = len(torch_atomic_numbers)
    blocks = _require_slots(
        rep_params,
        "ProductBlock_",
        len(config.descriptor.interactions),
        context="products",
    )
    for k in range(len(config.descriptor.interactions)):
        block = blocks[k]
        target = block["weight"]
        n_species_apax, basis_dim, mul = target.shape

        # Build the full-CG-aware conversion through mace-jax's reference
        # adapter; ``target_template`` only carries the desired (basis, mul)
        # shape and dtype, never values.
        torch_sc = torch_model.products[k].symmetric_contractions
        native_template = jnp.zeros((n_torch, basis_dim, mul), dtype=target.dtype)
        converted = np.asarray(
            convert_native_weights(torch_sc, target_template=native_template)
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
        )


def _map_readouts(
    state: dict[str, np.ndarray],
    readout_params: dict,
    *,
    num_interactions: int,
) -> None:
    """Map per-layer readout weights into ``MaceReadout_0/readout_{k}``.

    Layers ``0 .. N-2`` use a :class:`LinearReadoutBlock` (one
    ``e3nn.flax.Linear``); the last layer uses a
    :class:`NonLinearReadoutBlock` (``linear_1`` then ``linear_2``). Each
    Linear has a single weight slot keyed by an irreps string; we discover the
    slot name by reading the apax pytree rather than hard-coding values that
    are tied to a specific hidden width.

    Parameters
    ----------
    state : dict of str to np.ndarray
        Torch state_dict.
    readout_params : dict
        Readout subtree of the apax pytree (mutated in place).
    num_interactions : int
        Total number of interaction layers; equals number of readouts.
    """
    subs = _require_slots(
        readout_params, "readout_", num_interactions, context="readouts"
    )
    for k in range(num_interactions):
        sub = subs[k]
        if k < num_interactions - 1:
            _scatter_o3_linear_blocks(
                sub["linear"],
                state[f"readouts.{k}.linear.weight"],
            )
        else:
            _scatter_o3_linear_blocks(
                sub["linear_1"],
                state[f"readouts.{k}.linear_1.weight"],
            )
            _scatter_o3_linear_blocks(
                sub["linear_2"],
                state[f"readouts.{k}.linear_2.weight"],
            )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


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
        raise ValueError("NaN leaves after conversion:\n  - " + "\n  - ".join(bad))


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
    except (ImportError, AttributeError):
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
    except (ImportError, AttributeError):
        return "unknown"
