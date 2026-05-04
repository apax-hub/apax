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

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from apax.config.train_config import Config


# Map torch interaction class names to apax MaceModelConfig.interaction_cls values.
_INTERACTION_CLS_MAP = {
    "RealAgnosticInteractionBlock": "RealAgnostic",
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
    head: str = "default",
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
    head : str, default = "default"
        For multi-head foundation models (e.g. MPA), which head to retain.
        Single-head models accept the literal string ``"default"``.
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
    params_template = energy_derivative_model.init(
        jax.random.PRNGKey(0),
        R_dummy,
        Z_dummy,
        neigh_dummy,
        box_dummy,
        offsets_dummy,
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


def _extract_config_from_torch(model, head: str) -> dict:
    """Return a dict that matches :class:`MaceModelConfig` schema.

    Reads hyperparameters off a torch ``ScaleShiftMACE`` foundation-model
    instance. Many of the fields are not exposed as top-level attributes; this
    helper walks the submodule tree to discover them.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded MACE torch model (typically a ``ScaleShiftMACE``).
    head : str
        Head selector for multi-head models. Single-head models ignore this
        argument apart from validating it is in ``model.heads`` (or the literal
        ``"default"``).

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
    if head not in heads:
        raise ValueError(
            f"head={head!r} not in available heads {heads}. "
            f"Pass --head <name> from that list."
        )

    inter0_cls = type(model.interactions[0]).__name__
    interaction_cls = _INTERACTION_CLS_MAP.get(inter0_cls)
    if interaction_cls is None:
        raise NotImplementedError(
            f"Unsupported interaction class {inter0_cls!r}; "
            f"supported: {sorted(_INTERACTION_CLS_MAP)}"
        )

    hidden_irreps = str(model.products[0].linear.irreps_out)

    sph_irreps = e3nn.o3.Irreps(str(model.spherical_harmonics.irreps_out))
    max_ell = max(ir.l for _, ir in sph_irreps)

    sc0 = model.products[0].symmetric_contractions.contractions[0]
    correlation = 1
    while hasattr(sc0, f"U_matrix_{correlation + 1}"):
        correlation += 1

    cfg = {
        "r_max": float(model.r_max),
        "num_bessel": int(model.radial_embedding.bessel_fn.bessel_weights.shape[0]),
        "num_polynomial_cutoff": int(model.radial_embedding.cutoff_fn.p),
        "max_ell": int(max_ell),
        "hidden_irreps": hidden_irreps,
        "num_interactions": int(model.num_interactions),
        "correlation": int(correlation),
        "interaction_cls": interaction_cls,
        "use_cueq": False,
        "readout_kind": "mace",
        "MLP_irreps": "16x0e",
        # Float64 throughout for parity with the torch foundation model loaded
        # with default_dtype="float64".
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
) -> dict:
    """Translate torch ``state_dict`` arrays → a linen pytree matching ``template``.

    The returned pytree mirrors the output of
    :meth:`MaceBuilder.build_energy_derivative_model().init(...)`, so its top
    level is ``{"params": {"energy_model": {...}}}``.

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
    _map_products(state, rep, torch_atomic_numbers, config)
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
    matrix = flat.reshape(n_torch, hidden)
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
    for torch_idx, Z in enumerate(torch_atomic_numbers):
        if 0 <= Z < n_species:
            shift[Z, 0] = global_shift + float(atomic_energies[torch_idx])
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

        interactions.k.linear_up.weight       (M_in * M_out,)
        interactions.k.linear.weight          (n_paths * M * M,)
        interactions.k.skip_tp.weight         (M_in * E * M_out,)  per-element
        interactions.k.conv_tp_weights.layerN.weight   radial-MLP layer N

    The current apax :class:`InteractionBlock` is scalar-only (``128x0e`` in
    and out). The torch interaction emits the full ``128x0e+128x1o+128x2e+128x3o``
    irreps. To get a non-NaN pytree we extract:

    - ``linear_up`` : direct reshape, identical layout (both ``128x0e → 128x0e``).
    - ``linear_down`` : the leading ``0e → 0e`` block of the full torch ``linear``.
    - ``skip_linear`` : average of the per-element torch ``skip_tp`` over species.
    - ``radial_mlp`` Dense_{0..2} : direct reshape of torch layer 0..2.
      Dense_3 takes only the first ``n_paths_apax`` (= ``M`` for ``128x0e``
      output) columns of the torch last-layer weight (which has ``n_paths`` for
      the full torch irreps).

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
    M = hidden_irreps.filter("0e").dim  # scalar channel count

    for k in range(int(config.num_interactions)):
        block = rep_params[f"InteractionBlock_{k}"]
        prefix = f"interactions.{k}."

        # 1. linear_up : 128x0e -> 128x0e, flat reshape directly.
        target = block["linear_up"]["w[0,0] 128x0e,128x0e"]
        flat = state[prefix + "linear_up.weight"]
        if flat.size != target.size:
            raise ValueError(
                f"linear_up size mismatch for block {k}: "
                f"torch {flat.size} vs apax {target.size}"
            )
        block["linear_up"]["w[0,0] 128x0e,128x0e"] = flat.reshape(
            target.shape
        ).astype(target.dtype)

        # 2. linear_down : take the leading 0e->0e block of torch's full linear.
        target = block["linear_down"]["w[0,0] 128x0e,128x0e"]
        flat = state[prefix + "linear.weight"]
        # Torch ordering is sorted by output irrep index; for irreps
        # "128x0e+128x1o+128x2e+128x3o" -> "128x0e+...", first M*M elements are
        # the 0e->0e block (see e3nn.o3.Linear instructions).
        n_per_block = M * M
        if flat.size < n_per_block:
            raise ValueError(
                f"interactions.{k}.linear.weight too small ({flat.size}) for "
                f"the leading {n_per_block}-element 0e→0e block"
            )
        block["linear_down"]["w[0,0] 128x0e,128x0e"] = (
            flat[:n_per_block].reshape(M, M).astype(target.dtype)
        )

        # 3. skip_linear : torch skip_tp is per-element (M_in, E, M_out); apax
        #    skip is element-agnostic (M_in, M_out). Average over the torch
        #    element dimension as a reasonable single-tensor projection.
        target = block["skip_linear"]["w[0,0] 128x0e,128x0e"]
        flat = state[prefix + "skip_tp.weight"]
        expected = M * n_torch * M
        if flat.size != expected:
            raise ValueError(
                f"interactions.{k}.skip_tp.weight size {flat.size} != "
                f"expected {expected} (= M_in * E * M_out)"
            )
        per_elem = flat.reshape(M, n_torch, M)
        block["skip_linear"]["w[0,0] 128x0e,128x0e"] = (
            per_elem.mean(axis=1).astype(target.dtype)
        )

        # 4. radial MLP layers.
        radial = block["radial_mlp"]
        for layer_idx in range(4):
            target = radial[f"Dense_{layer_idx}"]["kernel"]
            torch_w = state[
                prefix + f"conv_tp_weights.layer{layer_idx}.weight"
            ]
            if torch_w.shape[0] != target.shape[0]:
                raise ValueError(
                    f"radial_mlp Dense_{layer_idx} input dim mismatch: "
                    f"torch {torch_w.shape[0]} vs apax {target.shape[0]}"
                )
            # The last layer's torch output dim is the number of TP paths for
            # the full torch irreps; apax keeps only the leading block of size
            # `target.shape[1]` (= scalar n_paths).
            cols = min(torch_w.shape[1], target.shape[1])
            sliced = torch_w[:, :cols]
            new = np.zeros_like(target)
            new[:, :cols] = sliced
            radial[f"Dense_{layer_idx}"]["kernel"] = new.astype(target.dtype)


def _map_products(
    state: dict[str, np.ndarray],
    rep_params: dict,
    torch_atomic_numbers: tuple[int, ...],
    config,
) -> None:
    """Map ``products.k.symmetric_contractions.*`` into ``ProductBlock_k/weight``.

    Both the torch ``SymmetricContraction`` and apax's cuequivariance-based
    :class:`ProductBlock` materialise a per-element weight tensor of shape
    ``(num_elements, basis_dim, mul)``. For the small MP-0 model with
    scalar-only ``hidden_irreps="128x0e"`` and ``correlation=3``:

    - torch native: ``weights_max (E, b_max, M)`` + ``weights.{m} (E, b_m, M)``
      concatenated to ``(E, b_native, M)``.
    - apax cue: ``(E, basis_dim, M)`` directly.

    For the scalar-only case the cuequivariance descriptor uses
    ``basis_dim == correlation`` (3), and the torch native blocks for each
    degree are 1-D scalars per correlation order; they map slot-by-slot when
    ``basis_dim == correlation``. We populate the leading
    ``min(basis_dim, native_dim)`` slots and leave any remainder zero.

    Per-element rows are scattered by physical Z; rows for missing species
    stay at their template init.

    Parameters
    ----------
    state : dict of str to np.ndarray
        Torch state_dict.
    rep_params : dict
        Representation subtree of the apax pytree (mutated in place).
    torch_atomic_numbers : tuple of int
        Maps each torch element row to its physical Z value.
    config : MaceModelConfig
        Used for ``num_interactions`` and ``correlation``.
    """
    correlation = int(config.correlation)
    n_torch = len(torch_atomic_numbers)
    for k in range(int(config.num_interactions)):
        target = rep_params[f"ProductBlock_{k}"]["weight"]
        n_species, basis_dim, M = target.shape
        prefix = f"products.{k}.symmetric_contractions.contractions.0."

        # Stack native blocks degree-high to degree-low (same order mace-jax
        # uses): weights_max, weights.0, weights.1, ...
        blocks: list[np.ndarray] = []
        max_arr = state[prefix + "weights_max"]  # (E, b_max, M)
        # The corr=highest block typically has the largest "b" dim and
        # corresponds to the top-correlation order; lower-degree weights live
        # in `weights.{m}`.
        # For the scalar-only descriptor (mace-jax's projection is square 3×3),
        # we only need the per-degree scalar entries, which are typically the
        # full block for scalar irreps.
        blocks.append(max_arr)
        for m in range(correlation - 1):
            key = prefix + f"weights.{m}"
            if key in state:
                blocks.append(state[key])
        native = np.concatenate(blocks, axis=1)  # (E, b_native_total, M)

        # Take the leading `basis_dim` columns. For the scalar-only descriptor
        # this is exact (basis_dim=3, native concat dim=23+4+1=28, but only the
        # first `basis_dim` dims map directly to the cue scalar weights). The
        # remainder are higher-order-CG tensors that vanish for scalar output.
        cols = min(basis_dim, native.shape[1])
        new = np.zeros_like(target)
        for torch_idx, Z in enumerate(torch_atomic_numbers):
            if 0 <= Z < n_species:
                new[Z, :cols, :] = native[torch_idx, :cols, :]
        # Quick sanity assertion that the torch row count matches the table.
        if native.shape[0] != n_torch:
            raise ValueError(
                f"product k={k}: native weight row count {native.shape[0]} "
                f"!= torch element count {n_torch}"
            )
        rep_params[f"ProductBlock_{k}"]["weight"] = new.astype(target.dtype)


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
