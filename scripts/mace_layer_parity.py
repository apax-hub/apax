"""Layer-by-layer parity diff: apax-converted MACE vs reference torch-mace.

Loads a single ``ase.Atoms`` (default: ``s22[20]`` indole-benzene T-shape —
the worst-case parity offender from ``tmp/main.py``). Runs both forward
passes with intermediate capture: torch via ``register_forward_hook``,
apax via Flax ``sow``. Diffs each named boundary and writes a parity
report to stdout and to ``tmp/mace_parity_report.json``.

The apax sow instrumentation lives in:

* :mod:`apax.layers.descriptor.basis_functions` — ``radial_embedding``
* :mod:`apax.layers.descriptor.mace_blocks` — ``interactions[k].linear_up``,
  ``interactions[k].conv_tp``, ``interactions[k].linear``,
  ``interactions[k].skip_tp`` and ``products[k]``
* :mod:`apax.layers.readout` — ``readouts[k]``
* :mod:`apax.layers.empirical` — ``pair_repulsion``
* :mod:`apax.nn.models` — ``scale_shift_out``

Sown intermediates are no-ops in production paths (Flax skips them when
the ``debug`` collection is not in ``mutable``).

Usage
-----
``uv run python scripts/mace_layer_parity.py [--system-idx 20]``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Torch hook outputs land here while a forward pass is running.  Cleared at
# the end of :func:`run_torch` so a second invocation starts clean.
_TORCH_CAPS: dict[str, np.ndarray] = {}


def _torch_hook(name: str, *, tuple_index: int | None = None):
    """Build a forward-hook callback that stashes the module output by name.

    Parameters
    ----------
    name : str
        Slot name under which the output should be stored in
        :data:`_TORCH_CAPS`.
    tuple_index : int or None, optional
        If the wrapped module returns a ``(tensor, ...)`` tuple, capture
        only this element under ``name``. Defaults to ``None`` which
        captures every tensor element under ``name[i]``.

    Returns
    -------
    callable
        A function suitable for :meth:`torch.nn.Module.register_forward_hook`.
    """

    def _hook(module, inputs, output):  # noqa: ARG001 — pytorch hook signature
        import torch as _torch

        if isinstance(output, _torch.Tensor):
            _TORCH_CAPS[name] = output.detach().cpu().double().numpy()
        elif isinstance(output, (tuple, list)):
            if tuple_index is not None:
                o = output[tuple_index]
                if isinstance(o, _torch.Tensor):
                    _TORCH_CAPS[name] = o.detach().cpu().double().numpy()
                return
            for i, o in enumerate(output):
                if isinstance(o, _torch.Tensor):
                    _TORCH_CAPS[f"{name}[{i}]"] = (
                        o.detach().cpu().double().numpy()
                    )

    return _hook


def run_torch(model_path, atoms):
    """Run torch-mace and capture per-block outputs via forward hooks.

    Parameters
    ----------
    model_path : os.PathLike
        Path to the torch ``.model`` checkpoint.
    atoms : ase.Atoms
        Structure to evaluate; will be copied so the input is untouched.

    Returns
    -------
    energy : float
        Total potential energy in eV.
    forces : np.ndarray
        Per-atom forces, shape ``(n_atoms, 3)``.
    captures : dict of str to np.ndarray
        Per-block outputs keyed by the same names used by the apax sow
        instrumentation.
    """
    from mace.calculators.mace import MACECalculator

    calc = MACECalculator(
        model_paths=str(model_path), default_dtype="float64", device="cpu",
    )
    model = calc.models[0]

    handles = [
        # Torch's ``RadialEmbeddingBlock.forward`` returns the tuple
        # ``(radial * cutoff, sph_or_None)``; we only diff the first
        # element against apax's sown ``radial_embedding`` (which is
        # exactly that product).
        model.radial_embedding.register_forward_hook(
            _torch_hook("radial_embedding", tuple_index=0)
        ),
    ]
    for i, inter in enumerate(model.interactions):
        for slot in ("linear_up", "conv_tp", "linear", "skip_tp"):
            sub = getattr(inter, slot, None)
            if sub is not None:
                handles.append(
                    sub.register_forward_hook(
                        _torch_hook(f"interactions[{i}].{slot}")
                    )
                )
    for i, prod in enumerate(model.products):
        handles.append(
            prod.register_forward_hook(_torch_hook(f"products[{i}]"))
        )
    for i, ro in enumerate(model.readouts):
        handles.append(
            ro.register_forward_hook(_torch_hook(f"readouts[{i}]"))
        )
    if hasattr(model, "scale_shift"):
        handles.append(
            model.scale_shift.register_forward_hook(
                _torch_hook("scale_shift_out")
            )
        )
    if hasattr(model, "pair_repulsion_fn"):
        # Torch's ``ZBLBasis`` returns the per-edge contribution; apax
        # sums + scales internally and sows the scalar total.  Sum the
        # torch output here so the slot diffs a like quantity.
        def _zbl_hook(module, inputs, output):  # noqa: ARG001 — pytorch hook
            import torch as _torch

            if isinstance(output, _torch.Tensor):
                # Apply the same global ``scale`` that torch's
                # ``ScaleShiftBlock`` would apply downstream — apax folds
                # it into ``MaceZBLPairRepulsion.output_scale``.
                scale = float(model.scale_shift.scale.detach().cpu())
                _TORCH_CAPS["pair_repulsion"] = (
                    output.detach().cpu().double().sum().numpy() * scale
                )

        handles.append(model.pair_repulsion_fn.register_forward_hook(_zbl_hook))

    a = atoms.copy()
    a.calc = calc
    energy = float(a.get_potential_energy())
    forces = np.asarray(a.get_forces())

    caps = dict(_TORCH_CAPS)
    _TORCH_CAPS.clear()
    for h in handles:
        h.remove()
    return energy, forces, caps


def run_apax(apax_dir, atoms):
    """Run apax forward and capture sown intermediates from the ``debug`` collection.

    The :class:`apax.md.ase_calc.ASECalculator` does not retain enough
    state across calls to re-run with a different ``mutable`` setting, so
    we build a fresh non-derivative :class:`~apax.nn.models.EnergyModel`
    here, mirroring the inputs that the calculator's normal pathway
    would produce, and call ``apply(..., mutable=['debug'])`` directly.

    Parameters
    ----------
    apax_dir : os.PathLike
        Path to the apax model directory (e.g.
        ``tmp/mace-mpa-0-medium``).
    atoms : ase.Atoms
        Structure to evaluate.

    Returns
    -------
    energy : float
        Total potential energy in eV.
    forces : np.ndarray
        Per-atom forces, shape ``(n_atoms, 3)``.
    captures : dict of str to np.ndarray
        Per-block outputs from the sown ``debug`` collection.
    """
    import jax
    import jax.numpy as jnp
    import numpy as _np
    from flax.core.frozen_dict import freeze, unfreeze
    from vesin import NeighborList

    from apax.md.ase_calc import ASECalculator

    calc = ASECalculator(apax_dir)
    a = atoms.copy()
    a.calc = calc
    # Trigger initialisation + a normal forward (energy + forces).
    energy = float(a.get_potential_energy())
    forces = _np.asarray(a.get_forces())

    # Now build a parallel non-derivative ``EnergyModel`` so we can drive
    # ``apply`` with ``mutable=['debug']`` and harvest sow outputs.  Mirrors
    # ``ASECalculator.initialize`` minus the gradient/stress/hessian wrap.
    config = calc.model_config
    Builder = config.model.get_builder()
    builder = Builder(config.model.model_dump(), n_species=119)
    box = jnp.asarray(atoms.cell.array, dtype=jnp.float64).T
    energy_model = builder.build_energy_model(
        apply_mask=True,
        init_box=_np.array(box),
        inference_disp_fn=None,
    )

    # Reuse the neighbour list that the calculator already built during
    # the forward pass above: this preserves whatever masking convention
    # apax's normal pipeline uses (jax-md self-pairs + padding for
    # gas-phase, vesin full-list otherwise).  Without this we'd get
    # different edge counts than apax's production path.
    r_max = float(config.model.basis.r_max)
    if calc.neigbor_from_jax:
        # ``self.neighbors`` is a jax-md NeighborList with ``.idx``
        # (shape ``(2, n_edges)``) including self-pair padding.  apax
        # masks self-pairs at the edge-feature level so we keep them.
        idx_jax = jnp.asarray(calc.neighbors.idx, dtype=jnp.int32)
        offsets_jax = jnp.zeros((idx_jax.shape[1], 3), dtype=jnp.float64)
    else:
        idx_jax = jnp.asarray(calc.neighbors, dtype=jnp.int32)
        offsets_jax = jnp.asarray(calc.offsets, dtype=jnp.float64)

    positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
    Z = jnp.asarray(atoms.numbers, dtype=jnp.int32)
    box_jax = jnp.asarray(atoms.cell.array, dtype=jnp.float64).T

    # For periodic systems the EnergyModel internally transforms
    # positions to fractional coordinates; the calculator's normal
    # ``step_fn`` does this transform up front.  Mirror it here.
    if _np.any(atoms.cell.array > 1e-6):
        from jax_md import space  # local import — only needed periodic.

        inv_box = jnp.linalg.inv(box_jax)
        positions = space.transform(inv_box, positions)

    # Unwrap the ``EnergyDerivativeModel`` layer in ``calc.params``: the
    # serialised tree is ``{params: {energy_model: ...}, buffers:
    # {energy_model: ...}}``.  ``canonicalize_energy_model_parameters``
    # would do this but it drops the ``buffers`` collection along the way,
    # which we need for ``AgnesiTransform`` / ``MaceZBLPairRepulsion``.
    raw = unfreeze(calc.params)
    inner: dict = {}
    for col in ("params", "buffers"):
        if col in raw and "energy_model" in raw[col]:
            inner[col] = raw[col]["energy_model"]
        elif col in raw:
            inner[col] = raw[col]
    params = freeze(inner)

    (energy_apax, _props), sown = energy_model.apply(
        params,
        positions,
        Z,
        idx_jax,
        box_jax,
        offsets_jax,
        mutable=["debug"],
    )

    # apax's jax-md NL pads with self-pairs (i, i) that downstream
    # masking zeroes out — torch's NL skips them entirely.  Stash a mask
    # of "real" edges so the diff table can drop padding rows from
    # per-edge captures (radial_embedding, conv_tp).
    edge_mask = _np.asarray(idx_jax[0] != idx_jax[1])
    debug = sown.get("debug", {})

    flat: dict[str, np.ndarray] = {}

    def _materialize(val):
        # Flax ``sow`` stores values as a tuple per call site (default
        # ``init_fn=tuple``, ``reduce_fn=lambda xs, x: xs + (x,)``).  For
        # our single-call sites the tuple has length 1.
        if isinstance(val, tuple) and len(val) == 1:
            val = val[0]
        # Sown values written from inside ``jax.vmap`` (e.g. the
        # ``readouts[k]`` slot, since ``EnergyModel`` calls
        # ``jax.vmap(self.readout)``) leak out as ``BatchTracer`` objects;
        # ``BatchTracer.val`` is the underlying batched array.  Using
        # :func:`flax.linen.vmap` instead would lift sown collections
        # automatically, but switching the production path is out of
        # scope for this harness.
        if hasattr(val, "val") and hasattr(val, "batch_dim"):
            return _np.asarray(val.val)
        return _np.asarray(val)

    def _walk(prefix, node):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(f"{prefix}/{k}" if prefix else k, v)
            return
        flat[prefix] = _materialize(node)

    _walk("", debug)

    # Drop self-pair padding rows from per-edge captures so the leading
    # edge-count axis aligns with the torch NL.
    n_edges_apax = int(edge_mask.shape[0])
    for raw, val in list(flat.items()):
        if val.ndim >= 1 and val.shape[0] == n_edges_apax:
            flat[raw] = val[edge_mask]

    # Strip the ``InteractionBlock_{k}`` / ``ProductBlock_{k}`` /
    # ``readout_{k}`` Linen module prefixes so the names match the torch
    # hook keys exactly.
    aliased: dict[str, np.ndarray] = {}
    for raw, val in flat.items():
        # Normalise the trailing slot, e.g.
        # "representation/InteractionBlock_0/interactions[0].linear_up" ->
        # "interactions[0].linear_up"; "representation/ProductBlock_0/products[0]"
        # -> "products[0]"; "readout/readouts[0]" -> "readouts[0]";
        # "representation/radial_embedding/radial_embedding" ->
        # "radial_embedding"; "scale_shift_out" -> "scale_shift_out".
        slot = raw.split("/")[-1]
        aliased[slot] = val
    return float(energy_apax), forces, aliased


def _edge_filter(arr_torch, arr_apax):
    """Best-effort edge-count alignment for per-edge captures.

    apax's jax-md neighbour list pads with self-pairs ``(i, i)`` whose
    feature vectors are masked to zero downstream; torch's NL skips them.
    When the leading dim disagrees and the apax extra rows are exactly
    zero (padding), drop them so the diff is meaningful.

    Parameters
    ----------
    arr_torch, arr_apax : np.ndarray
        Per-edge feature tensors.  Aligned on the trailing axes already;
        only the leading edge-count axis may differ.

    Returns
    -------
    arr_torch, arr_apax : np.ndarray or None
        Aligned tensors or ``(None, None)`` when alignment fails.
    """
    if arr_torch.shape == arr_apax.shape:
        return arr_torch, arr_apax
    if arr_torch.ndim == 0 or arr_apax.ndim == 0:
        return None, None
    if arr_torch.shape[1:] != arr_apax.shape[1:]:
        return None, None
    n_t = arr_torch.shape[0]
    n_a = arr_apax.shape[0]
    if n_a > n_t:
        # Drop trailing all-zero rows from apax (jax-md self-pair pad).
        nonzero_mask = np.any(arr_apax != 0, axis=tuple(range(1, arr_apax.ndim)))
        kept = arr_apax[nonzero_mask]
        if kept.shape[0] == n_t:
            return arr_torch, kept
    return None, None


def diff_table(torch_caps, apax_caps):
    """Build a per-block diff row list from two capture dicts.

    Parameters
    ----------
    torch_caps, apax_caps : dict of str to np.ndarray
        Captures keyed by the same slot names.  Missing entries on either
        side are reported as ``"missing torch"`` / ``"missing apax"``.

    Returns
    -------
    list of tuple
        Each tuple is ``(name, status, max_abs, mean_abs)``.  When the
        shapes don't match the status reads ``"shape t=... a=..."`` and
        the diff metrics are ``nan``.
    """
    rows = []
    keys = sorted(set(torch_caps) | set(apax_caps))
    for k in keys:
        if k not in torch_caps:
            rows.append((k, "missing torch", float("nan"), float("nan")))
            continue
        if k not in apax_caps:
            rows.append((k, "missing apax", float("nan"), float("nan")))
            continue
        t_arr = np.asarray(torch_caps[k])
        a_arr = np.asarray(apax_caps[k])
        status = "ok"
        # Accept differences in trailing singleton axes (e.g. torch's
        # readouts return ``(n_atoms, 1)`` while apax's vmapped readout
        # collapses to ``(n_atoms, n_out)`` with ``n_out=1``).
        if t_arr.shape != a_arr.shape:
            if t_arr.squeeze().shape == a_arr.squeeze().shape:
                t_arr = t_arr.squeeze()
                a_arr = a_arr.squeeze()
            else:
                t_aligned, a_aligned = _edge_filter(t_arr, a_arr)
                if t_aligned is None:
                    rows.append(
                        (
                            k,
                            f"shape t={t_arr.shape} a={a_arr.shape}",
                            float("nan"),
                            float("nan"),
                        )
                    )
                    continue
                t_arr, a_arr = t_aligned, a_aligned
                status = "ok (aligned)"
        d = np.abs(a_arr.ravel() - t_arr.ravel())
        rows.append((k, status, float(d.max()), float(d.mean())))
    return rows


def main():
    """Run the layer-by-layer parity harness."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system-idx", type=int, default=20)
    p.add_argument(
        "--apax-dir", type=Path, default=Path("tmp/mace-mpa-0-medium")
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path("tmp/mace-mpa-0-medium.model"),
    )
    p.add_argument(
        "--out", type=Path, default=Path("tmp/mace_parity_report.json")
    )
    args = p.parse_args()

    from ase.collections import s22

    atoms = list(s22)[args.system_idx]
    print(
        f"System idx={args.system_idx}  N_atoms={len(atoms)}  "
        f"formula={atoms.get_chemical_formula()}"
    )

    e_t, _, caps_t = run_torch(args.model_path, atoms)
    e_a, _, caps_a = run_apax(args.apax_dir, atoms)

    print(f"E_torch = {e_t:.10f} eV")
    print(f"E_apax  = {e_a:.10f} eV")
    print(f"diff    = {e_a - e_t:+.3e} eV")
    print()

    rows = diff_table(caps_t, caps_a)
    header = (
        f"{'block':<48} {'status':<22} {'max_abs':>14} {'mean_abs':>14}"
    )
    print(header)
    print("-" * len(header))
    for name, status, mx, mn in rows:
        if status == "ok":
            print(
                f"{name:<48} {status:<22} {mx:>14.3e} {mn:>14.3e}"
            )
        else:
            print(
                f"{name:<48} {status:<22} {'nan':>14} {'nan':>14}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "system_idx": args.system_idx,
                "energy_torch": e_t,
                "energy_apax": e_a,
                "energy_diff": e_a - e_t,
                "rows": [
                    {
                        "name": n,
                        "status": s,
                        "max_abs": mx,
                        "mean_abs": mn,
                    }
                    for n, s, mx, mn in rows
                ],
            },
            indent=2,
        )
    )
    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
