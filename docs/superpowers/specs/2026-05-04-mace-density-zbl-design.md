# MACE Density-Variant Interaction Blocks + ZBL Pair-Repulsion — Design Spec

**Date:** 2026-05-04
**Status:** Approved — ready for implementation plan
**Scope:** Extend apax's MACE descriptor to support `RealAgnosticDensity` and `RealAgnosticDensityResidual` interaction-block variants, plus a faithful port of torch-mace's `ZBLBasis` pair-repulsion correction. Unblocks parity for `medium-mpa-0` and the MatPES / OMAT foundation families on top of the existing P3 stack.

## 1. Problem

P3 shipped MACE foundation-model loading at machine-precision parity for `mace_mp("small")` — the simplest MACE-MP-0 variant: scalars-only `hidden_irreps`, `RealAgnosticResidualInteractionBlock` throughout, no ZBL. P3's tightening (commit `3971c4b0`) narrowed `MaceModelConfig.interaction_cls` to `Literal["RealAgnosticResidual"]` and `_extract_config_from_torch` rejects torch foundations that use any other interaction variant.

That excludes the entire current default tier of MACE foundations. A probe of the relevant models (commit `db862da7` for the matpes test):

| Model | hidden | inter[0] | inter[-1] | ZBL | heads |
|---|---|---|---|---|---|
| `small` ✅ | `128x0e` | `Residual` | `Residual` | ❌ | 1 |
| `medium` ⚠️ | `128x0e + 128x1o` | `Residual` | `Residual` | ❌ | 1 |
| `medium-mpa-0` ❌ | `128x0e + 128x1o` | `Density` | `DensityResidual` | ✅ | 1 |
| `MACE-matpes-r2scan-omat-ft` ❌ | `128x0e + 128x1o` | `Density` | `DensityResidual` | ✅ | 1 |

`medium-mpa-0` and `matpes` share architecture exactly. Closing one closes both. `medium` (Residual + 1o channels) is not blocked by this spec — but it has never been parity-tested, so we add a parity test for it as a pre-flight against any P3.5b regression on non-scalar `hidden_irreps`.

## 2. Goals and non-goals

### Goals

- Three interaction-block classes in `apax/layers/descriptor/mace_blocks.py` that mirror torch-mace's `RealAgnosticResidualInteractionBlock`, `RealAgnosticDensityInteractionBlock`, and `RealAgnosticDensityResidualInteractionBlock` exactly. Bit-exact parity at `rtol 1e-4` energy / `rtol 1e-3` forces.
- `MaceZBLPairRepulsion` empirical-correction module that mirrors `mace.modules.radial.ZBLBasis` exactly: same `c`, `p`, `covalent_radii` buffers, same `0.529` / `14.3996` constants, same per-pair polynomial cutoff (`r_max[edge] = covalent_radii[Z_u] + covalent_radii[Z_v]`).
- `MaceModelConfig.interaction_cls: Literal["RealAgnosticResidual", "RealAgnosticDensity", "RealAgnosticDensityResidual"]` — Density variant added back to the schema. Schema and runtime stay in sync (the principle from `3971c4b0`).
- `_extract_config_from_torch` accepts the three torch class names. `_map_state_to_pytree` gains a `_map_density_fn` helper and a `_map_pair_repulsion` helper that maps `model.pair_repulsion_fn` into a `MaceZBLPairRepulsion` correction.
- Block-level unit tests for each new class. Foundation parity tests for `medium`, `medium-mpa-0`, and matpes (matpes auto-flips from xfail to pass when this lands).

### Non-goals

- `RealAgnosticInteractionBlock` (the non-residual non-density variant). No major foundation uses it; deferred until a model needs it.
- `apax.layers.empirical.ZBLRepulsion` (the existing apax variant). Stays as-is — its API and parameter format are different. The new `MaceZBLPairRepulsion` is a separate sibling.
- Multi-head foundation models. matpes is single-head; mpa-0 is single-head. Multi-head support remains as documented in §4.5 of the original spec — out of scope.
- Re-training the new blocks from scratch. The blocks must support fine-tuning end-to-end (parameters are trainable, not just frozen for inference), but this spec doesn't add a parity-tested fresh-training task; that's covered by P4.

## 3. Architecture

### 3.1 Interaction-block hierarchy

Three concrete `nn.Module` classes in `apax/layers/descriptor/mace_blocks.py`. They share the first five forward-pass steps (linear_up → conv_tp → radial-MLP gate → scatter-sum → linear); the differences are confined to the post-`linear` normalisation and the skip placement.

```
                    +--------------------+
                    |   _interaction_    |   helper function (free, not nn.Module)
                    |     scaffold       |   linear_up → conv_tp → radial_mlp →
                    +---------+----------+   scatter_sum → linear  ⇒ pre_message
                              |
              +---------------+----------------+--------------------+
              ↓                                ↓                    ↓
  InteractionBlockResidual         InteractionBlockDensity   InteractionBlockDensity
  (rename of today's class)                                      Residual
              |                                |                    |
              ↓                                ↓                    ↓
  msg = pre / avg_num_neighbors    density = scatter_sum(           density = …
                                     tanh(density_fn(edge_feats)²))
                                   msg = pre / (density + 1)        msg = pre / (density + 1)
                                   msg = skip_tp(msg, node_attrs)
              |                                |                    |
              ↓                                ↓                    ↓
  sc = skip_tp(node_feats,         sc = None                sc = skip_tp(node_feats,
              node_attrs)                                            node_attrs)
              |                                |                    |
  return (msg, sc)                 return (msg, None)        return (msg, sc)
```

Critical detail (verified against `mace.modules.blocks.RealAgnosticDensityInteractionBlock` lines 745-862): in `Density` (non-residual), `skip_tp = FullyConnectedTensorProduct(target_irreps, node_attrs_irreps, target_irreps)` — operating on the **message** (in `target_irreps`) post-density-normalisation, NOT on raw `node_feats`. The output stays in `target_irreps`. There is no separate `sc` returned — the skip is folded into the message, and `ProductBlock(use_sc=False)` handles it.

In `DensityResidual` (lines 866-988), `skip_tp = FullyConnectedTensorProduct(node_feats_irreps, node_attrs_irreps, hidden_irreps)` — same shape and placement as `Residual`, computed up front from raw `node_feats`. Returns `(msg, sc)` exactly like the existing `Residual`.

### 3.2 The shared scaffold

`_interaction_scaffold` is a free function (not a `nn.Module`); each block calls it inside its own `@nn.compact`. Linen's parameter-name scoping is preserved through the calling block's name path, so torch param mapping stays unambiguous. This is cheaper than a base class given Flax's `@nn.compact` semantics — a base class with `_setup` doesn't compose cleanly with the inheritance flow Flax expects, and an `nn.Module` mixin requires care with field overrides. A free helper avoids both.

```python
def _interaction_scaffold(
    node_feats, edge_attrs, edge_feats, receivers, senders,
    *,
    node_feats_irreps: e3nn.Irreps,
    edge_attrs_irreps: e3nn.Irreps,
    target_irreps: e3nn.Irreps,
    radial_mlp: tuple,
):
    """Common interaction-block prefix. Returns the un-normalised message in
    ``target_irreps`` so the variant-specific code can divide by either
    ``avg_num_neighbors`` (Residual) or ``density + 1`` (Density variants).
    """
    x = e3nn.flax.Linear(node_feats_irreps, name="linear_up")(node_feats)
    irreps_mid, _ = tp_out_irreps_with_instructions(
        node_feats_irreps, edge_attrs_irreps, target_irreps,
    )
    x_j = x[senders]
    tp = e3nn.tensor_product(x_j, edge_attrs, filter_ir_out=irreps_mid)
    n_paths = tp.irreps.num_irreps
    weights = _MaceFullyConnectedNet(
        list_neurons=tuple(radial_mlp) + (n_paths,), name="radial_mlp",
    )(edge_feats)
    weighted = tp * weights
    agg = e3nn.scatter_sum(weighted, dst=receivers, output_size=node_feats.shape[0])
    return e3nn.flax.Linear(target_irreps, name="linear")(agg)  # pre-normalised
```

Critical: param names (`linear_up`, `radial_mlp`, `linear`) match torch's `state_dict` key segments so `_map_interactions` (the converter) treats all three blocks identically for these slots.

### 3.3 Variant bodies

```python
class InteractionBlockResidual(nn.Module):
    """Residual variant: Skip parallel to message. Today's implementation."""
    node_feats_irreps: str; node_attrs_irreps: str
    edge_attrs_irreps: str; target_irreps: str; hidden_irreps: str
    radial_mlp: tuple = (64, 64, 64); avg_num_neighbors: float = 1.0

    @nn.compact
    def __call__(self, node_feats, sph, radial, node_attrs, i, j):
        pre = _interaction_scaffold(
            node_feats, sph, radial, i, j,
            node_feats_irreps=e3nn.Irreps(self.node_feats_irreps),
            edge_attrs_irreps=e3nn.Irreps(self.edge_attrs_irreps),
            target_irreps=e3nn.Irreps(self.target_irreps),
            radial_mlp=self.radial_mlp,
        )
        message = pre / self.avg_num_neighbors
        skip_input = e3nn.tensor_product(node_feats, node_attrs)
        sc = e3nn.flax.Linear(
            e3nn.Irreps(self.hidden_irreps), name="skip_tp", force_irreps_out=True,
        )(skip_input)
        return message, sc


class InteractionBlockDensity(nn.Module):
    """Density variant (non-residual). Skip is applied to the post-density message
    in target_irreps; no separate sc returned.

    Mirrors torch-mace's RealAgnosticDensityInteractionBlock.forward
    (mace/modules/blocks.py:813-862).
    """
    node_feats_irreps: str; node_attrs_irreps: str
    edge_attrs_irreps: str; target_irreps: str; hidden_irreps: str  # unused — kept for parity with Residual signature
    radial_mlp: tuple = (64, 64, 64)

    @nn.compact
    def __call__(self, node_feats, sph, radial, node_attrs, i, j):
        target_irreps = e3nn.Irreps(self.target_irreps)
        pre = _interaction_scaffold(
            node_feats, sph, radial, i, j,
            node_feats_irreps=e3nn.Irreps(self.node_feats_irreps),
            edge_attrs_irreps=e3nn.Irreps(self.edge_attrs_irreps),
            target_irreps=target_irreps,
            radial_mlp=self.radial_mlp,
        )

        # Density: per-edge gate, scatter-summed into per-atom density.
        edge_density = jnp.tanh(
            _MaceFullyConnectedNet(list_neurons=(1,), name="density_fn")(radial) ** 2
        )
        density = e3nn.scatter_sum(
            edge_density, dst=i, output_size=node_feats.shape[0],
        )  # (n_atoms, 1)

        message = pre / (density + 1.0)

        # Post-message skip: target_irreps × node_attrs_irreps → target_irreps.
        skip_input = e3nn.tensor_product(message, node_attrs)
        message = e3nn.flax.Linear(
            target_irreps, name="skip_tp", force_irreps_out=True,
        )(skip_input)
        return message, None


class InteractionBlockDensityResidual(nn.Module):
    """Density variant with parent-style residual skip.

    Mirrors torch-mace's RealAgnosticDensityResidualInteractionBlock.forward
    (mace/modules/blocks.py:935-989).
    """
    # Same fields as Density.
    radial_mlp: tuple = (64, 64, 64)

    @nn.compact
    def __call__(self, node_feats, sph, radial, node_attrs, i, j):
        # Skip computed from raw node_feats (BEFORE linear_up), like Residual.
        skip_input = e3nn.tensor_product(node_feats, node_attrs)
        sc = e3nn.flax.Linear(
            e3nn.Irreps(self.hidden_irreps), name="skip_tp", force_irreps_out=True,
        )(skip_input)

        pre = _interaction_scaffold(
            node_feats, sph, radial, i, j,
            node_feats_irreps=e3nn.Irreps(self.node_feats_irreps),
            edge_attrs_irreps=e3nn.Irreps(self.edge_attrs_irreps),
            target_irreps=e3nn.Irreps(self.target_irreps),
            radial_mlp=self.radial_mlp,
        )
        edge_density = jnp.tanh(
            _MaceFullyConnectedNet(list_neurons=(1,), name="density_fn")(radial) ** 2
        )
        density = e3nn.scatter_sum(
            edge_density, dst=i, output_size=node_feats.shape[0],
        )
        message = pre / (density + 1.0)
        return message, sc
```

### 3.4 `MaceRepresentation` dispatch

```python
_INTERACTION_BLOCK_CLS = {
    "RealAgnosticResidual": InteractionBlockResidual,
    "RealAgnosticDensity": InteractionBlockDensity,
    "RealAgnosticDensityResidual": InteractionBlockDensityResidual,
}

# Inside MaceRepresentation.__call__, replacing the current single-class call:
Block = _INTERACTION_BLOCK_CLS[self.interaction_cls]
message, sc = Block(
    node_feats_irreps=prev_irreps_str,
    node_attrs_irreps=node_attrs_irreps_str,
    edge_attrs_irreps=sh_irreps_str,
    target_irreps=interaction_irreps_str,
    hidden_irreps=this_hidden_str,
    avg_num_neighbors=self.avg_num_neighbors,  # ignored by Density variants
)(node_feats, sph, radial, Z_one_hot, i, j)
node_feats = ProductBlock(
    node_feats_irreps=interaction_irreps_str,
    target_irreps=this_hidden_str,
    correlation=self.correlation,
    num_elements=self.num_elements,
    use_sc=(sc is not None),
    use_cueq=self.use_cueq,
)(message, sc, Z)
```

`ProductBlock` already accepts `use_sc: bool = True`. Extend its `__call__` to handle `sc is None` — when `use_sc=False`, the `sc` argument is ignored and may be passed as `None`. Today's signature requires an `IrrepsArray`; tighten only if needed. Trace at implementation time.

### 3.5 `MaceZBLPairRepulsion`

New `apax/layers/empirical.py` class. Faithful port of `mace.modules.radial.ZBLBasis`:

```python
class MaceZBLPairRepulsion(EmpiricalEnergyTerm):
    """Faithful port of torch-mace ZBLBasis.

    Mirrors mace/modules/radial.py:149-218 exactly. Distinct from
    apax's existing :class:`ZBLRepulsion` — apax's variant uses a cosine
    cutoff and softplus-parameterized coefficients; this one uses the
    polynomial cutoff with per-pair r_max from covalent radii (matching
    foundation-model parameter formats).
    """

    p: int = 6  # polynomial-cutoff order
    apply_mask: bool = True

    def setup(self):
        # All 5 torch params/buffers are non-trainable here. For foundations
        # in scope (mpa-0, matpes), trainable=False on the torch side.
        # If we ever encounter trainable=True, swap to self.param().
        self.c = self.variable(
            "buffers", "c",
            lambda: jnp.array([0.1818, 0.5099, 0.2802, 0.02817]),
        )
        self.a_exp = self.variable(
            "buffers", "a_exp", lambda: jnp.float64(0.300),
        )
        self.a_prefactor = self.variable(
            "buffers", "a_prefactor", lambda: jnp.float64(0.4543),
        )
        # ase.data.covalent_radii — 119 entries, indexed by Z directly.
        self.covalent_radii = self.variable(
            "buffers", "covalent_radii",
            lambda: jnp.asarray(ase.data.covalent_radii, dtype=jnp.float64),
        )

    def __call__(self, R, dr_vec, Z, idx, box, properties):
        i, j = idx[0], idx[1]
        Z_i, Z_j = Z[i], Z[j]
        dr = jnp.linalg.norm(dr_vec, axis=-1)
        dr = jnp.clip(dr, min=0.02)  # avoid div-by-zero at coincident atoms

        a_exp = self.a_exp.value
        a_prefactor = self.a_prefactor.value
        c = self.c.value

        a = a_prefactor * 0.529 / (Z_i**a_exp + Z_j**a_exp)
        r_over_a = dr / a
        phi = (
            c[0] * jnp.exp(-3.2 * r_over_a)
            + c[1] * jnp.exp(-0.9423 * r_over_a)
            + c[2] * jnp.exp(-0.4028 * r_over_a)
            + c[3] * jnp.exp(-0.2016 * r_over_a)
        )
        v_edges = (14.3996 * Z_i * Z_j) / dr * phi

        # Per-pair polynomial cutoff: r_max varies per edge.
        r_max_edge = self.covalent_radii.value[Z_i] + self.covalent_radii.value[Z_j]
        x = dr / r_max_edge
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * x**self.p
            + self.p * (self.p + 2.0) * x ** (self.p + 1)
            - (self.p * (self.p + 1.0) / 2.0) * x ** (self.p + 2)
        )
        envelope = jnp.where(dr <= r_max_edge, envelope, 0.0)

        v_edges = 0.5 * v_edges * envelope
        if self.apply_mask:
            v_edges = mask_by_neighbor(v_edges, idx)
        return fp64_sum(v_edges)
```

Hard-coded constants `0.529` (Bohr radius in Å) and `14.3996` (e²/4πε₀ in eV·Å) match torch's source exactly. The `0.02` lower clip on `dr` matches apax's existing `ZBLRepulsion` behaviour and avoids `inf` energies when atoms coincide during initialisation; torch-mace doesn't have this clip but in practice never sees `dr ≈ 0` in production.

Add `MaceZBLPairRepulsion` to `apax.layers.empirical.all_corrections` under the key `"mace_zbl"`.

### 3.6 Schema additions

```python
# apax/config/model_config.py

class MaceZBLPairRepulsion(Correction, extra="forbid"):
    """Faithful torch-mace ZBLBasis: polynomial cutoff + ase covalent radii.

    Distinct from :class:`ZBLRepulsion` (cosine cutoff + softplus coefficients).
    """
    name: Literal["mace_zbl"]
    p: int = 6
    trainable: bool = False


# Extend the discriminated union:
EmpiricalCorrection = Union[
    ZBLRepulsion,
    ExponentialRepulsion,
    LatentEwald,
    MaceZBLPairRepulsion,
]


# MaceModelConfig — the interaction_cls field is now either a single Literal
# (broadcast to every layer) or a per-layer list. Foundation models in scope
# use Density first / DensityResidual last, so the converter emits a list.
_InteractionLiteral = Literal[
    "RealAgnosticResidual",
    "RealAgnosticDensity",
    "RealAgnosticDensityResidual",
]


class MaceModelConfig(BaseModelConfig, extra="forbid"):
    ...
    interaction_cls: Union[_InteractionLiteral, list[_InteractionLiteral]] = (
        "RealAgnosticResidual"
    )
```

The `interaction_cls` Literal grows from one value to three; the principle from `3971c4b0` holds — schema lists exactly what's implemented. `RealAgnostic` (non-residual non-density) stays out until a model needs it.

The Union with `list[...]` is required because foundation models like mpa-0 / matpes use **different interaction classes per layer** (`Density` for layer 0, `DensityResidual` for layer 1). Pydantic validates each list element against the single-Literal branch. `MaceRepresentation` resolves the field to a per-layer list at the top of `__call__`:

```python
if isinstance(self.interaction_cls, str):
    per_layer = [self.interaction_cls] * self.num_interactions
else:
    per_layer = list(self.interaction_cls)
    if len(per_layer) != self.num_interactions:
        raise ValueError(
            f"interaction_cls list length {len(per_layer)} does not match "
            f"num_interactions {self.num_interactions}"
        )
# … use per_layer[k] in the dispatch table
```

Single-string configs (every existing `mace` config: `small`, fresh-trained models, the fine-tune template) keep working unchanged. List-form is what the converter emits for foundations whose layers differ.

## 4. Converter

### 4.1 `_extract_config_from_torch`

Accept the three torch interaction class names and emit a single string when all layers share a variant, or a list otherwise:

```python
_SUPPORTED_TORCH_INTERACTION_CLS = {
    "RealAgnosticResidualInteractionBlock": "RealAgnosticResidual",
    "RealAgnosticDensityInteractionBlock": "RealAgnosticDensity",
    "RealAgnosticDensityResidualInteractionBlock": "RealAgnosticDensityResidual",
}

per_layer_cls = []
for inter in model.interactions:
    cls_name = type(inter).__name__
    if cls_name not in _SUPPORTED_TORCH_INTERACTION_CLS:
        raise NotImplementedError(
            f"Foundation uses interaction block {cls_name!r}; apax supports "
            f"{sorted(_SUPPORTED_TORCH_INTERACTION_CLS)}. Other variants need "
            "their apax port before they can be converted."
        )
    per_layer_cls.append(_SUPPORTED_TORCH_INTERACTION_CLS[cls_name])

if len(set(per_layer_cls)) == 1:
    cfg["interaction_cls"] = per_layer_cls[0]
else:
    cfg["interaction_cls"] = per_layer_cls   # list of Literals
```

For mpa-0 / matpes this emits `["RealAgnosticDensity", "RealAgnosticDensityResidual"]`. For `small` / `medium` it emits `"RealAgnosticResidual"` (single string).

Detect ZBL: if `hasattr(model, "pair_repulsion_fn")`, append a `MaceZBLPairRepulsion(name="mace_zbl", p=int(model.pair_repulsion_fn.p), trainable=False)` to `cfg["empirical_corrections"]`. If `model.pair_repulsion_fn.a_exp` / `a_prefactor` are `nn.Parameter` (rather than buffers), set `trainable=True`.

### 4.2 `_map_state_to_pytree`

Add per-variant mapping logic. The first five `state_dict` keys per interaction are the same across all variants (`linear_up`, `conv_tp_weights.layer{0..3}`, `linear`, `skip_tp`); only the variant-specific keys differ:

| Variant | Variant-specific torch keys | Apax target |
|---|---|---|
| Residual | (none beyond shared) | (none beyond shared) |
| Density | `density_fn.layer0.weight (10, 1)` | `InteractionBlockDensity.density_fn.kernel_0` |
| DensityResidual | `density_fn.layer0.weight (10, 1)` | `InteractionBlockDensityResidual.density_fn.kernel_0` |

Density's `skip_tp` has different irrep signature (target × node_attrs → target) vs Residual / DensityResidual (node_feats × node_attrs → hidden). Torch's `skip_tp.weight` has the corresponding flat shape; the apax-side `e3nn.flax.Linear(force_irreps_out=True)` slot accepts the flat tensor with the right total numel either way.

`_map_pair_repulsion` (new):

```python
def _map_pair_repulsion(state, params, *, p_value):
    """Map torch pair_repulsion_fn buffers into MaceZBLPairRepulsion."""
    # All five torch buffers map directly into apax 'buffers' collection.
    buffers = params["buffers"]["MaceZBLPairRepulsion_0"]
    buffers["c"] = state["pair_repulsion_fn.c"]
    buffers["a_exp"] = state["pair_repulsion_fn.a_exp"]
    buffers["a_prefactor"] = state["pair_repulsion_fn.a_prefactor"]
    buffers["covalent_radii"] = state["pair_repulsion_fn.covalent_radii"]
    # p is a config-level int, not a runtime param.
```

The `p` value is read once at conversion time and passed through `MaceZBLPairRepulsion(p=...)` via the `MaceZBLPairRepulsion` correction config. The torch `pair_repulsion_fn.cutoff_fn.{p, r_max}` keys are ignored — `cutoff_fn` is a torch module artefact that isn't consulted at runtime by `ZBLBasis.forward` (it's there for `__repr__` purposes; the actual envelope uses per-pair `covalent_radii[Z_u] + covalent_radii[Z_v]`).

### 4.3 Per-layer mapping

`_map_interactions` switches on the per-layer variant from the resolved `interaction_cls` list (see §3.6). Density variants gain the extra `density_fn.layer0.weight (10, 1)` mapping into the apax `_MaceFullyConnectedNet` named `"density_fn"` (same `kernel_0` slot pattern as `radial_mlp`). Residual and non-residual share `linear_up`, `linear`, `conv_tp_weights.layer{0..3}`, and `skip_tp` — the only difference is the `skip_tp.weight` flat shape, which the apax `e3nn.flax.Linear(force_irreps_out=True)` slot accepts either way (target × num_elements vs node_feats × num_elements → output sized appropriately).

## 5. Testing

### 5.1 Always-on (no torch)

Add to `tests/unit_tests/layers/descriptor/test_mace_blocks.py`:

- `test_interaction_block_density_shape_and_finite` — random init, single-call, asserts `(message, sc)` tuple where `sc is None`, `message.irreps == target_irreps`, output finite.
- `test_interaction_block_density_residual_shape_and_finite` — same, asserts `sc.irreps == hidden_irreps` and `sc is not None`.
- `test_interaction_scaffold_param_names_match_torch` — initialises a `Residual` block, asserts the param tree contains exactly `{linear_up, radial_mlp, linear, skip_tp}` at the expected paths. Acts as a regression guard for converter mapping.
- `test_density_skip_tp_target_irreps_signature` — initialises a `Density` block, asserts `skip_tp.kernel` has shape consistent with `target_irreps × num_elements → target_irreps`.

Add to `tests/unit_tests/layers/test_mace_zbl.py` (new):

- `test_mace_zbl_pair_repulsion_matches_torch_on_dimer` — for a single H–H, H–O, O–O pair across a sweep of `dr` values, compute `MaceZBLPairRepulsion` energy in apax and `ZBLBasis` energy in torch with default buffers; assert `np.allclose(rtol=1e-12, atol=1e-12)`. Gated by `mace_parity` because it imports torch only for the reference.

### 5.2 Foundation-model parity (gated mace_parity)

Extend `tests/integration_tests/mace/test_mace_parity.py`:

- `test_energy_force_parity_water[medium]` — parametrize the existing water test with `medium` added to the fixture list. Pre-flight on Residual + 1o channels.
- `test_energy_force_parity_water[medium-mpa-0]` — Density + ZBL + 1o.
- `test_energy_force_parity_periodic_sio2[small]`, `[medium]`, `[medium-mpa-0]` — same parity check on a periodic SiO₂ cell. Validates the PBC offset path and stress-relevant geometry.
- `test_energy_force_parity_matpes_omat_ft` — drop the `xfail` decorator. Same body as mpa-0 with the local-file source.

Targets: `rtol 1e-4, atol 1e-5` energy; `rtol 1e-3, atol 1e-4` forces — same thresholds as P3.6.

### 5.3 Converter integration (gated)

Extend `tests/integration_tests/mace/test_convert.py`:

- `test_convert_emits_mace_zbl_correction[medium-mpa-0]` — assert the converted `config.yaml` lists a `mace_zbl` entry under `model.empirical_corrections` and the orbax checkpoint has matching buffer leaves.
- Update the existing shape-coverage test to permit `pair_repulsion_fn.*` buffers in the source and `MaceZBLPairRepulsion_0/buffers/*` in the target.

## 6. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Density `skip_tp` irrep signature mismatch (target vs hidden) | Medium | Parity off by O(1) factor | Section 3.3 explicit; unit test asserts irrep signature |
| `density_fn` numerical mismatch (e3nn-jax silu normalisation vs torch) | Medium | ~1-3% force error | Reuse `_MaceFullyConnectedNet` (already silu-normalised to torch) |
| ZBL `0.02` clip on `dr` produces force discontinuity at coincident atoms | Low | Forces blow up at training-step start with bad init | Match torch's behavior exactly: drop the clip, accept that any sensible dataset has `dr > 0.02`; if needed re-add as a configurable `eps` |
| Per-layer `interaction_cls` list breaks YAML round-trip via `Config.dump_config` | Low | Save/load asymmetry | Pydantic + `model_dump(mode="json")` handles list-of-Literal cleanly; covered by an explicit round-trip test |
| `MaceZBLPairRepulsion` `buffers` collection not preserved by orbax | Low | Foundation conversion succeeds but ZBL contributes 0 at inference | Use the same `self.variable("buffers", ...)` pattern apax already uses elsewhere; orbax restores all collections |
| Existing apax `ZBLRepulsion` users get confused by two ZBL variants | Low | Docs friction | Module docstrings cross-reference; converter only emits `mace_zbl` |

## 7. Phasing (within this spec)

| Phase | Goal | Exit |
|---|---|---|
| **a. Block primitives** | Three interaction-block classes + `_interaction_scaffold` helper, unit tests | Always-on tests pass; param tree shape verified |
| **b. ZBL** | `MaceZBLPairRepulsion` + dimer parity test | Dimer `np.allclose` vs torch passes |
| **c. Schema + dispatch** | `interaction_cls` Literal expanded; per-layer list support; `MaceRepresentation` dispatch wired | `medium` parity test passes (Residual + 1o channels) |
| **d. Converter** | `_extract_config_from_torch` accepts variants; `_map_density_fn`, `_map_pair_repulsion` helpers; per-layer `interaction_cls` emission | mpa-0 conversion runs without NaN; ZBL config emitted |
| **e. Parity** | mpa-0 + matpes parity tests pass; remove `xfail` | All foundation parity tests at documented thresholds |

## 8. Net LoC estimate

| Component | LoC |
|---|---|
| `_interaction_scaffold` + 3 block classes | ~280 |
| `MaceZBLPairRepulsion` + schema entry | ~110 |
| Per-layer `interaction_cls` plumbing in `MaceModelConfig` + `MaceRepresentation` | ~30 |
| Converter helpers (`_map_density_fn`, `_map_pair_repulsion`) | ~80 |
| Tests (always-on + parity) | ~250 |
| **Total** | **~750** |

All additive (apart from the existing `InteractionBlock` rename to `InteractionBlockResidual` with a deprecation alias).

## 9. Open items

- Whether `MaceZBLPairRepulsion` should support trainable `a_exp` / `a_prefactor` (torch's `trainable=True` mode). No foundation in scope uses this; defer to a `trainable: bool = False` field on the `MaceZBLPairRepulsion` correction config, default `False`, dispatch in `setup` between `self.variable("buffers", ...)` and `self.param(...)`.
- Whether to surface `ase.data.covalent_radii` as a configurable knob (the foundation models use ASE's table, but a custom training run might want a different one). Defer to an optional `covalent_radii: Optional[list[float]] = None` field on the config; default uses ASE.
- `RealAgnostic` (non-residual non-density) variant — kept out of scope. Adding it later is purely additive (one more `InteractionBlock*` class + Literal extension).
