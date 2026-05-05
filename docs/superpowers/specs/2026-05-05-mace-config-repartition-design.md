# MACE config re-partition + descriptor refactor

**Branch:** `feat/mace-foundation-integration`
**Status:** design (not yet implemented)
**Date:** 2026-05-05

## Motivation

`MaceModelConfig` was ported from torch-mace's flat hyperparameter naming and now holds 13 sibling fields. Three of them — `r_max`, `num_bessel`, the optional `distance_transform` — duplicate or shadow data that lives elsewhere:

- `model.r_max` and `model.basis.r_max` are written from the same source (`apax convert-mace`) and must stay in sync. The converter mirrors them and asserts equality as a tripwire (`apax/transfer_learning/mace_foundation.py:404-409`). All neighbour-list builders in `apax/md/`, `apax/bal/`, `apax/train/` read **only** `config.model.basis.r_max`; the MACE descriptor reads **only** `model.r_max`.
- `model.num_bessel` mirrors `model.basis.n_basis` for the same reason.

`MaceModelConfig` also breaks the structural pattern used by every other apax model. `GMNNConfig`, `EquivMPConfig`, `So3kratesConfig` keep architecture knobs flat alongside an inherited nested `basis: BasisConfig` and consume the basis as a **pre-built submodule** built by `ModelBuilder.build_basis_function()`. `MaceRepresentation` instead bakes the basis hyperparameters in as Linen-attribute scalars and constructs the radial embedding internally, which is what forced the duplication in the first place.

Lastly, `interaction_cls` carries the Python anti-pattern `_cls` suffix and is typed as an awkward `Union[Literal, list[Literal], tuple[Literal, ...]]` with a runtime length check against a separate `num_interactions` integer — a second drift class.

The branch has not merged and **no released MACE configs exist in the wild**, so the schema can be broken freely.

## Goals

1. Eliminate the `r_max` / `num_bessel` duplication. `model.basis` becomes the single source of truth.
2. Group MACE-specific fields into nested sub-configs that mirror the forward pass: `basis → radial_embedding → descriptor → readout`.
3. Replace `interaction_cls: Union[Literal, list[...], tuple[...]]` with `descriptor.interactions: list[InteractionConfig]` using a discriminated pydantic union per variant.
4. Drop `num_interactions`; use `len(descriptor.interactions)` instead.
5. Refactor `MaceRepresentation` to accept a pre-built `radial_embedding` Linen submodule, structurally matching how GMNN / EquivMP / So3krates take pre-built basis submodules.
6. Unify basis-function dispatch: `BesselBasisConfig` gains a `variant: kocer | standard` field so `ModelBuilder.build_basis_function()` is the single point of dispatch for *all* models including MACE. Default `kocer` preserves every existing GMNN / EquivMP / So3krates config; MACE configs override to `standard` (`MaceBesselBasis`).
7. Preserve the torch→jax converter slot-key map at `apax/transfer_learning/mace_foundation.py` — no parity-test churn.

## Non-goals

- Changing the *formula* for the existing `BesselBasis` Kocer-symmetrised class, or replacing it as the default for non-MACE models. The legacy form stays the default; the new `variant: standard` value opts in to `MaceBesselBasis`.
- Changing the discriminator-tagging convention (`name:` field) used by every other apax discriminated union. The interaction-config tag stays `name:` for codebase-wide consistency.
- Adding new interaction variants. The set stays {RealAgnosticResidual, RealAgnosticDensity, RealAgnosticDensityResidual}.
- Backward-compatibility shims for the old flat schema.

## Schema (`apax/config/model_config.py`)

`MaceModelConfig` becomes an assembly of four nested groups, mirroring the forward pass.

### Extended — `BesselBasisConfig`

Adds a `variant` field so the same discriminated `BasisConfig` slot serves every model, including MACE:

```python
class BesselBasisConfig(BaseModel, extra="forbid"):
    name: Literal["bessel"] = "bessel"
    variant: Literal["kocer", "standard"] = "kocer"   # NEW
    n_basis: PositiveInt = 16
    r_max: PositiveFloat = 5.0
```

- `kocer` (default) — Kocer 2019 symmetrised form, what GMNN / EquivMP / So3krates already use today via `BesselBasis`. Existing user configs are unaffected because the field defaults to this value.
- `standard` — the textbook spherical-Bessel form `√(2/r_max)·sin(nπr/r_max)/r`, what torch-mace and most modern descriptors use, implemented by `MaceBesselBasis`.

`ModelBuilder.build_basis_function()` dispatches on `(name, variant)` → returns either `BesselBasis` or `MaceBesselBasis`. After this change the MACE descriptor stops constructing its basis directly; it goes through the same builder helper as every other apax model.

### Reused unchanged

`GaussianBasisConfig`, `DistanceTransformConfig`, `AgnesiTransformConfig`, every base-config field.

### New — `MaceRadialEmbeddingConfig`

Owns the cutoff envelope and the optional length transform that compose with the basis to produce the per-edge radial features.

```python
class MaceRadialEmbeddingConfig(BaseModel, extra="forbid"):
    num_polynomial_cutoff: PositiveInt = 5
    distance_transform: Optional[DistanceTransformConfig] = None
```

### New — interaction discriminated union

Each variant is a typed pydantic class with an empty body today; the `name` Literal is the discriminator. Per-variant fields (e.g. a future `RealAgnosticAttention` with `num_heads`) drop in additively without breaking the schema.

```python
class RealAgnosticResidualConfig(BaseModel, extra="forbid"):
    name: Literal["RealAgnosticResidual"] = "RealAgnosticResidual"

class RealAgnosticDensityConfig(BaseModel, extra="forbid"):
    name: Literal["RealAgnosticDensity"] = "RealAgnosticDensity"

class RealAgnosticDensityResidualConfig(BaseModel, extra="forbid"):
    name: Literal["RealAgnosticDensityResidual"] = "RealAgnosticDensityResidual"

InteractionConfig = Annotated[
    Union[
        RealAgnosticResidualConfig,
        RealAgnosticDensityConfig,
        RealAgnosticDensityResidualConfig,
    ],
    Field(discriminator="name"),
]
```

### New — `MaceDescriptorConfig`

Message-passing architecture knobs. `interactions` is the typed per-layer list. The number of interaction layers is `len(interactions)`; there is no separate `num_interactions` field.

```python
class MaceDescriptorConfig(BaseModel, extra="forbid"):
    max_ell: PositiveInt = 3
    hidden_irreps: str = "128x0e + 128x1o"
    correlation: PositiveInt = 3
    interactions: list[InteractionConfig] = Field(
        default_factory=lambda: [
            RealAgnosticResidualConfig(),
            RealAgnosticResidualConfig(),
        ],
        min_length=1,
    )
    avg_num_neighbors: PositiveFloat = 1.0
    use_cueq: bool = False
```

### New — `MaceReadoutConfig`

```python
class MaceReadoutConfig(BaseModel, extra="forbid"):
    kind: Literal["mace", "standard"] = "mace"
    MLP_irreps: str = "16x0e"
```

`kind:` is a Literal field, not a pydantic discriminator — only the `interactions` list and the inherited `basis` / `empirical_corrections` / `distance_transform` use discriminators.

### Re-shaped `MaceModelConfig`

```python
class MaceModelConfig(BaseModelConfig, extra="forbid"):
    name: Literal["mace"] = "mace"
    basis: BesselBasisConfig = BesselBasisConfig(
        variant="standard", n_basis=8, r_max=5.0,
    )
    radial_embedding: MaceRadialEmbeddingConfig = MaceRadialEmbeddingConfig()
    descriptor: MaceDescriptorConfig = MaceDescriptorConfig()
    readout: MaceReadoutConfig = MaceReadoutConfig()
```

The `basis` default is overridden per-subclass to (`variant="standard"`, `n_basis=8`, `r_max=5.0`) — preserves today's MACE numerics (the standard-Bessel formula plus today's defaults) without requiring users to spell `variant: standard` in fresh MACE configs. `BaseModelConfig.basis` remains (`variant="kocer"`, `n_basis=16`, `r_max=5.0`) for non-MACE models.

### Removed from `MaceModelConfig`

`r_max`, `num_bessel`, `num_polynomial_cutoff`, `max_ell`, `hidden_irreps`, `num_interactions`, `correlation`, `interaction_cls`, `use_cueq`, `readout_kind`, `MLP_irreps`, `avg_num_neighbors`, `distance_transform`. All redistributed into the four sub-configs above.

## Linen descriptor refactor

Two intertwined edits in `apax/layers/descriptor/`.

### `MaceRadialEmbedding` becomes purely radial and consumes a pre-built basis

Two changes at once:

1. `max_ell` and the spherical-harmonics computation leave; SH are angular features and now live in `MaceRepresentation` proper. The submodule's contract becomes: take a per-edge displacement, return a per-edge radial-feature tensor.
2. The basis function is **injected** instead of constructed inline. The submodule no longer knows about `n_basis`, `r_max`, or which Bessel variant is used — it only owns the polynomial cutoff and the optional distance transform composition.

```python
class MaceRadialEmbedding(nn.Module):
    basis_fn: nn.Module                              # injected by builder
    num_polynomial_cutoff: int
    r_max: float                                     # forwarded to PolynomialCutoff
    distance_transform: Optional[Any] = None

    @nn.compact
    def __call__(self, dr_vec, Z, idx):
        r_ij = jnp.linalg.norm(dr_vec, axis=-1)
        cutoff = PolynomialCutoff(p=self.num_polynomial_cutoff, r_max=self.r_max)(r_ij)
        if self.distance_transform is not None:
            r_ij = self.distance_transform(r_ij, Z, idx)
        bessel = self.basis_fn(r_ij)
        return (bessel * cutoff[..., None]).astype(dr_vec.dtype)
```

`r_max` stays as a scalar attribute because `PolynomialCutoff` needs it; the basis_fn already carries its own `r_max` (set when the builder constructed it). The two are sourced from the same `model.basis.r_max` so they cannot drift.

Return type changes from `(radial, sph)` to `radial`. Every consumer of `MaceRadialEmbedding(...)(...)` is inside `MaceRepresentation`, so the public surface is unaffected.

### Linen-level placement of `distance_transform` (load-bearing)

Today the `AgnesiTransform` Linen module is attached as a field on `MaceRepresentation`, **not** on `MaceRadialEmbedding`. Its parameters land in the apax pytree at:

```
buffers/energy_model/representation/distance_transform/{a, q, p, covalent_radii}
```

The torch→jax converter (`apax/transfer_learning/mace_foundation.py:1033-1046`) targets that exact path. **This wiring must be preserved across the refactor**: the converter slot key is the contract.

Resolution: `distance_transform` lives in `MaceRadialEmbeddingConfig` at the **config** level (it groups naturally with the rest of the radial embedding) but stays a Linen field on `MaceRepresentation` at the **module** level. The builder threads the constructed module into both — one as a `MaceRepresentation` field, one as a `MaceRadialEmbedding` constructor argument — and Linen registers it once at the parent's slot. **Config-level grouping ≠ Linen-level field placement.**

### `MaceRepresentation` accepts a pre-built `radial_embedding`

```python
class MaceRepresentation(nn.Module):
    radial_embedding: nn.Module                       # injected by builder
    distance_transform: Optional[nn.Module]            # Linen field — see slot-key note
    max_ell: int
    hidden_irreps: str
    correlation: int
    interactions: tuple[dict, ...]                    # discriminated dicts
    avg_num_neighbors: float
    num_elements: int
    use_cueq: bool
    apply_mask: bool
    dtype: Any

    @nn.compact
    def __call__(self, dr_vec, Z, idx):
        radial = self.radial_embedding(dr_vec, Z, idx)
        sph = e3nn.spherical_harmonics(
            e3nn.Irreps.spherical_harmonics(self.max_ell),
            dr_vec, normalize=True, normalization="component",
        )
        for k, inter_cfg in enumerate(self.interactions):
            Block = _INTERACTION_BLOCK_CLS[inter_cfg["name"]]
            ...   # same body as today
```

`distance_transform` stays a Linen field on `MaceRepresentation` for the slot-key reason above; the builder passes the same Linen instance into the `MaceRadialEmbedding` constructor so the radial submodule can apply it without owning its parameter slot.

The single-string broadcast and length-vs-`num_interactions` validator are deleted.

`num_interactions` consumed elsewhere becomes `len(self.interactions)`.

### Slot-key preservation

The torch→jax converter has three slot-key paths into the radial-embedding region of the apax pytree:

1. `…/representation/radial_embedding/…` — the radial-embedding submodule itself. Preserved: `radial_embedding` as a Linen field on `MaceRepresentation` is auto-named `"radial_embedding"` exactly like today's explicit `name="radial_embedding"`.
2. `…/representation/distance_transform/…` — the Agnesi parameters (`a`, `q`, `p`, `covalent_radii`). Preserved: `distance_transform` stays a Linen field on `MaceRepresentation`, not pushed down into `MaceRadialEmbedding`. See the "Linen-level placement" note above.
3. `…/representation/radial_embedding/{MaceBesselBasis_0|basis_fn}/…` — irrelevant in practice. The converter only **reads** `bessel_weights.shape[0]` from the torch model to extract `num_bessel`; it does not copy the bessel weights or polynomial-cutoff state into the apax pytree (apax recomputes both from `n_basis` / `r_max` / `p` in `setup()`). Switching from inline construction to a `basis_fn` field changes the inner sub-name but doesn't touch any slot the converter writes to.

`InteractionBlock_{k}` names continue to be set explicitly in the construction loop. The converter's torch→jax slot-key map is **unchanged**.

## Builder rewiring (`apax/nn/builder.py`)

`MaceBuilder` reads from the four nested groups and constructs the radial submodule itself, then injects it into `MaceRepresentation`.

```python
class ModelBuilder:
    def build_basis_function(self):
        basis = self.config["basis"]
        if basis["name"] == "gaussian":
            return GaussianBasis(
                n_basis=basis["n_basis"], r_min=basis["r_min"], r_max=basis["r_max"],
                spacing=basis["spacing"], dtype=self.config["descriptor_dtype"],
            )
        if basis["name"] == "bessel":
            if basis["variant"] == "kocer":
                return BesselBasis(
                    n_basis=basis["n_basis"], r_max=basis["r_max"],
                    dtype=self.config["descriptor_dtype"],
                )
            if basis["variant"] == "standard":
                return MaceBesselBasis(
                    n_basis=basis["n_basis"], r_max=basis["r_max"],
                    dtype=self.config["descriptor_dtype"],
                )
        raise ValueError(f"unknown basis: {basis}")


class MaceBuilder(ModelBuilder):
    def build_descriptor(self, apply_mask):
        from apax.layers.descriptor.mace import MaceRepresentation
        from apax.layers.descriptor.basis_functions import MaceRadialEmbedding

        basis    = self.config["basis"]
        re_cfg   = self.config["radial_embedding"]
        desc_cfg = self.config["descriptor"]

        basis_fn = self.build_basis_function()                       # respects variant
        dt       = self._build_distance_transform(re_cfg["distance_transform"])

        radial_embedding = MaceRadialEmbedding(
            basis_fn=basis_fn,
            num_polynomial_cutoff=re_cfg["num_polynomial_cutoff"],
            r_max=basis["r_max"],
            distance_transform=dt,                                   # same Linen instance
        )
        return MaceRepresentation(
            radial_embedding=radial_embedding,
            distance_transform=dt,                                   # parent slot, see note
            max_ell=desc_cfg["max_ell"],
            hidden_irreps=desc_cfg["hidden_irreps"],
            correlation=desc_cfg["correlation"],
            interactions=tuple(desc_cfg["interactions"]),
            avg_num_neighbors=desc_cfg["avg_num_neighbors"],
            use_cueq=desc_cfg["use_cueq"],
            num_elements=self.n_species,
            apply_mask=apply_mask,
            dtype=self.config["descriptor_dtype"],
        )

    def build_readout(self, head_config, is_feature_fn=False, only_use_n_layers=None):
        readout = self.config["readout"]
        if readout["kind"] != "mace" or is_feature_fn:
            return super().build_readout(head_config, is_feature_fn, only_use_n_layers)

        import e3nn_jax as e3nn
        from apax.layers.readout import MaceReadout

        n_shallow_ensemble = 0
        ens = head_config.get("ensemble") if isinstance(head_config, dict) else None
        if ens and ens.get("kind") == "shallow":
            n_shallow_ensemble = ens["n_members"]

        desc_cfg = self.config["descriptor"]
        return MaceReadout(
            num_interactions=len(desc_cfg["interactions"]),
            hidden_dim=e3nn.Irreps(desc_cfg["hidden_irreps"]).filter("0e").dim,
            MLP_irreps=readout["MLP_irreps"],
            n_shallow_ensemble=n_shallow_ensemble,
            dtype=self.config["readout_dtype"],
        )
```

The `list → tuple` coercion for `interactions` happens once, inside the builder, for the same reason as today: Linen's mutable-default protection rejects list-typed fields.

`_build_distance_transform` is a small private helper that maps the discriminated `distance_transform` dict to its Linen module (`AgnesiTransform` today; same dispatch the current builder does inline).

## Converter rewrite (`apax/transfer_learning/mace_foundation.py`)

Add a single mapping table from torch class names to apax discriminator values:

```python
_TORCH_TO_APAX_INTERACTION = {
    "RealAgnosticResidualInteractionBlock":         "RealAgnosticResidual",
    "RealAgnosticDensityInteractionBlock":          "RealAgnosticDensity",
    "RealAgnosticDensityResidualInteractionBlock":  "RealAgnosticDensityResidual",
}
```

`_extract_config_from_torch` emits the nested shape directly:

```python
cfg = {
    "name": "mace",
    "basis": {
        "name": "bessel",
        "variant": "standard",
        "n_basis": int(model.radial_embedding.bessel_fn.bessel_weights.shape[0]),
        "r_max": float(model.r_max),
    },
    "radial_embedding": {
        "num_polynomial_cutoff": int(model.radial_embedding.cutoff_fn.p),
        "distance_transform": distance_transform_cfg,    # None or {name: agnesi, ...}
    },
    "descriptor": {
        "max_ell": int(max_ell),
        "hidden_irreps": hidden_irreps,
        "correlation": int(correlation),
        "interactions": [
            {"name": _TORCH_TO_APAX_INTERACTION[type(blk).__name__]}
            for blk in model.interactions
        ],
        "avg_num_neighbors": avg_num_neighbors,
        "use_cueq": False,
    },
    "readout": {"kind": "mace", "MLP_irreps": "16x0e"},
    "empirical_corrections": empirical_corrections,
    "descriptor_dtype": "fp64",
    "readout_dtype": "fp64",
    "scale_shift_dtype": "fp64",
}
```

The two mirror-asserts on `r_max` / `num_bessel` are deleted — the duplication class no longer exists.

Foundation models that ship unknown torch class names raise an explicit `KeyError` from the mapping lookup; this is desirable (fast failure).

## Tests

- **Update fixtures** under `tests/` that hand-write a `MaceModelConfig` to use the nested shape. The legacy parity-test fixtures referenced by recent commits (s22, MatPES, MPA-0) are config-shape edits only; no logic change.
- **Regenerate** `tmp/mace-mpa-0-medium/config.yaml` by re-running `apax convert-mace` after the converter is updated, so the in-tree example reflects the new shape.
- **Add** unit coverage in `tests/test_config.py` (or equivalent):
  - `MaceModelConfig` round-trips through pydantic with all three interaction variants in a list.
  - Rejects bare strings in the `interactions` list (e.g. `"RealAgnosticResidual"` without the `name:` wrap).
  - Rejects unknown variant names.
  - `interactions: []` is rejected by pydantic (`min_length=1` on the field).
  - `BesselBasisConfig(variant="kocer")` builds `BesselBasis`; `variant="standard"` builds `MaceBesselBasis`.
  - Existing GMNN / EquivMP / So3krates configs (no `variant` key) parse and dispatch to `BesselBasis` exactly as before — verified by an existing-fixture round-trip plus a numerical-output snapshot at one r value.
- **Run** the existing parity harness (s22 + MPA-0 + MatPES) end-to-end against MACE-MP-0 small, MACE-MPA-0 medium, MACE-matpes-r2scan — all three foundations should pass with no parity-test code changes.

## Docs

- Re-write the in-tree fine-tune template referenced by commit `619ddba5` (`docs(mace): add fine-tune template config using TransferLearningConfig`) to use the nested shape.
- Sweep `docs/source/` for any sphinx pages that document the flat MACE schema and bring them into line with the nested shape. The branch's existing planning/spec markdown under `docs/superpowers/` was removed prior to this design; any remaining references in the user-facing sphinx tree are the only docs left to update.

## What stays put

| component | status |
|---|---|
| `BaseModelConfig`, `EnsembleConfig`, `EmpiricalCorrection`, `PropertyHead`, `DistanceTransformConfig` | unchanged |
| GMNN / EquivMP / So3krates configs and builders | unchanged (`variant` defaults to `kocer`, dispatch path identical) |
| Existing `BesselBasis` Kocer formula | unchanged |
| Neighbour-list code reading `config.model.basis.r_max` (`apax/md/`, `apax/bal/`, `apax/train/`) | unchanged |
| `_INTERACTION_BLOCK_CLS` lookup, per-block Linen modules, symmetric-contraction code | unchanged |
| Torch→jax converter slot-key map | unchanged |
| `MaceBesselBasis`, `PolynomialCutoff`, `AgnesiTransform` Linen modules | unchanged (only the wrapper `MaceRadialEmbedding` loses `max_ell` and gains a `basis_fn` field) |

## Risk register

| concern | mitigation |
|---|---|
| Slot-key drift breaks the parity converter | The `radial_embedding` and `distance_transform` field names are preserved; `InteractionBlock_{k}` names stay explicit. Verify by running the parity harness against MACE-MP-0 / MPA-0 / MatPES after the refactor. |
| Pydantic discriminated union with empty per-variant configs | Standard pydantic v2 pattern via `Annotated[..., Field(discriminator="name")]`. No exotic features. |
| Inner basis-fn slot rename (`MaceBesselBasis_0` → `basis_fn`) breaks something | Slot is unobserved by the converter (apax recomputes the bessel constants in `setup()` from `n_basis` / `r_max`). Verified by inspecting `_extract_config_from_torch` and the converter's `_map_*` helpers — none reference the basis-weight slot. |
| Default `BesselBasisConfig.variant="kocer"` accidentally selected for MACE | `MaceModelConfig.basis` overrides to `variant="standard"` at the subclass level; the converter writes `variant: standard` explicitly into emitted yaml. Tests assert this default. |
| User-facing yaml is more verbose | One-time cost. Configs are auto-generated by the converter for foundation models; freshly trained apax users write 4 nested groups instead of 13 flat fields, grouped by physical role. |
| Tests in `pytest.mark.protected` reference flat fields | Read-only review: no protected test should depend on the MACE schema; if any does, schedule a separate change to update it via the appropriate channel. |
