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
6. Preserve the torch→jax converter slot-key map at `apax/transfer_learning/mace_foundation.py` — no parity-test churn.

## Non-goals

- Replacing the legacy `BesselBasis` (Kocer-symmetrised) with `MaceBesselBasis` for non-MACE models. Out of scope; would touch GMNN parity tests.
- Changing the discriminator-tagging convention (`name:` field) used by every other apax discriminated union. The interaction-config tag stays `name:` for codebase-wide consistency.
- Adding new interaction variants. The set stays {RealAgnosticResidual, RealAgnosticDensity, RealAgnosticDensityResidual}.
- Backward-compatibility shims for the old flat schema.

## Schema (`apax/config/model_config.py`)

`MaceModelConfig` becomes an assembly of four nested groups, mirroring the forward pass.

### Reused unchanged

`BesselBasisConfig`, `DistanceTransformConfig`, `AgnesiTransformConfig`, every base-config field.

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
    basis: BesselBasisConfig = BesselBasisConfig(n_basis=8, r_max=5.0)
    radial_embedding: MaceRadialEmbeddingConfig = MaceRadialEmbeddingConfig()
    descriptor: MaceDescriptorConfig = MaceDescriptorConfig()
    readout: MaceReadoutConfig = MaceReadoutConfig()
```

The `basis` default is overridden per-subclass to (`n_basis=8`, `r_max=5.0`) to preserve today's MACE defaults; `BaseModelConfig.basis` remains (`n_basis=16`, `r_max=5.0`) for non-MACE models.

### Removed from `MaceModelConfig`

`r_max`, `num_bessel`, `num_polynomial_cutoff`, `max_ell`, `hidden_irreps`, `num_interactions`, `correlation`, `interaction_cls`, `use_cueq`, `readout_kind`, `MLP_irreps`, `avg_num_neighbors`, `distance_transform`. All redistributed into the four sub-configs above.

## Linen descriptor refactor

Two intertwined edits in `apax/layers/descriptor/`.

### `MaceRadialEmbedding` becomes purely radial

`max_ell` and the spherical-harmonics computation move out. The submodule's contract becomes: take a per-edge displacement, return a per-edge radial-feature tensor. SH are angular features and now live in `MaceRepresentation` proper.

```python
class MaceRadialEmbedding(nn.Module):
    r_max: float
    num_bessel: int
    num_polynomial_cutoff: int
    distance_transform: Optional[Any] = None

    @nn.compact
    def __call__(self, dr_vec, Z, idx):
        r_ij = jnp.linalg.norm(dr_vec, axis=-1)
        cutoff = PolynomialCutoff(p=self.num_polynomial_cutoff, r_max=self.r_max)(r_ij)
        if self.distance_transform is not None:
            r_ij = self.distance_transform(r_ij, Z, idx)
        bessel = MaceBesselBasis(n_basis=self.num_bessel, r_max=self.r_max,
                                 dtype=dr_vec.dtype)(r_ij)
        return (bessel * cutoff[..., None]).astype(dr_vec.dtype)
```

Return type changes from `(radial, sph)` to `radial`. Every consumer of `MaceRadialEmbedding(...)(...)` is inside `MaceRepresentation`, so the public surface is unaffected.

### `MaceRepresentation` accepts a pre-built `radial_embedding`

```python
class MaceRepresentation(nn.Module):
    radial_embedding: nn.Module                       # injected by builder
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

The single-string broadcast and length-vs-`num_interactions` validator are deleted.

`num_interactions` consumed elsewhere becomes `len(self.interactions)`.

### Slot-key preservation

`radial_embedding` as a Linen field is auto-named `"radial_embedding"` — same key the converter targets at `…/MaceRepresentation_0/radial_embedding/...`. `InteractionBlock_{k}` names continue to be set explicitly in the construction loop. The converter's torch→jax slot-key map is **unchanged**.

## Builder rewiring (`apax/nn/builder.py`)

`MaceBuilder` reads from the four nested groups and constructs the radial submodule itself, then injects it into `MaceRepresentation`.

```python
class MaceBuilder(ModelBuilder):
    def build_descriptor(self, apply_mask):
        from apax.layers.descriptor.mace import MaceRepresentation
        from apax.layers.descriptor.basis_functions import MaceRadialEmbedding

        basis    = self.config["basis"]
        re_cfg   = self.config["radial_embedding"]
        desc_cfg = self.config["descriptor"]

        dt = self._build_distance_transform(re_cfg["distance_transform"])
        radial_embedding = MaceRadialEmbedding(
            r_max=basis["r_max"],
            num_bessel=basis["n_basis"],
            num_polynomial_cutoff=re_cfg["num_polynomial_cutoff"],
            distance_transform=dt,
        )
        return MaceRepresentation(
            radial_embedding=radial_embedding,
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
- **Run** the existing parity harness (s22 + MPA-0 + MatPES) end-to-end against MACE-MP-0 small, MACE-MPA-0 medium, MACE-matpes-r2scan — all three foundations should pass with no parity-test code changes.

## Docs

- Re-write the in-tree fine-tune template referenced by commit `619ddba5` (`docs(mace): add fine-tune template config using TransferLearningConfig`) to use the nested shape.
- Sweep `docs/source/` for any sphinx pages that document the flat MACE schema and bring them into line with the nested shape. The branch's existing planning/spec markdown under `docs/superpowers/` was removed prior to this design; any remaining references in the user-facing sphinx tree are the only docs left to update.

## What stays put

| component | status |
|---|---|
| `BaseModelConfig`, `EnsembleConfig`, `EmpiricalCorrection`, `PropertyHead`, `DistanceTransformConfig` | unchanged |
| GMNN / EquivMP / So3krates configs and builders | unchanged |
| Neighbour-list code reading `config.model.basis.r_max` (`apax/md/`, `apax/bal/`, `apax/train/`) | unchanged |
| `_INTERACTION_BLOCK_CLS` lookup, per-block Linen modules, symmetric-contraction code | unchanged |
| Torch→jax converter slot-key map | unchanged |
| `MaceBesselBasis`, `PolynomialCutoff`, `AgnesiTransform` Linen modules | unchanged (only the wrapper `MaceRadialEmbedding` loses `max_ell`) |

## Risk register

| concern | mitigation |
|---|---|
| Slot-key drift breaks the parity converter | The `radial_embedding` field name is preserved; `InteractionBlock_{k}` names stay explicit. Verify by running the parity harness against MACE-MP-0 / MPA-0 / MatPES after the refactor. |
| Pydantic discriminated union with empty per-variant configs | Standard pydantic v2 pattern via `Annotated[..., Field(discriminator="name")]`. No exotic features. |
| `MaceBesselBasis` slot key inside `radial_embedding` | Verify the converter map. `MaceRadialEmbedding` still constructs `MaceBesselBasis` internally so the inner slot key is unchanged. |
| User-facing yaml is more verbose | One-time cost. Configs are auto-generated by the converter for foundation models; freshly trained apax users write 4 nested groups instead of 13 flat fields, grouped by physical role. |
| Tests in `pytest.mark.protected` reference flat fields | Read-only review: no protected test should depend on the MACE schema; if any does, schedule a separate change to update it via the appropriate channel. |
