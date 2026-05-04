# MACE Foundation Model Integration — Design Spec

**Date:** 2026-04-20 (revised 2026-04-21)
**Status:** Approved — ready for implementation plan
**Scope:** Native MACE descriptor in apax + foundation-model loading (MACE-MP, MACE-MPA) + fine-tuning with apax's shallow-ensemble & property heads + JAX-MD integration

**Revision note (2026-04-21):** The original spec introduced a parallel `.apax/`-dir format (`params.msgpack` + `config.json` + `metadata.json`), a dedicated `load_mace_foundation` loader, and a separate `MaceFoundationEnergyModel`. Verification against the code base found that this creates two parallel load paths and an ASECalculator branch that shouldn't exist. The spec now mandates a single path: the converter produces apax's existing training-output layout (`config.yaml` + `best/` orbax checkpoint), and every consumer (`ASECalculator`, `apax md`, `apax eval`, BAL) reads it via the existing `restore_parameters` code. MACE's foundation-specific forward structure (per-layer readouts, global scale/shift + per-element E0) slots into the stock `EnergyModel(representation, readout, scale_shift)` via a new `MaceReadout` module and the existing `PerElementScaleShift`. See §3 and §4 below.

## 1. Problem

apax is a JAX/Flax (linen) framework for training atomistic interatomic potentials. It supports GMNN, So3krates, and EquivMP. Users want to:

1. Load pretrained MACE foundation models (MACE-MP and MACE-MPA) into apax.
2. Fine-tune them, notably by replacing the readout head with apax's shallow-ensemble head to obtain uncertainty-aware models.
3. Use the fine-tuned models in JAX-MD and via the ASE calculator through apax's existing MD paths.
4. Train additional property heads (charges, dipoles, stresses, custom per-atom properties) on top of the frozen or fine-tuned MACE backbone.
5. Do all of the above without adding `torch`, `mace-torch`, or `mace-jax` as apax runtime dependencies, while remaining performance-competitive with upstream mace-jax.

The sibling projects exist on-disk for reference: `/Users/fzills/tools/mace` (torch-mace), `/Users/fzills/tools/mace-jax` (Flax-NNX port with cuequivariance). They are *not* dependencies of apax; they may be used at model-conversion time only.

## 2. Goals and non-goals

### Goals

- `apax[mace]` extra installs everything needed at runtime (e3nn-jax, cuequivariance-jax).
- `apax convert-mace <source> <dst>/` CLI where `<source>` is either a canonical MACE foundation name (`medium`, `medium-mpa-0`, …) resolved via upstream `mace.calculators.foundations_models.mace_mp(return_raw_model=True)` — which handles bundled-local → cache → download — or a local `.model` path. Torch is imported lazily via try/except.
- `MaceRepresentation` is a Flax linen `nn.Module` matching apax's existing descriptor contract.
- **The converter output is identical in shape to any apax training-output directory.** Loading a foundation is byte-for-byte the same code path as loading a user-trained model.
- Parity with upstream torch-mace on energies (rtol 1e-5) and forces (rtol 1e-4) for MACE-MP-0 and MACE-MPA variants.
- Shallow-ensemble fine-tuning with the existing `TransferLearningConfig` works through a YAML toggle; no new training-loop code.
- `apax md`, `apax eval`, `apax.md.ase_calc`, and BAL work end-to-end with MACE; no changes to those modules.
- `PropertyHead` reuses MACE's scalar features to train charges, dipoles, polarizabilities, stresses, and arbitrary per-atom fields; ensemble uncertainty is automatic.
- CPU inference within 1.2× of upstream mace-jax; training within 1.3×; GPU with `use_cueq=True` within 1.1× of cuequivariance-accelerated mace-jax.

### Non-goals (deferred follow-ups)

- MACE-OFF, MACE-ANICC, MACE-OMOL foundation models.
- **Training** multi-head MACE models (`num_heads > 1`) in apax. We *consume* multi-head foundation weights by selecting a single head at convert time (see §4.4); what's out of scope is producing new multi-head models during fine-tuning.
- Equivariant property heads that consume MACE's l≥1 irrep features directly.
- LAMMPS MLIAP export.
- Changes to distributed-training infrastructure.
- Any port or upstreaming of mace-jax code into apax.

## 3. Architecture

### 3.1 Dependency strategy

No hard deps added. One new optional extra + one dev-only group:

```toml
[project.optional-dependencies]
mace = [
    "e3nn-jax>=0.21.0",
    "cuequivariance-jax>=0.9.1",
    "cuequivariance>=0.9.1",
]

[dependency-groups]
mace-convert = [
    "torch>=2.1",
    "mace-torch>=0.3",
    "mace-jax",          # reference implementation, used only by the converter iteration
]
```

`torch`, `mace-torch`, and `mace-jax` are never in `[project]`. The CLI converter does a runtime `try: import torch, mace` and raises a typer error telling the user how to install them if they invoke the conversion step. `mace-jax` is referenced from a pinned local path in `[tool.uv.sources]` during development for parameter-mapping validation; it does not need to be available to library consumers.

### 3.2 Module layout (all additive)

```
apax/
├── layers/
│   ├── descriptor/
│   │   ├── mace.py                   # MaceRepresentation
│   │   └── mace_blocks.py            # InteractionBlock, ProductBlock, LinearNodeEmbedding,
│   │                                  # LinearReadoutBlock, NonLinearReadoutBlock
│   └── readout.py                    # + MaceReadout (per-layer sum) alongside AtomisticReadout
├── nn/
│   └── builder.py                    # + MaceBuilder (overrides build_descriptor + build_readout)
├── config/
│   └── model_config.py               # + MaceModelConfig
├── cli/
│   └── convert_mace.py               # apax convert-mace subcommand
├── transfer_learning/
│   └── mace_foundation.py            # run_conversion(source, dst, ...) ONLY
└── tests/
    ├── unit_tests/
    │   ├── layers/descriptor/
    │   │   ├── test_mace_descriptor.py
    │   │   └── test_mace_blocks.py   # includes LinearReadoutBlock + NonLinearReadoutBlock
    │   └── layers/
    │       └── test_mace_readout.py
    ├── integration_tests/
    │   └── mace/
    │       ├── test_convert.py       # gated @pytest.mark.mace_parity
    │       ├── test_mace_parity.py   # gated @pytest.mark.mace_parity
    │       └── test_mace_md.py       # gated
    └── …
```

Zero modifications to `EnergyModel`, `ShallowEnsembleModel`, `EnergyDerivativeModel`, `PropertyHead`, `AtomisticReadout`, `PerElementScaleShift`, the base `ModelBuilder`, `trainer.py`, `simulate.py`, `ase_calc.py`, or `restore_parameters`. MACE slots into the existing composition.

**Deleted / not present:**
- ~~`apax/nn/mace_foundation_model.py`~~ — no separate full-energy model; `MaceReadout` + stock `EnergyModel` reproduces the foundation forward pass.
- ~~`load_mace_foundation`, `_resolve_short_name`~~ — loading is `restore_parameters(model_dir)`, which every other apax consumer already uses.
- ~~`_is_mace_foundation_dir`~~ in `ase_calc.py` — not needed; ASECalculator reads one format.

### 3.3 Descriptor contract (unchanged)

All descriptors follow:

```python
def __call__(self, dr_vec, Z, idx) -> Array:  # (n_atoms, n_features)
```

Distances are computed in `EnergyModel` before the descriptor is called. Output is always plain scalar features — never `IrrepsArray` — so downstream readouts receive their expected shape.

### 3.4 `MaceRepresentation`

```python
class MaceRepresentation(nn.Module):
    r_max: float = 5.0
    num_bessel: int = 8
    num_polynomial_cutoff: int = 5
    max_ell: int = 3
    hidden_irreps: str = "128x0e + 128x1o"
    num_interactions: int = 2
    correlation: int = 3
    interaction_cls: Literal[
        "RealAgnostic", "RealAgnosticResidual",
        "RealAgnosticDensity", "RealAgnosticDensityResidual",
    ] = "RealAgnosticResidual"
    num_elements: int = 119          # apax convention; indexed by Z directly
    use_cueq: bool = False
    apply_mask: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, dr_vec, Z, idx) -> Array:
        ...
        return features   # (n_atoms, num_interactions * hidden_scalar_dim)
```

Internally the descriptor builds the **full multi-irrep MACE pipeline**:

```
sh_irreps          = Irreps.spherical_harmonics(max_ell)              # 1x0e + 1x1o + … + 1x{L}{e|o}
num_features       = hidden_irreps.count(Irrep(0,1))                  # 128 for "128x0e"
node_attr_irreps   = Irreps([(num_elements, (0, 1))])                 # 119x0e — one-hot Z
node_feats_irreps  = Irreps([(num_features, (0, 1))])                 # 128x0e (initial scalars)
interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()  # 128x0e + 128x1o + 128x2e + 128x3o
```

Per layer:

1. `(message, sc) = InteractionBlock(node_feats_irreps=prev, node_attrs_irreps=node_attr_irreps,
   edge_attrs_irreps=sh_irreps, target_irreps=interaction_irreps,
   hidden_irreps=this_layer_hidden)(node_feats, sph, radial, Z_one_hot, i, j)`
2. `node_feats = ProductBlock(node_feats_irreps=interaction_irreps, target_irreps=this_layer_hidden,
   correlation=correlation, num_elements=num_elements, use_sc=True)(message, sc, Z)`
3. The scalar-only slice of `node_feats` is appended to the per-layer output buffer.

`this_layer_hidden` is `hidden_irreps` for layers `0..N-2` and `Irreps(str(hidden_irreps[0]))` for the last layer (the "first-irrep collapse"); for `hidden_irreps="128x0e"` every layer ends in `128x0e`, for `"128x0e+128x1o"` only the last layer collapses to `128x0e`. After the loop the per-layer scalars are concatenated, masked, and returned — `MaceReadout` (§3.5) consumes that scalar tail.

This is the same forward composition as upstream torch-mace `MACE.__init__` + `MACE.forward` (`mace/modules/models.py:148-273`) and mace-jax `MACE.__init__` (`mace_jax/modules/models.py:249-303`).

### 3.4a `InteractionBlock` and `ProductBlock` contracts

These primitives live in `apax/layers/descriptor/mace_blocks.py`. Both take explicit irrep specifications so `MaceRepresentation` can wire them with the multi-irrep `interaction_irreps` target.

#### `InteractionBlock`

```python
class InteractionBlock(nn.Module):
    node_feats_irreps: str            # input node features (e.g. "128x0e")
    node_attrs_irreps: str            # one-hot element irreps (e.g. "119x0e")
    edge_attrs_irreps: str            # spherical harmonics, e.g. "1x0e + 1x1o + 1x2e + 1x3o"
    target_irreps: str                # multi-irrep target (interaction_irreps)
    hidden_irreps: str                # post-product target (used by skip_tp slot)
    interaction_cls: str = "RealAgnosticResidual"
    radial_mlp: tuple = (64, 64, 64)

    @nn.compact
    def __call__(self, node_feats, sph, radial, Z_one_hot, i, j):
        # 1. linear_up:  node_feats   -> node_feats_irreps         (in-place mix)
        # 2. conv_tp:    (node_feats_j × sph)  filtered to target_irreps via
        #                tp_out_irreps_with_instructions
        # 3. radial MLP gates the per-edge TP weights from `radial`
        # 4. scatter-sum into receivers
        # 5. linear:     irreps_mid   -> target_irreps
        # 6. skip_tp:    e3nn.flax.Linear(node_feats × Z_one_hot -> hidden_irreps)
        return message, sc          # message: target_irreps; sc: hidden_irreps
```

`skip_tp` is the parity-critical per-element bilinear — `node_feats × node_attrs` projected to `hidden_irreps`. In e3nn-jax this is `e3nn.flax.Linear(hidden_irreps)(e3nn.tensor_product(node_feats, Z_one_hot))`. It corresponds to torch's `FullyConnectedTensorProduct(node_feats_irreps, node_attrs_irreps, hidden_irreps)` and stores `(num_elements × n_paths)` weights — **not** an element-agnostic Linear.

`tp_out_irreps_with_instructions(irreps_in1, irreps_in2, target)` is a small helper (ported from `mace_jax/modules/blocks.py`) that returns `(irreps_mid, instructions)` where `irreps_mid` is the simplified intersection of `irreps_in1 ⊗ irreps_in2` with `target`. Used so the conv-TP only computes paths that contribute to `target_irreps`.

#### `ProductBlock`

```python
class ProductBlock(nn.Module):
    node_feats_irreps: str            # input (= InteractionBlock.target_irreps)
    target_irreps: str                # output (= this_layer_hidden, e.g. "128x0e")
    correlation: int = 3
    num_elements: int = 119
    use_sc: bool = True
    use_cueq: bool = False

    @nn.compact
    def __call__(self, node_feats, sc, Z):
        # 1. SymmetricContraction (cuequivariance):
        #      node_feats_irreps -> target_irreps with per-element weights
        # 2. Linear(target_irreps -> target_irreps, name="linear")
        # 3. + sc   (if use_sc and sc is not None)
        return out                    # target_irreps
```

The `Linear` after symmetric contraction matches torch's `products.{k}.linear.weight`; without it, the post-SC features would skip the `+ sc` path's linear-mixing step.

### 3.4b Last-layer collapse

For `num_interactions = N` and `hidden_irreps = H`:

| Layer index | `this_layer_hidden`              |
|-------------|----------------------------------|
| `0..N-2`    | `H` (full)                       |
| `N-1`       | `Irreps(str(H[0]))` (first slot) |

For `H = "128x0e"`, every layer ends in `128x0e` (the collapse is a no-op). For `H = "128x0e + 128x1o"`, only the last layer collapses to `"128x0e"` — its product retains scalars only since the readout is scalar-only.

**Z-indexing convention:** `num_elements` defaults to 119 (matches GMNN's `n_species`), so the one-hot embedding is keyed by `Z` directly. Foundation weights (trained on 89 specific elements) are zero-padded at convert time into the 119-row slot (§4.3). No `atomic_numbers` LUT inside the model.

### 3.5 `MaceReadout` — the readout slot for MACE

A standard readout module that fits the `AtomisticReadout` slot in `EnergyModel`. It takes the concatenated per-layer scalar features from `MaceRepresentation` and reproduces MACE's per-layer readout structure:

```python
class MaceReadout(nn.Module):
    num_interactions: int
    hidden_dim: int                      # per-layer scalar channel count (e.g. 128)
    MLP_irreps: str = "16x0e"            # hidden dim of the NonLinear tail
    n_shallow_ensemble: int = 0          # feeds through to the final Linear's output dim
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):                # x: (num_interactions * hidden_dim,)  — post-vmap
        layers = x.reshape(self.num_interactions, self.hidden_dim)
        n_out = self.n_shallow_ensemble if self.n_shallow_ensemble > 0 else 1
        E = jnp.zeros((n_out,), dtype=x.dtype)
        for k in range(self.num_interactions):
            feat = e3nn.IrrepsArray(f"{self.hidden_dim}x0e", layers[k])
            if k < self.num_interactions - 1:
                E = E + LinearReadoutBlock(n_out=n_out, name=f"readout_{k}")(feat)
            else:
                E = E + NonLinearReadoutBlock(
                    MLP_irreps=self.MLP_irreps, n_out=n_out, name=f"readout_{k}")(feat)
        return E
```

Called under `jax.vmap(self.readout)(g)` inside `EnergyModel.__call__`, so each invocation sees a single atom's feature vector. Shallow-ensemble support is automatic: the `n_shallow_ensemble` argument sizes `n_out`, and `EnergyModel` already detects the ensemble case via `E_i.shape[1] > 1`.

`LinearReadoutBlock` (wrap `e3nn.flax.Linear` to `n_out x 0e`) and `NonLinearReadoutBlock` (`Linear → SiLU → Linear → n_out x 0e`) live in `apax/layers/descriptor/mace_blocks.py` alongside the existing blocks.

### 3.6 Scale + shift + atomic energies — via existing `PerElementScaleShift`

MACE's foundation forward is:

```
E_atom = global_scale · raw_atom_energy + global_shift + atomic_energies[Z]
```

The existing `PerElementScaleShift(x, Z)` is:

```python
out = self.scale_param[Z] * x + self.shift_param[Z]   # scale_param, shift_param shape (n_species, 1)
```

These line up exactly when we set:

- `scale_param[Z] = global_scale` (same value for all Z)
- `shift_param[Z] = global_shift + atomic_energies[Z]` (per-element; zero for unseen Z)

The converter performs this fold at write time. No new module, no runtime branching.

### 3.7 `MaceBuilder`

Overrides `build_descriptor` and `build_readout`. `build_energy_model` is inherited unchanged.

```python
class MaceBuilder(ModelBuilder):
    def build_descriptor(self, apply_mask):
        return MaceRepresentation(
            r_max=self.config["r_max"],
            num_bessel=self.config["num_bessel"],
            num_polynomial_cutoff=self.config["num_polynomial_cutoff"],
            max_ell=self.config["max_ell"],
            hidden_irreps=self.config["hidden_irreps"],
            num_interactions=self.config["num_interactions"],
            correlation=self.config["correlation"],
            interaction_cls=self.config["interaction_cls"],
            num_elements=self.n_species,
            use_cueq=self.config["use_cueq"],
            apply_mask=apply_mask,
            dtype=self.config["descriptor_dtype"],
        )

    def build_readout(self, head_config, is_feature_fn=False, only_use_n_layers=None):
        if self.config.get("readout_kind", "mace") == "mace" and not is_feature_fn:
            n_shallow = 0
            ens = head_config.get("ensemble")
            if ens and ens.get("kind") == "shallow":
                n_shallow = ens["n_members"]
            hidden_dim = e3nn.Irreps(self.config["hidden_irreps"]).filter("0e").dim
            return MaceReadout(
                num_interactions=self.config["num_interactions"],
                hidden_dim=hidden_dim,
                MLP_irreps=self.config["MLP_irreps"],
                n_shallow_ensemble=n_shallow,
                dtype=self.config["readout_dtype"],
            )
        return super().build_readout(head_config, is_feature_fn, only_use_n_layers)
```

Downstream `build_energy_model`, `build_energy_derivative_model`, `build_feature_model`, `build_property_heads`, `build_corrections` are all inherited verbatim. Wrapping by `ShallowEnsembleModel`, `EnergyDerivativeModel`, MD stepping, ASECalculator, and BAL all work through the same paths as every other apax model.

### 3.8 `MaceModelConfig`

Added to the Pydantic discriminated union in `apax/config/model_config.py`:

```python
class MaceModelConfig(BaseModelConfig):
    name: Literal["mace"] = "mace"
    r_max: PositiveFloat = 5.0
    num_bessel: PositiveInt = 8
    num_polynomial_cutoff: PositiveInt = 5
    max_ell: PositiveInt = 3
    hidden_irreps: str = "128x0e + 128x1o"
    num_interactions: PositiveInt = 2
    correlation: PositiveInt = 3
    interaction_cls: Literal[
        "RealAgnostic", "RealAgnosticResidual",
        "RealAgnosticDensity", "RealAgnosticDensityResidual",
    ] = "RealAgnosticResidual"
    use_cueq: bool = False
    readout_kind: Literal["mace", "standard"] = "mace"
    MLP_irreps: str = "16x0e"
```

**Explicitly not on `MaceModelConfig`:** `pretrained`, `freeze_backbone`, `unfreeze_backbone_epoch`, `atomic_numbers`, `atomic_energies`, `num_elements`, `scale`, `shift`. Fine-tuning uses the existing top-level `Config.transfer_learning: TransferLearningConfig` (§5). Per-element scale/shift (including atomic-energy references) lives in the `PerElementScaleShift` params in the orbax checkpoint, not in the YAML config.

## 4. Converter

### 4.1 Converted checkpoint format — identical to apax training output

```
<dst>/
├── config.yaml                 # full apax Config dumped via Config.dump_config()
├── best/                       # orbax CheckpointManager dir (step 0)
│   └── …                       # orbax internal layout
└── converter_metadata.json     # source name, resolved path, torch-mace version,
                                #   sha256, head selected, conversion date
                                #   (optional / purely provenance, never loaded)
```

`config.yaml` validates against the pydantic `Config` schema. The orbax checkpoint stores `{"model": {"params": params}, "epoch": 0}` — the shape that `load_state` already restores. `restore_parameters(dst)` returns `(Config, params)`; every apax consumer uses this.

Torch is never imported at runtime. The converter is the only module that imports `torch` or `mace`.

### 4.2 Converter CLI

```python
# apax/cli/convert_mace.py
@app.command("convert-mace")
def convert_mace(
    source: str,                       # canonical name or .model path
    dst: Path,                         # output directory
    head: str = "default",             # head selection for multi-head models
    family: str = "mace_mp",           # initial scope; mace_off/mace_anicc deferred
):
    try:
        import torch  # noqa: F401
        import mace   # noqa: F401
    except ImportError as e:
        raise typer.BadParameter(
            "Converting MACE foundation models requires torch and mace-torch. "
            "Install them with `uv sync --group mace-convert --extra mace`. "
            f"Missing module: {e.name}"
        )
    from apax.transfer_learning.mace_foundation import run_conversion
    run_conversion(source, dst, head=head, family=family)
```

Upstream resolution path — `source` is treated as:

- A canonical name (validated against `mace.calculators.foundations_models.mace_mp_names`), resolved via `mace_mp(source, return_raw_model=True, default_dtype="float64", device="cpu")`. Upstream handles bundled `mace-mpa-0-medium.model` included in the `mace-torch` package → `~/.cache/mace/` → GitHub-releases download, in that order.
- A path to a local `.model` file, loaded directly via `torch.load`.

### 4.3 `run_conversion` algorithm

```python
def run_conversion(source, dst, *, head="default", family="mace_mp") -> None:
    # 1. Load torch foundation model + resolve source path
    torch_model, resolved_path = _load_torch_foundation_model(source, family=family)

    # 2. Extract architecture hyperparameters → MaceModelConfig fields
    mace_cfg_fields = _extract_config_from_torch(torch_model, head=head)
    mace_cfg = MaceModelConfig(**mace_cfg_fields)

    # 3. Synthesize a full Config with placeholder training-only fields
    full_cfg = _synthesize_full_config(mace_cfg, dst)
    #   data: DataConfig(directory=str(dst), experiment="converted",
    #                    data_path="placeholder.extxyz")
    #   loss: [LossConfig(name="energy")]
    #   optimizer: OptimizerConfig()         # all defaults
    #   n_epochs: 1
    full_cfg = Config.model_validate(full_cfg.model_dump())   # re-validate

    # 4. Build the same model the trainer / ASECalculator would build
    Builder = full_cfg.model.get_builder()
    builder = Builder(full_cfg.model.model_dump(), n_species=119)
    energy_model = builder.build_energy_model()
    params_template = energy_model.init(rng, R_dummy, Z_dummy, neigh_dummy, box_dummy, offsets_dummy)

    # 5. Map torch state_dict → params pytree (see 4.4)
    torch_state = {k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()}
    params = _map_state_to_pytree(
        torch_state, params_template, mace_cfg, head=head,
        torch_atomic_numbers=tuple(torch_model.atomic_numbers.cpu().numpy().tolist()),
    )
    _validate_no_nan(params)

    # 6. Persist via the stock apax writers
    dst.mkdir(parents=True, exist_ok=True)
    full_cfg.dump_config(dst)                              # writes dst/config.yaml
    _write_orbax_checkpoint(dst / "best", params, epoch=0) # orbax CheckpointManager

    # 7. Provenance
    (dst / "converter_metadata.json").write_text(json.dumps({
        "source": str(source),
        "source_resolved_path": str(resolved_path) if resolved_path else None,
        "source_sha256": _sha256(resolved_path) if resolved_path else None,
        "torch_mace_version": _torch_mace_version(),
        "apax_version": _apax_version(),
        "head_selected": head,
        "family": family,
        "converted_at": datetime.now(tz=timezone.utc).isoformat(),
    }, indent=2))
```

`_write_orbax_checkpoint` uses `orbax.checkpoint.CheckpointManager` with `args=ocp.args.StandardSave({"model": {"params": params}, "epoch": 0})` — the inverse of what `load_state` does.

### 4.4 Torch → apax parameter mapping

With the §3.4/3.4a forward pass, every torch `state_dict` weight maps **directly** to an apax leaf — no projections, slicing, or per-element averaging. The only transformations are zero-padding along the element axis (89 → 119) and torch ↔ e3nn-jax row/column conventions where they differ.

- **`node_embedding.linear.weight`** `(N_torch_elts * hidden_scalar,)` → apax `(119, hidden_scalar)` with rows scattered at `torch_atomic_numbers` indices.
- **`interactions.{k}.linear_up.weight`** → apax `InteractionBlock.linear_up.kernel`, full `node_feats_irreps × node_feats_irreps` weight (no slicing — `linear_up` is in-place mix).
- **`interactions.{k}.linear.weight`** → apax `InteractionBlock.linear.kernel`, full `irreps_mid × target_irreps` (`128x0e+128x1o+128x2e+128x3o → 128x0e+128x1o+128x2e+128x3o`). Direct copy.
- **`interactions.{k}.skip_tp.weight`** → apax `InteractionBlock.skip_tp.kernel`, full `(node_feats_irreps × node_attrs_irreps → hidden_irreps)` per-element FCTP weight `(num_elements, n_paths)` zero-padded along the element axis. Direct copy of weights at `torch_atomic_numbers` rows.
- **`interactions.{k}.conv_tp_weights.layer{0..3}.weight`** — per-layer MLP weights. Our `e3nn.flax.MultiLayerPerceptron` stores them as `kernel_{j}`; direct assignment of `(in, out)` tensors. Output dim of `layer3` is the FULL `conv_tp.weight_numel` (e.g. `512` for `target_irreps="128x0e+128x1o+128x2e+128x3o"` with `max_ell=3`).
- **`products.{k}.symmetric_contractions.contractions.{c}.weights_max` + `.weights.{m}`** — torch stores per-correlation-order weights separately; cuequivariance fuses them into `(num_elements, weight_numel, mul)`. The cuex `SymmetricContraction` descriptor must be built from `node_feats_irreps=interaction_irreps` (the multi-irrep input) and `target_irreps=this_layer_hidden`. Use `/Users/fzills/tools/mace-jax/mace_jax/adapters/cuequivariance/symmetric_contraction.py` as the fusion rule. All correlation orders for all `l` channels are mapped — no truncation.
- **`products.{k}.linear.weight`** → apax `ProductBlock.linear.kernel`, `target_irreps × target_irreps`. Direct copy.
- **`readouts.0.linear.weight`** `(hidden_scalar,)` → `MaceReadout.readout_0.linear.kernel` with shape `(hidden_scalar, 1)`.
- **`readouts.-1.linear_1.weight`** / **`linear_2.weight`** → `MaceReadout.readout_{N-1}.linear_1.kernel`, `linear_2.kernel` of shapes `(hidden_scalar, MLP_dim)` and `(MLP_dim, 1)`.
- **`scale_shift.scale` + `scale_shift.shift` + `atomic_energies_fn.atomic_energies`** → folded into `PerElementScaleShift.scale_per_element (119, 1)` (constant at `global_scale`) and `shift_per_element (119, 1)` (zero-padded `atomic_energies[Z] + global_shift`, at `torch_atomic_numbers` indices).
- **Radial basis parameters** — `radial_embedding.bessel_fn.bessel_weights`, `.prefactor`, `cutoff_fn.p`, `cutoff_fn.r_max` — mapped into `BesselBasis` / `PolynomialCutoff` params.
- **`normalize2mom` activation constant** — extracted at conversion time via the mace-jax `_extract_norm_consts` recipe, stored as a non-trainable `constants` variable in the pytree (or as the `silu_normalization` field on `NonLinearReadoutBlock` — see §3.5).

Invariants:
- `_validate_no_nan(params)` passes (every float leaf finite).
- For every torch `state_dict` key, there exists an apax leaf with the **same total `numel`** (modulo the 89→119 element-axis padding). The shape-coverage test (P3.5b.6) enforces this.

### 4.5 Multi-head foundation models

MACE-MPA is a multi-head pretrained model: a single shared backbone with `num_heads > 1` parallel readout stacks. apax `MaceReadout` has one readout stack — we **select a single head at conversion time** and drop the others. `_extract_config_from_torch` validates `head` against `model.heads`, raising `ValueError` with the available names if the requested head is unknown. `_map_state_to_pytree` restricts the `readouts.*` mapping to the selected head's slice.

`converter_metadata.json` records which head was selected.

## 5. Fine-tuning flow

Driven entirely by YAML, using the existing `Config.transfer_learning: TransferLearningConfig` (already in `apax/config/train_config.py`):

```yaml
# configs/finetune_mace_mp_small.yaml
n_epochs: 50

data:
  directory: ./runs/
  experiment: finetune_mp_small
  data_path: ./my_dataset.extxyz
  n_train: 800
  n_valid: 100

model:
  name: mace
  r_max: 6.0
  num_bessel: 10
  num_polynomial_cutoff: 5
  max_ell: 3
  hidden_irreps: 128x0e
  num_interactions: 2
  correlation: 3
  interaction_cls: RealAgnosticResidual
  readout_kind: standard              # swap MACE readout for apax's MLP head
  nn: [256, 256]
  ensemble:
    kind: shallow
    n_members: 8
    force_variance: true
  property_heads:
    - { name: charges, mode: l0, aggregation: none, nn: [64, 32] }

transfer_learning:
  base_model_checkpoint: ./mace-mp-0-small.apax      # the converted directory
  reset_layers: [MaceReadout_0]                       # drop the MACE readout params;
                                                      # start fresh with AtomisticReadout
  freeze_layers: [MaceRepresentation_0]               # optional: freeze descriptor

loss:
  - { name: energy, loss_type: crps }
  - { name: forces, loss_type: crps }
  - { name: charges, loss_type: weighted_mse }
```

`transfer_parameters(state, ckpt_config)` loads `base_model_checkpoint` via the existing `load_params`, then `black_list_param_transfer` applies the reset-list filter. Freezing is enforced via the optimizer (set per-group LR=0 through `OptimizerConfig` if needed; a proper `freeze_layers` plumbing is a separate refactor outside this spec).

### 5.1 Ensembles on top of MACE

- **Shallow:** `ensemble.kind=shallow, n_members=N`. `MaceBuilder.build_readout` passes `n_shallow_ensemble=N` to `MaceReadout` — its final `LinearReadoutBlock` / `NonLinearReadoutBlock` emits `(n_atoms, N)` scalars. `EnergyModel` auto-detects via `E_i.shape[1] > 1`; `ShallowEnsembleModel` wraps.
- **Full:** `ensemble.kind=full, n_members=N` stacks N independent MACE models via `stack_parameters`. No MACE-specific code.

### 5.2 Property heads

`PropertyHead` (see `apax/layers/properties.py`) consumes the scalar features `g` from `MaceRepresentation` (the per-layer-concatenated scalar tensor) and, when present, auto-emits mean predictions and uncertainty for each head. This works identically for GMNN and MACE backbones.

### 5.3 Zero-shot evaluation

Downstream path for "use the foundation as-is with no retraining": user runs `apax convert-mace <name> ./out/`, then `apax.md.ase_calc.ASECalculator("./out/")` — the same entry point any other apax-trained model uses. Parity vs torch-mace's `MACECalculator` is the P3 exit criterion.

## 6. Performance

(unchanged from initial spec — carried forward)

### 6.1 Baked-in decisions

- **Fixed-shape batching**: reuse apax's padded-neighbor-list strategy.
- **Single JIT boundary**: MACE lives inside `EnergyDerivativeModel`'s jit; no separate compilation surface.
- **cueq dispatch internal to blocks**: `use_cueq=True` toggles kernel choice in `InteractionBlock` / `ProductBlock`; forward semantics identical.
- **Precision policy**: fp64 for `scale_shift` / reductions (reuse `fp64_sum`), fp32 for tensor products; configurable via existing `descriptor_dtype` / `readout_dtype` / `scale_shift_dtype` knobs.
- **Lazy cueq descriptor is forced at init**: cuequivariance constructs TP descriptors on first use; we force construction in `setup()` so it's part of the frozen graphdef.

### 6.2 Benchmark harness

Lives in `benchmarks/mace/`, not run under pytest. Entry points:

```
python -m benchmarks.mace.bench_inference --model mace-mp-0-medium --n-atoms 512
python -m benchmarks.mace.bench_training  --model mace-mp-0-medium --batch 16
python -m benchmarks.mace.bench_md        --model mace-mpa-medium  --n-atoms 1024
python -m benchmarks.mace.report
```

### 6.3 Success thresholds

| Metric | Threshold |
|---|---|
| Inference latency | ≤ 1.2× upstream mace-jax (same XLA) |
| Training throughput | ≤ 1.3× upstream mace-jax |
| GPU with `use_cueq=True` | ≤ 1.1× cueq-accelerated mace-jax |
| Energy parity vs torch-mace | rtol 1e-5, atol 1e-6 |
| Force parity vs torch-mace | rtol 1e-4, atol 1e-5 |
| Existing GMNN / So3krates benchmarks | no regression |

## 7. Testing

### 7.1 Always-on tests (no torch)

- `test_mace_descriptor.py`: `MaceRepresentation(dr_vec, Z, idx)` returns `(n_atoms, n_features)` with correct dtype; responds to `apply_mask`; works on a batched input via `jax.vmap`.
- `test_mace_blocks.py`: `LinearNodeEmbedding`, `InteractionBlock`, `ProductBlock`, `LinearReadoutBlock`, `NonLinearReadoutBlock` — shape + init tests, no parity.
- `test_mace_readout.py`: `MaceReadout` returns `(1,)` or `(n_members,)` as configured; gradient flows; `jax.vmap(readout)(g)` gives `(n_atoms, n_out)`.
- `test_mace_builder.py`: `MaceBuilder` produces an `EnergyModel` that runs end-to-end with random params on a minimal config, including `readout_kind="standard"` → `AtomisticReadout` fallback.
- `test_mace_shallow_ensemble.py`: `ShallowEnsembleModel(EnergyModel(...))` with MACE backbone emits energies of shape `(n_members,)` and per-atom force variance.

### 7.2 Parity tests (opt-in, require torch + mace-torch)

Gated by `@pytest.mark.mace_parity`. Not run by default:

- `test_convert.py`: converts `small` (MACE-MP-0 small) and `medium-mpa-0`; asserts `<dst>/config.yaml` validates against `Config`; asserts `restore_parameters(dst)` returns `(Config, params)` without error; asserts `_validate_no_nan` passes post-conversion.
- `test_mace_parity.py`: for `small` (and later `medium`, `medium-mpa-0`), converts, loads via `ASECalculator(<dst>)` — unmodified ASECalculator — evaluates on an isolated water molecule and a small periodic SiO₂ cell; compares energies, forces, stresses against upstream `mace_mp(name)` `MACECalculator`. Thresholds from §6.3.
- Autodiff-consistency: apax forces match finite-difference ∇E on the same system within 1e-3. Independent of torch.
- CI: `workflow_dispatch` job installs the dev group, runs `uv run pytest -m mace_parity`.

### 7.3 CI matrix

```
Always-on (every PR):
  uv run pytest -m "not slow and not mace_parity"
MACE parity (manual / nightly):
  uv sync --group mace-convert --extra mace
  uv run pytest -m mace_parity
Benchmarks (nightly, GPU runner):
  uv run python -m benchmarks.mace.report
```

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CG / normalization drift (cueq vs torch) | Medium | Silent parity failure | Pin `group=O3_e3nn`, `use_reduced_cg=True`; NaN-leaf post-check at convert time |
| `cuequivariance-jax` pulls hard CUDA dep | Low | Install breaks on CPU-only | Audit wheels; pin known-good version |
| Torch state_dict → apax params layout drift | High (first pass) | Conversion fails NaN check | `_validate_no_nan` covers every float leaf; mace-jax reference implementations give per-module mappings |
| Synthesized `config.yaml` placeholder fields feel weird to users | Low | UX friction | `converter_metadata.json` documents origin; fields never break at load time |
| Per-layer feature dim mismatch on init-vs-load | Medium | Head shape mismatch | Orbax restore hits the template pytree built by the same `MaceBuilder` code path; shapes are forced consistent |
| `reset_layers` matching the MACE readout path name changes | Low | Transfer-learning test fails | Pin the module name in `MaceBuilder.build_readout`; documented in §5 YAML |
| fp32 TP loses accuracy at high `max_ell` | Low | Force error grows with L | Keep reductions in fp64; benchmark fp32-full vs mixed |
| Foundation uses unsupported interaction variant (Density) | Medium | Parity fails for specific models | P3 ships only the `RealAgnosticResidual` variant (covers MACE-MP-0 small/medium, `medium-0b*`, `large*`). Density-variant models deferred |
| torch pickle format changes | Low | Converter breaks on new torch | Pin converter's torch-mace range in dep group |

## 9. Phasing

| Phase | Goal | Exit |
|---|---|---|
| P0 Plumbing | `MaceModelConfig`, `MaceBuilder.build_descriptor`, skeleton `MaceRepresentation` (random features) | `uv sync --extra mace` succeeds; dummy build runs |
| P1 Native forward | Port InteractionBlock, ProductBlock, LinearNodeEmbedding, cutoffs, radial, SH in linen | Random-weight forward shape-correct |
| P2 cueq dispatch | `use_cueq=True` path | Identical outputs with/without cueq |
| P3 Converter + readout | `LinearReadoutBlock`, `NonLinearReadoutBlock`, `MaceReadout`, `MaceBuilder.build_readout`; `apax convert-mace` produces `config.yaml` + `best/`; `ASECalculator(<dst>)` works unchanged; parity vs `mace_mp("small")` on water | Energy rtol 1e-4, force rtol 1e-3 on small foundation |
| P4 Fine-tune | Config-driven fine-tuning via existing `TransferLearningConfig`; shallow ensemble on MACE backbone | Val loss drops on a small benchmark dataset; uncertainty calibrates |
| P5 MD + ASE | End-to-end tests with existing infrastructure (no ASECalculator changes needed) | jax-md NVT on 256-atom water; ASECalculator parity vs training |
| P6 Benchmarks | Perf harness, optimization pass if below thresholds | Report in `benchmarks/mace/REPORT.md` |

**Revision note:** previous spec's P3 ("Converter + loader") and separate P3.5 ("ASECalc wiring") collapse into this P3. P4 shrinks — the recipe is YAML config, no new runtime code.

## 10. Net LoC estimate (revised)

| Component | LoC |
|---|---|
| `MaceRepresentation` + descriptor blocks | ~1100 |
| `LinearReadoutBlock`, `NonLinearReadoutBlock` | ~80 |
| `MaceReadout` | ~60 |
| cueq dispatch + irreps utils | ~250 |
| `MaceModelConfig` + `MaceBuilder` | ~100 |
| Weight converter (`run_conversion` + helpers) | ~500 |
| CLI subcommand | ~60 |
| Tests (always-on + parity) | ~600 |
| Benchmarks | ~300 |
| **Total** | **~3050** |

All additive. Net delta vs previous spec: −150 LoC (no `MaceFoundationEnergyModel`, no `load_mace_foundation`, no ASECalculator branch), +140 (`MaceReadout` + readout blocks).

## 11. Open items

None blocking this spec. Items deferred to implementation:

- Exact torch-state-dict → linen-pytree parameter-path map (enumerated at P3 time against `small` first; widened to `medium` and `medium-mpa-0` iteratively).
- Shipping pre-converted `.apax/` directories on Hugging Face for MACE-MP and MACE-MPA, or leaving conversion to users. Preferred: initial release asks users to run `apax convert-mace <name>` once (upstream handles the download/cache); HF artifacts are a follow-up.
- Whether to add a `apax mace download <name>` command that fetches pre-converted directories from HF — nice to have, deferred.
- Whether `TransferLearningConfig.freeze_layers` should be plumbed into the optimizer more forcefully than "set per-group LR to 0" — out of scope for this spec; filed against P4.
