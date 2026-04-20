# MACE Foundation Model Integration — Design Spec

**Date:** 2026-04-20
**Status:** Approved — ready for implementation plan
**Scope:** Native MACE descriptor in apax + foundation-model loading (MACE-MP, MACE-MPA) + fine-tuning with apax's shallow-ensemble & property heads + JAX-MD integration

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
- `apax convert-mace <source> <dst.apax/>` CLI where `<source>` is either a canonical MACE foundation name (`medium`, `medium-mpa-0`, …) resolved via upstream `mace.calculators.foundations_models.mace_mp(return_raw_model=True)` — which handles bundled-local → cache → download — or a local `.model` path. Torch is imported lazily via try/except.
- `MaceRepresentation` is a Flax linen `nn.Module` matching apax's existing descriptor contract.
- Parity with upstream torch-mace on energies (rtol 1e-5) and forces (rtol 1e-4) for MACE-MP-0 and MACE-MPA variants.
- Shallow-ensemble fine-tuning with `freeze_backbone: true` works through a YAML toggle; no new training-loop code.
- `apax md` and `apax.md.ase_calc` work end-to-end with MACE; no changes to those modules.
- `PropertyHead` reuses MACE's scalar features to train charges, dipoles, polarizabilities, stresses, and arbitrary per-atom fields; ensemble uncertainty is automatic.
- CPU inference within 1.2× of upstream mace-jax; training within 1.3×; GPU with `use_cueq=True` within 1.1× of cuequivariance-accelerated mace-jax.

### Non-goals (deferred follow-ups)

- MACE-OFF, MACE-ANICC, MACE-OMOL foundation models.
- **Training** multi-head MACE models (`num_heads > 1`) in apax. We *consume* multi-head foundation weights by selecting a single head at load time (see §4.5); what's out of scope is producing new multi-head models during fine-tuning.
- Equivariant property heads that consume MACE's l≥1 irrep features directly.
- LAMMPS MLIAP export.
- Changes to distributed-training infrastructure.
- Any port or upstreaming of mace-jax code into apax.

## 3. Architecture

### 3.1 Dependency strategy

No hard deps added. One new optional extra:

```toml
[project.optional-dependencies]
mace = [
    "e3nn-jax>=0.21.0",
    "cuequivariance-jax>=0.9.1",
    "cuequivariance>=0.9.1",
]
```

`torch` and `mace-torch` are never in `pyproject.toml`. The CLI converter does a runtime `try: import torch, mace` and raises a typer error telling the user how to install them if they invoke the conversion step.

### 3.2 New module layout (all additive)

```
apax/
├── layers/descriptor/
│   ├── mace.py                     # MaceRepresentation
│   └── mace_blocks.py              # InteractionBlock, ProductBlock, LinearNodeEmbedding, Readout
├── nn/
│   ├── builder.py                  # + MaceBuilder
│   └── mace_foundation_model.py    # MaceFoundationEnergyModel (parity path only)
├── config/
│   └── model_config.py             # + MaceModelConfig
├── cli/
│   └── convert.py                  # apax convert-mace subcommand
├── transfer_learning/
│   └── mace_foundation.py          # load_mace_foundation(src) → (params, cfg)
├── utils/
│   └── mace_irreps.py              # Irreps helpers, cueq dispatch utilities
└── tests/mace/
    ├── test_descriptor_shape.py
    ├── test_builder_wiring.py
    ├── test_shallow_ensemble.py
    └── test_convert_parity.py      # gated by @pytest.mark.mace_parity
```

Zero modifications to `EnergyModel`, `AtomisticReadout`, `ShallowEnsembleModel`, `PropertyHead`, the base `ModelBuilder`, `trainer.py`, `simulate.py`, or `ase_calc.py`.

### 3.3 Descriptor contract (mirrored from existing apax descriptors)

All existing descriptors follow:

```python
def __call__(self, dr_vec, Z, idx) -> Array:  # (n_atoms, n_features)
```

Distances are computed in `EnergyModel` before the descriptor is called. Output is always plain scalar features — never IrrepsArray — so `AtomisticReadout` receives its expected shape.

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
    num_elements: int = 119
    use_cueq: bool = False
    apply_mask: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, dr_vec, Z, idx) -> Array:
        ...
        return features   # (n_atoms, n_features)
```

Internally: node embedding → N× (interaction → product) → concatenate scalar (l=0) features from every layer → apply node mask → return. Non-scalar irreps are discarded at the apax boundary because `AtomisticReadout` expects scalars.

### 3.5 `MaceFoundationEnergyModel` (parity path only)

A separate module used exclusively for (a) parity tests versus torch-mace and (b) zero-shot inference from a pretrained model without a new head. It includes MACE's per-layer internal readouts and the per-element atomic-energy reference — i.e., it reproduces the full upstream forward pass.

```python
class MaceFoundationEnergyModel(nn.Module):
    representation: MaceRepresentation  # internal: exposes per-layer node_feats
    readouts: list[Readout]
    atomic_energies: jnp.ndarray
    init_box: np.ndarray
```

Fine-tuning, shallow ensemble, property heads, MD, and ASE paths all use the standard `EnergyModel(MaceRepresentation, AtomisticReadout, PerElementScaleShift)` — not `MaceFoundationEnergyModel`.

### 3.6 `MaceBuilder`

Mirrors `GMNNBuilder` and `So3kratesBuilder`:

```python
class MaceBuilder(ModelBuilder):
    def build_descriptor(self, apply_mask):
        return MaceRepresentation(...)   # all fields from self.config
```

Inherits `build_readout`, `build_scale_shift`, `build_property_heads`, `build_corrections`, `build_energy_model`, `build_energy_derivative_model`, `build_feature_model` from `ModelBuilder` with no changes.

### 3.7 `MaceModelConfig`

Added to the Pydantic discriminated union in `apax/config/model_config.py`:

```python
class MaceModelConfig(BaseModelConfig):
    name: Literal["mace"] = "mace"
    r_max: float = 5.0
    num_bessel: int = 8
    num_polynomial_cutoff: int = 5
    max_ell: int = 3
    hidden_irreps: str = "128x0e + 128x1o"
    num_interactions: int = 2
    correlation: int = 3
    interaction_cls: Literal[...] = "RealAgnosticResidual"
    use_cueq: bool = False
    pretrained: str | Path | None = None        # name or path to apax-native dir
    freeze_backbone: bool = False
    unfreeze_backbone_epoch: int | None = None
```

## 4. Converter and foundation-model loader

### 4.1 Converted checkpoint format

```
<name>.apax/
├── config.json        # MACE architecture hyperparameters (feeds MaceModelConfig)
├── params.msgpack     # flax.serialization.to_bytes(pytree)
└── metadata.json      # source path, torch-mace version, sha256, convert date, apax version
```

Loading is pure JAX — the runtime reads msgpack + JSON and merges the params into a fresh `MaceRepresentation` (or `MaceFoundationEnergyModel`) pytree. Torch is never imported at runtime.

### 4.2 Converter CLI

```python
# apax/cli/convert_mace.py
@app.command("convert-mace")
def convert_mace(
    source: str,                  # canonical name OR .model path
    dst: Path,
    head: str = "mp",
    family: str = "mace_mp",      # scope initial: mace_mp (covers MP-0/0b/0b2/0b3 + MPA)
):
    try:
        import torch
        import mace
    except ImportError as e:
        raise typer.BadParameter(
            "Converting MACE foundation models requires torch and mace-torch. "
            f"Missing: {e.name}"
        )
    _convert(source, dst, head=head, family=family)
```

Upstream resolution path — `source` is treated as:
- A canonical name (validated against `mace.calculators.foundations_models.mace_mp_names`), resolved via `mace_mp(source, return_raw_model=True)`. Upstream handles bundled `mace-mpa-0-medium.model` included in the `mace-torch` package → `~/.cache/mace/` → GitHub-releases download, in that order.
- A path to a local `.model` file, loaded directly via `torch.load`.

Conversion steps:

1. Load torch model (via `mace_mp` or `torch.load`).
2. Validate `use_reduced_cg=True`, `group=O3_e3nn` (raise otherwise — we don't support the full-CG path initially).
3. Walk torch state dict; for each parameter, map its name to the linen pytree path and convert dtype (fp64 for scale/shift/reference, fp32 for tensor-product weights by default).
4. Extract the `normalize2mom` activation constant from torch's e3nn into a non-trainable param.
5. Validate: after conversion, scan for any NaN leaves — raise listing the path of the first failure.
6. Write `params.msgpack` via `flax.serialization.to_bytes`; write `config.json` and `metadata.json` (records the canonical name, resolved on-disk path, sha256, torch-mace version, head selected).

### 4.3 `load_mace_foundation`

```python
def load_mace_foundation(
    source: str | Path,
) -> tuple[PyTree, MaceModelConfig]:
    """Load a converted MACE foundation model.

    ``source`` may be:
      - A directory path pointing to a converted .apax/ dir.
      - A short name like 'mace-mp-0-medium' — resolved via huggingface-hub (optional).
    """
```

Short-name resolution uses `huggingface-hub` if available, falling back to a clear error if the name is unknown. The HF-based flow is optional; local paths always work.

### 4.4 Supported interactions at P3

Initial scope: `RealAgnosticResidualInteractionBlock` (what MACE-MP and MACE-MPA use). Other variants are supported at the module level but not validated for parity until later phases.

### 4.5 Multi-head foundation models (MACE-MPA)

MACE-MPA is a **multi-head** pretrained model: a single shared backbone with `num_heads > 1` parallel energy readout stacks, each trained on a different dataset (materials, organic, etc.). At inference time the upstream user picks which head to evaluate.

apax's `EnergyModel` has one energy readout. To consume MACE-MPA we **select a single head at conversion time** and drop the others. Default: the MP head. The `convert-mace` CLI exposes a flag:

```bash
apax convert-mace mace-mpa-medium.model ./out.apax/ --head mp     # default
apax convert-mace mace-mpa-medium.model ./out.apax/ --head pbe    # alternative
```

`metadata.json` records which head was selected. The converted `.apax/` directory thereafter looks identical to a single-head MACE model; apax never sees the multi-head structure at runtime.

Training a *new* multi-head MACE inside apax is a non-goal. Users who want to retain the multi-head backbone must convert each head separately or use upstream mace-jax.

## 5. Fine-tuning flow

Driven entirely by YAML. Example:

```yaml
model:
  name: mace
  pretrained: mace-mp-0-medium
  freeze_backbone: true
  ensemble:
    kind: shallow
    n_members: 8
    force_variance: true
  property_heads:
    - name: charges
      mode: l0
      aggregation: none
      nn: [64, 32]
    - name: dipole
      mode: l1
      aggregation: sum
      nn: [64, 32]

loss:
  - { name: energy,  loss_type: crps }
  - { name: forces,  loss_type: crps }
  - { name: stress,  loss_type: weighted_mse }
  - { name: charges, loss_type: weighted_mse }
```

### 5.1 Freezing

Uses `apax.transfer_learning.parameter_transfer.black_list_param_transfer` with a new parameter-path predicate targeting `MaceRepresentation`. If `unfreeze_backbone_epoch` is set, a training-loop callback re-enables gradients for those paths at the specified epoch.

### 5.2 Shallow-ensemble head

When `ensemble.kind == "shallow"`, `ModelBuilder.build_readout` already constructs an `AtomisticReadout` with `n_shallow_ensemble=n_members`. `build_energy_derivative_model` wraps the resulting `EnergyModel` in `ShallowEnsembleModel`, which computes force variance via `jacrev`. No new code is needed for the MACE path.

### 5.3 Property heads

`PropertyHead` (see `apax/layers/properties.py:44-134`) consumes the scalar features `g` from `MaceRepresentation` and, when the feature tensor has ensemble dim > 1, automatically emits both the mean prediction and its ensemble uncertainty. Modes: `l0` (scalar), `l1` (vector from position relative to COM), `symmetric_l2`, `symmetric_traceless_l2`. Aggregations: `none`, `sum`, `mean`.

## 6. Performance

### 6.1 Baked-in decisions

- **Fixed-shape batching**: reuse apax's padded-neighbor-list strategy (lock to `n_max_neighbors`). MACE is shape-sensitive; shape polymorphism causes XLA recompiles.
- **Single JIT boundary**: MACE lives inside the existing `EnergyDerivativeModel` jit; no separate compilation surface.
- **cueq dispatch internal to blocks**: `use_cueq=True` toggles the internal kernel choice in `InteractionBlock` / `ProductBlock` / symmetric-contraction; forward semantics identical.
- **Neighbor lists**: vesin for ASE calc, jax-md `partition` for MD — unchanged from existing apax. matscipy is not introduced.
- **Precision policy**: fp64 for `scale_shift` / reductions (reuse `fp64_sum`), fp32 for tensor products; configurable via existing `descriptor_dtype` / `readout_dtype` / `scale_shift_dtype` knobs.
- **Lazy cueq descriptor is forced at init**: cuequivariance constructs TP descriptors on first use; we force construction in `setup()` so it's part of the frozen graphdef.

### 6.2 Benchmark harness

Lives in `benchmarks/mace/`, not run under pytest. Script entry points:

```
python -m benchmarks.mace.bench_inference --model mace-mp-0-medium --n-atoms 512
python -m benchmarks.mace.bench_training  --model mace-mp-0-medium --batch 16
python -m benchmarks.mace.bench_md        --model mace-mpa-medium  --n-atoms 1024
python -m benchmarks.mace.report
```

Target matrix: MACE-MP-0 small/medium at 64 and 512 atoms, MACE-MPA medium at 512 atoms, with/without `use_cueq`, CPU + GPU. Compared against upstream mace-jax at equal config.

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

- `test_descriptor_shape.py`: `MaceRepresentation(dr_vec, Z, idx)` returns `(n_atoms, n_features)` with correct dtype; responds to `apply_mask`; works on a batched input via `jax.vmap`.
- `test_builder_wiring.py`: `MaceBuilder` produces an `EnergyModel` that runs end-to-end with random params on a minimal config.
- `test_shallow_ensemble.py`: `ShallowEnsembleModel` around MACE emits energies of shape `(n_members,)` and per-atom force variance.

### 7.2 Parity tests (opt-in, require torch + mace-torch)

Gated by `@pytest.mark.mace_parity`. Not run by default:

- For `medium-mpa-0` (default MACE-MPA-0, bundled with mace-torch) and `medium` (MACE-MP-0 medium, downloaded on first run from ACEsuit/mace-mp GitHub releases): convert, load, evaluate on an isolated water molecule and a small periodic SiO₂ cell; compare energies, forces, and stresses against the upstream `MACECalculator` ASE wrapper (not just the raw torch module — the ASE wrapper is what downstream users see).
- Additional autodiff-consistency test: apax forces match finite-difference ∇E on the same system within 1e-3. Independent of torch, so catches apax-side autodiff bugs even in environments without torch.
- Dev-only dependency group `mace-convert` (adds `torch`, `mace-torch`); engineers opt-in via `uv sync --group mace-convert --extra mace`. The group is never installed by default.
- CI: `workflow_dispatch` job installs the dev group, runs `uv run pytest -m mace_parity`. First invocation triggers upstream downloads into the runner cache; re-runs hit the cache.

### 7.3 CI matrix

```
Always-on (every PR):
  uv run pytest -m "not slow and not mace_parity"
MACE parity (manual / nightly):
  # CI job installs torch + mace-torch into the job env, then:
  uv run pytest -m mace_parity
Benchmarks (nightly, GPU runner):
  uv run python -m benchmarks.mace.report
```

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CG / normalization drift (cueq vs torch) | Medium | Silent parity failure | Pin `group=O3_e3nn`, `use_reduced_cg=True`; NaN-leaf post-check at convert time |
| `cuequivariance-jax` pulls hard CUDA dep | Low | Install breaks on CPU-only | Audit wheels; pin known-good version |
| Per-layer feature dim mismatch on load | Medium | Head shape mismatch at init | Builder reads dim from converted `config.json`, not user config |
| Apax fixed-pad clashes with MACE batching | Medium | Recompile churn or wrong energies | Lock apax's policy; assert in InteractionBlock |
| fp32 TP loses accuracy at high `max_ell` | Low | Force error grows with L | Keep reductions in fp64; benchmark fp32-full vs mixed |
| Foundation model uses unsupported feature | Medium | Parity fails for specific models | P3 starts with one concrete model; expand model-by-model |
| torch pickle format changes | Low | Converter breaks on new torch | Pin converter's known-good torch range in its check |

## 9. Phasing

| Phase | Goal | Exit |
|---|---|---|
| P0 Plumbing | `MaceModelConfig`, `MaceBuilder`, skeleton `MaceRepresentation` (random features) | `uv sync --extra mace` succeeds; dummy train runs |
| P1 Native forward | Port InteractionBlock, ProductBlock, LinearNodeEmbedding, cutoffs, radial, SH in linen | Random-weight forward matches mace-jax to fp32 |
| P2 cueq dispatch | `use_cueq=True` path | Identical outputs with/without cueq |
| P3 Converter + loader | `apax convert-mace` CLI; `MaceFoundationEnergyModel` | Parity test passes for mace-mp-0-medium and mace-mpa-medium |
| P4 Fine-tune | `freeze_backbone`, `unfreeze_backbone_epoch`, ensemble + property heads | Val loss drops on a small benchmark dataset |
| P5 MD + ASE | End-to-end tests with existing infrastructure | jax-md NVT on 256-atom water; ASE calc parity vs training |
| P6 Benchmarks | Perf harness, optimization pass if below thresholds | Report in `benchmarks/mace/REPORT.md` |

## 10. Net LoC estimate

| Component | LoC |
|---|---|
| `MaceRepresentation` + blocks | ~1100 |
| `MaceFoundationEnergyModel` | ~150 |
| cueq dispatch + irreps utils | ~250 |
| Config + builder | ~150 |
| Weight converter + loader | ~400 |
| CLI subcommand | ~200 |
| Tests (always-on + parity) | ~600 |
| Benchmarks | ~300 |
| **Total** | **~3200** |

All additive.

## 11. Open items

None blocking this spec. Items deferred to implementation:

- Exact torch-state-dict → linen-pytree parameter-path map (enumerated at P3 time, one key per file for maintainability).
- Whether to ship pre-converted apax checkpoints on Hugging Face for MACE-MP and MACE-MPA, or leave conversion to users. Preferred: initial release asks users to run `apax convert-mace <name>` once (upstream handles the download/cache); we can add pre-converted HF artifacts as a follow-up for users who don't want to install torch even once.
- Whether to add a `apax mace download <name>` command that fetches from HF — nice to have, deferred.
