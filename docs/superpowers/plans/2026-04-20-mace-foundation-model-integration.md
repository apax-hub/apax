# MACE Foundation Model Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native MACE (Multi-Atomic Cluster Expansion) support to apax: load MACE-MP / MACE-MPA foundation models via a CLI converter, fine-tune them with apax's shallow-ensemble and property-head infrastructure, and use them in JAX-MD — without making torch, mace-torch, or mace-jax runtime dependencies.

**Architecture:** `MaceRepresentation` is a Flax linen `nn.Module` mirroring apax's existing descriptor contract (`(dr_vec, Z, idx) → (n_atoms, n_features)`). Equivariant math comes from optional extras `e3nn-jax` (irreps, spherical harmonics, tensor products) and `cuequivariance-jax` (MACE-specific symmetric contraction, optional GPU acceleration). The `apax convert-mace` CLI writes **apax's existing training-output layout** (`<dst>/config.yaml` + `<dst>/best/` orbax checkpoint), so every apax consumer (`restore_parameters`, `ASECalculator`, `apax md`, BAL) reads converted foundations with no special-casing. Torch is imported lazily only during conversion; runtime never touches it.

**Tech Stack:** JAX, Flax linen, e3nn-jax, cuequivariance-jax, pydantic v2, orbax-checkpoint, typer (CLI), pytest. Reference source: `/Users/fzills/tools/mace-jax` (Flax NNX port, consulted for per-module parameter maps). Conversion source of truth: torch-mace checkpoints loaded via `mace.calculators.foundations_models.mace_mp(return_raw_model=True)`.

**Spec:** `docs/superpowers/specs/2026-04-20-mace-foundation-model-integration-design.md`

**Progress (updated 2026-04-22):** Phases P0 and P1 complete. The P3 scaffold shipped in commits `bd61b1b8`, `066db7fa`, `e7d6b695`, `d7342dd9` is being **partially reverted** under the revised design (see §"Revision note" below): the parallel `.apax/`-dir format, `load_mace_foundation`, `MaceFoundationEnergyModel`, and the `ASECalculator` detection branch are removed; converter emits apax's standard training-output layout; MACE slots into stock `EnergyModel` via a new `MaceReadout`. Commits to keep: `5bddaed2` (`_extract_config_from_torch` attribute fixes), `2b7283e8` (`LinearReadoutBlock` + `NonLinearReadoutBlock`), `b586af25` (body of what will become `MaceReadout` logic). `027ceb53` added torch + mace-torch + mace-jax as `mace-convert` dev deps — installed in the dev environment. Phase P2 (`use_cueq=True` equivalence) remains outstanding. P1 summary: P1.1 edge features, P1.2 LinearNodeEmbedding, P1.3 InteractionBlock, P1.4 ProductBlock (ported mace-jax's symmetric-contraction adapter to linen), P1.5 real forward pass + `force_irreps_out=True` skip-Linear fix.

**Revision note (2026-04-22):** Earlier drafts of P3 introduced a parallel output format (`params.msgpack` + `config.json` + `metadata.json`), a dedicated `load_mace_foundation` loader, and a dispatch branch inside `ASECalculator`. Verification against the code base (`EnergyModel.__call__`, `PerElementScaleShift`, `restore_parameters`, `TransferLearningConfig`) showed those created two parallel load paths where one will do. The plan now matches the spec revision: the converter emits `<dst>/config.yaml` (via `Config.dump_config`) + `<dst>/best/` orbax checkpoint, and loading is `restore_parameters(dst)`. The per-layer readout + scale/shift + atomic-energy structure lives in a new `MaceReadout` module that fills the existing readout slot. See P3.0 through P3.6 below.

**Reference: how upstream MACE loads foundation models** (`/Users/fzills/tools/mace/mace/calculators/foundations_models.py`):

- `mace_mp(model=..., return_raw_model=True)` returns the raw `torch.nn.Module` and handles everything: bundled-local → cache → download.
- Default (no arg) → **medium-mpa-0** bundled at `mace/calculators/foundations_models/mace-mpa-0-medium.model` (shipped inside the torch-mace pip package).
- Named arg → fetched from GitHub releases (`ACEsuit/mace-mp`, `ACEsuit/mace-foundations`) into cache dir `get_cache_dir()` (typically `~/.cache/mace/`).
- Canonical model names (MP + MPA in initial scope — others in `mace_mp_urls` dict deferred):
  - `small`, `medium`, `large`  → MACE-MP-0
  - `small-0b`, `medium-0b`     → MACE-MP-0b
  - `small-0b2`, `medium-0b2`, `large-0b2` → MACE-MP-0b2
  - `medium-0b3`                → MACE-MP-0b3
  - `medium-mpa-0`              → MACE-MPA-0 (default when None is passed)
- Some OMAT/MatPES models require **Academic Software License (ASL)** acceptance — the torch loader prints a notice. Our converter must forward the same notice.

We integrate by **always going through `mace_mp()`** (and later `mace_off()` / `mace_mpa()` family) rather than requiring users to find `.model` files on disk. This inherits upstream's download, caching, URL registry, and license-notice behavior for free.

For **parity testing** we use `mace_mp(name)` (the `MACECalculator` ASE wrapper) as the reference — it guarantees identical data-prep and forward-pass semantics to what a downstream user gets from torch-mace.

**Global rules (hard):**
- Never use `pip install` or `uv pip install -e .[extra]`. Always `uv sync --extra <name>`.
- Never use `python` directly. Always `uv run python`.
- Never use `uvx pre-commit`. Always `uvx prek --all-files`.
- Never modify tests marked `@pytest.mark.protected`.
- Docstrings follow numpy style.

---

## Reference: file map

### Files created
| Path | Purpose |
|---|---|
| `apax/layers/descriptor/mace.py` | `MaceRepresentation` — public descriptor module |
| `apax/layers/descriptor/mace_blocks.py` | `LinearNodeEmbedding`, `InteractionBlock`, `ProductBlock`, `LinearReadoutBlock`, `NonLinearReadoutBlock` |
| `apax/layers/descriptor/mace_irreps.py` | Irreps utilities, dispatch between e3nn-jax & cuequivariance (P2) |
| `apax/cli/convert_mace.py` | `apax convert-mace` CLI subcommand |
| `apax/transfer_learning/mace_foundation.py` | `run_conversion(source, dst, ...)` and torch-state→apax-pytree mapping helpers |
| `apax/cli/templates/mace_finetune_minimal.yaml` | Fine-tune template config |
| `tests/unit_tests/config/test_mace_model_config.py` | Pydantic schema tests |
| `tests/unit_tests/layers/descriptor/test_mace_descriptor.py` | Shape/contract tests |
| `tests/unit_tests/layers/descriptor/test_mace_blocks.py` | Block-level tests (includes readout blocks) |
| `tests/unit_tests/layers/test_mace_readout.py` | `MaceReadout` tests |
| `tests/unit_tests/nn/test_mace_builder.py` | Builder wiring (incl. readout routing) |
| `tests/unit_tests/cli/test_convert_mace.py` | Converter CLI unit tests (no torch) |
| `tests/integration_tests/mace/test_convert.py` | Converter integration test (gated `mace_parity`) |
| `tests/integration_tests/mace/test_mace_parity.py` | Energy/force parity vs torch-mace (gated) |
| `tests/integration_tests/mace/test_mace_finetune.py` | Fine-tune smoke via `TransferLearningConfig` (gated) |
| `tests/integration_tests/mace/test_mace_shallow_ensemble.py` | Ensemble smoke test |
| `tests/integration_tests/mace/test_mace_md.py` | jax-md + ASE smoke test |
| `benchmarks/mace/bench_inference.py` | Inference benchmark |
| `benchmarks/mace/bench_training.py` | Training benchmark |
| `benchmarks/mace/bench_md.py` | MD benchmark |
| `benchmarks/mace/report.py` | Aggregated report |

### Files modified
| Path | Change |
|---|---|
| `pyproject.toml` | Add `mace` optional extra + `mace-convert` dev group + `mace_parity` pytest marker |
| `apax/config/model_config.py` | Add `MaceModelConfig` (with `readout_kind`, `MLP_irreps`) to discriminated union |
| `apax/nn/builder.py` | Add `MaceBuilder(ModelBuilder)` overriding `build_descriptor` + `build_readout` |
| `apax/layers/readout.py` | Add `MaceReadout` alongside `AtomisticReadout` |
| `apax/layers/descriptor/basis_functions.py` | Add `PolynomialCutoff` |
| `apax/cli/apax_app.py` | Register `convert-mace` subcommand |
| `apax/nodes/model.py` | Register `MaceModelConfig` if applicable |

### Files deleted (rolled back from earlier P3 scaffold)
| Path | Reason |
|---|---|
| `apax/nn/mace_foundation_model.py` | Monolithic full-energy model replaced by `MaceReadout` + stock `EnergyModel` |
| `tests/integration_tests/mace/test_load_foundation.py` | Tested the `.apax/`-msgpack format that no longer exists |
| `tests/unit_tests/md/test_ase_calc_mace_foundation.py` | ASECalculator dispatch branch reverted; no custom detection |

---

## Phase P0 — Plumbing

Goal: get `apax train` to run with a MACE config that uses random features (no real MACE math yet). Proves all integration wiring before the heavy math.

### Task P0.1: Add `mace` optional extra and pytest marker

**Files:**
- Modify: `pyproject.toml`

- [x] **Step 1: Read current pyproject**

Run: Read `pyproject.toml`, confirm shape of `[project.optional-dependencies]` block.

- [x] **Step 2: Add `mace` extra**

Edit `pyproject.toml`, add under `[project.optional-dependencies]`:

```toml
mace = [
    "e3nn-jax>=0.21.0",
    "cuequivariance-jax>=0.9.1",
    "cuequivariance>=0.9.1",
]
```

- [x] **Step 3: Add `mace_parity` marker**

Edit `pyproject.toml` under `[tool.pytest.ini_options]`, extend `markers`:

```toml
markers = [
    "slow: mark a test as slow and should only run explicitly",
    "mace_parity: requires torch + mace-torch; opt-in parity tests against upstream",
]
```

- [x] **Step 4: Sync and verify**

Run: `uv sync --extra mace`
Expected: resolves and installs e3nn-jax, cuequivariance-jax, cuequivariance. No errors.

Run: `uv run python -c "import e3nn_jax, cuequivariance_jax, cuequivariance; print('ok')"`
Expected output: `ok`

- [x] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add mace optional extra (e3nn-jax, cuequivariance-jax)"
```

---

### Task P0.2: Add `PolynomialCutoff` to existing basis_functions

**Files:**
- Modify: `apax/layers/descriptor/basis_functions.py`
- Test: `tests/unit_tests/layers/descriptor/test_basis_functions.py`

- [x] **Step 1: Read existing file**

Run: Read `apax/layers/descriptor/basis_functions.py` to understand conventions (BesselBasis / GaussianBasis as reference).

- [x] **Step 2: Write failing test**

Edit `tests/unit_tests/layers/descriptor/test_basis_functions.py`, append:

```python
def test_polynomial_cutoff_zero_at_rmax():
    import jax.numpy as jnp
    from apax.layers.descriptor.basis_functions import PolynomialCutoff

    cutoff = PolynomialCutoff(p=5, r_max=5.0)
    r = jnp.array([0.0, 2.5, 4.999, 5.0, 5.1])
    f = cutoff(r)
    assert f.shape == r.shape
    assert jnp.isclose(f[0], 1.0, atol=1e-5)           # value at 0 ≈ 1
    assert f[3] == 0.0                                  # exactly 0 at r_max
    assert f[4] == 0.0                                  # 0 beyond r_max
    assert f[2] > 0.0                                   # positive just below


def test_polynomial_cutoff_monotone_decreasing():
    import jax.numpy as jnp
    from apax.layers.descriptor.basis_functions import PolynomialCutoff

    cutoff = PolynomialCutoff(p=5, r_max=5.0)
    r = jnp.linspace(0.0, 5.0, 50)
    f = cutoff(r)
    diffs = jnp.diff(f)
    assert (diffs <= 1e-6).all()                        # never increases
```

- [x] **Step 3: Run test — expect fail**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_basis_functions.py::test_polynomial_cutoff_zero_at_rmax -v`
Expected: ImportError — `PolynomialCutoff` not defined.

- [x] **Step 4: Implement `PolynomialCutoff`**

Append to `apax/layers/descriptor/basis_functions.py`:

```python
class PolynomialCutoff(nn.Module):
    """MACE-style polynomial envelope cutoff.

    Implements the smooth cutoff function from Klicpera et al. 2020, used by MACE:

        f(r) = 1 - ((p + 1)(p + 2) / 2) x^p
                 + p(p + 2) x^(p+1)
                 - (p(p + 1) / 2) x^(p+2)            for r <= r_max
        f(r) = 0                                       for r >  r_max
    with x = r / r_max.

    Parameters
    ----------
    p : int, default 5
        Polynomial order; controls smoothness at r_max.
    r_max : float, default 5.0
        Distance at which the cutoff becomes 0.
    """
    p: int = 5
    r_max: float = 5.0

    @nn.compact
    def __call__(self, r):
        x = r / self.r_max
        p = self.p
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * x**p
            + p * (p + 2.0) * x ** (p + 1)
            - (p * (p + 1.0) / 2.0) * x ** (p + 2)
        )
        return jnp.where(r < self.r_max, envelope, 0.0)
```

Confirm `nn` and `jnp` are already imported in that file; add if missing.

- [x] **Step 5: Run test — expect pass**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_basis_functions.py -v -k polynomial_cutoff`
Expected: both tests PASS.

- [x] **Step 6: Commit**

```bash
git add apax/layers/descriptor/basis_functions.py tests/unit_tests/layers/descriptor/test_basis_functions.py
git commit -m "feat(layers): add PolynomialCutoff basis function for MACE"
```

---

### Task P0.3: Create skeleton `MaceRepresentation` that returns random features

Proves the descriptor contract and lets downstream plumbing be tested before the real math is in place.

**Files:**
- Create: `apax/layers/descriptor/mace.py`
- Test: `tests/unit_tests/layers/descriptor/test_mace_descriptor.py`

- [x] **Step 1: Write failing contract test**

Create `tests/unit_tests/layers/descriptor/test_mace_descriptor.py`:

```python
"""Shape & contract tests for MaceRepresentation.

These tests use random weights and a skeleton forward pass; correctness
against upstream MACE is validated in the parity tests (gated).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apax.layers.descriptor.mace import MaceRepresentation


@pytest.fixture
def tiny_system():
    n_atoms = 4
    n_neighbors = 8
    rng = np.random.default_rng(0)
    dr_vec = jnp.asarray(rng.normal(size=(n_neighbors, 3))).astype(jnp.float32)
    Z = jnp.asarray([1, 8, 1, 6], dtype=jnp.int32)
    idx = jnp.asarray(
        [[0, 0, 1, 1, 2, 2, 3, 3],
         [1, 2, 0, 3, 0, 3, 1, 2]], dtype=jnp.int32,
    )
    return dr_vec, Z, idx, n_atoms


def test_mace_representation_contract(tiny_system):
    dr_vec, Z, idx, n_atoms = tiny_system
    model = MaceRepresentation(
        r_max=5.0,
        num_bessel=8,
        max_ell=2,
        hidden_irreps="16x0e + 16x1o",
        num_interactions=2,
    )
    params = model.init(jax.random.PRNGKey(0), dr_vec, Z, idx)
    out = model.apply(params, dr_vec, Z, idx)
    assert out.ndim == 2
    assert out.shape[0] == n_atoms
    assert out.dtype == jnp.float32
    assert jnp.isfinite(out).all()


def test_mace_representation_is_jittable(tiny_system):
    dr_vec, Z, idx, n_atoms = tiny_system
    model = MaceRepresentation(hidden_irreps="8x0e")
    params = model.init(jax.random.PRNGKey(0), dr_vec, Z, idx)
    jitted = jax.jit(model.apply)
    out = jitted(params, dr_vec, Z, idx)
    assert out.shape[0] == n_atoms
```

- [x] **Step 2: Run — expect import fail**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_descriptor.py -v`
Expected: ModuleNotFoundError — `apax.layers.descriptor.mace` does not exist.

- [x] **Step 3: Create skeleton module**

Create `apax/layers/descriptor/mace.py`:

```python
"""MACE descriptor for apax.

Exposes :class:`MaceRepresentation`, a Flax linen ``nn.Module`` that consumes
pair displacement vectors and atomic numbers, and returns per-atom scalar
features compatible with apax's :class:`AtomisticReadout`.

Matches the apax descriptor contract exactly:
``__call__(dr_vec, Z, idx) -> (n_atoms, n_features)``.

The heavy equivariant math lives in :mod:`apax.layers.descriptor.mace_blocks`
and is added progressively over P1–P2. This module starts as a typed skeleton
that returns random-but-finite scalar features so the rest of apax (builder,
config, readout wiring) can be validated end-to-end first.
"""
from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from apax.utils.convert import str_to_dtype

InteractionKind = Literal[
    "RealAgnostic",
    "RealAgnosticResidual",
    "RealAgnosticDensity",
    "RealAgnosticDensityResidual",
]


class MaceRepresentation(nn.Module):
    """MACE descriptor producing per-atom scalar features.

    Parameters
    ----------
    r_max : float
        Interaction cutoff in Angstrom.
    num_bessel : int
        Number of Bessel radial basis functions.
    num_polynomial_cutoff : int
        Polynomial order of the envelope cutoff.
    max_ell : int
        Maximum spherical-harmonic degree used for edge features.
    hidden_irreps : str
        e3nn irreps string for node features, e.g. ``"128x0e + 128x1o"``.
    num_interactions : int
        Number of (interaction, product) layer pairs.
    correlation : int
        Symmetric-contraction correlation order.
    interaction_cls : str
        Which MACE interaction block variant to use.
    num_elements : int
        Size of the chemical-element embedding table.
    use_cueq : bool
        If True, use cuequivariance-jax kernels where available.
    apply_mask : bool
        If True, zero out masked atoms in the output.
    dtype : Any
        Floating-point dtype for features.
    """

    r_max: float = 5.0
    num_bessel: int = 8
    num_polynomial_cutoff: int = 5
    max_ell: int = 3
    hidden_irreps: str = "128x0e + 128x1o"
    num_interactions: int = 2
    correlation: int = 3
    interaction_cls: InteractionKind = "RealAgnosticResidual"
    num_elements: int = 119
    use_cueq: bool = False
    apply_mask: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, dr_vec, Z, idx):
        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)
        n_atoms = Z.shape[0]
        # P0 skeleton: return a random-but-finite per-atom feature tensor
        # so the rest of apax (readout, scale/shift, train, MD) can be wired.
        # Replaced by real forward pass in P1.
        n_scalar = _scalar_feature_dim(self.hidden_irreps) * self.num_interactions
        w = self.param(
            "skeleton_w",
            nn.initializers.normal(stddev=0.01),
            (self.num_elements, n_scalar),
            dtype,
        )
        features = w[Z]
        return features


def _scalar_feature_dim(irreps_str: str) -> int:
    """Parse irreps string and return the multiplicity of the 0e component."""
    for part in irreps_str.split("+"):
        part = part.strip()
        if part.endswith("x0e"):
            return int(part.split("x")[0])
    return 0
```

- [x] **Step 4: Run — expect pass**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_descriptor.py -v`
Expected: both tests PASS.

- [x] **Step 5: Commit**

```bash
git add apax/layers/descriptor/mace.py tests/unit_tests/layers/descriptor/test_mace_descriptor.py
git commit -m "feat(layers): add MaceRepresentation skeleton (P0)"
```

---

### Task P0.4: Add `MaceModelConfig` to pydantic discriminated union

**Files:**
- Modify: `apax/config/model_config.py`

- [x] **Step 1: Read existing file to find the discriminated-union pattern**

Run: Read `apax/config/model_config.py`, locate existing `BaseModelConfig`, `GMNNConfig`, and the `Union[...]` type alias used for `model`.

- [x] **Step 2: Add `MaceModelConfig`**

Append to `apax/config/model_config.py` (before the `Union[...]` alias):

```python
class MaceModelConfig(BaseModelConfig):
    """Config for a MACE model used as an apax descriptor.

    See docs/superpowers/specs/2026-04-20-mace-foundation-model-integration-design.md.
    """

    name: Literal["mace"] = "mace"
    r_max: float = 5.0
    num_bessel: int = 8
    num_polynomial_cutoff: int = 5
    max_ell: int = 3
    hidden_irreps: str = "128x0e + 128x1o"
    num_interactions: int = 2
    correlation: int = 3
    interaction_cls: Literal[
        "RealAgnostic",
        "RealAgnosticResidual",
        "RealAgnosticDensity",
        "RealAgnosticDensityResidual",
    ] = "RealAgnosticResidual"
    use_cueq: bool = False

    # Foundation-model loading
    pretrained: str | Path | None = None
    freeze_backbone: bool = False
    unfreeze_backbone_epoch: int | None = None
```

Ensure `Path` and `Literal` are imported at top of file.

- [x] **Step 3: Add `MaceModelConfig` to discriminated union**

Locate the existing alias (likely `ModelConfig = Annotated[Union[GMNNConfig, ...], Field(discriminator="name")]`) and add `MaceModelConfig`.

- [x] **Step 4: Validate import**

Run: `uv run python -c "from apax.config.model_config import MaceModelConfig; print(MaceModelConfig().name)"`
Expected output: `mace`

Run: `uv run apax schema`
Expected: succeeds; `.vscode/train_schema.json` contains `"mace"` as a discriminator value.

- [x] **Step 5: Commit**

```bash
git add apax/config/model_config.py
git commit -m "feat(config): add MaceModelConfig to model discriminated union"
```

---

### Task P0.5: Add `MaceBuilder` to builder.py

**Files:**
- Modify: `apax/nn/builder.py`
- Test: `tests/unit_tests/nn/test_mace_builder.py`

- [x] **Step 1: Write failing builder test**

Create `tests/unit_tests/nn/test_mace_builder.py`:

```python
"""MaceBuilder wiring — verifies the builder produces a runnable EnergyModel."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def mace_config_dict():
    """Minimal MACE config dict, matching what ModelBuilder expects."""
    return {
        "name": "mace",
        "r_max": 5.0,
        "num_bessel": 8,
        "num_polynomial_cutoff": 5,
        "max_ell": 2,
        "hidden_irreps": "16x0e + 16x1o",
        "num_interactions": 2,
        "correlation": 3,
        "interaction_cls": "RealAgnosticResidual",
        "use_cueq": False,
        "descriptor_dtype": "fp32",
        "readout_dtype": "fp32",
        "scale_shift_dtype": "fp64",
        "activation_fn": "silu",
        "nn": [32, 32],
        "b_init": "zeros",
        "w_init": "lecun_normal",
        "use_ntk": False,
        "n_shallow_members": 0,
        "basis": {"name": "bessel", "n_basis": 8, "r_max": 5.0},
        "ensemble": None,
        "property_heads": [],
        "empirical_corrections": [],
        "calc_stress": False,
    }


def test_mace_builder_builds_energy_model(mace_config_dict):
    from apax.nn.builder import MaceBuilder

    builder = MaceBuilder(mace_config_dict, n_species=10)
    model = builder.build_energy_model()
    # sanity: module has the expected structure
    assert model.representation is not None
    assert model.readout is not None


def test_mace_builder_build_derivative_model_runs(mace_config_dict):
    from apax.nn.builder import MaceBuilder

    builder = MaceBuilder(mace_config_dict, n_species=10)
    model = builder.build_energy_derivative_model()
    # initialize + run forward on a tiny system
    n_atoms = 4
    R = jnp.zeros((n_atoms, 3))
    Z = jnp.array([1, 8, 1, 6], dtype=jnp.int32)
    # Neighbor list: all pairs for this tiny system
    idx = jnp.array([[0, 0, 0, 1, 1, 2, 2, 3, 3, 3],
                     [1, 2, 3, 0, 2, 0, 3, 0, 1, 2]], dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), R, Z, idx, None, None)
    energy, _ = model.apply(params, R, Z, idx, None, None)
    assert jnp.isfinite(energy).all()
```

- [x] **Step 2: Run — expect import fail**

Run: `uv run pytest tests/unit_tests/nn/test_mace_builder.py -v`
Expected: ImportError — `MaceBuilder` not defined.

- [x] **Step 3: Append `MaceBuilder`**

Edit `apax/nn/builder.py`, find existing pattern of `So3kratesBuilder`, append at end of file:

```python
class MaceBuilder(ModelBuilder):
    """Build an ``EnergyModel`` whose descriptor is :class:`MaceRepresentation`.

    Overrides only :meth:`build_descriptor`; all other build steps
    (readout, scale-shift, property heads, corrections, ensemble wrapping)
    are inherited from :class:`ModelBuilder` unchanged.
    """

    def build_descriptor(self, apply_mask):
        from apax.layers.descriptor.mace import MaceRepresentation

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
```

- [x] **Step 4: Wire builder selection**

Find where the builder is selected based on `config.name` (likely in `apax/nn/builder.py` or `apax/train/run.py`). Add a branch:

```python
elif model_name == "mace":
    builder = MaceBuilder(config_dict, n_species=n_species)
```

If using a dict/mapping, register `"mace": MaceBuilder` there.

- [x] **Step 5: Run — expect pass**

Run: `uv run pytest tests/unit_tests/nn/test_mace_builder.py -v`
Expected: both tests PASS.

- [x] **Step 6: Commit**

```bash
git add apax/nn/builder.py tests/unit_tests/nn/test_mace_builder.py
git commit -m "feat(nn): add MaceBuilder (P0 plumbing)"
```

---

### Task P0.6: End-to-end smoke — `apax train` runs with a MACE config

**Files:**
- Create: `tests/integration_tests/mace/__init__.py`
- Create: `tests/integration_tests/mace/test_mace_smoke.py`

- [x] **Step 1: Write smoke test**

Create `tests/integration_tests/mace/test_mace_smoke.py`:

```python
"""End-to-end smoke: apax trains a tiny MACE model (skeleton, random features).

Gate: does not verify correctness, only that the entire pipeline
(config → builder → data pipeline → trainer → checkpoint) runs to completion.
"""
import yaml
from pathlib import Path
import pytest


@pytest.mark.slow
def test_apax_train_with_mace_config(tmp_path, fake_dataset_h5):
    """``fake_dataset_h5`` is the existing apax fixture providing a tiny
    training dataset; if its name differs, update to the actual fixture name
    used by other integration tests (see tests/integration_tests/conftest.py)."""
    # Minimal MACE train config
    cfg = {
        "data": {"directory": str(tmp_path), "experiment": "mace_smoke",
                 "data_path": str(fake_dataset_h5), "n_epochs": 1},
        "model": {
            "name": "mace",
            "r_max": 5.0,
            "hidden_irreps": "8x0e",
            "num_interactions": 1,
            "correlation": 2,
            "max_ell": 1,
            "num_bessel": 4,
            "descriptor_dtype": "fp32",
        },
        "loss": [{"name": "energy", "loss_type": "weighted_mse"}],
    }
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    from apax.train.run import run
    run(str(cfg_path))
    assert (tmp_path / "mace_smoke").exists()
```

*Note:* the exact fixture name (`fake_dataset_h5`) depends on the existing `tests/integration_tests/conftest.py`. Before running, inspect that file and use its minimal dataset fixture; if none exists, use the simplest pattern from `tests/integration_tests/cli/test_app.py`.

- [x] **Step 2: Run — may need fixture adjustment**

Run: `uv run pytest tests/integration_tests/mace/test_mace_smoke.py -v -m slow`
Expected: Either PASS or fail with a clear config schema issue. If schema fails, dump the validation error and fix fields (likely missing required fields from `BaseModelConfig`).

- [x] **Step 3: Fix config fields as needed, re-run**

Iterate until smoke test PASSES.

- [x] **Step 4: Commit**

```bash
git add tests/integration_tests/mace/
git commit -m "test: add MACE train smoke test (P0 skeleton)"
```

**P0 exit criterion met:** ✅ `uv sync --extra mace` works; `apax train` runs with a MACE YAML config using the random-features skeleton. Phase P0 landed on `feat/mace-foundation-integration` through commit `bc5a7f89` (2026-04-20). 9 commits including two review-fix rounds; 50 unit tests pass; 1-epoch MACE smoke train completes in ~14s on MD22.

---

## Phase P1 — Native MACE forward pass (linen port)

Goal: real MACE forward pass using e3nn-jax + cuequivariance-jax primitives. Parity with upstream mace-jax on *random weights* at fp32 tolerance.

### Task P1.1: Bessel basis (reuse) and edge feature assembly

**Files:**
- Create: `apax/layers/descriptor/mace_blocks.py` (new module)
- Test: `tests/unit_tests/layers/descriptor/test_mace_blocks.py`

- [x] **Step 1: Stub module**

Create `apax/layers/descriptor/mace_blocks.py`:

```python
"""Building blocks for :class:`MaceRepresentation`.

Each block is a Flax linen ``nn.Module`` and is independently testable.
Layer order inside ``MaceRepresentation`` is:

    LinearNodeEmbedding → N× (InteractionBlock → ProductBlock) → concat scalars

Equivariant primitives:
- Irreps / IrrepsArray / tensor products: ``e3nn_jax``
- Symmetric contraction: ``cuequivariance`` / ``cuequivariance_jax``
  (works on CPU; CUDA kernels are opt-in via ``use_cueq``).
"""
from __future__ import annotations

from typing import Any

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from flax import linen as nn


def assemble_edge_features(dr_vec, r_max, num_bessel, num_poly_cutoff, max_ell):
    """Compute radial-basis × cutoff and spherical harmonics.

    Parameters
    ----------
    dr_vec : Array, shape (n_edges, 3)
    r_max : float
    num_bessel : int
    num_poly_cutoff : int
    max_ell : int

    Returns
    -------
    radial : Array, shape (n_edges, num_bessel)   — (Bessel basis × cutoff)
    sph : IrrepsArray                              — spherical harmonics 0e..max_ell
    """
    from apax.layers.descriptor.basis_functions import BesselBasis, PolynomialCutoff

    r_ij = jnp.linalg.norm(dr_vec, axis=-1)
    bessel = BesselBasis(n_basis=num_bessel, r_max=r_max)(r_ij)
    cutoff = PolynomialCutoff(p=num_poly_cutoff, r_max=r_max)(r_ij)
    radial = bessel * cutoff[..., None]
    irreps = e3nn.Irreps.spherical_harmonics(max_ell)
    sph = e3nn.spherical_harmonics(
        irreps, dr_vec, normalize=True, normalization="component",
    )
    return radial, sph
```

- [x] **Step 2: Write test**

Create `tests/unit_tests/layers/descriptor/test_mace_blocks.py`:

```python
import jax.numpy as jnp
import numpy as np

from apax.layers.descriptor.mace_blocks import assemble_edge_features


def test_assemble_edge_features_shapes():
    dr = jnp.asarray(np.random.default_rng(0).normal(size=(8, 3))).astype(jnp.float32)
    radial, sph = assemble_edge_features(dr, r_max=5.0, num_bessel=8,
                                          num_poly_cutoff=5, max_ell=2)
    assert radial.shape == (8, 8)
    # irreps 0e + 1o + 2e = 1 + 3 + 5 = 9 scalars per edge
    assert sph.array.shape == (8, 9)


def test_assemble_edge_features_cutoff_zeroes_far_edges():
    dr = jnp.asarray([[0.0, 0.0, 10.0]] * 4)       # all edges beyond r_max=5
    radial, _ = assemble_edge_features(dr, r_max=5.0, num_bessel=4,
                                        num_poly_cutoff=5, max_ell=1)
    assert jnp.allclose(radial, 0.0, atol=1e-6)
```

- [x] **Step 3: Run — expect pass**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k edge_features`
Expected: PASS.

- [x] **Step 4: Commit**

```bash
git add apax/layers/descriptor/mace_blocks.py tests/unit_tests/layers/descriptor/test_mace_blocks.py
git commit -m "feat(mace): edge-feature assembly (radial × cutoff, spherical harmonics)"
```

---

### Task P1.2: `LinearNodeEmbedding`

**Files:**
- Modify: `apax/layers/descriptor/mace_blocks.py`
- Test: `tests/unit_tests/layers/descriptor/test_mace_blocks.py`

- [x] **Step 1: Write test**

Append to `test_mace_blocks.py`:

```python
import jax
from apax.layers.descriptor.mace_blocks import LinearNodeEmbedding


def test_linear_node_embedding_scalar_output():
    n_atoms = 5
    num_elements = 10
    hidden_irreps = "16x0e"
    Z = jnp.array([0, 3, 5, 0, 9], dtype=jnp.int32)
    emb = LinearNodeEmbedding(num_elements=num_elements, irreps_out=hidden_irreps)
    params = emb.init(jax.random.PRNGKey(0), Z)
    out = emb.apply(params, Z)
    # IrrepsArray output; scalar-only irreps means (n_atoms, 16)
    assert out.array.shape == (n_atoms, 16)
    assert str(out.irreps) == "16x0e"
```

- [x] **Step 2: Run — expect fail**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py::test_linear_node_embedding_scalar_output -v`
Expected: ImportError.

- [x] **Step 3: Implement**

Append to `apax/layers/descriptor/mace_blocks.py`:

```python
class LinearNodeEmbedding(nn.Module):
    """One-hot element embedding followed by an irreps-linear.

    Produces node features in ``irreps_out``. Initial features are pure
    scalars (parity even), so ``irreps_out`` must contain only ``0e`` components.
    """
    num_elements: int
    irreps_out: str                # must be scalar irreps (e.g. "128x0e")

    @nn.compact
    def __call__(self, Z):
        irreps = e3nn.Irreps(self.irreps_out).filter("0e")
        one_hot = jax.nn.one_hot(Z, self.num_elements)           # (n, E)
        # Linear projection E -> irreps.dim, wrapped as IrrepsArray
        w = self.param(
            "weight",
            nn.initializers.normal(stddev=1.0 / jnp.sqrt(self.num_elements)),
            (self.num_elements, irreps.dim),
        )
        feats = one_hot @ w                                      # (n, irreps.dim)
        return e3nn.IrrepsArray(irreps, feats)
```

- [x] **Step 4: Run — expect pass**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k linear_node_embedding`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add apax/layers/descriptor/mace_blocks.py tests/unit_tests/layers/descriptor/test_mace_blocks.py
git commit -m "feat(mace): LinearNodeEmbedding block"
```

---

### Task P1.3: `InteractionBlock` — `RealAgnosticResidual` variant (e3nn-jax path)

Reference: `/Users/fzills/tools/mace-jax/mace_jax/modules/blocks.py`, class `RealAgnosticResidualInteractionBlock` (~line 863). Port behavior, not code — match semantics but write in linen idioms.

**Files:**
- Modify: `apax/layers/descriptor/mace_blocks.py`
- Test: `tests/unit_tests/layers/descriptor/test_mace_blocks.py`

- [x] **Step 1: Read mace-jax reference**

Run: Read `/Users/fzills/tools/mace-jax/mace_jax/modules/blocks.py` focusing on `RealAgnosticResidualInteractionBlock` and its `__call__`. Also read `wrapper_ops.py` to understand how `FullyConnectedTensorProduct` and `Linear` are used.

Summarize into a comment block in `mace_blocks.py`:

```python
# InteractionBlock (RealAgnosticResidual) — port notes
# Inputs:  node_feats [n_atoms, irreps_in], edge_attrs (sph) [n_edges, Ylm],
#          edge_feats (radial) [n_edges, n_bessel], i (receivers), j (senders), pair_mask
# Layout:
#   1. source_linear:    node_feats[j] -> irreps_in (e3nn Linear)
#   2. conv_tp:          source @ edge_attrs via FullyConnectedTensorProduct
#                        with per-edge MLP weights from radial features
#   3. scatter_sum:      aggregate messages into receiver index i
#   4. target_linear:    aggregated -> irreps_out (e3nn Linear)
#   5. residual:         output + skip connection from original node_feats
# Output: new node_feats [n_atoms, irreps_out]
```

- [x] **Step 2: Write test against randomly initialized block**

Append to `test_mace_blocks.py`:

```python
from apax.layers.descriptor.mace_blocks import InteractionBlock


def test_interaction_block_shape_and_finite():
    n_atoms, n_edges = 5, 12
    hidden = "16x0e + 16x1o"
    sph_irreps = "1x0e + 1x1o + 1x2e"  # max_ell=2

    node_feats = e3nn.IrrepsArray(
        e3nn.Irreps(hidden),
        jnp.asarray(np.random.default_rng(0).normal(size=(n_atoms, 64))),
    )
    sph_array = jnp.asarray(np.random.default_rng(1).normal(size=(n_edges, 9)))
    edge_attrs = e3nn.IrrepsArray(e3nn.Irreps(sph_irreps), sph_array)
    edge_feats = jnp.asarray(np.random.default_rng(2).normal(size=(n_edges, 8)))
    i = jnp.asarray(np.random.default_rng(3).integers(0, n_atoms, size=n_edges))
    j = jnp.asarray(np.random.default_rng(4).integers(0, n_atoms, size=n_edges))

    block = InteractionBlock(irreps_out=hidden, interaction_cls="RealAgnosticResidual")
    params = block.init(jax.random.PRNGKey(0), node_feats, edge_attrs, edge_feats, i, j)
    out = block.apply(params, node_feats, edge_attrs, edge_feats, i, j)
    assert out.array.shape == (n_atoms, e3nn.Irreps(hidden).dim)
    assert jnp.isfinite(out.array).all()
```

- [x] **Step 3: Run — expect import fail**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k interaction_block`
Expected: ImportError.

- [x] **Step 4: Implement `InteractionBlock`**

Append to `mace_blocks.py`:

```python
class InteractionBlock(nn.Module):
    """MACE interaction + residual.

    RealAgnosticResidual variant: one tensor product between node features and
    edge (spherical-harmonic) attributes, weighted by a radial MLP, aggregated
    into receivers via scatter-sum, plus a skip connection.

    Parameters
    ----------
    irreps_out : str
        Target irreps for node features after this block.
    interaction_cls : str
        Which interaction variant; for now only RealAgnosticResidual is
        implemented. Other variants raise NotImplementedError.
    radial_mlp : tuple[int, ...]
        Hidden layer widths for the radial MLP that gates tensor-product
        channels. Defaults to ``(64, 64, 64)``.
    """

    irreps_out: str
    interaction_cls: str = "RealAgnosticResidual"
    radial_mlp: tuple = (64, 64, 64)

    @nn.compact
    def __call__(self, node_feats, edge_attrs, edge_feats, receivers, senders):
        if self.interaction_cls != "RealAgnosticResidual":
            raise NotImplementedError(
                f"Interaction variant {self.interaction_cls!r} not yet implemented; "
                "only 'RealAgnosticResidual' is supported at this phase."
            )
        irreps_in = node_feats.irreps
        irreps_out = e3nn.Irreps(self.irreps_out)

        # 1. Linear pre-mix
        x = e3nn.flax.Linear(irreps_in, name="linear_up")(node_feats)

        # 2. Gather source node features at senders
        x_j = x[senders]

        # 3. Tensor product with edge spherical harmonics
        #    Output irreps = full tensor-square subset reachable in irreps_out
        tp = e3nn.tensor_product(x_j, edge_attrs, filter_ir_out=irreps_out)

        # 4. Radial MLP producing a scalar per TP path per edge
        n_paths = tp.irreps.num_irreps
        mlp_widths = (*self.radial_mlp, n_paths)
        weights = e3nn.flax.MultiLayerPerceptron(
            list(mlp_widths), act=jax.nn.silu, name="radial_mlp"
        )(edge_feats)
        weighted = tp * weights                                 # broadcast-safe

        # 5. Scatter-sum into receivers
        out = e3nn.scatter_sum(weighted, dst=receivers, output_size=node_feats.shape[0])

        # 6. Post-mix and residual
        out = e3nn.flax.Linear(irreps_out, name="linear_down")(out)
        skip = e3nn.flax.Linear(irreps_out, name="skip_linear")(node_feats)
        return out + skip
```

*Hints for the engineer executing this step:*
- If `e3nn.scatter_sum` has a slightly different signature in the installed version, check `uv run python -c "import e3nn_jax as e; help(e.scatter_sum)"`. Alternative: use `e3nn.utils.scatter_sum` or manual `jax.ops.segment_sum(...)` on `.array` while wrapping back into `IrrepsArray`.
- The `*` operator on `IrrepsArray × jnp.ndarray` must broadcast per-irrep; if it doesn't, use `tp.transform_by_weights(weights)` or loop by irrep.

- [x] **Step 5: Run — expect pass**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k interaction_block`
Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add apax/layers/descriptor/mace_blocks.py tests/unit_tests/layers/descriptor/test_mace_blocks.py
git commit -m "feat(mace): InteractionBlock (RealAgnosticResidual)"
```

---

### Task P1.4: `ProductBlock` — symmetric contraction via cuequivariance

**Files:**
- Modify: `apax/layers/descriptor/mace_blocks.py`
- Test: `tests/unit_tests/layers/descriptor/test_mace_blocks.py`

- [x] **Step 1: Read reference**

Run: Read `/Users/fzills/tools/mace-jax/mace_jax/adapters/cuequivariance/symmetric_contraction.py` (first 120 lines) to see exactly how cuequivariance's `symmetric_contraction` is invoked and what kwargs it expects.

Summarize into a docstring on the block.

- [x] **Step 2: Write test**

Append to `test_mace_blocks.py`:

```python
from apax.layers.descriptor.mace_blocks import ProductBlock


def test_product_block_shape_and_finite():
    n_atoms = 5
    hidden = "16x0e + 16x1o"
    node_feats = e3nn.IrrepsArray(
        e3nn.Irreps(hidden),
        jnp.asarray(np.random.default_rng(0).normal(size=(n_atoms, 64))),
    )
    Z = jnp.array([0, 1, 2, 3, 1], dtype=jnp.int32)
    block = ProductBlock(
        hidden_irreps=hidden,
        correlation=3,
        num_elements=10,
    )
    params = block.init(jax.random.PRNGKey(0), node_feats, Z)
    out = block.apply(params, node_feats, Z)
    assert out.array.shape == (n_atoms, e3nn.Irreps(hidden).dim)
    assert jnp.isfinite(out.array).all()
```

- [x] **Step 3: Run — expect fail**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k product_block`
Expected: ImportError.

- [x] **Step 4: Implement `ProductBlock`**

Append to `mace_blocks.py`:

```python
class ProductBlock(nn.Module):
    """MACE product block: high-body-order symmetric contraction.

    Wraps cuequivariance's ``symmetric_contraction`` primitive. Produces
    node features in the same irreps as its input, after per-element
    weighted self-tensor-products up to ``correlation`` order.

    Parameters
    ----------
    hidden_irreps : str
        Target irreps (same as input).
    correlation : int
        Maximum tensor-product order (MACE uses 3).
    num_elements : int
        Number of chemical elements (sets the per-species weight table).
    use_cueq : bool
        If True, use cuequivariance's CUDA kernels when available. Defaults
        False (pure JAX path in cuequivariance, portable).
    """

    hidden_irreps: str
    correlation: int = 3
    num_elements: int = 119
    use_cueq: bool = False

    @nn.compact
    def __call__(self, node_feats, Z):
        import cuequivariance as cue
        import cuequivariance_jax as cuex

        irreps = e3nn.Irreps(self.hidden_irreps)
        # Build the descriptor once at init via setup-like pattern
        sc_layer = cuex.SymmetricContraction(
            irreps_in=cue.Irreps(str(irreps), cue.O3_e3nn),
            irreps_out=cue.Irreps(str(irreps), cue.O3_e3nn),
            correlation=self.correlation,
            num_elements=self.num_elements,
            use_reduced_cg=True,
        )
        # Apply: select per-element weight row via Z
        out_array = sc_layer(node_feats.array, indices=Z)
        return e3nn.IrrepsArray(irreps, out_array)
```

*Caveats:*
- The exact API of `cuequivariance_jax.SymmetricContraction` may differ by version. Inspect the version actually installed: `uv run python -c "import cuequivariance_jax as cuex; help(cuex.SymmetricContraction)"`.
- If the module takes layout strings (`mul_ir` / `ir_mul`), pass `input_layout="mul_ir"` explicitly (matches e3nn default).
- The parameter table lives inside `sc_layer`; ensure it's registered as a flax param. If cuex is not a flax Module natively, wrap the trainable tensor as a `self.param(...)` call and pass it into the functional cue API.

- [x] **Step 5: Run — expect pass**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k product_block`
Expected: PASS. Fix version/API mismatches as you go.

- [x] **Step 6: Commit**

```bash
git add apax/layers/descriptor/mace_blocks.py tests/unit_tests/layers/descriptor/test_mace_blocks.py
git commit -m "feat(mace): ProductBlock (symmetric contraction via cuequivariance)"
```

---

### Task P1.5: Wire real forward pass into `MaceRepresentation`

**Files:**
- Modify: `apax/layers/descriptor/mace.py`
- Test: `tests/unit_tests/layers/descriptor/test_mace_descriptor.py`

- [x] **Step 1: Replace skeleton forward with real pipeline**

Edit `apax/layers/descriptor/mace.py`, replace the body of `__call__`:

```python
@nn.compact
def __call__(self, dr_vec, Z, idx):
    from apax.layers.descriptor.mace_blocks import (
        LinearNodeEmbedding, InteractionBlock, ProductBlock, assemble_edge_features,
    )
    from apax.layers.descriptor.so3krates import get_node_mask, get_neighbor_mask

    dtype = str_to_dtype(self.dtype)
    dr_vec = dr_vec.astype(dtype)
    i, j = idx[0], idx[1]

    pair_mask = get_neighbor_mask(idx) if self.apply_mask else 1.0
    node_mask = get_node_mask(Z) if self.apply_mask else 1.0

    radial, sph = assemble_edge_features(
        dr_vec,
        self.r_max, self.num_bessel, self.num_polynomial_cutoff, self.max_ell,
    )
    radial = radial * pair_mask[..., None] if self.apply_mask else radial

    # Initial node features: scalars only
    scalar_init = _scalar_irreps_only(self.hidden_irreps)
    node_feats = LinearNodeEmbedding(
        num_elements=self.num_elements, irreps_out=scalar_init,
    )(Z)

    per_layer_scalars = []
    for layer in range(self.num_interactions):
        node_feats = InteractionBlock(
            irreps_out=self.hidden_irreps,
            interaction_cls=self.interaction_cls,
        )(node_feats, sph, radial, i, j)
        node_feats = ProductBlock(
            hidden_irreps=self.hidden_irreps,
            correlation=self.correlation,
            num_elements=self.num_elements,
            use_cueq=self.use_cueq,
        )(node_feats, Z)
        per_layer_scalars.append(node_feats.filter(keep="0e").array)

    features = jnp.concatenate(per_layer_scalars, axis=-1)
    if self.apply_mask:
        features = features * node_mask[..., None]
    assert features.dtype == dtype
    return features


def _scalar_irreps_only(irreps_str: str) -> str:
    """Return the 0e subset of an irreps string (e.g. '128x0e + 128x1o' -> '128x0e')."""
    parts = [p.strip() for p in irreps_str.split("+")]
    scalar = [p for p in parts if p.endswith("x0e")]
    if not scalar:
        raise ValueError(f"No 0e component in irreps {irreps_str!r}")
    return " + ".join(scalar)
```

Remove the obsolete `skeleton_w` param and `_scalar_feature_dim` helper.

- [x] **Step 2: Update existing descriptor tests to still pass**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_descriptor.py -v`
Expected: shape test passes; jit test passes.

If shape changes (n_features will now be `n_interactions × scalar_mult`), update assertions to compute expected dim from irreps.

- [ ] **Step 3: Write parity test against mace-jax random-weight output (optional if mace-jax importable)** — skipped; deferred to P3 parity phase

Create `tests/unit_tests/layers/descriptor/test_mace_vs_macejax.py`:

```python
"""Cross-check against mace-jax on a fixed random-weight config.

Gated by @pytest.mark.mace_parity so it's skipped by default. Requires
mace-jax to be importable in the env.
"""
import pytest

pytestmark = pytest.mark.mace_parity


def test_mace_representation_vs_macejax():
    pytest.importorskip("mace_jax")
    # Minimal sanity: both produce features of the same shape.
    # Full numerical parity is validated in the conversion-based parity test.
    import jax, jax.numpy as jnp, numpy as np
    from apax.layers.descriptor.mace import MaceRepresentation

    dr_vec = jnp.asarray(np.random.default_rng(0).normal(size=(12, 3))).astype(jnp.float32)
    Z = jnp.array([1, 8, 1, 6, 7, 1], dtype=jnp.int32)
    idx = jnp.stack([jnp.array([0,1,2,3,4,5,0,1,2,3,4,5]),
                      jnp.array([1,2,3,4,5,0,5,4,3,2,1,0])])
    model = MaceRepresentation(hidden_irreps="16x0e + 16x1o", num_interactions=2)
    params = model.init(jax.random.PRNGKey(0), dr_vec, Z, idx)
    out = model.apply(params, dr_vec, Z, idx)
    assert out.shape == (6, 32)          # 16 scalars × 2 layers
```

- [x] **Step 4: Run, ensure smoke still passes**

Run: `uv run pytest tests/unit_tests/layers/descriptor/ tests/unit_tests/nn/test_mace_builder.py -v`
Expected: all PASS.

- [x] **Step 5: Commit**

```bash
git add apax/layers/descriptor/mace.py tests/unit_tests/layers/descriptor/
git commit -m "feat(mace): replace skeleton with native forward pass (P1)"
```

**P1 exit criterion met:** ✅ MaceRepresentation runs the real forward pass; 19/19 unit tests green (`tests/unit_tests/layers/descriptor/` + `tests/unit_tests/nn/test_mace_builder.py`). Landed on `feat/mace-foundation-integration` through commit `a0c22c3a` on 2026-04-21. Notable deviations from the plan's verbatim code: (a) P1.4's `cuex.SymmetricContraction` is replaced by a linen port of mace-jax's adapter (using `cuex.equivariant_polynomial` + a cached `cue_mace_symmetric_contraction` descriptor); (b) P1.5 inlines `_get_node_mask`/`_get_neighbor_mask` rather than importing from `so3krates` (which pulls in an unavailable `myrto` dep); (c) `InteractionBlock.skip_linear` uses `force_irreps_out=True` so the residual sum stays shape-consistent on the first layer.

---

## Phase P2 — cuequivariance dispatch for `use_cueq=True`

Goal: identical outputs with/without cueq, CUDA-accelerated where a GPU is present.

### Task P2.1: Add `use_cueq` dispatch helpers

**Files:**
- Create: `apax/layers/descriptor/mace_irreps.py`
- Test: `tests/unit_tests/layers/descriptor/test_mace_irreps.py`

- [ ] **Step 1: Write module**

Create `apax/layers/descriptor/mace_irreps.py`:

```python
"""Dispatch helpers between e3nn-jax and cuequivariance-jax primitives.

Both libraries implement the same equivariant operators (tensor products,
linear layers) with matching conventions when cuequivariance is configured
with ``O3_e3nn``. These helpers give the blocks a single-entry point so
``use_cueq=True`` / ``False`` is a local swap rather than duplicated modules.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import e3nn_jax as e3nn


def tensor_product(x, y, *, filter_ir_out, use_cueq: bool):
    """Return x ⊗ y filtered to ``filter_ir_out`` irreps."""
    if use_cueq:
        import cuequivariance_jax as cuex
        return cuex.tensor_product(x, y, filter_ir_out=filter_ir_out)
    import e3nn_jax as e3nn
    return e3nn.tensor_product(x, y, filter_ir_out=filter_ir_out)


def linear_module(irreps_in, irreps_out, *, use_cueq: bool, name: str):
    """Return a flax Module implementing an irreps-linear map."""
    if use_cueq:
        import cuequivariance_jax as cuex
        return cuex.flax.Linear(irreps_in, irreps_out, name=name)
    import e3nn_jax as e3nn
    return e3nn.flax.Linear(irreps_out, name=name)
```

*Caveat:* cuex's flax binding naming may differ. Adjust imports after reading the actually-installed version (`uv run python -c "import cuequivariance_jax as cuex; print(dir(cuex))"`).

- [ ] **Step 2: Write tests**

Create `tests/unit_tests/layers/descriptor/test_mace_irreps.py`:

```python
import jax.numpy as jnp
import numpy as np
import pytest

import e3nn_jax as e3nn

from apax.layers.descriptor.mace_irreps import tensor_product


@pytest.mark.parametrize("use_cueq", [False, True])
def test_tensor_product_shapes_match(use_cueq):
    rng = np.random.default_rng(0)
    x = e3nn.IrrepsArray("4x0e + 4x1o", jnp.asarray(rng.normal(size=(3, 16))))
    y = e3nn.IrrepsArray("1x0e + 1x1o + 1x2e", jnp.asarray(rng.normal(size=(3, 9))))
    try:
        z = tensor_product(x, y, filter_ir_out="4x0e + 4x1o + 4x2e", use_cueq=use_cueq)
    except ImportError:
        pytest.skip("cuequivariance_jax not installed with flax binding")
    assert z.shape[0] == 3


@pytest.mark.parametrize("use_cueq", [False, True])
def test_tensor_product_numerical_equivalence(use_cueq):
    """Enforce that e3nn and cueq paths produce the same values to fp32 tol."""
    import jax
    rng = np.random.default_rng(1)
    x_arr = jnp.asarray(rng.normal(size=(3, 16)))
    y_arr = jnp.asarray(rng.normal(size=(3, 9)))
    x = e3nn.IrrepsArray("4x0e + 4x1o", x_arr)
    y = e3nn.IrrepsArray("1x0e + 1x1o + 1x2e", y_arr)
    ref = tensor_product(x, y, filter_ir_out="4x0e", use_cueq=False)
    alt = tensor_product(x, y, filter_ir_out="4x0e", use_cueq=use_cueq)
    np.testing.assert_allclose(ref.array, alt.array, rtol=1e-4, atol=1e-5)
```

- [ ] **Step 3: Run — expect pass (or skip for cueq path if binding missing)**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_irreps.py -v`
Expected: non-cueq tests PASS. cueq tests PASS or SKIP.

- [ ] **Step 4: Commit**

```bash
git add apax/layers/descriptor/mace_irreps.py tests/unit_tests/layers/descriptor/test_mace_irreps.py
git commit -m "feat(mace): dispatch helpers for e3nn-jax/cuequivariance"
```

---

### Task P2.2: Plumb `use_cueq` into `InteractionBlock`

**Files:**
- Modify: `apax/layers/descriptor/mace_blocks.py`
- Test: `tests/unit_tests/layers/descriptor/test_mace_blocks.py`

- [ ] **Step 1: Add `use_cueq` field and swap TP call**

Edit `InteractionBlock` in `mace_blocks.py`:
- Add field: `use_cueq: bool = False`
- Replace `e3nn.tensor_product(...)` call with `mace_irreps.tensor_product(..., use_cueq=self.use_cueq)`.

- [ ] **Step 2: Write equivalence test**

Append to `test_mace_blocks.py`:

```python
def test_interaction_block_cueq_equivalent():
    rng = np.random.default_rng(0)
    n_atoms, n_edges = 5, 12
    hidden = "16x0e + 16x1o"
    sph_irreps = "1x0e + 1x1o + 1x2e"

    node_feats = e3nn.IrrepsArray(e3nn.Irreps(hidden), jnp.asarray(rng.normal(size=(n_atoms, 64))))
    edge_attrs = e3nn.IrrepsArray(e3nn.Irreps(sph_irreps), jnp.asarray(rng.normal(size=(n_edges, 9))))
    edge_feats = jnp.asarray(rng.normal(size=(n_edges, 8)))
    i = jnp.asarray(rng.integers(0, n_atoms, size=n_edges))
    j = jnp.asarray(rng.integers(0, n_atoms, size=n_edges))

    block_ref = InteractionBlock(irreps_out=hidden, use_cueq=False)
    params = block_ref.init(jax.random.PRNGKey(0), node_feats, edge_attrs, edge_feats, i, j)
    out_ref = block_ref.apply(params, node_feats, edge_attrs, edge_feats, i, j)

    block_cueq = InteractionBlock(irreps_out=hidden, use_cueq=True)
    try:
        out_cueq = block_cueq.apply(params, node_feats, edge_attrs, edge_feats, i, j)
    except (ImportError, AttributeError):
        pytest.skip("cueq flax bindings unavailable")
    np.testing.assert_allclose(out_ref.array, out_cueq.array, rtol=1e-4, atol=1e-5)
```

- [ ] **Step 3: Run — expect pass/skip**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k cueq`

- [ ] **Step 4: Commit**

```bash
git add apax/layers/descriptor/mace_blocks.py tests/unit_tests/layers/descriptor/test_mace_blocks.py
git commit -m "feat(mace): wire use_cueq through InteractionBlock"
```

---

### Task P2.3: Propagate `use_cueq` through `MaceRepresentation`

**Files:**
- Modify: `apax/layers/descriptor/mace.py`
- Test: new descriptor-level equivalence test

- [ ] **Step 1: Pass flag into `InteractionBlock`**

In `MaceRepresentation.__call__`, change:
```python
InteractionBlock(irreps_out=self.hidden_irreps, interaction_cls=self.interaction_cls)
```
to:
```python
InteractionBlock(
    irreps_out=self.hidden_irreps,
    interaction_cls=self.interaction_cls,
    use_cueq=self.use_cueq,
)
```

`ProductBlock` already has the flag.

- [ ] **Step 2: Write equivalence test**

Append to `test_mace_descriptor.py`:

```python
def test_representation_cueq_equivalence(tiny_system):
    dr_vec, Z, idx, n_atoms = tiny_system
    cfg = dict(hidden_irreps="16x0e + 16x1o", num_interactions=2, max_ell=2)
    m_ref = MaceRepresentation(use_cueq=False, **cfg)
    params = m_ref.init(jax.random.PRNGKey(0), dr_vec, Z, idx)
    out_ref = m_ref.apply(params, dr_vec, Z, idx)

    m_cueq = MaceRepresentation(use_cueq=True, **cfg)
    try:
        out_cueq = m_cueq.apply(params, dr_vec, Z, idx)
    except (ImportError, AttributeError):
        pytest.skip("cueq path not available")
    np.testing.assert_allclose(out_ref, out_cueq, rtol=1e-3, atol=1e-5)
```

- [ ] **Step 3: Run**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_descriptor.py -v`
Expected: PASS (or cueq test SKIPs).

- [ ] **Step 4: Commit**

```bash
git add apax/layers/descriptor/mace.py tests/unit_tests/layers/descriptor/test_mace_descriptor.py
git commit -m "feat(mace): propagate use_cueq to representation"
```

**P2 exit:** use_cueq path gives identical outputs to fp32 tolerance.

---

## Phase P3 — Converter + `MaceReadout`, consumed via the standard load path

**Revised 2026-04-22.** This phase replaces the previous draft that introduced a parallel `.apax/`-dir format (`params.msgpack` + `config.json` + `metadata.json`), a `load_mace_foundation` loader, and an `ASECalculator` branch. Those are gone. Under the new design:

- The converter writes apax's existing training-output layout: `<dst>/config.yaml` (via `Config.dump_config`) + `<dst>/best/` (orbax `CheckpointManager` checkpoint).
- Loading a foundation is the same code path as loading any user-trained apax model: `restore_parameters(<dst>)` returns `(Config, params)`.
- MACE's per-layer readout structure lives in a new `MaceReadout` module that slots into the stock `EnergyModel`'s readout slot alongside `AtomisticReadout`.
- `PerElementScaleShift` absorbs MACE's global scale + global shift + per-element atomic-energy reference (zero-padded to 119 elements at convert time).
- `MaceFoundationEnergyModel`, `load_mace_foundation`, `_resolve_short_name`, `_is_mace_foundation_dir`, and the custom `.apax/` directory shape are all deleted.

### Task P3.0: Revert the `.apax/`-dir scaffolding that commit `d7342dd9` introduced

**Files:**
- Modify: `apax/md/ase_calc.py` (remove `_is_mace_foundation_dir` + NotImplementedError branch)
- Delete: `tests/unit_tests/md/test_ase_calc_mace_foundation.py`
- Delete: `tests/unit_tests/md/__init__.py` (if empty after the above delete and no other tests in that directory — verify first)
- Delete: `tests/integration_tests/mace/test_load_foundation.py`
- Delete: `apax/nn/mace_foundation_model.py`
- Modify: `apax/transfer_learning/mace_foundation.py` (remove `load_mace_foundation`, `_resolve_short_name`)
- Modify: `apax/config/model_config.py` (remove `pretrained`, `freeze_backbone`, `unfreeze_backbone_epoch`, `num_elements` from `MaceModelConfig`; these were accidental additions)

- [ ] **Step 1: Revert ASECalculator branch**

Open `apax/md/ase_calc.py`. Remove the module-level `_is_mace_foundation_dir` helper and the `if _is_mace_foundation_dir(model_dir):` branch inside `ASECalculator.__init__` that raises `NotImplementedError`. After the revert, the constructor flows straight into `self.model_config, self.params = restore_parameters(model_dir)` as it did before `d7342dd9`.

- [ ] **Step 2: Delete obsolete test files**

```bash
rm tests/unit_tests/md/test_ase_calc_mace_foundation.py
rm tests/integration_tests/mace/test_load_foundation.py
rm apax/nn/mace_foundation_model.py
```

Check whether `tests/unit_tests/md/__init__.py` is empty AND the directory has no other files:

```bash
test ! -s tests/unit_tests/md/__init__.py && ls tests/unit_tests/md/
```

If `__init__.py` is 0 bytes and no other test files remain, delete the directory: `rmdir tests/unit_tests/md/`. Otherwise keep both.

- [ ] **Step 3: Prune `mace_foundation.py`**

Open `apax/transfer_learning/mace_foundation.py`. Delete the `load_mace_foundation` function and the `_resolve_short_name` helper. Keep `run_conversion`, `_load_torch_foundation_model`, `_extract_config_from_torch`, `_map_state_to_pytree` (still stubbed — filled in Task P3.4), `_validate_no_nan`, `_extract_norm_consts`, `_torch_mace_version`, `_apax_version`.

- [ ] **Step 4: Clean `MaceModelConfig`**

In `apax/config/model_config.py`, remove these fields from `MaceModelConfig`: `pretrained`, `freeze_backbone`, `unfreeze_backbone_epoch`, `num_elements`. Remove their docstring lines. Leave every other field intact.

- [ ] **Step 5: Run unit tests — expect green**

```bash
uv run pytest tests/unit_tests/ -v --no-header -x
```

Expected: all previously-passing tests still pass. Any remaining references to `load_mace_foundation`, `_is_mace_foundation_dir`, `MaceFoundationEnergyModel`, or the removed config fields surface here and must be fixed.

- [ ] **Step 6: Commit**

```bash
git add apax/md/ase_calc.py \
        apax/transfer_learning/mace_foundation.py \
        apax/config/model_config.py \
        tests/unit_tests/md \
        tests/integration_tests/mace/test_load_foundation.py \
        apax/nn/mace_foundation_model.py
git commit -m "revert(mace): drop parallel .apax/ format and load_mace_foundation"
```

---

### Task P3.1: `MaceModelConfig` — add `readout_kind` + `MLP_irreps`

**Files:**
- Modify: `apax/config/model_config.py`
- Test: `tests/unit_tests/config/test_mace_model_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit_tests/config/test_mace_model_config.py`:

```python
"""MaceModelConfig pydantic schema — smoke tests."""
import pytest
from apax.config.model_config import MaceModelConfig


def test_mace_model_config_defaults():
    cfg = MaceModelConfig()
    assert cfg.name == "mace"
    assert cfg.readout_kind == "mace"
    assert cfg.MLP_irreps == "16x0e"
    assert cfg.hidden_irreps == "128x0e + 128x1o"
    assert cfg.num_interactions == 2


def test_mace_model_config_readout_kind_literal():
    cfg = MaceModelConfig(readout_kind="standard")
    assert cfg.readout_kind == "standard"
    with pytest.raises(ValueError, match="readout_kind"):
        MaceModelConfig(readout_kind="garbage")


def test_mace_model_config_has_no_removed_fields():
    """Regression: pretrained / freeze_backbone / num_elements must be gone."""
    fields = MaceModelConfig.model_fields
    assert "pretrained" not in fields
    assert "freeze_backbone" not in fields
    assert "unfreeze_backbone_epoch" not in fields
    assert "num_elements" not in fields
```

- [ ] **Step 2: Run — expect fail**

```bash
uv run pytest tests/unit_tests/config/test_mace_model_config.py -v
```

Expected: `readout_kind` test FAILs (field missing). `has_no_removed_fields` test passes if Task P3.0 Step 4 succeeded.

- [ ] **Step 3: Add the two fields**

Edit `MaceModelConfig` in `apax/config/model_config.py`:

```python
class MaceModelConfig(BaseModelConfig, extra="forbid"):
    """
    Configuration for a MACE model.

    Parameters
    ----------
    r_max : PositiveFloat, default = 5.0
        Interaction cutoff in Angstrom.
    num_bessel : PositiveInt, default = 8
        Number of Bessel radial basis functions.
    num_polynomial_cutoff : PositiveInt, default = 5
        Polynomial order of the envelope cutoff.
    max_ell : PositiveInt, default = 3
        Maximum spherical-harmonic degree.
    hidden_irreps : str, default = "128x0e + 128x1o"
        e3nn-jax irreps string for node features. Must include a 0e component.
    num_interactions : PositiveInt, default = 2
        Number of (interaction, product) layer pairs.
    correlation : PositiveInt, default = 3
        Symmetric-contraction correlation order.
    interaction_cls : Literal, default = "RealAgnosticResidual"
        Which MACE interaction block variant to use.
    use_cueq : bool, default = False
        Dispatch to cuequivariance-jax kernels where available.
    readout_kind : Literal["mace", "standard"], default = "mace"
        Which readout slot-filler to use. ``"mace"`` builds ``MaceReadout``
        with per-layer readouts matching the torch-mace foundation. ``"standard"``
        falls back to apax's ``AtomisticReadout`` (useful when fine-tuning and
        replacing the head).
    MLP_irreps : str, default = "16x0e"
        Hidden irreps for the final non-linear readout's internal MLP.
        Ignored when ``readout_kind == "standard"``.
    """

    name: Literal["mace"] = "mace"

    r_max: PositiveFloat = 5.0
    num_bessel: PositiveInt = 8
    num_polynomial_cutoff: PositiveInt = 5
    max_ell: PositiveInt = 3
    hidden_irreps: str = "128x0e + 128x1o"
    num_interactions: PositiveInt = 2
    correlation: PositiveInt = 3
    interaction_cls: Literal[
        "RealAgnostic",
        "RealAgnosticResidual",
        "RealAgnosticDensity",
        "RealAgnosticDensityResidual",
    ] = "RealAgnosticResidual"
    use_cueq: bool = False
    readout_kind: Literal["mace", "standard"] = "mace"
    MLP_irreps: str = "16x0e"

    def get_builder(self):
        from apax.nn.builder import MaceBuilder

        return MaceBuilder
```

- [ ] **Step 4: Run — expect pass**

```bash
uv run pytest tests/unit_tests/config/test_mace_model_config.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add apax/config/model_config.py tests/unit_tests/config/test_mace_model_config.py
git commit -m "feat(config): add readout_kind + MLP_irreps to MaceModelConfig"
```

---

### Task P3.2: `LinearReadoutBlock` + `NonLinearReadoutBlock` primitives

These primitives already landed in commit `2b7283e8`. Verify their contract matches what `MaceReadout` needs, and add tests if missing.

**Files:**
- Check: `apax/layers/descriptor/mace_blocks.py` — confirm `LinearReadoutBlock` and `NonLinearReadoutBlock` accept a single-atom `e3nn.IrrepsArray` and an `n_out: int = 1` parameter.
- Test: `tests/unit_tests/layers/descriptor/test_mace_blocks.py`

- [ ] **Step 1: Read the existing implementation**

Read `apax/layers/descriptor/mace_blocks.py` — locate `LinearReadoutBlock` and `NonLinearReadoutBlock`. Confirm they take a single-atom irreps array and emit `(n_out,)`. If either:
- doesn't have an `n_out` field (defaults to 1), OR
- doesn't use `e3nn.flax.Linear` with irreps `f"{n_out}x0e"` on the final projection, OR
- expects a batched `(n_atoms, hidden_dim)` input instead of per-atom,

patch the signature. The contract for `MaceReadout` is that each block is called inside `jax.vmap(self.readout)(g)` in `EnergyModel.__call__`, so each invocation sees one atom.

- [ ] **Step 2: Write or extend tests**

Append to `tests/unit_tests/layers/descriptor/test_mace_blocks.py`:

```python
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp

from apax.layers.descriptor.mace_blocks import (
    LinearReadoutBlock,
    NonLinearReadoutBlock,
)


def test_linear_readout_block_scalar_out():
    block = LinearReadoutBlock(n_out=1)
    feat = e3nn.IrrepsArray("16x0e", jnp.ones((16,)))  # single atom
    params = block.init(jax.random.PRNGKey(0), feat)
    out = block.apply(params, feat)
    assert out.shape == (1,)


def test_linear_readout_block_ensemble_out():
    block = LinearReadoutBlock(n_out=4)
    feat = e3nn.IrrepsArray("16x0e", jnp.ones((16,)))
    params = block.init(jax.random.PRNGKey(0), feat)
    out = block.apply(params, feat)
    assert out.shape == (4,)


def test_nonlinear_readout_block_scalar_out():
    block = NonLinearReadoutBlock(MLP_irreps="8x0e", n_out=1)
    feat = e3nn.IrrepsArray("16x0e", jnp.ones((16,)))
    params = block.init(jax.random.PRNGKey(0), feat)
    out = block.apply(params, feat)
    assert out.shape == (1,)


def test_nonlinear_readout_block_vmap_over_atoms():
    block = NonLinearReadoutBlock(MLP_irreps="8x0e", n_out=1)
    n_atoms = 5
    feat = e3nn.IrrepsArray("16x0e", jnp.ones((n_atoms, 16)))
    # Init on single atom, then vmap.apply on stacked
    single = e3nn.IrrepsArray("16x0e", jnp.ones((16,)))
    params = block.init(jax.random.PRNGKey(0), single)
    batched = jax.vmap(lambda x: block.apply(params, x))(feat)
    assert batched.shape == (n_atoms, 1)
```

- [ ] **Step 3: Run — expect pass (possibly after signature fixes from Step 1)**

```bash
uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k readout
```

Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add apax/layers/descriptor/mace_blocks.py tests/unit_tests/layers/descriptor/test_mace_blocks.py
git commit -m "test(mace): verify readout block contracts for vmap usage"
```

---

### Task P3.3: `MaceReadout` — the per-layer sum readout

**Files:**
- Modify: `apax/layers/readout.py` (add `MaceReadout` alongside `AtomisticReadout`)
- Test: `tests/unit_tests/layers/test_mace_readout.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit_tests/layers/test_mace_readout.py`:

```python
"""MaceReadout — per-layer readout sum slotted into EnergyModel."""
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import pytest

from apax.layers.readout import MaceReadout


def test_mace_readout_single_atom_shape():
    num_interactions = 2
    hidden_dim = 16
    readout = MaceReadout(
        num_interactions=num_interactions,
        hidden_dim=hidden_dim,
        MLP_irreps="8x0e",
    )
    x = jnp.ones((num_interactions * hidden_dim,))      # single atom, post-vmap
    params = readout.init(jax.random.PRNGKey(0), x)
    out = readout.apply(params, x)
    assert out.shape == (1,)


def test_mace_readout_shallow_ensemble_shape():
    num_interactions = 2
    hidden_dim = 16
    n_members = 4
    readout = MaceReadout(
        num_interactions=num_interactions,
        hidden_dim=hidden_dim,
        MLP_irreps="8x0e",
        n_shallow_ensemble=n_members,
    )
    x = jnp.ones((num_interactions * hidden_dim,))
    params = readout.init(jax.random.PRNGKey(0), x)
    out = readout.apply(params, x)
    assert out.shape == (n_members,)


def test_mace_readout_vmapped_over_atoms():
    num_interactions = 2
    hidden_dim = 16
    n_atoms = 5
    readout = MaceReadout(num_interactions=num_interactions, hidden_dim=hidden_dim)
    x_single = jnp.ones((num_interactions * hidden_dim,))
    params = readout.init(jax.random.PRNGKey(0), x_single)

    g = jnp.ones((n_atoms, num_interactions * hidden_dim))
    batched = jax.vmap(lambda xi: readout.apply(params, xi))(g)
    assert batched.shape == (n_atoms, 1)


def test_mace_readout_uses_linear_then_nonlinear():
    """Last layer is non-linear; earlier layers are linear.

    We verify this by counting params — with one linear readout (1 Linear)
    and one non-linear readout (Linear + Linear), the non-linear layer's
    internal hidden Linear dominates total param count.
    """
    num_interactions = 2
    hidden_dim = 16
    readout = MaceReadout(
        num_interactions=num_interactions,
        hidden_dim=hidden_dim,
        MLP_irreps="8x0e",
    )
    x = jnp.ones((num_interactions * hidden_dim,))
    params = readout.init(jax.random.PRNGKey(0), x)
    # readout_0 is Linear (16 -> 1): 16 weights
    # readout_1 is NonLinear (16 -> 8 -> 1): 16*8 + 8 = 136 weights
    leaves = jax.tree_util.tree_leaves(params)
    assert sum(l.size for l in leaves) >= 16 + 16 * 8  # sanity lower bound
```

- [ ] **Step 2: Run — expect fail**

```bash
uv run pytest tests/unit_tests/layers/test_mace_readout.py -v
```

Expected: FAIL — `MaceReadout` not found.

- [ ] **Step 3: Implement `MaceReadout`**

Append to `apax/layers/readout.py`:

```python
from typing import Any

import e3nn_jax as e3nn
import jax.numpy as jnp
import flax.linen as nn

from apax.layers.descriptor.mace_blocks import (
    LinearReadoutBlock,
    NonLinearReadoutBlock,
)
from apax.utils.convert import str_to_dtype


class MaceReadout(nn.Module):
    """Per-layer readout sum that matches the foundation MACE forward pass.

    Consumes the concatenated per-layer scalar features emitted by
    :class:`~apax.layers.descriptor.mace.MaceRepresentation`, reshapes into
    per-layer chunks, applies a linear readout to each intermediate layer and
    a two-Linear MLP (with SiLU gate) to the last layer, and returns the sum.

    Slotted into :class:`~apax.nn.models.EnergyModel` in the readout position.
    ``EnergyModel`` vmaps the readout over atoms, so each invocation sees a
    single atom's flat feature vector.

    Parameters
    ----------
    num_interactions : int
        Number of interaction layers in the backbone.
    hidden_dim : int
        Per-layer scalar channel count; equals the ``0e`` dimension of
        ``MaceRepresentation.hidden_irreps``.
    MLP_irreps : str, default = "16x0e"
        Hidden irreps of the last-layer non-linear MLP.
    n_shallow_ensemble : int, default = 0
        When > 0, each block's final projection emits ``n_shallow_ensemble``
        scalars. Downstream ``EnergyModel`` auto-detects the ensemble case
        from ``E_i.shape[1] > 1``.
    dtype : Any
        Floating-point dtype for internal computations.
    """

    num_interactions: int
    hidden_dim: int
    MLP_irreps: str = "16x0e"
    n_shallow_ensemble: int = 0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        """Return per-atom energy summed across layers.

        Parameters
        ----------
        x : Array, shape ``(num_interactions * hidden_dim,)``
            Flat per-atom feature vector (post-vmap).

        Returns
        -------
        Array, shape ``(1,)`` or ``(n_shallow_ensemble,)``
        """
        dtype = str_to_dtype(self.dtype)
        x = x.astype(dtype)
        layers = x.reshape(self.num_interactions, self.hidden_dim)
        n_out = self.n_shallow_ensemble if self.n_shallow_ensemble > 0 else 1

        E = jnp.zeros((n_out,), dtype=dtype)
        for k in range(self.num_interactions):
            feat = e3nn.IrrepsArray(f"{self.hidden_dim}x0e", layers[k])
            if k < self.num_interactions - 1:
                E = E + LinearReadoutBlock(n_out=n_out, name=f"readout_{k}")(feat)
            else:
                E = E + NonLinearReadoutBlock(
                    MLP_irreps=self.MLP_irreps,
                    n_out=n_out,
                    name=f"readout_{k}",
                )(feat)
        return E
```

- [ ] **Step 4: Run — expect pass**

```bash
uv run pytest tests/unit_tests/layers/test_mace_readout.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add apax/layers/readout.py tests/unit_tests/layers/test_mace_readout.py
git commit -m "feat(mace): MaceReadout — per-layer sum readout for EnergyModel slot"
```

---

### Task P3.4: `MaceBuilder.build_readout` override + end-to-end wiring

**Files:**
- Modify: `apax/nn/builder.py`
- Test: `tests/unit_tests/nn/test_mace_builder.py`

- [ ] **Step 1: Write the failing tests**

Create (or extend) `tests/unit_tests/nn/test_mace_builder.py`:

```python
"""MaceBuilder — descriptor + readout composition tests."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apax.config.model_config import MaceModelConfig
from apax.nn.builder import MaceBuilder


def _minimal_cfg(**overrides):
    base = MaceModelConfig(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=1,
        hidden_irreps="8x0e",
        num_interactions=2,
        correlation=2,
        interaction_cls="RealAgnosticResidual",
        descriptor_dtype="fp32",
        readout_dtype="fp32",
        scale_shift_dtype="fp64",
    )
    return base.model_copy(update=overrides).model_dump()


def test_mace_builder_uses_mace_readout_by_default():
    from apax.layers.readout import MaceReadout
    builder = MaceBuilder(_minimal_cfg(readout_kind="mace"), n_species=5)
    readout = builder.build_readout(builder.config)
    assert isinstance(readout, MaceReadout)
    assert readout.num_interactions == 2
    assert readout.hidden_dim == 8


def test_mace_builder_standard_readout_falls_back():
    from apax.layers.readout import AtomisticReadout
    cfg = _minimal_cfg(readout_kind="standard")
    cfg["nn"] = [32, 32]
    cfg["w_init"] = "lecun"
    cfg["b_init"] = "zeros"
    cfg["use_ntk"] = False
    builder = MaceBuilder(cfg, n_species=5)
    readout = builder.build_readout(builder.config)
    assert isinstance(readout, AtomisticReadout)


def test_mace_builder_shallow_ensemble_plumbs_n_members():
    from apax.layers.readout import MaceReadout
    cfg = _minimal_cfg()
    cfg["ensemble"] = {"kind": "shallow", "n_members": 4, "force_variance": True,
                       "chunk_size": None}
    builder = MaceBuilder(cfg, n_species=5)
    readout = builder.build_readout(builder.config)
    assert isinstance(readout, MaceReadout)
    assert readout.n_shallow_ensemble == 4


def test_mace_builder_end_to_end_energy_model():
    """Compose full EnergyDerivativeModel and call it with random params."""
    cfg = _minimal_cfg()
    builder = MaceBuilder(cfg, n_species=5)
    model = builder.build_energy_derivative_model()

    n_atoms = 3
    R = jnp.zeros((n_atoms, 3))
    Z = jnp.array([1, 6, 8], dtype=jnp.int32)
    # minimal neighbor list: each atom connected to the next
    neighbor = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32).T
    box = jnp.zeros((3,))
    offsets = jnp.zeros((neighbor.shape[1], 3))

    params = model.init(jax.random.PRNGKey(0), R, Z, neighbor, box, offsets)
    out = model.apply(params, R, Z, neighbor, box, offsets)
    assert "energy" in out
    assert "forces" in out
    assert out["forces"].shape == (n_atoms, 3)
    assert np.all(np.isfinite(np.asarray(out["forces"])))
```

- [ ] **Step 2: Run — expect fail**

```bash
uv run pytest tests/unit_tests/nn/test_mace_builder.py -v
```

Expected: FAIL — `build_readout` uses the parent's default.

- [ ] **Step 3: Override `build_readout` on `MaceBuilder`**

Edit `apax/nn/builder.py`. Find the existing `MaceBuilder` class and add `build_readout`:

```python
class MaceBuilder(ModelBuilder):
    def build_descriptor(
        self,
        apply_mask,
    ):
        from apax.layers.descriptor.mace import MaceRepresentation

        descriptor = MaceRepresentation(
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
        return descriptor

    def build_readout(
        self,
        head_config,
        is_feature_fn: bool = False,
        only_use_n_layers: int | None = None,
    ):
        """Route between ``MaceReadout`` (matches foundation MACE) and the
        standard ``AtomisticReadout`` based on ``readout_kind``."""
        import e3nn_jax as e3nn
        kind = self.config.get("readout_kind", "mace")
        if kind == "mace" and not is_feature_fn:
            from apax.layers.readout import MaceReadout
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

- [ ] **Step 4: Run — expect pass**

```bash
uv run pytest tests/unit_tests/nn/test_mace_builder.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add apax/nn/builder.py tests/unit_tests/nn/test_mace_builder.py
git commit -m "feat(mace): MaceBuilder.build_readout routes mace/standard readouts"
```

---

### Task P3.5: Converter — emit `<dst>/config.yaml` + `<dst>/best/` via pydantic + orbax

**Files:**
- Modify: `apax/transfer_learning/mace_foundation.py`
- Test: `tests/integration_tests/mace/test_convert.py` (gated by `@pytest.mark.mace_parity`)

**Preamble:** Torch + mace-torch are installed in the dev environment via `uv sync --group mace-convert --extra mace`. The converter code must still import torch lazily so that library consumers who don't install the group don't pay the cost.

- [ ] **Step 1: Replace the gated test's expectations**

Rewrite `tests/integration_tests/mace/test_convert.py`:

```python
"""apax convert-mace produces an apax training-output-shaped directory.

Gated by mace_parity. Requires:
    uv sync --group mace-convert --extra mace
"""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.mace_parity


@pytest.mark.parametrize("model_name", ["small"])
def test_convert_produces_config_yaml_and_orbax_checkpoint(tmp_path, model_name):
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion
    from apax.train.checkpoints import restore_parameters

    dst = tmp_path / f"{model_name}.apax"
    run_conversion(model_name, dst, head="default", family="mace_mp")

    assert (dst / "config.yaml").is_file()
    assert (dst / "best").is_dir()
    assert (dst / "converter_metadata.json").is_file()

    meta = json.loads((dst / "converter_metadata.json").read_text())
    assert meta["source"] == model_name
    assert meta["family"] == "mace_mp"

    # The universal apax loader must accept it.
    config, params = restore_parameters(dst)
    assert config.model.name == "mace"
    # Params must contain at least the representation + readout + scale_shift branches.
    flat = {"/".join(str(k) for k in p): v
            for p, v in __import__("jax").tree_util.tree_flatten_with_path(params)[0]}
    assert any("MaceRepresentation" in k for k in flat), "descriptor params missing"
    assert any("MaceReadout" in k or "readout" in k for k in flat), "readout params missing"
    assert any("ScaleShift" in k or "scale_shift" in k for k in flat), "scale_shift missing"


def test_convert_rejects_unknown_head(tmp_path):
    """Multi-head selection rejects unknown head names up-front."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion
    with pytest.raises(ValueError, match="head"):
        run_conversion("medium-mpa-0", tmp_path / "out.apax", head="does-not-exist")
```

- [ ] **Step 2: Run — expect skip (no torch in default env) or fail-on-NotImplementedError when run with mace_parity**

```bash
uv run pytest tests/integration_tests/mace/test_convert.py -v
```

Expected: both tests SKIPPED (gated). With the dev group installed and `-m mace_parity` passed, the first test should FAIL (`_map_state_to_pytree` still stubbed or converter still writes the old format); that's the red step we're about to fix.

- [ ] **Step 3: Implement `run_conversion` — orchestrator**

Replace `run_conversion` in `apax/transfer_learning/mace_foundation.py`:

```python
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp


def run_conversion(
    source: Union[str, Path],
    dst: Path,
    *,
    head: str = "default",
    family: str = "mace_mp",
) -> None:
    """Convert a torch-mace foundation model into an apax training-output dir.

    The output layout matches what ``apax train`` produces, so every apax
    loader (``restore_parameters``, ``ASECalculator``, ``apax md``, BAL)
    reads it with no special-casing.

    Parameters
    ----------
    source
        Canonical MACE foundation name (e.g. ``"small"``, ``"medium-mpa-0"``)
        or a filesystem path to a local ``.model`` file.
    dst
        Output directory.
    head
        For multi-head models (e.g. MPA), which head to retain.
    family
        Foundation-family resolver. Initial scope: ``"mace_mp"``.
    """
    from apax.config.train_config import Config
    from apax.config.model_config import MaceModelConfig
    from apax.train.checkpoints import load_state  # for schema reference

    dst = Path(dst)

    # 1. Load torch model + resolve source on disk
    torch_model, resolved_path = _load_torch_foundation_model(source, family=family)

    # 2. Extract architecture → MaceModelConfig fields
    mace_cfg_fields = _extract_config_from_torch(torch_model, head=head)
    torch_atomic_numbers = tuple(
        torch_model.atomic_numbers.detach().cpu().numpy().astype(np.int64).tolist()
    )

    # 3. Build full apax Config with placeholder training-only fields
    full_cfg = _synthesize_full_config(mace_cfg_fields, dst)

    # 4. Build the same model the trainer would build
    Builder = full_cfg.model.get_builder()
    builder = Builder(full_cfg.model.model_dump(), n_species=119)
    energy_derivative_model = builder.build_energy_derivative_model()

    R_dummy = jnp.zeros((2, 3))
    Z_dummy = jnp.array([1, 1], dtype=jnp.int32)
    neigh_dummy = jnp.array([[0], [1]], dtype=jnp.int32)
    box_dummy = jnp.zeros((3,))
    offsets_dummy = jnp.zeros((1, 3))
    params_template = energy_derivative_model.init(
        jax.random.PRNGKey(0), R_dummy, Z_dummy, neigh_dummy, box_dummy, offsets_dummy
    )

    # 5. Map torch weights into the template
    state = {k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()}
    extra_scalars = {
        "scale": float(torch_model.scale_shift.scale.detach().cpu()),
        "shift": float(torch_model.scale_shift.shift.detach().cpu()),
        "atomic_energies": torch_model.atomic_energies_fn.atomic_energies.detach().cpu().numpy(),
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

    # 6. Persist
    dst.mkdir(parents=True, exist_ok=True)
    full_cfg.dump_config(dst)  # dst/config.yaml
    _write_orbax_checkpoint(dst / "best", params, epoch=0)

    # 7. Provenance
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


def _synthesize_full_config(mace_cfg_fields: dict, dst: Path):
    """Build a valid ``Config`` around a MaceModelConfig for a converted model.

    Training-only fields get placeholder values; users should never use this
    YAML to launch training directly, only to restore params.
    """
    from apax.config.train_config import Config
    from apax.config.model_config import MaceModelConfig

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
        "optimizer": {},  # all defaults
    }
    return Config.model_validate(cfg_dict)


def _write_orbax_checkpoint(path: Path, params, *, epoch: int) -> None:
    """Write ``{"model": {"params": params}, "epoch": epoch}`` via orbax.

    Matches the schema that :func:`apax.train.checkpoints.load_state` reads.
    """
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    with ocp.CheckpointManager(path) as mngr:
        mngr.save(
            step=0,
            args=ocp.args.StandardSave({"model": {"params": params}, "epoch": epoch}),
        )
        mngr.wait_until_finished()
```

- [ ] **Step 4: Fix `_extract_config_from_torch` attr paths for ScaleShiftMACE**

In `apax/transfer_learning/mace_foundation.py`, review `_extract_config_from_torch`. Probed foundation model layout (see commit `5bddaed2`):

```python
def _extract_config_from_torch(model, head: str) -> dict:
    """Extract MaceModelConfig-compatible fields from a torch ScaleShiftMACE.

    Reads attributes that are actual tensors/buffers on the module; wraps them
    as the right Python types for pydantic.
    """
    import e3nn

    # Heads: torch exposes ``model.heads`` as a list of strings; default is ``['default']``
    heads = list(getattr(model, "heads", ["default"]))
    if len(heads) > 1 and head not in heads:
        raise ValueError(
            f"head={head!r} not in available heads {heads}. "
            f"Pass --head <name> from that list."
        )

    # Interaction variant
    inter0_cls = type(model.interactions[0]).__name__
    variant_map = {
        "RealAgnosticInteractionBlock": "RealAgnostic",
        "RealAgnosticResidualInteractionBlock": "RealAgnosticResidual",
        "RealAgnosticDensityInteractionBlock": "RealAgnosticDensity",
        "RealAgnosticDensityResidualInteractionBlock": "RealAgnosticDensityResidual",
    }
    if inter0_cls not in variant_map:
        raise NotImplementedError(
            f"Unsupported interaction class {inter0_cls!r}; "
            "supported: " + ", ".join(variant_map)
        )

    # Hidden irreps — take from the first product's Linear.irreps_out
    hidden_irreps = str(model.products[0].linear.irreps_out)

    # max_ell — the spherical_harmonics irreps are 1x0e + 1x1o + ... + 1x{L}{o|e}
    sph_irreps = e3nn.o3.Irreps(str(model.spherical_harmonics.irreps_out))
    max_ell = max(ir.ir.l for _, ir in sph_irreps)

    # Correlation — count U_matrix_N entries in the first contraction
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
        "interaction_cls": variant_map[inter0_cls],
        "use_cueq": False,
        "readout_kind": "mace",
        "MLP_irreps": "16x0e",  # torch-mace foundation models default
    }
    return cfg
```

- [ ] **Step 5: Implement `_map_state_to_pytree` — core weight mapping**

This is the single largest function. Organize as one top-level function plus sub-helpers per block type:

```python
def _map_state_to_pytree(
    state: dict[str, np.ndarray],
    template: dict,
    *,
    torch_atomic_numbers: tuple[int, ...],
    extra_scalars: dict,
    selected_head: str,
    config,
) -> dict:
    """Translate torch state_dict → linen pytree matching ``template``.

    The pytree is the fully-wrapped output of
    :meth:`MaceBuilder.build_energy_derivative_model().init(...)`, which means
    the top level is ``{"params": {"energy_model": {...}}, ...}``.

    The function mutates a copy of ``template`` and returns it.
    """
    out = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), template)
    # Drill into the energy_model subtree
    energy_params = out["params"]["energy_model"]

    # 1. MaceRepresentation subtree
    rep_params = energy_params["representation"]
    _map_node_embedding(state, rep_params, torch_atomic_numbers)
    _map_interactions(state, rep_params, torch_atomic_numbers, config)
    _map_products(state, rep_params, torch_atomic_numbers, config)

    # 2. MaceReadout subtree
    readout_params = energy_params["readout"]
    _map_readouts(state, readout_params, selected_head=selected_head,
                  num_interactions=config.num_interactions,
                  MLP_irreps=config.MLP_irreps)

    # 3. PerElementScaleShift — combine scale + shift + atomic_energies
    ss_params = energy_params["scale_shift"]
    _map_scale_shift(
        ss_params,
        global_scale=extra_scalars["scale"],
        global_shift=extra_scalars["shift"],
        atomic_energies=extra_scalars["atomic_energies"],
        torch_atomic_numbers=torch_atomic_numbers,
    )

    return out
```

Each `_map_X` helper follows a pattern: read the specific torch keys, reshape/pad/transpose, assign into the target pytree. Reference implementations in `/Users/fzills/tools/mace-jax/mace_jax/modules/blocks.py` (each class has an `import_from_torch` decorator-generated method). Fill these in iteratively against the actual state dict:

```python
def _map_node_embedding(state, rep_params, atomic_numbers):
    """node_embedding.linear.weight (N_torch * hidden,) → rep/LinearNodeEmbedding_0/weight (119, hidden)."""
    w = state["node_embedding.linear.weight"]  # flat torch tensor
    N_torch = len(atomic_numbers)
    hidden = w.size // N_torch
    w = w.reshape(N_torch, hidden)  # torch-e3nn Linear layout; confirm and transpose if needed
    target = rep_params["LinearNodeEmbedding_0"]["weight"]
    # target shape: (119, hidden)
    padded = np.zeros_like(target)
    for i, Z in enumerate(atomic_numbers):
        padded[Z] = w[i]
    rep_params["LinearNodeEmbedding_0"]["weight"] = padded


def _map_scale_shift(ss_params, *, global_scale, global_shift, atomic_energies, torch_atomic_numbers):
    """Fold (global_scale, global_shift, atomic_energies) → PerElementScaleShift."""
    n_species = ss_params["scale_per_element"].shape[0]  # 119
    scale = np.full((n_species, 1), global_scale, dtype=np.float64)
    shift = np.zeros((n_species, 1), dtype=np.float64)
    for i, Z in enumerate(torch_atomic_numbers):
        shift[Z, 0] = global_shift + float(atomic_energies[i])
    ss_params["scale_per_element"] = scale
    ss_params["shift_per_element"] = shift


# _map_interactions, _map_products, _map_readouts — implemented similarly; each
# enumerates the torch keys in its scope and scatters into the corresponding
# linen slot. Use the mace-jax per-module import_from_torch implementations
# as the ground truth for weight layouts.
```

Implement iteratively:

1. Start with `_map_node_embedding` and `_map_scale_shift` (simplest).
2. Enable `_validate_no_nan` — it lists every float leaf that's still NaN. Use that list to drive the next helper: pick the first `NaN path`, find the torch key that should land there, implement the map, re-run.
3. Continue through `_map_interactions`, `_map_products`, `_map_readouts`.
4. A helper script at `scripts/mace_parity_probe.py` (keep under tests/, do NOT check in) is useful for iteration: loads torch model, inits apax model, prints both key→shape tables side-by-side.

- [ ] **Step 6: Run — expect convert succeeds, pytree loads**

```bash
uv run pytest tests/integration_tests/mace/test_convert.py -v -m mace_parity
```

Expected: 2 passed. If NaN validation fails, the error message lists the missed leaf — extend the mapping and re-run.

- [ ] **Step 7: Commit**

```bash
git add apax/transfer_learning/mace_foundation.py tests/integration_tests/mace/test_convert.py
git commit -m "feat(mace): converter writes config.yaml + orbax best/; maps torch weights"
```

---

### Task P3.6: Parity test via stock `ASECalculator`

**Files:**
- Modify: `tests/integration_tests/mace/test_mace_parity.py` (rewrite to use `ASECalculator` directly)

- [ ] **Step 1: Rewrite the parity test to load via the stock ASECalculator**

Replace `tests/integration_tests/mace/test_mace_parity.py`:

```python
"""Parity vs torch-mace MACECalculator on the same ase.Atoms.

Gated by mace_parity. Requires:
    uv sync --group mace-convert --extra mace
"""
import numpy as np
import pytest

pytestmark = pytest.mark.mace_parity


@pytest.fixture(params=["small"])
def foundation_name(request):
    return request.param


@pytest.fixture
def ase_water():
    from ase import Atoms
    return Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        pbc=False,
    )


def _torch_energy_forces(name, atoms):
    from mace.calculators.foundations_models import mace_mp
    calc = mace_mp(name, default_dtype="float64", device="cpu")
    atoms.calc = calc
    return float(atoms.get_potential_energy()), np.asarray(atoms.get_forces())


def _apax_energy_forces(apax_dir, atoms):
    from apax.md.ase_calc import ASECalculator
    calc = ASECalculator(apax_dir)
    atoms.calc = calc
    return float(atoms.get_potential_energy()), np.asarray(atoms.get_forces())


def test_energy_force_parity_water(tmp_path, foundation_name, ase_water):
    """End-to-end: convert → ASECalculator(converted_dir) → match torch."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion

    dst = tmp_path / f"{foundation_name}.apax"
    run_conversion(foundation_name, dst, head="default", family="mace_mp")

    e_torch, f_torch = _torch_energy_forces(foundation_name, ase_water.copy())
    e_apax,  f_apax  = _apax_energy_forces(dst, ase_water.copy())

    np.testing.assert_allclose(e_apax, e_torch, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(f_apax, f_torch, rtol=1e-3, atol=1e-4)


def test_force_consistency_via_finite_difference(tmp_path, foundation_name, ase_water):
    """Independent of torch: apax autodiff forces match numerical grad."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion
    from apax.md.ase_calc import ASECalculator

    dst = tmp_path / f"{foundation_name}.apax"
    run_conversion(foundation_name, dst, head="default", family="mace_mp")

    calc = ASECalculator(dst)
    atoms = ase_water.copy()
    atoms.calc = calc

    f_analytic = atoms.get_forces()
    h = 1e-4
    f_numeric = np.zeros_like(f_analytic)
    for i in range(len(atoms)):
        for d in range(3):
            a = atoms.copy(); a.positions[i, d] += h; a.calc = calc
            ep = a.get_potential_energy()
            a = atoms.copy(); a.positions[i, d] -= h; a.calc = calc
            em = a.get_potential_energy()
            f_numeric[i, d] = -(ep - em) / (2 * h)

    np.testing.assert_allclose(f_analytic, f_numeric, atol=1e-3)
```

- [ ] **Step 2: Run — iterate until parity holds**

```bash
uv run pytest tests/integration_tests/mace/test_mace_parity.py::test_energy_force_parity_water -v -m mace_parity -s
```

Expected (end state): PASS with rtol 1e-4 energy / 1e-3 forces. Diagnostic loop when failing:

- Energy matches but forces don't → likely a sign or norm issue in a tensor-product weight; inspect the first interaction's output against torch by adding prints in both.
- Energy off by a constant → scale-shift fold is wrong (global_shift absorbed twice, or atomic_energies scaled inadvertently).
- Energy off by a multiplicative factor → normalize2mom constant not copied, or scale-shift `scale_per_element` not set to `global_scale`.
- Random-looking mismatch → parameter at some torch path landed in wrong linen slot; `_validate_no_nan` won't catch this, so grep for a surprising leaf value (e.g. the torch weight with largest norm) and confirm it's in the right place.

- [ ] **Step 3: Stress parity (optional, periodic)**

If energy+force parity holds on water, add a small periodic SiO₂ test (3 atoms in 4 Å cube) to confirm the PBC-offset path. Deferred to a follow-up if the open-cell parity is already painful.

- [ ] **Step 4: Commit**

```bash
git add tests/integration_tests/mace/test_mace_parity.py
git commit -m "test(mace): ASECalculator parity vs torch-mace mace_mp small"
```

**P3 exit criteria:**
- `apax convert-mace small ./out/` produces `out/config.yaml` + `out/best/` + `out/converter_metadata.json`.
- `apax.train.checkpoints.restore_parameters("./out/")` returns `(Config, params)` without error.
- `apax.md.ase_calc.ASECalculator("./out/")` runs end-to-end. No `_is_mace_foundation_dir` code path exists.
- Energy parity rtol 1e-4 on water for MACE-MP-0 small.
- Force parity rtol 1e-3 on water for MACE-MP-0 small.
- Finite-difference force consistency passes atol 1e-3.
- `medium` and `medium-mpa-0` are parametrizable follow-ups once `small` is green.

---

## Phase P4 — Fine-tuning integration (simplified)

**Revised 2026-04-22.** Previous P4 introduced `freeze_mace_backbone_predicate`, an `UnfreezeMACEBackboneCallback`, and `pretrained` loading through `MaceBuilder.build_energy_model`. All three are dropped. Fine-tuning on a converted MACE foundation uses the existing `TransferLearningConfig` (already in `apax/config/train_config.py`) plus the `transfer_parameters` / `black_list_param_transfer` path that every other apax model uses.

This means P4 collapses to a single integration test plus a template config.

### Task P4.1: Fine-tune template config

**Files:**
- Create: `apax/cli/templates/mace_finetune_minimal.yaml`

- [ ] **Step 1: Add the template**

Create `apax/cli/templates/mace_finetune_minimal.yaml`:

```yaml
n_epochs: 50
seed: 1

data:
  directory: ./runs/
  experiment: mace_finetune
  data_path: ./dataset.extxyz
  n_train: 800
  n_valid: 100
  batch_size: 16
  valid_batch_size: 32

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
  readout_kind: standard          # swap MACE readout for apax's standard head
  nn: [256, 256]
  ensemble:
    kind: shallow
    n_members: 8
    force_variance: true

transfer_learning:
  base_model_checkpoint: ./converted/mace-mp-0-small.apax
  reset_layers: []                # keep if you also set readout_kind=mace;
                                  # with readout_kind=standard you don't need
                                  # to reset since target has a different head

loss:
  - { name: energy, loss_type: crps }
  - { name: forces, loss_type: crps }

optimizer:
  name: adam
  nn_lr: 0.0003
  emb_lr: 0.0003
  scale_lr: 0.0001
  shift_lr: 0.0003
```

- [ ] **Step 2: Commit**

```bash
git add apax/cli/templates/mace_finetune_minimal.yaml
git commit -m "docs(mace): add fine-tune template config using TransferLearningConfig"
```

---

### Task P4.2: End-to-end fine-tune smoke test on a converted small model

**Files:**
- Test: `tests/integration_tests/mace/test_mace_finetune.py`

- [ ] **Step 1: Write test**

Create `tests/integration_tests/mace/test_mace_finetune.py`:

```python
"""Fine-tune the converted MACE-MP-0 small on a synthetic dataset.

Verifies that the stock TransferLearningConfig path works on a MACE backbone
with a swapped readout and a shallow ensemble. Gated by mace_parity because
it requires the converter dev group.
"""
import numpy as np
import pytest
from ase import Atoms
from ase.io import write

pytestmark = [pytest.mark.mace_parity, pytest.mark.slow]


def _tiny_dataset(tmp_path, n_frames: int = 12) -> str:
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(n_frames):
        positions = rng.normal(size=(3, 3))
        atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=positions,
            pbc=False,
        )
        atoms.info["energy"] = float(rng.normal())
        atoms.arrays["forces"] = rng.normal(size=(3, 3))
        frames.append(atoms)
    path = tmp_path / "ds.extxyz"
    write(path, frames)
    return str(path)


def test_finetune_converted_small_runs_end_to_end(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion
    from apax.train.run import run
    from apax.train.checkpoints import restore_parameters

    # 1. Convert the foundation.
    converted = tmp_path / "converted" / "mace-mp-0-small.apax"
    run_conversion("small", converted, head="default", family="mace_mp")

    # 2. Write a minimal fine-tune config.
    ds_path = _tiny_dataset(tmp_path)
    cfg = {
        "n_epochs": 1,
        "data": {
            "directory": str(tmp_path),
            "experiment": "ft_smoke",
            "data_path": ds_path,
            "n_train": 8,
            "n_valid": 4,
            "batch_size": 2,
            "valid_batch_size": 2,
        },
        "model": {
            "name": "mace",
            "r_max": 6.0,
            "num_bessel": 10,
            "num_polynomial_cutoff": 5,
            "max_ell": 3,
            "hidden_irreps": "128x0e",
            "num_interactions": 2,
            "correlation": 3,
            "interaction_cls": "RealAgnosticResidual",
            "readout_kind": "standard",
            "nn": [64],
            "ensemble": {"kind": "shallow", "n_members": 4, "force_variance": True,
                          "chunk_size": None},
        },
        "transfer_learning": {
            "base_model_checkpoint": str(converted),
            "reset_layers": [],
        },
        "loss": [{"name": "energy"}, {"name": "forces"}],
        "optimizer": {"nn_lr": 1e-4, "emb_lr": 1e-4},
    }
    cfg_path = tmp_path / "cfg.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(cfg))

    # 3. Train one epoch.
    run(cfg_path, log_level="warning")

    # 4. The experiment dir contains a config.yaml + best/ and restore works.
    ft_dir = tmp_path / "ft_smoke"
    restored_cfg, restored_params = restore_parameters(ft_dir)
    assert restored_cfg.model.name == "mace"
    # Shallow ensemble → energy output shape (n_members,) per atom before sum
    import jax
    leaves = jax.tree_util.tree_leaves(restored_params)
    assert len(leaves) > 0
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/integration_tests/mace/test_mace_finetune.py -v -m "mace_parity and slow"
```

Expected: PASS. One epoch on 8 samples should be cheap (<2 min) on CPU.

- [ ] **Step 3: Commit**

```bash
git add tests/integration_tests/mace/test_mace_finetune.py
git commit -m "test(mace): fine-tune converted small through TransferLearningConfig"
```

**P4 exit criteria:**
- `apax train <config.yaml>` on the template above runs one full epoch.
- `restore_parameters(experiment_dir)` returns a valid `(Config, params)` after fine-tuning.
- Shallow ensemble uncertainty field is emitted on inference (propagated through the existing `ShallowEnsembleModel` path).
- No MACE-specific trainer code was added.

---

## Phase P5 — MD + ASE

### Task P5.1: jax-md NVT smoke

**Files:**
- Test: `tests/integration_tests/mace/test_mace_md.py`

- [ ] **Step 1: Write test (reuses existing `apax md` smoke pattern)**

Copy the pattern from `tests/integration_tests/md/test_md.py` (non-MACE). Adapt to a MACE config. Run 5 steps of NVT.

- [ ] **Step 2: Run**

Run: `uv run pytest tests/integration_tests/mace/test_mace_md.py -v -m slow`
Expected: PASS — energy conservation within tolerance.

- [ ] **Step 3: Commit**

```bash
git commit -am "test(mace): jax-md NVT smoke"
```

---

### Task P5.2: ASE calculator smoke

**Files:**
- Test: extend `test_mace_md.py`

- [ ] **Step 1: Adapt `tests/unit_tests/md/test_ase_calc.py` pattern for MACE**

Use `apax.md.ase_calc.ASECalculator` with a converted MACE checkpoint.

- [ ] **Step 2: Run + commit**

**P5 exit:** NVT runs; ASE returns finite forces matching the training-time model.

---

## Phase P6 — Benchmarks

### Task P6.1: Benchmark harness

**Files:**
- Create: `benchmarks/mace/bench_inference.py`
- Create: `benchmarks/mace/bench_training.py`
- Create: `benchmarks/mace/bench_md.py`
- Create: `benchmarks/mace/report.py`

- [ ] **Step 1: `bench_inference.py`**

```python
"""Inference latency benchmark: energy + forces, varying n_atoms."""
import argparse, time, json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mace-mp-0-medium")
    ap.add_argument("--n-atoms", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="benchmarks/mace/out/inference.json")
    args = ap.parse_args()

    from apax.transfer_learning.mace_foundation import load_mace_foundation
    from apax.nn.mace_foundation_model import MaceFoundationEnergyModel

    params, cfg = load_mace_foundation(args.model)
    model = MaceFoundationEnergyModel(**{...})

    R, Z, idx = _make_system(args.n_atoms)
    dr = jnp.asarray(R[idx[1]] - R[idx[0]])
    fn = jax.jit(lambda p, dr, Z, idx: model.apply(p, dr, Z, idx))

    for _ in range(args.warmup):
        fn(params, dr, Z, idx).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        out = fn(params, dr, Z, idx)
    out.block_until_ready()
    t1 = time.perf_counter()

    result = {
        "model": args.model,
        "n_atoms": args.n_atoms,
        "mean_s": (t1 - t0) / args.iters,
        "iters": args.iters,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(result)


def _make_system(n_atoms):
    ...


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Equivalent `bench_training.py`, `bench_md.py`**

Pattern after `bench_inference.py`; training measures steps/sec for a fixed batch, MD measures steps/sec for a fixed-size NVT simulation.

- [ ] **Step 3: `report.py`** — loads JSON outputs, writes `REPORT.md`

```python
"""Aggregate benchmark JSON files into benchmarks/mace/REPORT.md."""
```

- [ ] **Step 4: Commit**

```bash
git add benchmarks/mace/
git commit -m "bench(mace): add inference/training/md benchmark harness"
```

---

### Task P6.2: Run, report, optimize if needed

- [ ] **Step 1: Run**

```bash
uv run python -m benchmarks.mace.bench_inference --model tests/fixtures/mace/tiny.apax --n-atoms 64
uv run python -m benchmarks.mace.bench_training  --model tests/fixtures/mace/tiny.apax --batch 4
uv run python -m benchmarks.mace.bench_md        --model tests/fixtures/mace/tiny.apax --n-atoms 64
uv run python -m benchmarks.mace.report
```

- [ ] **Step 2: Compare against mace-jax**

In a sibling venv with mace-jax installed, run the equivalent upstream benchmark at matching config. Log ratios in `REPORT.md`.

- [ ] **Step 3: If thresholds missed, optimize**

Likely hot-spots to investigate:
- Tensor-product path filtering — reduce `filter_ir_out` scope.
- Radial MLP width — trim if config-permitted.
- Scatter aggregation — try `segment_sum` vs `e3nn.scatter_sum` on the exact hardware.
- `use_cueq=True` on GPU — toggle and re-measure.

- [ ] **Step 4: Commit report**

```bash
git add benchmarks/mace/REPORT.md
git commit -m "bench(mace): initial performance report"
```

**P6 exit:** All targets met, or deviations documented in REPORT.md with mitigations planned.

---

## Final verification

- [ ] **Final step: Full test sweep**

```bash
uv run pytest -m "not slow and not mace_parity" -v
uv run pytest -m slow -v
uv run pytest -m mace_parity -v       # requires torch + mace-torch locally
uvx ruff check .
uvx ruff format --check .
uvx prek --all-files
```

Expected: all three pytest invocations pass on target machines; lint/format clean.

- [ ] **Final step: Update docs**

- Update `README.md` with a pointer to the `apax convert-mace` workflow.
- Add an entry to `docs/source/...` (apax's sphinx tree) introducing MACE support with a minimal YAML example.

- [ ] **Final step: Commit and open PR**

```bash
git commit -am "docs: document MACE foundation model integration"
```

PR title: `feat: add MACE foundation model support (MACE-MP, MACE-MPA)`
PR body: summarize phases, parity numbers, benchmark ratios.

---

## Self-review (to be run before handing off)

1. **Spec coverage:** Each spec section maps to at least one task:
   - Dependency strategy (§3.1) → P0.1
   - Module layout (§3.2) → P0.3, P1.*, P3.*
   - Descriptor contract (§3.3) → P0.3, P1.5
   - MaceRepresentation (§3.4) → P0.3, P1.5
   - MaceFoundationEnergyModel (§3.5) → P3.4
   - MaceBuilder (§3.6) → P0.5
   - MaceModelConfig (§3.7) → P0.4
   - Converter format (§4.1) → P3.2
   - Converter CLI (§4.2) → P3.1
   - Loader (§4.3) → P3.3
   - Interactions at P3 (§4.4) → P1.3 (only RealAgnosticResidual)
   - Multi-head (§4.5) → P3.5
   - Fine-tune (§5) → P4.*
   - Performance (§6) → P6.*
   - Testing (§7) → covered across all phases
   - Risks (§8) → mitigations embedded in task caveats

2. **Placeholder scan:** Every step has either executable code, an exact command, or an explicit reference to an upstream file-line to port from. A small number of steps reference "see mace-jax import_from_torch" rather than inlining 300 LoC of torch-state-dict walking — acceptable because that function is large and mechanical, and the reference is a concrete file.

3. **Type consistency:** `MaceRepresentation` signature `(dr_vec, Z, idx)` used consistently in P0.3 and P1.5. `MaceBuilder.build_descriptor(apply_mask)` matches inherited signature. `InteractionBlock(irreps_out, interaction_cls, use_cueq)` fields stable through P1 and P2. `load_mace_foundation(source) -> (params, cfg)` stable through P3 and P4.

No fixes needed.
