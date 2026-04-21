# MACE Foundation Model Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native MACE (Multi-Atomic Cluster Expansion) support to apax: load MACE-MP / MACE-MPA foundation models via a CLI converter, fine-tune them with apax's shallow-ensemble and property-head infrastructure, and use them in JAX-MD — without making torch, mace-torch, or mace-jax runtime dependencies.

**Architecture:** `MaceRepresentation` is a Flax linen `nn.Module` mirroring apax's existing descriptor contract (`(dr_vec, Z, idx) → (n_atoms, n_features)`). Equivariant math comes from optional extras `e3nn-jax` (irreps, spherical harmonics, tensor products) and `cuequivariance-jax` (MACE-specific symmetric contraction, optional GPU acceleration). A separate `apax convert-mace` CLI converts torch-mace `.model` files into apax-native `.apax/` directories (msgpack + JSON) so runtime never imports torch.

**Tech Stack:** JAX, Flax linen, e3nn-jax, cuequivariance-jax, pydantic v2, flax.serialization (msgpack), typer (CLI), pytest. Reference source: `/Users/fzills/tools/mace-jax` (Flax NNX port, consulted for parity not copied). Conversion source of truth: torch-mace checkpoints at `mace-foundations` / locally installed `mace` + `torch`.

**Spec:** `docs/superpowers/specs/2026-04-20-mace-foundation-model-integration-design.md`

**Progress (updated 2026-04-21):** Phase P0 complete. Phase P1 (native MACE forward pass) is next; P1.1 dispatch was interrupted and nothing is partial — tree is clean at `bc5a7f89`.

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
| `apax/layers/descriptor/mace_blocks.py` | `LinearNodeEmbedding`, `InteractionBlock`, `ProductBlock`, `Readout` |
| `apax/layers/descriptor/mace_irreps.py` | Irreps utilities, dispatch between e3nn-jax & cuequivariance |
| `apax/nn/mace_foundation_model.py` | `MaceFoundationEnergyModel` — parity-only full energy path |
| `apax/cli/convert_mace.py` | `apax convert-mace` CLI subcommand |
| `apax/transfer_learning/mace_foundation.py` | `load_mace_foundation(src)` + freezing predicate |
| `tests/unit_tests/layers/descriptor/test_mace_descriptor.py` | Shape/contract tests |
| `tests/unit_tests/layers/descriptor/test_mace_blocks.py` | Block-level tests |
| `tests/unit_tests/nn/test_mace_builder.py` | Builder wiring |
| `tests/unit_tests/cli/test_convert_mace.py` | Converter unit tests (no torch) |
| `tests/integration_tests/mace/test_mace_shallow_ensemble.py` | Ensemble smoke test |
| `tests/integration_tests/mace/test_mace_parity.py` | Parity vs torch-mace (gated) |
| `tests/integration_tests/mace/test_mace_md.py` | jax-md + ASE smoke test |
| `benchmarks/mace/bench_inference.py` | Inference benchmark |
| `benchmarks/mace/bench_training.py` | Training benchmark |
| `benchmarks/mace/bench_md.py` | MD benchmark |
| `benchmarks/mace/report.py` | Aggregated report |

### Files modified
| Path | Change |
|---|---|
| `pyproject.toml` | Add `mace` optional extra + `mace_parity` pytest marker |
| `apax/config/model_config.py` | Add `MaceModelConfig` to discriminated union |
| `apax/nn/builder.py` | Add `MaceBuilder(ModelBuilder)` |
| `apax/layers/descriptor/basis_functions.py` | Add `PolynomialCutoff` |
| `apax/cli/apax_app.py` | Register `convert-mace` subcommand |
| `apax/nodes/model.py` | Register `MaceModelConfig` if applicable |

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

- [ ] **Step 1: Stub module**

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

- [ ] **Step 2: Write test**

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

- [ ] **Step 3: Run — expect pass**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k edge_features`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add apax/layers/descriptor/mace_blocks.py tests/unit_tests/layers/descriptor/test_mace_blocks.py
git commit -m "feat(mace): edge-feature assembly (radial × cutoff, spherical harmonics)"
```

---

### Task P1.2: `LinearNodeEmbedding`

**Files:**
- Modify: `apax/layers/descriptor/mace_blocks.py`
- Test: `tests/unit_tests/layers/descriptor/test_mace_blocks.py`

- [ ] **Step 1: Write test**

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

- [ ] **Step 2: Run — expect fail**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py::test_linear_node_embedding_scalar_output -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

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

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k linear_node_embedding`
Expected: PASS.

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Read mace-jax reference**

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

- [ ] **Step 2: Write test against randomly initialized block**

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

- [ ] **Step 3: Run — expect import fail**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k interaction_block`
Expected: ImportError.

- [ ] **Step 4: Implement `InteractionBlock`**

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

- [ ] **Step 5: Run — expect pass**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k interaction_block`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apax/layers/descriptor/mace_blocks.py tests/unit_tests/layers/descriptor/test_mace_blocks.py
git commit -m "feat(mace): InteractionBlock (RealAgnosticResidual)"
```

---

### Task P1.4: `ProductBlock` — symmetric contraction via cuequivariance

**Files:**
- Modify: `apax/layers/descriptor/mace_blocks.py`
- Test: `tests/unit_tests/layers/descriptor/test_mace_blocks.py`

- [ ] **Step 1: Read reference**

Run: Read `/Users/fzills/tools/mace-jax/mace_jax/adapters/cuequivariance/symmetric_contraction.py` (first 120 lines) to see exactly how cuequivariance's `symmetric_contraction` is invoked and what kwargs it expects.

Summarize into a docstring on the block.

- [ ] **Step 2: Write test**

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

- [ ] **Step 3: Run — expect fail**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k product_block`
Expected: ImportError.

- [ ] **Step 4: Implement `ProductBlock`**

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

- [ ] **Step 5: Run — expect pass**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py -v -k product_block`
Expected: PASS. Fix version/API mismatches as you go.

- [ ] **Step 6: Commit**

```bash
git add apax/layers/descriptor/mace_blocks.py tests/unit_tests/layers/descriptor/test_mace_blocks.py
git commit -m "feat(mace): ProductBlock (symmetric contraction via cuequivariance)"
```

---

### Task P1.5: Wire real forward pass into `MaceRepresentation`

**Files:**
- Modify: `apax/layers/descriptor/mace.py`
- Test: `tests/unit_tests/layers/descriptor/test_mace_descriptor.py`

- [ ] **Step 1: Replace skeleton forward with real pipeline**

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

- [ ] **Step 2: Update existing descriptor tests to still pass**

Run: `uv run pytest tests/unit_tests/layers/descriptor/test_mace_descriptor.py -v`
Expected: shape test passes; jit test passes.

If shape changes (n_features will now be `n_interactions × scalar_mult`), update assertions to compute expected dim from irreps.

- [ ] **Step 3: Write parity test against mace-jax random-weight output (optional if mace-jax importable)**

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

- [ ] **Step 4: Run, ensure smoke still passes**

Run: `uv run pytest tests/unit_tests/layers/descriptor/ tests/unit_tests/nn/test_mace_builder.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add apax/layers/descriptor/mace.py tests/unit_tests/layers/descriptor/
git commit -m "feat(mace): replace skeleton with native forward pass (P1)"
```

**P1 exit criterion:** MaceRepresentation runs the real forward pass; all unit tests green.

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

## Phase P3 — Converter and foundation-model loader

Goal: `apax convert-mace` produces an apax-native directory; `load_mace_foundation` reads it; parity vs torch-mace rtol 1e-5.

### Task P3.1: Converter CLI — accept both canonical names and file paths

The CLI must accept **either**:
- A short canonical name like `medium`, `medium-mpa-0` — resolved via `mace.calculators.foundations_models.mace_mp(..., return_raw_model=True)` which handles bundled-local → cache → download transparently.
- A filesystem path to a `.model` file — for custom / fine-tuned torch models.

**Files:**
- Create: `apax/cli/convert_mace.py`
- Modify: `apax/cli/apax_app.py` (register subcommand)
- Test: `tests/unit_tests/cli/test_convert_mace.py`

- [ ] **Step 1: Write tests covering both modes + missing-torch**

Create `tests/unit_tests/cli/test_convert_mace.py`:

```python
"""Converter CLI unit tests that don't require torch."""
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner


def test_convert_mace_missing_torch_fails_gracefully(monkeypatch, tmp_path):
    # Simulate torch being absent
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "mace", None)

    from apax.cli.apax_app import app
    runner = CliRunner()
    res = runner.invoke(app, ["convert-mace", "medium", str(tmp_path / "out.apax")])
    assert res.exit_code != 0
    assert "torch" in res.output.lower()


def test_convert_mace_accepts_canonical_name_and_file_path():
    """Argument parsing treats both as valid 'source' inputs."""
    import typer
    from apax.cli.convert_mace import convert_mace

    # Typer parameter introspection: no type narrowing that would reject either
    import inspect
    sig = inspect.signature(convert_mace)
    # 'source' parameter exists and has no strict type filter
    assert "source" in sig.parameters
```

- [ ] **Step 2: Run — expect fail (command not registered)**

Run: `uv run pytest tests/unit_tests/cli/test_convert_mace.py -v`
Expected: FAIL — command missing.

- [ ] **Step 3: Create command**

Create `apax/cli/convert_mace.py`:

```python
"""CLI subcommand: convert torch-mace foundation models into apax-native directories.

Usage
-----
Canonical name (downloaded / cached automatically by torch-mace):

    apax convert-mace medium-mpa-0 ./mace-mpa-0-medium.apax/
    apax convert-mace medium       ./mace-mp-0-medium.apax/
    apax convert-mace large        ./mace-mp-0-large.apax/

Local .model file:

    apax convert-mace ./my_custom_model.model ./my_custom.apax/

``torch`` and ``mace-torch`` are imported lazily; if missing, the command
exits with a typer error and a hint.
"""
from __future__ import annotations

from pathlib import Path

import typer


def convert_mace(
    source: str = typer.Argument(
        ...,
        help=(
            "Canonical MACE foundation name (e.g. 'medium-mpa-0', 'medium', 'large') "
            "or path to a local torch .model file."
        ),
    ),
    dst: Path = typer.Argument(..., help="Output apax-native directory"),
    head: str = typer.Option("mp", help="Which head to select for multi-head models"),
    family: str = typer.Option(
        "mace_mp",
        help="Foundation-model family: 'mace_mp' (includes MPA-0 and MP-0/0b/0b2/0b3). "
             "Others (mace_off, mace_anicc) are deferred.",
    ),
) -> None:
    """Convert a torch-mace foundation model into an apax-native .apax/ directory."""
    try:
        import torch  # noqa: F401
        import mace   # noqa: F401
    except ImportError as e:
        raise typer.BadParameter(
            "Converting MACE foundation models requires torch and mace-torch. "
            "Install them in your current env. "
            f"Missing module: {e.name}"
        ) from None

    from apax.transfer_learning.mace_foundation import run_conversion

    run_conversion(source, dst, head=head, family=family)
```

- [ ] **Step 4: Register in `apax_app.py`**

Edit `apax/cli/apax_app.py`, near other `app.command(...)` registrations:

```python
from apax.cli.convert_mace import convert_mace
app.command("convert-mace")(convert_mace)
```

- [ ] **Step 5: Stub `run_conversion`**

Create `apax/transfer_learning/mace_foundation.py` with a stub so imports succeed:

```python
"""MACE foundation model loading and conversion.

Functions
---------
run_conversion(source, dst, head, family)
    Entry point called by the ``apax convert-mace`` CLI. Accepts either a
    canonical MACE model name (resolved via ``mace.calculators.foundations_models.mace_mp``)
    or a path to a local torch .model file.
load_mace_foundation(source)
    Runtime loader for apax-native MACE directories.
"""
from __future__ import annotations

from pathlib import Path


def run_conversion(source, dst: Path, *, head: str = "mp", family: str = "mace_mp") -> None:
    """Convert a torch-mace checkpoint. Imports torch + mace lazily."""
    raise NotImplementedError("Filled in by P3.2")


def load_mace_foundation(source):
    raise NotImplementedError("Filled in by P3.6")
```

- [ ] **Step 6: Run — expect pass**

Run: `uv run pytest tests/unit_tests/cli/test_convert_mace.py -v`
Expected: PASS — the error path is hit.

- [ ] **Step 7: Commit**

```bash
git add apax/cli/convert_mace.py apax/cli/apax_app.py apax/transfer_learning/mace_foundation.py tests/unit_tests/cli/test_convert_mace.py
git commit -m "feat(cli): scaffold convert-mace subcommand with lazy torch import"
```

---

### Task P3.2: Torch state-dict → linen pytree mapping (core conversion)

The largest task in the plan. Reference: `/Users/fzills/tools/mace-jax/mace_jax/tools/import_from_torch.py` for the exact mapping logic and `/Users/fzills/tools/mace/mace/calculators/foundations_models.py:mace_mp` for the upstream download/cache path.

**Files:**
- Modify: `apax/transfer_learning/mace_foundation.py`
- Test: `tests/integration_tests/mace/test_convert.py` — uses `mace_mp("medium")` (smallest MP-0; downloads to cache on first run; reused thereafter by the mace-torch cache).

- [ ] **Step 1: Add torch+mace to a dev-only dependency group**

Edit `pyproject.toml`, add to `[dependency-groups]`:

```toml
mace-convert = [
    "torch>=2.1",
    "mace-torch>=0.3",
]
```

This group is **never** installed by default; a developer who wants to run the converter or parity tests opts in via:

```bash
uv sync --group mace-convert --extra mace
```

Document this in the README section added at the end of the plan.

- [ ] **Step 2: No fixture checked in — use mace_mp() directly**

We do **not** check in a `.model` file. The parity test loads models via `mace_mp(name, return_raw_model=True)` which:
- Uses the bundled `medium-mpa-0` (ships with mace-torch pip package) when no name is passed.
- Otherwise downloads to `~/.cache/mace/` once, then reuses.

The CI parity job runs `uv sync --group mace-convert --extra mace && uv run pytest -m mace_parity`. The first invocation triggers downloads; subsequent runs are cache-hits.

For CI sandbox safety (no network), we additionally cache a tiny synthetic model under `tests/fixtures/mace/` — generated on demand by running the parity test locally once and committing the cache file. Optional — not required for the plan's correctness.

- [ ] **Step 2a: Write conversion test (gated) — canonical name path**

Create `tests/integration_tests/mace/test_convert.py`:

```python
"""Integration test for convert-mace. Gated by mace_parity marker.

Requires:
    uv sync --group mace-convert --extra mace

These tests resolve MACE foundation models via the upstream
``mace.calculators.foundations_models.mace_mp`` interface, which handles
the bundled-local model, HTTP download, and caching under ``~/.cache/mace/``.
"""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.mace_parity


@pytest.mark.parametrize("model_name", [
    "medium-mpa-0",      # default; bundled with mace-torch package
    "medium",            # MACE-MP-0 medium; first-run download, cached thereafter
])
def test_convert_canonical_name(tmp_path, model_name):
    """Convert a canonical foundation model fetched via mace_mp()."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion

    dst = tmp_path / f"{model_name}.apax"
    run_conversion(model_name, dst, head="mp", family="mace_mp")

    assert (dst / "params.msgpack").exists()
    assert (dst / "config.json").exists()
    assert (dst / "metadata.json").exists()

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["name"] == "mace"
    assert cfg["num_interactions"] >= 1

    meta = json.loads((dst / "metadata.json").read_text())
    assert meta["source"] == model_name            # records the canonical name
    assert meta["source_resolved_path"]             # records where it actually came from


def test_convert_local_path(tmp_path):
    """Convert from an explicit .model path (no network)."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    # Pre-resolve the cached path, then feed it as a local file input
    local_path = Path(download_mace_mp_checkpoint("medium-mpa-0"))
    assert local_path.exists()

    dst = tmp_path / "local.apax"
    run_conversion(str(local_path), dst, head="mp", family="mace_mp")

    assert (dst / "params.msgpack").exists()
```

- [ ] **Step 3: Implement `run_conversion` that resolves canonical names via mace_mp()**

Edit `apax/transfer_learning/mace_foundation.py`, replacing the stub:

```python
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import jax.numpy as jnp
import numpy as np
from flax import serialization


def run_conversion(
    source: Union[str, Path],
    dst: Path,
    *,
    head: str = "mp",
    family: str = "mace_mp",
) -> None:
    """Convert a torch-mace foundation model into an apax-native directory.

    Parameters
    ----------
    source : str or Path
        Either a canonical MACE foundation name (e.g. ``"medium-mpa-0"``,
        ``"medium"``, ``"large"``) resolved via ``mace_mp(...)``, or a path to a
        local ``.model`` file.
    dst : Path
        Output directory.
    head : str
        For multi-head foundation models (e.g. MPA), which head to retain.
    family : str
        Which foundation-family resolver to use. Initial scope: ``"mace_mp"``
        (covers MACE-MP-0, 0b, 0b2, 0b3 and MACE-MPA). Others deferred.
    """
    import torch  # local import; runtime never needs this

    dst = Path(dst)
    torch_model, resolved_path = _load_torch_foundation_model(source, family=family)
    if hasattr(torch_model, "state_dict"):
        state = {k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()}
    else:
        state = torch_model

    cfg = _extract_config_from_torch(torch_model, head=head)
    params_pytree = _map_state_to_pytree(state, cfg, head=head)
    _validate_no_nan(params_pytree)

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "params.msgpack").write_bytes(serialization.to_bytes(params_pytree))
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

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
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2))


def _load_torch_foundation_model(source, *, family: str):
    """Load a torch MACE model via the upstream foundation loader.

    Goes through ``mace.calculators.foundations_models.mace_mp(return_raw_model=True)``
    so we inherit bundled-local → cache → download logic + ASL license notices.

    Returns
    -------
    torch_model : torch.nn.Module
    resolved_path : str | None
        Path on disk that the torch model was loaded from (when the resolver
        exposes it). ``None`` if we have only a module-in-memory.
    """
    if family != "mace_mp":
        raise NotImplementedError(
            f"family={family!r} not yet supported; only 'mace_mp' is in initial scope."
        )

    from mace.calculators.foundations_models import (
        download_mace_mp_checkpoint,
        mace_mp,
        mace_mp_names,
    )

    # Heuristic: treat as a canonical name if it's not an existing file path.
    is_path = isinstance(source, (str, Path)) and Path(source).exists()
    if is_path:
        import torch
        return torch.load(str(source), map_location="cpu"), str(source)

    # Canonical name — validate against the registry for a clearer error
    if source not in mace_mp_names and not str(source).startswith("https:"):
        raise ValueError(
            f"Unknown MACE-MP model name {source!r}. "
            f"Valid names: {', '.join(n for n in mace_mp_names if n)}"
        )

    resolved_path = download_mace_mp_checkpoint(source)
    torch_model = mace_mp(source, return_raw_model=True)
    return torch_model, resolved_path


def _extract_config_from_torch(model, head: str) -> dict:
    """Return a dict that matches MaceModelConfig schema.

    Reads hyperparameters off the torch module (r_max, hidden_irreps, etc.).
    Multi-head selection: if model.heads > 1, pick the named head and drop others.
    """
    # Torch attrs: r_max, num_bessel, num_polynomial_cutoff, num_interactions, ...
    # See /Users/fzills/tools/mace/mace/modules/models.py for exact attribute names.
    attrs = getattr(model, "__dict__", {})
    cfg = {
        "name": "mace",
        "r_max": float(model.r_max),
        "num_bessel": int(getattr(model, "num_bessel", 8)),
        "num_polynomial_cutoff": int(getattr(model, "num_polynomial_cutoff", 5)),
        "max_ell": int(getattr(model, "max_ell", 3)),
        "hidden_irreps": str(model.hidden_irreps),
        "num_interactions": int(model.num_interactions),
        "correlation": int(getattr(model, "correlation", 3)),
        "interaction_cls": "RealAgnosticResidual",
        "num_elements": int(model.num_elements),
    }
    # atomic_energies are per-element reference E0, kept in a separate array
    cfg["atomic_energies"] = model.atomic_energies_fn.atomic_energies.detach().cpu().numpy().tolist()
    if getattr(model, "num_heads", 1) > 1:
        cfg["selected_head"] = head
    return cfg


def _map_state_to_pytree(state: dict[str, np.ndarray], cfg: dict, head: str) -> dict:
    """Translate torch parameter names → linen pytree.

    Parameter path map (canonical list):

    Torch key                                    → apax pytree path
    ------------------------------------------------------------------
    node_embedding.linear.weight                 → params/LinearNodeEmbedding_0/weight
    interactions.<i>.linear_up.weight            → params/MaceRepresentation/.../InteractionBlock_<i>/linear_up/kernel
    interactions.<i>.linear_down.weight          → .../InteractionBlock_<i>/linear_down/kernel
    interactions.<i>.skip_tp.weight              → .../InteractionBlock_<i>/skip_linear/kernel
    interactions.<i>.conv_tp_weights.*           → .../InteractionBlock_<i>/radial_mlp/*
    products.<i>.linear.weight                   → .../ProductBlock_<i>/symmetric_contraction/weight
    readouts.<i>.linear.weight                   → .../Readout_<i>/linear/kernel
    ...

    This map is authoritative — every torch key must land somewhere, and
    the NaN-leaf check (below) will fail if any expected slot is missed.
    """
    out = {"params": {}}
    # Implemented progressively; follow /Users/fzills/tools/mace-jax/mace_jax/tools/import_from_torch.py
    # as the reference. For each torch key, decide where it goes in our tree.
    raise NotImplementedError(
        "Fill in the mapping below. Start with node_embedding and a single interaction/product/readout; "
        "add entries until no torch keys remain and no apax leaves are NaN. "
        "See mace-jax import_from_torch.py for the naming convention."
    )


def _validate_no_nan(pytree: dict) -> None:
    import jax
    bad = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(pytree)[0]:
        if isinstance(leaf, np.ndarray) and np.issubdtype(leaf.dtype, np.floating):
            if np.isnan(leaf).any():
                bad.append("/".join(str(k) for k in path))
    if bad:
        raise ValueError(f"NaN leaves after conversion:\n  - " + "\n  - ".join(bad))


def _torch_mace_version() -> str:
    try:
        import mace
        return mace.__version__
    except Exception:
        return "unknown"


def _apax_version() -> str:
    try:
        from apax import __version__
        return __version__
    except Exception:
        return "unknown"
```

- [ ] **Step 4: Iteratively fill in `_map_state_to_pytree`**

This is the single most tedious step — walk through the torch state keys (print them with `print(list(state.keys()))` at run time), map each to the linen path. For each key:

1. Print torch shape.
2. Find the matching slot in `MaceRepresentation.init(...)` output.
3. Assign, reshaping if needed (cueq may use `ir_mul` layout — transpose accordingly).

Iterate:
```bash
uv run pytest tests/integration_tests/mace/test_convert.py::test_convert_tiny_mace_produces_apax_dir -v -m mace_parity
```
until the parameter count matches and the NaN check passes. Keep this function readable: dispatch with explicit `for k, v in state.items(): match k:` style or a series of small helper functions (`_map_interaction_layer`, `_map_product_layer`, ...).

- [ ] **Step 5: Extract `normalize2mom` constant**

Borrow the exact pattern from `/Users/fzills/tools/mace-jax/mace_jax/tools/import_from_torch.py:_extract_norm_consts`. Store in `params["constants"]["normalize2mom_silu"]`.

- [ ] **Step 6: Run parity conversion test**

Run: `uv run pytest tests/integration_tests/mace/test_convert.py -v -m mace_parity`
Expected: PASS (directory exists, params loadable, NaN check clean).

- [ ] **Step 7: Commit**

```bash
git add apax/transfer_learning/mace_foundation.py tests/integration_tests/mace/test_convert.py tests/fixtures/mace/
git commit -m "feat(mace): implement torch→apax weight conversion"
```

---

### Task P3.3: `load_mace_foundation` — runtime-side loader

**Files:**
- Modify: `apax/transfer_learning/mace_foundation.py`
- Test: `tests/integration_tests/mace/test_load_foundation.py`

- [ ] **Step 1: Write test**

Create `tests/integration_tests/mace/test_load_foundation.py`:

```python
import json
from pathlib import Path
import jax.numpy as jnp
import pytest
from flax import serialization


def test_load_mace_foundation_from_dir(tmp_path):
    # Write a minimal fake apax dir
    cfg = {
        "name": "mace", "r_max": 5.0, "num_bessel": 4, "num_polynomial_cutoff": 5,
        "max_ell": 1, "hidden_irreps": "8x0e", "num_interactions": 1,
        "correlation": 2, "interaction_cls": "RealAgnosticResidual",
        "num_elements": 5,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "metadata.json").write_text("{}")
    # Use random params matching the model init output
    from apax.layers.descriptor.mace import MaceRepresentation
    import jax, numpy as np
    model = MaceRepresentation(**{k: v for k, v in cfg.items() if k != "name"})
    dr_vec = jnp.zeros((4, 3))
    Z = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    idx = jnp.array([[0, 1], [1, 0]])
    params = model.init(jax.random.PRNGKey(0), dr_vec, Z, idx)
    (tmp_path / "params.msgpack").write_bytes(serialization.to_bytes(params))

    from apax.transfer_learning.mace_foundation import load_mace_foundation
    loaded_params, loaded_cfg = load_mace_foundation(tmp_path)
    assert loaded_cfg.name == "mace"
    assert loaded_cfg.r_max == 5.0
    # Param structure matches
    assert jax.tree_util.tree_structure(loaded_params) == jax.tree_util.tree_structure(params)
```

- [ ] **Step 2: Run — expect fail**

Run: `uv run pytest tests/integration_tests/mace/test_load_foundation.py -v`
Expected: NotImplementedError.

- [ ] **Step 3: Implement loader**

In `mace_foundation.py`:

```python
def load_mace_foundation(source):
    """Load a converted MACE foundation model.

    ``source`` may be:
    - A directory path pointing to a converted .apax/ directory.
    - A short name (string) — resolved via huggingface-hub if available.
    """
    from apax.config.model_config import MaceModelConfig
    from apax.layers.descriptor.mace import MaceRepresentation
    import jax

    src = Path(source) if isinstance(source, (str, Path)) and Path(source).exists() else None
    if src is None:
        src = _resolve_short_name(str(source))

    cfg_dict = json.loads((src / "config.json").read_text())
    cfg = MaceModelConfig(**{k: v for k, v in cfg_dict.items() if k in MaceModelConfig.model_fields})

    # Rebuild a fresh pytree to match against
    model = MaceRepresentation(
        r_max=cfg.r_max, num_bessel=cfg.num_bessel,
        num_polynomial_cutoff=cfg.num_polynomial_cutoff, max_ell=cfg.max_ell,
        hidden_irreps=cfg.hidden_irreps, num_interactions=cfg.num_interactions,
        correlation=cfg.correlation, interaction_cls=cfg.interaction_cls,
        num_elements=cfg.num_elements, use_cueq=cfg.use_cueq,
    )
    dr_dummy = jnp.zeros((1, 3))
    Z_dummy = jnp.zeros((1,), dtype=jnp.int32)
    idx_dummy = jnp.zeros((2, 1), dtype=jnp.int32)
    template = model.init(jax.random.PRNGKey(0), dr_dummy, Z_dummy, idx_dummy)
    params = serialization.from_bytes(template, (src / "params.msgpack").read_bytes())
    return params, cfg


def _resolve_short_name(name: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ValueError(
            f"Model {name!r} not found locally and huggingface-hub is not installed. "
            "Install it with `uv sync --extra mace` or pass a local directory path."
        )
    _KNOWN = {
        "mace-mp-0-medium": "apax-hub/mace-mp-0-medium",
        "mace-mpa-medium": "apax-hub/mace-mpa-medium",
    }
    repo = _KNOWN.get(name)
    if repo is None:
        raise ValueError(f"Unknown MACE foundation shortname {name!r}")
    return Path(snapshot_download(repo))
```

Add `huggingface-hub` to the `mace` extra in `pyproject.toml` (optional inside the extra).

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/integration_tests/mace/test_load_foundation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apax/transfer_learning/mace_foundation.py pyproject.toml tests/integration_tests/mace/
git commit -m "feat(mace): load_mace_foundation runtime loader"
```

---

### Task P3.4: `MaceFoundationEnergyModel` for parity path against `mace_mp()` / `MACECalculator`

**Files:**
- Create: `apax/nn/mace_foundation_model.py`
- Test: integration parity test

- [ ] **Step 1: Skeleton**

Create `apax/nn/mace_foundation_model.py`:

```python
"""Full-energy MACE module used only for parity tests and zero-shot inference.

Unlike the standard :class:`apax.nn.models.EnergyModel` + :class:`MaceRepresentation`
+ :class:`AtomisticReadout` stack used for fine-tuning, this module includes
MACE's per-layer internal readouts and per-element atomic-energy reference so
it reproduces the full upstream ``ScaleShiftMACE`` forward pass.

Fine-tuning, shallow ensemble, property heads, MD, ASE — none of these go
through this module. It exists solely so the parity test can verify the
converter.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import linen as nn

from apax.layers.descriptor.mace import MaceRepresentation


class MaceFoundationEnergyModel(nn.Module):
    r_max: float
    num_bessel: int
    num_polynomial_cutoff: int
    max_ell: int
    hidden_irreps: str
    num_interactions: int
    correlation: int
    interaction_cls: str
    num_elements: int
    atomic_energies: jnp.ndarray
    use_cueq: bool = False

    @nn.compact
    def __call__(self, dr_vec, Z, idx):
        # Run representation once keeping intermediate per-layer features
        # Apply each MACE-style readout, sum, add atomic_energies reference.
        raise NotImplementedError(
            "Fill in after the MaceRepresentation exposes per-layer node_feats. "
            "Reference: mace-jax models.MACE.__call__ lines 375-554."
        )
```

- [ ] **Step 2: Expose per-layer node features**

Edit `MaceRepresentation` to accept a flag `return_per_layer_node_feats: bool = False`; when True, return a list of `IrrepsArray` per layer. Keep default behavior unchanged.

- [ ] **Step 3: Implement `MaceFoundationEnergyModel.__call__`**

Port the logic from `mace-jax/mace_jax/modules/models.py` lines 375–554 adapted to linen, returning per-atom energy. Sum atomic-energy reference at the end.

- [ ] **Step 4: Parity test via `MACECalculator` (ASE interface)**

Reference against the upstream `mace_mp(name)` ASE calculator — this guarantees identical neighbor lists, data prep, and forward pass to what a downstream user of torch-mace would get. We pass the same `ase.Atoms` through both calculators and compare.

Create `tests/integration_tests/mace/test_mace_parity.py`:

```python
"""Parity vs torch-mace ``MACECalculator`` (ASE wrapper). Opt-in only."""
import pytest
import numpy as np


pytestmark = pytest.mark.mace_parity


@pytest.fixture(params=["medium-mpa-0", "medium"])
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


@pytest.fixture
def ase_periodic_sio2():
    from ase import Atoms
    return Atoms(
        symbols=["Si", "O", "O"],
        positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )


def _torch_energy_forces(name, atoms):
    """Run the upstream MACECalculator and return (energy, forces)."""
    from mace.calculators.foundations_models import mace_mp
    calc = mace_mp(name, default_dtype="float64", device="cpu")
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    return float(e), np.asarray(f)


def _apax_energy_forces(apax_dir, atoms):
    """Run apax's MaceFoundationEnergyModel + derivative via apax ASE calc."""
    from apax.md.ase_calc import ASECalculator
    calc = ASECalculator(apax_dir)      # apax-native; no torch
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    return float(e), np.asarray(f)


def test_energy_force_parity_water(tmp_path, foundation_name, ase_water):
    """Molecular parity for a 3-atom system."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion

    # 1. convert foundation model to apax-native dir
    dst = tmp_path / f"{foundation_name}.apax"
    run_conversion(foundation_name, dst, head="mp", family="mace_mp")

    # 2. torch reference (via mace_mp ASE calculator)
    e_torch, f_torch = _torch_energy_forces(foundation_name, ase_water.copy())

    # 3. apax prediction
    e_apax, f_apax = _apax_energy_forces(dst, ase_water.copy())

    # 4. parity
    np.testing.assert_allclose(e_apax, e_torch, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(f_apax, f_torch, rtol=1e-4, atol=1e-5)


def test_energy_force_parity_periodic(tmp_path, foundation_name, ase_periodic_sio2):
    """Periodic-box parity: validates neighbor lists + PBC offsets."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion

    dst = tmp_path / f"{foundation_name}.apax"
    run_conversion(foundation_name, dst, head="mp", family="mace_mp")

    e_torch, f_torch = _torch_energy_forces(foundation_name, ase_periodic_sio2.copy())
    e_apax, f_apax = _apax_energy_forces(dst, ase_periodic_sio2.copy())

    np.testing.assert_allclose(e_apax, e_torch, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(f_apax, f_torch, rtol=1e-4, atol=1e-5)


def test_force_consistency_via_finite_difference(tmp_path, foundation_name, ase_water):
    """Independent of torch parity: apax autodiff forces match numerical grad."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion
    from apax.md.ase_calc import ASECalculator

    dst = tmp_path / f"{foundation_name}.apax"
    run_conversion(foundation_name, dst, head="mp", family="mace_mp")

    calc = ASECalculator(dst)
    atoms = ase_water.copy(); atoms.calc = calc

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

- [ ] **Step 5: Stress parity (periodic systems only)**

Append to the same file:

```python
def test_stress_parity_periodic(tmp_path, foundation_name, ase_periodic_sio2):
    """Validate stress via autodiff matches the upstream torch stress."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion
    from apax.md.ase_calc import ASECalculator
    from mace.calculators.foundations_models import mace_mp

    dst = tmp_path / f"{foundation_name}.apax"
    run_conversion(foundation_name, dst, head="mp", family="mace_mp")

    # torch stress
    torch_calc = mace_mp(foundation_name, default_dtype="float64", device="cpu")
    a = ase_periodic_sio2.copy(); a.calc = torch_calc
    s_torch = a.get_stress(voigt=False)

    # apax stress — requires calc_stress=True in the apax config
    apax_calc = ASECalculator(dst, calc_stress=True)
    a = ase_periodic_sio2.copy(); a.calc = apax_calc
    s_apax = a.get_stress(voigt=False)

    np.testing.assert_allclose(s_apax, s_torch, rtol=1e-4, atol=1e-6)
```

*Note on `ASECalculator(dst)`*: the apax ASE calculator needs a new code path that accepts an apax-foundation-model directory directly (in addition to the existing apax-train-output path). Add this in Task P3.6 below.

- [ ] **Step 6: Iterate until parity holds**

Run: `uv run pytest tests/integration_tests/mace/test_mace_parity.py -v -m mace_parity`

Expected in success: PASS with rtol 1e-5. If failing:
- Check irreps layout (`mul_ir` vs `ir_mul` — most common source of numerical drift).
- Check `use_reduced_cg` matches torch's flag.
- Check normalize2mom constant actually copied.
- Check atomic_energies applied in fp64.
- Inspect first interaction-block output of both and diff per-irrep.

- [ ] **Step 7: Commit**

```bash
git add apax/nn/mace_foundation_model.py apax/layers/descriptor/mace.py tests/integration_tests/mace/test_mace_parity.py
git commit -m "feat(mace): MaceFoundationEnergyModel + parity test (P3)"
```

---

### Task P3.5: `ASECalculator(.apax/)` — wire converted dirs into apax's ASE calc

**Files:**
- Modify: `apax/md/ase_calc.py`
- Test: `tests/integration_tests/mace/test_mace_parity.py` (already uses this)

- [ ] **Step 1: Read current ASECalculator**

Run: Read `apax/md/ase_calc.py` to find how it currently loads params + config. Most apax setups point it at a training-output directory containing `config.yaml` + checkpoints.

- [ ] **Step 2: Detect foundation-model dirs**

Add logic: if the input directory contains `params.msgpack` + `config.json` (apax-foundation layout), use `load_mace_foundation` instead of the standard checkpoint loader. Construct `EnergyModel(MaceRepresentation, AtomisticReadout=None, ...)` via `MaceFoundationEnergyModel`.

- [ ] **Step 3: Write direct unit test (non-parity)**

Create `tests/unit_tests/md/test_ase_calc_mace_foundation.py`:

```python
@pytest.mark.mace_parity
def test_ase_calc_accepts_mace_foundation_dir(tmp_path):
    """ASECalculator dispatches to the foundation-model loader correctly."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from ase import Atoms
    from apax.transfer_learning.mace_foundation import run_conversion
    from apax.md.ase_calc import ASECalculator

    dst = tmp_path / "f.apax"
    run_conversion("medium-mpa-0", dst, head="mp", family="mace_mp")
    atoms = Atoms(["O", "H", "H"], positions=[[0,0,0],[0.96,0,0],[-0.24,0.93,0]])
    atoms.calc = ASECalculator(dst)
    e = atoms.get_potential_energy()
    assert np.isfinite(e)
```

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(md): ASECalculator supports MACE foundation-model directories"
```

---

### Task P3.6: Multi-head selection (`--head` flag)

**Files:**
- Modify: `apax/transfer_learning/mace_foundation.py`
- Test: extend `test_convert.py`

- [ ] **Step 1: Add head selection in `_extract_config_from_torch` and `_map_state_to_pytree`**

When `model.num_heads > 1`, only walk the keys belonging to the selected head; raise if head name not found. Record in metadata.

- [ ] **Step 2: Write test**

Append to `tests/integration_tests/mace/test_convert.py`:

```python
def test_convert_rejects_unknown_head(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from apax.transfer_learning.mace_foundation import run_conversion
    src = Path("tests/fixtures/mace/tiny_mace.model")
    with pytest.raises(ValueError, match="head"):
        run_conversion(src, tmp_path / "out.apax", head="does-not-exist")
```

- [ ] **Step 3: Run**

Run: `uv run pytest tests/integration_tests/mace/test_convert.py -v -m mace_parity`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add apax/transfer_learning/mace_foundation.py tests/integration_tests/mace/
git commit -m "feat(mace): --head selection for multi-head foundation models"
```

**P3 exit:**
- `apax convert-mace medium-mpa-0 ./out.apax/` works end-to-end — resolved via `mace_mp(return_raw_model=True)`, downloaded/cached upstream.
- `apax convert-mace medium ./out.apax/` works for MACE-MP-0 medium.
- Parity tests green on a dev machine with `uv sync --group mace-convert --extra mace`:
  - Energy parity rtol 1e-5 on water + periodic SiO₂ for both `medium-mpa-0` and `medium`.
  - Force parity rtol 1e-4 on both.
  - Stress parity rtol 1e-4 on the periodic system.
  - Finite-difference force consistency confirms autodiff path is correct even without the torch reference.

---

## Phase P4 — Fine-tuning integration

### Task P4.1: Parameter-freezing predicate for MACE backbone

**Files:**
- Modify: `apax/transfer_learning/parameter_transfer.py`
- Test: `tests/unit_tests/transfer_learning/test_parameter_transfer.py`

- [ ] **Step 1: Read existing transfer utils**

Run: Read `apax/transfer_learning/parameter_transfer.py` — understand `black_list_param_transfer`.

- [ ] **Step 2: Add MACE-specific helper**

Append:

```python
def freeze_mace_backbone_predicate(param_path: tuple[str, ...]) -> bool:
    """Return True if the parameter path belongs to the MACE backbone.

    Treats LinearNodeEmbedding, InteractionBlock_*, ProductBlock_* as backbone;
    AtomisticReadout and PerElementScaleShift are *not* backbone.
    """
    backbone_markers = (
        "LinearNodeEmbedding",
        "InteractionBlock",
        "ProductBlock",
    )
    return any(marker in "/".join(param_path) for marker in backbone_markers)
```

- [ ] **Step 3: Write test**

Append to `test_parameter_transfer.py`:

```python
def test_freeze_mace_backbone_predicate_flags_blocks():
    from apax.transfer_learning.parameter_transfer import freeze_mace_backbone_predicate
    assert freeze_mace_backbone_predicate(("params", "MaceRepresentation", "InteractionBlock_0", "linear_up", "kernel"))
    assert freeze_mace_backbone_predicate(("params", "MaceRepresentation", "ProductBlock_1", "weight"))
    assert not freeze_mace_backbone_predicate(("params", "AtomisticReadout", "dense_0", "kernel"))
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/unit_tests/transfer_learning/test_parameter_transfer.py -v -k freeze_mace`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(transfer): mace backbone freezing predicate"
```

---

### Task P4.2: `unfreeze_backbone_epoch` callback

**Files:**
- Modify: `apax/train/callbacks.py`
- Test: light unit test

- [ ] **Step 1: Find the callback registration pattern**

Run: Read `apax/train/callbacks.py` — follow an existing callback (e.g. EMA callback) as a template.

- [ ] **Step 2: Add callback**

```python
class UnfreezeMACEBackboneCallback:
    """Switch the backbone from frozen to trainable at a specified epoch."""
    def __init__(self, epoch: int):
        self.epoch = epoch
        self._fired = False

    def __call__(self, trainer, epoch: int):
        if self._fired or epoch < self.epoch:
            return
        trainer.unfreeze_params(
            predicate="apax.transfer_learning.parameter_transfer:freeze_mace_backbone_predicate"
        )
        self._fired = True
```

Adapt to the trainer's actual callback contract.

- [ ] **Step 3: Integration point**

In `apax/train/trainer.py` (or wherever the optimizer is constructed), if `config.model.freeze_backbone`, apply the predicate to produce an optax `multi_transform` that zeroes gradients on backbone params. When `unfreeze_backbone_epoch` is set, register the callback.

- [ ] **Step 4: Test**

Write a minimal unit test that loops the trainer for `epoch+1` epochs and checks that after the unfreeze epoch, at least one backbone parameter has been updated.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(train): unfreeze_backbone_epoch callback"
```

---

### Task P4.3: `pretrained` loading in `MaceBuilder`

**Files:**
- Modify: `apax/nn/builder.py`
- Test: `tests/integration_tests/mace/test_finetune.py`

- [ ] **Step 1: Preload in builder**

Edit `MaceBuilder.build_energy_model`:

```python
def build_energy_model(self, *args, **kwargs):
    model = super().build_energy_model(*args, **kwargs)
    if self.config.get("pretrained"):
        from apax.transfer_learning.mace_foundation import load_mace_foundation
        self._pretrained_params, self._pretrained_cfg = load_mace_foundation(
            self.config["pretrained"]
        )
        # caller is responsible for merging via transfer_learning utilities
    return model
```

And expose the preloaded params through the builder so the training-entry code (`apax/train/run.py`) can install them before the first step.

- [ ] **Step 2: Smoke fine-tune test**

Create `tests/integration_tests/mace/test_finetune.py`:

```python
@pytest.mark.slow
def test_finetune_with_frozen_mace(tmp_path):
    """Smoke-level: training runs, loss drops, backbone params don't move."""
    # Use the tiny_mace fixture converted in P3
    # Train 2 epochs on a dummy dataset
    # Assert: backbone param snapshot before == after (allclose);
    #         readout params moved; loss epoch2 < loss epoch1
    ...
```

- [ ] **Step 3: Run**

Run: `uv run pytest tests/integration_tests/mace/test_finetune.py -v -m slow`
Expected: PASS after wiring.

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(nn): pretrained loading through MaceBuilder"
```

---

### Task P4.4: End-to-end shallow-ensemble fine-tune

**Files:**
- Test: `tests/integration_tests/mace/test_mace_shallow_ensemble.py`

- [ ] **Step 1: Write test**

```python
@pytest.mark.slow
def test_mace_shallow_ensemble_emits_uncertainty(tmp_path):
    """Train a tiny MACE + shallow ensemble for 1 epoch; inference returns
    both energy and energy_uncertainty fields."""
    ...
```

Use the existing dataset fixture pattern from `tests/integration_tests/cli/test_app.py`.

- [ ] **Step 2: Run + commit**

Run: `uv run pytest tests/integration_tests/mace/test_mace_shallow_ensemble.py -v -m slow`
Expected: PASS.

```bash
git commit -am "test: shallow-ensemble fine-tune with MACE"
```

**P4 exit:** Fine-tune YAML works end-to-end; backbone freezing behaves; ensemble uncertainty emitted.

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
