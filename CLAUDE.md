# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`apax` is a JAX-based framework for training and deploying atomistic machine-learned interatomic potentials (MLIPs). It implements the Gaussian Moment Neural Network (GMNN) model and supports multiple architectures, molecular dynamics, and active learning.

## Commands

```bash
# Run tests (excluding slow ones)
uv run coverage run -m pytest -k "not slow"

# Run a single test file
uv run pytest tests/unit_tests/layers/test_descriptor.py

# Run tests with a specific marker or keyword
uv run pytest -k "test_name_pattern"

# Lint / format
uv run ruff check apax/
uv run ruff format apax/

# Train a model
apax train config.yaml

# Run MD simulation
apax md config.yaml md_config.yaml

# Validate config files
apax validate train config.yaml
apax validate md md_config.yaml

# Generate JSON schema for YAML autocompletion
apax schema

# Generate config templates
apax template train        # minimal template
apax template train --full # all options
apax template md
```

## Architecture

### Configuration system (`apax/config/`)
All configuration is done via YAML files parsed into Pydantic models. The main configs are:
- `Config` (`train_config.py`) — top-level training config with nested `DataConfig`, `ModelConfig`, `OptimizerConfig`, `LossConfig`, etc.
- `MDConfig` (`md_config.py`) — molecular dynamics config
- `parse_config()` in `common.py` — unified entry point that accepts a path or dict

`ModelConfig` is a union discriminated on `name`: `"gmnn"` → `GMNNConfig`, `"equiv-mp"` → `EquivMPConfig`, `"so3krates"` → `So3kratesConfig`.

### Model construction (`apax/nn/`)
Models are built via the Builder pattern:
1. `ModelConfig.get_builder()` returns the appropriate `ModelBuilder` subclass (`GMNNBuilder`, `EquivMPBuilder`, `So3kratesBuilder`)
2. The builder assembles Flax `nn.Module` components: `descriptor → readout → scale_shift → corrections → property_heads`
3. `builder.build_energy_derivative_model()` wraps the core in `EnergyDerivativeModel` (or `ShallowEnsembleModel` for shallow ensembles), which computes forces via `jax.grad` and optionally stress

Key Flax modules in `apax/nn/models.py`:
- `EnergyModel` — descriptor + readout + scale/shift + empirical corrections
- `EnergyDerivativeModel` — wraps `EnergyModel`, adds force/stress via autodiff
- `ShallowEnsembleModel` — last-layer ensemble with uncertainty quantification
- `FeatureModel` — descriptor + readout without energy head (for active learning)

### Descriptor/layers (`apax/layers/`)
- `apax/layers/descriptor/` — `GaussianMomentDescriptor` (GMNN), `EquivMPRepresentation`, `So3kratesRepresentation`; all take `(dr_vec, Z, idx)` and return per-atom features
- `apax/layers/distances.py` — neighbor distance computation, handles periodic boundary conditions
- `apax/layers/readout.py` — `AtomisticReadout`: per-atom MLP mapping features to scalar energies
- `apax/layers/scaling.py` — `PerElementScaleShift`: per-species learned scale and shift
- `apax/layers/empirical.py` — optional corrections (ZBL, exponential repulsion, latent Ewald)
- `apax/layers/properties.py` — `PropertyHead` for additional per-atom or global properties (charges, dipoles, etc.)

### Data pipeline (`apax/data/`)
Three dataset variants selected via `DataConfig.dataset.processing`:
- `"cached"` (`CachedDataset`) — tf.data with disk cache, best for uniform system sizes
- `"otf"` (`OTFDataset`) — on-the-fly neighbor list, tf.data generator
- `"pbp"` (`PBPDataset`) — power-of-two padding with parallel workers, best for varied system sizes

All datasets in `input_pipeline.py` subclass `InMemoryDataset` and output batches of `(positions, Z, neighbor_idx, box, offsets)`.

### Training (`apax/train/`)
`run.py:run()` orchestrates: config parsing → dataset init → model build → optimizer → `fit()`. `trainer.py:fit()` is the main training loop using Flax's train state pattern. Checkpointing via orbax (`checkpoints.py`). Optional EMA weight averaging, transfer learning, and MLflow/TensorBoard/CSV callbacks.

### Molecular dynamics (`apax/md/`)
- `ase_calc.py` — ASE calculator interface for use in external MD frameworks
- `simulate.py` / `run_md()` — built-in NVT via JaxMD (NHC thermostat)
- `function_transformations.py` — wraps trained model into a JaxMD-compatible energy function

### Batch active learning (`apax/bal/`)
`api.py:kernel_selection()` selects informative structures from a pool using last-layer gradient features and MaxDist selection. Uses `FeatureModel` to extract representations.

### Utilities
- `apax/utils/jax_md_reduced/` — vendored subset of JaxMD (neighbor lists, space transforms, partition)
- `apax/utils/convert.py` — ASE Atoms ↔ apax input format conversion
- `apax/utils/transform.py` — function-level model transformations (e.g. `make_energy_only_model`)

## Key Conventions

**JAX/precision:** Default is float32 for descriptors and readouts. Positions, cell vectors, and final energy/force predictions must be float64. `fp64_sum` in `apax/utils/math.py` is used for numerically stable energy accumulation.

**Functional style:** Prefer pure functions. Model components are Flax `nn.Module`s; stateful operations are avoided. `jax.jit` requires static shapes — dynamic padding to the maximum system size is handled in the data pipeline.

**Neighbor list convention:** `idx` is shape `(2, n_neighbors)`, `offsets` is shape `(n_neighbors, 3)`. The JaxMD-reduced neighbor list in `apax/utils/jax_md_reduced/partition.py` is used for MD; `vesin` is used for training neighbor lists.

**Test structure:** `tests/unit_tests/` for individual components, `tests/integration_tests/` for end-to-end training runs, `tests/regression_tests/` for numerical reproducibility. Slow tests are marked `@pytest.mark.slow` and excluded from default runs.
