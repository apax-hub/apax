# MACE Foundation Integration — Review Remediation Design

**Date:** 2026-05-05
**Branch:** `feat/mace-foundation-integration`
**Base:** `main`
**Status:** Approved for plan-writing

## 1. Background

The `feat/mace-foundation-integration` branch ports the MACE equivariant
message-passing architecture into apax. It (a) lets apax train MACE-style
models from scratch and (b) consumes pretrained MACE foundation models
(e.g. `mace-mpa-0-medium`) by converting torch weights into apax's
JAX/Flax parameter tree.

This spec covers the work needed to bring the branch to a mergeable
state against current `main`: synchronise with main, fix two real
defects, and root-cause the s22 parity gap with systematic debugging.

## 2. Work items

| ID | Description | Resolution |
|----|-------------|------------|
| C1 | s22 single-point energy parity max \|ΔE\| = 3.56e-2 eV vs reference MACE on the same `.model` (target ≤ 1e-5 eV) | Phase 1 (systematic debugging) |
| C2 | `apax convert-mace --head 'mp'` default crashes on models whose only head is `default` | Phase 0.2 |
| C4 | `pyproject.toml` pins `mace-jax` by absolute local path (`/Users/fzills/tools/mace-jax`) | Phase 0.3 |
| I2 | Parity tests use only 3-atom systems; can't catch size-scaling bugs | Phase 2.F |
| I4 | `_scatter_o3_linear_blocks` slot-key ↔ torch instruction-order assumption untested for multi-irrep linear blocks | Phase 2.G |
| I6 | `MaceModelConfig.interaction_cls` accepts `list`, `MaceRepresentation` expects `tuple`; no coercion in builder | Phase 2.H |
| I7 | ZBL `output_scale` assumes a single global scalar — silently miscomputes for any per-element scale model | Phase 2.H |

## 3. Goals and non-goals

**Goals**

- Bring `feat/mace-foundation-integration` to a mergeable state against
  current `main`.
- Achieve s22 single-point energy parity ≤ 1e-5 eV (max abs diff)
  against reference `MACECalculator` on the same `.model` file, for the
  `mace-mpa-0-medium` foundation checkpoint.
- Make the branch buildable on any machine (no absolute local-path
  dependencies).
- Add tests that lock in the parity fix and the multi-irrep slot-key
  invariant.

**Non-goals**

- Closing the inference performance gap vs reference MACE. The 27×
  wall-clock difference observed on s22 with varying atom counts is
  caused by JIT recompilation per shape. Confirmed: with a fixed-shape
  conformer set (e.g. `smiles2conformers("CCO", numConfs=250)`) the gap
  closes. This is a usage characteristic, not a code defect, and is out
  of scope for this PR.
- Restoring or reimplementing any feature that the branch already
  contains.
- Refactors unrelated to the MACE work or the listed findings.

## 4. Phase 0 — Sync with `main` and trivial Critical fixes

Each step is its own small, reviewable commit. 0.1 must come first
because it changes line numbers that 0.2 and 0.3 then edit.

### 0.1 Merge `origin/main` into the branch

Conflict surface (files modified on both sides since merge-base
`df2b8084`):

- `apax/cli/apax_app.py` — branch added two MACE-convert command
  registrations (`+2` lines); main rewrote the file (`+297` lines).
  **Resolution:** take main's structure, then re-add the MACE-convert
  command registrations on top.
- `apax/config/model_config.py` — branch added `MaceModelConfig` and
  extended the `BaseModelConfig` discriminator union; main also
  changed schemas. **Resolution:** union — keep main's edits and
  re-apply the MACE additions.
- `apax/layers/empirical.py` — branch added `MaceZBLPairRepulsion`;
  main also changed empirical contributions. **Resolution:** union.
- `apax/nn/builder.py` — branch added the MACE builder branch
  (`+96`); main added runtime overrides for `calc_stress`,
  `calc_hessian`, `force_variance` (`+20`). **Resolution:**
  composable; both keep.
- `pyproject.toml` — both sides changed dependencies and the `mace`
  extra. **Resolution:** union; the `mace-jax` `[tool.uv.sources]`
  entry will be removed by Phase 0.3, so its conflict is moot.
- `uv.lock` — auto-regenerated. **Resolution:** `git checkout --theirs
  uv.lock` then `uv sync --extra mace` to regenerate against the
  merged `pyproject.toml`. Stage and commit the regenerated lockfile.

Post-merge sanity:

- `uv run pytest tests/integration_tests/mace -x` (must still pass).
- `uv run pytest tests/unit_tests/layers/descriptor/test_mace_blocks.py
  -x` (must still pass).

### 0.2 Fix `--head` default in `convert-mace`

`apax/cli/convert_mace.py:25` currently defaults `head="mp"`, which
raises `ValueError: head='mp' not in available heads ['default']` on
any model that does not have an `mp` head (including the user's local
foundation runs). Fix:

- CLI: change `head: str = typer.Option("mp", ...)` to
  `head: str | None = typer.Option(None, "--head", ...)`.
- `apax/transfer_learning/mace_foundation.py:_extract_config_from_torch`:
  when `head is None`, set `head = list(model.heads)[0]`. When `head`
  is given but not in `model.heads`, raise the existing `ValueError`
  unchanged.
- Test: extend `tests/unit_tests/cli/test_convert_mace.py` with a case
  asserting that omitting `--head` picks `heads[0]`.

### 0.3 Vendor `_convert_native_weights`; drop `mace-jax` local-path dep

The branch's only consumer of `mace-jax` is the symmetric-contraction
weight converter at `apax/transfer_learning/mace_foundation.py:1058`
(`from mace_jax.adapters.cuequivariance.symmetric_contraction import
_convert_native_weights`). `mace-jax` is not on PyPI and is currently
pinned by absolute local path in `pyproject.toml`, which makes the
branch unbuildable on any other machine.

- Create `apax/transfer_learning/_mace_jax_compat.py`. Copy the helper
  function and any private dependencies it pulls in (Clebsch–Gordan
  coefficient builders) verbatim, with a top-of-file comment citing
  the upstream source path and revision (`mace-jax` commit SHA at
  copy time) and the upstream MIT license header.
- Replace the import in `mace_foundation.py:1058` with the local
  module.
- Remove `mace-jax` from `[project.optional-dependencies].mace` and
  from `[tool.uv.sources]` in `pyproject.toml`. Run `uv sync --extra
  mace` to regenerate the lockfile.
- Verify by running the converter end-to-end on
  `mace-mpa-0-medium.model` and confirming the resulting apax model
  loads.

## 5. Phase 1 — Systematic debugging for C1 (parity gap)

**Iron Law:** no fixes proposed until layer-by-layer evidence is
captured. The s22 error pattern (machine precision on small systems,
1e-3 to 1e-2 eV on dense aromatic stacks) does not point uniquely to
any single layer; treat every block as a candidate until the harness
data narrows it down.

### 5.1 Build a layer-by-layer parity harness

- Pick the worst-case s22 system from the `tmp/main.py` run: index 20
  (indole–benzene, T-shape), |ΔE| = 3.56e-2 eV. Lock this system as
  the regression target throughout Phase 1.
- Torch side: register `forward_hook`s on every block of the loaded
  reference `MACECalculator` model:
  - `radial_embedding` (output → `R_l(r)`)
  - per layer `i ∈ {0, …, num_interactions-1}`:
    - `interactions[i].linear_up`
    - `interactions[i].conv_tp` (the message)
    - `interactions[i].linear`
    - `interactions[i].skip_tp`
    - `products[i]` (symmetric contraction output)
    - `readouts[i]`
  - `scale_shift`
  - `pair_repulsion` (ZBL)
- Apax side: thread Flax `sow('intermediates', name, value)` calls (or
  equivalent post-hoc capture) at the matching boundaries inside
  `MaceRepresentation` / `MaceInteraction` / `ProductBlock` /
  `MaceReadout` / `MaceZBLPairRepulsion`. Slot names already match
  torch's per the `mace_blocks.py` slot-pinning convention; reuse them.
- Run both on the same atoms (system 20). For each pair of named
  intermediates, compute `max_abs_diff` and `mean_abs_diff` per atom ×
  per irrep slot. Persist the table as a parity report (CSV or JSON)
  under `tmp/` for inspection.

### 5.2 Identify the first divergence

The first row in the report whose `max_abs_diff` exceeds ~1e-6 (rough
single-precision noise floor; tighter for float64 paths) is the
culprit. **No speculation about which layer it will be.** Possible
outcomes — each is a hypothesis to be tested only once the evidence
points there:

- Radial fails first → `_map_distance_transform` wiring or
  `covalent_radii` mismatch.
- `interactions[i].linear` fails first → `_scatter_o3_linear_blocks`
  slot-ordering bug for multi-irrep targets.
- `interactions[i].skip_tp` fails first → `path_weight = 1/sqrt(M_in
  * n_species)` rescale wrong.
- `products[i]` fails first → ProductBlock symmetric contraction
  ordering bug (`cuex.equivariant_polynomial(method="naive")`
  permutation vs torch's einsum).
- `scale_shift` fails first → ZBL/scale interaction.
- `readouts[i]` fails first → `_MaceFullyConnectedNet` silu-norm
  constant bit-mismatch.

### 5.3 Drill, hypothesise, fix minimally, verify

- Form a single hypothesis from the report.
- Make the smallest possible code change to test it.
- Re-run the harness on system 20 and confirm:
  - the previously-failing layer now matches at the float64 noise
    floor;
  - all subsequent layers also match (a fix at layer N can leave
    layer N+1 still off if there's an independent bug there).
- If the hypothesis fails, do not stack a second fix on top; revert
  and form a new hypothesis from the same evidence.
- After each successful round, re-run on all 22 s22 systems via
  `tmp/main.py` and record the new max |ΔE|. Iterate until max |ΔE|
  ≤ 1e-5 eV across all 22.

### 5.4 Architectural escalation

If three hypothesis–fix–verify cycles fail, stop. Three failed fixes
is a signal that the architecture (e.g. the converter's structural
assumption that torch and apax instruction orders correspond) is the
problem, not any single bug. At that point: pause, document the
evidence, and surface the question — do we need to change the
converter strategy? — before spending more effort on point fixes.

## 6. Phase 2 — Lock-in tests + robustness

### 6.1 (F) Parametrized s22 parity test

- New file: `tests/integration_tests/mace/test_mace_s22_parity.py`.
- Uses the same converted `mace-mpa-0-medium` model as `tmp/main.py`.
- `@pytest.mark.parametrize("idx", range(22))` iterates over every s22
  dimer.
- Asserts both energy parity (`atol=1e-5 eV`) and forces parity
  (`atol=1e-4 eV/Å`) against the reference `MACECalculator`.
- Marked as integration; gated behind a fixture that skips if the
  pretrained `.model` file is not present locally (CI gate to be
  decided separately).

### 6.2 (G) Multi-irrep slot-key pin test for `_scatter_o3_linear_blocks`

- New unit test: `tests/unit_tests/transfer_learning/test_scatter_o3_linear_blocks.py`.
- Constructs a torch `e3nn.o3.Linear` with multi-irrep input and
  multi-irrep output (e.g. `8x0e + 8x1o → 4x0e + 4x1o + 4x2e`).
- Constructs the matching apax `e3nn.flax.Linear`.
- Asserts that `_scatter_o3_linear_blocks` produces apax weights such
  that, given the same input, both Linears yield identical output to
  float64 noise.
- Locks the slot-key ↔ instruction-order invariant against future
  changes in either e3nn-jax's slot naming or torch's instruction
  enumeration.

### 6.3 (H) I6 and I7 robustness

- I6 — `apax/nn/builder.py`: at the point that reads
  `self.config["interaction_cls"]`, coerce list → tuple before
  passing to `MaceRepresentation`. Update
  `apax/config/model_config.py` `MaceModelConfig.interaction_cls`
  type annotation to be unambiguous about the runtime contract.
- I7 — `apax/transfer_learning/mace_foundation.py:316-323`: before
  reading `model.scale_shift.scale`, assert
  `torch.unique(model.scale_shift.scale).numel() == 1` and raise
  `NotImplementedError("Per-element ZBL scale not supported")`
  otherwise. Add a brief regression unit test that fabricates a
  multi-element scale tensor and confirms the assertion fires.

## 7. Validation gate before merge

All of the following must pass:

- `uv run python tmp/main.py` shows max |ΔE| ≤ 1e-5 eV on s22
  (record the actual number in the PR description).
- `uv run pytest tests/integration_tests/mace/test_mace_s22_parity.py`
  passes for all 22 systems (Phase 2.F).
- `uv run pytest tests/integration_tests/mace tests/unit_tests/layers/descriptor/test_mace_blocks.py
  tests/unit_tests/cli/test_convert_mace.py
  tests/unit_tests/config/test_mace_model_config.py
  tests/unit_tests/layers/test_mace_readout.py
  tests/unit_tests/nn/test_mace_builder.py
  tests/unit_tests/transfer_learning/test_scatter_o3_linear_blocks.py`
  passes.
- Full apax test suite passes: `uv run pytest`.
- `apax convert-mace` without `--head` succeeds on the local
  `mace-mpa-0-medium.model`.
- `pyproject.toml` contains no absolute paths and no `mace-jax`
  reference; `uv sync --extra mace` succeeds in a clean environment.

## 8. Subagent fan-out plan

0.1 must come before 0.2 and 0.3 (the merge changes line numbers
they edit). 0.2 and 0.3 are independent of each other and can run in
parallel after 0.1 lands. Phase 1 is sequential by nature (evidence
→ hypothesis → fix → verify). Phase 2 is parallel-friendly:

| Phase | Subagent | Independent? |
|-------|----------|--------------|
| 0.1 | merge driver | sequential — must come first |
| 0.2 | C2 fix | after 0.1 |
| 0.3 | C4 vendor | after 0.1 |
| 1   | parity-harness driver | sequential, single thread |
| 2.F | s22 parity test | parallel — depends only on Phase 1 fix |
| 2.G | slot-key pin test | parallel — depends only on Phase 0.1 |
| 2.H | I6 + I7 robustness | parallel — depends only on Phase 0.1 |

## 9. Risks and open questions

- **Parity may not reach 1e-5 eV with a single fix.** The s22 error
  pattern (machine precision on small/few-element systems, 1e-3 to
  1e-2 eV on dense aromatic stacks) suggests a many-body or
  high-multipole accumulation rather than a per-edge bias. Phase 1's
  iteration plan accounts for this; the architectural-escalation
  clause in 5.4 prevents indefinite thrashing.
- **Vendoring `_convert_native_weights` decouples apax from
  upstream mace-jax fixes.** If mace-jax later corrects a CG
  coefficient bug, we will need to re-pull. Mitigation: the vendored
  file's header records the upstream commit SHA, so the diff is
  traceable.
- **Inference performance.** Out of scope per Section 3, but should
  be mentioned in the PR description so reviewers don't expect
  parity *and* speed in this PR.

## 10. Out of scope (deferred to follow-up issues)

- Closing the apax-vs-torch inference perf gap (JIT recompilation
  across varying atom counts).
- Adding a `nn.scan`-based readout for ensemble-of-foundations.
- Migrating the `tmp/main.py` ad-hoc benchmark into a proper
  `tests/perf/` harness.
- Making `apax convert-mace` accept a remote URL or HuggingFace
  reference for the `.model` file.
