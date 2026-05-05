# MACE transfer-learning + property-heads — Design

> **For agentic workers:** the implementation plan written from this spec is the
> authoritative artefact for execution. This document defines the design and
> rationale; it does not prescribe step-by-step code edits.

## Goal

Three follow-ups to PR #558 that close real gaps surfaced while reviewing the
foundation → fine-tune workflow. None of them depend on the schema-repartition
that just landed in PR #558; they can be implemented in any order, but they
share a single review surface so we ship them together.

1. **(a) Shape-mismatch transfer learning.** `apax.transfer_learning.parameter_transfer.black_list_param_transfer` currently copies source-checkpoint values into the target-model pytree leaf-by-leaf without checking shapes. When a user fine-tunes a converted MACE foundation with a shallow ensemble (or any architecture change that widens a slot — different `nn` topology, different `n_species`), the source `(M, 1)` weight is silently written into the target's `(M, n_members)` slot. JAX may broadcast or crash later; either way the user has no actionable signal. Fix: detect shape mismatches loudly with an error message that names every offending leaf and emits a copy-pasteable `reset_layers` snippet.
2. **(b)+(c) MACE property-head support.** `MaceBuilder.build_readout` ignores its `head_config` argument and always returns a `MaceReadout` configured from the model-level `model.readout`. Property heads' per-head `nn`, `n_shallow_members`, `dtype`, etc. are silently dropped. Fix: add an explicit `kind: Literal["standard", "mace"]` field to `PropertyHead`, route property heads through the appropriate readout, and error loudly on cross-model misuse (`kind="standard"` on MACE; `kind="mace"` on non-MACE).
3. **Template wording.** `apax/cli/templates/mace_finetune_minimal.yaml` currently ships `transfer_learning.reset_layers: []`, which after (a) lands will produce a guided shape-mismatch error on the first run. Update the template's preamble comment to describe the run-once-then-paste-`reset_layers` workflow so users hit the error and recover smoothly.

## Why this matters

The user-facing motivation: foundation → shallow-ensemble fine-tune is a
canonical workflow for these models, and supporting that workflow (plus
property heads like charges/dipole that build on top of MACE) is the whole
point of integrating MACE foundations into apax. The current state has two
loud silent failures (transfer learning shape mismatch; property-head config
drop) and no good answer to "how do I add a `[64, 64]` MLP charges head on
top of MPA-0?". This spec closes both.

## Architecture

The fix touches three files in `apax/` plus three test files plus the
template. No schema changes outside `PropertyHead`; no migration concerns.

| File | Role |
|---|---|
| `apax/transfer_learning/parameter_transfer.py` | (a) — replace the unconditional `flat_target[p] = v` with a shape-checked transfer that collects mismatches and raises with an actionable error; extend `reset_layers` semantics to accept full leaf paths in addition to the legacy `p[-2]` suffix. |
| `apax/config/model_config.py` | (b)+(c) — add `kind: Literal["standard", "mace"] = "standard"` and `MLP_irreps: str = "16x0e"` fields to `PropertyHead`. |
| `apax/nn/builder.py` | (b)+(c) — `MaceBuilder.build_readout`: discriminate energy head vs property head by `head_config is self.config`; for property heads, dispatch on `head_config["kind"]`. `ModelBuilder.build_readout`: error on `kind="mace"` since AtomisticReadout-only paths can't honor it. |
| `apax/cli/templates/mace_finetune_minimal.yaml` | Preamble comment update — describe the run-once-paste-`reset_layers` workflow. |
| `tests/unit_tests/transfer_learning/test_shape_mismatch_transfer.py` (new) | (a) unit tests — ensemble widening errors; matching shapes pass; full-leaf-path entries skip correctly; legacy suffix entries still work. |
| `tests/unit_tests/nn/test_property_heads.py` (new) | (b)+(c) unit tests — MACE+`kind=mace`, MACE+`kind=standard`(error), GMNN+`kind=mace`(error), GMNN default behavior unchanged. |
| `tests/integration_tests/mace/test_mace_finetune.py` | Add a parametrized test asserting that fine-tuning the converted small foundation with `n_members > 1` raises the actionable shape-mismatch error when `reset_layers` is empty, and succeeds when the suggested keys are pasted in. |

## Detailed design

### (a) Shape-mismatch transfer learning

`black_list_param_transfer` becomes:

```
For each leaf path p in source_params:
    skip_leaf = (p[-2] in reset_layers) or ("/".join(map(str, p)) in reset_layers)
    if skip_leaf: continue
    if source[p].shape != target[p].shape:
        record (p, source.shape, target.shape) for error
        continue   # do NOT copy
    target[p] = source[p]

If any mismatches recorded:
    raise TransferLearningShapeMismatchError(mismatches)
```

The error message format:

```
Transfer learning shape mismatch on N parameter(s):

  params/energy_model/readout/readout_0/linear/kernel
    source: (128, 1)   target: (128, 8)
  ...

Add the following to your config to re-initialize these slots from the model's
default initialization:

  transfer_learning:
    reset_layers:
      - params/energy_model/readout/readout_0/linear/kernel
      - ...
```

The `reset_layers` semantic change is purely additive: a leaf is skipped if
EITHER its second-to-last path component matches an entry (legacy) OR its
full `/`-joined path matches an entry (new). Existing GMNN/EquivMP/So3krates
configs that use the legacy form keep working unchanged.

**Why error rather than silent skip:** silent skip masks unintentional shape
mismatches (typos in the architecture config). The error message tells the
user exactly which keys mismatched and how to suppress them, so the
intentional case (foundation→ensemble) takes one extra run-then-paste
iteration but is unambiguous. A `transfer_strict_shape: false` permissive
mode was considered and rejected as redundant — the actionable error message
is itself the friction reducer.

### (b)+(c) MACE property-head support

**Schema change** on `apax/config/model_config.py:PropertyHead`:

```python
kind: Literal["standard", "mace"] = "standard"
MLP_irreps: str = "16x0e"   # used only when kind="mace"
```

`kind="standard"` is the schema-wide default, preserving GMNN/EquivMP/So3krates
behavior. MACE users must explicitly set `kind: mace` per property head; the
builder errors otherwise (inside `MaceBuilder.build_readout`, after pydantic validation).

**`MaceBuilder.build_readout`** logic:

```
def build_readout(self, head_config, is_feature_fn=False, only_use_n_layers=None):
    is_energy_head = head_config is self.config
    if is_energy_head:
        # existing energy logic — reads self.config["readout"], dispatches on kind
        ...
        return MaceReadout(...)   or fall through to super() on standard / is_feature_fn
    # Property head path
    if head_config["kind"] == "standard":
        raise ValueError(
            f"property_head '{head_config['name']}' uses kind='standard' but the "
            "model is MACE; AtomisticReadout doesn't respect MACE's per-layer body-"
            "order structure. Set kind='mace' explicitly, or change the model."
        )
    # head_config["kind"] == "mace"
    return MaceReadout(
        num_interactions=len(self.config["descriptor"]["interactions"]),
        hidden_dim=e3nn.Irreps(self.config["descriptor"]["hidden_irreps"]).filter("0e").dim,
        MLP_irreps=head_config["MLP_irreps"],
        n_shallow_ensemble=head_config["n_shallow_members"],
        dtype=head_config["dtype"],
    )
```

**`ModelBuilder.build_readout`** (parent, used by GMNN/EquivMP/So3krates):

At the top, before the existing AtomisticReadout build:

```
if head_config is not self.config and head_config.get("kind") == "mace":
    raise ValueError(
        f"property_head '{head_config['name']}' uses kind='mace' but the model is "
        f"{self.config['name']}; MaceReadout requires the per-layer-concatenated "
        "feature shape that only MaceRepresentation produces."
    )
```

The energy path (where `head_config is self.config`) doesn't have a `kind`
field — that's a property-head-only schema field — so the energy path is
unaffected by this guard.

**Why both errors loud, both opt-in explicit:** matches the principle the
user articulated — no implicit cross-model defaults; one schema default
(`"standard"`); MACE users opt in explicitly. The cost is YAML verbosity for
MACE property heads; the benefit is no silent config drift.

**Behavior with the user's example YAML:**

```yaml
model:
  name: mace
  ...
  property_heads:
    - name: charges
      nn: [64, 64]                # default kind="standard" → ERROR
```

→ `ValueError` raised inside `MaceBuilder.build_readout` when the model is constructed (after pydantic validation, before training starts). User adds `kind: mace`:

```yaml
property_heads:
  - name: charges
    kind: mace
    n_shallow_members: 0          # MaceReadout uses these...
    # MLP_irreps: 16x0e           # ...not these:
    # nn: [64, 64]                # silently ignored under kind=mace
```

For GMNN, `kind="standard"` is implicit and AtomisticReadout uses `nn`,
`n_shallow_members`, etc. as today.

### Template fix

`apax/cli/templates/mace_finetune_minimal.yaml` keeps `reset_layers: []` but
the preamble comment changes from "head-swapping is incompatible with
black_list_param_transfer" to:

```yaml
# Foundation → shallow-ensemble fine-tune workflow:
#
#   1. Run `apax train mace_finetune_minimal.yaml` once.
#   2. The first run fails with a "shape mismatch" error listing every
#      readout slot that needs to be re-initialized for the new ensemble
#      width. Copy the suggested `reset_layers:` block into this config.
#   3. Re-run; the listed slots stay random-initialized (the classical
#      "drop scalar head, attach N-output head, train" pattern), every
#      other slot transfers from the foundation.
#
# This is the standard apax mechanism for going from a single-output
# foundation to an n_members shallow ensemble.
```

No code changes to the template's `model:` or `transfer_learning:` blocks.

## Testing strategy

Three new test files, plus one extension to an existing integration test.

**`tests/unit_tests/transfer_learning/test_shape_mismatch_transfer.py`** — unit tests for (a):
- `test_matching_shapes_transfer_succeeds`: source and target identical shapes → no error, target gets source values.
- `test_mismatched_shapes_raise_with_actionable_message`: a single mismatched leaf → error names that leaf, mentions both shapes, contains a `reset_layers:` YAML snippet with that leaf's full path.
- `test_full_leaf_path_in_reset_layers_skips_transfer`: mismatched leaf added to `reset_layers` by full path → no error, target value unchanged.
- `test_legacy_suffix_in_reset_layers_skips_transfer`: leaf added by `p[-2]` suffix (legacy form) → no error.
- `test_multiple_mismatches_collected_in_one_error`: three mismatched leaves → one raise with all three.

**`tests/unit_tests/nn/test_property_heads.py`** — unit tests for (b)+(c):
- `test_mace_property_head_kind_mace_builds_mace_readout`: MACE config + `property_heads: [{name: charges, kind: mace}]` → builder produces a MaceReadout with `num_interactions=len(descriptor.interactions)`.
- `test_mace_property_head_kind_standard_raises`: MACE config + `kind: standard` → ValueError naming the head.
- `test_mace_property_head_default_kind_is_standard_so_it_raises_on_mace`: schema default → MACE rejects, message tells user to set `kind: mace`.
- `test_gmnn_property_head_default_kind_builds_atomistic_readout`: GMNN config + default property_head → AtomisticReadout with the per-head `nn`.
- `test_gmnn_property_head_kind_mace_raises`: GMNN + `kind: mace` → ValueError naming the head.
- `test_mace_property_head_mace_readout_n_shallow_members_propagates`: per-head `n_shallow_members=4` → MaceReadout's `n_shallow_ensemble == 4` (independent of the energy head's ensemble).

**`tests/integration_tests/mace/test_mace_finetune.py`** — extend with:
- `test_finetune_foundation_to_ensemble_errors_when_reset_layers_empty`: convert small foundation, write fine-tune config with `n_members=4` and `reset_layers: []`, attempt to start training, assert TransferLearningShapeMismatchError with at least the readout-final-Linear keys named.
- `test_finetune_foundation_to_ensemble_succeeds_with_suggested_reset_layers`: parse the error message's suggested keys, paste them into `reset_layers`, re-run, assert one epoch trains and the readout slots in the saved checkpoint match `(M, 4)` shape.

## Risks

| concern | mitigation |
|---|---|
| Legacy `reset_layers: ["linear"]`-style configs broken by the new full-path check | Backward-compatible OR-of-conditions: a leaf is skipped if EITHER condition matches. Test `test_legacy_suffix_in_reset_layers_skips_transfer` locks this. |
| Identity check `head_config is self.config` fragile if a future caller passes a copy | Documented at the call site; unit-tested by both the energy path and the property path. The discriminator is correct because `build_energy_model` and `build_property_heads` are the only two callers in the current tree, and both call patterns are stable. |
| User pastes the suggested `reset_layers` snippet but the foundation's parameter layout changes between runs | Out of scope — the error is regenerated on every run, so a layout-changed foundation produces a fresh suggestion. The user re-pastes. |
| Silent broadcasting in JAX masks the error if our shape check is missed | The new code path raises BEFORE writing to `flat_target`, so JAX never sees the malformed pytree. |
| `kind="mace"` on a property head with a MACE descriptor that uses `kind="standard"` for the energy readout | Sanity-check during MaceBuilder.build_readout: this is allowed. The energy readout config is independent of the per-head readout config. The integration test case explicitly does not exercise this combination but the unit tests do. |

## Out of scope

- Auto-resetting mismatched layers without an explicit `reset_layers` entry. The user articulated that errors are the right default; this proposal honors that and treats `reset_layers` as the only escape valve.
- Restoring partial slots (e.g., copying `(M, 1)` into the first column of `(M, 8)`). The classical "drop scalar head, attach N-output head, train" pattern is what users want; this proposal implements exactly that.
- Per-layer-frozen fine-tuning beyond what `transfer_learning.freeze_layers` already supports. Orthogonal feature.
- Allowing `MaceReadout` on non-MACE descriptors. MaceReadout's input shape contract `(K·M,)` is structurally tied to MACE's per-layer concatenation; supporting it elsewhere would require a new readout interface entirely.
- Refactoring `black_list_param_transfer` away from path-suffix matching toward something more typed (e.g., regex or glob). The full-leaf-path extension is enough for the current use cases; future precision improvements are additive.
