# MACE Distance Transforms (Agnesi) — Design Spec

**Date:** 2026-05-04
**Status:** Pending implementation (companion to `2026-05-04-mace-density-zbl-design.md`)
**Scope:** Add a per-edge `distance_transform` stage to apax's MACE radial embedding so foundations that ship `mace.modules.radial.AgnesiTransform` (MACE-MPA-0, MatPES-r2scan-omat-ft, OMAT) can convert and run at parity. Without this, the Density-variant blocks land but mpa-0 / matpes parity is still blocked at the radial-embedding boundary.

## 1. Problem

The Density + ZBL spec assumed the radial embedding stays
``radial = bessel(r) * cutoff(r)``. That holds for `small` and `medium`. It
does **not** hold for `medium-mpa-0`, `MACE-matpes-r2scan-omat-ft`, or any
foundation trained with a non-trivial `radial_embedding.distance_transform`.

torch-mace's `RadialEmbeddingBlock.forward`:

```python
cutoff = self.cutoff_fn(r)
if hasattr(self, "distance_transform"):
    r = self.distance_transform(r, node_attrs, edge_index, atomic_numbers)
radial = self.bessel_fn(r)         # uses *transformed* r
return radial * cutoff, None       # cutoff still uses *original* r
```

Both mpa-0 and matpes carry a non-trainable `AgnesiTransform(a=1.0805,
q=0.9183, p=4.5791)`. A probe confirmed identical default parameters
across the two models.

Empirically (water dimer, mpa-0): without the transform, apax's `radial`
norm is 3.13 vs torch's 11.12 — a 3.5× systematic drift that propagates
through the radial MLP and explodes downstream (per-atom force error
~165 eV/Å vs torch's 0.94 eV/Å).

## 2. Goals and non-goals

### Goals

- Faithful port of `mace.modules.radial.AgnesiTransform` as a Flax linen
  module in `apax/layers/descriptor/basis_functions.py`. Bit-exact parity
  vs the torch reference at `rtol 1e-12, atol 1e-12` on a dimer sweep.
- Refactored radial embedding pipeline that threads per-edge atomic
  numbers (`Z`, `idx`) through to the transform without breaking the
  existing `assemble_edge_features` API for callers that don't need a
  transform.
- `MaceModelConfig.distance_transform: Optional[DistanceTransformConfig]`
  with `None` (default, current behaviour) or a `Literal["agnesi"]`
  discriminated entry that carries the three scalar parameters and a
  `trainable: bool` toggle. Schema and runtime stay in sync — exactly the
  pattern that worked for `interaction_cls` / `empirical_corrections`.
- `_extract_config_from_torch` detects the torch `distance_transform`
  attribute and emits the right config; `_map_state_to_pytree` maps the
  `radial_embedding.distance_transform.{a,q,p,covalent_radii}` buffers /
  parameters into the apax slot.
- mpa-0 + matpes parity tests pass at the documented thresholds
  (`rtol 1e-4, atol 1e-5` energy; `rtol 1e-3, atol 1e-4` forces).

### Non-goals

- `SoftTransform` (the tanh-based variant in `mace/modules/radial.py:285`).
  Not used by any foundation in scope today. Adding it later is purely
  additive (one more discriminated-union entry + one more Linen module).
- Trainable distance-transform parameters in apax fresh training. The
  port supports `trainable: bool` for round-trip fidelity, but no fresh
  training is parity-tested in this spec — that work is downstream.
- `AgnesiTransform.covalent_radii` swap-out. The torch buffer comes from
  `ase.data.covalent_radii` verbatim; we register the same default and
  copy the buffer over from torch on conversion (so any foundation that
  shipped a custom table will round-trip).

## 3. Architecture

### 3.1 The transform module

```python
# apax/layers/descriptor/basis_functions.py

class AgnesiTransform(nn.Module):
    """Faithful port of ``mace.modules.radial.AgnesiTransform``.

    Per-pair length transform driven by element-pair covalent radii::

        r_0   = 0.5 * (covalent_radii[Z_u] + covalent_radii[Z_v])
        T(r)  = 1 / (1 + a * (r/r_0)^q / (1 + (r/r_0)^(q-p)))

    All three scalars (``a``, ``q``, ``p``) are stored as buffers when
    ``trainable=False`` (foundation-model regime) and as parameters when
    ``trainable=True`` (so a future fresh-training run can fine-tune them).
    """

    a_init: float = 1.0805
    q_init: float = 0.9183
    p_init: float = 4.5791
    trainable: bool = False

    def setup(self):
        self.covalent_radii = self.variable(
            "buffers", "covalent_radii",
            lambda: jnp.asarray(ase.data.covalent_radii, dtype=jnp.float64),
        )
        if self.trainable:
            self.a = self.param("a", lambda _: jnp.asarray(self.a_init, dtype=jnp.float64))
            self.q = self.param("q", lambda _: jnp.asarray(self.q_init, dtype=jnp.float64))
            self.p = self.param("p", lambda _: jnp.asarray(self.p_init, dtype=jnp.float64))
        else:
            self.a = self.variable("buffers", "a",
                lambda: jnp.asarray(self.a_init, dtype=jnp.float64))
            self.q = self.variable("buffers", "q",
                lambda: jnp.asarray(self.q_init, dtype=jnp.float64))
            self.p = self.variable("buffers", "p",
                lambda: jnp.asarray(self.p_init, dtype=jnp.float64))

    def _scalar(self, x):
        return x.value if hasattr(x, "value") else x

    def __call__(self, r, Z, idx):
        # r: (n_edges,), Z: (n_atoms,), idx: (2, n_edges)
        i, j = idx[0], idx[1]
        Z_u, Z_v = Z[i], Z[j]
        r0 = 0.5 * (self.covalent_radii.value[Z_u] + self.covalent_radii.value[Z_v])
        a, q, p = self._scalar(self.a), self._scalar(self.q), self._scalar(self.p)
        x = r / r0
        denom = 1.0 + a * (x ** q) / (1.0 + (x ** (q - p)))
        return 1.0 / denom
```

Note: `idx` follows apax's convention (`idx[0]=receiver, idx[1]=sender`)
which is the **opposite** of torch's `edge_index` ordering (sender first,
receiver second). The transform is symmetric in `Z_u`/`Z_v` (their sum
appears in `r_0`) so the convention difference does not change values.

### 3.2 Refactor `assemble_edge_features` → `MaceRadialEmbedding`

Today's `assemble_edge_features` is a free function that takes only
`dr_vec`. To plumb the transform through cleanly we promote it to a Linen
module so it can own the (optional) child transform without forcing
every caller to construct one:

```python
class MaceRadialEmbedding(nn.Module):
    """Composable radial embedding: bessel × cutoff with optional transform.

    Mirrors torch-mace ``RadialEmbeddingBlock``. The optional
    ``distance_transform`` is applied between ``cutoff_fn`` and
    ``bessel_fn`` and consumes per-edge atomic numbers.

    Forward returns ``(radial, sph)`` where ``radial = bessel(T(r)) * cutoff(r)``
    (or ``bessel(r) * cutoff(r)`` when no transform is configured).
    """

    r_max: float
    num_bessel: int
    num_polynomial_cutoff: int
    max_ell: int
    distance_transform: Optional[Any] = None  # nn.Module instance or None

    @nn.compact
    def __call__(self, dr_vec, Z, idx):
        dtype = dr_vec.dtype
        r_ij = jnp.linalg.norm(dr_vec, axis=-1)
        cutoff = PolynomialCutoff(p=self.num_polynomial_cutoff, r_max=self.r_max)(r_ij)
        if self.distance_transform is not None:
            r_ij = self.distance_transform(r_ij, Z, idx)
        bessel = MaceBesselBasis(
            n_basis=self.num_bessel, r_max=self.r_max, dtype=dtype,
        )(r_ij)
        radial = (bessel * cutoff[..., None]).astype(dtype)
        sph = e3nn.spherical_harmonics(
            e3nn.Irreps.spherical_harmonics(self.max_ell),
            dr_vec, normalize=True, normalization="component",
        )
        return radial, sph
```

Critical: cutoff is computed on the **original** `r`, transform is then
applied, and bessel uses the **transformed** value. This matches torch
exactly. Reordering breaks parity.

`assemble_edge_features` stays as a thin wrapper for callers that don't
care about a transform (existing tests, fresh-trained models):

```python
def assemble_edge_features(dr_vec, r_max, num_bessel, num_poly_cutoff, max_ell):
    """Back-compat shim — calls MaceRadialEmbedding without a transform."""
    mod = MaceRadialEmbedding(
        r_max=r_max, num_bessel=num_bessel,
        num_polynomial_cutoff=num_poly_cutoff, max_ell=max_ell,
        distance_transform=None,
    )
    return mod.apply({}, dr_vec, jnp.zeros((), dtype=jnp.int32), jnp.zeros((2, 0), dtype=jnp.int32))
```

(The dummy `Z`/`idx` are fine because the no-transform path doesn't touch
them; document this in the wrapper's docstring.)

### 3.3 `MaceRepresentation` integration

Replace the current free-function call:

```python
radial, sph = assemble_edge_features(dr_vec, ...)
```

with a Linen submodule call:

```python
radial, sph = MaceRadialEmbedding(
    r_max=self.r_max,
    num_bessel=self.num_bessel,
    num_polynomial_cutoff=self.num_polynomial_cutoff,
    max_ell=self.max_ell,
    distance_transform=self._build_distance_transform(),
    name="radial_embedding",
)(dr_vec, Z, idx)
```

where `_build_distance_transform` instantiates the configured transform
(or returns `None`). The submodule's name is `"radial_embedding"` —
matches torch's state-dict prefix exactly so the converter keys map
1-to-1.

### 3.4 Schema — `DistanceTransformConfig`

```python
# apax/config/model_config.py

class DistanceTransform(BaseModel, extra="forbid"):
    name: str

class AgnesiTransformConfig(DistanceTransform, extra="forbid"):
    name: Literal["agnesi"]
    a: float = 1.0805
    q: float = 0.9183
    p: float = 4.5791
    trainable: bool = False

DistanceTransformConfig = Union[AgnesiTransformConfig]  # extend as more land

class MaceModelConfig(BaseModelConfig, extra="forbid"):
    ...
    distance_transform: Optional[DistanceTransformConfig] = None
```

Keep the shape consistent with `EmpiricalCorrection`: discriminated union,
each entry a typed config with hard-coded `name` Literal. Adding
`SoftTransform` later is a one-class extension to the union.

### 3.5 Builder dispatch

```python
# apax/nn/builder.py inside MaceBuilder.build_descriptor

dt_cfg = self.config.get("distance_transform")
if dt_cfg is None:
    distance_transform = None
elif dt_cfg["name"] == "agnesi":
    distance_transform = AgnesiTransform(
        a_init=dt_cfg.get("a", 1.0805),
        q_init=dt_cfg.get("q", 0.9183),
        p_init=dt_cfg.get("p", 4.5791),
        trainable=dt_cfg.get("trainable", False),
    )
else:
    raise NotImplementedError(f"distance_transform {dt_cfg['name']!r}")
descriptor = MaceRepresentation(
    ...,
    distance_transform=distance_transform,
)
```

`MaceRepresentation` gains a `distance_transform: Optional[Any] = None`
field that it threads into `MaceRadialEmbedding`. The field carries the
constructed Linen module (not the config dict) — matches how other
Mace-side dispatched modules are passed today.

## 4. Converter

### 4.1 `_extract_config_from_torch`

```python
distance_transform_cfg = None
if hasattr(model.radial_embedding, "distance_transform"):
    dt = model.radial_embedding.distance_transform
    cls_name = type(dt).__name__
    if cls_name == "AgnesiTransform":
        is_trainable = isinstance(dt.a, torch.nn.Parameter)
        distance_transform_cfg = {
            "name": "agnesi",
            "a": float(dt.a.detach().cpu()),
            "q": float(dt.q.detach().cpu()),
            "p": float(dt.p.detach().cpu()),
            "trainable": is_trainable,
        }
    else:
        raise NotImplementedError(
            f"Foundation uses distance_transform {cls_name!r}; apax supports "
            f"['AgnesiTransform']. Other variants need their apax port."
        )
cfg["distance_transform"] = distance_transform_cfg
```

### 4.2 `_map_state_to_pytree` — `_map_distance_transform`

```python
def _map_distance_transform(state, out):
    """Copy radial_embedding.distance_transform.{a,q,p,covalent_radii} from torch."""
    rep_buf = out["buffers"]["energy_model"]["representation"]["radial_embedding"]
    dt_buf = rep_buf["distance_transform"]
    dt_buf["covalent_radii"] = state["radial_embedding.distance_transform.covalent_radii"]
    for name in ("a", "q", "p"):
        torch_arr = state[f"radial_embedding.distance_transform.{name}"]
        if name in dt_buf:
            dt_buf[name] = torch_arr.astype(dt_buf[name].dtype)
        else:
            par = out["params"]["energy_model"]["representation"]["radial_embedding"]
            par["distance_transform"][name] = torch_arr.astype(...)
```

`_map_state_to_pytree` calls this after `_map_node_embedding` if the
config carries a `distance_transform`. Keeps the spec's existing
"if hasattr(...): map" pattern that already exists for `pair_repulsion_fn`.

### 4.3 Path slot in apax pytree

Linen names submodules by their attribute name. With the
`MaceRadialEmbedding(name="radial_embedding")` placement in §3.3, the
slot lives at:

```
params/energy_model/representation/radial_embedding/distance_transform/{a,q,p}
buffers/energy_model/representation/radial_embedding/distance_transform/{a,q,p,covalent_radii}
```

(`a`/`q`/`p` go under `params/...` only when `trainable=True`; otherwise
all four leaves live under `buffers/...`.)

## 5. Testing

### 5.1 Always-on (no torch)

Add to `tests/unit_tests/layers/descriptor/test_basis_functions.py`:

- `test_agnesi_transform_shape_and_finite` — sanity check on a small
  `(n_edges,)` input.
- `test_agnesi_transform_pair_symmetric` — `T(r; Z_u, Z_v)` =
  `T(r; Z_v, Z_u)` because `r_0` depends on the symmetric sum.
- `test_agnesi_transform_param_collections` — assert `trainable=False`
  puts everything under `buffers`; `trainable=True` puts `a`/`q`/`p`
  under `params`.

### 5.2 Dimer parity (gated mace_parity)

Add to `tests/integration_tests/mace/test_mace_distance_transform.py`
(new file):

- `test_agnesi_transform_matches_torch_on_dimer[H-H,H-O,O-O,Si-O]` —
  for a dimer at a sweep of `r` values, compare apax `AgnesiTransform`
  output to torch's. `np.allclose(rtol=1e-12, atol=1e-12)`.

### 5.3 Foundation parity (gated mace_parity)

Already covered by `tests/integration_tests/mace/test_mace_parity.py`
in the parent spec. Once this work lands, the existing `[medium-mpa-0]`
parametrization passes and the matpes test (xfail dropped in the parent
spec) passes too.

### 5.4 Converter integration

Extend `test_torch_to_apax_param_coverage_no_projections` so the
`mpa-0` path runs end-to-end and the buffer leaves
(`radial_embedding/distance_transform/{a,q,p,covalent_radii}`) are
populated and equal to the torch source.

## 6. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cutoff applied to transformed `r` instead of original | Medium | Off-by-O(1) energy, off forces | Section 3.2 explicit; unit test compares to torch radial output exactly |
| `idx` convention swap (apax vs torch edge_index ordering) | Low | None for Agnesi (symmetric in Z_u, Z_v) | Stays low even for SoftTransform (also symmetric); document in module docstring |
| `assemble_edge_features` callers break after refactor | Medium | Parity for small/medium regresses | Back-compat shim with dummy `Z`/`idx`; existing tests that import `assemble_edge_features` keep passing without modification |
| Pyright Optional[Any] for `distance_transform` field | Low | Type imprecision | Acceptable — Linen's pattern allows nn.Module instances to be passed as fields and the runtime type is dynamic |
| Foundation with `apply_cutoff=False` | Low | Off forces if encountered | Detect and reject at the converter boundary with a clear NotImplementedError; no in-scope foundation uses it |

## 7. Phasing

| Phase | Goal | Exit |
|---|---|---|
| **a. Transform primitive** | `AgnesiTransform` Linen module + always-on unit tests | Shape + symmetry tests pass |
| **b. Radial pipeline** | `MaceRadialEmbedding` module + `assemble_edge_features` shim; `MaceRepresentation` wired to it | Existing small/medium parity still passes |
| **c. Schema + dispatch** | `DistanceTransformConfig` discriminated union, builder dispatch, `MaceModelConfig.distance_transform` | Schema round-trip test passes |
| **d. Converter** | `_extract_config_from_torch` detects `distance_transform`; `_map_distance_transform` plumbed in | mpa-0 conversion runs, dimer parity test passes |
| **e. Foundation parity** | mpa-0 + matpes parity (water + periodic SiO₂) | All gated parity tests at documented thresholds |

## 8. Net LoC estimate

| Component | LoC |
|---|---|
| `AgnesiTransform` + tests | ~110 |
| `MaceRadialEmbedding` Linen module + shim | ~70 |
| `MaceRepresentation` wiring + builder dispatch | ~40 |
| Schema (`DistanceTransformConfig` union) | ~30 |
| Converter (`_extract_config_from_torch` + `_map_distance_transform`) | ~60 |
| Tests (always-on + dimer + foundation) | ~120 |
| **Total** | **~430** |

All additive — no breaking changes to fresh-trained MACE models.

## 9. Open items

- Whether to expose the `covalent_radii` table as a configurable knob.
  Defer to an optional `covalent_radii: Optional[list[float]] = None`
  field on `AgnesiTransformConfig`; default copies ASE's table.
- `SoftTransform` port — purely additive when needed.
- A future "no-transform sanity" CI guard that converts `small` /
  `medium` and asserts the converted config has
  `distance_transform == None`. Cheap to add, prevents regressions where
  a future converter change accidentally always emits a transform.
