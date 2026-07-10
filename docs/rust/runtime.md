# Rust Runtime Contract

Status date: 2026-07-09

This document defines the operational contract for the Rust extension: backend
resolution, stage defaults, failure semantics, verification gates, and the risk
register. Active benchmark baselines live in `baselines.md`.

---

## Goal

Primary Rust gate commands are the canonical `scripts/rust_suite.py` workflows
documented in [baselines.md](baselines.md):

```
uv run python scripts/rust_suite.py compare ...
uv run python scripts/rust_suite.py transfer-mini ...
uv run python scripts/rust_suite.py stress-rebuild ...
```

Archived transfer scripts are historical and are not the primary runtime gate.

Project goals:
1. Keep quality parity with Python.
2. Keep or improve latency on maintained train/eval workloads.
3. Keep Rust peak RSS non-regressed vs Python (inference and train/eval).
4. Treat `s2and-rust` as a required install dependency. Explicit Python mode
   currently controls featurization/constraints but native production-model
   scoring still calls Rust; making it a true zero-Rust rollback is tracked in
   [../work_plan.md](../work_plan.md#b16-backendpython-is-not-a-zero-rust-rollback).
5. Keep rollback controls and explicit-Rust override behavior.
6. Reach full train/eval + inference Rust unification only after all gates pass.

---

## Runtime contract

| Install | Default runtime |
|---|---|
| `uv pip install s2and` | Installs S2AND plus required `s2and-rust`; `auto` uses Rust for capable stages |
| `uv pip install "s2and[rust]"` | Compatibility alias; equivalent runtime dependency set |

Python fallback paths remain available via explicit backend and stage overrides for stages that still support them.

### Backend resolution

- `S2AND_BACKEND` accepts: `python`, `rust`, `auto`.
- Unset `S2AND_BACKEND` resolves as `auto`.
- `auto` behavior:
  - If Rust core capability is unavailable: resolve to Python.
  - If Rust core capability is available: resolve to Rust.
- Invalid values raise `ValueError`.
- Capability detection is centralized in `s2and/runtime.py`.
- Core runtime capability requires extension importability plus the current
  Rust markers used by production Arrow paths: direct Arrow ingest, indexed
  featurization, constraints, seed updates, and name-count index support.
  Rust featurization enters through Arrow IPC artifacts only. Classic
  `ANDData`/JSON datasets remain supported through the Python featurizer unless
  validated Arrow artifacts are attached.

### Stage defaults (resolved backend = `rust`)

| Stage | Default |
|---|---|
| `ingest_preprocess` | Rust |
| `constraints` | Rust |
| `pair_featurization` | Rust |

- Direct Arrow inputs are the production inference boundary. Explicit Rust
  production prediction fails fast when required Arrow paths are incomplete;
  auto/default prediction falls back to Python for non-Arrow datasets.
- Arrow-ingested training datasets (`s2and/arrow_training.py`) attach
  `rust_featurizer_arrow_paths` and build through `from_arrow_paths`. When
  that Rust attachment is enabled, Python-side SPECTER embedding materialization
  is skipped by default; pass `load_python_specter=True` to
  `build_training_anddata_from_arrow(...)` only for Python reference work or
  direct Python embedding access.
- Train/eval and classic `ANDData` payloads without Arrow artifacts use Python
  featurization.
- `S2AND_BACKEND` controls featurization, constraints, indexed subblocking, and
  promoted-linker routing. It does not yet control native production booster
  scoring, which is a known contract defect rather than an intentional stage
  exception.

### Failure semantics

| Backend | Failure behavior |
|---|---|
| Explicit `python` | Python featurization/constraints, but native production booster scoring still calls Rust until work-plan B16 lands. |
| Explicit `rust` | Strict fail-fast on any Rust-stage execution error. |
| `auto` (resolved to Python) | Python only; no Rust fallback needed. |
| `auto` (resolved to Rust) | Fail-fast on runtime Rust-stage errors. Fallback only happens during initial backend resolution. |

---

## Verification gates

These gates must pass before promoting any Rust defaults further.

| Gate | Threshold |
|---|---|
| Quality parity | `1e-6` absolute for continuous features and exact for discrete/count/boolean fields |
| Latency | No regression worse than `+10%` vs the pinned migration protocol unless explicitly accepted |
| Peak RSS | No regression worse than `+10%` vs the pinned migration protocol unless explicitly accepted |
| CI release | Both `py-only` and `rust-enabled` CI lanes green |
| Full-unification | Train/eval and inference both pass latency + RSS gates before removing mode-specific path logic |

---

## Unification status

**Already unified** (train/eval and inference):
1. `constraints` stage backend selection and Rust execution.
2. `pair_featurization` hot path (`many_pairs_featurize` Rust batch path).
3. Rust featurizer cache/build lifecycle core machinery.

**Intentionally divergent** (by design):
1. Direct Arrow inference uses typed runtime files that classic JSON/`ANDData`
   workflows do not require.
2. Classic train/eval still starts from `ANDData`; Rust train/eval requires an
   Arrow-ingested dataset with attached `rust_featurizer_arrow_paths`.

---

## Cache semantics

- Public `use_cache` remains the pair-feature persistent-cache knob across training and inference.
- `use_cache=True` enables the pair-feature SQLite cache.
- Same-process Rust featurizer reuse is independent of `use_cache`.
- Published Arrow/count inputs are immutable content-addressed generations.
  Reuse is bound to the exact material paths, full generation inventory,
  non-seed settings, and seed version. Filesystem watches invalidate same-path
  mutation before reuse, including same-size/restored-mtime rewrites.
- See [../caching.md](../caching.md) for the full cache layout and operational guidance.

---

## Implementation notes

Key design decisions and their rationale (in order of implementation):

- **Batch constraint APIs** (`get_constraints_matrix_indexed`,
  `get_constraints_block_upper_triangle_indexed`): integrated across
  `distance_matrix_helper` and `_predict_incremental_helper`.
- **Compact `CounterData`**: replaced `HashMap<String, f64>` with `Vec<(u64, f32)>` sorted by
  FNV-1a 64-bit hash; `counter_jaccard_data` uses binary search. ~400 MB savings on kisti.
  Disk-cache version bumped to 5. Note: 64-bit birthday collision risk is very low at million-scale
  keys (~2.7e-8 at 1M; ~2.7e-6 at 10M), but a collision would merge counts silently.
- **Windows memory budgeting without `psutil`**: total RAM via `GlobalMemoryStatusEx`; RSS via
  `GetProcessMemoryInfo` (working set).
- **Arrow training deferred preprocessing**: `s2and.arrow_training` can mark
  Arrow-ingested datasets so paper preprocessing and signature n-gram/field
  materialization are deferred to Rust Arrow readers.
- **L1b cleanup boundary**: `scripts/transfer_experiment_seed_paper.py` runs targeted
  `evict_rust_featurizer(dataset)` + `gc.collect()` after LightGBM fit; emits
  `Telemetry: post_rust_cleanup ...`.
- **Rust batch chunk-budget control**: max chunk budget 256 MB; startup fixed-overhead calibration
  hardened with page-touch probe and conservative adoption (never decreases `fixed_overhead_bytes`).

---

## Arrow Training Deferred Preprocessing

Arrow-ingested training datasets can skip Python paper preprocessing
(`preprocess_papers_parallel`) and signature n-gram/field materialization
because Rust reads the preprocessed Arrow bundle directly through
`RustFeaturizer.from_arrow_paths`.

### Gating

Python skips these build steps only when all of the following hold:

- Backend resolves to Rust (`S2AND_BACKEND=rust` or `auto` resolved to Rust).
- `preprocess=True`.
- The dataset is constructed through `s2and.arrow_training` or otherwise passes
  `rust_arrow_featurization=True`.

Code pointers:
- Lifecycle decision: `s2and/rust_lifecycle.py` (`build_rust_lifecycle_policy`).
- Python skip behavior: `s2and/data.py` (skips `preprocess_papers_parallel` when
  `skip_python_paper_preprocess=True`).
- Rust ingestion: `s2and/feature_port.py`
  (`build_rust_featurizer_from_arrow_paths`).

### How to verify

Maintenance checklist:
1. Build release extension: `uv run maturin develop -m s2and_rust/Cargo.toml --release`
2. Run focused tests:
   - `uv run pytest -q tests/test_arrow_training_ingestion.py tests/test_preprocess_papers_parallel_defaults.py tests/test_rust_lifecycle.py tests/test_rust_capabilities.py`
3. Optional: rerun transfer-mini compare and write the JSON under `scratch/baselines_YYYYMMDD/` (see `baselines.md`).

Current watchlist items for this area are tracked in
[../work_plan.md](../work_plan.md), especially the blocked normalization
migration.

---

## Verification commands

**Full tests, Python only:**
```
uv sync --extra dev --frozen
S2AND_BACKEND=python uv run pytest -q
```

**Full tests, Rust enabled:**
```
uv sync --extra dev --extra rust --frozen
uv run maturin develop -m s2and_rust/Cargo.toml
uv run pytest -q
```

**Runtime policy coverage:**
```
uv run pytest -q tests/test_runtime.py tests/test_rust_lifecycle.py
```
