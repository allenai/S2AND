# Rust Runtime And Performance Plan

Status date: 2026-02-28

## Current Frontier

Active optimization frontier is tracked in:
`docs/rust/roadmap.md`

Baselines and promotion rules live in:
`docs/rust/baselines.md`.

## Goal

Primary train/eval command:

`uv run python scripts/archive/transfer_experiment_internal.py --experiment_name inventors_s2and_union_eval --leave_self_in --skip_individual_models --random_seed 1 --n_jobs 8`

Project goals:

1. Keep quality parity with Python.
2. Keep or improve latency on maintained train/eval workloads.
3. Keep Rust peak RSS non-regressed (inference and train/eval).
4. Keep install-aware runtime defaults:
   `s2and` => Python by default, `s2and[rust]` => Rust on beneficial stages by default.
5. Keep rollback controls and strict explicit-Rust behavior.
6. Reach full train/eval + inference Rust-path unification only after gates are met.


## Runtime Contract

1. `uv pip install s2and`:
   default runtime is Python end-to-end.
2. `uv pip install "s2and[rust]"`:
   if Rust extension is importable and core-capable, default runtime uses Rust for beneficial stages.
3. Python path remains available via explicit backend and stage overrides.

## Implemented

1. Backend resolution supports `python`, `rust`, `auto`.
2. Unset `S2AND_BACKEND` resolves through capability-aware `auto`.
3. `auto` capability check is centralized in `s2and/rust_capabilities.py`:
   - core runtime requires extension importability + `RustFeaturizer.from_dataset`
4. Startup logging emits one-time resolved backend with capability reason.
5. Per-stage process-tree RSS snapshots added to `scripts/rust_suite.py transfer-mini` output JSON (`stage_rss_gb`) plus per-dataset RSS checkpoints.
6. CI runs two gated lanes:
   - `py-only` (full suite with Python backend)
   - `rust-enabled` (extension build + parity guards + full suite)
7. JSON-ingest filtered payload temp dirs are managed with `TemporaryDirectory`.
8. Rust extension capability/version gating is centralized and enforced for `auto` runtime resolution.
9. Rust batch featurization uses chunk-budget control as the supported memory governor:
   - max chunk budget: 256 MB
10. Runtime env surface is centralized in `README.md` (stable + advanced/internal runtime controls).
11. `scripts/rust_suite.py transfer-mini --mode compare` now runs `python` vs `rust` only.
12. `from_dataset` ref_details extraction gated behind `compute_reference_features` (matching existing `from_json_paths` gate); disk-cache version bumped to 4; featurizer cache add-before-evict race fixed.
13. Compact `CounterData`: replaced `HashMap<String, f64>` with `Vec<(u64, f32)>` sorted by FNV-1a 64-bit hash; `counter_jaccard_data` uses binary search. Disk-cache version bumped to 5. Measured savings: ~400 MB for kisti. Phase 1 RSS gate: all three seeds pass (≤+2.2% vs +5% threshold).
14. L1 + P0 completion (2026-02-27):
   - Added repeated-build stress gate with fast default test and opt-in heavy AMiner loop (`scripts/rust_suite.py stress-rebuild`, `tests/test_rust_from_json_paths.py`).
   - Added Rust batch constraint APIs (`get_constraints_matrix`, `get_constraints_matrix_indexed`) plus Python/model integration across `distance_matrix_helper`, `predict_incremental_helper`, and `_phase_a_seed_distances`.
   - Added regression coverage for batch-constraint parity/fallback/incremental invariants.
15. Phase-split incremental surfaces partial Phase A explicitly:
    - return field `phase_a_accumulator_overflow_early_stop`
    - log line `Telemetry: phase_split_phase_a_overflow ...`
    - test: `tests/test_cluster_incremental.py::test_phase_a_overflow_surfaces_in_result_and_telemetry`
16. Rust batch startup fixed-overhead calibration hardening:
    - page-touch probe allocations before RSS sampling
    - calibration only adopted when it increases conservatism (never decreases `fixed_overhead_bytes` used for chunk sizing)
    - test: `tests/test_rust_batch_chunking.py::test_rust_batch_plan_never_decreases_fixed_overhead`
17. Windows memory budgeting without `psutil`:
    - total RAM fallback: `GlobalMemoryStatusEx`
    - RSS fallback: `GetProcessMemoryInfo` (working set)
    - tests: `tests/test_memory_budget.py` (Windows fallbacks monkeypatched; no real WinAPI calls)
18. Training-mode deferred paper preprocessing in Rust `from_dataset` is implemented and capability-gated:
    - capability marker: `SUPPORTS_FROM_DATASET_PAPER_PREPROCESS`
    - chunked deferred-paper compute in Rust (`FROM_DATASET_PAPER_PREPROCESS_CHUNK_SIZE=4096`)
    - Python lifecycle gate includes `compute_reference_features=False` safeguard
    - coverage: `tests/test_rust_from_dataset_contract.py`, `tests/test_preprocess_papers_parallel_defaults.py`
19. L1b production training-path cleanup boundary is now ported beyond harness-only:
    - `scripts/transfer_experiment_seed_paper.py` runs targeted
      `evict_rust_featurizer(dataset)` + `gc.collect()` boundaries
    - emits `Telemetry: post_rust_cleanup ...`; no global inference/subblocking cache clear
20. L6 local artifact hygiene defaults now route key local generators to `scratch/` by default
    (`export_name_counts_for_rust`, inventors subset/histogram/embedding scripts), with matching tests.
21. Transfer-mini diagnostics now include explicit runtime toggles and per-trial timing telemetry:
    - toggles: `--rust-cleanup-boundary`, `--force-python-paper-preprocess`
    - stage telemetry: hyperopt trial duration summaries + parameter hashes + fitted tree counts

## Runtime Policy Spec

### Backend resolution

- `S2AND_BACKEND` accepts: `python`, `rust`, `auto`.
- Unset `S2AND_BACKEND` is treated as `auto`.
- `auto` behavior:
  - if Rust core capability is unavailable, resolve to Python
  - if Rust core capability is available, resolve to Rust
- Invalid values raise `ValueError`.

### Stage defaults

- Resolved backend `python`:
  all stages run Python.
- Resolved backend `rust` defaults:
  - `ingest_preprocess`: Rust stage enabled
  - `constraints`: Rust
  - `pair_featurization`: Rust
- Ingest nuance:
  - Rust inference defaults to JSON ingest (`from_json_paths`) when JSON paths are available.
  - Train/eval and non-path inference payloads use `from_dataset`.
- `S2AND_BACKEND` controls all stages uniformly.

### Failure semantics

- Explicit `python` backend:
  zero Rust calls.
- Explicit `rust` backend:
  strict fail-fast on Rust-stage execution errors.
- `auto` backend:
  fallback to Python only during backend resolution.
  If `auto` resolves to Rust, runtime Rust-stage errors still fail fast.

## Verification Gates

Use these concrete gates before promoting any Rust defaults further.

1. Quality parity:
   no metric regression beyond `1e-6` absolute on maintained parity tests.
2. Latency gate:
   no regression worse than `+5%` versus Python baseline on maintained workloads.
3. Peak RSS gate:
   no regression worse than `+5%` versus Python baseline on maintained workloads.
4. CI release gate:
   both `py-only` and `rust-enabled` lanes must be green.
5. Full-unification gate:
   maintained train/eval and inference must both pass latency and RSS gates on current-code artifacts before removing mode-specific path logic.

## Benchmark Evidence (active source of truth)

Active benchmark baselines and promotion workflow are centralized in:
`docs/rust/baselines.md`

For active promoted baselines and latest verification snapshots, see:
`docs/rust/baselines.md`.

Workload-ID gate behavior verified:
1. Smoke vs full (`--mode gate`) fails with workload mismatch.
2. Full vs full (`--mode gate`) passes with `violations: 0`.

## Unification Status (as implemented)

Already unified across train/eval and inference:

1. `constraints` stage backend selection and Rust execution.
2. `pair_featurization` hot path (`many_pairs_featurize` Rust batch path).
3. Rust featurizer cache/build lifecycle core machinery.

Intentionally divergent (by design):

1. Inference-only JSON ingest (`from_json_paths`) — requires file paths that train/eval does not have.


## High-Impact Risk Register

### Accepted (low severity, monitored)

1. Featurizer cold-start serialization:
   global cache lock spans full featurizer build/load path. Lock scope is correct and necessary for atomicity (cache-hit check + insert must be atomic). Contention under tested n_jobs=4-8 is not observed.
2. Name-count precedence drift risk:
   canonicalization shims (`_canonicalize_last_for_counts`, `_lasts_equivalent_for_constraint`) introduce implicit precedence assumptions. No NaN placeholders in current code. Shims will be removed after normalization migration phase 4.

### Resolved

1. Train/eval RSS non-regression (was primary blocker):
   Phase 1 gate passed after compact `CounterData` — all three seeds ≤+2.2% vs +5% threshold.
2. Weak-reference cache race:
   `WeakKeyDictionary` iteration in `_auto_evict_rust_featurizers` is guarded by snapshot materialization (`list(keys())`) plus `_RUST_FEATURIZER_CACHE_LOCK`.

### Mitigated recently (keep regression watch)

1. JSON-ingest API skew hard-fail:
   runtime now enforces the current `from_json_paths` contract (normalization-version args required); old-extension adapters were removed to avoid silent drift.
2. Double-parse cost scope reduced:
   normalization version validation is delegated to Rust for artifact-backed JSON ingest (no Python-side json gate).
3. Incremental RAM-input policy risk (big-block track):
    phase-split incremental now accepts explicit `total_ram_bytes` input and otherwise
    auto-detects RAM with a `0.8` safety factor before deriving budgets.
4. Rust signature-preprocess cohort drift (2026-02-24):
   `compute_single_letter_cohort_baseline.py` cohort predicate falls back to `author_info_first` when `author_info_first_normalized_without_apostrophe` is `None` (Rust deferred-field path), matching `_signature_first_for_rules` in `model.py`.

## Verification Commands

1. Full tests (`py-only`):
   - `uv sync --extra dev --frozen`
   - `S2AND_BACKEND=python uv run pytest -q`
2. Full tests (`rust-enabled`):
   - `uv sync --extra dev --extra rust --frozen`
   - `uv run maturin develop -m s2and_rust/Cargo.toml`
   - `uv run pytest -q`
3. Runtime policy coverage:
   - `uv run pytest -q tests/test_runtime.py tests/test_runtime_policy.py`
4. Deferred-field and contract suites:
   - `uv run pytest -q tests/test_rust_signature_preprocess.py tests/test_rust_from_dataset_contract.py tests/test_feature_port_parity.py`
5. JSON-ingest policy/cache suites:
   - `uv run pytest -q tests/test_rust_lifecycle.py tests/test_rust_native_ingest_version_gate.py tests/test_specter_pickle_contract.py tests/test_rust_sinonym_json_ingest.py tests/test_feature_port_cache.py tests/test_runtime_policy.py`
6. Inference comparator gate:
   - `uv run --no-project python scripts/rust_suite.py compare --dataset inspire --limit 5000 --pair-count 5000 --n-jobs 4 --require-non-dev-rust 0 --require-rust-release 1 --write-json scratch/baselines_20260227/compare_inspire_5k_<change>.json`
7. Maintained mini-transfer full acceptance baseline:
   - `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode compare --preset full --target kisti --n-jobs 4 --n-train-pairs 10000 --n-iter 5 --require-rust-release 1 --write-json scratch/baselines_20260227/profile_transfer_mini_full_<change>.json`
8. Smoke mini-transfer gate:
   - `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode compare --preset smoke --target kisti --n-jobs 2 --n-train-pairs 300 --n-iter 1 --require-rust-release 1 --write-json scratch/baselines_20260227/profile_transfer_mini_smoke_<change>.json`
9. API-skew and incremental/container risk regression slice:
   - `uv run pytest -q tests/test_feature_port_cache.py tests/test_ingest_contract.py tests/test_rust_native_ingest_version_gate.py tests/test_compare_python_vs_rust.py tests/test_cluster_incremental.py`
10. L1/P0 hot-path verification slice:
   - `uv run pytest -q tests/test_rust_from_json_paths.py tests/test_feature_port_parity.py tests/test_regression_fixes.py tests/test_cluster_incremental.py`
11. Stress rebuild RSS-series baseline:
   - `uv run --with psutil python scripts/rust_suite.py stress-rebuild --dataset aminer --build-path from_json_paths --repeats 6 --num-threads 1 --rss-sample-ms 50 --require-rust-release 1 --write-json scratch/baselines_20260227/stress_rust_from_json_paths_aminer_6x_<change>.json`
12. Stress rebuild threshold enforcement check:
   - `uv run --with psutil python scripts/rust_suite.py stress-rebuild --dataset aminer --build-path from_json_paths --repeats 2 --num-threads 1 --rss-sample-ms 50 --rss-growth-max-fraction 0.05 --require-rust-release 1 --write-json scratch/baselines_20260227/stress_rust_from_json_paths_aminer_2x_gate_<change>.json`
13. Largest-block canonical compare smoke:
    - `uv run --no-project python scripts/rust_suite.py largest-block --mode compare --dataset aminer --block "j wang" --n-jobs 4 --max-block-size 200 --timeout-hours 0.5 --require-rust-release 1 --write-json scratch/baselines_20260227/largest_block_compare_smoke_200_<change>.json`
14. Transfer-mini latency diagnostics matrix (Rust single-backend):
    - `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode single --backend rust --preset full --target kisti --n-jobs 4 --n-train-pairs 10000 --n-iter 5 --require-rust-release 1 --run-label rust_default_diag --write-json scratch/diagnostics/transfer_mini_diag_default_<change>.json`
    - `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode single --backend rust --preset full --target kisti --n-jobs 4 --n-train-pairs 10000 --n-iter 5 --require-rust-release 1 --rust-cleanup-boundary 0 --run-label rust_no_cleanup_diag --write-json scratch/diagnostics/transfer_mini_diag_no_cleanup_<change>.json`
    - `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode single --backend rust --preset full --target kisti --n-jobs 4 --n-train-pairs 10000 --n-iter 5 --require-rust-release 1 --force-python-paper-preprocess 1 --run-label rust_force_py_papers_diag --write-json scratch/diagnostics/transfer_mini_diag_force_py_papers_<change>.json`

## Rollout

1. Keep `auto` runtime semantics and dual-lane CI as required baseline.
2. Current baselines:
   - train/eval acceptance: `scratch/baselines_20260227/profile_transfer_mini_full_20260227.json`
   - train/eval sanity: `scratch/baselines_20260227/profile_transfer_mini_smoke_20260227.json`
   - inference: `scratch/baselines_20260227/compare_inspire_5k_20260227.json`
   - stress: `scratch/baselines_20260227/stress_rust_from_json_paths_aminer_6x_20260227.json`
   - largest-block compare smoke: `scratch/baselines_20260227/largest_block_compare_smoke_200_20260227.json`
   - full baseline ownership, freshness rules, and historical archive pointers live in `docs/rust/baselines.md`.

## Artifact divergence

Artifact-level divergences between Python and Rust paths (and the format migration plan) live in:
`docs/rust/artifact_divergence.md`.

