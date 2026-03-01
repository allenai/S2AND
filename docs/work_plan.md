# Current Work Plan (Engineering Bundles)

Status date: 2026-02-28

This doc is a short "where to start next" map. It is not a replacement for:
- `docs/rust/runtime.md` (runtime contract + verification commands)
- `docs/rust/baselines.md` (canonical gate commands + baseline artifacts)
- `docs/rust/roadmap.md` (optimization frontier / worklist)
- `docs/rust/training_preprocessing_plan.md` (detailed design + phased execution for training paper preprocessing)
- `docs/rust/artifact_divergence.md` (artifact divergence map + migration plan)
- `docs/normalization_migration.md` (normalization unification migration plan)

## Evidence Policy (Important)

Evidence artifacts are **local-only for now**.

- Put JSON/log artifacts under `scratch/` (already gitignored).
- "Promotion" means documenting: command line, `workload_id` (when present), and key deltas in docs.
- When the policy changes to shared artifacts, revisit `docs/rust/baselines.md` and the migration docs.

## Bundle 1: Training Paper Preprocessing -> Rust (Already Staffed)

Status: **Implemented in working tree and locally verified (2026-02-28).**

Scope: phases 0-3 of `docs/rust/training_preprocessing_plan.md`.

Start here:
- `docs/rust/training_preprocessing_plan.md` (authoritative checklist + verification plan)

Completed implementation + verification highlights:
- Rust `from_dataset` now handles deferred paper preprocessing for training raw papers, including bounded
  chunked compute (`FROM_DATASET_PAPER_PREPROCESS_CHUNK_SIZE`) and language/coauthor parity.
- Python lifecycle gating is capability-aware (`from_dataset_paper_preprocess_available`) and stage-safe
  (`compute_reference_features=False` guard).
- Release build verified: `uv run maturin develop -m s2and_rust/Cargo.toml --release`.
- Parity/regression tests passed:
  - `uv run pytest -q tests/test_rust_from_dataset_contract.py tests/test_preprocess_papers_parallel_defaults.py tests/test_rust_lifecycle.py tests/test_rust_capabilities.py` (`79 passed`).
  - `uv run pytest -q tests/test_feature_port_parity.py tests/test_rust_from_json_paths.py` (`25 passed, 1 skipped`).

Phase evidence (local):
- Phase 0 microbench artifact: `scratch/bench_paper_preprocess_bundle1_20260228.log`.
- Phase 3 transfer gate artifact: `scratch/profile_transfer_mini_bundle1_4_20260228.json`.
- Supporting inference compare artifact: `scratch/compare_bundle1_4_inspire5k_20260228.json`.

Notes:
- This bundle is the highest-impact remaining training perf item, but it can increase Rust-side
  `from_dataset` work. If the process then runs Python-only LightGBM stages, allocator residue can
  dominate wall time on Windows. See Bundle 2 for the mitigation boundary.

## Bundle 2: L1b Production Port + L6 Hygiene (Python-side)

Goal: port the "Rust -> LightGBM cleanup boundary" from the benchmark harness into the **real** training path.

Scope:
- L1b: targeted per-dataset Rust featurizer eviction + `gc.collect()` at the Rust->LightGBM boundary.
  Do **not** clear global caches on inference/subblocking paths.
- L6 hygiene: default local generators to write into `scratch/` (or require explicit `--output` under `data/`)
  and ensure `.gitignore` covers common local artifacts.

Start here:
- L1b design/status: `docs/rust/roadmap.md` (L1b section)
- Current harness implementation: `scripts/_rust_suite/transfer_mini_cmd.py` (cleanup boundary lives here today)
- Artifact hygiene policy context: `docs/rust/roadmap.md` (L6 section)

Status (2026-02-28): **Implemented and numerically verified.**

L1b production-port updates:
- Added targeted Rust cleanup boundary to real training scripts:
  - `scripts/transfer_experiment_seed_paper.py`
- Boundary behavior: per-dataset `evict_rust_featurizer(...)` + `gc.collect()` with telemetry
  (`Telemetry: post_rust_cleanup ...`), no global inference/subblocking cache clear.

L6 hygiene updates:
- Local generator defaults moved to `scratch/`:
  - `scripts/export_name_counts_for_rust.py`
  - `scripts/make_inventors_s2and_subset.py`
  - `scripts/make_inventors_hf_specter_embeddings.py`
  - `scripts/make_inventors_split_and_histograms.py`
- `.gitignore` now explicitly ignores `data/name_counts_rust.json`.

Verification evidence:
- `scratch/profile_transfer_mini_bundle1_4_20260228.json`
  - `post_rust_cleanup` present: `4.188 GB`.
  - Stage ratios vs python:
    - `union_pairwise_fit_seconds`: `62.797 / 53.996 = 1.163` (passes `<=1.25`).
    - `union_nameless_pairwise_fit_seconds`: `45.595 / 43.286 = 1.053` (passes `<=1.25`).
  - RSS cleanup delta: `4.301 - 4.188 = 0.113 GB` (passes `>=0.05 GB`).

Latency caveat follow-up (diagnostic matrix, 2026-02-28):
- Added transfer-mini diagnostics knobs:
  - `--rust-cleanup-boundary {0,1}`
  - `--force-python-paper-preprocess {0,1}` (sets `S2AND_RUST_FORCE_PYTHON_PAPER_PREPROCESS=1`)
- Added per-stage hyperopt trial telemetry in transfer-mini artifacts (`trial_seconds_*`, `trial_param_hashes`,
  and LightGBM fit summaries).
- Diagnostic artifacts:
  - `scratch/diagnostics/transfer_mini_diag_default_20260228.json`
  - `scratch/diagnostics/transfer_mini_diag_default_r2_20260228.json`
  - `scratch/diagnostics/transfer_mini_diag_default_r3_20260228.json`
  - `scratch/diagnostics/transfer_mini_diag_no_cleanup_20260228.json`
  - `scratch/diagnostics/transfer_mini_diag_force_py_papers_20260228.json`
  - `scratch/diagnostics/transfer_mini_diag_force_py_papers_no_cleanup_20260228.json`
- Readout:
  - Deferred-paper (default) runs keep peak RSS stable (`4.787-4.795 GB`) but show high single-run latency spread
    (`153.828s`, `172.326s`, `200.574s`), concentrated in LightGBM trial time.
  - Across those runs, hyperopt parameter hashes and fitted tree counts are identical (`pairwise=2270`,
    `nameless=2098`), indicating runtime variance instead of search/config drift.
  - Forcing Python paper preprocessing raises ANDData build time (`8.7s` -> `27.8-29.4s`), peak RSS
    (`4.795 GB` -> `5.601-5.662 GB`), and total runtime (`+53.9s` to `+57.9s` vs the fastest default run).

## Bundle 3: P3 + P4 Rust Micro-Optimizations (Rust-side)

Scope (see `docs/rust/roadmap.md`):
- P3: cache immutable ID order per featurizer; relax repeated validation on internal trusted paths.
- P4: add two-pointer merge path for `counter_jaccard_data` when both vectors are similarly sized.

Start here:
- `docs/rust/roadmap.md` (P3/P4 sections)
- `docs/rust/baselines.md` (canonical gates to re-run for deltas)

Guardrails:
- Can start immediately on current hotspots.
- Re-profile after Bundles 1/2 land to confirm wins are still material; if hotspots moved, pivot.
- Keep parity gates as the primary stop condition.

Status (2026-02-28): **Implemented, parity-validated, and independently benchmarked.**
- P3: immutable signature ID order/full feature-count caching + reduced repeated internal lookup work in indexed constraints.
- P4: `counter_jaccard_data` now uses two-pointer merge for similarly sized vectors and binary-search path for skew.
- Parity gates: `uv run pytest -q tests/test_feature_port_parity.py tests/test_rust_from_json_paths.py` (`25 passed, 1 skipped`).
- Runtime/regression gate: `scratch/profile_transfer_mini_bundle1_4_20260228.json` and `scratch/compare_bundle1_4_inspire5k_20260228.json`.
- Independent P3/P4 validation artifacts (2026-02-28):
  - `scratch/profile_transfer_mini_p3p4_20260228.json`: rust `177.0s` / `4.790 GB` vs python `299.0s` / `5.520 GB`
    (`1.69x` speedup, `-13.2%` RSS, quality parity: B3 `0.960`, Cluster `0.976`, ClusterMacro `0.933` — identical).
  - `scratch/compare_p3p4_inspire5k_20260228.json`: `4.32x` speedup, `-38.8%` RSS, feature parity pass.

## Bundle 4: Memory Calibration Broadening (Ops)

Scope:
- Run `calibrate-phase-a` / `calibrate-rust-batch` on 2-3 additional workload shapes that emit the
  relevant telemetry and record recommended constants + evidence paths.

Status (2026-02-28): **Completed with additional local evidence artifacts.**

Phase A calibration artifacts:
- `scratch/calibrate_phase_a_shape_4000_l5_pairbuffix_20260228.json` -> recommended `163`.
- `scratch/calibrate_phase_a_shape_10000_p2p4_20260228.json` -> recommended `192`.
- `scratch/calibrate_phase_a_shape_14995_l5_pairbuffix_20260228.json` -> recommended `151`.
- `scratch/calibrate_phase_a_shape_4000_l5_overhead_20260228.json` -> recommended `461` (legacy-overhead outlier).

Rust-batch calibration artifacts:
- `scratch/calibrate_rust_batch_shape_4000_l5_overhead_20260228.json` -> recommended `37`.
- `scratch/calibrate_rust_batch_shape_4000_l5_pairbuffix_20260228.json` -> recommended `37`.
- `scratch/calibrate_rust_batch_shape_14995_l5_pairbuffix_20260228.json` -> recommended `49`.

Operational readout:
- The newer pair-buffer telemetry shapes cluster around `phase_a ~151-192` and `rust_batch ~37-49`.
- Keep `4000_l5_overhead` (`461`) as a historical outlier until old-overhead logs are excluded from calibration sets.

Promoted defaults (2026-02-28):
- `INCREMENTAL_ACCUMULATOR_ENTRY_BYTES`: kept at `200` (P95 ~192, already within ~4% margin).
- `RUST_BATCH_PERSISTENT_ROW_OVERHEAD_BYTES`: tightened from `64` to `52` (P95=49 across 4 workloads, +6% margin).
- `PHASE_A_PAIR_BUFFER_ENTRY_BYTES`: kept at `80` (not recalibrated in this bundle).
- All 19 `tests/test_memory_budget.py` tests pass with updated defaults.



## A1 Batch-Oriented Distance Matrices (Rust-path optimization; keep Python path)

Owner: Codex

Goal: speed up the **Rust-backed predict path** by removing per-pair Python loop scaffolding on the hot path
(`distance_matrix_helper()` generator + per-row matrix write loop) in favor of a batch/block-oriented pipeline.

Non-goal: deleting or deprecating the Python-only path. The existing Python implementation must remain available and
correct under `S2AND_BACKEND=python` and as a fallback when the Rust extension is unavailable.

Scope (staged; keep memory bounded; Rust path only):
1. Add a Rust-backend-only block/chunk implementation (inside `make_distance_matrices()` or a helper) activated only
   when the runtime context resolves to Rust. Keep the current Python generator path intact for the Python backend.
2. For the Rust backend, replace the per-pair generator with a batch API that returns per-chunk arrays:
   - within-block `(i, j)` indices for the chunk
   - constraint labels (`np.nan` = needs model; negative = partial supervision/constraint)
   - block key (one string per chunk)
3. On the Rust backend, fill float64 (what fastcluster wants) distance matrices with contiguous slice writes (FastCluster pdist),
   avoiding `tqdm`/Python per-row loops on the hot path.
4. Rust integration:
   - Use indexed batch constraints + indexed batch featurization on the same pair order (no per-pair FFI).
   - Optional follow-up (bigger win): add a fused Rust API that generates upper-triangle pairs internally
     from a block's signature indices (avoid `np.triu_indices` + Python pair list construction).

Start here:
- `s2and/model.py`: `distance_matrix_helper()`, `make_distance_matrices()`
- `s2and/featurizer.py`: `many_pairs_featurize()` (Rust matrix APIs + chunk planning)
- Roadmap context: `docs/rust/roadmap.md` (A1/A0 sections)
- Hotspot evidence: `docs/rust/profiling/2026-02-26.md`
- Canonical gates: `docs/rust/baselines.md`

Validation (must-have):
- Pytest regression:
  - Python-only path: `S2AND_BACKEND=python uv run pytest -q tests/test_cluster.py::TestClusterer::test_make_distance_matrix_fastcluster`
  - Rust path: `S2AND_BACKEND=rust uv run pytest -q tests/test_cluster.py::TestClusterer::test_make_distance_matrix_fastcluster`
  - Update/add tests covering both FastCluster (flattened) and square-matrix modes for the new blockwise path.
  - Ensure Rust indexed constraint API is still exercised when enabled (update
    `tests/test_regression_fixes.py::test_distance_matrix_helper_uses_indexed_constraint_api` or replace it).
- Canonical runtime gates (release extension; write artifacts under `scratch/`):
  - Build: `uv run maturin develop -m s2and_rust/Cargo.toml --release`
  - Inference comparator:
    `uv run --no-project python scripts/rust_suite.py compare --dataset inspire --limit 5000 --pair-count 5000 --n-jobs 4 --require-non-dev-rust 0 --require-rust-release 1 --write-json scratch/compare_next_up_a1_inspire5k.json`
  - Transfer-mini (acceptance workload):
    `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode compare --preset full --target kisti --n-jobs 4 --n-train-pairs 10000 --n-iter 5 --require-rust-release 1 --write-json scratch/profile_transfer_mini_next_up_a1.json`
  - Largest-block smoke:
    `uv run --no-project python scripts/rust_suite.py largest-block --mode compare --dataset aminer --block "j wang" --n-jobs 4 --max-block-size 200 --timeout-hours 0.5 --require-rust-release 1 --write-json scratch/largest_block_next_up_a1_200.json`
- Profiling evidence:
  - Capture a before/after cProfile on the same workload; `distance_matrix_helper` should disappear from the top
    or drop materially (and total predict seconds should improve). Record deltas + artifact paths.

Implementation update (2026-02-28):
- Landed Rust-only chunk/block helper path in `s2and/model.py`:
  - `_distance_matrix_chunk_helper_rust(...)` emits per-chunk block key, within-block `(i, j)` arrays, labels, and pair IDs.
  - `_predict_distance_matrix_chunk(...)` centralizes chunk featurize + predict.
- `make_distance_matrices(...)` now branches by runtime backend:
  - Rust backend: chunked pipeline + contiguous FastCluster slice writes.
  - Python backend: existing `distance_matrix_helper()` generator path unchanged.
- Fused predict path (`predict_helper(..., dists=None)`) now also uses the Rust chunked pipeline when backend resolves to Rust.
- Rust FastCluster distance buffers are now allocated/fill-written as `float64` on the chunked path.
- Regression coverage added:
  - `tests/test_rust_distance_matrix_blockwise.py` (FastCluster flattened + square-matrix modes).
  - `tests/test_regression_fixes.py::test_make_distance_matrices_rust_blockwise_uses_indexed_constraint_api`.
  - Kept `tests/test_regression_fixes.py::test_distance_matrix_helper_uses_indexed_constraint_api`.

Validation run (2026-02-28):
- `S2AND_BACKEND=python uv run pytest -q tests/test_cluster.py::TestClusterer::test_make_distance_matrix_fastcluster` -> passed.
- `S2AND_BACKEND=rust uv run pytest -q tests/test_cluster.py::TestClusterer::test_make_distance_matrix_fastcluster` -> passed.
- `uv run pytest -q tests/test_rust_distance_matrix_blockwise.py tests/test_regression_fixes.py::test_make_distance_matrices_rust_blockwise_uses_indexed_constraint_api` -> passed.
- `uv run ruff check s2and/model.py tests/test_cluster.py tests/test_regression_fixes.py tests/test_rust_distance_matrix_blockwise.py` -> passed.
- `uv run ty check s2and/model.py tests/test_rust_distance_matrix_blockwise.py tests/test_cluster.py` -> passed with existing non-blocking warnings.

Fused follow-up update (2026-02-28):
- Implemented optional fused upper-triangle API path:
  - Rust extension now exposes:
    - `get_constraints_block_upper_triangle_indexed(...)` (returns chunk-local `(i, j)` + constraint values).
    - `featurize_block_upper_triangle_matrix_indexed(...)` (builds feature rows directly from block indices + offset/range).
  - Python wrappers added in `s2and/feature_port.py`:
    - `get_constraints_block_upper_triangle_indexed_rust(...)`
    - `build_block_upper_triangle_feature_matrix_indexed_rust(...)`
  - Rust `Clusterer` path uses fused APIs when safe (`indexed` constraints mode, Rust backend, `use_cache=False`, APIs present):
    - avoids Python `np.triu_indices` + per-pair string list construction on that path.
    - keeps automatic fallback to prior chunked indexed path when fused APIs are unavailable.
- Added fused-path regression:
  - `tests/test_rust_distance_matrix_blockwise.py::test_make_distance_matrices_rust_fused_upper_triangle_api`
    (asserts fused APIs are exercised and legacy pair helper / `many_pairs_featurize` are bypassed).

Fused follow-up risk analysis (lightweight-gate scope):
- Correctness risk: pair-order drift between constraints and feature rows.
  - Mitigation: both fused Rust APIs use the same upper-triangle range planner (`start_offset`, `max_pairs`) so
    `(i, j)` order is shared by construction.
- Correctness risk: partial-supervision precedence drift versus prior path.
  - Mitigation: Python side maps supervision to condensed-index offsets for each block, then applies the same
    override precedence logic on fused chunk labels before model scoring.
- Correctness risk: behavioral regression when fused APIs are unavailable in older wheels.
  - Mitigation: runtime capability checks + automatic fallback to prior chunked indexed Rust path.
- Compatibility/behavior risk: fused path is unavailable in unsupported modes (`use_cache=True`, constraints not in
  `indexed` mode), which can surprise operators expecting fused performance.
  - Clarification: in current `Clusterer` code, "non-indexed constraints" means the non-fused constraint path
    (`constraint_api_mode != "indexed"`, i.e., Python constraint evaluation / fallback path), not the fused indexed Rust APIs.
  - Mitigation: fused path is intentionally gated to safe preconditions; unsupported modes stay on legacy path.
- Operational risk: extension lock/build environment friction on Windows.
  - Mitigation: keep fused APIs optional and non-breaking; fallback path remains production-safe.

Decision for this step:
- Skip heavy canonical runtime gates/profiling in this pass; keep lightweight regression/lint/build checks only.

Additional validation run (2026-02-28):
- `S2AND_BACKEND=python uv run pytest -q tests/test_cluster.py::TestClusterer::test_make_distance_matrix_fastcluster tests/test_cluster.py::TestClusterer::test_fused_path_equivalence` -> passed.
- `S2AND_BACKEND=rust uv run pytest -q tests/test_cluster.py::TestClusterer::test_make_distance_matrix_fastcluster tests/test_cluster.py::TestClusterer::test_fused_path_equivalence` -> passed.
- `uv run pytest -q tests/test_rust_distance_matrix_blockwise.py tests/test_regression_fixes.py::test_make_distance_matrices_rust_blockwise_uses_indexed_constraint_api tests/test_regression_fixes.py::test_distance_matrix_helper_uses_indexed_constraint_api` -> passed.
- `uv run ruff check s2and/model.py s2and/feature_port.py tests/test_rust_distance_matrix_blockwise.py tests/test_cluster.py tests/test_regression_fixes.py` -> passed.
- `uv run ty check s2and/model.py s2and/feature_port.py tests/test_rust_distance_matrix_blockwise.py tests/test_cluster.py` -> passed with existing non-blocking warnings.
- `uv run pytest -q tests/test_rust_distance_matrix_blockwise.py tests/test_regression_fixes.py::test_make_distance_matrices_rust_blockwise_uses_indexed_constraint_api tests/test_regression_fixes.py::test_distance_matrix_helper_uses_indexed_constraint_api tests/test_cluster.py::TestClusterer::test_make_distance_matrix_fastcluster tests/test_cluster.py::TestClusterer::test_fused_path_equivalence` -> passed (`7 passed`).
- `uv run ruff check s2and/model.py s2and/feature_port.py tests/test_rust_distance_matrix_blockwise.py` -> passed.
- `uv run maturin develop -m s2and_rust/Cargo.toml --release` -> release build succeeded; install/copy blocked by locked `_s2and_rust` binary (`os error 32`).
- `cargo check --manifest-path s2and_rust/Cargo.toml` -> not a code failure; local interpreter check failed because Python `3.14` exceeds current `PyO3` support in this direct cargo env (the `uv` path uses Python `3.11`).

Perf delta run (2026-02-28, after local release rebuild):
- Inference comparator (new artifact):
  - `uv run --no-project python scripts/rust_suite.py compare --dataset inspire --limit 5000 --pair-count 5000 --n-jobs 4 --require-non-dev-rust 0 --require-rust-release 1 --write-json scratch/compare_a1_fused_inspire5k_20260228.json`
  - Result snapshot: python `19.984s` / `1.508 GB`; rust `4.467s` / `0.917 GB`; feature parity pass (language + non-language).
  - Delta vs `scratch/compare_p3p4_inspire5k_20260228.json`:
    - Rust latency: `5.014s -> 4.467s` (`-0.547s`, `-10.91%`).
    - Rust peak RSS: `0.922 GB -> 0.917 GB` (`-0.005 GB`, `-0.54%`).
    - Python/Rust speedup: `4.322x -> 4.474x`.
- Transfer-mini full compare (new artifact):
  - `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode compare --preset full --target kisti --n-jobs 4 --n-train-pairs 10000 --n-iter 5 --require-rust-release 1 --write-json scratch/profile_transfer_mini_a1_fused_20260228.json`
  - Result snapshot: python `310.934s` / `5.488 GB`; rust `133.866s` / `4.707 GB`; quality parity (`B3 F1=0.960`, `Cluster F1=0.976`, `ClusterMacro F1=0.933`).
  - Delta vs `scratch/profile_transfer_mini_p3p4_20260228.json`:
    - Rust latency: `177.038s -> 133.866s` (`-43.172s`, `-24.39%`).
    - Rust peak RSS: `4.790 GB -> 4.707 GB` (`-0.083 GB`, `-1.73%`).
    - B3/cluster quality: unchanged (`B3 F1 delta = 0.000`; cluster F1 and cluster-macro F1 unchanged).
    - Python/Rust speedup improved: `1.689x -> 2.323x`.
- Gate check against prior transfer-mini artifact:
  - `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode gate --baseline-json scratch/profile_transfer_mini_p3p4_20260228.json --current-json scratch/profile_transfer_mini_a1_fused_20260228.json --gate-run-label rust`
  - Passed with `violations: 0`, workload_id match, runtime delta fraction `-0.243857`, peak RSS delta fraction `-0.017328`, B3 F1 drop `0.0`.

Notes:
- `uv run maturin develop -m s2and_rust/Cargo.toml --release` compiled successfully but install/copy failed due a locked `_s2and_rust` binary (`os error 32`) in this environment, so live validation against a freshly installed wheel is pending unlock.
- Heavy canonical `rust_suite.py` runtime gates + before/after cProfile artifacts remain pending.

Status (2026-02-28): **A1 core + fused follow-up implementation landed; canonical heavy gates/profiling pending.**

## Explicitly Not Starting Yet

## Bundle 5: Artifact Format Unification (Infra/Plumbing)

Owner: TBD

Goal: remove the largest remaining Rust/Python artifact divergences without changing normalization policy.

Authoritative divergence map:
- `docs/rust/artifact_divergence.md`

Proposed scope (staged delivery recommended):
1. `name_counts`: Python pickle + Rust JSON -> one MessagePack artifact readable natively by both.
2. `specter`: pickle -> safetensors (eliminate hidden Python FFI dependency in Rust ingest).
3. `name_tuples`: collapse to one default runtime variant (keep non-default variants as explicit offline inputs).

Notes:
- The current `artifact_divergence.md` keeps ORCID first-k counts as JSON for now; treat any ORCID
  format change as ask-first.
- ORCID first-k counts: out of scope for this bundle (stays JSON; regeneration deferred to Bundle 6 /
  normalization migration).
- Do this as "dual-read + converters + fixtures" first; defer regenerating huge real artifacts until
  you're ready to make the evidence policy shareable.

## Bundle 6: Normalization Migration (Phase 1-2)

Owner: TBD

Scope: phases 1-2 of `docs/normalization_migration.md`.

Notes:
- Phase 1 (policy + canonical example table + pytest invariants) can start immediately.
- Phase 2 (implementation) is easier after Bundle 5 so Phase 3 artifact regeneration goes straight to
  the target formats (MessagePack/Safetensors) instead of regenerating twice.

## Bundle 7: Giant-Block Subblocking Throughput + Phase-A Chunk Tuning (Proposal)

Owner: TBD

Status: **Track B complete (2026-02-28). Track A not started.**

Goal: reduce latency for single-block large-scale inference (100k-600k signatures) in subblocking mode while
preserving bounded-memory operation and explicit quality/semantics controls (`phase_b_mode`).

Scope (two coordinated tracks):
1. Subblock orchestration and lifecycle overhead reduction.
2. Phase-A chunk/memory tuning and parameterized throughput sweep.

Track A - orchestration/lifecycle scope:
- Process subblocks in deterministic cohorts using a pair-budget target, not strictly one-by-one.
- Hoist Phase A backend initialization and indexed signature map construction out of per-subblock loops.
- Replace full seed-sync-on-every-subblock with deterministic delta sync (or sync-every-N-subblocks) with parity gates.
- Remove expensive `deepcopy` patterns in hot subblock loops where object ownership allows safe shallow/structured copies.
- Determinism/aliasing contract: keep subblock iteration order stable, apply seed deltas in a deterministic order, and add
  a parity gate (`cluster_membership_digest` match in compare mode) before accepting any lifecycle changes.

Track B - chunk/memory tuning scope:
- Always pass explicit `total_ram_bytes` for giant-block/subblocked runs.
- Sweep `S2AND_PHASE_A_MAX_CHUNK_PAIRS` on representative 10k/15k subsets (`250k`, `500k`, `1M`, `2M`).
- Before sweeping, ensure the harness emits Phase A gate metrics in machine-readable JSON (avoid log-scraping):
  - plumb `phase_a_accumulator_overflow_early_stop` and an aggregated `phase_a_adaptive_halvings_max` (max across subblocks)
    through `predict_incremental(...)` results and `scripts/_rust_suite/big_block_incremental_cmd.py` single-run JSON output.
- Select the largest setting that satisfies all required gates (as recorded in JSON):
  - `phase_a_adaptive_halvings_max == 0`
  - `phase_a_accumulator_overflow_early_stop == false`
  - acceptable `phase_b_mode` (`exact` when parity is required)
  - best wall time among compliant settings
- If no setting is stable/fast enough, expose/tune `chunk_budget_fraction` and `accumulator_budget_fraction`
  as runtime knobs (currently code-level defaults in `memory_budget.py`).

Start here:
- `s2and/model.py`:
  - subblocked predict orchestration (`predict(...)` with `batching_threshold`)
  - phase-split incremental path (`_predict_incremental_phase_split`, `_phase_a_seed_distances`)
  - seed sync path (`_sync_rust_cluster_seeds`)
- `s2and/memory_budget.py`:
  - `compute_incremental_phase_split_limits(...)`
- `scripts/_rust_suite/big_block_incremental_cmd.py`:
  - sweep harness, compare outputs, and artifact emission
- Context docs:
  - `docs/subclustering.md`
  - `docs/stage_memory_estimates.md`
  - `docs/rust/roadmap.md`

Validation (must-have):
- Perf runs must use a **release** Rust extension build (or published wheel):
  - Build once: `uv run --with maturin maturin develop --release -m s2and_rust/Cargo.toml`
  - For sweeps, pass `--require-rust-release 1` so accidental debug builds don't pollute latency results.
- Keep canonical compare mode for parity-sensitive checks:
  - `uv run --no-project python scripts/rust_suite.py big-block-incremental --mode compare_phase_split --backend rust --subset-dir scratch/inventors_topblock_15k --total-signatures 10000 --seed-signatures 7500 --seed-cluster-count 1200 --batching-threshold 7500 --n-jobs 8 --total-ram-bytes 34359738368 --write-json scratch/big_block/compare_phase_split_bundle7_10k.json --require-rust-release 1 --full-run`
- Chunk-pairs sweep (PowerShell example loop; one JSON per candidate):
  - `$env:S2AND_PHASE_A_MAX_CHUNK_PAIRS='250000'; uv run --no-project python scripts/rust_suite.py big-block-incremental --mode single --backend rust --subset-dir scratch/inventors_topblock_15k --total-signatures 14995 --seed-signatures 11245 --seed-cluster-count 1800 --batching-threshold 7500 --n-jobs 4 --total-ram-bytes 34359738368 --single-write-json scratch/big_block/bundle7_chunkpairs_250k_14995.json --require-rust-release 1 --full-run`
  - `$env:S2AND_PHASE_A_MAX_CHUNK_PAIRS='500000'; uv run --no-project python scripts/rust_suite.py big-block-incremental --mode single --backend rust --subset-dir scratch/inventors_topblock_15k --total-signatures 14995 --seed-signatures 11245 --seed-cluster-count 1800 --batching-threshold 7500 --n-jobs 4 --total-ram-bytes 34359738368 --single-write-json scratch/big_block/bundle7_chunkpairs_500k_14995.json --require-rust-release 1 --full-run`
  - `$env:S2AND_PHASE_A_MAX_CHUNK_PAIRS='1000000'; uv run --no-project python scripts/rust_suite.py big-block-incremental --mode single --backend rust --subset-dir scratch/inventors_topblock_15k --total-signatures 14995 --seed-signatures 11245 --seed-cluster-count 1800 --batching-threshold 7500 --n-jobs 4 --total-ram-bytes 34359738368 --single-write-json scratch/big_block/bundle7_chunkpairs_1m_14995.json --require-rust-release 1 --full-run`
  - `$env:S2AND_PHASE_A_MAX_CHUNK_PAIRS='2000000'; uv run --no-project python scripts/rust_suite.py big-block-incremental --mode single --backend rust --subset-dir scratch/inventors_topblock_15k --total-signatures 14995 --seed-signatures 11245 --seed-cluster-count 1800 --batching-threshold 7500 --n-jobs 4 --total-ram-bytes 34359738368 --single-write-json scratch/big_block/bundle7_chunkpairs_2m_14995.json --require-rust-release 1 --full-run`
- Orchestration overhead check: add a second benchmark configuration that intentionally yields **many subblocks**
  (e.g., expose a `subblocking_maximum_size` knob for `make_subblocks(...)`, or choose a block known to split heavily),
  and record `subblock_count` + total overhead time separately from model/constraint runtime.
- Decision artifact:
  - Record selected setting + rejected settings with gate reasons in a summary JSON under `scratch/big_block/`.
  - Include run metadata so the decision is reproducible: git SHA, `rust_extension_identity`, `n_jobs`, `total_ram_bytes`,
    resolved `chunk_pairs` + budgets, `phase_b_mode`, peak RSS, wall time, and the Phase A gate metrics.

### Track B Results (2026-02-28)

Code changes (prerequisite for machine-readable gate metrics):
- `s2and/model.py`: plumbed `phase_a_adaptive_halvings_max` (max across subblocks) through
  `_predict_incremental_phase_split()` → `_build_incremental_result()`.
- `scripts/_rust_suite/big_block_incremental_cmd.py`: surfaced both `phase_a_accumulator_overflow_early_stop`
  and `phase_a_adaptive_halvings_max` in single-run JSON output.

Sweep configuration:
- Block: `j kim` (14995 signatures, 4 subblocks avg 3749 sigs each)
- Subset: `scratch/inventors_topblock_15k`
- Params: `--seed-signatures 11245 --seed-cluster-count 1800 --batching-threshold 7500 --n-jobs 4 --total-ram-bytes 34359738368`
- Rust extension: release build, crate v0.31.0, `debug_assertions=false`
- Machine: 32 GB RAM, Windows 10 Pro

Results:

| chunk_pairs | predict_s | total_s | peak_rss_gb | halvings_max | overflow | phase_b | digest |
|-------------|-----------|---------|-------------|--------------|----------|---------|--------|
| 250k | 1615.4 | 1620.9 | 2.590 | 0 | false | exact | 260096cc… |
| 500k (default) | 1524.7 | 1530.7 | 2.539 | 0 | false | exact | 260096cc… |
| **1M (selected)** | **1455.0** | **1459.9** | **2.379** | 0 | false | exact | 260096cc… |
| 2M | 1613.0 | 1617.7 | 2.670 | 0 | false | exact | 260096cc… |

All four settings pass every gate (`halvings_max==0`, `overflow==false`, `phase_b==exact`).
All produce identical `cluster_membership_digest` — bit-for-bit clustering parity confirmed.

Selected setting: **`S2AND_PHASE_A_MAX_CHUNK_PAIRS=1_000_000`**
- 10.5% wall-time improvement over the current 500k default (1455s vs 1525s).
- Lowest peak RSS of all candidates (2.38 GB vs 2.54 GB at 500k).
- 2M regresses because per-chunk model prediction time (~32s/chunk) exceeds the savings from fewer chunks.
- Sweet spot: 1M processes Phase A in 7 chunks (vs 13 at 500k) with ~17s/chunk prediction.

JSON artifacts: `scratch/big_block/bundle7_chunkpairs_{250k,500k,1m,2m}_14995.json`

Next steps:
- Change the default from 500k → 1M in `memory_budget.py` (or promote to a config knob).
- Track A orchestration work (lifecycle overhead, seed sync, deepcopy) remains available for further gains.

## The Rest
- A0 (fully fused Rust pipeline): high effort; revisit after the A1 batch/block distance-matrix pipeline lands.
- P1 (3b) full Vec-backed internal storage refactor: roadmap marks this ask-first; profile after P3/P4.
