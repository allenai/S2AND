# Rust Ideas Frontier

Status date: 2026-02-28

Execution bundles / where to start: `docs/work_plan.md`.

Latest updates:
- 2026-02-28: Work bundles + "where to start" pointers captured in `docs/work_plan.md`.
- 2026-02-28: Transfer-mini latency diagnostics added (cleanup/preprocess toggles + per-trial telemetry).
  Artifacts: `scratch/diagnostics/transfer_mini_diag_{default,default_r2,default_r3,no_cleanup,force_py_papers,force_py_papers_no_cleanup}_20260228.json`.
  Readout: peak RSS is stable in deferred-paper default runs (`~4.79 GB`) but LightGBM stages show high single-run wall-time variance; hyperopt parameter hashes and fitted tree counts are unchanged run-to-run.
- 2026-02-28: L1b production training-path port landed with targeted cleanup:
  `scripts/transfer_experiment_seed_paper.py` now runs
  per-dataset `evict_rust_featurizer(dataset)` + `gc.collect()` boundaries
  with `Telemetry: post_rust_cleanup ...` logging (no global inference-path clear).
- 2026-02-27: Session execution notes captured for L1b/P1/P2 with benchmark artifacts.
  L1b cleanup boundary first landed in benchmark harness
  (`scripts/_rust_suite/transfer_mini_cmd.py`) and was later ported to
  production training scripts on 2026-02-28.
  Median of 3 Rust-only `transfer-mini --preset full` runs improved from
  `203.833s` to `183.615s` (`-9.92%`) with pairwise stage medians:
  `union_pairwise_fit 69.741s -> 61.591s` and
  `union_nameless_pairwise_fit 50.010s -> 33.132s`; cleanup snapshot shows
  `per_dataset_complete 5.133 GB -> post_rust_cleanup 5.037 GB` (`-0.096 GB`).
  Artifacts:
  `scratch/baseline_transfer_mini_rust_run{1,2,3}_b9315f7_20260227_142343.json`,
  `scratch/after_l1b_transfer_mini_rust_run{1,2,3}_b9315f7_20260227_144537.json`.
  Important caveat: global cache clear can hurt large inference/subblocking and
  mixed predict workflows by forcing featurizer rebuild churn; if ported to
  production, prefer training-only targeted per-dataset eviction + GC at the
  Rust->LightGBM boundary (do not clear globally in inference paths).
- 2026-02-27: P1 (3a) landed in Rust for indexed constraints hot path.
  `get_constraints_matrix_indexed` now validates index bounds once, builds a
  one-time index-native lookup, and uses that lookup in the parallel loop
  (removes repeated per-pair string/hash lookups in this path).
  File: `s2and_rust/src/lib.rs` (`get_constraints_matrix_indexed`).
  Verification:
  `uv run pytest -q tests/test_feature_port_parity.py tests/test_rust_from_json_paths.py`
  => `25 passed, 1 skipped`.
  Compare artifact:
  `scratch/after_p1a_compare_b9315f7_20260227_145518.json`.
  Transfer compare artifact:
  `scratch/after_p1a_transfer_mini_b9315f7_20260227_145640.json`.
  Note: P1 (3b) full Vec storage refactor remains ask-first and was not started.
- 2026-02-27: Training-mode deferred paper preprocessing into Rust was not newly
  profiled/landed as part of this session execution plan. Keep the contract-test
  and memory-risk controls in `docs/rust/training_preprocessing_plan.md` as the
  source of truth before declaring this item complete.
- 2026-02-27: P2 marked implemented. Rust stores per-paper `specter_norm` and
  uses norm-aware cosine with compatibility fallback for older caches that lack
  precomputed norms.
- 2026-02-27: L1 stress gate landed with fast default + opt-in heavy AMiner loop. Heavy `from_json_paths` stress run passed `6/6` with no crash (`scratch/baselines_20260227/stress_rust_from_json_paths_aminer_6x_20260227.json`).
- 2026-02-27: P0 landed across all hot paths. Rust now exposes `get_constraints_matrix` and `get_constraints_matrix_indexed`, model paths batch unresolved constraints, and telemetry reports `constraint_batch` API mode/call counts. Latest canonical largest-block compare smoke (`scratch/baselines_20260227/largest_block_compare_smoke_200_20260227.json`): predict `4.331s -> 1.069s` (`4.05x`), total `375.726s -> 326.999s` (`1.15x`), peak RSS `13.652 GB -> 9.855 GB` (`-27.8%`), partition diff `0/200`.
- 2026-02-27: Phase-split incremental now surfaces Phase A accumulator overflow explicitly: return field
  `phase_a_accumulator_overflow_early_stop` + `Telemetry: phase_split_phase_a_overflow ...` log line.
- 2026-02-27: Rust batch startup fixed-overhead calibration now page-touches probe allocations and is only adopted when it
  increases conservatism (never reduces configured fixed overhead used for chunk planning).
- 2026-02-27: Windows RAM/RSS detection now has WinAPI fallbacks (no `psutil` required) for memory budgeting.
- 2026-02-26: L5 landed. Rust featurizer disk cache now keys and validates on artifact metadata (stale/mismatched metadata is treated as cache miss).
- 2026-02-26: Memory prediction P2–P4 landed (see `docs/stage_memory_estimates.md`); Phase A accumulator default is now calibrated (`INCREMENTAL_ACCUMULATOR_ENTRY_BYTES=200`).
- 2026-02-26: Memory-model follow-up landed: Phase A pair-buffer bytes are modeled, Rust batch persistent-row overhead is calibrated, and the 14,995-signature gate is now `underpredicted=False` for both Phase A and Rust batch telemetry.
- 2026-02-26: Added startup 3-probe machine-local fixed-overhead calibration for Rust batch planning (one-time per process; feeds `fixed_overhead_bytes` in chunk planning).
- 2026-02-25: L0, L3, L4 implemented (Python-side reuse + seed-sync dedupe + persistent warm).

This file tracks the current optimization frontier beyond the baseline/gate docs.
Use this as the working list for next Rust speed + memory wins.

## Context

Recent analysis shows:
- Rust is already faster and usually lower RSS on maintained small/medium gates.
- Remaining hot spots are now mostly in lifecycle overhead (featurizer acquire/build/save),
  plus per-pair loops once lifecycle is amortized.
- At very large block scale (100k+ signatures), the limiting factor is `O(U^2)` work/memory,
  not just ingest/build overhead.
- Largest-block profiling shows Rust can be slower than Python today when featurizer rebuilds
  are repeated inside a single predict; fixing L0/L1 is a prerequisite for 100k work.

Related references:
- `docs/rust/profiling/2026-02-26.md`
- `docs/subclustering.md`
- `docs/rust/baselines.md`
- `docs/stage_memory_estimates.md`

See also: `docs/stage_memory_estimates.md` has the integrated phasing plan that sequences lifecycle work
with memory-telemetry changes.

## Priority Opportunities

These opportunities assume the lifecycle issues in L0/L1 are addressed; otherwise, featurizer
rebuild cost dominates and per-pair optimizations will be drowned out at large-block scale.

### P0: Batch constraint evaluation API in Rust

Problem:
- Constraints are currently called pair-by-pair from Python.
- This creates high FFI overhead in large `U x S` loops.

Idea:
- Add Rust APIs like `get_constraints_matrix` and `get_constraints_matrix_indexed`.
- Evaluate large pair batches in one call, returning vector/matrix outputs.

Expected impact:
- Large speedup in constraint-heavy incremental phases.
- Slight memory reduction from less Python-side per-call overhead and temporary objects.

Risk:
- Medium implementation risk; low semantic risk if logic is reused.

Status:
- Implemented (2026-02-27).
- Rust APIs added on `RustFeaturizer`: `get_constraints_matrix(...)` and
  `get_constraints_matrix_indexed(...)`, reusing existing single-pair semantics.
- Python wrappers enforce the batch-constraint API contract (no per-pair compatibility fallback).
- Integrated across all current hotspots:
  `distance_matrix_helper` path, `predict_incremental_helper`, and
  `_phase_a_seed_distances`.
- Regression/parity coverage added in:
  `tests/test_feature_port_parity.py`,
  `tests/test_regression_fixes.py`,
  `tests/test_cluster_incremental.py`.
- Verification snapshot: `uv run pytest -q tests/test_rust_from_json_paths.py tests/test_feature_port_parity.py tests/test_regression_fixes.py tests/test_cluster_incremental.py`
  => `56 passed, 1 skipped`.

### P1: Index-native Rust featurizer hot path

Problem:
- Even in indexed batch mode, per-pair execution still does repeated map lookups.

Idea:
- Build contiguous index-addressable structures at featurizer build time.
- Keep string IDs for boundaries, but use integer IDs internally in hot loops.

Expected impact:
- Lower per-pair latency from better cache locality and fewer hash lookups.
- Lower memory overhead from reducing repeated key traversals and metadata churn.

Risk:
- Medium; needs careful parity validation and cache-version bump.

Status:
- 3a implemented (2026-02-27): `get_constraints_matrix_indexed` now uses
  index-bound validation + one-time lookup build + lookup-only parallel loop.
- 3a is low-risk and landed with parity tests passing.
- 3b (full Vec-backed internal storage refactor) remains optional/ask-first and
  is not implemented in this session.

### P2: Precompute SPECTER norms (or normalized vectors)

Status:
- Implemented (2026-02-27).
- `PaperData` now stores `specter_norm: Option<f64>` (`s2and_rust/src/lib.rs:117`).
- SPECTER cosine uses precomputed norms when present via
  `cosine_sim_with_norms(...)` (`s2and_rust/src/lib.rs:1820`, callsite at
  `s2and_rust/src/lib.rs:2218`).
- Backward-compat fallback remains in place for artifacts without norms:
  `cosine_sim_vec_f32(...)` (`s2and_rust/src/lib.rs:1801`), so older `.bin`
  caches can still load but run the slower path until rebuilt.

### P3: Reduce matrix-featurization fixed overhead

Problem:
- Batch methods perform full upfront validation and repeated ID-order cloning.

Idea:
- Cache immutable ID order once per featurizer.
- Remove/relax repeated validation on trusted internal call paths.

Expected impact:
- Moderate speedup for many repeated medium/large batches.
- Small memory reduction from fewer transient allocations.

Risk:
- Low to medium; guard with debug checks and tests.

Status:
- Implemented (2026-02-28, Bundle 3).
- `cached_signature_id_order: OnceLock<Vec<String>>` and
  `cached_full_feature_count: OnceLock<usize>` in `s2and_rust/src/lib.rs`
  cache immutable ID order and full feature count per featurizer instance.
- `signature_id_order()` computes and caches sorted IDs on first call;
  subsequent calls return the cached slice with zero allocation.
- Parity gates: `uv run pytest -q tests/test_feature_port_parity.py tests/test_rust_from_json_paths.py`
  => `25 passed, 1 skipped`.
- Runtime evidence: `scratch/profile_transfer_mini_bundle1_4_20260228.json`.

### P4: Counter kernel tuning (`counter_jaccard_data`)

Problem:
- Current intersection uses binary search per token.

Idea:
- Add two-pointer merge path when both vectors are sorted and similarly sized.
- Keep binary-search path for very skewed sizes.

Expected impact:
- Moderate speedup in text-feature heavy loops.
- Neutral memory.

Risk:
- Low.

Status:
- Implemented (2026-02-28, Bundle 3).
- `counter_jaccard_data` (`s2and_rust/src/lib.rs:1505`) now uses:
  - **Two-pointer merge** when `large.len() < small.len() * 4` (similarly sized vectors).
  - **Binary-search** fallback when one vector is much larger (skewed sizes).
- Threshold heuristic (`4×`) avoids the merge path when binary search would
  skip most of the larger vector.
- Parity gates: `uv run pytest -q tests/test_feature_port_parity.py tests/test_rust_from_json_paths.py`
  => `25 passed, 1 skipped`.
- Runtime evidence: `scratch/profile_transfer_mini_bundle1_4_20260228.json`,
  `scratch/compare_bundle1_4_inspire5k_20260228.json`.

## 100k+ Scale Memory + Speed Ideas

At 100k+ unassigned signatures **in a single Phase B invocation**, dense Phase B
(`U*(U-1)/2`) is the core bottleneck.

Operationally, giant blocks (e.g., a 600k-signature block) are made feasible by
subblocking into many small subblocks (order-of-magnitude: ~10k subblocks). In
that regime, any approach that tries to materialize global `U^2` buffers is not
viable even on 128GB RAM machines.

Dense Phase B condensed-vector bytes (float64) is:
`U * (U - 1) // 2 * 8`.

Examples:
- `U=100,000` => ~40 GB (condensed vector only; excludes overhead)
- `U=600,000` => ~1.3 TiB (condensed vector only; excludes overhead)

Subblocking feasibility sanity check (illustrative):
- 600k signatures split into ~10k subblocks ⇒ ~60 signatures/subblock on average.
- Dense Phase B within a ~60-signature subblock is tiny:
  `60 * 59 // 2 * 8 = 14,160 bytes` for the float64 condensed vector.


## Lifecycle & Architecture Opportunities

Profiling evidence (see `docs/rust/profiling/2026-02-26.md`) shows featurizer lifecycle overhead
dominates Rust predict time: 83% on large blocks, 54% on medium blocks. The ideas
below target this layer, which sits above the per-pair hot path addressed by P0–P4.

### L0: Fix featurizer caching / reuse architecture

Problem:
- `Clusterer.__init__` defaults `use_cache=False` (`model.py:384`).
- With this default, every call to `_get_rust_featurizer()` triggers a full rebuild.
- There are 3+ independent callsites per `predict_helper` invocation:
  `distance_matrix_helper()`, `_sync_rust_cluster_seeds()`, and once per batch in
  `many_pairs_featurize()`.
- On aminer large block (157K sigs): 5 consecutive rebuilds at 60–80s each = 237.7s
  of pure waste (83% of predict time).
- `warm_rust_featurizer()` intends to amortize cold-start cost, but it only helps if
  it populates a reuse mechanism; otherwise it builds then drops the featurizer.
- `Clusterer.use_cache` currently overloads two concerns: Python pair-feature caching
  (expensive/unbounded) and Rust featurizer reuse (cheap/high-ROI). They should be
  controlled independently.

Idea:
- Ensure `_get_rust_featurizer()` builds once per dataset and reuses across all
  callsites within a predict. Options:
  - Thread the featurizer instance through the call chain instead of re-acquiring
    from the cache each time (avoids cache-invalidation subtleties).
  - Add a Rust-featurizer-specific reuse flag (preferred) rather than flipping
    `Clusterer.use_cache` globally (keep Python pair-feature caching off by default).
  - Consider a per-run reuse handle keyed by `runtime_context.run_id` so repeated
    callsites (constraints, seeds, featurize) share one instance even if the dataset
    object is re-instantiated.
  - If changing the default to `use_cache=True`, confirm Python-side caching behavior
    won't balloon RSS on long runs.

Expected impact:
- Large blocks: 237.7s → ~0s rebuild overhead. Rust predict drops from ~287s to ~50s
  (Python is 205s).
- Medium blocks (kisti): 10–15s savings.
- This is a Python-side change, not a Rust change.

Risk:
- Low semantic risk if reuse is dataset-scoped and seed updates are applied to the
  reused featurizer instance.
- Residual correctness risk is low now that L5 cache identity + metadata validation
  is implemented; stale cache artifacts are now treated as misses.

Status:
- Implemented (2026-02-25). Rust featurizer reuse now persists even when
  `use_cache=False` (decoupled from Python pair-feature caching). Disk cache
  still follows `use_cache`. Tests: `tests/test_feature_port_cache.py`.

### L1: Fix segfault on repeated featurizer builds with large datasets

Problem:
- Rust crashes (0xC0000005) on the 5th consecutive `from_json_paths` build with
  157K-signature datasets. Blocks >1M pairs cannot use Rust at all.
- Root cause: likely use-after-free or double-free in FFI layer when dropping and
  rebuilding featurizers in rapid succession.

Idea:
- Audit FFI ownership model for the featurizer lifecycle. Check for dangling
  references when a previous featurizer is dropped while a new one is building.
- Add a stress test: build/drop 10× in a loop on a large dataset.

Expected impact:
- Unblocks Rust for all block sizes. Currently a hard showstopper.

Risk:
- Medium; may require rethinking FFI object lifecycle.

Status:
- Stress regression gate implemented (2026-02-27).
- Added reusable runner:
  `scripts/rust_suite.py stress-rebuild` with loop telemetry, explicit
  `del` + `gc.collect()`, and JSON artifact output.
- Added tests:
  `tests/test_rust_from_json_paths.py` (fast default smoke + opt-in heavy AMiner).
- Heavy opt-in gate command:
  `uv run --with psutil python scripts/rust_suite.py stress-rebuild --dataset aminer --build-path from_json_paths --repeats 6 --num-threads 1 --rss-sample-ms 50 --require-rust-release 1 --write-json scratch/baselines_20260227/stress_rust_from_json_paths_aminer_6x_20260227.json`
  passed `6/6` (no crash reproduced).

### L1b: Rust allocation residue degrades subsequent Python-only stages

Problem:
- Even when no Rust code is running, Python-only stages (LightGBM hyperopt) run
  2.5–3.4× slower in the Rust process than in an equivalent Python-only process.
- Evidence: `profile_transfer_mini` (2026-02-25, 0b3e877) shows `union_pairwise_fit`
  at 117.8s (Rust process) vs 46.2s (Python process) on identical pre-computed numpy
  arrays with the same hyperopt seed. `union_nameless_pairwise_fit` is 99.7s vs 29.4s.
- Individual LightGBM hyperopt trials show up to 8.8× slowdown on the same
  hyperparameters / data (e.g., nameless trial 4: 35s vs 4s).
- The Rust process carries ~700 MB higher RSS at the start of LightGBM stages
  (5.1 GB vs 4.4 GB) from unreturned `from_dataset` allocations.
- The `from_dataset` build cost itself doubles under memory pressure: early kisti
  builds cost ~11s, late builds cost 22–25s at higher RSS, suggesting heap/allocator
  degradation compounds over the run.
- Likely contributors: Windows CRT heap fragmentation from repeated ~500 MB
  build/teardown cycles; CPU L3 cache pollution from large dead allocations;
  possible scheduling contention between parked Rayon threads and OMP threads.

Idea:
- After the per-dataset featurization phase (before LightGBM hyperopt), explicitly
  release all Rust featurizer references and force `gc.collect()` to return memory
  to the OS. On Windows, consider `ctypes.windll.kernel32.SetProcessWorkingSetSize`
  or `malloc_trim` equivalent to actually shrink RSS.
- Investigate Rayon thread pool lifetime: if the default pool (num_cpus threads) is
  parked during LightGBM training, either shut it down between stages or pin its
  size to 0/1 when not in use.
- Long-term (with L0 fixed): if the featurizer is kept alive for reuse, its memory
  is legitimately in use and shouldn't cause fragmentation. The issue largely
  disappears when L0 eliminates repeated build/teardown cycles.

Expected impact:
- Transfer-mini workflow: ~142s savings on pairwise fit stages alone (the second
  largest contributor after `from_dataset` rebuilds).
- Partially masked by L0: once L0 eliminates repeated builds, there is less
  fragmentation and the LightGBM degradation should shrink. But RSS residue from a
  single large featurizer (~500 MB for kisti) may still affect cache behavior.

Risk:
- Low for the `gc.collect()` / explicit-release approach.
- Medium for Rayon pool management (may need PyO3 API changes or a Rust-side
  shutdown hook).

Status:
- Harness boundary implemented (2026-02-27) in
  `scripts/_rust_suite/transfer_mini_cmd.py`:
  clear Rust featurizer cache references, call `gc.collect()`, and record
  `post_rust_cleanup` RSS snapshot before LightGBM stages.
- Verified in `transfer-mini` artifacts (see latest-updates bullets above).
- Production training-path port landed (2026-02-28):
  `scripts/transfer_experiment_seed_paper.py` now applies targeted
  per-dataset eviction + GC boundaries (`Telemetry: post_rust_cleanup ...`).
- Keep guardrail: avoid global cache clear in production inference/subblocking
  paths where rebuild churn can dominate and hurt latency.

### L2: Move featurizer disk save outside the cache lock

Problem:
- `RustFeaturizer.save` takes 7.1s (kisti) and executes inside
  `_RUST_FEATURIZER_CACHE_LOCK`, blocking all concurrent threads from acquiring a
  featurizer for the full serialization duration.

Idea:
- Build and insert into the memory cache inside the lock, then release the lock
  and write to disk outside it. Other threads can proceed with the in-memory
  featurizer while the save completes.

Expected impact:
- 7s concurrent unblock in multi-threaded predict.
- No latency change for single-threaded use.

Risk:
- Low. Disk save is idempotent; a concurrent reader that misses the disk cache
  simply takes the in-memory path.

Status:
- Implemented (2026-02-26). `_get_rust_featurizer` now inserts the in-memory
  cache entry under lock and defers
  `_save_rust_featurizer_cache_best_effort(...)` until after lock release.
- Regression test:
  `tests/test_feature_port_cache.py::test_disk_cache_save_runs_outside_global_cache_lock`.
- Artifacts:
  `scratch/compare_save_outside_lock_before.json`,
  `scratch/compare_save_outside_lock_after.json`,
  `scratch/profile_rust_featurizer_reuse_before.json`,
  `scratch/profile_rust_featurizer_reuse_after.json`,
  `scratch/profile_transfer_mini_save_outside_lock.json`.

### L3: Reduce `_sync_rust_cluster_seeds` call frequency

Problem:
- `_sync_rust_cluster_seeds` is called at multiple points: top of `predict_helper`,
  after every single-letter subblock, etc. (`model.py:940, 991, 998, 1148`).
- Without caching, each call triggers a full featurizer rebuild.
- Even with caching, the lock acquisition and seed-update work is repeated
  redundantly when seeds haven't changed.

Idea:
- Track a seed-version counter. Only call the Rust `update_cluster_seeds` when seeds
  have actually changed since the last sync.
- Alternatively, push seed updates lazily: mark seeds dirty, sync once before the
  next operation that needs them.

Expected impact:
- Large blocks: 75.7s → near-zero (with L0 also fixed).
- Medium blocks: eliminates redundant lock acquisitions.

Risk:
- Low if dirty-tracking is straightforward.

Status:
- Implemented (2026-02-25). Dataset seed versioning skips redundant Rust
  `update_cluster_seeds` calls. Tests: `tests/test_regression_fixes.py`.

### L4: Make warm/prewarm actually persistent

Problem:
- `warm_rust_featurizer()` is documented as "preload into memory", but if Rust
  featurizer reuse is disabled it only builds then drops the featurizer.
- Benchmark scripts often call warm but keep `Clusterer.use_cache=False`, so the
  warm step doesn't reduce later builds.

Idea:
- Ensure warm uses the Rust-featurizer reuse mechanism (even when Python pair-feature
  caching is off).
- Add a small validation hook in benchmarking scripts: log `rust_featurizer_build_count`
  before/after predict to confirm reuse.

Expected impact:
- Removes cold-start overhead from measured predict latency when desired.
- Makes profiling scripts reflect the intended steady-state.

Risk:
- Low.

Status:
- Implemented (2026-02-25). `warm_rust_featurizer()` now persists the Rust
  featurizer even when Python pair-feature caching is off.

### L5: Strengthen Rust featurizer disk cache identity + validation

Problem:
- Disk cache keying can be too coarse (e.g., keyed on dataset name/sizes) and may
  load a featurizer built from different underlying artifacts.
- This can silently break parity when JSON-ingest inputs or name-count artifacts differ.

Idea:
- Encode artifact identity into the cache key (absolute paths + mtime/size or content
  hash for signatures/papers/specter/name-count JSON).
- Persist a metadata manifest inside the cache file and verify it on load (treat
  mismatches as cache miss).
- Bump `RUST_FEATURIZER_CACHE_VERSION` and surface cache-hit/miss reasons in telemetry.

Expected impact:
- Makes disk caching safe to rely on across runs and across datasets.
- Improves debuggability when caches are invalidated.

Risk:
- Low ongoing risk; metadata mismatch now degrades to cache miss instead of stale-load behavior.

Status:
- Implemented (2026-02-26).
- `_get_rust_featurizer` now computes artifact metadata, validates sidecar metadata
  before `RustFeaturizer.load`, and writes sidecar metadata on cache save.
- New tests:
  `tests/test_feature_port_cache.py::test_disk_cache_metadata_mismatch_skips_load`
  and
  `tests/test_feature_port_cache.py::test_disk_cache_metadata_match_loads_without_rebuild`.
- Supporting artifacts:
  `scratch/compare_l5_mem_tune_final_20260226.json`,
  `scratch/profile_rust_featurizer_reuse_l5_mem_tune_final_20260226.json`,
  `scratch/baselines_20260227/profile_transfer_mini_full_20260227.json`.

### L6: Keep large Rust ingest artifacts out of the repo by default

Problem:
- Rust-ingest artifacts like `name_counts_rust.json` can be hundreds of MB and are
  easy to accidentally leave untracked or commit.
- Benchmark subsets (e.g., inventors artifacts) can similarly bloat the working tree.

Idea:
- Default generators to write into `scratch/` (or require explicit `--output` under `data/`).
- Add `.gitignore` patterns for common local artifacts (`data/name_counts_rust.json`,
  `data/inventors_s2and/*`, etc.).

Expected impact:
- Less repo churn and lower risk of accidental giant commits.
- Faster local iteration (fewer huge files in status/diff).

Risk:
- Low.

## Pipeline Architecture Opportunities

These ideas restructure the predict pipeline itself rather than optimizing within
the current architecture.

### A0: Fused constraint + featurize pipeline in Rust

Problem:
- Current flow: Python generator (`distance_matrix_helper`) yields one pair at a
  time → Python/Rust checks constraint per pair → surviving pairs batch-sent to Rust
  for featurization → back to Python for LightGBM → Python fills distance matrix.
- This means 3.3M Python loop iterations, 3.3M FFI calls for constraints, and a
  separate batch FFI call for featurization.

Idea:
- Send all pair indices for a block/batch to Rust in one call. Rust evaluates
  constraints internally, skips rejected pairs, featurizes survivors, returns the
  feature matrix (or even the distance vector if LightGBM is also callable from Rust).
- Subsumes P0 (batch constraints) and goes further.

Expected impact:
- Eliminates the entire `distance_matrix_helper` generator loop.
- Python constraint cost: 77.7s for 3.3M pairs → eliminated.
- Rust per-pair FFI constraint cost: 11.6s → <1s (no FFI boundary per pair).
- Also enables internal Rust parallelism (rayon) over pairs, which the Python
  generator cannot do.

Risk:
- High implementation effort. Requires Rust to understand constraint semantics,
  cluster seeds, and partial supervision — all currently Python-side.
- Can be staged: batch constraints first (P0), then fused featurize, then fused
  predict.

### A1: Replace per-pair generator with batch-oriented block pipeline

Problem:
- `distance_matrix_helper()` interleaves constraint evaluation with pair generation
  in a single-threaded Python generator. Each pair is yielded individually. This is
  inherently sequential and prevents overlap between constraint evaluation and
  featurization.

Idea:
- For each block, generate all pair indices (vectorized numpy), batch-evaluate
  constraints (one Rust call or vectorized Python), partition into
  constrained/unconstrained, featurize only unconstrained pairs, predict, fill
  distance matrix.
- This is a less invasive version of A0: keeps the Python orchestration but moves
  from per-pair to per-block batches.

Expected impact:
- Eliminates Python-loop overhead for pair generation and constraint dispatch.
- Enables numpy vectorization for pair index generation.
- Natural fit for batch constraint API (P0).

Risk:
- Medium. Must handle cross-batch block boundaries (current generator tracks block
  transitions for incremental clustering).
- Partial supervision dict lookup may need vectorization.

## Suggested Execution Order

The best order depends on workload shape, but for **giant blocks handled via subblocking**
(order-of-magnitude: ~10k subblocks), the multiplier is per-subblock overhead, so start by
amortizing lifecycle work and removing Python per-pair loops.

Recommended execution order (giant-block / subblocked focus):

1. L0 (featurizer reuse) — stop rebuilding within a run [done 2026-02-25]
2. L4 (make warm/prewarm persistent) — ensure reuse is actually exercised [done 2026-02-25]
3. L3 (reduce seed sync frequency) — avoid per-subblock redundant sync/lock work [done 2026-02-25]
4. L1 (stress gate + repeated-build segfault triage/fix) — correctness/robustness before deep optimization [done 2026-02-27]
5. L2 (save outside lock) — remove avoidable lock contention [done 2026-02-26]
6. P0 (batch constraints across hot paths) — eliminate per-pair Python/FFI constraint overhead [done 2026-02-27]; keep A1 as optional follow-up if generator overhead remains dominant
7. P3 (reduce matrix-featurization fixed overhead) — high leverage when many batches/subblocks [done 2026-02-28]
8. P2 (precompute SPECTER norms) — cheap per-pair win if embeddings are hot [done 2026-02-27]
9. P1 (3a) + P4 (per-pair micro-optimizations) — only after the above [done 2026-02-28]; P1 (3b) full Vec refactor remains ask-first
10. L5 (disk cache identity + validation) — enable safe cross-run reuse [done 2026-02-26]
11. A0 (fused Rust pipeline) — long-term architectural target
12. L6 (artifact hygiene) — reduce repo churn and accidental giant commits [done 2026-02-28]

If you need **better-than-subblock-local semantics** at large `U` (phase-split incremental / sparse modes):

1. L0/L4/L3 (lifecycle + seed sync) to avoid drowning in overhead
2. P0 or A1 (batch constraints / batch pipeline) to remove per-pair Python loops
3. P1–P4 micro-optimizations

## Validation Expectations

For each change:
- Preserve current parity gates on maintained workloads.
- Track latency and peak RSS deltas in baseline scripts.
- For large-scale modes, add explicit quality-vs-cost reporting (partition diff, B3 delta, wall time, peak RSS).
