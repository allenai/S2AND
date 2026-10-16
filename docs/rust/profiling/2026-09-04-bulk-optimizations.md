# Bulk Arrow optimizations: exact outputs, time, and peak memory

Four changes passed the requested gates on the measured workloads: at least
10% less time in the affected category, exact outputs, and no more than 10%
additional peak memory. No public API, artifact schema, or production
dependency changed.

## Full workload

The benchmark calls `Clusterer.predict_from_arrow` with `n_jobs=10`, both
native classifiers, average linkage, and no subblocking threshold. AMiner's
five largest natural blocks contain 9,267 signatures and 8,762,170 pairs.
The explicit limit of 2,500 signatures per block truncates none of these
blocks. There are 11 feature chunks, each at most 1,000,000 pairs, and
3,087,972 pairs scored by each model.

Both baseline and candidate use optimized release extensions, verified with
`debug_assertions=False`. The same historical v1.21 pairwise models,
feature contracts, name counts, Arrow bytes, block order, and EPS are used.
This is a performance benchmark, not a current-model quality evaluation.
Hardware: AMD Ryzen Threadripper PRO 3945WX, 12 physical / 24 logical cores.

The fresh full baseline took **87.771 s**. Three optimized requests took
**45.478, 44.517, and 44.458 s**: a **49.28% reduction** using the optimized
median. Timing includes trace hashing equally in both implementations, but
excludes imports, dataset opening/validation, build time, and initial scratch
artifact preparation. This is separate from the earlier uninstrumented
profiling baseline whose median was 84.091 s.

Windows process high-water marks, measured through `GetProcessMemoryInfo`
without sampling:

| Full-process measurement | Baseline bytes | Optimized maximum bytes | Change |
|---|---:|---:|---:|
| Peak working set | 2,971,725,824 | 2,981,769,216 | +0.34% |
| Peak committed memory | 2,615,214,080 | 2,629,324,800 | +0.54% |

The baseline and candidate use fresh processes. The candidate maximum covers
all three requests. Individual changes were also checked separately below.
These are actual OS measurements, not S2AND's fallback RSS estimates printed
when psutil is unavailable. Tests that deliberately hold both reference and
candidate arrays are excluded from performance/memory measurements.

## Retained changes

| Category | Before | After | Reduction | Isolated peak working-set change |
|---|---:|---:|---:|---:|
| Native scoring | 74.613 s | 35.668 s | 52.20% | -0.42% |
| Feature construction/projection | 6.233 s | 4.069 s | 34.72% | -0.88% |
| Constraints, including Python conversion | 1.822 s | 0.502 s | 72.43% | -16.2% for conversion |
| Python label construction | 1.557 s | 0.847 s | 45.64% | -0.2% |

Scoring times come from the full real-scoring runs. The other phase times
are medians of three real-data requests with scoring bypassed to isolate
those categories. Feature comparison holds the label and constraint changes
constant; label/constraint comparison uses the copied original implementation.
Consequently these rows are separate comparisons, not additive components
of one request. Scorer memory is measured with 50,000 real feature rows per
model in fresh processes; label/wrapper memory uses million-pair fixtures
and three fresh processes per retained variant. Feature memory uses the
whole five-block workload with only the feature path changed.

- **Tree reuse across 64 rows:** process a small row tile through each tree
  before advancing to the next tree. Every row retains its original tree
  accumulation order, initial value, comparisons, and sigmoid operation.
  Both float64 and float32 paths benefit. On 50,000 real rows per model,
  alternating baseline/candidate runs reduced float64 probability time by
  49.4% for the main model and 48.3% for nameless; float32 reductions were
  45.9% and 42.2%. Peak committed memory in the scorer probe fell 0.042%.
- **Skip features for resolved pairs:** when supervision resolves a majority
  of a chunk, use the existing indexed-pair native API for the rows that will
  be scored. Preserve full matrix shapes and column order. Unscored internal
  rows contain NaN placeholders; the existing label mask excludes them from
  classifiers. Classifier call sizes and row order stay unchanged. Free
  temporary pairs, indices, and sparse feature buffers before projection.
  Isolated peak committed memory fell 1.10%.
- **Remove redundant constraint conversions:** PyO3 already returns fresh
  built-in lists of integers and float-or-None values. Return these directly
  instead of copying/coercing every element. The native-inclusive phase
  reduction above is distinct from the 95.5% conversion-only microbenchmark.
- **Stream label construction:** without override dictionaries, use
  `np.fromiter` with the exact original scalar subtraction. This avoids
  dictionary probes and per-row array assignment without a temporary list.
  The original loop remains for overrides, preserving directional precedence.
  Isolated peak committed memory was effectively unchanged.

## Rejected alternatives

- A list comprehension reduced label time by 49.6% in isolation but increased
  peak working set by **36.3%**. It was rejected despite passing the speed gate.
- Removing dictionary probes but retaining scalar array assignment was 40.5%
  faster; the retained streaming version was faster still with similar memory.
- A 256-row scoring tile also improved scoring, but did not deliver a reliable
  additional 10% improvement over 64 rows. The smaller tile was retained.
- An initial sparse-feature version retained temporary buffers through
  projection. Those lifetimes were shortened before final acceptance.

## Exactness evidence

No tolerance-based comparisons were used for acceptance:

- All candidate runs match baseline label and returned distance bytes for
  every one of the 8,762,170 pairs, including array shapes and dtypes.
- All three real-scoring runs match the exact cluster dictionary, including
  keys and list order, with 3,139 clusters. Dictionary SHA-256:
  `5aa0b47cf472b7e9ab531fb8bff764f2da74fe14df825e1f05a259134075eb04`.
- A separate whole-workload dense/sparse comparison checks the bits of every
  scored feature row; placeholders are never passed to a classifier.
- Real-model scorer comparisons check all 50,000 rows per model, both dtypes,
  and both raw margins and probabilities against the original extension.
- Regression tests cover tile boundaries, cancellation-sensitive sums, all
  missing-value decision modes, noncontiguous arrays, NaNs and payloads,
  infinities, signed zero, thresholds, duplicate/reordered feature columns,
  chunk boundaries, disabled constraints, override precedence, both distance
  storage forms, and feature-sensitive/classifier-batch observations.
- The full baseline and candidate completion records report no Python source
  changes during their runs. Native extension builds were fixed per process.

The selected regression suite passed **130 pytest tests** using the final
release extension. The scorer's **13 Rust tests** also passed. Ruff checks,
format checks, and `ty check` for both changed Python modules passed.

These measurements cover the stated models and workloads, not every possible
dataset or machine. Giant-block subblocking/incremental attachment was not
benchmarked. Preservation of arithmetic order and classifier inputs is the
implementation invariant; the exact tests and real-data traces verify it.

## Reproduction and retained artifacts

All experiment scripts, original source copies, measurements, and built
extensions are under `scratch/bulk_fixes_20260904/` (git-ignored). The original
release extension remains in `scratch/bulk_profile_20260904/release_runtime/`.
The final wheel is in `final_wheels/`, extracted into `final_runtime/`;
the default workspace extension was not overwritten.

```powershell
uv run --no-sync maturin build --release --manifest-path s2and_rust/Cargo.toml --out scratch/bulk_fixes_20260904/final_wheels
uv run --no-sync python scratch/bulk_fixes_20260904/profile_compare.py --dataset aminer --limit 2500 --blocks 5 --repeat 1 --baseline --output scratch/bulk_fixes_20260904/full_baseline_rerun --runtime scratch/bulk_profile_20260904/release_runtime
uv run --no-sync python scratch/bulk_fixes_20260904/profile_compare.py --dataset aminer --limit 2500 --blocks 5 --repeat 3 --output scratch/bulk_fixes_20260904/full_candidate_rerun --runtime scratch/bulk_fixes_20260904/final_runtime
```

Output directories must be fresh. The harness prepends the chosen release
runtime before importing S2AND, since repository imports otherwise shadow it.
For feature isolation add `--skip-scoring`; `--dense-features` supplies the
dense control and `--check-features` performs the separate exact comparison.
Do not time independent variants concurrently.

Primary evidence: `acceptance.json`, `full_baseline/`, `full_candidate/`,
`features_dense_control/`, `features_final/`, `features_exact/`,
`tile64_bench.json`, `tile256_bench.json`, `scorer_*_memory.json`,
`labels_benchmark.json`, `labels_memory_*.json`, and `LABELS.md`.
Each full run includes input/model identities, source hashes, per-chunk
digests, phase logs, timings, memory high-water marks, and completion status.

Regression command (explicitly preload the rebuilt release):

```powershell
uv run --no-sync python -c "import sys; sys.path.insert(0,'scratch/bulk_fixes_20260904/final_runtime'); import s2and_rust; assert not s2and_rust.get_build_info()['debug_assertions']; import pytest; raise SystemExit(pytest.main(['-q','tests/test_bulk_sparse_features.py','tests/test_bulk_constraint_labels.py','tests/test_model_distance_matrix_orchestration.py','tests/test_rust_distance_matrix_parity.py','tests/test_rust_lightgbm_tiled_scoring.py','tests/test_rust_lightgbm_booster_parity.py','tests/test_rust_batch_chunking.py','tests/test_feature_port_parity.py']))"
```
