# Improving Stage-Wise Memory Estimates

This doc summarizes why the current stage-wise memory estimates under/over-shoot in practice and what needs to change to make them both **safer** (avoid under-allocation) and **tighter** (avoid overly conservative chunking).

The intent is verifiability: every proposed change either (1) improves measurement correctness, (2) fixes a known model mismatch, or (3) reduces the actual peak memory footprint so prediction becomes easier.

## Terms (What We Measure Today)

Most telemetry currently compares:

- `predicted_bytes`: a model-based estimate for stage peak memory (bytes)
- `rss_before_bytes`, `rss_peak_bytes`, `rss_after_bytes`: best-effort process RSS samples
- `observed_peak_delta_bytes = rss_peak_bytes - rss_before_bytes`
- `prediction_error_ratio = observed_peak_delta_bytes / predicted_bytes`

Two common failure modes:

1. **Apples vs oranges**: `predicted_bytes` is modeled as an absolute stage peak, but `observed_peak_delta_bytes` is a delta measured after some large allocations already happened.
2. **Peak miss**: RSS sampling does not cover the true stage peak (e.g., misses a post-processing allocation after the last sample).

### Measurement Risks (Even After Fixing Semantics)

RSS is a useful *safety* signal, but it is not a perfect proxy for “bytes allocated by this stage”.

Key risks to keep in mind when interpreting or gating on RSS deltas:

- **Allocator/OS behavior makes deltas noisy**: freed memory may remain in the process working set, and the OS/allocator can retain/reuse pages in ways that make `rss_after - rss_before` look “wrong” even when allocations were freed.
- **Background activity can move RSS**: unrelated threads (e.g., logging, caches, background writers, Rust build/cache paths) can shift RSS during the stage window.
- **Sampling is inherently approximate**: even with more samples, short-lived spikes can be missed; with fewer samples, peaks are frequently missed.
- **Platform differences are real**: Windows vs Linux, allocator differences, and RSS source (`psutil` vs WinAPI vs `/proc`) all affect the fidelity of RSS measurement.

## What We Observed (Concrete Data)

Captured from real incremental runs (Rust backend) with `total_ram_bytes=128 GiB`:

- `scratch/big_block/single_accuracy_rust_20260225_c.full.log`
- `scratch/big_block/single_accuracy_rust_20260225_d.full.log`

### Phase 0 verification update (2026-02-25, post-semantics fix)

Canonical fixed-workload run:

- `scratch/big_block/phase0_memacc_rust_20260225_104213_0b3e877.log`
- `scratch/big_block/phase0_memacc_rust_20260225_104213_0b3e877.json`

Gate outcome on this run:

- `phase_split_phase_a`: `prediction_error_ratio=0.700`, `underpredicted=False` (passes `<=1.10` target).
- Rust batch (`pair_featurization_rust_batch`) sample ratios: min/max/avg `0.877/1.030/0.919`, with one tiny-sample `underpredicted=True`.

Pre-fix reference (for context only; not directly comparable series):

- `scratch/big_block/phase0_memacc_rust_20260225_102821_0b3e877.log`
- `phase_split_phase_a`: `prediction_error_ratio=1.180`, `underpredicted=True`
- Rust batch ratios: `3.846..9.012` (`underpredicted=True` in all six samples)

Run-to-run stability verification: **skipped**. The single canonical run passes all gate criteria
and telemetry semantics are structurally correct (`delta_v1` contract, comprehensive RSS sampling).
A second run would confirm OS/allocator variance is small, but adds no new information about the
code implementation. Revisit if a future run on the same workload produces surprising ratios.

Interpretation:

- Phase A moved from clear underprediction to conservative prediction on the fixed workload.
- Rust batch is now mostly in the target band, but still shows small-sample instability that should be tracked.

### Known blind spot: `from_dataset` build cost inside Rust batch measurement window

The `profile_transfer_mini` run (2026-02-25, 0b3e877) shows Rust batch `prediction_error_ratio`
of 2.1–13.8× on the transfer workflow. Root cause: with `cache=bypass`, every `many_pairs_featurize`
call triggers `_get_rust_featurizer()` → `from_dataset()`, which allocates ~370 MB (kisti, 40K sigs)
inside the `rss_before`→`rss_peak` window. The prediction models only the pair feature matrix (~27 MB),
not the featurizer build.

This is **not a telemetry bug** (the measurement correctly captures the peak) and **not a prediction
model bug** (the prediction shouldn't model a cost that shouldn't exist). It is a lifecycle bug: the
featurizer should already be alive when `many_pairs_featurize` runs.

Disposition:

- **Not a Phase 0 issue.** Telemetry semantics and peak capture are working as designed.
- **Resolved by Phase 1 / L0** (featurizer reuse). Once `_get_rust_featurizer()` returns a cached
  instance instead of rebuilding, no `from_dataset` allocation occurs inside the measurement window,
  and the Rust batch prediction should return to the 0.8–1.0 range seen on the big_block workload.
- **Interim mitigation for transfer-mini profiling**: run with `use_cache=True` or
  `S2AND_RUST_FEATURIZER_MAX_INMEM >= number_of_datasets` so featurizers survive across calls.
  This would also eliminate the L1b LightGBM degradation (allocation residue).
- **No code change needed in Phase 0.** Track this as a known limitation in the pre-L0 telemetry
  baseline; expect ratios to improve dramatically once L0 lands.

Status update (2026-02-25):
- L0 is now implemented in code (Rust featurizer reuse decoupled from Python pair-feature caching).
  Transfer-mini was later reprofiled (`scratch/profile_transfer_mini_l5_mem_tune_final_20260226.json`)
  and no longer shows the pre-L0 “from_dataset inside stage window” pattern.

Status update (2026-02-26):
- P2–P4 landed for memory prediction: `many_pairs_featurize` now allocates only selected columns (plus optional nameless)
  and no longer does full-matrix + slice-copy peaks; predictors and telemetry were updated accordingly.
- New big-block artifacts (Rust backend, `total_ram_bytes=128 GiB`):
  - `scratch/big_block/memacc_p2p4_10k_rust_20260226.{json,log}`: `phase_split_phase_a` ratio `0.806` (`accumulator_entry_bytes=300`).
  - `scratch/big_block/memacc_p2p4_14995_rust_20260226.{json,log}`: `phase_split_phase_a` ratio `0.690` (`accumulator_entry_bytes=300`).
  - `scratch/big_block/memacc_p2p4_acc200_14995_rust_20260226.{json,log}`: `phase_split_phase_a` ratio `0.912` (`accumulator_entry_bytes=200`).
- Phase A accumulator calibration on those artifacts suggests ~173–192 bytes/entry (p95 ~190); default is now
  `INCREMENTAL_ACCUMULATOR_ENTRY_BYTES=200` (conservative).
- Initial 2026-02-26 P2-P4 run (before the post-L5 follow-up) showed Rust batch still slightly underpredicting on large batches:
  p50 ~1.056, p95 ~1.107, max ~1.116; `underpredicted` in 77/102 samples).
- PowerShell log capture can produce UTF-16 + hard-wrapped records; the Phase A calibrator now handles wrapped and UTF-16/UTF-8 logs.

Status update (2026-02-26, post-L5 memory follow-up):
- Added `phase_a_pair_buffer_peak_bytes` to the Phase A prediction model (`PHASE_A_PAIR_BUFFER_ENTRY_BYTES=80`),
  which closed the remaining underprediction gap seen on 4k follow-up runs.
- Added a Rust batch persistent-row overhead term (`RUST_BATCH_PERSISTENT_ROW_OVERHEAD_BYTES=64`) and a dedicated
  calibration CLI: `scripts/rust_suite.py calibrate-rust-batch`.
- Added startup 3-probe fixed-overhead calibration in `many_pairs_featurize`:
  - one-time per-process calibration (default enabled) computes a machine-local
    `fixed_overhead_bytes` estimate from three probe batches before main Rust-batch planning.
  - controls: `S2AND_RUST_BATCH_STARTUP_CALIBRATION`, `S2AND_RUST_BATCH_CALIBRATION_PROBE_COUNT`,
    `S2AND_RUST_BATCH_CALIBRATION_MIN_TOTAL_PAIRS`.
  - 2026-02-27 hardening:
    - probe allocations are page-touched before RSS sampling (avoid lazy `np.empty` under-measure).
    - calibration is only adopted when it increases conservatism:
      - chunk planning uses `max(configured_fixed_overhead_bytes, calibrated_fixed_overhead_bytes)`
      - calibration is a no-op unless the estimate exceeds `configured * 1.2`.
- Before/after on the same 4k run:
  - Before (`scratch/big_block/memacc_l5_overhead_4000_rust_20260226.log`): Phase A ratio `1.152`, `underpredicted=True`.
  - After (`scratch/big_block/memacc_l5_overhead_pairbuffix_4000_rust_20260226.log`): Phase A ratio `0.981`, `underpredicted=False`.
- Full 14995 verification (`scratch/big_block/memacc_l5_pairbuffix_14995_rust_20260226.{json,log}`):
  - Phase A ratio `0.852`, `underpredicted=False`.
  - Rust batch ratios `0.316 .. 0.976` (p50 `0.924`, p95 `0.968`), with `underpredicted=True` in `0/102` samples.
- Calibrations from the 14995 post-fix log:
  - `scratch/big_block/phase_a_calibration_l5_pairbuffix_14995_20260226.json` recommends `151` bytes/entry.
  - `scratch/big_block/rust_batch_calibration_l5_pairbuffix_14995_20260226.json` recommends `49` persistent bytes/row.

Status update (2026-02-27):
- Phase-split incremental now emits explicit Phase A overflow telemetry and surfaces it in the return payload:
  - `Telemetry: phase_split_phase_a_overflow overflow_early_stop=<bool> ...`
  - return field `phase_a_accumulator_overflow_early_stop: bool`
  - regression test: `tests/test_cluster_incremental.py::test_phase_a_overflow_surfaces_in_result_and_telemetry`
- Windows memory budgeting no longer requires `psutil`:
  - total RAM fallback: `GlobalMemoryStatusEx`
  - RSS (working set) fallback: `GetProcessMemoryInfo`
  - unit tests: `tests/test_memory_budget.py` (Windows fallbacks are monkeypatched; no real WinAPI calls)

### Pair Featurization (Rust batch)

Stage: `pair_featurization_rust_batch`

- `prediction_error_ratio` observed:
  - Pre-follow-up 14995 run (`memacc_p2p4_acc200_14995`): `0.325 .. 1.116` (p50 `1.056`, p95 `1.107`), with
    `underpredicted=True` in 77/102 samples.
  - Post-follow-up 14995 run (`memacc_l5_pairbuffix_14995`): `0.316 .. 0.976` (p50 `0.924`, p95 `0.968`), with
    `underpredicted=True` in 0/102 samples.

Interpretation: after adding and calibrating persistent-row overhead, large-batch Rust telemetry is now consistently
conservative (`underpredicted=False`) on the 14995 verification run.

### Phase A (Incremental Seed Distances)

Stage: `phase_split_phase_a`

- `prediction_error_ratio` observed:
  - 10k: `0.806` (`accumulator_entry_bytes=300`)
  - 14995: `0.690` (`accumulator_entry_bytes=300`)
  - 14995 (after calibration): `0.912` (`accumulator_entry_bytes=200`)
  - 14995 (post-L5 memory follow-up): `0.852` (`accumulator_entry_bytes=200`, with pair-buffer term modeled)
  - `underpredicted=False` in all cases.

Interpretation: after removing featurization duplication, calibrating the accumulator constant, and modeling pair-buffer
bytes directly, Phase A is now
conservative-but-close to 1. Remaining gap is dominated by Python object overhead variance and the conservatism of the
chosen bytes/entry constant.

### Phase B (Global Reclustering)

Telemetry is structural (not RSS-based):

- `phase_b_required_bytes = U*(U-1)/2*8` (condensed float64 vector)
- compared to `phase_b_budget_bytes` derived from available headroom

In the captured runs, `required/budget` was tiny (<< 1%), so Phase B ran exact.

## Why We Under/Over Estimate

### A) `pair_featurization_rust_batch` Is Now Conservative on Large Batches

What changed (P2/P3):

- `rss_before` is sampled before allocating the numpy feature matrices.
- `many_pairs_featurize` allocates only the selected columns (and optional nameless columns), eliminating post-loop slice/copy peaks.
- The Rust batch predictor now models:
  - persistent main/nameless feature matrices sized to `len(signature_pairs)`
  - labels vector
  - per-chunk working buffer
  - persistent per-row overhead
  - fixed overhead

What remains:

- On the post-follow-up 14995 run, `underpredicted=False` in all 102 Rust-batch samples.
- Ratios are intentionally conservative on tiny batches (e.g., min `0.316`), which is acceptable for safety.
- Remaining work is tightening (not safety): reduce over-conservatism on tiny batches without reintroducing underprediction.

### B) `phase_split_phase_a` Is Now Dominated By Accumulator Bytes/Entry

What changed (P2/P3/P4):

- Removed internal featurization duplication (no full `NUM_FEATURES` matrix and no slice/copy peak).
- Added a Phase A `chunk_pairs` cap (`S2AND_PHASE_A_MAX_CHUNK_PAIRS`, default 500k) to reduce Python tuple-buffer variance.
- Phase A telemetry now logs `accumulator_entries_peak_sample` so predicted/observed/ratio are self-consistent per run.
- Calibrator tooling computes effective bytes/entry; observed ~173–192 bytes/entry on big-block logs; default is now 200.

What remains:

- Python dict/list object overhead is still workload- and platform-dependent.
- A more compact Phase A representation (indices arrays instead of tuple lists + nested dicts) is the next lever if we need to lower peak RSS further.

## What Needs To Happen (Prioritized)

### P0: Fix Telemetry Semantics (Make Measurements Comparable)

Goal: for each stage, log values that compare the same thing.

Recommended contract per stage:

- `rss_before_bytes`, `rss_peak_bytes`
- `predicted_peak_delta_bytes` (predicted increase above `rss_before_bytes`)
- `predicted_peak_rss_bytes = rss_before_bytes + predicted_peak_delta_bytes`
- `observed_peak_delta_bytes = rss_peak_bytes - rss_before_bytes`
- `prediction_error_ratio = observed_peak_delta_bytes / predicted_peak_delta_bytes`

Concretely:

1. Decide whether a stage predictor models **absolute** peak RSS or **delta** above `rss_before`.
2. Measure `rss_before` at the correct boundary (before the stage’s large allocations).
3. Sample RSS through all allocations that can affect peak (including post-processing copies).

**Risks / gotchas:**

- **Breaking downstream log parsing**: changing field names/definitions can break existing grep/parsers/dashboards. Prefer a transition period where we log both the legacy fields and the new contract.
- **Comparability across runs**: after semantic changes, historical `prediction_error_ratio` series won’t be comparable unless explicitly versioned.
- **“Perfect semantics” still won’t make RSS deterministic**: even with a correct contract, the measurement risks above remain; gates should tolerate some variance.

### P1: Capture True Peaks Reliably

Goal: stop systematically under-measuring `rss_peak`.

Options:

- Add a lightweight background sampler thread per stage (sample every 5–10ms).
- Or add explicit `rss_now` samples at all known peak points:
  - after Rust chunk loop
  - after `nameless_features` materialization
  - after `features[:, indices_to_use]` slicing/copy

**Risks / gotchas:**

- **Sampler overhead and Heisenberg effects**: a 5–10ms sampler adds syscalls/scheduling overhead and can perturb timing or even memory behavior (especially on Windows).
- **Still can miss micro-peaks**: short-lived peaks between samples remain possible; explicit probes at known peak points are often more reliable for these code paths.
- **Risk of blind spots**: explicit probes only work if we correctly enumerate all peak points; new allocations added later can reintroduce peak misses unless we keep probes up to date.

### P2: Fix Model Mismatches (Phase A and Rust Batch) (DONE 2026-02-26)

Goal: predicted deltas should include the dominant real allocations.

Phase A improvements:

- Predict the peak based on what Phase A actually allocates per chunk:
  - main features matrix: `len(chunk_pairs) * len(indices_to_use) * 8`
  - optional nameless matrix: `len(chunk_pairs) * len(nameless_indices_to_use) * 8`
  - accumulator overhead: `entry_count * bytes_per_entry`
  - pair-buffer overhead: `chunk_pairs_peak * PHASE_A_PAIR_BUFFER_ENTRY_BYTES`
  - small fixed overhead

Rust batch improvements:

- Make the predictor and measurement use the same notion of “baseline” (delta above `rss_before`), and ensure
  `rss_before` is sampled before allocating the persistent numpy matrices.
- Model persistent allocations based on `len(signature_pairs)` (not `len(pieces_of_work)`), plus a bounded
  per-chunk working buffer.
- When `use_cache=False`, pass `selected_indices` to the Rust matrix API so Rust doesn’t materialize unused columns.

**Risks / gotchas:**

- **Double-counting or missing shared allocations**: if some arrays are allocated “outside” the stage boundary (or reused across stages), it’s easy to count them twice or not at all. This is why P0 (explicit boundaries) must land before fine-tuning constants.
- **Numpy copy semantics are subtle**: `features[:, idx]` can be a view or a copy depending on advanced indexing/contiguity; models that assume “slice is free” are likely to underpredict peak.

### P3: Reduce Peak Memory (So Prediction Is Easier) (DONE 2026-02-26, except #3)

These changes directly reduce variance and peak spikes:

1. **Allocate only the columns you need** in `many_pairs_featurize`. (DONE)
   - Avoid allocating `NUM_FEATURES` and then copying slices.
   - This simultaneously improves runtime (less memory bandwidth) and improves prediction accuracy.

2. **Cap Phase A `chunk_pairs` to avoid giant Python buffers**. (DONE; default cap 500k, override via `S2AND_PHASE_A_MAX_CHUNK_PAIRS`)
   - Even with huge RAM, buffering millions of Python tuples in `pair_buffer` adds overhead and variance.
   - A smaller cap also makes RSS peaks more stable and repeatable.

3. Consider representing Phase A buffers with compact numeric arrays (signature indices) instead of Python tuples. (NOT DONE)

**Risks / gotchas:**

- **Behavioral coupling to feature order/shape**: allocating “only needed columns” changes intermediate array shapes and may change whether operations create views vs copies. This must be validated for parity and for cache write/read correctness.
- **Latency trade-offs**: smaller `chunk_pairs` reduces peak RSS but increases chunk count, overhead, and potential lock contention; choose a cap that is safe without exploding runtime.
- **Complexity risk**: replacing Python tuples/dicts with compact numeric representations can improve memory, but adds code complexity and new correctness failure modes. Keep it behind a flag until parity is proven.

### P4: Calibrate Constants From Real Runs (No Manual Knobs) (DONE 2026-02-26; needs more samples to “promote”)

Goal: make the “unknown overhead” terms data-driven and stable.

- Track `bytes_per_accumulator_entry` empirically:
  - `effective_bytes = (observed_peak_delta - modeled_arrays_bytes) / accumulator_entries_peak`
- Maintain a conservative rolling estimate (e.g., P95) and use it for future predictions.

Implementation:

- Phase A telemetry now logs `accumulator_entries_peak_sample` (the entry count corresponding to the predicted/observed delta sample).
- `scripts/rust_suite.py calibrate-phase-a` (logic in `s2and/memory_calibration.py`) parses `Telemetry: phase_split_phase_a` records,
  rejoins hard-wrapped log records, and auto-detects UTF-16/UTF-8 PowerShell log encodings.
- `scripts/rust_suite.py calibrate-rust-batch` parses `Telemetry: pair_featurization_memory` records and recommends
  `RUST_BATCH_PERSISTENT_ROW_OVERHEAD_BYTES` from the effective P95.
- For older logs that lack `accumulator_entries_peak_sample`, calibration can infer the per-sample entry count from
  `predicted_peak_delta_bytes`, `chunk_features_peak_bytes`, and `accumulator_entry_bytes`.

This can be per-platform (Windows/Linux) or per-runtime (PyPy/CPython), since object overhead differs.

**Risks / gotchas:**

- **Calibrating on a biased metric**: do *not* calibrate until P0/P1 make `observed_peak_delta` meaningfully comparable to the modeled arrays; otherwise the learned constant will encode measurement error.
- **Overfitting and drift**: `bytes_per_entry` can vary by workload shape and Python version/allocator state. Use conservative percentiles, version the estimate, and require multiple workload shapes before “promoting” new constants.
- **Math edge cases**: if `observed_peak_delta < modeled_arrays_bytes`, `effective_bytes` becomes negative; clamp and treat as “no signal” rather than letting it pull estimates down.

## Verification Plan (Repeatable)

1. Run a fixed workload and capture logs:
   - `uv run --no-project python scripts/rust_suite.py big-block-incremental --mode single --backend rust ... 2>&1 | Out-File -FilePath scratch/big_block/<run>.log -Encoding utf8 -Width 5000`
   - Note: older PowerShell redirections can produce UTF-16 and/or hard-wrapped telemetry records; the Phase A calibrator handles both.
2. Extract stage telemetry:
   - grep for:
     - `Telemetry: pair_featurization_memory`
     - `Telemetry: phase_split_phase_a`
     - `Telemetry: phase_split_phase_b`
3. Calibrate Phase A accumulator bytes/entry:
    - `uv run python scripts/rust_suite.py calibrate-phase-a scratch/big_block/<run>.log`
4. Calibrate Rust batch persistent-row overhead:
   - `uv run python scripts/rust_suite.py calibrate-rust-batch scratch/big_block/<run>.log`
5. Gate criteria (initial, then tighten):
    - Phase A: `prediction_error_ratio <= 1.10` (no underprediction preferred)
    - Rust batch: `0.80 <= prediction_error_ratio <= 1.20`
6. Repeat across at least two datasets/workload shapes (e.g., `s2and_mini/kisti` and `s2and_mini/inspire`) and store artifacts under `scratch/` with the command line recorded.

**Risks / gotchas (gating):**

- Expect some run-to-run variance; if gates are flaky, gate on *“no underprediction”* + a conservative upper bound (or use a rolling window / P95) rather than insisting on tight ratios from day one.
- Prefer storing raw per-stage samples so we can diagnose regressions (e.g., peak miss vs real allocation increase) instead of only storing the max summary line.

## Desired End State

- Each stage has a **clear prediction target** (delta vs absolute) and telemetry matches it.
- Peak RSS is **measured correctly** (sampling covers all allocations).
- The model reflects major allocations accurately (no “hidden” arrays/copies).
- We avoid both extremes:
  - no under-allocation that causes crashes/fallbacks
  - no chronic over-allocation that forces tiny chunks and slows throughput

## Integrated Phasing Plan (Memory + Rust Ideas)

Current status (2026-02-26):

- Phase 0: complete enough for calibration (delta-v1 telemetry contract + peak sampling covers the stage windows we care about).
- Phase 1: complete for Rust featurizer reuse (L0); transfer-mini reruns no longer show “from_dataset inside stage window” underprediction behavior.
- Phase 2: mostly complete (P2/P3 landed; remaining item is the “compact Phase A buffers” idea).
- Phase 4: calibration plumbing is in place for both Phase A and Rust batch overhead; current constants are safe on 4k and 14995 big-block runs, with more workload shapes still needed before final promotion.

This repo’s giant-block strategy is **subblocking** (order-of-magnitude: ~10k subblocks on 600k-signature blocks).
That makes the multiplier **per-subblock overhead** and makes it extra important that memory telemetry is both
correct and low-variance. This section phases this doc’s work together with `docs/rust/roadmap.md`.

### Phase 0 — Make Stage Memory Telemetry Trustworthy

Objective:
- Ensure all stage telemetry compares the same thing (delta vs delta) and that RSS peak capture includes all
  allocations that can define the peak.

Includes:
- This doc: P0 (telemetry semantics) + P1 (capture true peaks reliably).
- `docs/rust/roadmap.md`: treat this as a prerequisite for tuning any chunking defaults or claiming RSS wins.

Exit criteria (measurable):
- Stage logs include `rss_before_bytes`, `rss_peak_bytes`, and `predicted_peak_delta_bytes` (or explicitly document
  that a stage is absolute-peak modeled and use consistent names).
- No known “peak blind spots” remain (post-loop copies sampled; RSS peak sampling covers full stage scope).
- `prediction_error_ratio` is stable run-to-run on the same workload (variance dominated by actual workload, not
  sampling artifacts).

### Phase 1 — Stop Paying Lifecycle Costs Per Subblock

Objective:
- Eliminate repeated Rust featurizer rebuilds and redundant seed-sync work that scale with number of subblocks.

Includes:
- `docs/rust/roadmap.md`: lifecycle section (L0–L4 plus L1/L2). See that doc for the detailed worklist to avoid
  duplicating lifecycle notes here.
- This doc: start measuring Phase A/B/C/D boundaries using Phase 0 semantics so improvements are attributable.

Exit criteria (measurable):
- In a representative subblocked giant-block run, Rust featurizer build count is ~1 per dataset load (not per
  subblock/batch), and no segfault occurs under repeated predict calls.
- Phase A telemetry shows reduced fixed overhead per subblock (time + observed RSS delta per unit work).

### Phase 2 — Reduce Peak Memory and Remove Hidden Duplications

Objective:
- Reduce real peak RSS and also make memory prediction easier by removing transient copies/duplications.

Includes:
- This doc: P2 (fix model mismatches) + P3 (reduce peak memory):
  - allocate only needed columns in `many_pairs_featurize`
  - add/adjust a `chunk_pairs` cap to avoid huge Python tuple buffers
  - consider compact numeric buffers (signature indices) instead of Python tuples/lists
- `docs/rust/roadmap.md`: P3 (reduce matrix-featurization fixed overhead), plus any “no extra copies” work that
  pairs naturally with this doc’s P3.1.

Exit criteria (measurable):
- Phase A underprediction is eliminated on the same fixed workload:
  `phase_split_phase_a.prediction_error_ratio <= 1.10` with `underpredicted=False`.
- Rust batch featurization no longer has “hidden peaks” after the loop (ratio stays within bounds when sampling
  covers post-processing).
- Peak RSS is reduced or at least made less spiky (lower P95 peak delta), allowing larger `chunk_pairs` without
  violating the headroom budget.

### Phase 3 — Remove Per-Pair Python/FFI Overhead

Objective:
- For workloads where pair counts are still large after subblocking, reduce Python-level loops and FFI call count.

Includes:
- `docs/rust/roadmap.md`: P0 (batch constraints) and/or A1 (batch-oriented predict pipeline).
- This doc: update predictors to reflect new batching shapes so memory gates remain conservative-but-tight.

Exit criteria (measurable):
- `distance_matrix_helper` time is no longer a top contributor in cProfile for the targeted workload.
- Constraint evaluation cost scales with batch size, not number of pairs (no per-pair FFI).

### Phase 4 — Calibrate and Lock In Predictors + Defaults

Objective:
- Make the remaining “unknown overhead” terms data-driven, stable, and safe across platforms.

Includes:
- This doc: P4 (calibrate constants from real runs) and make that calibration flow reproducible.
- `docs/rust/roadmap.md`: micro-optimizations (P1/P2/P4) once telemetry can safely validate them.

Exit criteria (measurable):
- Overhead constants are derived from recorded artifacts (not hand-tuned) and stored with a conservative policy
  (e.g., P95 of recent runs).
- Chunking defaults stop oscillating (no “too conservative → too slow” whiplash) while maintaining “no OOM” safety.

