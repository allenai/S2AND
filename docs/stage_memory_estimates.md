# Improving Stage-Wise Memory Estimates

Status date: 2026-03-02

This doc explains what S2AND stage-wise memory predictors/telemetry measure, why they can be wrong, and what work remains to make them both **safe** (avoid under-allocation) and **tight** (avoid overly conservative chunking).

## Terms (what we measure)

Most telemetry compares:
- `predicted_peak_delta_bytes`: model-based estimate for the stage’s peak RSS increase above `rss_before_bytes`
- `rss_before_bytes`, `rss_peak_bytes`, `rss_after_bytes`: best-effort process RSS samples
- `observed_peak_delta_bytes = rss_peak_bytes - rss_before_bytes`
- `prediction_error_ratio = observed_peak_delta_bytes / predicted_peak_delta_bytes`

## Measurement caveats (RSS is not “bytes allocated by the stage”)

- Allocator/OS behavior can retain memory after frees (RSS deltas can look “wrong”).
- Background activity can shift RSS during a stage window.
- Sampling can miss short-lived peaks.
- Fidelity differs by platform and RSS source (Windows vs Linux, psutil vs WinAPI vs `/proc`).

## Telemetry contract (delta-based)

Recommended per-stage contract:
- `rss_before_bytes`, `rss_peak_bytes`
- `predicted_peak_delta_bytes`
- `predicted_peak_rss_bytes = rss_before_bytes + predicted_peak_delta_bytes`
- `observed_peak_delta_bytes = rss_peak_bytes - rss_before_bytes`
- `prediction_error_ratio = observed_peak_delta_bytes / predicted_peak_delta_bytes`

Boundary rules:
1. Decide whether the predictor models **delta above `rss_before`** (recommended) vs an **absolute** peak.
2. Sample `rss_before` before the stage’s large allocations start.
3. Ensure RSS sampling covers all allocations that can define the peak (including post-processing copies).

## What remains (next work)

No additional work is planned right now. These predictors are treated as best-effort; we prioritize stable telemetry contracts and regression tests over further tightening.

Regression coverage:
- `tests/test_cluster_incremental.py::test_phase_a_memory_prediction_logged_and_bounded`
- `tests/test_rust_batch_chunking.py::test_rust_batch_prediction_matches_observed_real_workload`
- `tests/test_memory_calibration.py`

## Where to look / how to calibrate

- Calibration logic: `s2and/memory_calibration.py`
- CLIs:
  - `uv run python scripts/rust_suite.py calibrate-phase-a`
  - `uv run python scripts/rust_suite.py calibrate-rust-batch`

## Verification (repeatable)

1. Run a fixed workload and capture logs (use a reproducible command line and record `n_jobs` / thread env vars).
2. Extract stage telemetry records and confirm:
   - contract fields are present
   - peaks are not missed (probes/sampling covers the real peak window)
3. Calibrate and check safety:
   - underprediction is eliminated (`underpredicted=false` where surfaced)
   - ratios are stable enough to gate (don’t over-tighten on noisy RSS deltas)
