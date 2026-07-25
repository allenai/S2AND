# Improving Stage-Wise Memory Estimates

Status date: 2026-07-24

This doc explains what S2AND stage-wise memory predictors and telemetry
measure, why they can be wrong, and how to calibrate them.

## Terms (what we measure)

Most telemetry compares:
- `predicted_peak_delta_bytes`: model-based estimate for the stage's peak RSS increase above `rss_before_bytes`
- `rss_before_bytes`, `rss_peak_bytes`, `rss_after_bytes`: best-effort process RSS samples
- `observed_peak_delta_bytes = rss_peak_bytes - rss_before_bytes`
- `prediction_error_ratio = observed_peak_delta_bytes / predicted_peak_delta_bytes`

## Measurement caveats (RSS is not "bytes allocated by the stage")

- Allocator/OS behavior can retain memory after frees (RSS deltas can look wrong).
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
2. Sample `rss_before` before the stage's large allocations start.
3. Ensure RSS sampling covers all allocations that can define the peak (including post-processing copies).

## Status

No dedicated memory-estimate backlog is active right now. These predictors are
treated as best-effort; predictor remediation belongs in
[work_plan.md](work_plan.md). The v1.3 release's measured runtime/RSS protocol
and acceptance gate are in
[1_3_release_todo.md](1_3_release_todo.md#stage-8-release-candidate-evaluation).

Regression coverage:
- `tests/test_memory_budget.py`
- `tests/test_memory_calibration.py`
- `tests/test_memory_telemetry_summary.py`
- `tests/test_rust_batch_chunking.py::test_rust_batch_prediction_matches_observed_real_workload`

## Where to look / how to calibrate

- Calibration logic: `s2and/memory_calibration.py`
- Structured memory telemetry is written as JSONL when
  `--memory-telemetry-jsonl <path>` is passed to `scripts/rust_suite.py` before
  the command name. Human-readable logs can be captured separately with
  `--log-file <path>`.
- CLIs:
  - `uv run python scripts/rust_suite.py calibrate-phase-a scratch/run_memory_telemetry.jsonl`
  - `uv run python scripts/rust_suite.py calibrate-rust-batch scratch/run_memory_telemetry.jsonl`
  - `uv run python scripts/rust_suite.py summarize-memory-telemetry scratch/run_memory_telemetry.jsonl`

## Verification (repeatable)

1. Run a fixed workload with `--memory-telemetry-jsonl` (use a reproducible command line and record `n_jobs` / thread env vars).
2. Extract stage telemetry records and confirm:
   - contract fields are present
   - peaks are not missed (probes/sampling covers the real peak window)
3. Calibrate and check safety:
   - underprediction is eliminated (`underpredicted=false` where surfaced)
   - ratios are stable enough to gate (do not over-tighten on noisy RSS deltas)
