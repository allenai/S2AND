# Prediction state isolation

- [x] Pass owned Python seed overrides through incremental and subblocked prediction.
- [x] Share native feature data while isolating each call's seed overlay.
- [x] Move prediction telemetry from model attributes to request state.
- [x] Verify independent/interleaved calls and failure cleanup on tiny fixtures.
- [x] Run focused regression, lint, type, and native checks; document ownership.

Focused regression command:

```powershell
uv run --no-sync pytest -q tests/test_prediction_state.py tests/test_rust_seed_overlays.py tests/test_feature_port_cache.py tests/test_model_distance_matrix_orchestration.py
```

The five public prediction method signatures remain unchanged. Dataset feature
and presplit caches remain reusable; prediction seeds and diagnostics belong to
the current operation. See `docs/production_inference.md` for ownership details.

## Distance precision alignment

- [x] Use float64 for stored FastCluster matrices in both execution paths.
- [x] Verify threshold decisions, stored/streaming parity, and allocation budgets.
- [x] Re-run strict component parity and regression checks.

Verification: 1065 pytest tests passed, including nine new precision and memory
regressions. The bounded existing-script parity harness matched raw distances,
features, constraints, and clusters across eight Python/Rust comparisons.
Logs: `scratch/distance-precision-pytest.log` and
`scratch/distance-precision-parity.log`. Ruff and configured ty checks passed.

## Inference import boundary

- [x] Extract pure clustering metrics and scope evaluation plotting style.
- [x] Defer optimizer and Genie imports to their use sites.
- [x] Verify fresh-process imports, plotting, optimization, and full regressions.

Verification: `uv run --no-sync pytest -q` passed 1074 tests, including fresh
subprocess import checks, scoped plotting style, and actual Hyperopt fitting.
Ruff, formatting, and configured ty checks passed. Full-suite log:
`scratch/inference-import-boundary-pytest-final.log`.

## Pure seed-link assignment

- [x] Extract seed-link decisions with explicit inputs and owned results.
- [x] Preserve lazy name access, diagnostic logging, and residual orchestration.
- [x] Verify direct decision cases and full regressions.

Verification: 1090 pytest tests passed, including 16 direct decision cases and
existing Python/Arrow incremental completion tests. Ruff, formatting, and
configured ty checks passed. Log: `scratch/seed-link-assignment-pytest.log`.

## Deferred calibration setup

- [x] Keep default optimizer setup out of Clusterer construction and bundle loading.
- [x] Resolve the default space explicitly in Python and Arrow calibration.
- [x] Verify fresh-process bundle inference, calibration parity, and full local CI.

Full local CI passed: 1095 Python tests, 119 Rust tests, 87.26% coverage.
Native bundle scoring avoids Hyperopt; Python and Arrow calibration match the
explicit default search space exactly on fixed fixtures. Log:
`scratch/deferred-calibration-local-ci.log`.

## Authoritative feature metadata

- [x] Capture the existing 33-column contract before replacing metadata definitions.
- [x] Generate Rust named columns from one ordered specification with a CI freshness gate.
- [x] Derive Python selection metadata and replace native positional writes.
- [x] Verify all group subsets and native feature/prediction parity.

Verification: the original metadata matches all 4096 group subsets; 1111 pytest
tests and 119 native tests passed. Eight bounded component comparisons matched
features, constraints, distances, and clusters exactly. Schema freshness, lint,
formatting, and type checks pass for the changed feature files. Final combined
local CI passed after the concurrent constraint fixes: 1119 Python tests,
119 Rust tests, and 86.00% coverage.
Logs: `scratch/feature-schema-pytest.log`,
`scratch/feature-schema-component-parity.log`, and
`scratch/feature-schema-local-ci-final.log`, and
`scratch/completed-work-local-ci.log`.
