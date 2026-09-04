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
