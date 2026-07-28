# Promoted incremental performance report

The release uses one Rust performance workload: promoted incremental linking
against the reviewed Arrow replay root and complete production model. Quality,
feature parity, packaging, and installed-runtime checks have their own focused
commands in the release runbook.

Run a bounded profile first:

```powershell
uv run --with psutil python scripts/verification/profile_promoted_incremental_arrow.py `
  --evaluation-plan path/to/run/evaluation_plan.json `
  --model-path path/to/production_model_vX.Y `
  --run-binding path/to/run/run_binding.json `
  --require-rust-release `
  --write-json path/to/run/reports/performance_evaluation_report.json
```

The report path must be fresh. The frozen evaluation plan supplies the Arrow
root and exact workload. Workloads above the built-in query or seed limits
also require `--full-run`; use it only for a reviewed release workload.

The command records the exact workload, Rust extension identity, per-run
timing and process-tree RSS, and summary statistics. Release evaluation
consumes:

- the prepared run's `run_binding_sha256`;
- the exact reviewed `workload`;
- `summary.predict_seconds.p50`; and
- `summary.peak_rss_gb.max`.

Keep ordinary command logs beside the report. Historical measurements remain
in [profiling/](profiling/); they are evidence snapshots, not active commands.
