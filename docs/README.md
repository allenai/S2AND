# S2AND Docs

This folder contains internal design + operational notes for working on S2AND.

## Start here (most used)

- Current work plan (bundles): `docs/work_plan.md`
- Rust runtime contract + verification commands: `docs/rust/runtime.md`
- Rust benchmark baselines + promotion rules: `docs/rust/baselines.md`
- Rust optimization frontier / worklist: `docs/rust/roadmap.md`
- Stage-wise memory telemetry + prediction model: `docs/stage_memory_estimates.md`
- Subclustering (subblocking) for large blocks: `docs/subclustering.md`
- Training paper preprocessing plan (Bundle 1): `docs/rust/training_preprocessing_plan.md`
- Artifact divergence map + format migration (Bundle 5): `docs/rust/artifact_divergence.md`
- Preprocessing parallelism analysis: `docs/preprocessing_parallelism.md`

## Keep separate (not currently executing)

- Normalization unification migration plan (Bundle 6): `docs/normalization_migration.md`

## Where things go

- New profiling refreshes: add dated snapshots under `docs/rust/profiling/` (see `docs/rust/profiling/README.md`).
- Historical/forensics-only notes: `docs/archive/` (see `docs/archive/README.md`).
