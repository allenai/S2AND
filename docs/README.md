# S2AND Docs

This folder contains internal design + operational notes for working on S2AND.

## Start here (most used)

- Current work plan (next steps): `docs/work_plan.md`
- Rust runtime contract + verification commands: `docs/rust/runtime.md`
- Rust benchmark baselines + promotion rules: `docs/rust/baselines.md`
- Rust optimization frontier / backlog: `docs/work_plan.md` (Backlog section)
- Stage-wise memory telemetry + prediction model: `docs/stage_memory_estimates.md`
- Subclustering (subblocking) for large blocks: `docs/subclustering.md`
- Cluster-summary retrieval experiment plan (production incremental routing): `docs/cluster_retrieval_experiment.md`
- Training paper preprocessing (Rust training-mode bypass): `docs/rust/runtime.md` (Training-mode deferred paper preprocessing section)
- Artifact divergence map + format migration (Bundle 5): `docs/rust/artifact_divergence.md`
- Threading and parallelism (incl. preprocessing defaults): `docs/threading.md`
- Environment variables (centralized reference): `docs/environment.md`

## Keep separate (not currently executing)

- Normalization unification migration plan [BLOCKED]: `docs/normalization_migration_blocked.md`

## Where things go

- New profiling refreshes: add dated snapshots under `docs/rust/profiling/` (indexed in `docs/rust/baselines.md`).
