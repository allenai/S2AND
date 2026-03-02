# S2AND Docs

This folder contains internal design + operational notes for working on S2AND.

## Start here (most used)

- Current work plan (next steps): `docs/work_plan.md`
- Rust runtime contract + verification commands: `docs/rust/runtime.md`
- Rust benchmark baselines + promotion rules: `docs/rust/baselines.md`
- Rust optimization frontier / backlog: `docs/work_plan.md` (Backlog section)
- Stage-wise memory telemetry + prediction model: `docs/stage_memory_estimates.md`
- Subclustering (subblocking) for large blocks: `docs/subclustering.md`
- Training paper preprocessing (Rust training-mode bypass): `docs/rust/runtime.md` (Training-mode deferred paper preprocessing section)
- Artifact divergence map + format migration (Bundle 5): `docs/rust/artifact_divergence.md`
- Threading and parallelism (incl. preprocessing defaults): `docs/threading.md`

## Keep separate (not currently executing)

- Normalization unification migration plan (blocked on data readiness): `docs/normalization_migration.md`

## Where things go

- New profiling refreshes: add dated snapshots under `docs/rust/profiling/` (see `docs/rust/profiling/README.md`).
- Historical/forensics-only notes: `docs/archive/` (see `docs/archive/README.md`).
