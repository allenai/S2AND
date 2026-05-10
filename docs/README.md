# S2AND Docs

This directory holds both user-facing guides and internal engineering notes.

## Start here

- Install and setup: [install.md](install.md)
- Production inference: [production_inference.md](production_inference.md)
- Training and evaluation: [training.md](training.md)
- Development workflow: [development.md](development.md)
- Rust runtime contract and verification commands: [rust/runtime.md](rust/runtime.md)
- Rust and platform backlog: [work_plan.md](work_plan.md)

## User guides

- Install and setup: [install.md](install.md)
- Data download and config: [data.md](data.md)
- Production inference: [production_inference.md](production_inference.md)
- Training and evaluation: [training.md](training.md)
- Development workflow: [development.md](development.md)
- Reproducibility and paper-era notes: [reproducibility.md](reproducibility.md)

## Runtime and operations

- Rust runtime contract and verification commands: [rust/runtime.md](rust/runtime.md)
- Environment variables: [environment.md](environment.md)
- Cache semantics and layout: [caching.md](caching.md)
- Threading and parallelism: [threading.md](threading.md)
- Subblocking for large blocks: [subclustering.md](subclustering.md)
- Rust promotion baselines and gate commands: [rust/baselines.md](rust/baselines.md)
- Historical profiling snapshots: [rust/profiling/README.md](rust/profiling/README.md)

## Deep dives and engineering notes

- Stage-wise memory telemetry notes: [stage_memory_estimates.md](stage_memory_estimates.md)

## Planning and migration docs

- Rust and platform backlog: [work_plan.md](work_plan.md)
- Rust artifact divergence and migration plan: [rust/artifact_divergence.md](rust/artifact_divergence.md)
- Normalization migration plan [blocked]: [normalization_migration_blocked.md](normalization_migration_blocked.md)

## Scope notes

- Runtime and operations docs describe current behavior, knobs, and verification commands.
- Deep dives explain subsystem behavior or preserve historical context for active areas.
- Planning and migration docs can describe proposed or blocked work that is not yet part of the runtime contract.
