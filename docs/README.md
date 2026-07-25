# S2AND Docs

This directory holds both user-facing guides and internal engineering notes.

Documentation status: reviewed against the canonical-v2 branch and the v1.3
release plan on 2026-07-24. Current-behavior docs describe the unreleased
worktree; they do not imply that canonical artifacts or a v1.3 model have
already been published.

For release work, authority is intentionally split:

1. [1_3_release_todo.md](1_3_release_todo.md) owns execution order, blockers,
   approvals, immutable records, and release gates.
2. [normalization_migration_blocked.md](normalization_migration_blocked.md)
   owns the frozen canonical-name contract and acceptance thresholds.
3. [work_plan.md](work_plan.md) is only a remediation ledger. It is not an
   executable sequence.

## User guides

- Install and setup: [install.md](install.md)
- Data download and config: [data.md](data.md)
- Production inference: [production_inference.md](production_inference.md)
- Training and evaluation: [training.md](training.md)
- Development workflow: [development.md](development.md)
- Reproducibility and paper-era notes: [reproducibility.md](reproducibility.md)

## Runtime and operations

- Rust documentation index and scope: [rust/README.md](rust/README.md)
- Rust runtime contract and verification commands: [rust/runtime.md](rust/runtime.md)
- Environment variables: [environment.md](environment.md)
- Cache semantics and layout: [caching.md](caching.md)
- Threading and parallelism: [threading.md](threading.md)
- Subblocking for large blocks: [subblocking.md](subblocking.md)
- Rust promotion baselines and gate commands: [rust/baselines.md](rust/baselines.md)
- Rust artifact formats: [rust/artifact_formats.md](rust/artifact_formats.md)
- Direct Rust Arrow dataset schema: [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md)

## Deep dives and engineering notes

- Stage-wise memory telemetry notes: [stage_memory_estimates.md](stage_memory_estimates.md)
- Release notes: [release_notes.md](release_notes.md)
- Retained historical Rust profiling evidence:
  [rust/profiling/2026-05-28-promoted-incremental-arrow.md](rust/profiling/2026-05-28-promoted-incremental-arrow.md)

## Planning and migration docs

- v1.3 artifact regeneration, retraining, and release operator runbook:
  [1_3_release_todo.md](1_3_release_todo.md)
- Rust and platform backlog: [work_plan.md](work_plan.md)
- Normalization migration plan [blocked]: [normalization_migration_blocked.md](normalization_migration_blocked.md)

## Script documentation

- Maintained, verification, and archived script catalog:
  [../scripts/README.md](../scripts/README.md)
- Production artifact command reference:
  [../scripts/production/README.md](../scripts/production/README.md)

## Scope notes

- Runtime and operations docs describe current behavior, knobs, and verification commands.
- Deep dives explain subsystem behavior or preserve historical context for active areas.
- Planning and migration docs can describe proposed or blocked work that is not yet part of the runtime contract.
- Dated profiling snapshots are immutable historical evidence. Their old dates
  and commands are preserved intentionally and are not current release gates.
- `AGENTS.md` is repository working policy rather than product documentation;
  the vendored `s2and_rust/vendor/cld2/README.md` is third-party documentation.
