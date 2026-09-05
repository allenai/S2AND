# S2AND Docs

This directory holds both user-facing guides and internal engineering notes.

Documentation status: reviewed against the simplified `1.0.0` worktree on
2026-09-04. Current-behavior docs do not imply that canonical artifacts or a
v1.3 model have already been published.

[release.md](release.md) is the sole v1.3 release policy and runbook.
[../scripts/production/README.md](../scripts/production/README.md) is its
command reference. Canonical-name rules live in [data.md](data.md).

## Repository entry points

- Project overview and public quick start: [../README.md](../README.md)
- Native `s2and-rust` package build and test guide:
  [../s2and_rust/README.md](../s2and_rust/README.md)

## User guides

- Install and setup: [install.md](install.md)
- Data download and config: [data.md](data.md)
- Production inference: [production_inference.md](production_inference.md)
- Training and evaluation: [training.md](training.md)
- Development workflow: [development.md](development.md)
- Reproducibility and paper-era notes: [reproducibility.md](reproducibility.md)

## Runtime and operations

- v1.3 release policy and runbook: [release.md](release.md)
- Rust documentation index and scope: [rust/README.md](rust/README.md)
- Rust runtime contract and verification commands: [rust/runtime.md](rust/runtime.md)
- Environment variables: [environment.md](environment.md)
- Threading and parallelism: [threading.md](threading.md)
- Subblocking for large blocks: [subblocking.md](subblocking.md)
- Promoted incremental performance report: [rust/baselines.md](rust/baselines.md)
- Public Arrow/name-count formats and direct Rust dataset schema:
  [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md)

## Deep dives and engineering notes

- Version-authority rationale and implementation record:
  [../version_simplification.md](../version_simplification.md)
- Release notes: [release_notes.md](release_notes.md)
- Retained historical Rust profiling evidence:
  [rust/profiling/2026-05-28-promoted-incremental-arrow.md](rust/profiling/2026-05-28-promoted-incremental-arrow.md)

## Script documentation

- Maintained, verification, and archived script catalog:
  [../scripts/README.md](../scripts/README.md)
- Production artifact command reference:
  [../scripts/production/README.md](../scripts/production/README.md)

## Scope notes

- Runtime and operations docs describe current behavior, knobs, and verification commands.
- Deep dives explain subsystem behavior or preserve historical context for active areas.
- Dated profiling snapshots are immutable historical evidence. Their old dates
  and commands are preserved intentionally, are exempt from live-link checks,
  and are not current release gates.
- `AGENTS.md` is repository working policy rather than product documentation;
  the vendored `s2and_rust/vendor/cld2/README.md` is third-party documentation.
