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
- Reproducibility and paper-era notes: [root README](../README.md#reproducibility)

## Runtime and operations

- v1.3 release policy and runbook: [release.md](release.md)
- Runtime routing verification: [development.md](development.md#runtime-routing-verification)
- Environment variables: [environment.md](environment.md)
- Threading and parallelism: [threading.md](threading.md)
- Subblocking for large blocks: [subblocking.md](subblocking.md)
- Promoted incremental performance report:
  [production command reference](../scripts/production/README.md#promoted-incremental-performance-report)
- Public Arrow/name-count formats and direct Rust dataset schema:
  [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md)

## Deep dives and engineering notes

- Release notes: [release_notes.md](release_notes.md)
- September test audit and verification record: [test_audit_2026_09.md](test_audit_2026_09.md)
- Retained historical Rust profiling evidence:
  [rust/profiling/2026-05-28-promoted-incremental-arrow.md](rust/profiling/2026-05-28-promoted-incremental-arrow.md)
- Bulk Arrow optimization measurements:
  [initial optimizations](rust/profiling/2026-09-04-bulk-optimizations.md) and
  [specialization follow-up](rust/profiling/2026-09-04-bulk-followup.md)

## Script documentation

- Maintained, verification, and archived script catalog:
  [../scripts/README.md](../scripts/README.md)
- Production artifact command reference:
  [../scripts/production/README.md](../scripts/production/README.md)

## Scope notes

- Runtime and operations docs describe current behavior, knobs, and verification commands.
- Deep dives explain subsystem behavior or preserve historical context for active areas.
- Dated profiling snapshots belong under `rust/profiling/` and are immutable
  historical evidence. Their old dates
  and commands are preserved intentionally, are exempt from live-link checks,
  and are not current release gates.
- The vendored `s2and_rust/vendor/cld2/README.md` is third-party documentation.
