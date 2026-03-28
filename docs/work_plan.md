# Work Plan (Rust/Platform Backlog)

Status date: 2026-03-27

This doc is not the active giant-block experiment plan. Active `h_wang` work lives in
`TODO.md` and `TASK.md`.

This file only tracks Rust/platform items that are still open and worth revisiting later.
It intentionally excludes items that are already done.

Start here:
- Threading and preprocessing defaults: `docs/threading.md`
- Rust runtime contract and verification commands: `docs/rust/runtime.md`
- Rust gate commands: `docs/rust/baselines.md`
- Artifact divergence and migration plan: `docs/rust/artifact_divergence.md`
- Environment variables: `docs/environment.md`
- Stage-wise memory telemetry: `docs/stage_memory_estimates.md`

## Partial

### Artifact format unification (Ask-first)

Status:
- Still open. Python and Rust artifact handling have converged somewhat, but they are not unified.

What remains:
- `name_counts` still diverges between Python pickle handling and Rust JSON ingest.
- `specter` still depends on pickle-era artifact assumptions in parts of the stack.
- `name_tuples` still has more runtime variation than we likely want.

When to consider it:
- After the current giant-block chooser work, or sooner if artifact drift blocks parity or debugging.

Verification bar:
- Dual-read loaders first.
- Tiny fixture round-trips before any large artifact regeneration.
- Gate with the existing Rust baseline commands.

### Reference-features deprecation

Status:
- Effectively soft-deprecated in production, but not removed from the codebase.

Current state:
- The 39-dim feature contract still reserves reference-feature slots at indices `16..21`.
- Current production paths do not rely on reference features.
- Legacy training, reproducibility paths, and tests still support them.

Open decision:
- Hard-deprecate and fail fast for any model that requests `reference_features`.
- Or keep soft-deprecate behavior until every such model is retired.

Guardrail:
- Preserve the feature index contract unless there is an explicit migration plan.

### Configuration surface cleanup

Status:
- Partially improved, not finished.

What remains:
- Env var parsing and validation are still spread across multiple modules and scripts.
- Some scripts still set runtime knobs through ambient env vars instead of explicit parameters.

Why it matters:
- Reproducibility is better when run-critical settings live in CLI or typed API surfaces instead of implicit process state.

## Backlog

### Rust frontier ideas (Ask-first)

1. **Fused constraint and featurize pipeline in Rust**
   - Replace the remaining Python-side per-pair orchestration with one Rust batch call that applies constraints internally and returns features or distances.
   - Only revisit this if fresh profiling shows Python orchestration is still a real bottleneck after the current batching wins.

2. **Further Vec-backed internal storage refactors**
   - Some hot-path Rust structures already moved in this direction; do more only if profiling shows the remaining hash-map overhead is still material.

### Small refactor candidates

- `s2and/model.py`: further separate clusterer, incremental assignment, constraints, and pairwise orchestration responsibilities.
- `s2and/featurizer.py`: keep `many_pairs_featurize` as the public orchestrator but continue extracting cache lifecycle and worklist construction pieces.
- `s2and/data.py`: keep breaking up `ANDData.__init__` stages and flatten deeply nested preprocessing.
- `s2and/feature_port.py`: further separate runtime gating, artifact selection, cache IO, constraints, and batch bridge logic.
- `s2and_rust/src/lib.rs`: reduce duplication between `from_dataset` and `from_json_paths` when there is clear shared stage logic.

### Separate blocked track

- Normalization migration remains blocked: `docs/normalization_migration_blocked.md`
