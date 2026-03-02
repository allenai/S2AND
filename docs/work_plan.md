# Work Plan (Next Steps + Backlog)

Status date: 2026-03-02

This doc is intentionally short and forward-looking (no execution logs).
The long-term backlog also lives here under the `Backlog` section.

Start here:
- Threading + preprocessing defaults: `docs/threading.md`
- Rust runtime contract + verification commands: `docs/rust/runtime.md`
- Rust gate commands (write JSON to `scratch/`): `docs/rust/baselines.md`
- Artifact divergence + migration plan: `docs/rust/artifact_divergence.md`
- Stage-wise memory telemetry + predictors: `docs/stage_memory_estimates.md`

## Next work (active)

### Bundle: artifact format unification (Ask-first)

Goal: remove the remaining Rust/Python artifact divergences **without changing normalization policy**.

Scope (recommended order):
1. `name_counts`: Python pickle + Rust JSON → one MessagePack artifact readable natively by both.
2. `specter`: pickle → safetensors (eliminate hidden Python FFI dependency in Rust ingest).
3. `name_tuples`: collapse runtime to one default variant (keep other variants as explicit offline inputs).

Work style:
- Start with dual-read loaders + tiny fixtures.
- Defer regenerating huge artifacts until dual-read is proven.


## Decisions (need explicit call)

### Reference-features deprecation

Guardrails:
- Keep the full **39-dim feature index contract** stable; reserve legacy reference slots at indices `16..21` (filled with `NaN` when refs are disabled).

Choose one:
- Hard-deprecate: any model that requests `reference_features` fails fast.
- Soft-deprecate: keep the legacy reference data + preprocessing path until all such models are retired.

## Separate tracks

- Normalization migration (blocked until data is ready): `docs/normalization_migration.md`

## Backlog

Ask-first / long-term ideas. Keep this section high-level; when something becomes active work, move it into
`Next work (active)` above with explicit verification gates.

### Rust frontier ideas (Ask-first)

1. **Fused constraint + featurize pipeline in Rust**
   - Replace the per-pair Python generator loop with "one Rust call per block/batch" that evaluates constraints internally, featurizes survivors, and returns a feature matrix (or distances).
   - Primary risks: constraint semantics drift, determinism regressions, rollout/rollback complexity.

2. **Vec-backed internal storage refactor (Rust)**
   - Reduce per-pair hash-map overhead in Rust hot loops via index-native Vec-backed internal stores.
   - Only pursue if profiling shows per-pair map overhead dominates after existing batching wins plateau.

### Refactor candidates (keep small, staged)

- `s2and/model.py`: consider splitting clusterer/incremental/constraints/pairwise responsibilities.
- `s2and/featurizer.py`: keep `many_pairs_featurize` as an orchestrator; extract cache lifecycle + worklist construction + telemetry finalize.
- `s2and/data.py`: extract `ANDData.__init__` stages; flatten deep nesting in signature preprocessing.
- `s2and/feature_port.py`: separate runtime gating/build selection/cache IO/constraints/batch bridge logic.
- `s2and_rust/src/lib.rs`: factor shared stages between `from_dataset` and `from_json_paths` to reduce duplication.

### Configuration surface cleanup

- Centralize env var parsing/validation to reduce drift across modules.
- Prefer explicit parameters (API/CLI) over ambient env vars for reproducibility where possible.
