# Canonical-v2 Release Work Plan

Status date: 2026-07-21

This file tracks only work that remains before a canonical-v2 release. The
frozen normalization contract, acceptance thresholds, rollback requirements,
and artifact requirements are in
[normalization_migration_blocked.md](normalization_migration_blocked.md).

## Current release state

- The artifact-independent canonical-v2 implementation is in place:
  normalization is `canonical_v2`, `FEATURIZER_VERSION = 10`, production
  inference requires validated Arrow inputs, and production loading requires
  an explicit complete bundle.
- Canonical tuple, name-count, and ORCID artifact contracts are provenance
  bound. The canonical ORCID bytes and sidecar have not yet been generated or
  packaged.
- No canonical v1.3 production bundle or implicit default is packaged. The
  historical v1.21 bundle remains available for source/parity work but is not
  accepted by the canonical loader.
- Full warehouse generation, retraining, release-scale profiling, and
  publication remain external work. Run them only from a clean, exact commit
  after the owner approves the reviewed inputs and full commands.

## Canonical artifacts and retraining

Run a tiny fixture for every generator and training entry point before any
expensive or internal-data operation. Then:

1. Generate immutable canonical `name_counts` data and its binary
   `name_counts_index` from the reviewed source snapshot.
2. Run the ORCID query on a bounded warehouse sample and verify that emitted
   canonical keys are monotonic before full prefix-count generation.
3. Publish the ORCID direct JSON, runtime metadata sidecar, and producer
   generation report; verify that their identities agree.
4. Regenerate canonical v3 name-tuple data and metadata from the reviewed
   source file.
5. Replace the excluded legacy ORCID JSON with the reviewed canonical JSON and
   sidecar, and make distribution checks require exactly those bytes.
6. Record source snapshot IDs, generation IDs, selected/rejected row counts,
   cardinalities, byte sizes, and SHA-256 digests for every generated artifact.
7. Audit nullable `signatures.author_position` in every release dataset;
   repair source data or explicitly retain nullability.
8. Re-export benchmark names by signature-ID join and report join and
   divergence counts.
9. Train the pairwise model from the exact immutable dataset, count, tuple,
   and ORCID identities.
10. Train the promoted linker against that pairwise bundle and target digest.
    Reuse promoted features only when every materialization sidecar verifies
    against the current inputs.
11. Finalize and reload the complete bundle, then verify every cross-artifact
    identity before evaluation.
12. Run pairwise, clustering, subblocking, Python/Rust parity, quality,
    throughput, and peak-RSS gates.
13. Build clean Python and Rust distributions, install them outside the source
    checkout, and run pairwise and incremental smoke tests with the explicit
    complete bundle path.

Record the exact commit, commands, logs, artifact identities, metrics, runtime,
and peak RSS in the release run report. Do not copy changing test totals here.

## Verification gates

Run focused checks while changing a boundary:

```powershell
uv run pytest -q tests/test_incremental_linking_production.py
uv run pytest -q tests/test_feature_snapshot_cache.py tests/test_name_counts_binding.py
uv run pytest -q tests/test_arrow_training_ingestion.py
uv run pytest -q tests/test_generate_orcid_name_prefix_counts.py
uv run pytest -q tests/test_incremental_linking_artifact.py tests/test_production_model.py tests/test_production_bundle_publication.py
uv run pytest -q tests/test_promoted_linker_training_cli.py tests/test_production_model_cli_flow.py
```

Before handoff, artifact generation, or retraining, run the complete local
gates from a clean checkout:

```powershell
uv lock --check
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run ty check s2and --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global --python-version 3.11 --python-platform linux
uv run ty check scripts --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global --python-version 3.11 --python-platform linux
$env:PYO3_PYTHON=(Resolve-Path '.venv\Scripts\python.exe').Path
uv run cargo fmt --manifest-path s2and_rust/Cargo.toml -- --check
uv run cargo clippy --manifest-path s2and_rust/Cargo.toml --lib --no-deps -- -D clippy::correctness -D clippy::suspicious
uv run cargo test --manifest-path s2and_rust/Cargo.toml --lib --no-default-features
uv run --no-project python scripts/sync_version.py --check
git diff --check
```

Run the installed-distribution and release-scale quality/performance gates from
[normalization_migration_blocked.md](normalization_migration_blocked.md) after
the canonical bundle exists.

## Standing release decisions

- Correctness and provenance override compatibility.
- `s2and.arrow_inputs` is the production Arrow-validation authority.
- Full scans are explicit compatibility/test opt-ins, never silent fallbacks.
- Name-count and name-tuple loaders have one artifact authority; do not add
  legacy fallbacks or duplicate validation paths.
- No production model or implicit default is packaged until the canonical v1.3
  bundle passes all artifact, quality, parity, and performance gates.
- Large generation, retraining, release-scale profiling, paid APIs, and
  internal warehouse queries require a tiny fixture first and explicit owner
  approval.
- Do not add compatibility shims for pre-canonical artifacts unless a concrete
  supported consumer is identified and the owner approves the contract.
- Public API, schema, and serialized-output changes require separate approval
  and focused compatibility evidence.
- Comparison-only CLIs and tests should be reassessed after canonical-v2
  comparison ends; retain any independent strict oracle or cost guardrail.
