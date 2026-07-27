# Canonical-v2 Release Work Plan

Status date: 2026-07-25

> **Execution order:** Use [1_3_release_todo.md](1_3_release_todo.md) for the
> reviewed v1.3 operator sequence, approvals, commands, and release gates. This
> file is a remediation ledger and is not safe to execute in numbered order.
> Use [1_3_release_blockers.md](1_3_release_blockers.md) for current blocker
> status and closure evidence.

This file tracks only work remaining before a canonical-v2 release.
`release_spec.json` is the authority for acceptance and rollback decisions;
[normalization_migration_blocked.md](normalization_migration_blocked.md)
retains the migration history.

## Current release state

- The artifact-independent canonical-v2 implementation is in place:
  normalization is `canonical_v2`, `FEATURIZER_VERSION = 10`, production
  inference requires validated Arrow inputs, and production loading requires
  an explicit complete bundle.
- Canonical tuple, name-count, and ORCID schemas bind their content identities,
  but full warehouse source provenance is not yet independently bound (B27/B28).
  The canonical ORCID data and manifest have not yet been generated. The
  current pre-release tree packages neither; Stage 1 of the v1.3 runbook adds
  and declares both together.
- No canonical v1.3 production bundle or implicit default is packaged. The
  historical v1.21 bundle remains available for comparison work but is not
  accepted by the canonical loader.
- Full warehouse generation, retraining, release-scale profiling, and
  publication remain external work. Run them only from a clean, exact commit
  after the owner approves the reviewed inputs and full commands.

## Remaining remediation groups

Do not execute this section as a pipeline. The complete, dependency-safe order
is the staged flow in [1_3_release_todo.md](1_3_release_todo.md). The separate
[blocker ledger](1_3_release_blockers.md) is the source of truth for current blocker
status and closure evidence.

- **Release specification (B01, B12, B24, B30):** freeze the package/model
  identity, EPS, acceptance thresholds, and rollback decision in
  `release_spec.json`.
- **Data (B02-B11, B19, B21, B23, B25, B27-B28):** generate immutable canonical
  inputs, validate leakage and provenance, and publish one complete data
  manifest.
- **Model (B13-B15):** fit the linker once against the final pairwise model,
  write one complete v5 bundle with its replay target, and reload those exact
  bytes.
- **Evaluation and publication (B16-B17, B26, B32):** produce one
  evaluation report, stage the exact approved bytes, and let the workflow write
  `SHA256SUMS`. Its transport index is not another semantic authority.

Every expensive or warehouse operation requires a tiny fixture, reviewed exact
command, owner approval, and durable logs. Detached jobs use the supported
launcher so command, inputs, PID, and logs remain reproducible. Logs are
operational records, not additional release authorities.

## Verification gates

Run focused checks while changing a boundary:

```powershell
uv run pytest -q tests/test_incremental_linking_production.py
uv run pytest -q tests/test_feature_snapshot_cache.py tests/test_name_counts_binding.py tests/test_generate_name_counts_script.py
uv run pytest -q tests/test_arrow_training_ingestion.py
uv run pytest -q tests/test_generate_orcid_name_prefix_counts.py
uv run pytest -q tests/test_incremental_linking_artifact.py tests/test_production_model.py tests/test_production_bundle_publication.py
uv run pytest -q tests/test_promoted_linker_training_cli.py
uv run pytest -q tests/test_train_pairwise_script.py tests/test_eval_prod_models.py tests/test_release_workflow.py
```

Before handoff, artifact generation, or retraining, run the complete local
gates from a clean checkout:

```powershell
uv lock --check
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run ty check s2and --ignore unresolved-import --ignore unresolved-reference --ignore unresolved-attribute --ignore possibly-missing-attribute --ignore unused-type-ignore-comment --python-version 3.11 --python-platform linux
uv run ty check scripts --exclude scripts/archive --ignore unresolved-import --ignore unresolved-reference --ignore unresolved-attribute --ignore possibly-missing-attribute --ignore unused-type-ignore-comment --python-version 3.11 --python-platform linux
$env:PYO3_PYTHON=(Resolve-Path '.venv\Scripts\python.exe').Path
uv run cargo fmt --manifest-path s2and_rust/Cargo.toml -- --check
uv run cargo clippy --manifest-path s2and_rust/Cargo.toml --lib --no-deps -- -D clippy::correctness -D clippy::suspicious
uv run cargo test --manifest-path s2and_rust/Cargo.toml --lib --no-default-features
uv run --no-project python scripts/sync_version.py --check
git diff --check
```

The retired `scripts/archive` tree is outside this gate. Static type errors
remain blocking. After the canonical bundle exists, run the installed-package
and release-scale gates in
[normalization_migration_blocked.md](normalization_migration_blocked.md).

## Standing release decisions

- Correctness and provenance override compatibility.
- Production Arrow, name-count, and name-tuple inputs each have one validation
  authority; full scans and legacy fallbacks are never implicit.
- The v1.3 model is an immutable external bundle, not a packaged default.
- Large generation, retraining, profiling, paid APIs, and warehouse queries
  require a tiny fixture and owner approval.
- Compatibility, public-API, schema, or serialization changes require a
  concrete consumer, separate approval, and focused evidence.
