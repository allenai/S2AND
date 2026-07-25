# Canonical-v2 Release Work Plan

Status date: 2026-07-24

> **Execution order:** Use [1_3_release_todo.md](1_3_release_todo.md) for the
> reviewed v1.3 operator sequence, approvals, commands, and release gates. This
> file is a remediation ledger and is not safe to execute in numbered order.

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
  bound. The canonical ORCID data and manifest have not yet been generated or
  packaged.
- No canonical v1.3 production bundle or implicit default is packaged. The
  historical v1.21 bundle remains available for source/parity work but is not
  accepted by the canonical loader.
- Full warehouse generation, retraining, release-scale profiling, and
  publication remain external work. Run them only from a clean, exact commit
  after the owner approves the reviewed inputs and full commands.

## Remaining remediation groups

Do not execute this section as a pipeline. The complete, dependency-safe order
is the staged flow in [1_3_release_todo.md](1_3_release_todo.md); its blocker
table is the source of truth for status and acceptance evidence.

- **Release identity and publication (B01, B15-B18, B24, B26-B29):** decide
  package/model/data versions and model distribution policy; make distribution
  inventories complete; assemble and verify immutable remote data; publish only
  the exact reviewed workflow bytes through machine-enforced quality,
  attestation, installed-real-model, and public-index gates.
- **Generators and immutable inputs (B02-B10, B19, B25):** make count commands
  clean-clone runnable; bind warehouse snapshots to independently verifiable
  evidence; generate tuples before the ORCID artifact that binds them; re-export
  benchmark names into fields actually consumed by training; produce
  leakage-safe linker assignments and complete byte inventories; prove whether
  linker candidate members are EPS-independent.
- **Pairwise and EPS (B11-B12, B21-B23, B30):** move all pair-overlap checks
  into preflight; create bounded fixed-pair smoke inputs; persist selection
  evidence and sealed test identities; add a publication-boundary smoke,
  validation-only EPS calibration/finalization, and separately invocable
  one-shot pairwise and clustering test evaluators.
- **Linker lifecycle (B13-B14, B20):** keep target inputs outside fresh output
  directories; retain the exact evaluated candidate and deterministic
  query-level predictions; implement one reviewed no-retraining
  candidate-to-production transition that preserves learned bytes and candidate
  ancestry while making target/artifact lifecycle digests agree.

Every expensive or warehouse operation still requires a tiny fixture,
reviewed exact command, explicit owner approval, detached execution, durable
logs, and a completion record. Do not copy changing test totals into this
ledger.

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
