# Canonical-v2 Release Work Plan

Status date: 2026-07-20

This is the active plan for completing the canonical-v2 migration and producing
the v1.3 release candidate. Completed audit ledgers, stale test totals, and
historical microbenchmark tables have been removed from this file. The frozen
normalization contract, artifact requirements, and quality thresholds remain in
[normalization_migration_blocked.md](normalization_migration_blocked.md).

## Current release state

- Phase 0 and Phase 1 implementation work is landed through `88c6f19` on
  `canonical-v2-migration`.
- Later-phase and schema-changing candidates are preserved in stash commit
  `328c79d12f15` (`phase2-schema-and-generation-candidates-after-phase1`,
  currently `stash@{0}`). Do not apply that stash wholesale; two candidate
  groups have confirmed blockers and mixed files require hunk-level recovery.
- The artifact-independent canonical-v2 implementation is substantially in
  place: normalization is `canonical_v2`, `FEATURIZER_VERSION = 10`, production
  inference uses validated Arrow inputs, model loading requires an explicit
  complete bundle, and canonical tuple/count/ORCID artifacts are provenance
  bound.
- Require-constraint extraction, pair-cache lineage, strict training
  `author_block`, mandatory name-count binding, ORCID producer ordering, and
  Python/Rust whitespace parity are implemented and verified.
- No canonical v1.3 production bundle or implicit default is packaged. The
  historical v1.21 bundle remains a source/parity artifact and is not accepted
  by the canonical loader.
- Full warehouse generation, retraining, release-scale profiling, and
  publication require a clean exact commit plus explicit owner approval.

## Status vocabulary

- **Complete:** landed in the reviewed commit stack and focused verification
  passes.
- **Preserved candidate:** retained in the named stash; not accepted.
- **Blocked:** a concrete correctness gap prevents acceptance.
- **Open:** confirmed work remains.
- **Decision:** the recommended change affects an API, serialized output, or
  broad source ownership and needs explicit owner approval.
- **External:** blocked on reviewed datasets, immutable generated artifacts, or
  an approved expensive run.
- **Deferred:** deliberately excluded from this release plan.

## Phase 0: stabilize and split the working tree

**Status: Complete**

The 2026-07-20 dirty tree was inventoried and split into the following Phase 1
stack:

| Commit | Boundary |
| --- | --- |
| `ff3f3f2` | Retraining input lineage, required blocks, require signals, and mandatory name-count binding |
| `4c2fbdc` | Python-compatible Rust whitespace for canonical names and email rejection |
| `c933268` | Canonical ORCID producer ordering and shared prefix-count artifact contract |
| `25814a3` | Pairwise-model name-count binding CLI-flow evidence |
| `46f6e00` | Name-count `--limit` help contract |
| `88c6f19` | Pairwise output-directory preflight before artifact loading |

All residual changes were classified. Phase 2, name-tuple v3, and incomplete
name-count publication work were preserved in the named stash. No production
artifact generation or retraining was run.

Exit condition met: the active branch has a reviewable Phase 1 stack and no
unexplained product-code changes.

## Phase 1: correctness required before retraining

### P0 - require-constraint extraction

**Status: Complete (`ff3f3f2`)**

- Read `constraint_require_count` only from `decision_row_signals`, the
  authoritative merged constraint state.
- Do not fall back between decision and pairwise distance signal mappings.
- Retain the real extraction regression and add a production-level cross-batch
  case proving a required component cannot be excluded during rescore.

### P0 - pair-feature cache lineage

**Status: Superseded** — the persistent pair-feature SQLite cache was removed
entirely (2026-07-20) in favor of the training-boundary feature snapshot cache
(`s2and/feature_cache.py`, see docs/caching.md); snapshot keys fingerprint the
ingested source bytes, name-count binding, name tuples, featurizer
  configuration, and exact pair lists.

### P0 - mandatory name-count binding

**Status: Complete (`ff3f3f2`, `25814a3`)**

- Any model selecting `name_counts` must carry the complete four-field
  generation/source binding.
- Treat this as an intentional artifact/runtime contract change, not mechanical
  cleanup.
- Gate calls on whether the model uses name-count features, then require the
  complete binding without a legacy optional mode.
- Do not retain backward compatibility unless a concrete supported artifact
  requires it and the owner approves it.

### P0 - training block integrity

**Status: Complete (`ff3f3f2`)**

- Training ingestion must reject missing, null, and empty `author_block`.
- Keep this as a training-profile invariant; do not make the field globally
  non-null for every Arrow inference surface without the pending dataset audit.
- Do not introduce an `"unknown"` block sentinel or a downstream grouping
  fallback.

### P0 - ORCID warehouse ordering

**Status: Complete in code (`c933268`); warehouse execution remains external**

- Make the warehouse query produce and order by the same canonical ORCID key
  consumed by `build_prefix_counts_from_sorted_rows`.
- Retain the streaming monotonicity assertion.
- Do not buffer and sort the full result in Python; that would hide producer
  drift and defeat bounded processing.
- Add bounded fixtures for embedded valid ORCIDs with leading/trailing junk,
  lowercase `x`, invalid/missing rows, and repeated canonical groups.

### P1 - Python/Rust whitespace parity

**Status: Complete (`4c2fbdc`)**

- Add one narrow Rust compatibility predicate matching Python whitespace
  behavior for U+001C through U+001F.
- Use it only in canonical first-name splitting and email rejection.
- Test all four separators, adjacent controls, ordinary Unicode whitespace, and
  canonical-name/email parity.
- Do not globally rewrite unrelated Rust `split_whitespace` call sites.

### P2 - small retrain-safety corrections

**Status: Complete (`46f6e00`, `88c6f19`)**

- The name-count `--limit` help now matches its optional bounded-row behavior.
- Pairwise training rejects an existing output directory before loading
  artifacts or starting expensive work.

Phase 1 exit condition met: every behavior has focused regression coverage,
grouped tests pass, and retraining cannot silently use stale pair features,
unbound counts, invalid blocks, or the wrong require/ORCID semantics.

## Preserved blockers and owner decisions

### Name-count source/index publication

**Status: Blocked; preserved in `328c79d12f15`**

- The preserved callback/rollback candidate is not transactional across the
  source and binary-index manifests.
- A crash after publishing the source pointer leaves source=new/index=old. A
  late failure after publishing the index can roll the source back while
  leaving index=new.
- The current synthetic callback test never inspects actual index state, and
  catching `BaseException` conflicts with the repository's narrow-exception
  rule.
- Redesign around one authoritative pointer that binds both generations, or
  explicitly narrow the contract and add real integration failure tests before
  recovering this candidate.

### Name-tuple artifact v3

**Status: Complete (recovered from `328c79d12f15`)**

- v3 adds the `dropped_duplicate_canonical` generation count and a load-time
  invariant (Python and Rust) that every input pair is accounted for. The v2
  metadata left 3,768 of 9,925 input pairs unaccounted.
- Owner approved the serialization-format change 2026-07-20. The Python loader,
  packaged metadata, generator, Rust loader, tests, and migration documentation
  landed as one unit, before canonical artifact generation, so the immutable
  v1.3 tuple artifact is generated once under the complete-accounting schema.

## Phase 2: artifact and bundle integrity

### Linker artifact v4 and target identity

**Status: Complete (recovered from `328c79d12f15`)**

- `incremental_linking_artifact_v4` binds the canonical digest of the complete
  training target JSON; `save_incremental_linking_artifact` requires the
  target spec.
- Final bundle assembly and production loading both reject a target changed
  after checksums or manifests are refreshed (regression-tested, including a
  refreshed-manifest tamper case).
- The target-digest field is verified through writer, loader, finalizer,
  corruption, and installed-smoke paths.

### Precomputed promoted-feature reuse

**Status: Blocked; preserved in `328c79d12f15`**

- The candidate identity omits per-dataset Arrow manifests/files and the
  effective default name-count index even though materialization consumes
  them. `--reuse-existing-features` can therefore silently reuse stale output.
- Persisted bundle-relative paths currently use `str(Path)`, which writes
  backslashes on Windows and breaks replay on POSIX. Persist portable paths
  with `.as_posix()`.
- Bind reuse to source and pairwise bundles, target JSON, feature schema, NaN
  policy, exemplar cap, selected rows, every consumed Arrow generation, and
  the explicit or effective name-count index.
- Reuse must fail closed when any materialization input differs.
- Portable reusable metadata uses bundle-relative paths; replay must not depend
  on historical scratch paths.

### Pairwise fixture and config hardening

**Status: Preserved candidate in `328c79d12f15`**

- Fixed prediction-fixture tolerances, probability bounds, Boolean numeric
  rejection, and direct required-manifest access are coherent but belong in a
  separate Phase 2 commit.
- Recover only their hunks from the mixed production bundle/model files.

### Single pairwise-binding authority

**Status: Complete**

- `pairwise_bundle_binding` is an explicit required argument on
  `save_incremental_linking_artifact` and metadata `build`; only the validated
  top-level field persists.
- New `audit_metadata` rejects the reserved `pairwise_bundle_binding` key.
- Existing artifacts load an old nested audit copy as inert historical
  metadata (regression-tested).
- Owner approved the callable API and serialized-output change 2026-07-20; no
  canonical bundle was generated under the old layout.

### Single-pass bundle validation

**Status: Open**

- Keep public `pairwise_bundle_binding(path)` as a validating boundary.
- During model loading, derive the binding from the already validated manifest
  and clusterer config instead of hashing the whole bundle a second time.
- Add a checksum-call-count regression showing one hash per declared file on a
  normal load while preserving tamper rejection.

### Artifact save validation

**Status: Open**

- Remove only the explicit contract-validation call immediately after metadata
  construction; `__post_init__` already performs it.
- Keep the staged reload because it verifies serialized bytes, booster checksum,
  Rust loading, and prediction fixtures.
- Keep atomic staging, fsync, publication locks, and manifest hashing.

### Rust/Arrow production boundary coverage

**Status: Open**

- Add a production-entrypoint test proving filtered or incremental Arrow
  prediction rejects unindexed input.
- Keep small direct checks for intentionally removed public Rust APIs.
- Do not restore repository-wide regex scanners or tests whose only assertion is
  that no call sites exist.

Phase 2 exit condition: artifact metadata has one authority for each identity,
normal loads do not repeat full checksum work, and all save/load/finalize
boundaries fail closed under mutation.

## Phase 3: slim the pull request

### Pair-ablation study

**Status: Complete**

- Owner approved (2026-07-20) removing the complete 2,544-line one-off study
  from this migration PR. Its two CLIs, `_pair_ablation/`, documentation, and
  five tests were deleted together and preserved verbatim in the ignored
  `scratch/pair_ablation_study/` workspace with a provenance README; git
  history before the removal commit retains the tracked copies.
- If the study's conclusion should inform maintained documentation, recover it
  from the preserved `docs/pair_ablation.md`; the executable machinery does
  not ship.

### One-off benchmark machinery

**Status: Open**

- Move or delete `scripts/bench_python_name_counts.py` from maintained source.
- Remove the in-file Rust benchmark harness that recreates the retired dense
  layout. Preserve relevant conclusions in documentation rather than obsolete
  executable machinery.

### Proven dead private code

**Status: Open**

Delete after retargeting any useful assertions to live paths:

- `model._missing_arrow_prediction_artifacts_error`;
- `production._summarize_query_views`;
- `production.promoted_incremental_observed_probe`;
- `production._raw_arrow_plan_window_size/_raw_arrow_plan_windows`;
- `policy.arrow_paths_have_name_counts_index`;
- `subblocking._load_canonical_orcid_prefix_counts`.

For `ANDData._compute_signature_name_counts`, move canonical-key coverage to the
live batched preprocessing path before deleting the private single-row helper.

### Rust and test cleanup

**Status: Open**

- Remove the six dead serde derives together so the compiler verifies the
  transitive cleanup.
- Remove the broken reference to an ignored canonical-example generator unless
  regeneration becomes a supported tracked workflow.
- Correct the enabled aarch64 matrix comment.
- Keep release-policy coverage, but prefer parsed workflow/policy assertions to
  brittle formatting snapshots.

Phase 3 exit condition: branch-local experimental code is either explicitly
owned or removed, dead private paths are gone, and no public API is deleted by
accident.

## Canonical artifact generation and retraining

These tasks remain external until Phases 0 through 3 are complete and the owner
approves the expensive commands.

1. Run tiny local fixtures for name-count and ORCID generation.
2. Generate the immutable canonical `name_counts` generation and binary
   `name_counts_index` from the reviewed source snapshot.
3. Run the ORCID query on a bounded internal warehouse sample and verify its
   emitted canonical key is monotonic before full prefix-count generation.
4. Record and validate source snapshot IDs, generation IDs, selected/rejected
   row counts, cardinalities, byte sizes, and SHA-256 digests.
5. Audit nullable `signatures.author_position` in every intended release
   dataset; repair source data or explicitly retain nullability.
6. Re-export benchmark names by signature-ID join and report join/divergence
   counts.
7. Train the pairwise model from the exact immutable dataset/count/tuple/ORCID
   identities.
8. Train the promoted linker against that exact pairwise bundle and target
   digest. Do not reuse promoted features until the blocked materialization
   identity is corrected and verified.
9. Finalize the complete bundle, reload it, and verify every cross-artifact
   identity before evaluation.
10. Run pairwise, clustering, subblocking, parity, quality, throughput, and
    peak-RSS gates.
11. Build clean Python and Rust distributions, install them outside the source
    checkout, and run pairwise plus incremental smoke using the explicit
    complete bundle path.

For commands, acceptance thresholds, and rollback requirements, use
[normalization_migration_blocked.md](normalization_migration_blocked.md).

## Verification gates

Run focused tests first:

```powershell
uv run pytest -q tests/test_incremental_linking_production.py
uv run pytest -q tests/test_feature_snapshot_cache.py tests/test_name_counts_binding.py
uv run pytest -q tests/test_arrow_training_ingestion.py
uv run pytest -q tests/test_generate_orcid_name_prefix_counts.py
uv run pytest -q tests/test_incremental_linking_artifact.py tests/test_production_model.py tests/test_production_bundle_publication.py
uv run pytest -q tests/test_promoted_linker_training_cli.py tests/test_production_model_cli_flow.py
```

Before handoff or retraining:

```powershell
uv lock --check
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run ty check s2and --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global
$env:PYO3_PYTHON=(Resolve-Path '.venv\Scripts\python.exe').Path
uv run cargo fmt --manifest-path s2and_rust/Cargo.toml -- --check
uv run cargo test --manifest-path s2and_rust/Cargo.toml --lib --no-default-features
uv run --no-project python scripts/sync_version.py --check
git diff --check
```

Do not copy historical test totals into this plan. Record exact commands,
commit SHA, logs, artifact IDs, metrics, runtime, and peak RSS in the release
run report produced for the candidate.

## Deferred or explicitly rejected for this release

- Deleting `feature_block_from_arrow_paths` without a separately approved
  public-API decision.
- Deleting public-looking test-only helpers solely because the repository has
  no production caller.
- Removing `unidecode_char_map` plumbing without a measured, reviewed patch.
- Deleting `final_limits` behavior; the empty-query telemetry path is real.
- Treating `_n_features_in` as write-only.
- Removing tested boundary guards because ordinary callers currently satisfy
  their invariants.
- Reverting archive safety/provenance warnings.
- Pinning literal trained LightGBM probabilities in unit tests.
- Wholesale deletion of release/CI policy tests.
- Building a generic validation/checksum framework or broad test-helper
  refactor during the migration.
- Adding compatibility fallbacks that mix legacy and canonical artifacts.
- Fixing the unreachable empty-ID Rust self-skip discrepancy while all readers
  already reject empty IDs.

## Standing release decisions

- Correctness and provenance override compatibility.
- `s2and.arrow_inputs` is the production Arrow validation authority.
- Full scans are explicit compatibility/test opt-ins, never silent fallbacks.
- No production model or implicit default is packaged until the canonical v1.3
  bundle passes all gates.
- Large generation, retraining, release-scale profiling, paid APIs, and
  internal warehouse queries require a tiny fixture first and explicit owner
  approval.
