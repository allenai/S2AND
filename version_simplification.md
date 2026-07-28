# Version simplification for the v1.3 release

Status: corrected design and implementation record

This document replaces the original version-simplification proposal. The
original was directionally right about deleting redundant counters and schema
labels, but several proposed safeguards would have created more lifecycle and
compatibility machinery than they removed. This version records the
independent critique, the smaller resulting design, and the checks required to
show that the deletion is safe.

## Outcome

S2AND has three version authorities:

| Authority | Value for this release | Meaning |
|---|---:|---|
| Runtime | `1.0.0` | The installed `s2and` and `s2and-rust` packages |
| Artifact release | `1.3` | The coordinated production model and final public-data publication |
| Public-data format | `1` | The independently readable Arrow/name-count representation |

One additional field is deliberately not a version: pairwise bundles retain
`eps_calibration: "pending" | "calibrated"`. Pending and calibrated pairwise
bundles have the same file inventory, and a genuinely calibrated EPS may equal
the placeholder value `0.5`. This lifecycle fact cannot be inferred safely.

The runtime implements one normalization and featurization behavior. The
separate `NORMALIZATION_VERSION` and `FEATURIZER_VERSION` authorities are
deleted. Production models instead record the exact runtime that generated
them, and the loader requires exact equality. Public data records its one
format epoch. Private binary formats keep their magic bytes.

## Independent critique of the original proposal

### What it got right

- `FEATURIZER_VERSION` and `NORMALIZATION_VERSION` duplicated a compatibility
  decision that is already made by shipping the Python and Rust runtimes
  together.
- The persistent feature-snapshot cache had no production caller. Its module,
  documentation, and dedicated tests were pure maintenance surface.
- Many JSON `schema_version` values were labels on fixed-role files with one
  writer, one reader, strict required fields, and content hashes. They did not
  enable an actual migration path.
- The checked-in `production_model_v1.21` directory was neither package data
  nor a supported runtime model. Most of its roughly 39 MB existed only to
  support tests that skipped when the files were absent.
- Model/data release numbering needed one owner-authored source instead of a
  separate `--production-version` choice.

### What it got wrong

1. **Inventory cannot replace EPS lifecycle state.** A pairwise bundle before
   and after calibration contains the same files. Inferring calibration from
   EPS `0.5` is also invalid because `0.5` can be the selected optimum.
2. **Same-major runtime compatibility was an unsupported promise.** A rule
   such as "generator is older than reader and both have major version 1"
   does not prove that feature values are compatible. Supporting it would
   require a model corpus, download/update policy, semantic fixture format,
   and long-lived compatibility tests.
3. **The proposed embedded semantic fixture was new artifact machinery.**
   Existing Python/Rust feature parity, exact ordered feature contracts,
   booster-width checks, stored-matrix scorer fixtures, and installed-wheel
   smoke tests cover the useful failure modes without adding a new model file.
4. **A frozen `ReleaseContext` object was broader than the problem.** Only
   `release_version` needs a human-owned release choice. Runtime and format
   values are code constants; source and candidate identities are already
   content-bound by the prepared plans and `run_binding.json`.
5. **Candidate-wheel registries, stage-by-stage wheel hashes, PyPI/tag
   orchestration, and import-origin gates were release-process expansion.**
   Clean-source checks, synchronized package versions, candidate builds,
   distribution inspection, and installed-wheel smoke remain operating
   procedure, not a new compatibility subsystem.
6. **A text-scanning `test_version_marker_budget.py` would test spelling, not
   behavior.** The repository can use an ad-hoc marker inventory during this
   migration; durable tests should exercise readers, writers, parity, and
   failure boundaries.
7. **Renaming private counters is not simplification.** The memory telemetry
   field and the pickle-surviving altered-presplit generation either remain
   private implementation details or are deleted when unused; they are not
   renamed merely to fit a naming taxonomy.
8. **The proposal described stale current state.** Before this work the live
   production bundle writer emitted v5, not v6, and component release versions
   had already been removed.

### Remaining costs and assumptions

- Exact runtime matching is intentionally conservative: any package release
  that changes model-visible behavior needs regenerated artifacts. It also
  assumes a published version is immutable; the release workflow, not another
  manifest field, prevents two different builds from being published as
  `1.0.0`.
- Public format `1` is a clean break. Existing generated Arrow and name-count
  roots are not migrated or accepted; the release needs newly generated data.
- "One name-count index" means one copy in each independently publishable,
  self-contained root. It does not mean a machine-local absolute path or a
  network dependency shared by all publications.
- Homogeneous validation matrices may execute named inputs in one compact
  loop when they share the same setup and failure layer. High-risk boundaries
  that need independent isolation or reporting remain separate or
  parametrized. Corruption checks, cross-language parity, and release gates
  remain even when several logical inputs share one collected pytest node.

## Corrected durable contracts

### Runtime

`VERSION` remains the package release authority. The Python package exposes
that installed version, and the Rust loader requires the extension's
`__version__` to match it exactly. `scripts/sync_version.py` synchronizes the
project and Rust packaging metadata; it does not maintain a second runtime
constant.

There is intentionally no same-major compatibility range. Relaxing exact
equality later requires evidence from real released artifacts and is a
separate decision.

### Production model

The model root manifest has these responsibilities only:

```json
{
  "kind": "s2and_model",
  "release_version": "1.3",
  "generated_by_runtime": "1.0.0",
  "eps_calibration": "calibrated",
  "sha256": {
    "clusterer.json": "<sha256>",
    "incremental_linker/booster.lgb": "<sha256>",
    "incremental_linker/metadata.json": "<sha256>",
    "pairwise/main.lgb": "<sha256>",
    "pairwise/main_prediction_fixture.json": "<sha256>",
    "pairwise/nameless.lgb": "<sha256>",
    "pairwise/nameless_prediction_fixture.json": "<sha256>",
    "reproducibility/incremental_linker_training_target.json": "<sha256>"
  }
}
```

The example shows the smallest supported complete runtime inventory. Production
training additionally records both
`reproducibility/pairwise_training_config.json` and
`reproducibility/pairwise_training_summary.json`; the manifest accepts both or
neither. Release calibration requires the config, so a release candidate always
carries the pair. Synthetic runtime smoke bundles may omit them rather than
fabricating training provenance.

The checksum-key inventory determines whether the bundle is pairwise-only or
complete. Callers may require one of those roles, but the manifest does not
serialize a derivable `bundle_kind`.

Pairwise bundles may be pending or calibrated. Complete bundles must be
calibrated. This is enforced alongside exact inventory validation.

Clusterer configuration, scorer fixtures, and incremental-linker metadata use
their fixed role, exact required fields, content hashes, ordered feature names,
and model-width validation. They do not carry decorative schema or
featurizer/normalization counters. A standalone linker records
`kind: "s2and_incremental_linker"` and `generated_by_runtime`; a linker inside
a model is also bound by the model-root hash inventory.

### Release authority

The owner-authored `release.json` contains one top-level
`release_version`. `prepare-run` copies it into the generated model plan.
Training obtains the model release from that verified plan; there is no
independent `--production-version` argument. The final public-data assembler
obtains the same value from the prepared plan rather than asking the operator
to type it again.

`run_binding.json` remains the location-independent binding among the model
plan, evaluation plan, reviewed baseline record, finalized candidate manifest,
and public-data root manifest. It is identity evidence, not another version
authority.

### Public data

The final publication root records:

```json
{
  "kind": "s2and_public_data",
  "release_version": "1.3",
  "format_version": 1,
  "dataset_manifests": {
    "pubmed": {
      "path": "pubmed/manifest.json",
      "sha256": "<sha256>"
    }
  }
}
```

Generic local Arrow conversion roots are not artifact releases and therefore
omit `release_version`. The final assembler writes the strict published root.

An independently opened Arrow dataset records a stable kind,
`format_version: 1`, portable `paths`, and a flat content inventory. Its
collection root already supplies the dataset name. The inventory maps semantic
roles to byte count and SHA-256; it does not repeat `kind: "file"`, physical
paths, or a derived `generation_id`.

The name-count manifest records a stable kind, the same
`format_version: 1`, and facts for the four fixed binary roles. Each role's
path is derived as `<role>.bin`; the manifest does not repeat it.

The public format epoch covers persisted field meaning as well as framing.
Changing name canonicalization, count-key eligibility/composition, missing-key
meaning, Arrow table interpretation, or a public sidecar encoding requires a
format bump and regenerated public data.

Physical Arrow batch sizes and batch-index validity are checked from the
actual IPC/index bytes. They are metrics in producer or release reports, not a
second manifest authority. Private batch-index and name-count binary magic
bytes remain because they guard independently parsed binary layouts.

### Fixed-role JSON

The following fixed-role files omit schema labels:

- clusterer configuration and pairwise prediction fixtures;
- incremental-linker metadata, logistic-gate artifacts, and retained training
  targets;
- prepared model/evaluation plans and release component/final reports;
- transient raw/labeled candidate plans where producer and consumer are in
  the same runtime.

Readers continue to validate every field they consume, numeric ranges, model
widths, hashes, and run bindings. Removing a schema label is not permission to
accept a missing behavioral input.

Private formats that cross a genuine persistence boundary keep their own
guard. In particular, private binary magic values and the altered-presplit
cache generation are outside this consolidation.

## Historical model cleanup

The unsupported `s2and/data/production_model_v1.21/` tree is removed.

- A realistic main LightGBM booster and its probability fixture may live under
  `tests/fixtures/` only if they cover behavior not represented by the bounded
  synthetic parity models. Declared fixtures must fail when absent; they must
  never silently skip.
- The small 53-feature linker target moves to a test/release-evidence fixture
  and remains byte-identified where the release runbook needs it.
- The large nameless and linker boosters are not retained merely to multiply
  the same evaluator test.

Historical documentation may name v1.21 when describing provenance, but live
instructions must not present its deleted path as an executable input.

## Test-suite simplification

The test suite protects behavior, not every deleted representation.

Keep:

- reader/writer round trips and malformed-boundary cases for current public
  artifacts;
- Python/Rust feature and scorer parity;
- model inventory, checksum, exact-runtime, and EPS lifecycle failures;
- release-plan identity/leakage checks and gate calculations;
- a small number of end-to-end trainer/assembler/installed-artifact smokes.

Delete or consolidate:

- tests whose only purpose is rejecting every retired schema/version string;
- cache tests for the deleted persistent cache;
- repeated CLI-parser and JSON-key tests already exercised through one
  end-to-end command;
- three copies of the same test for pairwise, nameless, and linker boosters
  when one realistic booster plus bounded synthetic adversaries covers the
  evaluator;
- implementation-coupled assertions for fields removed from the durable
  contract.

Distinct inputs may be named cases in a compact loop when they exercise one
validation boundary; separate nodes are retained when their isolation is
useful. The verified suite has 995 collected pytest nodes, not merely 995
logical inputs or behaviors. Reducing that number is not itself the goal;
removing redundant setup, compatibility branches, and representation-only
assertions is.

## Verification

The implementation is complete only when all of the following hold:

1. `VERSION`, Python distribution metadata, Rust crate metadata, lockfiles,
   and installed Python/Rust `__version__` values agree at `1.0.0`.
2. No live `FEATURIZER_VERSION`, `NORMALIZATION_VERSION`, or
   `REQUIRED_RUST_EXTENSION_VERSION` authority remains.
3. Production model loading requires exact runtime identity, exact checksum
   inventory, model widths, artifact hashes, and valid EPS lifecycle.
4. Public Arrow and name-count readers require kind plus public format `1`,
   validate real files, and retain binary magic checks.
5. The release version is selected once in `release.json` and reaches the
   model and final public-data roots without a second CLI choice.
6. Missing declared test fixtures fail instead of skip.
7. Current documentation contains no executable references to the deleted
   cache, deleted v1.21 bundle, or retired schema/version fields.
8. Focused Python/Rust, model, data, release, and distribution tests pass,
   followed by the complete test suite, Ruff, formatting, and configured type
   checks.

Before publication, rerun the external check that package version `1.0.0` and
the intended tag are still unused. That check is a release precondition, not a
new in-repository version subsystem.
