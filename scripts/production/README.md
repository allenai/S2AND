# Production Release Artifacts

Status date: 2026-07-24.

Scripts in this directory create or validate release-candidate artifacts. The
model artifact is a native `production_model_vX.Y/` directory; whether v1.3 is
external or packaged under `s2and/data/` remains release blocker B15. Do not
create production pickles.

This is a component command reference, not the v1.3 execution order. No full
warehouse query or training run is currently authorized by this document.
Follow [../../docs/1_3_release_todo.md](../../docs/1_3_release_todo.md) for the
current blockers, bounded preflights, EPS stage, test-unblinding rules,
candidate promotion, approvals, and exact-byte publication. Examples below use
`X.Y` as the model-bundle version; the coordinated Python/Rust package version
is a separate open decision.

## 1. Train Pairwise

```powershell
$RunRoot = "D:/local-unsynced/s2and-vX.Y"

uv run python scripts/production/model/train_pairwise.py `
  --production-version X.Y `
  --data-dir path/to/canonical_benchmark_data `
  --matrix-work-dir "$RunRoot/matrix-work" `
  --output-dir "$RunRoot/pairwise_stage/production_model_vX.Y" `
  --run-full
```

This writes the pairwise-only bundle stage:

```text
production_model_vX.Y/
  clusterer.json
  manifest.json
  pairwise/
    main.lgb
    nameless.lgb
    main_prediction_fixture.json
    nameless_prediction_fixture.json
  reproducibility/
    pairwise_training_config.json
    pairwise_training_summary.json
```

This stage is loadable for training/finalization, but it is not a complete
runtime production model until the linker is added.

Preflight resolves and hashes every selected dataset input before loading it.
The operator must independently confirm that `--matrix-work-dir` is local and
unsynced and that the reviewed size estimate fits. The CLI requires that
directory to exist and be empty, verifies it with a temporary flushed and
synced write, removes the probe, and records measured free bytes. Concrete
matrix writes also check their actual byte requirement, and emitted metrics
must be finite. Sampled clustered datasets honor the requested pair limits,
but fixed-pair CSV datasets are currently loaded in full. Release blocker B22
therefore requires a separate deterministic pre-sampled smoke root; the
pair-size flags alone do not bound that work. Use `--preflight-only` for a
readiness check that leaves the matrix-work directory empty and the output
directory absent.

Passing `--datasets ...` means smoke mode: it may exercise training, but it
never creates a model bundle. A production run uses `--run-full`, the complete
fixed dataset set, and the reviewed `--pairwise-test-manifest-sha256`; the
trainer records that digest but never opens the manifest or featurizes/scores
test pairs. Full release training does not use `--feature-cache-dir`.
`--datasets` and `--run-full` are mutually exclusive. This smoke stops before
bundle publication and Rust fixture reload, so it does not satisfy B21.

The current full-run report still lacks the complete selection evidence
required by B23. The sealed test manifest is opened only by
`release_pairwise.py evaluate-pairs` in Stage 8.

## 2. Select and freeze EPS

Do not train the v1.3 linker directly from the raw pairwise stage. Select EPS
on validation identities only, persist every trial and weighting definition,
then freeze the reviewed pairwise stage. If the trainer-selected EPS is
accepted, designate that exact stage; if review selects a different EPS, use
`release_pairwise.py finalize-eps`. Validation calibration is `calibrate-eps`;
the sealed Stage-8 routes are `evaluate-pairs` and `evaluate-clusters`. Stage 6
of the release runbook is authoritative.

```powershell
uv run python scripts/production/model/release_pairwise.py calibrate-eps `
  --pairwise-model "$RunRoot/pairwise_stage/production_model_vX.Y" `
  --eps 0.55 0.60 0.65 0.70 `
  --output-json "$RunRoot/reports/eps/calibration.json"

uv run python scripts/production/model/release_pairwise.py finalize-eps `
  --source-bundle "$RunRoot/pairwise_stage/production_model_vX.Y" `
  --expected-manifest-sha256 REVIEWED_SOURCE_MANIFEST_SHA256 `
  --expected-old-eps REVIEWED_OLD_EPS `
  --new-eps REVIEWED_NEW_EPS `
  --output-bundle "$RunRoot/pairwise_calibrated/production_model_vX.Y"
```

## 3. Train Linker And Finalize

```powershell
$PairwiseModel = "$RunRoot/pairwise_calibrated/production_model_vX.Y"
# If validation accepted the trainer-selected EPS, point this at pairwise_stage instead.

uv run python scripts/production/model/train_linker_and_finalize.py publish `
  --source-bundle-root path/to/official_linker_source_bundle `
  --target-json "$RunRoot/release_inputs/incremental_linker_training_target.json" `
  --pairwise-model-path "$PairwiseModel" `
  --output-dir "$RunRoot/production_linker_vX.Y" `
  --publish-to "$RunRoot/release_candidate/production_model_vX.Y"
```

`--source-bundle-root` is the official linker source-bundle root, not a generic
benchmark Arrow directory. It must contain `bundle.json`, labels, candidate
members, leakage-safe split assignments and summaries, featureless support
tables, and the nested canonical Arrow manifests inventoried by the release
record. The target JSON must live in an immutable input directory outside the
fresh `--output-dir`.

The command saves and reloads the exact trained, calibrated, and evaluated
linker under
`$RunRoot/production_linker_vX.Y/incremental_linker_artifact/`.
When `--publish-to` is present, it infers the bundle version from the pairwise
stage, requires the destination basename to agree, and atomically publishes a
complete bundle to that fresh destination. The pairwise stage remains
unchanged. The final bundle contains:

```text
production_model_vX.Y/
  clusterer.json
  pairwise/
    main.lgb
    nameless.lgb
    main_prediction_fixture.json
    nameless_prediction_fixture.json
  incremental_linker/
    booster.lgb
    metadata.json
  reproducibility/
    pairwise_training_config.json
    pairwise_training_summary.json
    incremental_linker_training_target.json
  manifest.json
```

Linker training has one feature path: it materializes a fresh Arrow/Rust
feature bundle under the requested output directory, then trains from that
bundle. The feature-bundle destination must not already exist. Use the
`materialize` command with `--limit-rows N`, optionally with `--tables` or
`--datasets`, for a bounded smoke run before approving `candidate` or
`publish`.
Use `preflight` to validate the currently implemented target
feature/parameter/metric fields, pairwise/name-count bindings, source tables,
and fresh output paths without creating the output directory. It does not yet
enforce B20's target lifecycle fields. Selector-based runs are
materialization-only, and zero-row, unknown-selector, mixed-count-generation,
pairwise/count-binding, or split-identity overlap fails before publication.
Production policy is fixed in code; the script has no hyperparameter-search or
policy-tuning CLI. Current preflight is not yet the complete B08/B10/B19
source-path and byte-inventory gate.

The `candidate` command writes `candidate_target.json`, retains the exact
evaluated learned artifact, and writes deterministic query-level predictions
with a complete digest inventory; it cannot publish. B20 still requires an
atomic no-retraining candidate-to-production lifecycle transition that
preserves learned bytes and candidate ancestry.

Only after the runbook's lifecycle, one-shot evaluation, and release gates have
passed do users load the complete model with:

```python
from s2and.production_model import load_production_model

clusterer = load_production_model("/path/to/production_model_vX.Y")
```

There is no implicit default model. Runtime callers must pass the complete
bundle path; pairwise-only stages are accepted only by internal training code.

## Arrow Release Validation

For local release-root smoke checks that do not touch S3 or scan large Arrow
tables, run:

```powershell
uv run python scripts/verification/validate_local_arrow_release.py `
  --release-root s2and/data
```

This verifies manifest checksums, required local files, raw-planner batch-index
paths, replay-bundle manifest references, and `name_counts_index/manifest.json`
targets. Use `scripts/convert_to_arrow.py validate --dataset-dir ...` for
deeper per-dataset Arrow schema/table validation.

## Count Artifacts

The `counts/` scripts are guarded producers for production count artifacts:

- `counts/generate_name_counts.py` writes a content-bound,
  provenance-carrying immutable
  `name_counts_index/` into a previously absent target. It requires an explicit
  source snapshot, verifies selected-row content, and supports bounded fixture
  runs before any authorized warehouse run. The writer builds the complete
  directory in a temporary sibling and publishes it with one rename. Python
  and Rust runtime paths share that verified mmap index; neither unpickles nor
  retains the full dictionaries. Models compare the exact manifest SHA-256
  before feature work. Regeneration uses a new output directory; there is no
  in-place overwrite mode.
- `counts/generate_orcid_name_prefix_counts.py` writes canonical unordered
  ORCID prefix pairs directly to `first_k_letter_counts_from_orcid.json` with
  one adjacent `.manifest.json`. That manifest is the single authority for
  normalization, pair semantics, source provenance, tuple binding, generator
  parameters, cardinalities, and the data SHA-256. The producer reloads the
  staged pair before publishing its fresh output directory. The canonical
  tuple input and expected SHA-256 are explicit, and the reviewed
  `max_names_per_orcid` guard is checked before quadratic pair expansion.
  Install its JSON serializer with `uv sync --extra orcid-counts`.

Both scripts are import-safe without the internal warehouse package. Module
execution is the supported interface:

```powershell
uv run --no-sync python -m scripts.production.counts.generate_name_counts --help
uv run --no-sync python -m scripts.production.counts.generate_orcid_name_prefix_counts --help
```

Use `--dry-run` or a small fixture: `--fixture-input` for the name-count
producer and `--input-json` for the ORCID producer. A full internal query requires
`--run-full`, an explicit `--output-dir`, a reviewed `--source-snapshot-id`,
and `--guardrails-json`. A reviewed name-count guardrail file supplies
`min_source_rows`, `max_source_rows`, `min_keys_per_mapping`, and
`max_keys_per_mapping`; an ORCID guardrail file supplies `min_source_rows`,
`max_source_rows`, `max_names_per_orcid`, `min_orcid_pair_keys`, and
`max_pair_keys`. The warehouse query includes `LIMIT max_source_rows + 1`,
which bounds the client result but not the warehouse scan. `--limit` is
fixture-only. Pairwise production preflight rejects fixture provenance.

The snapshot ID is currently a caller-supplied label, not independent proof of
warehouse snapshot identity, and the ORCID `selected_rows_sha256` is computed
after rows reach the client. B27 and B28 must bind both full producers to
verifiable warehouse snapshot/query-result evidence before release use.
The code-only checkout declares neither ORCID file as package data. Stage 3
promotes the reviewed JSON and manifest and adds both declarations in one
commit; distribution verification derives its required inventory from those
declarations.

Full warehouse generation and model training are deliberately not part of the
local verification suite.
