# Production Release Artifacts

Status date: 2026-07-26.

Scripts in this directory create or validate release artifacts. The
model artifact is a native `production_model_vX.Y/` directory. v1.3 uses an
immutable external model; B15 remains partial only for distribution-verifier
enforcement of that fixed decision. Do not create production pickles.

This is a component command reference, not the v1.3 execution order. No full
warehouse query or training run is currently authorized by this document.
Follow [../../docs/1_3_release_todo.md](../../docs/1_3_release_todo.md) for the
bounded preflights, EPS stage, direct complete-bundle finalization, one
evaluation report, approval, and exact-byte publication; use
[../../docs/1_3_release_blockers.md](../../docs/1_3_release_blockers.md) for
current blockers. Examples below use `X.Y` as the model-bundle version; the
coordinated Python/Rust package version is a separate open decision.

## 1. Train Pairwise

```powershell
$RunRoot = "D:/local-unsynced/s2and-vX.Y"
$NameCountsIndexRoot = "path/to/name_counts_index"

uv run python scripts/production/model/release_pairwise.py `
  preflight-training-inputs `
  --manifest path/to/pairwise_inputs_manifest.json `
  --expected-manifest-sha256 REVIEWED_INPUTS_SHA256 `
  --output-plan "$RunRoot/pairwise_training_plan.json"

New-Item -ItemType Directory -Path "$RunRoot/matrix-work" | Out-Null

uv run python scripts/production/model/train_pairwise.py `
  --production-version X.Y `
  --training-plan "$RunRoot/pairwise_training_plan.json" `
  --expected-training-plan-sha256 REVIEWED_PLAN_SHA256 `
  --name-counts-index-root "$NameCountsIndexRoot" `
  --matrix-work-dir "$RunRoot/matrix-work" `
  --output-dir "$RunRoot/pairwise_stage/production_model_vX.Y" `
  --validation-pairs-size REVIEWED_VALIDATION_PAIRS_SIZE `
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

Pairwise training has one release-only mode. It requires `--run-full`, the
digest-bound test-path-free plan from `preflight-training-inputs`, and an
external name-count index. Canonical tuple and ORCID artifacts come from the
packaged artifact authority. There is no dataset selector, smoke mode,
feature-cache mode, or preflight-only mode.

The operator must confirm that `--matrix-work-dir` is local, empty, unsynced,
and large enough. The trainer checks it before loading inputs, verifies actual
matrix write requirements, and requires finite metrics.

`preflight-training-inputs` validates sealed test members and strips their
paths from the plan; the trainer never opens them. Training provenance is
covered by reproducibility members in the final complete-model manifest.

## 2. Select and freeze EPS

Do not train the v1.3 linker directly from the raw pairwise stage. Use the
single validation-only `calibrate-eps` command driven by the frozen
EPS rule in `release_spec.json`; the grid is never typed into an operator
command.
It uses packaged tuple/ORCID artifacts plus the external name-count index and
always writes one fresh calibrated pairwise bundle whether or not the selected
EPS changes. Use the exact tested command in Stage 4 of the release runbook.
There is no second EPS-finalization command or separate release authority.

## 3. Train Linker and finalize the complete model

The direct finalization command is the mandatory linker retrain after EPS
freezes the pairwise bundle. It rematerializes features, fits once, writes a
complete v5 bundle, reloads it, and evaluates those exact bytes. A linker bound
to another pairwise manifest cannot be reused.

```powershell
$PairwiseModel = "$RunRoot/pairwise_calibrated/production_model_vX.Y"

uv run python scripts/production/model/train_linker_and_finalize.py `
  --source-bundle-root path/to/official_linker_source_bundle `
  --target-json "$RunRoot/inputs/targets/incremental_linker_training_target.json" `
  --pairwise-model-path "$PairwiseModel" `
  --name-counts-index-root "$NameCountsIndexRoot" `
  --output-dir "$RunRoot/linker_release" `
  --n-jobs REVIEWED_N_JOBS `
  --total-ram-bytes REVIEWED_LINKER_RAM_BYTES
```

The only durable outputs are
`$RunRoot/linker_release/production_model_vX.Y` and
`$RunRoot/linker_release/linker_evaluation_report.json`.

`--source-bundle-root` is the official linker source-bundle root, not a generic
benchmark Arrow directory. It must contain `bundle.json`, labels, candidate
members, leakage-safe split assignments and summaries, featureless support
tables, and the nested canonical Arrow manifests inventoried by the release
record. The target JSON must live in an immutable input directory outside the
fresh `--output-dir`.

Feature materialization and staging are temporary implementation details. The
fresh `--output-dir` receives only the complete model and linker evaluation
report after validation succeeds. Production policy is fixed in code; there is
no lifecycle, selector, search, or tuning CLI.

The bundle retains
`reproducibility/incremental_linker_training_target.json`, and linker metadata
binds it plus the final pairwise contract. Any pairwise or EPS change requires
another fit. The command cannot approve publication; the runbook's one
evaluation authority applies all release-spec gates.

`release_pairwise.py evaluate-release` verifies the pinned nine-member
`s2and_release_evidence_manifest_v1`, applies the frozen gates, and writes the
one `s2and_release_evaluation_report_v1` report. Its exact command is in Stage 6
of the release runbook.

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
verifiable warehouse snapshot/query-result evidence before release use. The
current full-query implementation still imports retired `pys2`; it is not a
v1.3 release path and must be replaced under B28.
The current pre-release tree declares neither ORCID file as package data.
Stage 1 adds the reviewed JSON and manifest plus both declarations in the final
release commit; distribution verification derives its required inventory from
those declarations.

Full warehouse generation and model training are deliberately not part of the
local verification suite.
