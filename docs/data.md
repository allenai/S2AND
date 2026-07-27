# Data and Models

This document covers dataset download, checked-in model artifacts, and `path_config.json`.

> **Canonical-v2 migration status (2026-07-24):** the published Arrow release,
> shared name counts, and checked-in v1.21 model are legacy inputs. They are not
> a compatible production release unit for this branch. Canonical counts and the
> v1.3 bundle are pending; see the
> [v1.3 release runbook](1_3_release_todo.md). Here, “v1.3” names the coordinated
> model/data release. The Python/Rust package version is still an explicit
> release decision.

## Dataset download

Download the Arrow-native production runtime release into `s2and/data/` for
Rust/Arrow prediction and evaluation:

```bash
uvx --from awscli aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release-arrow s2and/data/
```

Expected size is about `10.1 GiB`. The currently published release root contains
benchmark dataset directories, the legacy shared `name_counts_index/`, the
legacy `production_model_v1.21/`, and the promoted-linker replay bundle.

Download the legacy JSON/pickle S2AND release only when you need paper-era
JSON/pickle inputs for canonical S2-block `ANDData` workflows. Canonical-v2
does not restore the release's original/given-block partition inside
`ANDData`:

```bash
uvx --from awscli aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release s2and/data/
```

Expected legacy release size is about `55.5 GiB`.

The promoted-linker replay subbundle can also be downloaded by itself:

```bash
uvx --from awscli aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release-arrow/s2and_and_big_blocks_linker_dataset_20260525 s2and/data/s2and_and_big_blocks_linker_dataset_20260525
```

`s2and/data/s2and_and_big_blocks_linker_dataset_20260525` is the conventional
local name for the previously published replay subbundle. Its manifests predate
the canonical generation contract, so it is now a legacy historical input and
is rejected by strict v1.3 validation. B09 in
[1_3_release_blockers.md](1_3_release_blockers.md) requires a regenerated
replay bundle before linker training.

The Arrow release stores runtime signatures, papers, paper authors, and SPECTER
rows as Arrow IPC files. It intentionally does not duplicate legacy `raw/`,
`embeddings/`, or precomputed `features_corrected/` directories.

Both Arrow/Rust inference and Python `ANDData` consume the shared
`name_counts_index/`. Python callers pass `NAME_COUNTS_INDEX_PATH` or an open
`NameCountsIndex` handle. The native manifest is the publication and model
identity; its `source_provenance` retains warehouse audit lineage.

The previous production model source bundle is checked into this repo under
`s2and/data/production_model_v1.21/`. Canonical-v2 rejects it; it is retained
only as an explicitly named migration and historical validation input until
v1.3 replaces it.

## Previous production model bundle

The previous production model is a native bundle directory:

- `s2and/data/production_model_v1.21/manifest.json`
- `s2and/data/production_model_v1.21/clusterer.json`
- `s2and/data/production_model_v1.21/pairwise/main.lgb`
- `s2and/data/production_model_v1.21/pairwise/nameless.lgb`
- `s2and/data/production_model_v1.21/pairwise/metadata.json` (legacy v1 only)
- `s2and/data/production_model_v1.21/pairwise/main_prediction_fixture.json`
- `s2and/data/production_model_v1.21/pairwise/nameless_prediction_fixture.json`
- `s2and/data/production_model_v1.21/incremental_linker/booster.lgb`
- `s2and/data/production_model_v1.21/incremental_linker/metadata.json`
- `s2and/data/production_model_v1.21/reproducibility/incremental_linker_training_target.json`

See [production_inference.md](production_inference.md) for what each file is
for.

The source bundle is excluded from package data, the obsolete v1.0-v1.2 model
pickles have been removed, and no default production model declaration is
distributed during cutover. Evaluation and validation tools must receive an
explicit model bundle path. The v1.3 decision is an immutable external artifact,
not a packaged default. B15 remains partial until the release-candidate
distribution verifier enforces that absence.

New production releases use immutable native bundle directories. The component
entry points are `scripts/production/model/train_pairwise.py` and
`scripts/production/model/train_linker_and_finalize.py`, with release-only
  calibration/evaluation in `scripts/production/model/release_pairwise.py`;
  stage, validate, and rename the complete bundle rather than mutating a live
  directory. They are not by themselves the full v1.3 protocol: EPS selection,
  a fresh linker fit against the calibrated pairwise bundle, no-second-fit
  complete-bundle assembly, sealed evaluation, protected approval, and
  exact-byte publication remain governed by
  [1_3_release_todo.md](1_3_release_todo.md). Do not create new production
  pickles.

The replay target for rebuilding/auditing the promoted incremental linker lives
at:

```text
s2and/data/production_model_v1.21/reproducibility/incremental_linker_training_target.json
```

Prediction logic does not consume it, but bundle load validation includes its
manifest checksum. It records feature order and training params for the replay
script.

The previous promoted-linker replay data is published under the Arrow release
prefix. Use the standalone download only to audit historical inputs; do not use
it as the v1.3 source bundle without the B07-B10/B19 regeneration, assignment,
and inventory work.

Pass the downloaded source bundle explicitly with `--source-bundle-root` to
`scripts/production/model/train_linker_and_finalize.py`; the release command
has no implicit replay-bundle default.

## Configuring `s2and/data/path_config.json`

Some scripts look up the main data root through `s2and/data/path_config.json`
or the `S2AND_PATH_CONFIG` environment variable. This config points at the
downloaded benchmark dataset root; it is separate from the package data checked
in under `s2and/data/`.

Example:

```json
{
  "main_data_dir": "absolute path to your downloaded S2AND data",
  "internal_data_dir": ""
}
```

Guidance:

- Set `main_data_dir` to the directory containing your downloaded S2AND datasets.
- `internal_data_dir` is only relevant for internal AI2 workflows and can be left empty.
- If your data lives in this repo's `s2and/data/` directory, the default placeholder config already resolves there.

## Dataset file expectations

Arrow production/eval workflows use each dataset's `manifest.json` to resolve:

- `signatures.arrow`
- `papers.arrow`
- `paper_authors.arrow`
- one selected embedding table under manifest key `specter`; production and
  evaluation bundles use physical `specter2.arrow` (an explicit historical
  SPECTER1 research-training bundle may instead use `specter.arrow`)
- raw-planner `*_batch_index.bin` sidecars
- shared `name_counts_index/`
- eval-only clusters JSON when metrics are requested

Production manifests must declare `canonical_v2` and include the canonical
content-addressed `artifact_generation` inventory for every immutable table,
batch index, and count-index manifest. Request-time query/seed sidecars are not
part of that generation. Older local Arrow directories without this inventory
are intentionally rejected and must be reconverted; relabeling their manifests
is not sufficient.

Legacy-input workflows use the standard S2AND JSON files below, but current
`ANDData` always groups them by `author_info.block`; it ignores any
`author_info.given_block` field:

- signatures
- papers
- clusters
- optional cluster seeds
- SPECTER embeddings

The tutorial script supports Arrow by default when a dataset manifest exists,
and JSON only when `--input-format json` is requested or Arrow artifacts are
absent. JSON mode supports:

- mini-dataset naming such as `<dataset>_papers.json`
- plain fixture naming such as `papers.json`

See [production_inference.md](production_inference.md) for the minimal inference
input contract, and [training.md](training.md) for training-mode dataset
requirements.
