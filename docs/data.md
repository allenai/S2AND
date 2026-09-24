# Data and Models

This document covers dataset download, public artifact layout, and
`path_config.json`. Canonical-name rules live in
[canonical_names.md](canonical_names.md).

> **Update at v1.3 release:** the currently published Arrow release and shared
> name counts predate public format `1`. They are not a compatible production
> release unit for this branch, and current validation rejects them. At
> release, replace the download commands below with the published v1.3 data
> location and size. See the [v1.3 release runbook](release.md).

## Dataset download

Download the currently published historical Arrow release into `s2and/data/`
only for migration, research, or source-data work:

```bash
uvx --from awscli aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release-arrow s2and/data/
```

Expected size is about `10.1 GiB`. Its manifests predate public format `1`, so
current runtime/release validation deliberately rejects it. A v1.3 production
root must be regenerated through the release runbook.

> **Update at v1.3 release:** add the published v1.3 public-data root here as
> the primary download, and label the command above as the historical release.

Download the legacy JSON/pickle S2AND release only when you need paper-era
JSON/pickle inputs for canonical S2-block `ANDData` workflows. The current API
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
public format `1`, so it is now a legacy historical input and
is rejected by strict v1.3 validation. Regenerate the replay bundle before
linker training as required by the
[v1.3 runbook](release.md#stage-2-build-training-and-evaluation-data).

The Arrow release stores runtime signatures, papers, paper authors, and SPECTER
rows as Arrow IPC files. It intentionally does not duplicate legacy `raw/`,
`embeddings/`, or precomputed `features_corrected/` directories.

Both Arrow/Rust inference and Python `ANDData` consume the shared
`name_counts_index/`. Python callers pass `NAME_COUNTS_INDEX_PATH` or an open
`NameCountsIndex` handle. The SHA-256 of the validated native manifest is the
publication and model identity. Its manifest contains exactly
`kind: "s2and_name_counts"`, `format_version: 1`, and the byte count plus
SHA-256 for each fixed binary role. One self-contained published root contains
one index; its benchmark and replay dataset manifests reference that shared
directory.

## Production model bundles

No default production model is distributed; evaluation and validation tools
take an explicit bundle path. The v1.3 model is an external bundle, not package
data. Its layout and manifest are specified in
[production_inference.md](production_inference.md#complete-model-bundles), and
[release.md](release.md) owns how it is trained, calibrated, and evaluated.

The reviewed 53-feature linker target is retained as
`tests/fixtures/incremental_linker_training_target.json`. Prediction logic does
not consume it, but finalization embeds the reviewed target in the complete
bundle and the model manifest binds its checksum.

The previous promoted-linker replay data is published under the Arrow release
prefix. Use the standalone download only to audit historical inputs; do not use
it as the v1.3 source bundle without regenerated assignments, Arrow data, and
pairwise-derived inputs. Pass a source bundle explicitly with
`--source-bundle-root` to `scripts/production/model/train_linker_and_finalize.py`;
the release command has no implicit replay-bundle default.

## Configuring `s2and/data/path_config.json`

`s2and/data/path_config.json`, or the file named by the `S2AND_PATH_CONFIG`
environment variable, sets `main_data_dir`. Its one consumer is the default
name-count index path for classic `ANDData`,
`<main_data_dir>/name_counts_index` (`s2and.consts.NAME_COUNTS_INDEX_PATH`).
Arrow routes and the production scripts take explicit paths instead.

Example:

```json
{
  "main_data_dir": "/absolute/path/to/your/s2and/data"
}
```

Guidance:

- Set `main_data_dir` to the directory that contains `name_counts_index/`.
- The checked-in file keeps the placeholder value
  `"absolute path of wherever you downloaded the data to"`. Only that exact
  value falls back to the package's `s2and/data/` directory (with a warning);
  any other value must be an existing directory.
- Archived AI2-internal scripts under `scripts/archive/` read an extra
  `internal_data_dir` key; add it to your own config if you rerun one. Current
  code does not use it.

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

Dataset manifests require `kind: "s2and_arrow_dataset"`,
`format_version: 1`, portable `paths`, and a flat `files` inventory mapping
each immutable semantic role to `byte_count` and lowercase `sha256`.
Request-time query/seed sidecars are not part of that inventory. Older local
Arrow directories without this contract must be reconverted; relabeling their
manifests is not sufficient.

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
absent. Present but invalid Arrow artifacts, including the historical
pre-format-1 release, raise an error instead of falling back to JSON. JSON mode
supports:

- mini-dataset naming such as `<dataset>_papers.json`
- plain fixture naming such as `papers.json`

See [production_inference.md](production_inference.md) for the minimal inference
input contract, and [training.md](training.md) for training-mode dataset
requirements.
