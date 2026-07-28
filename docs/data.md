# Data and Models

This document covers dataset download, public artifact layout, and
`path_config.json`.

> **1.0.0 release status (2026-07-27):** the currently published Arrow
> release and shared name counts predate public format `1`. They are not a
> compatible production release unit for this branch. See the
> [v1.3 release runbook](release.md).

## Canonical name contract

`s2and.text.canonicalize_name_parts` and its Rust equivalent are the
authorities. Given raw `first`, `middle`, and `last` values, they:

1. normalize Unicode spacing and remove soft hyphen and zero-width joiner;
2. normalize apostrophe-like marks, transliterate, lowercase, and delete
   apostrophes rather than making token boundaries;
3. treat supported Unicode dash variants as one separator;
4. replace remaining nonletters with spaces and collapse whitespace;
5. drop at most one leading title from first name, except `md`;
6. retain a dash-bound first-name group, otherwise keep the first token and
   spill remaining first-name tokens into middle; and
7. normalize last name independently, preserving spaces and surname particles.

Examples: `Anne-Marie Claire` becomes first `anne marie`, middle `claire`;
apostrophe variants of `O'Connor` become `oconnor`; and `Ou-Yang` and
`Ou Yang` both become last `ou yang`.

`s2and.text.canonical_name_count_keys` emits no sentinel keys. It emits
`first` only when first has more than one character, `last` when present,
`first_last` when both are informative, and `last_first_initial` when both
components exist.

At comparison time, `same_prefix_tokens` is symmetric: every aligned token in
the shorter canonical first name must prefix its counterpart, and empty input
is missing evidence. Alias tuples are unordered. Canonical last names retain
spaces; only documented count/block projections compact them, while
`canonical_lasts_equivalent` treats dash/space variants as equivalent.

The live authorities are `s2and.text`, its Rust implementation, and
[the frozen examples](../tests/fixtures/canonical_name_examples.json).
Tuple and ORCID runtime rules live in
[production_inference.md](production_inference.md); Arrow and name-count
formats live in [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md).
The retained manual
tuple review is [release_evidence/name_tuple_legacy_adjudication_v1.md](release_evidence/name_tuple_legacy_adjudication_v1.md).

## Dataset download

Download the currently published historical Arrow release into `s2and/data/`
only for migration, research, or source-data work:

```bash
uvx --from awscli aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release-arrow s2and/data/
```

Expected size is about `10.1 GiB`. Its manifests predate public format `1`, so
current runtime/release validation deliberately rejects it. A v1.3 production
root must be regenerated through the release runbook.

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

The obsolete production pickles and previous v1.21 repository bundle have been
removed. Their history remains available from the compatible prior release and
Git history, but no current command accepts their former paths.

No default production model declaration is distributed during cutover.
Evaluation and validation tools must receive an explicit model bundle path.
The v1.3 model is an external bundle, not package data. Its manifest records
`kind: "s2and_model"`, `release_version: "1.3"`, exact
`generated_by_runtime`, EPS calibration state, and the runtime-file checksum
inventory.

New production releases use fresh native bundle directories. The component
entry points are `scripts/production/model/train_pairwise.py` and
`scripts/production/model/train_linker_and_finalize.py`, with release-only
calibration/evaluation in `scripts/production/model/release_pairwise.py`.
The v1.3 sequence trains the final pairwise boosters, selects EPS on validation
data, materializes the calibrated linker inputs, fits one fresh linker, reloads the
complete bundle, and evaluates it as described in
[release.md](release.md). Do not create new production
pickles.

The reviewed 53-feature linker target is retained as
`tests/fixtures/incremental_linker_training_target.json`. Prediction logic does
not consume it, but finalization embeds the reviewed target in the complete
bundle and the model manifest binds its checksum.

The previous promoted-linker replay data is published under the Arrow release
prefix. Use the standalone download only to audit historical inputs; do not use
it as the v1.3 source bundle without regenerated assignments, Arrow data, and
pairwise-derived inputs.

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
absent. JSON mode supports:

- mini-dataset naming such as `<dataset>_papers.json`
- plain fixture naming such as `papers.json`

See [production_inference.md](production_inference.md) for the minimal inference
input contract, and [training.md](training.md) for training-mode dataset
requirements.
