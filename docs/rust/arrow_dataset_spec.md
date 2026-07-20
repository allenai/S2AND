# Arrow Dataset Specification

Status date: 2026-07-10

This document defines the Arrow artifact contract for engineers assembling
datasets for direct Rust S2AND routes. These artifacts are used by
`Clusterer.predict_from_arrow_paths(...)`,
`Clusterer.predict_incremental_from_arrow_paths(...)`, and
`build_training_anddata_from_arrow(...)`. Classic `Clusterer.predict(...)` and
`Clusterer.predict_incremental(...)` operate on `ANDData` through Python.

Production Arrow is a raw runtime input contract, not a serialized
`ANDData(preprocess=True)` cache. The goal is feature parity with the current
`ANDData(preprocess=True)` representation after the local runtime preprocesses
the Arrow rows. Arrow text/name columns are preprocessing inputs, not
authoritative precomputed feature values: Rust normalizes titles, venues,
journals, signature names, paper-author names, and computes language-dependent
paper state while building the scoring view.

---

## Summary

Each Arrow dataset is a directory of Arrow IPC file-format tables plus a
manifest. The hot path reads these files directly from Rust or through
memory-mapped Arrow readers.

Required for full-block prediction:

- `signatures.arrow`
- `papers.arrow`
- `paper_authors.arrow`
- `specter.arrow` or `specter2.arrow` when the model uses
  `embedding_similarity`

Required in addition for seeded prediction or incremental prediction promoted
through Arrow:

- request-local `query_signatures.arrow` for raw incremental candidate planning
- `cluster_seeds.arrow`

Optional for seeded prediction or incremental prediction promoted through Arrow:

- `cluster_seed_disallows.arrow` when pairwise seed disallow constraints exist

Required when incremental input contains altered claimed profiles:

- `altered_cluster_signatures.arrow`

Offline evaluation datasets may also include:

- `<dataset>_clusters.json`

Do not create per-dataset `name_pairs.arrow` files for production datasets.
Name aliases are a shared runtime resource.
Do not include `name_pairs` or `name_tuples` path keys in production manifests
or runtime path bundles.

---

## Layout

Preferred on-disk layout:

```text
<arrow_root>/
  manifest.json
  <dataset>/
    manifest.json
    signatures.arrow
    papers.arrow
    paper_authors.arrow
    specter.arrow
    specter2.arrow
    signatures.signatures_batch_index.bin
    papers.papers_batch_index.bin
    paper_authors.paper_authors_batch_index.bin
    specter.specter_batch_index.bin
    specter2.specter_batch_index.bin
    query_signatures.arrow
    cluster_seeds.arrow
    cluster_seed_disallows.arrow
    altered_cluster_signatures.arrow
    <dataset>_clusters.json
```

Notes:

- `query_signatures.arrow` is the request-local query table consumed by the
  raw incremental planner. Runtime helpers may materialize it from existing
  Python request arguments; producers that already have a typed request should
  pass it under the `query_signatures` path key.
- `cluster_seeds.arrow` is one accepted seed source for seeded/incremental
  datasets. It can be omitted for unseeded full prediction, offline eval, and
  incremental production requests that provide seed assignments through another
  normalized request/dataset mapping such as `dataset.cluster_seeds_require`.
  Promoted Rust incremental prediction still requires a seed source; when the
  source is not a physical Arrow sidecar, the runtime materializes a
  request-local `cluster_seeds.arrow`.
- `cluster_seed_disallows.arrow` preserves pairwise seed disallow constraints.
  Hand-authored artifacts can omit it when the request has no seed disallows;
  converters may emit an empty table instead. An explicit path must exist when
  present.
- When using `scripts.arrow_conversion_helpers.write_feature_block_arrow_from_anddata(...)` to publish physical
  seeded/incremental seed sidecars, pass `include_empty_cluster_seeds=True` so
  empty seed/disallow tables are still emitted.
- `altered_cluster_signatures.arrow` is required for incremental datasets whose
  seed clusters include altered claimed profiles. When an in-memory
  `ANDData.altered_cluster_signatures` request value is present it is
  authoritative; otherwise the Arrow file is the producer-owned request
  artifact for this condition. `altered_cluster_signatures.txt` is not a valid
  production Arrow sidecar; it remains only for older fixtures and
  ANDData-compatible training tooling.
- `<dataset>_clusters.json` is ground truth for offline evaluation only. It is
  not part of production inference scoring.
- `specter.arrow` is the SPECTER v1 embedding table. `specter2.arrow` is the
  SPECTER v2 embedding table. Include whichever model family will be used; eval
  bundles usually include both.
- If embeddings are requested but no block papers have embeddings, emit a valid
  zero-row `specter.arrow` rather than omitting the table so production
  prediction degrades through missing-vector features.
- The Arrow files must be Arrow IPC file format, not Arrow stream format. The
  current writer uses `pyarrow.ipc.new_file(...)`; readers use
  `pyarrow.ipc.open_file(...)` and memory maps.

The machine-readable column contract lives at
`s2and/arrow_schema_contract.json`. It is a parity guard for producer/consumer
drift; runtime readers still enforce their local validation rules directly.

The explicit `*_from_arrow_paths` APIs receive this mapping directly. The fixed
Rust-training constructor stores the validated mapping once as the immutable
`dataset.arrow_paths` authority. Rust routes do not infer sibling
`<data_root>_arrow/<dataset>` directories or declare optional sidecars merely
because files are present on disk. The mapping uses these keys:

| Key | Meaning |
|---|---|
| `signatures` | Path to `signatures.arrow` |
| `papers` | Path to `papers.arrow` |
| `paper_authors` | Path to `paper_authors.arrow` |
| `specter` | Path to the embedding table selected for the current model, even if the file is physically named `specter2.arrow` |
| `query_signatures` | Request-local path to `query_signatures.arrow` for raw incremental candidate planning |
| `cluster_seeds` | Optional path to `cluster_seeds.arrow` for incremental/seeded prediction; required only when this sidecar is the seed source |
| `cluster_seed_disallows` | Optional path to `cluster_seed_disallows.arrow` for pairwise seed disallow constraints |
| `altered_cluster_signatures` | Path to `altered_cluster_signatures.arrow` when altered claimed profiles are present |
| `clusters` | Path to eval-only ground-truth clusters JSON |
| `name_counts_index` | Required manifest-declared shared/global name-count index directory when the selected model uses `name_counts` |
| `signatures_batch_index` | S2AND-generated lookup index for `signatures.arrow`; required for production filtered reads |
| `papers_batch_index` | S2AND-generated lookup index for `papers.arrow`; required for production filtered reads |
| `paper_authors_batch_index` | S2AND-generated lookup index for `paper_authors.arrow`; required for production filtered reads |
| `specter_batch_index` | S2AND-generated lookup index for the selected embedding path passed as `specter`; required for production filtered reads when embeddings are used. The sidecar filename follows the selected file stem, for example `specter.specter_batch_index.bin` or `specter2.specter_batch_index.bin` |

---

## Large-Block Physical Layout

The schema above is the semantic artifact contract. Large-block incremental
serving also needs a physical layout that makes indexed raw candidate planning
cheap. This layout is not required for correctness, but it is required for the
scalable performance path on large blocks such as common family-name blocks.

For large block artifacts, producers should write the lookup tables below as
Arrow IPC file-format files with bounded record batches. Do not write these
tables as one giant record batch when the row count exceeds the limit.

| Table | Lookup key | Maximum rows per IPC record batch |
|---|---|---:|
| `signatures.arrow` | `signature_id` | 16,384 |
| `papers.arrow` | `paper_id` | 16,384 |
| `paper_authors.arrow` | `paper_id` | 16,384 |
| `specter.arrow` / `specter2.arrow` | `paper_id` | 2,048 |

The smaller request-scoped tables do not need a random-access physical layout:

| Table | Layout guidance |
|---|---|
| `query_signatures.arrow` | Read fully by the raw planner; no bounded-batch requirement. |
| `cluster_seeds.arrow` | Read fully by the raw planner; no bounded-batch requirement. |
| `cluster_seed_disallows.arrow` | Read fully when present; no bounded-batch requirement. |
| `altered_cluster_signatures.arrow` | Read as request metadata; bounded batches do not address altered-profile pre-splitting cost. |

Implementation notes for producers:

- Use Arrow IPC file format, not stream format.
- Prefer S2AND's `write_arrow_ipc_table(..., max_record_batch_rows=<limit>)`
  helper. Independent PyArrow writers should use `pyarrow.ipc.new_file(...)`
  and `writer.write_table(table, max_chunksize=<limit>)`, then verify the
  emitted record batches with `arrow_ipc_physical_layout(...)` or an equivalent
  check.
- Preserve `signatures.arrow` row order. Record-batch boundaries must not
  change row contents or row order.
- Keep `paper_authors.arrow` grouped by `paper_id`, then ordered by `position`
  where practical. This improves locality when all authors for a paper are read.
- One record batch is acceptable only when
  `row_count <= maximum rows per IPC record batch`.
- For embedding files, the 2,048-row limit is intentionally lower because each
  row contains a dense vector. If the embedding dimension changes enough that a
  batch becomes much larger than roughly 8-16 MiB, lower this limit rather than
  raising it.

S2AND binary batch indexes are derived artifacts over the final Arrow files.
The preferred handoff is for producers to supply bounded Arrow IPC files and
for an S2AND prep step to generate these indexes. Producers may include indexes
only when they are generated with S2AND tooling, such as
`s2and.incremental_linking.feature_block.write_raw_arrow_batch_lookup_indexes`.
Do not hand-write the binary format in an independent pipeline. Do not generate
these indexes before a later rewrite or deployment copy that changes the source
Arrow file metadata; regenerate the indexes from the final files in their
serving location.

Every script that produces S2AND runtime Arrow artifacts should use the shared
writers instead of open-coding the table or sidecar formats:

- `scripts.arrow_conversion_helpers.write_feature_block_arrow_from_anddata(...)` or
  `write_feature_block_arrow_tables(...)` for semantic Arrow IPC tables.
- `write_raw_arrow_batch_lookup_indexes(...)` after the final table write for
  raw-planner sidecars.
- `raw_planner_arrow_physical_layout(...)` for manifest/report layout metrics.

Recommended sidecar filenames are stem-qualified:

```text
signatures.signatures_batch_index.bin
papers.papers_batch_index.bin
paper_authors.paper_authors_batch_index.bin
specter.specter_batch_index.bin
specter2.specter_batch_index.bin
```

The double stem is intentional: the first stem identifies the Arrow file and the
trailing `<table>_batch_index` stem matches the manifest path key.

When both `specter.arrow` and `specter2.arrow` are present, write one embedding
index per file. At runtime, the selected embedding file is passed under the
`specter` path key, and S2AND uses the adjacent
`<embedding-stem>.specter_batch_index.bin` sidecar when present.

The batch-index format is S2AND-owned. Current writers and readers require
`arrow_batch_lookup_index` / `S2ABI002`, which records the key-column hash and
full-file source fingerprint in addition to key-to-batch records. Each record maps a
64-bit FNV-1a hash of the lookup key to an IPC record-batch index; the Rust
reader verifies exact ids after loading the selected batches, so hash collisions
do not change results. The file body must contain exactly the header-declared
record count, records must be ordered by nondecreasing key hash, and every stored
batch index must be smaller than the Arrow file's IPC record-batch count.

---

## Runtime Input Semantics

Rows must provide the source values needed for the local Rust runtime to produce
the same feature view that S2AND would expose after normal preprocessing:

- `block_type="s2"`
- `name_tuples=None`
- `name_counts_index/` available when the selected model uses name-count features

`author_orcid` is optional. If the column is absent, every signature is treated
as having no ORCID evidence; all non-ORCID features and constraints remain
available. Name-count artifacts always use the canonical initial-character key.

Use the script-only `FeatureBlock` conversion writer as the reference
implementation for Arrow physical layout and for benchmark/replay bundles whose
inputs are derived from `ANDData`:
`scripts.arrow_conversion_helpers.write_feature_block_arrow_from_anddata`.
That writer returns table paths and does not write `manifest.json`. All in-repo
producers pass those paths through
`s2and.arrow_inputs.build_arrow_artifact_manifest(...)` and
`write_arrow_artifact_manifest(...)`, which own the portable paths,
normalization, immutable-generation inventory, and publication format.
Producer-specific metadata cannot override those canonical fields.
`scripts/convert_to_arrow.py` is the reference producer for deployable
dataset metadata and current batch-index sidecars.
`scripts/verification/compare_full_predict_arrow_parity.py` is the reference
bounded parity producer: it writes current batch-index sidecars, resolves a
canonical name-count index, and publishes an artifact-generation manifest for
its temporary Arrow bundle before validation. An independent assembly pipeline is fine, but
production producers should send source/raw text and name inputs plus the same
manifest contract as this document. Parity is measured after Rust preprocessing,
not by requiring producer-side Python preprocessing before Arrow construction.

Important parity details:

- Preserve source signature order. The current converter writes
  `signature_ids=list(dataset_obj.signatures)` for this reason.
- Store ids as strings, even if an upstream source stores numeric ids.
- Text/name fields should be source/raw values where practical. Rust owns the
  normalization, ngram, unidecode, name splitting, and language-detection work
  needed for production scoring.
- Keep `abstract` as an abstract-presence signal, not raw abstract text. The
  current `FeatureBlock` encoding writes `"Has Abstract"` when the preprocessed
  paper has an abstract and `""` otherwise.
- Include all paper-author rows needed for coauthor features.
- Do not include embedded name-count columns in `signatures.arrow`; use the
  shared `name_counts_index/` sidecar.

---

## Table Schemas

### `signatures.arrow`

One row per signature. Columns marked optional in the meaning text may be
omitted; when present, they must have the listed type.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id` | `string` | no | Stable signature id |
| `paper_id` | `string` | no | Referenced paper id |
| `author_first` | `string` | yes | Source author first-name field used as runtime preprocessing input |
| `author_middle` | `string` | yes | Source author middle-name field used as runtime preprocessing input |
| `author_last` | `string` | yes | Source author last-name field used as runtime preprocessing input |
| `author_suffix` | `string` | yes | Source author suffix field used as runtime preprocessing input |
| `author_affiliations` | `list<string>` | yes | Author affiliations; prefer empty list over null |
| `author_orcid` | `string` | yes | Optional column containing ORCID evidence when available |
| `author_position` | `int64` | yes | Author position on the paper |
| `author_block` | `string` | yes | S2 block key, needed for block reconstruction/eval |
| `author_email` | `string` | yes | Author email |
| `source_author_ids` | `list<string>` | yes | Upstream author ids |

Name-count values are intentionally not part of the signature table.

Migration warning: although this table currently permits null
`author_position`, full Rust featurization rejects it and correct coauthor
exclusion cannot be reconstructed without a focal position. The canonical
target is a required/non-null field after intended release datasets are audited
and repaired; see [../work_plan.md B14](../work_plan.md#b14-nullable-signature-author_position-has-contradictory-contracts).

### `papers.arrow`

One row per paper referenced by `signatures.arrow`. Columns:

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `paper_id` | `string` | no | Stable paper id |
| `title` | `string` | yes | Source/raw paper title text used as runtime preprocessing input |
| `abstract` | `string` | yes | Abstract-presence signal: `"Has Abstract"` or `""` |
| `venue` | `string` | yes | Venue text used as runtime preprocessing input |
| `journal_name` | `string` | yes | Journal text used as runtime preprocessing input |
| `year` | `int64` | yes | Optional publication year |
| `predicted_language` | `string` | yes | Optional cached/compatibility language override |
| `is_reliable` | `bool` | yes | Cached/compatibility reliability flag; required with a non-null `predicted_language` and paired with `language_reliability` |
| `language_reliability` | `float64` | yes | Cached detector confidence in `[0.0, 1.0]`; required with a non-null `predicted_language` and paired with `is_reliable` |

Production `papers.arrow` should keep source/raw title, venue, and journal
text. Consumers must not assume these text fields are already normalized. If
`predicted_language` is null, Rust detects language locally from the raw title.
If `predicted_language` is non-null, Rust treats it as a producer-owned
precomputed override. Such an override is complete only when `is_reliable` and
`language_reliability` are also non-null; consumers reject partial overrides.
`language_reliability` must be finite and in `[0.0, 1.0]`, and it must be exactly
`0.0` when `is_reliable` is `false`. Offline compatibility bundles may contain
the complete override triple, but production producers should leave all three
fields null unless the same approved local detector already produced them before
Arrow handoff.

### `paper_authors.arrow`

One row per paper-author child row. Required columns:

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `paper_id` | `string` | no | Referenced paper id |
| `position` | `int64` | no | Author position |
| `author_name` | `string` | no | Source paper-author name string used as runtime preprocessing input for coauthor features |

Rows should be ordered by `paper_id` then `position` where practical. Ordering is
not the identity contract, but stable ordering makes diffs and validation easier.

### `specter.arrow` and `specter2.arrow`

One row per embedded paper. Required columns:

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `paper_id` | `string` | no | Referenced paper id |
| `embedding` | `fixed_size_list<float32>[dimension]` | no | SPECTER vector |

All vectors in one file must have the same dimension, and `paper_id` values
must be unique. A missing embedding means there is no row for that `paper_id`;
do not represent missing vectors with a null `embedding` value. If the model
uses `embedding_similarity`, every paper referenced by `signatures.arrow` should
have an embedding row for the selected embedding version. Missing embeddings can
change scores and should fail validation unless the target model explicitly
permits them.

### `query_signatures.arrow`

Request-local query table for raw incremental candidate planning. The Rust
planner reads this table before candidate retrieval and uses it as the planner
query set and per-query view policy.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id` | `string` | no | Query signature id |
| `query_view` | `string` | no | Requested view: `auto`, `full`, or `initial_only` |
| `query_author` | `string` | no | Caller-visible query author text; empty string is allowed |

`signature_id` values must be unique and non-empty. `query_view` values must be
valid. The planner derives scoring-time author evidence from `signatures.arrow`
and validates a non-empty `query_author` against that derived query author.

### `cluster_seeds.arrow`

One accepted seed source for incremental/seeded prediction through the Arrow
promoted path. Optional for unseeded full prediction and for incremental
production requests that provide seed assignments through another normalized
request/dataset mapping. Promoted Rust incremental prediction requires some seed
source; if the caller provides a non-Arrow mapping, the runtime writes a
request-local `cluster_seeds.arrow` before entering raw Arrow retrieval.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id` | `string` | no | Seed signature id |
| `cluster_id` | `string` | no | Required seed component/cluster id |

Only required seed assignments are persisted here. Pairwise seed disallow
constraints are persisted separately in `cluster_seed_disallows.arrow`.
`signature_id` values must be unique, and `cluster_id` values must be non-empty
strings.

### `cluster_seed_disallows.arrow`

Optional for incremental/seeded prediction through the Arrow promoted path.
Omit the file when no seed disallows are present, or emit a valid empty table
when using a converter configured to keep seed/disallow tables explicit. An
explicit path must exist when present.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id_1` | `string` | no | First signature id in the disallow pair |
| `signature_id_2` | `string` | no | Second signature id in the disallow pair |

Each id must exist in `signatures.arrow`. Runtime treats the pair as
undirected, matching existing `cluster_seeds_disallow` semantics.
Pairs must not be self-pairs. Duplicate pairs, including reversed duplicates,
should fail validation.

### `altered_cluster_signatures.arrow`

Required for incremental prediction when the request includes altered claimed
profiles. Omit it, or write an empty table, when no altered profiles are
present.

Required columns:

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id` | `string` | no | Seed signature id belonging to an altered claimed profile |

Each id must exist in `signatures.arrow` and in the active seed source. At
runtime, S2AND maps these signature ids through the seed assignments to identify
the claimed seed components that need altered-profile pre-splitting.
`signature_id` values must be unique.

`altered_cluster_signatures.txt` with one signature id per line is supported
only by classic ANDData/training inputs.
Production Arrow path mappings must point at the Arrow table.

### `<dataset>_clusters.json`

Eval-only truth data. Keep the same shape as existing S2AND clusters JSON:

```json
{
  "cluster_id": {
    "cluster_id": "cluster_id",
    "signature_ids": ["signature_a", "signature_b"],
    "model_version": -1
  }
}
```

The `signature_ids` field is required by the S2AND loader. Other fields, such as
`cluster_id` and `model_version`, are conventional metadata in existing bundles.

Production prediction does not need this file.

---

## Name Counts

Manifest expectations from this spec:

1. Provide a shared/global `name_counts_index/` sidecar referenced from
   manifests via the `name_counts_index` path key when the selected model uses
   name-count features.
2. Do not build a request-time pipeline that loads legacy name-count artifacts
   into Python dicts/lists. That defeats the purpose of this contract.

The on-disk layout, manifest schema (`schema_version: "name_counts_index_v1"`),
binary record format, and immutable-generation publication ritual are owned by
[`artifact_formats.md` -- Name Counts](artifact_formats.md#name-counts). New
writers must publish through that contract.

---

## Name Aliases

Production datasets must not contain per-dataset `name_pairs.arrow` files or
manifest path keys. The runtime default is the packaged canonical alias file:

```text
s2and_name_tuples_canonical.txt
```

If a non-default alias set is ever needed, make it an explicit shared/global
runtime artifact passed through the Python `name_tuples` argument, not something
duplicated into every dataset directory or hidden in path bundles.

---

## Manifests

Each dataset directory must contain `manifest.json`. The manifest is not the hot
path source of truth, but it is required for auditability and validation.

Required fields for every semantic Arrow manifest:

```json
{
  "schema": "feature_block_arrow_v2",
  "normalization_version": "canonical_v2",
  "dataset": "dataset_name",
  "signature_count": 0,
  "paper_count": 0,
  "paths": {
    "signatures": "signatures.arrow",
    "papers": "papers.arrow",
    "paper_authors": "paper_authors.arrow"
  },
  "artifact_generation": {
    "schema_version": "s2and_arrow_artifact_generation_v1",
    "generation_id": "sha256-of-canonical-file-inventory",
    "files": {}
  },
  "name_tuples": "default packaged canonical aliases"
}
```

The manifest `schema` value is the on-disk Arrow manifest schema. In Python it
is exposed as
`s2and.incremental_linking.feature_block.FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION`.
Do not use the in-memory `FeatureBlock` schema constant for manifest
validation.

The canonical manifest builder supplies `normalization_version`, `paths`, and
`artifact_generation`. Dataset producers supply fields such as `schema`,
`dataset`, and row counts as metadata.

Conditional `paths` entries:

- `specter` is required when the selected model uses `embedding_similarity`.
  This is the selected embedding file for the run, even when the physical file is
  named `specter2.arrow`.
- `specter2` may be included as bundle inventory when both embedding versions
  are shipped, but runtime callers still pass the selected embedding as
  `specter`.
- `cluster_seeds` is required only when the published Arrow sidecar is the seed
  source. Seeded or incremental Arrow prediction may instead receive a
  normalized request/dataset seed mapping and materialize request-local Arrow.
  `cluster_seed_disallows` is optional; omit it when no disallows are present.
- `altered_cluster_signatures` is required when altered claimed profiles are
  present.
- `clusters` is eval-only ground truth.
- `name_counts_index` is required when the selected model uses name-count
  features.
- `paths.name_pairs` or `paths.name_tuples` must not be present in manifests.
  Top-level `name_tuples` metadata is allowed to describe how the artifact was
  produced.

Large-block optimized artifacts should also include:

```json
{
  "paths": {
    "specter": "specter.arrow",
    "signatures_batch_index": "signatures.signatures_batch_index.bin",
    "papers_batch_index": "papers.papers_batch_index.bin",
    "paper_authors_batch_index": "paper_authors.paper_authors_batch_index.bin",
    "specter_batch_index": "specter.specter_batch_index.bin"
  },
  "physical_layout": {
    "schema": "s2and_arrow_physical_v1",
    "optimized_for": "incremental_raw_candidate_planning",
    "tables": {
      "signatures": {
        "key": "signature_id",
        "max_record_batch_rows": 16384,
        "row_count": 0,
        "record_batch_count": 0,
        "actual_max_batch_rows": 0,
        "batch_index_path_key": "signatures_batch_index",
        "batch_index_present": true
      },
      "papers": {
        "key": "paper_id",
        "max_record_batch_rows": 16384,
        "row_count": 0,
        "record_batch_count": 0,
        "actual_max_batch_rows": 0,
        "batch_index_path_key": "papers_batch_index",
        "batch_index_present": true
      },
      "paper_authors": {
        "key": "paper_id",
        "max_record_batch_rows": 16384,
        "row_count": 0,
        "record_batch_count": 0,
        "actual_max_batch_rows": 0,
        "batch_index_path_key": "paper_authors_batch_index",
        "batch_index_present": true
      },
      "specter": {
        "key": "paper_id",
        "max_record_batch_rows": 2048,
        "row_count": 0,
        "record_batch_count": 0,
        "actual_max_batch_rows": 0,
        "batch_index_path_key": "specter_batch_index",
        "batch_index_present": true
      }
    }
  }
}
```

Repeat the `physical_layout.tables` entry for every large lookup table shipped
for indexed raw planning. If both `specter.arrow` and `specter2.arrow` are
included, inventory both embedding layouts or clearly identify which embedding
is selected for the manifest.

Recommended additional fields:

- `cluster_count` for eval datasets.
- `source_dir` or source snapshot identifier.
- `generated_at`.
- `generator_version` or git commit.
- `specter` metadata with `row_count`, `dimension`, and source artifact id for
  each embedding file.
- `name_counts_index` metadata with the shared index path and schema version.
- `physical_layout.tables.<table>` entries for every large lookup table:
  `row_count`, `record_batch_count`, `actual_max_batch_rows`,
  `max_record_batch_rows`, lookup `key`, and batch-index presence.
- `raw_planner_batch_indexes` metrics when S2AND-generated sidecars are present.
- `validation` summary with row counts, duplicate counts, missing reference
  counts, physical-layout checks, and parity-check command/output location.

Root-level `manifest.json` should use schema `inference_arrow_bundle_v1` and
list dataset directories and their manifest paths in `dataset_manifests` when an
artifact bundle contains multiple datasets. Keep per-input `source_path` values
in dataset manifests; do not write a root-level `source_path`. Existing root
manifests without `schema: "inference_arrow_bundle_v1"` are rejected instead of
migrated in place.

---

## Validation Checklist

Validate every generated dataset before handing it to model evaluation or
production inference.

Required checks:

- Every Arrow file opens with `pyarrow.ipc.open_file(...)`.
- Required files exist for the intended use case.
- Required columns exist with the exact Arrow types above.
- `signature_id` values are unique.
- `paper_id` values in `papers.arrow` are unique.
- `paper_id` values in each selected embedding file are unique.
- `(paper_id, position)` values in `paper_authors.arrow` are unique.
- Every `signatures.paper_id` exists in `papers.arrow`.
- Every `paper_authors.paper_id` exists in `papers.arrow`.
- When embeddings are required, the selected SPECTER Arrow file exists and
  validates structurally. Require every referenced paper to have an embedding
  only for datasets whose source contract guarantees complete coverage.
- `query_signatures.signature_id` is unique, is a subset of
  `signatures.signature_id`, and every `query_view` is one of `auto`, `full`,
  or `initial_only`.
- `cluster_seeds.signature_id` is a subset of `signatures.signature_id`.
- `cluster_seeds.signature_id` values are unique and every `cluster_id` is a
  non-empty string.
- `cluster_seed_disallows.signature_id_1` and
  `cluster_seed_disallows.signature_id_2` are subsets of
  `signatures.signature_id`.
- `cluster_seed_disallows.arrow` contains no self-pairs and no duplicate
  undirected pairs.
- `altered_cluster_signatures.signature_id` is unique and is a subset of both
  `signatures.signature_id` and `cluster_seeds.signature_id`.
- `name_counts_index/manifest.json` exists when the selected model uses
  `name_counts`.
- Manifest row counts match the corresponding Arrow table row counts.
- `author_block` is present when the dataset will be used for block
  reconstruction or offline eval.
- Signature row order matches the source `ANDData` order or the documented
  source order for that dataset.
- Eval-only clusters JSON references only signatures present in
  `signatures.arrow`.

Required physical-layout checks for large-block optimized artifacts:

- `signatures.arrow`, `papers.arrow`, `paper_authors.arrow`, and the selected
  embedding file are bounded as specified in
  [Large-Block Physical Layout](#large-block-physical-layout).
- `physical_layout.schema` is `s2and_arrow_physical_v1`.
- `physical_layout.tables.<table>.actual_max_batch_rows` is less than or equal
  to `physical_layout.tables.<table>.max_record_batch_rows`.
- One-batch lookup tables have
  `row_count <= max_record_batch_rows`; otherwise they should be rejected as
  unoptimized for indexed raw planning.
- If batch-index sidecars are present, they were generated from the final Arrow
  files and the manifest path keys point to those sidecars.
- Batch-index validation must not require source file mtimes to match. Object
  store downloads can rewrite mtimes; validators use source size plus the
  stored full-file source fingerprint for portable release artifacts.

Recommended smoke checks:

PowerShell:

```powershell
uv run python scripts/convert_to_arrow.py validate `
  --dataset-dir s2and/data/qian `
  --require-embeddings `
  --require-name-counts-index
```

```powershell
uv run python scripts/eval_prod_models.py `
  --dataset full `
  --use-arrow `
  --datasets qian `
  --specter-suffixes _specter2.pkl `
  --n_jobs 4 `
  --seed 42
```

Bash:

```bash
uv run python scripts/convert_to_arrow.py validate \
  --dataset-dir s2and/data/qian \
  --require-embeddings \
  --require-name-counts-index

uv run python scripts/eval_prod_models.py \
  --dataset full \
  --use-arrow \
  --datasets qian \
  --specter-suffixes _specter2.pkl \
  --n_jobs 4 \
  --seed 42
```

The eval command should report `use_arrow=True` and `Arrow data root:
s2and/data` after the public Arrow release has been synced locally.

---

## Non-Goals

This Arrow dataset contract is not a full `ANDData` replacement. Do not include
training pair samples, train/val/test split construction artifacts, reference
features, upstream normalization artifacts, or pair-sampling policy state unless a
separate training/eval contract explicitly asks for them.

The direct Rust inference path should consume only the narrow feature-block
inputs it needs for scoring and clustering.
