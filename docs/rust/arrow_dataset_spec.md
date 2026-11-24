# Arrow Dataset Specification

This document defines the Arrow artifact contract for engineers assembling
datasets for direct Rust S2AND routes. These artifacts are used by
`ArrowDataset.open(root)`, `Clusterer.predict_from_arrow(...)`,
`Clusterer.predict_incremental_from_arrow(...)`, and
`build_training_anddata_from_arrow(...)`. Classic prediction methods operate on
`ANDData` through Python.

Production Arrow is a raw runtime input contract, not a serialized
`ANDData(preprocess=True)` cache. The goal is feature parity with the current
`ANDData(preprocess=True)` representation after the local runtime preprocesses
the Arrow rows. Arrow text/name columns are preprocessing inputs, not
authoritative precomputed feature values: Rust normalizes titles, venues,
journals, signature names, paper-author names, and computes language-dependent
paper state while building the scoring view.

---

## Summary

Each Arrow dataset is a directory of Arrow interprocess communication (IPC)
file-format tables plus a manifest. The hot path reads these files directly from
Rust or through memory-mapped Arrow readers.

Required by `ArrowDataset.open(...)` for every dataset:

- `signatures.arrow`, `papers.arrow`, and `paper_authors.arrow`
- their S2AND-generated batch indexes (`signatures_batch_index`,
  `papers_batch_index`, `paper_authors_batch_index`)

Declare at most one embedding table under `specter`, together with its
`specter_batch_index`. Both are required for subblocking and for models that
use `embedding_similarity`. Production and evaluation bundles use
`specter2.arrow`. Research training can select the historical `specter.arrow`.

The sidecars below are optional files for producers and validators.
`ArrowDataset.open(...)` ignores their manifest paths.
`ArrowDataset.open_trusted(...)` ignores `query_signatures` and raises `ValueError`
for the other sidecar keys. Pass request-local values explicitly to public
prediction.

| Sidecar | Prediction argument or reader |
|---|---|
| `cluster_seeds.arrow` | `cluster_seeds_require` |
| `cluster_seed_disallows.arrow` | `cluster_seeds_disallow` |
| `altered_cluster_signatures.arrow` | `altered_cluster_signatures` for incremental prediction. Bulk prediction uses the required seed groups unchanged. |
| `profile_exclusions.arrow` | `profile_exclusions` on `predict_from_arrow(...)` only. Read the pairs with `read_profile_exclusions_arrow(...)`. |
| `query_signatures.arrow` | [Internal planner tests](#query_signaturesarrow) |

`predict_incremental_from_arrow(...)` requires a nonempty `cluster_seeds_require`
mapping. It plans requested signatures absent from that mapping as queries in
automatic view mode and writes request-local seed and disallow tables from the
arguments as needed.

Offline evaluation datasets may also include labels:

- `<dataset>_clusters.json` for cluster-labeled benchmarks
- `train_pairs.csv`, `test_pairs.csv`, and optional `val_pairs.csv` for
  pair-labeled benchmarks such as MEDLINE

---

## Layout

Preferred on-disk layout:

```text
<arrow_root>/
  manifest.json
  name_counts_index/
    manifest.json
    first.bin
    last.bin
    first_last.bin
    last_first_initial.bin
  <dataset>/
    manifest.json
    signatures.arrow
    papers.arrow
    paper_authors.arrow
    specter2.arrow
    signatures.signatures_batch_index.bin
    papers.papers_batch_index.bin
    paper_authors.paper_authors_batch_index.bin
    specter2.specter_batch_index.bin
    query_signatures.arrow
    cluster_seeds.arrow
    cluster_seed_disallows.arrow
    altered_cluster_signatures.arrow
    profile_exclusions.arrow
    <dataset>_clusters.json
    train_pairs.csv
    val_pairs.csv
    test_pairs.csv
```

Write Arrow IPC file-format tables with `pyarrow.ipc.new_file(...)`. Readers
use `pyarrow.ipc.open_file(...)` and memory maps.

The machine-readable column contract lives at
`s2and/arrow_schema_contract.json`. It is a parity guard for producer/consumer
drift; runtime readers still enforce their local validation rules directly.

### Manifest paths

Callers open the dataset root once with `ArrowDataset.open(root)`. The handle
validates the manifest and retains the immutable tables, indexes, and optional
name-count index; training and prediction receive that handle directly. Rust
routes do not infer sibling `<data_root>_arrow/<dataset>` directories. The
manifest `paths` object uses these keys:

| Key | Meaning |
|---|---|
| `signatures` | Required path to `signatures.arrow` |
| `papers` | Required path to `papers.arrow` |
| `paper_authors` | Required path to `paper_authors.arrow` |
| `specter` | Path to the embedding table selected for the current model. See the [embedding requirements](#summary). |
| `query_signatures` | Optional path to `query_signatures.arrow` |
| `cluster_seeds` | Optional path to `cluster_seeds.arrow` |
| `cluster_seed_disallows` | Optional path to `cluster_seed_disallows.arrow` |
| `altered_cluster_signatures` | Optional path to `altered_cluster_signatures.arrow` |
| `profile_exclusions` | Optional path to `profile_exclusions.arrow` |
| `clusters` | Path to eval-only ground-truth clusters JSON |
| `train_pairs`, `val_pairs`, `test_pairs` | Paths to eval-only fixed-pair label CSVs for pair-labeled benchmarks; `val_pairs` is optional |
| `name_counts_index` | Required manifest-declared shared/global name-count index directory when the selected model uses `name_counts` |
| `signatures_batch_index` | S2AND-generated lookup index for `signatures.arrow`; always required by `ArrowDataset.open(...)` |
| `papers_batch_index` | S2AND-generated lookup index for `papers.arrow`; always required by `ArrowDataset.open(...)` |
| `paper_authors_batch_index` | S2AND-generated lookup index for `paper_authors.arrow`; always required by `ArrowDataset.open(...)` |
| `specter_batch_index` | S2AND-generated lookup index for the selected embedding table, typically named `<embedding-stem>.specter_batch_index.bin` |
| `name_pairs`, `name_tuples` | Omit these keys. The runtime loads [name aliases](#name-aliases) separately. |

---

## Large-Block Physical Layout

Bounded record batches limit how much data indexed candidate planning reads on
large blocks such as common family-name blocks. The batch-index writer and
validator reject lookup tables with record batches above these limits.

Producers must write the lookup tables below as Arrow IPC file-format files
with bounded record batches. Do not write these tables as one giant record
batch when the row count exceeds the limit.

| Table | Lookup key | Maximum rows per IPC record batch |
|---|---|---:|
| `signatures.arrow` | `signature_id` | 16,384 |
| `papers.arrow` | `paper_id` | 16,384 |
| `paper_authors.arrow` | `paper_id` | 16,384 |
| `specter.arrow` / `specter2.arrow` | `paper_id` | 2,048 |

The smaller request sidecars have no bounded-batch or lookup-index requirements:

| Table | Layout guidance |
|---|---|
| `query_signatures.arrow` | Internal planner test input |
| `cluster_seeds.arrow` | Read fully by the raw planner |
| `cluster_seed_disallows.arrow` | Read fully when present |
| `altered_cluster_signatures.arrow` | Bounded batches do not address altered-profile pre-splitting cost. |
| `profile_exclusions.arrow` | Read fully by `read_profile_exclusions_arrow(...)` |

Implementation notes for producers:

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
for an S2AND prep step to generate these indexes. Generate them from the final
Arrow bytes with S2AND tooling, such as
`s2and.incremental_linking.feature_block.write_raw_arrow_batch_lookup_indexes`.
Regenerate the index if the table is rewritten. Keep each table and index
together when copying files to their serving location.

Every script that produces S2AND runtime Arrow artifacts should use the shared
writers instead of open-coding the table or sidecar formats:

- `scripts.common.arrow_conversion_helpers.write_raw_planner_arrow_from_anddata(...)` or
  `write_raw_planner_arrow_tables(...)` for semantic Arrow IPC tables.
- `write_raw_arrow_batch_lookup_indexes(...)` after the final table write for
  raw-planner sidecars.
- `raw_planner_arrow_physical_layout(...)` for transient inspection of the
  final bytes when validation or reporting needs batch metrics.

Recommended sidecar filenames are stem-qualified:

```text
signatures.signatures_batch_index.bin
papers.papers_batch_index.bin
paper_authors.paper_authors_batch_index.bin
specter2.specter_batch_index.bin
```

The double stem is intentional: the first stem identifies the Arrow file and the
trailing `<table>_batch_index` stem matches the manifest path key.

The batch-index format is S2AND-owned. Current writers and readers require
format magic `S2ABI003`. Its 32-byte little-endian header is `<8sQQQ`: magic,
record count, source byte count, and key-column hash. Each 16-byte `<QII` record
contains a key hash, batch index, and reserved zero. There is no full-file source
fingerprint. Older batch indexes must be rebuilt with the current writer.
Manifest-backed opens verify SHA-256 checksums; `open_trusted(paths)` skips the
dataset manifest. Both retain source-size, schema, and index structure checks.
Standalone validation and `overwrite=False` reuse check structure and size,
not content identity; the producer must keep each table/index pair together.
Each record maps a
64-bit FNV-1a hash of the lookup key to an IPC record-batch index; the Rust
reader verifies exact ids after loading the selected batches, so hash collisions
do not change results. The file body must contain exactly the header-declared
record count, records must be ordered by nondecreasing key hash, and every stored
batch index must be smaller than the Arrow file's IPC record-batch count.

---

## Runtime Input Semantics

Rows must provide the source values needed for the local Rust runtime to produce
the same feature view that S2AND would expose after normal preprocessing:

- `author_block` is the sole supported blocking source
- Python `name_tuples=None`, resolved to validated packaged pairs before the
  Rust call
- `name_counts_index/` available when the selected model uses name-count features

`author_orcid` is optional. If the column is absent, every signature is treated
as having no ORCID evidence; all non-ORCID features and constraints remain
available. Name-count artifacts always use the canonical initial-character key.

`scripts/intake/convert_to_arrow.py` is the reference producer for deployable
dataset manifests and current batch-index sidecars. Use
`s2and.arrow_inputs.build_arrow_artifact_manifest(...)` and
`write_arrow_artifact_manifest(...)` to publish manifests from the paths the
shared table writers return. Measure feature parity after Rust preprocessing.

Important parity details:

- Preserve source signature order. The current converter writes
  `signature_ids=list(dataset.signatures)` for this reason.
- Store ids as strings, even if an upstream source stores numeric ids.
- Text/name fields should be source/raw values where practical. Rust owns the
  normalization, ngram, unidecode, name splitting, and language-detection work
  needed for production scoring.
- Keep `abstract` as an abstract-presence signal, not raw abstract text. The
  current direct Arrow conversion writes `"Has Abstract"` when the preprocessed
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
| `author_position` | `int64` | no | Author position on the paper |
| `author_block` | `string` | yes | Optional column containing the S2 block key; needed for block reconstruction/eval |
| `author_email` | `string` | yes | Optional column containing author email |
| `source_author_ids` | `list<string>` | yes | Optional column containing upstream author ids |

Name-count values are intentionally not part of the signature table.

Both full Rust featurization and raw candidate planning reject null
`author_position`: correct coauthor exclusion and local-window evidence cannot
be reconstructed without the focal position. Release datasets must satisfy
this required/non-null contract before training or evaluation; see the
[v1.3 training and evaluation data stage](../release.md#stage-2-build-training-and-evaluation-data).

### `papers.arrow`

One row per paper referenced by `signatures.arrow`. Columns:

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `paper_id` | `string` | no | Stable paper id |
| `title` | `string` | yes | Source/raw paper title text used as runtime preprocessing input |
| `abstract` | `string` | yes | Optional column containing an abstract-presence signal: `"Has Abstract"` or `""` |
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

A paper with an empty or absent source author list has no child rows.
Raw candidate planners accept this representation and use absent coauthor evidence.
Every signature must still reference an existing row in `papers.arrow`.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `paper_id` | `string` | no | Referenced paper id |
| `position` | `int64` | no | Author position |
| `author_name` | `string` | no | Source paper-author name string used as runtime preprocessing input for coauthor features |

Rows should be ordered by `paper_id` then `position` where practical. Ordering is
not the identity contract, but stable ordering makes diffs and validation easier.
An empty or whitespace-only `author_name` is valid and must remain a row so that
source author positions and list cardinality are preserved. Consumers apply their
existing preprocessing semantics to that retained value: modern raw-planner and
subblocking name evidence ignores names that normalize empty, while classic
pairwise preprocessing retains the legacy coauthor-set behavior. A null
`author_name` is invalid.

### Selected embedding table (`specter2.arrow` in production)

One row per embedded paper. Required columns:

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `paper_id` | `string` | no | Referenced paper id |
| `embedding` | `fixed_size_list<float32>[dimension]` | no | SPECTER vector |

All vectors in one file must have the same dimension, and `paper_id` values
must be unique. Represent missing embeddings by omitting their rows. Partial
coverage, including a valid zero-row table, is accepted. Missing vectors use
the runtime's missing-vector feature behavior and can change scores.
`scripts/intake/convert_to_arrow.py service-json` and the shared `ANDData` table writer
can emit an empty table. The `benchmark` command requires at least one matching
vector when it writes the embedding table.
For a source contract that guarantees complete coverage, validate with
`--require-complete-embeddings`; `--require-embeddings` checks table presence
and structure without requiring an embedding for every referenced paper.

### `query_signatures.arrow`

Query table for the internal `RawBlockQueryCandidatePlanner.from_query_signatures`
constructor, which tests use to exercise declared per-query view policies.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id` | `string` | no | Query signature id |
| `query_view` | `string` | no | Requested view: `auto`, `full`, or `initial_only` |
| `query_author` | `string` | no | Caller-visible query author text; empty string is allowed |

`signature_id` values must be unique and non-empty. `query_view` values must be
valid. The planner derives scoring-time author evidence from `signatures.arrow`
and validates a non-empty `query_author` against that derived query author.

### `cluster_seeds.arrow`

One row per required seed assignment.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id` | `string` | no | Seed signature id |
| `cluster_id` | `string` | no | Required seed component/cluster id |

`signature_id` values must be unique, and `cluster_id` values must be non-empty
strings.

### `cluster_seed_disallows.arrow`

Omit the file or write an empty table when no seed disallows are present.
Pass `include_empty_cluster_seeds=True` to `write_raw_planner_arrow_from_anddata(...)`
to emit both seed tables even when empty. The `validate` command checks the
disallow table when its declared file exists.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id_1` | `string` | no | First signature id in the disallow pair |
| `signature_id_2` | `string` | no | Second signature id in the disallow pair |

Each id must exist in `signatures.arrow`. Runtime treats the pair as
undirected, matching existing `cluster_seeds_disallow` semantics.
Pairs must not be self-pairs. Duplicate pairs, including reversed duplicates,
should fail validation.

### `altered_cluster_signatures.arrow`

Omit the file or write an empty table when no altered profiles are present.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id` | `string` | no | Seed signature id belonging to an altered claimed profile |

Each id must be unique and exist in `signatures.arrow` and in the active seed
assignments. Incremental prediction uses the seed assignments to identify the
claimed seed components that need altered-profile pre-splitting.

`altered_cluster_signatures.txt` with one signature id per line is supported
only by classic ANDData/training inputs.
The producer manifest path must point at the Arrow table.

### `profile_exclusions.arrow`

Each row keeps an excluded signature out of the cluster that contains its anchor.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `anchor_signature_id` | `string` | no | Signature id placed on the claimed profile |
| `excluded_signature_id` | `string` | no | Signature id excluded from the anchor's cluster |

Both columns are required. Pairs are directional and unique, with distinct ids
in each row. `scripts/intake/convert_to_arrow.py service-json` writes this sidecar when
the request supplies `profile_exclusions`. It checks the ids against the dataset
and rejects exclusions that contradict required seed assignments. The writer
sorts and deduplicates pairs. The `validate` command checks the declared file
when present. It uses the sidecar reader and checks both ids against
`signatures.arrow`.

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

### Fixed-pair label CSVs

Pair-labeled benchmarks such as MEDLINE have no clusters. Their labels ship as
`train_pairs.csv` and `test_pairs.csv`, plus an optional `val_pairs.csv`,
declared under the `train_pairs`, `test_pairs`, and `val_pairs` path keys.
Each file uses the `ANDData` fixed-pair contract: columns `signature_id_1`,
`signature_id_2`, and `label` (`YES`/`NO` or `1`/`0`). Pass the files to
`build_training_anddata_from_arrow(..., train_pairs=..., test_pairs=...)` or
`ANDData(..., train_pairs=..., test_pairs=...)`.

`scripts/intake/convert_to_arrow.py` publishes these files only after checking that
train and test are both present, every row satisfies the contract, every
signature exists in the dataset, and no unordered pair appears in two splits.
The converter accepts legacy `pair1`/`pair2` and `pairs1`/`pair2` input headers
and writes the canonical column names above. It rejects self-pairs and
conflicting labels for duplicate or reversed pairs; consistent repeated rows
retain their order and multiplicity. `--overwrite` removes a previously
published optional pair CSV when the replacement sources no longer include it.

In the current format, pair CSVs are declared only in `paths`; their contents
are not bound by the dataset manifest's `files` inventory or conversion
provenance. Release plans separately hash their declared pair inputs. Keep
published pair CSVs unchanged and verify them against that source authority
before using a public Arrow root for evaluation.

---

## Name Counts

Manifest expectations from this spec:

1. Provide one `name_counts_index/` per self-contained published root,
   referenced from every benchmark and replay dataset manifest via the
   `name_counts_index` path key when the selected model uses name-count
   features.
2. Do not build a request-time pipeline that loads legacy name-count artifacts
   into Python dicts/lists. That defeats the purpose of this contract.

The directory has exactly these fixed-role files:

```text
name_counts_index/
  manifest.json
  first.bin
  last.bin
  first_last.bin
  last_first_initial.bin
```

Its manifest has exactly this shape:

```json
{
  "kind": "s2and_name_counts",
  "format_version": 1,
  "files": {
    "first": {"byte_count": 0, "sha256": "lowercase-sha256"},
    "last": {"byte_count": 0, "sha256": "lowercase-sha256"},
    "first_last": {"byte_count": 0, "sha256": "lowercase-sha256"},
    "last_first_initial": {"byte_count": 0, "sha256": "lowercase-sha256"}
  }
}
```

Each role resolves to `<role>.bin`. `NameCountsIndex.open(path)` validates the
manifest, declared file sizes, fixed filenames, path containment, file headers,
and section bounds. Use `NameCountsIndex.open(path, verify=True)` or call
`verify()` on the open index to also check every file's SHA-256 digest and every
record. Python uses the opened handle's path and exact manifest SHA-256 for
orchestration and model binding.

Each binary file starts with private magic `S2NCI001` and stores:

```text
header: magic:8, record_count:u64, blob_offset:u64, blob_len:u64
record: hash1:u64, hash2:u64, name_offset:u64, name_len:u32, reserved:u32, count:f64
blob: concatenated UTF-8 name bytes
```

Two FNV-64 hashes narrow lookup and exact UTF-8 comparison prevents hash
collisions from producing false hits. Writers assemble the complete directory
in a temporary sibling and rename it once into an absent
`name_counts_index/` target. Publish changed counts under a new parent/root
rather than mutating an open index.

---

## Name Aliases

Keep per-dataset `name_pairs.arrow` files out of production datasets. The
runtime defaults to the shared, packaged text file below.

```text
s2and_name_tuples_canonical.txt
```

The Python loader directly validates canonical row shape, ordering, uniqueness,
and alias semantics. If a non-default alias set is needed, load its explicit
text path through the same loader and pass the validated pairs to Rust.

---

## Manifests

Each dataset directory must contain `manifest.json`. The manifest is not the hot
path source of truth, but it is required for auditability and validation.

Every dataset manifest requires stable kind `s2and_arrow_dataset`, public
format `1`, portable paths, and a flat content inventory. A shortened example
is:

```json
{
  "kind": "s2and_arrow_dataset",
  "format_version": 1,
  "paths": {
    "signatures": "signatures.arrow",
    "signatures_batch_index": "signatures.signatures_batch_index.bin",
    "papers": "papers.arrow",
    "papers_batch_index": "papers.papers_batch_index.bin",
    "paper_authors": "paper_authors.arrow",
    "paper_authors_batch_index": "paper_authors.paper_authors_batch_index.bin"
  },
  "files": {
    "signatures": {"byte_count": 0, "sha256": "lowercase-sha256"},
    "signatures_batch_index": {"byte_count": 0, "sha256": "lowercase-sha256"},
    "papers": {"byte_count": 0, "sha256": "lowercase-sha256"},
    "papers_batch_index": {"byte_count": 0, "sha256": "lowercase-sha256"},
    "paper_authors": {"byte_count": 0, "sha256": "lowercase-sha256"},
    "paper_authors_batch_index": {"byte_count": 0, "sha256": "lowercase-sha256"}
  }
}
```

`files` inventories the immutable Arrow tables and their batch indexes,
including `specter` and `specter_batch_index` when present. The
`name_counts_index` key appears only in `paths` and points to the index
directory. The model binds that index. `ArrowDataset.open()` ignores legacy
`files.name_counts_index` entries written before `1.0.0b11`. Optional request
sidecars also appear only in `paths`. File and directory kind labels and
generation IDs are derived at runtime.

The canonical manifest builder supplies exactly `kind`, `format_version`,
`paths`, and `files`. The collection root owns dataset names. Dataset manifests
accept no open-ended metadata and no second schema, normalization, generation,
or physical-layout authority.
`ArrowDataset.open()` validates format before opening payloads, then validates
manifest-relative paths, retained bytes, Arrow table schemas, batch-index
structure, and the [name-count index layout](#name-counts). The release-root
validator checks that datasets and declared replay collections use the same
index and verifies every index file digest and record.

See the [manifest paths table](#manifest-paths) for required, conditional, and
omitted keys.

A generic multi-dataset conversion root uses:

```json
{
  "kind": "s2and_arrow_collection",
  "format_version": 1,
  "dataset_manifests": {
    "pubmed": {
      "path": "pubmed/manifest.json",
      "sha256": "lowercase-sha256"
    }
  }
}
```

`replay_bundles`, when present, uses the same name-to-`{path, sha256}` shape
and points at one level of generic collection manifests. Those replay
collections cannot declare further `replay_bundles`. Only a final
self-contained publication may declare replay collections; it uses
`s2and_public_data` and adds the single
owner-selected `release_version`, for example `"1.3"`. Generic conversion
roots omit it. Every published benchmark and replay dataset resolves
`paths.name_counts_index` to that publication root's one shared
`name_counts_index/`.

Public format `1` covers persisted meaning as well as JSON framing. A change to
name canonicalization, count-key construction, missing-count semantics, Arrow
column interpretation, or a public sidecar encoding requires a format bump and
regenerated public data. Private binary magic remains an independent
corruption/layout guard.

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
- When `profile_exclusions.arrow` is present, both columns reference ids in
  `signatures.arrow`, and the pairs meet the
  [profile exclusions schema](#profile_exclusionsarrow).
- `name_counts_index/manifest.json` exists when the selected model uses
  `name_counts`.
- Manifest `files` entries match the retained files' byte counts and SHA-256
  digests. Row counts are derived from Arrow tables and are not manifest fields.
- `author_block` is present when the dataset will be used for block
  reconstruction or offline eval.
- Signature row order matches the source `ANDData` order or the documented
  source order for that dataset.
- Eval-only clusters JSON references only signatures present in
  `signatures.arrow`.
- Fixed-pair label CSVs meet the contract in
  [Fixed-pair label CSVs](#fixed-pair-label-csvs); the converter checks this.

`scripts/intake/convert_to_arrow.py validate` does not check signature row order or
clusters-JSON references; verify those in the producer.

Required physical-layout checks:

- `signatures.arrow`, `papers.arrow`, `paper_authors.arrow`, and the selected
  embedding file are bounded as specified in
  [Large-Block Physical Layout](#large-block-physical-layout).
- Inspect actual IPC record batches; reject any batch above its table's
  maximum. This fact is not copied into a second manifest authority.
- Generate the required batch indexes from the final Arrow files and point the
  manifest path keys to them.
- Batch-index validation must not require source file mtimes to match. Object
  store downloads can rewrite mtimes; index validators use source size and
  structure. Dataset manifests bind portable release files with SHA-256.

Validate a dataset required to supply both embeddings and name counts.
Set `--dataset-dir` to the directory containing its dataset manifest.

```bash
uv run python scripts/intake/convert_to_arrow.py validate \
  --dataset-dir /path/to/arrow_root/pubmed \
  --require-embeddings \
  --require-name-counts-index
```

For release-model evaluation, see the [release runbook](../release.md).

---

## Non-Goals

This Arrow dataset contract is not a full `ANDData` replacement. Do not include
training pair samples, train/val/test split construction artifacts, reference
features, upstream normalization artifacts, or pair-sampling policy state unless a
separate training/eval contract explicitly asks for them.

The direct Rust inference path should consume only the narrow raw-planner
inputs it needs for scoring and clustering.
