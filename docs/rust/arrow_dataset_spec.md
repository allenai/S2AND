# Arrow Dataset Specification

Status date: 2026-09-04

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

Each Arrow dataset is a directory of Arrow IPC file-format tables plus a
manifest. The hot path reads these files directly from Rust or through
memory-mapped Arrow readers.

Required for full-block prediction:

- `signatures.arrow`
- `papers.arrow`
- `paper_authors.arrow`
- one selected embedding table under the canonical `specter` manifest key when
  the model uses `embedding_similarity`; production/eval bundles use physical
  `specter2.arrow`

Public seeded prediction receives `cluster_seeds_require` explicitly.
`predict_incremental_from_arrow(...)` requires this mapping to be nonempty;
request-local disallows and altered profiles are supplied through
`cluster_seeds_disallow` and `altered_cluster_signatures`. Published request
sidecars do not supply these public API arguments.

The raw incremental planner uses request-local tables, materialized by the
runtime from the prediction request:

- request-local `query_signatures.arrow` for raw incremental candidate planning
- `cluster_seeds.arrow`

Optional producer/validation sidecars:

- `cluster_seed_disallows.arrow` for pairwise seed disallow constraints
- `altered_cluster_signatures.arrow` for altered claimed profiles

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
    <dataset>_clusters.json
```

Notes:

- `query_signatures.arrow` is the request-local query table consumed by the
  raw incremental planner. Runtime helpers may materialize it from existing
  Python request arguments; producers that already have a typed request should
  pass it under the `query_signatures` path key.
- `cluster_seeds.arrow` records seed assignments for producer/validation
  tooling and raw planner inputs. Public Arrow prediction uses the explicit
  `cluster_seeds_require` argument; incremental prediction requires a nonempty
  mapping and materializes a request-local `cluster_seeds.arrow` from it.
  Publishing a seed sidecar does not replace that argument.
- `cluster_seed_disallows.arrow` preserves pairwise seed disallow constraints.
  Hand-authored artifacts can omit it when the request has no seed disallows;
  converters may emit an empty table instead. An explicit path must exist when
  present. Public Arrow prediction receives these constraints through the
  explicit `cluster_seeds_disallow` argument.
- When using `scripts.arrow_conversion_helpers.write_raw_planner_arrow_from_anddata(...)` to publish physical
  seeded/incremental seed sidecars, pass `include_empty_cluster_seeds=True` so
  empty seed/disallow tables are still emitted.
- `altered_cluster_signatures.arrow` records altered claimed profiles for
  producer/validation tooling. Public Arrow prediction reads altered profiles
  from the explicit `altered_cluster_signatures` argument, with no fallback to
  this sidecar. Classic prediction reads `ANDData.altered_cluster_signatures`.
  `altered_cluster_signatures.txt` is not a valid
  production Arrow sidecar; it remains only for older fixtures and
  ANDData-compatible training tooling.
- `<dataset>_clusters.json` is ground truth for offline evaluation only. It is
  not part of production inference scoring.
- Each bundle contains exactly one selected embedding table under manifest key
  `specter`. Production and evaluation bundles select physical
  `specter2.arrow`. An explicit research-training bundle may instead select
  historical `specter.arrow`; it does not ship both tables.
- If embeddings are requested but no block papers have embeddings, emit a valid
  zero-row selected embedding table rather than omitting it so production
  prediction degrades through missing-vector features.
- The Arrow files must be Arrow IPC file format, not Arrow stream format. The
  current writer uses `pyarrow.ipc.new_file(...)`; readers use
  `pyarrow.ipc.open_file(...)` and memory maps.

The machine-readable column contract lives at
`s2and/arrow_schema_contract.json`. It is a parity guard for producer/consumer
drift; runtime readers still enforce their local validation rules directly.

Callers open the dataset root once with `ArrowDataset.open(root)`. The handle
validates the manifest and retains the immutable tables, indexes, and optional
name-count index; training and prediction receive that handle directly. Rust
routes do not infer sibling `<data_root>_arrow/<dataset>` directories. The
manifest `paths` object uses these keys:

| Key | Meaning |
|---|---|
| `signatures` | Path to `signatures.arrow` |
| `papers` | Path to `papers.arrow` |
| `paper_authors` | Path to `paper_authors.arrow` |
| `specter` | Path to the embedding table selected for the current model, even if the file is physically named `specter2.arrow` |
| `query_signatures` | Producer/validation path to request-local `query_signatures.arrow` |
| `cluster_seeds` | Producer/validation path to a seed sidecar; public prediction receives seed mappings explicitly |
| `cluster_seed_disallows` | Producer/validation path to pairwise seed-disallow constraints |
| `altered_cluster_signatures` | Producer/validation path for altered claimed profiles |
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

- `scripts.arrow_conversion_helpers.write_raw_planner_arrow_from_anddata(...)` or
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

At runtime, the one selected embedding file is passed under the `specter` path
key, and S2AND uses the adjacent
`<embedding-stem>.specter_batch_index.bin` sidecar. A historical SPECTER1
research-training bundle therefore uses `specter.specter_batch_index.bin`
instead of the production `specter2.specter_batch_index.bin`.

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

- `author_block` is the sole supported blocking source
- Python `name_tuples=None`, resolved to validated packaged pairs before the
  Rust call
- `name_counts_index/` available when the selected model uses name-count features

`author_orcid` is optional. If the column is absent, every signature is treated
as having no ORCID evidence; all non-ORCID features and constraints remain
available. Name-count artifacts always use the canonical initial-character key.

Use the script-only direct Arrow conversion writer as the reference
implementation for physical layout and for benchmark/replay bundles whose
inputs are derived from `ANDData`:
`scripts.arrow_conversion_helpers.write_raw_planner_arrow_from_anddata`.
That writer returns table paths and does not write `manifest.json`. All in-repo
producers pass those paths through
`s2and.arrow_inputs.build_arrow_artifact_manifest(...)` and
`write_arrow_artifact_manifest(...)`, which own the portable paths,
public-format identity, flat content inventory, and publication format.
`scripts/convert_to_arrow.py` is the reference producer for deployable
dataset manifests and current batch-index sidecars. Its `benchmark` command
requires explicit `--source-root` and `--output-root`; `service-json` requires
explicit `--input-json` and `--output-root`. Neither command discovers a
production source or destination root.
`scripts/verification/compare_full_predict_arrow_parity.py` is the reference
bounded parity producer: it writes current batch-index sidecars, resolves a
canonical name-count index, and publishes a public-format-1 manifest for
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
must be unique. A missing embedding means there is no row for that `paper_id`;
do not represent missing vectors with a null `embedding` value. If the model
uses `embedding_similarity`, the selected embedding table must exist, but
partial coverage (including a valid zero-row table) is accepted. Missing vectors
use the runtime's missing-vector feature behavior and can change scores.
For a source contract that guarantees complete coverage, validate with
`--require-complete-embeddings`; `--require-embeddings` checks table presence
and structure without requiring an embedding for every referenced paper.

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

Seed assignment table for producer/validation tooling and raw incremental
planning. Public Arrow prediction receives seeds through the explicit
`cluster_seeds_require` mapping. Incremental prediction requires a nonempty
mapping and writes a request-local `cluster_seeds.arrow` before entering raw
Arrow retrieval; it does not load seed assignments from a published sidecar.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id` | `string` | no | Seed signature id |
| `cluster_id` | `string` | no | Required seed component/cluster id |

Only required seed assignments are persisted here. Pairwise seed disallow
constraints are persisted separately in `cluster_seed_disallows.arrow`.
`signature_id` values must be unique, and `cluster_id` values must be non-empty
strings.

### `cluster_seed_disallows.arrow`

Optional producer/validation sidecar for seed disallow constraints. Public Arrow
prediction receives these pairs through the explicit `cluster_seeds_disallow`
argument. Omit the file when no seed disallows are present, or emit a valid empty
table when using a converter configured to keep seed/disallow tables explicit.
An explicit path must exist when present.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id_1` | `string` | no | First signature id in the disallow pair |
| `signature_id_2` | `string` | no | Second signature id in the disallow pair |

Each id must exist in `signatures.arrow`. Runtime treats the pair as
undirected, matching existing `cluster_seeds_disallow` semantics.
Pairs must not be self-pairs. Duplicate pairs, including reversed duplicates,
should fail validation.

### `altered_cluster_signatures.arrow`

Producer/validation sidecar describing altered claimed profiles. Public Arrow
prediction receives these ids through the explicit `altered_cluster_signatures`
argument and does not load them from this file. Omit it, or write an empty table,
when no altered profiles are present.

Required columns:

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id` | `string` | no | Seed signature id belonging to an altered claimed profile |

Each id must exist in `signatures.arrow` and in the active seed assignments. At
runtime, S2AND maps the explicitly supplied altered signature ids through the
seed assignments to identify the claimed seed components that need
altered-profile pre-splitting.
`signature_id` values must be unique.

`altered_cluster_signatures.txt` with one signature id per line is supported
only by classic ANDData/training inputs.
The producer manifest path must point at the Arrow table.

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

Paths are not serialized; each role resolves to `<role>.bin`. The Rust opener
validates the manifest, declared sizes and digests, fixed filenames, contained
paths, and every record. Python retains only the opened handle's path and exact
manifest SHA-256 for orchestration and model binding.

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

Production datasets must not contain per-dataset `name_pairs.arrow` files or
manifest path keys. The runtime default is one packaged text file:

```text
s2and_name_tuples_canonical.txt
```

The Python loader directly validates canonical row shape, ordering, uniqueness,
and alias semantics. If a non-default alias set is needed, load its explicit
text path through the same loader and pass the validated pairs to Rust. Do not
duplicate it into every dataset or hide it in Arrow path bundles.

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

Optional immutable roles such as `specter`, `specter_batch_index`, and
`name_counts_index` appear in both `paths` and `files`. A name-count file entry
describes the referenced index's `manifest.json`; its path points to the index
directory. Physical paths are serialized only in `paths`; file/directory kind
labels and derived generation IDs are not persisted.

The canonical manifest builder supplies exactly `kind`, `format_version`,
`paths`, and `files`. The collection root owns dataset names. Dataset manifests
accept no open-ended metadata and no second schema, normalization, generation,
or physical-layout authority.
`ArrowDataset.open()` validates format before opening payloads, then validates
manifest-relative paths, retained bytes, Arrow table schemas, batch-index
bytes, and the native name-count index. The release-root validator separately
owns shared-index topology, including datasets in declared replay collections.

Conditional `paths` entries:

- `specter` is required when the selected model uses `embedding_similarity`.
  This is the selected embedding file for the run, even when the physical file is
  named `specter2.arrow`.
- `cluster_seeds`, `cluster_seed_disallows`, and `altered_cluster_signatures`
  are optional producer/validation paths. Public Arrow prediction receives
  seeds, disallows, and altered profiles as explicit arguments; declaring these
  paths does not populate the request.
- `clusters` is eval-only ground truth.
- `name_counts_index` is required when the selected model uses name-count
  features.
- `paths.name_pairs` or `paths.name_tuples` must not be present in manifests.

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
- Manifest `files` entries match the retained files' byte counts and SHA-256
  digests. Row counts are derived from Arrow tables and are not manifest fields.
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
- Inspect actual IPC record batches; reject any batch above its table's
  maximum. This fact is not copied into a second manifest authority.
- Canonical production/eval roots contain batch indexes for signatures,
  papers, paper authors, and the selected embedding; they were generated from
  the final Arrow files and the manifest path keys point to them. Reduced
  non-production fixtures may omit only indexes their validation profile does
  not require.
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
  --arrow-data-root s2and/data `
  --datasets qian `
  --specter2-model-path path\to\production_model_bundle `
  --n_jobs 4
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
  --arrow-data-root s2and/data \
  --datasets qian \
  --specter2-model-path /path/to/production_model_bundle \
  --n_jobs 4
```

The eval command should report `use_arrow=True` and the resolved absolute
`Arrow data root` corresponding to the explicit `s2and/data` argument after the
public Arrow release has been synced locally. There is no implicit Arrow root.
JSON/ANDData evaluation or training instead requires
explicit `--json-data-root`, `--name-tuples-path`, and
`--name-counts-index-root`; it does not discover those inputs from package
defaults.
Production-bundle evaluation rejects an explicit `--seed`; it reads the
trainer's recorded `data_random_seed`. That reproduces a split only when input
bytes and ordering are identical. The v1.3 release uses persisted, digested
split identities and the one-shot evaluators in the release runbook instead of
treating a seed as identity evidence.

---

## Non-Goals

This Arrow dataset contract is not a full `ANDData` replacement. Do not include
training pair samples, train/val/test split construction artifacts, reference
features, upstream normalization artifacts, or pair-sampling policy state unless a
separate training/eval contract explicitly asks for them.

The direct Rust inference path should consume only the narrow raw-planner
inputs it needs for scoring and clustering.
