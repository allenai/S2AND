# Rust Artifact Formats

Status date: 2026-05-22

This is the current artifact-format decision table for Rust-backed inference.
It replaces the older artifact-divergence migration log.

## Current Targets

| Artifact / data family | Target format | Runtime role |
|---|---|---|
| Signatures | `signatures.arrow` Arrow IPC table | Required direct-Rust input. Contains identity, paper id, author name fields, affiliations, ORCID, position, optional block/email/source ids. It does not contain embedded name-count columns. |
| Papers | `papers.arrow` Arrow IPC table | Required direct-Rust input. Contains title, abstract-presence signal, venue, optional year, language, and reliability fields. |
| Paper authors | `paper_authors.arrow` Arrow IPC table | Required for coauthor and paper-author row signals. |
| Cluster seeds | `cluster_seeds.arrow` Arrow IPC table | Required for seeded/incremental Arrow prediction. Omit for unseeded full prediction. |
| Cluster seed disallows | `cluster_seed_disallows.arrow` Arrow IPC table | Optional for seeded/incremental Arrow prediction. Include it when pairwise seed disallow constraints are present; omitted means no disallows. |
| SPECTER | `specter.arrow` Arrow fixed-size-list `float32` table | Preferred direct-path embedding input. Include the embedding version required by the model. |
| Raw-planner batch indexes | `<arrow-stem>.<path-key>.bin` S2AND binary sidecar | Optional derived indexes for large-block raw planning. Current writers emit `arrow_batch_lookup_index` with magic `S2ABI002`; regenerate from the final Arrow IPC files. |
| Name counts | `s2and/data/name_counts_index/` sorted binary sidecar | Preferred Rust hot-path lookup artifact for models that use name-count features. |
| Name aliases | Packaged canonical text file | Shared runtime default. Avoid per-dataset alias artifacts unless running an explicit experiment. |
| Pairwise and linker models | Native LightGBM text plus JSON metadata | Current production model-bundle format. |
| Eval clusters | Existing clusters JSON | Offline evaluation truth only; not part of production inference scoring. |

## Name Counts

The preferred production publication layout is:

```text
s2and/data/name_counts_index/
  manifest.json
  generations/<publication-generation>/
    .published
    first.bin
    last.bin
    first_last.bin
    last_first_initial.bin
```

`manifest.json` must have `schema_version: "name_counts_index_v2"`,
`normalization_version: "canonical_v2"`, a complete
`name_counts_provenance_v3` `source_provenance`, and a `files` object with
`first`, `last`, `first_last`, and `last_first_initial` entries. Each entry
requires a nonempty contained `path`, unsigned `byte_count`, and lowercase
SHA-256. The declared size and digest must match the file, and the file's
directory must contain `.published`. A `record_count` may be descriptive but
is not the acceptance authority. Each path must equal
`generations/<publication-generation>/<kind>.bin`, and all four files must
share the same nonempty publication-generation directory. This directory name
is a storage identifier and need not equal `source_provenance.generation_id`.

`name_counts_provenance_v3` records the input cardinality once as
`source_row_count`; the duplicate `selected_row_count` field from v1 is not
accepted. It retains warehouse snapshot, query, selected-row, cardinality, and
generation audit facts, but does not name a separately published pickle.

The native Rust opener is the runtime authority for manifest, file-digest, and
record validation. Python freezes the provenance and resolved file facts
returned by that native handle for orchestration and model binding; it does
not run a second manifest-schema validator.

Writers require the final `name_counts_index/` target to be absent. They build
the complete layout above in a temporary sibling directory, fsync it, and
rename that directory once to `name_counts_index/`. Failed builds leave the
target absent. Published indexes are immutable: publish changed counts at a new
parent/path and update the enclosing release manifest instead of replacing an
existing index.

Each `.bin` file starts with magic `S2NCI001` and stores sorted records with
layout:

```text
header: magic:8, record_count:u64, blob_offset:u64, blob_len:u64
record: hash1:u64, hash2:u64, name_offset:u64, name_len:u32, reserved:u32, count:f64
blob: concatenated UTF-8 name bytes
```

Lookup uses two FNV-64 hashes plus exact byte-string verification, so hash
collisions do not produce false name-count hits.

Do not embed per-signature name-count values in `signatures.arrow`. That path
has been removed from the runtime direction. Rust production scoring and Python
`ANDData` both open this validated memory-mapped index. Python deduplicates each
2,048-signature key batch, resolves unique keys in one four-column native call,
and attaches only the resulting scalar counts. Do not build any runtime path
that loads `name_counts.pickle` into Python dicts/lists.

The legacy direct-file layout with `first.bin`, `last.bin`, `first_last.bin`,
and `last_first_initial.bin` directly under `name_counts_index/` is rejected.
Regenerate it as a manifest-backed generation. Production manifests must use
the `name_counts_index` key; runtime boundaries reject the old
`name_counts_index_dir` alias.

## Arrow Runtime Writers

`s2and.arrow_inputs.build_arrow_artifact_manifest(...)` and
`write_arrow_artifact_manifest(...)` are the in-repo authority for portable
paths, normalization, immutable-generation inventory, serialization, and
publication. Producer-specific metadata may be added, but cannot override
`normalization_version`, `paths`, or `artifact_generation`.

`scripts/convert_to_arrow.py` is the reference deployable Arrow-bundle producer.
It writes bounded Arrow IPC file-format tables, regenerates current raw-planner
batch-index sidecars (`S2ABI002`), records physical-layout metrics, and writes
dataset manifests. `scripts/verification/compare_full_predict_arrow_parity.py`
is the reference bounded parity writer and follows the same table and sidecar
helpers for temporary verification artifacts. It also binds those files and
the selected canonical name-count index into an immutable artifact-generation
manifest before calling a production validator.

New scripts that create S2AND runtime Arrow files should use
`scripts.arrow_conversion_helpers.write_raw_planner_arrow_from_anddata(...)`
or `write_raw_planner_arrow_tables(...)`, then call
`write_raw_arrow_batch_lookup_indexes(...)` when the artifact may be used by raw
planning. Do not hand-write the batch-index binary format.

## Deprioritized Or Rejected

| Format / approach | Current decision |
|---|---|
| Embedded `name_count_*` columns in `signatures.arrow` | Removed as a preferred/supporting Arrow hot path. Use `name_counts_index/`. |
| SQLite for name counts | Not better for the current exact static point-lookup workload. Revisit only for ad hoc queries, updates, or transaction requirements. |
| Pickle | Removed. The native index is the sole published representation. |
| JSON | Fine for fixtures and compatibility loaders; not the runtime target for large table-shaped inference data. |
| Arrow read into Python dict/list before Rust | Defeats the columnar boundary and was measured slower than keeping the hot path in Rust. |
| MessagePack as universal target | Better than JSON for nested legacy payloads, but it preserves the object shape the Rust path is trying to avoid. |
| Parquet as request/runtime hot path | Useful offline, but Arrow IPC is simpler for local runtime bundles and direct Rust readers. |
| Per-dataset `name_pairs.arrow` | Avoid for production. The default packaged canonical aliases are small enough as text. |

## Format Ownership

- [arrow_dataset_spec.md](arrow_dataset_spec.md) owns the table schemas and
  manifest checklist.
- This document owns artifact-format choices and rejected alternatives.
- [runtime.md](runtime.md) owns the Python/Rust execution boundary and failure
  semantics.
- [production_inference.md](../production_inference.md) owns production
  prediction operations.
