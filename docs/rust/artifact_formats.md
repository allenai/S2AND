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
| Name-count Arrow table | `name_counts.arrow` long-form Arrow table | Generation, inspection, and parity debugging only. It is not a request-time runtime fallback for `name_counts_index/`. |
| Name aliases | Packaged filtered text file | Shared runtime default. Avoid per-dataset alias artifacts unless running an explicit experiment. |
| Pairwise and linker models | Native LightGBM text plus JSON metadata | Current production model-bundle format. |
| Eval clusters | Existing clusters JSON | Offline evaluation truth only; not part of production inference scoring. |

## Name Counts

The preferred production publication layout is:

```text
s2and/data/name_counts_index/
  manifest.json
  generations/<generation-id>/
    first.bin
    last.bin
    first_last.bin
    last_first_initial.bin
```

`manifest.json` must have `schema_version: "name_counts_index_v1"` and a
`files` object with `first`, `last`, `first_last`, and `last_first_initial`
entries. Each entry contains a `path`, `record_count`, and `byte_count`; new
writers set each `path` to `generations/<generation-id>/<kind>.bin`. Current
packaged artifacts may still use direct manifest-relative paths such as
`first.bin`; readers follow the manifest and accept both shapes.

Writers publish by writing every binary file into a temporary generation
directory, renaming that directory into `generations/`, validating that every
manifest path exists, and replacing `manifest.json` last. Readers therefore see
either the old complete manifest or the new complete manifest. A failed
overwrite must leave the previous manifest and generation readable.

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
that loads `name_counts.arrow` or `name_counts.pickle` into Python dicts/lists.

The legacy direct-file layout with `first.bin`, `last.bin`, `first_last.bin`,
and `last_first_initial.bin` directly under `name_counts_index/` is accepted
only when referenced by `manifest.json`. New publication runs should regenerate
a manifest-backed generation instead of reusing direct files. Production
manifests should use the `name_counts_index` key; do not emit the old
`name_counts_index_dir` alias.

## Arrow Runtime Writers

`scripts/convert_to_arrow.py` is the reference deployable Arrow-bundle writer.
It writes bounded Arrow IPC file-format tables, regenerates current raw-planner
batch-index sidecars (`S2ABI002`), records physical-layout metrics, and writes
dataset manifests. `scripts/verification/compare_full_predict_arrow_parity.py`
is the reference bounded parity writer and follows the same table and sidecar
helpers for temporary verification artifacts.

New scripts that create S2AND runtime Arrow files should use
`scripts.arrow_conversion_helpers.write_feature_block_arrow_from_anddata(...)`
or `write_feature_block_arrow_tables(...)`, then call
`write_raw_arrow_batch_lookup_indexes(...)` when the artifact may be used by raw
planning. Do not hand-write the batch-index binary format.

## Deprioritized Or Rejected

| Format / approach | Current decision |
|---|---|
| Embedded `name_count_*` columns in `signatures.arrow` | Removed as a preferred/supporting Arrow hot path. Use `name_counts_index/`. |
| SQLite for name counts | Not better for the current exact static point-lookup workload. Revisit only for ad hoc queries, updates, or transaction requirements. |
| Pickle | Not a runtime name-count format. The generator still publishes a source pickle because the v1 provenance/model contract names its SHA; remove it when that contract is versioned. |
| JSON | Fine for fixtures and compatibility loaders; not the runtime target for large table-shaped inference data. |
| Arrow read into Python dict/list before Rust | Defeats the columnar boundary and was measured slower than keeping the hot path in Rust. |
| MessagePack as universal target | Better than JSON for nested legacy payloads, but it preserves the object shape the Rust path is trying to avoid. |
| Parquet as request/runtime hot path | Useful offline, but Arrow IPC is simpler for local runtime bundles and direct Rust readers. |
| Per-dataset `name_pairs.arrow` | Avoid for production. The default packaged filtered aliases are small enough as text. |

## Format Ownership

- `docs/rust/arrow_dataset_spec.md` owns the table schemas and manifest
  checklist.
- This document owns artifact-format choices and rejected alternatives.
- `docs/rust/inference_architecture.md` owns the runtime boundary and
  remaining Python-heavy paths.
