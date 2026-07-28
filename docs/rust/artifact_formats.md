# Rust Artifact Formats

Status date: 2026-07-24

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
| SPECTER | Physical `specter2.arrow` under manifest key `specter`, Arrow fixed-size-list `float32` | Canonical production/eval embedding input. An explicit historical research-training bundle may select `specter.arrow` under the same logical key. |
| Raw-planner batch indexes | `<arrow-stem>.<path-key>.bin` S2AND binary sidecar | Required for canonical filtered reads of signatures, papers, paper authors, and the selected embedding. Current writers emit `arrow_batch_lookup_index` with magic `S2ABI002`; regenerate from the final Arrow IPC files. |
| Name counts | `<main_data_dir>/name_counts_index/` sorted binary index | Sole supported runtime representation when the model selects name-count features; referenced from immutable Arrow manifests. |
| Name aliases | Packaged `s2and_name_tuples_canonical.txt` | Shared runtime default validated directly by Python and passed to Rust as explicit pairs. Avoid per-dataset alias artifacts. |
| Pairwise and linker models | Native LightGBM text plus JSON metadata | Current production model-bundle format. |
| Eval clusters | Existing clusters JSON | Offline evaluation truth only; not part of production inference scoring. |

## Name Counts

The production data-release layout is below. `<main_data_dir>` is selected by
`path_config.json`/`S2AND_PATH_CONFIG`; `s2and/data` is only the checkout's
default placeholder and the large index is not Python package data.

```text
<main_data_dir>/name_counts_index/
  manifest.json
  first.bin
  last.bin
  first_last.bin
  last_first_initial.bin
```

`manifest.json` has exactly `schema_version: "name_counts_index_v3"`,
`normalization_version: "canonical_v2"`, and a `files` object with `first`,
`last`, `first_last`, and `last_first_initial` entries. Each entry requires a
nonempty contained `path`, unsigned `byte_count`, and lowercase SHA-256. The
declared size and digest must match the file. Each path must equal `<kind>.bin`;
subdirectories and alternate filenames are rejected.

The native Rust opener is the runtime authority for manifest, file-digest, and
record validation. Python retains the manifest SHA-256, normalization version,
and resolved file facts returned by that native handle for orchestration and
model binding; it does not run a second manifest-schema validator. Producer
mode and output cardinalities are command metrics, not runtime manifest fields.

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

The retired `generations/<generation>/` nesting is rejected. Production
manifests must use the `name_counts_index` key; runtime boundaries reject the
old `name_counts_index_dir` alias.

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
