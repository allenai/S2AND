# Subblocking

Subblocking keeps pairwise distance construction bounded for large blocks.
`Clusterer.predict_from_arrow(..., batching_threshold=N)` and the EPS sweep both
use the native Rust planner over Arrow inputs.

`N` becomes the maximum subblock size. Every emitted subblock is expected to
have at most `N` signatures.

## Default fallback

Oversized groups that cannot be split by names use the native graph fallback.
The caller receives errors from graph preparation and clustering.

The graph configuration is available on `Clusterer`:

```python
clusterer.subblocking_graph_config = GraphSubblockingConfig()
```

## Partition flow

Subblocking proceeds in this order:

1. Give each compound first name one key token (see [compound first names](#compound-first-names)).
2. Split signatures by first-name prefix.
3. For first-name groups that remain too large, split by middle-name prefix.
4. For groups that are still too large, call the [fallback cluster function](#default-fallback).
5. Merge compatible small subblocks back up to the maximum size.
6. Run the optional same-ORCID repair pass.

All steps use the key names from step 1. Signatures whose key first name has at most one character form the
single-letter group, which is handled later in bulk prediction. Multi-letter subblocks are predicted first,
their resulting clusters become temporary cluster seeds, and single-letter subblocks run through a synthetic
incremental pass so initial-only signatures can attach back to established clusters. Prediction classifies each
subblock by the key first name of its first member. Between these two passes, the
[reunite-only step](#reunite-only-step) reattaches no-middle records that step 4 cut across subblocks.

Follow-on attachment preserves bulk clusters and omits internal hard negatives only
from its local request. See [caller supervision](production_inference.md#caller-supervision-and-dataset-constraints).

Arrow prediction packs active required seed components together within each original
input block. A component larger than the threshold in that block raises before
subblocking. Packing prefers the first full-name subblock in sorted order that
already holds a member. Members native to that subblock precede imported members
of the same component, preserving the destination's classification.
Packing respects the subblock cap. Final clusters can grow beyond that cap through attachment.

## Reunite-only step

A key first name with more than `maximum_size` records in a block is a first-name dead end. Prefix
subdivision cannot split it, so step 3 splits it by middle name. Its records without a key middle name share one
bucket. When that bucket is over the cap, the fallback in step 4 cuts it into pieces of at most the cap. Bulk
prediction runs on each subblock alone, so records of one person in different pieces never meet in bulk clustering.

A record is marked when all of these conditions are true:

- Its original block has more than `maximum_size` records.
- Its key first name has two or more characters and more than `maximum_size` records in that block.
- Its key middle name is empty.
- It is not a member of an active caller-required seed group.

The native Arrow planner returns one bucket per dead end for records that meet the first three conditions.
A record's piece is its final subblock after required seed groups are packed.

After bulk prediction, the step does these operations for each bucket of each original block:

1. Find the bulk clusters that hold only marked records of the bucket. When they occupy fewer than two pieces, the
   bucket does not change.
2. Keep the clusters of the piece with the most of these records as seeds. On a tie, use the first of these pieces
   in key order. The attachment pass needs at least one seed. A merged piece that holds only the small leftover of
   the cut can sort first, so the largest piece keeps the most bulk clusters.
3. Detach the clusters of the other pieces. Send the records of each piece, in key order, to the incremental
   linker as one query group.
4. Then run the initial-only groups with all clusters visible.

A group in step 3 sees only the seeds and the clusters that earlier groups of its bucket made. It never sees a
cluster of another bucket or a cluster that holds an unmarked record. A measurement showed that most attachments
of marked records to clusters with unmarked records were wrong. A bulk cluster with marked records of two buckets
stays as it is. Members of required seed groups are never marked, so their groups do not split or merge
differently. Explicit disallows and caller supervision apply as in initial-only attachment.

The promoted Arrow linker decides each query individually using the name checks of
[initial-only attachment](production_inference.md#incremental-decision-semantics).
No required member is visible, so the set of exempt members is empty.
The `subblocked_reunite` telemetry counts marked records, reunited buckets,
detached and seed clusters, query groups, and their records.
See [production_inference.md](production_inference.md#reunite-only-memory-and-time) for memory and time.

## Dash normalization

Current normalization handles every dash-like given-name separator uniformly. Dash-bound compounds stay together in the
canonical first name regardless of the dash code point: both `Sang-Min` and `Sang<U+2010>Min` normalize to
`first="sang min", middle=""`. Subblock keys then apply the [compound first-name rule](#compound-first-names).

## Compound first names

A canonical first name with two or more tokens is a compound, for example `hui min` from `Hui-Min` or `j d` from
`J.-D.`. A compound keys as one token, so it does not make a prefix bucket of its own. The rule changes only the
first and middle names that form subblock keys. Canonical names, features, name counts and constraints do not
change. The planner applies this rule to the whole request before the partition flow starts.

1. If every token has one character, the key first is the first token. The other tokens go before the canonical
   middle: `j d` with middle `t` keys as first `j`, middle `d t`, in the single-letter group.
2. Else the planner counts two forms in the request. `n_join` counts records with the joined canonical first
   (`huimin`). `n_split` counts records with canonical first `hui` and a middle whose first token is `min`.
3. The compound keys as the joined form with its canonical middle when `n_join <= maximum_size` and
   `(n_join + 1) / (n_split + 1) > 8`.
4. Else the key first is the first token (`hui`), and the key middle is the other tokens before the canonical
   middle (`min`).

A compound with three or more tokens joins all of its tokens (`xiaoyuting`) or keeps only its first token
(`xiao` with middle `yu ting`). The split-form count uses the first two tokens. The ratio keeps distinct names
apart on weak evidence, for example `qi an` and `qian`. A joined first name with more than `maximum_size` records
is already a dead end of prefix subdivision, and a few joined records would only change its middle-name and
fallback groups.

The counts depend on the whole original block. Thus the Arrow route reads the names of every member of an
oversized block to classify its subblocks. A block at or below the threshold is not subblocked, so its
classification uses canonical first names.

## ORCID policy

`Clusterer._subblocking_prefix_counts` selects the packaged merge priors by the
feature contract's `orcid_prefix_counts_data_sha256`. The wheel ships two sets.

| Model | Packaged directory |
| --- | --- |
| `1.3.x` | `s2and/data/` |
| `1.4` | `s2and/data/orcid_prefix_counts_v1_4/` |

Each directory contains `first_k_letter_counts_from_orcid.json` and
`first_k_letter_counts_from_orcid.manifest.json`, which records the name-tuple
data hash used to build the counts. The producer excludes 19 reviewed groups and
preserves the shared-surname rule. The [loader](../s2and/orcid_prefix_counts.py)
validates the count data hash, canonical prefix-pair keys, and positive integer
counts. Missing or malformed artifacts raise before partitioning. Models without
a contract hash use the `1.4` counts.

The planner considers these counts after exact-name and `same_prefix_tokens`
merge candidates.

The native planner supports optional same-ORCID repair. When enabled, the
final repair pass can merge whole subblocks that contain the same ORCID, but it never extracts only ORCID signatures
from an existing subblock. The merge runs only when the combined whole subblocks fit within `maximum_size`; otherwise
the split is preserved and telemetry records the capacity skip.

The ORCID key is canonicalized to match Rust Arrow ingestion: extract the first
structured token containing 15 ASCII digits followed by an ASCII digit or
`X`/`x`, with optional supported dash separators between four-character groups.
The token cannot be immediately adjacent to another ASCII digit or `X`/`x`.
URL and label prefixes are accepted; arbitrary punctuation inside the token is
not stripped. The result uses ASCII hyphens and an uppercase final `X`, for
example `ORCID: 000000021825009x` becomes `0000-0002-1825-009X`.
This validates the token's structure, not its checksum. Values with no matching
token are ignored.
This subblocking policy is independent from same-ORCID hard-link distance constraints.
The `use_orcid_subblocking` flag controls only that final same-ORCID repair; it
does not disable the prefix-pair merge priors described above.

## Graph fallback

The graph fallback builds a capacity-constrained graph over only the oversized fallback group. It uses normalized
SPECTER embeddings plus coauthor and affiliation evidence to score candidate edges, then greedily unions edges while
respecting `target_subblock_size`.

Default `GraphSubblockingConfig` behavior:

- `neighbor_mode="projection"` with `projection_count=12` and `projection_window=12`.
- `min_edge_score=0.30`.
- `component_pack_strategy="edge-greedy"` and `pack_components=True`.
- Exact k-nearest neighbors are available with `neighbor_mode="exact"` and capped by `max_exact_knn_group_size`.
- Sparse coauthor evidence is enabled by default with `sparse_evidence_max_posting_size=8`,
  `sparse_evidence_neighbors=1`, `sparse_evidence_min_weight=0.40`, and affiliations excluded. This adds only
  bounded coauthor-posting edges after projection edges have been scored.
- Adaptive projection, aggregate packing, and local moves are experimental knobs and are off by default.

The native planner requires Arrow signatures and the batch indexes for its
evidence tables. Missing required artifacts and invalid Arrow schemas raise
before clustering. Use the [native threshold sweep](../scripts/production/README.md#subblocking-quality)
to inspect partitions and ORCID-group preservation.

## Prediction

Call `Clusterer.predict_from_arrow(blocks, arrow_dataset, batching_threshold=N,
total_ram_bytes=...)` to partition oversized blocks with the native Rust graph
planner and reuse one Rust featurizer across the emitted subblocks.

`batching_threshold=None` keeps full-block prediction. A positive threshold caps
native subblock size and leaves blocks at or below the cap intact. `total_ram_bytes`
controls pair-batch sizing and allocation checks. Prediction raises if a required
Arrow table, embedding resource, batch index, or the incremental linker is missing.
See [Rust inference](production_inference.md#rust-inference) for initial-only attachment
scope and [caller supervision](production_inference.md#caller-supervision-and-dataset-constraints)
for disallow precedence.

## Telemetry

The native planner records partition telemetry, including:

- input and single-letter/multi-letter signature counts
- first-name dead-end counts
- fallback candidate and invocation counts
- pre-merge and final subblock counts
- ORCID repair capacity skips
- `*_specter_labeled_*` subblock and signature counts, which count every
  graph fallback subblock (keys containing `|specter=`)

## Incremental prediction

Subblocking applies to full-block prediction. On
`Clusterer.predict_incremental_from_arrow`, `batching_threshold` has a different
meaning: it caps the number of unassigned query signatures per linker batch.
See the [incremental contract](production_inference.md#incremental-decision-semantics).

The EPS sweep uses the same native graph planner as production. ORCID-B3 evaluation
keeps ORCID evidence hidden in subblocking and distance constraints to prevent label leakage.
