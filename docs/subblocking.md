# Subblocking

Subblocking keeps pairwise distance construction bounded for large blocks. It is
used by `Clusterer.predict(..., batching_threshold=N)` on the Python/`ANDData`
route and by `Clusterer.predict_from_arrow(..., batching_threshold=N)` on the
Rust Arrow route (see [Python and Rust routing](#python-and-rust-routing)). The
Python boundary is `make_subblocks_with_telemetry(...)`; `make_subblocks(...)`
returns only the final subblock dictionary.

`N` becomes the maximum subblock size. Every emitted subblock is expected to
have at most `N` signatures.

Both routes use the packaged ORCID prefix counts under the [ORCID policy](#orcid-policy).
The committed producer excludes the 14 reviewed groups and preserves the shared-surname rule.
The wheel contains the count JSON and its tuple-dependency manifest.

## Default fallback

Oversized groups that cannot be split by names go to a fallback cluster function:

- Prefix and middle-name splitting define the main subblock structure.
- `Clusterer.predict(...)` and the Arrow route use the graph fallback.
- Direct callers of `make_subblocks(...)` or `make_subblocks_with_telemetry(...)`
  that do not pass a cluster function get the SPECTER fallback,
  `cluster_with_specter(...)`.
- Graph preparation and clustering failures propagate; the selected algorithm does not change at runtime.

The graph configuration is available on `Clusterer`:

```python
clusterer.subblocking_graph_config = GraphSubblockingConfig()
```

## Partition flow

Subblocking proceeds in this order:

1. Give each compound first name one key token (see [compound first names](#compound-first-names)).
2. Split signatures by first-name prefix.
3. For first-name groups that remain too large, split by middle-name prefix.
4. For groups that are still too large, call the fallback cluster function (graph for `Clusterer`; see above).
5. Merge compatible small subblocks back up to the maximum size.
6. Run the optional same-ORCID repair pass.

All steps use the key names from step 1. Signatures whose key first name has at most one character form the
single-letter group, which is handled later in bulk prediction. Multi-letter subblocks are predicted first,
their resulting clusters become temporary cluster seeds, and single-letter subblocks run through a synthetic
incremental pass so initial-only signatures can attach back to established clusters. Prediction classifies each
subblock by the key first name of its first member.

Average linkage does not put a pair at exactly `LARGE_DISTANCE` in one cluster.
Other linkage methods give no such guarantee. The synthetic attachment pass
preserves the computed clusters and omits any internal hard negatives only from
its local request. Cross-group and fresh-signature exclusions remain active.
Caller-supplied incremental seed groups with internal disallows still raise.

Required profiles spanning several subblocks are kept together. When a profile
has a member in a full-name subblock, both Python and Arrow place it there,
preferring the first such subblock in sorted order. Members native to that
subblock stay first so an imported initial does not change its classification.
This keeps compatible full-name neighbours available during bulk clustering.
Packing still respects the subblock capacity; separate full-name groups can
remain separate when they do not fit. The threshold does not cap the size of a
final cluster after initial-only attachment.

## Dash normalization

Current normalization handles every dash-like given-name separator uniformly. Dash-bound compounds stay together in the
canonical first name regardless of the dash code point: both `Sang-Min` and `Sang<U+2010>Min` normalize to
`first="sang min", middle=""`. Subblock keys then apply the [compound first-name rule](#compound-first-names).

The Python implementation uses precomputed canonical first/middle fields when available and reconstructs them with the
same canonicalizer when Rust preprocessing deferred those fields. The Rust Arrow path applies the same
normalization while reading raw name columns.

## Compound first names

A canonical first name with two or more tokens is a compound, for example `hui min` from `Hui-Min` or `j d` from
`J.-D.`. A compound keys as one token, so it does not make a prefix bucket of its own. The rule changes only the
first and middle names that form subblock keys. Canonical names, features, name counts and constraints do not
change. `s2and.subblocking.subblocking_name_parts(...)` and the Rust Arrow planner apply the same rule to the
whole request before the partition flow starts.

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

After Stage 1 artifact promotion, the default merge-prior mapping is loaded lazily from the packaged
`first_k_letter_counts_from_orcid.json` and its required adjacent
`.manifest.json`, then resolved at subblocking entry. The loader validates the
canonical prefix-pair keys and positive integer counts directly; the minimal
manifest records the canonical name-tuple data hash used to generate them.
Missing or malformed files fail closed before partitioning, and production
contracts bind their computed hashes. (The current pre-release tree contains
neither file; see the release note at the top.) These learned prefix-pair
counts are considered
only after exact-name and `same_prefix_tokens` merge candidates, so they do not
override stronger compatibility evidence. A caller may supply an explicit
mapping for a controlled experiment.

`make_subblocks(...)` and `make_subblocks_with_telemetry(...)` have a `use_orcid_subblocking` flag. When enabled, the
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
- Exact kNN remains available with `neighbor_mode="exact"`, but it is capped by `max_exact_knn_group_size`.
- Sparse coauthor evidence is enabled by default with `sparse_evidence_max_posting_size=8`,
  `sparse_evidence_neighbors=1`, `sparse_evidence_min_weight=0.40`, and affiliations excluded. This adds only
  bounded coauthor-posting edges after projection edges have been scored.
- Adaptive projection, aggregate packing, and local moves are experimental knobs and are off by default.

The internal Arrow graph helper loads the union of required `signatures`,
`paper_authors`, and selected embedding rows through raw-planner batch lookup
indexes, then slices that in-memory evidence for each fallback group. It
refuses filtered full scans: `signatures_batch_index`,
`paper_authors_batch_index`, and the selected embedding index (`specter_batch_index`) must be
present, and malformed Arrow schemas or declared missing artifacts raise before
graph clustering starts. This helper is not selected by the public
Python/`ANDData` `Clusterer.predict(...)` route.

`Clusterer.predict(...)` selects the graph implementation directly, and graph
read/prepare/call failures propagate. Direct callers of `make_subblocks(...)`
that do not supply a cluster function retain the explicitly selected legacy
`cluster_with_specter(...)` behavior.

## Python and Rust routing

The public routes are method-based:

- **Python subblocking.** Call
  `Clusterer.predict(blocks, dataset, batching_threshold=N)`. This method takes
  `ANDData`, has no `backend` keyword, and rejects a Rust runtime context.
  Oversized fallback groups call the Python graph fallback.
- **Rust Arrow prediction.** Call
  `Clusterer.predict_from_arrow(blocks, arrow_dataset,
  batching_threshold=N, total_ram_bytes=...)`. This validates the indexed Arrow
  handle, partitions oversized blocks with the native Rust graph
  subblocker, and reuses one Rust featurizer across the emitted subblocks.

The two graph routes have a quality-equivalence contract, not an identical-partition
contract. In the default projection mode, Python and Rust use separate deterministic
random-number generators and seed derivations, so the projection edges and resulting
subblocks may differ for the same signatures. Verification requires both routes to
cover the same signatures, honor the size cap, and preserve required components; it
does not require matching unlabeled partitions. For a fixed candidate edge set,
score ties are resolved deterministically within each route.

`batching_threshold=None` retains full-block prediction. A positive threshold
is the maximum native subblock size; blocks at or below it are unchanged.
Multi-letter subblocks are predicted first, then initial-only groups attach to
those clusters through the promoted Rust incremental linker. `total_ram_bytes`
controls pair-batch sizing and allocation checks; it is not a block
partitioning knob. Missing required Arrow tables, embedding evidence, batch
indexes, or the incremental artifact raise instead of falling back to Python or
`ANDData`.

Both prediction routes keep active required seed components together within
each original input block. A component larger than `batching_threshold` in one
block raises `ValueError` before subblocking; the size of a profile across
different blocks does not trigger this check. Disabling required-seed supervision
also disables this packing and size check.

Initial-only groups can attach only to predictions from their original input
block. A block containing only initials uses bulk prediction, even when other
blocks contain full names. Restoring a profile spanning several input blocks
does not join their output clusters. With constraint supervision enabled,
explicit disallows remain effective during sequential attachment, including
pairs whose other endpoint is processed later. Ignoring required seeds with
`incremental_dont_use_cluster_seeds` does not disable these disallows.

## Telemetry

`make_subblocks_with_telemetry(...)` returns the final subblocks plus telemetry for the partition process, including:

- input and single-letter/multi-letter signature counts
- first-name dead-end counts
- fallback candidate and invocation counts
- pre-merge and final subblock counts
- ORCID repair capacity skips
- `*_specter_labeled_*` subblock and signature counts, which count every
  fallback-produced subblock (keys containing `|specter=`), whichever fallback
  ran

## Incremental prediction

Subblocking applies to full-block prediction. On
`Clusterer.predict_incremental_from_arrow`, `batching_threshold` has a different
meaning: it caps the number of unassigned query signatures per linker batch.
The Python `Clusterer.predict_incremental` API takes no batching parameter. See
[production_inference.md](production_inference.md#incremental-decision-semantics)
for the incremental contract.
