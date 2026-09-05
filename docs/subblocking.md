# Subblocking

Subblocking is used by the Python/`ANDData` call
`Clusterer.predict(..., batching_threshold=N)` to keep pairwise distance
construction bounded for large blocks. The core boundary is
`make_subblocks_with_telemetry(...)`; `make_subblocks(...)` returns only the
final subblock dictionary.

For `Clusterer.predict(..., batching_threshold=N)`, `N` becomes the maximum subblock size. Every emitted subblock is
expected to have at most `N` signatures.

## Current Default

Subblocking uses graph fallback for oversized groups that cannot be split by names:

- Prefix and middle-name splitting still define the main subblock structure.
- Oversized groups that cannot be split by names call the graph fallback instead of the old SPECTER fallback.
- Graph preparation and clustering failures propagate; the selected algorithm does not change at runtime.

The graph configuration is available on `Clusterer`:

```python
clusterer.subblocking_graph_config = GraphSubblockingConfig()
```

## Partition flow

Subblocking proceeds in this order:

1. Split signatures by first-name prefix.
2. For first-name groups that remain too large, split by middle-name prefix.
3. For groups that are still too large, call the fallback cluster function. By default this is the graph fallback.
4. Merge compatible small subblocks back up to the maximum size.
5. Run the optional same-ORCID repair pass.

The single-letter first-name path is handled later in bulk prediction. Multi-letter subblocks are predicted first,
their resulting clusters become temporary cluster seeds, and single-letter subblocks run through a synthetic
incremental pass so initial-only signatures can attach back to established clusters.

## Dash normalization

Current normalization handles every dash-like given-name separator uniformly. Dash-bound compounds stay together in the first
name for subblocking regardless of the dash code point: both `Sang-Min` and `Sang<U+2010>Min` normalize to
`first="sang min", middle=""`.

The Python implementation uses precomputed canonical first/middle fields when available and reconstructs them with the
same canonicalizer when Rust preprocessing deferred those fields. The Rust Arrow path applies the same
normalization while reading raw name columns.

## ORCID policy

After Stage 1 artifact promotion, the default merge-prior mapping is loaded lazily from the packaged
`first_k_letter_counts_from_orcid.json` and its required adjacent
`.manifest.json`, then resolved at subblocking entry. The loader validates the
canonical prefix-pair keys and positive integer counts directly; the minimal
manifest records the canonical name-tuple data hash used to generate them.
Missing or malformed files fail closed before partitioning, and production
contracts bind their computed hashes. The current pre-release tree contains
neither file. These learned prefix-pair counts are considered
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
- Adaptive projection, aggregate packing, and local moves are still experimental knobs and are off by default.

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
still controls pair-batch sizing and allocation checks; it is not a block
partitioning knob. Missing required Arrow tables, embedding evidence, batch
indexes, or the incremental artifact raise instead of falling back to Python or
`ANDData`.

## Telemetry

`make_subblocks_with_telemetry(...)` returns the final subblocks plus telemetry for the partition process, including:

- input and single-letter/multi-letter signature counts
- first-name dead-end counts
- fallback candidate and invocation counts
- pre-merge and final subblock counts
- ORCID repair capacity skips
- final SPECTER-labeled subblock counts

## Incremental routing

Incremental prediction has two supported routes:

- **Promoted Rust linker.** `Clusterer.predict_incremental_from_arrow` requires
  an open `ArrowDataset` and creates a Rust runtime context when none is supplied.
  An explicit context must select Rust. Retrieval and scoring run directly
  against its Arrow tables using the pinned native ABI.
- **Python helper.** `Clusterer.predict_incremental` operates on `ANDData` and
  creates a Python runtime context when none is supplied. An explicit context
  must select Python. It covers partition coverage but does not implement
  batched incremental routing.

The APIs do not inspect native capabilities or fall back between implementations. A missing or mismatched pinned
Rust extension is an error on the Arrow route.

`batching_threshold` has two separate entry points. On full-block `Clusterer.predict`, it caps subblock size. On
`Clusterer.predict_incremental_from_arrow`, it caps the number of unassigned query signatures per linker batch.
The standalone Python `Clusterer.predict_incremental` API does not take a batching parameter.

See [production_inference.md](production_inference.md#incremental-decision-semantics) for the full
caller-facing contract.
