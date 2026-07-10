# Subblocking

Subblocking is used by full-block `Clusterer.predict(...)` to keep pairwise distance construction bounded for large
blocks. The core boundary is `make_subblocks_with_telemetry(...)`; `make_subblocks(...)` returns only the final
subblock dictionary.

For `Clusterer.predict(..., batching_threshold=N)`, `N` becomes the maximum subblock size. Every emitted subblock is
expected to have at most `N` signatures.

## Current Default

Subblocking uses graph fallback for oversized groups that cannot be split by names:

- Prefix and middle-name splitting still define the main subblock structure.
- Oversized groups that cannot be split by names call the graph fallback instead of the old SPECTER fallback.
- For Python/`ANDData` graph fallback, graph preparation or fallback-call IO failures run the old Python
  `cluster_with_specter(...)` path for that fallback group. Arrow-backed Rust production graph failures raise instead
  of falling back to `ANDData`.

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

## Dash compatibility repair

Subblocking currently has a localized legacy repair for dash-like given names. Canonical first/middle normalization
keeps all dash-like given-name compounds together, but the subblocking key layer then preserves different behavior for
ASCII hyphen versus Unicode dash-like characters:

- ASCII-hyphen compounds stay together for subblocking, for example `Sang-Min` -> `first="sang min", middle=""`.
- Non-ASCII dash compounds spill to first + middle for subblocking, for example `Sang<U+2010>Min` ->
  `first="sang", middle="min"`.

This is not intended as permanent semantic policy. It is a measured compatibility repair for current artifacts: uniform
single-key alternatives regressed real subblocking quality on the active `s_lee`, `s_park`, and `h_wang` checks. Keep
the repair scoped to subblocking keys; do not copy it into generic `normalize_text(...)`, title/venue normalization,
ORCID parsing, or pairwise feature normalization. The Python implementation uses precomputed normalized first/middle
fields when present and only inspects the raw first name to detect the dash form. The Rust Arrow path reads
`author_first`/`author_middle`, normalizes them internally, and applies the same subblocking-only repair.

Any replacement must meet the current quality gates on full artifacts before landing. The migration notes in
[normalization_migration_blocked.md](normalization_migration_blocked.md) track the tested alternatives and the path to a
cleaner canonical policy.

## ORCID policy

`make_subblocks(...)` and `make_subblocks_with_telemetry(...)` have a `use_orcid_subblocking` flag. When enabled, the
final repair pass can merge whole subblocks that contain the same ORCID, but it never extracts only ORCID signatures
from an existing subblock. The merge runs only when the combined whole subblocks fit within `maximum_size`; otherwise
the split is preserved and telemetry records the capacity skip.

The ORCID key is canonicalized to match Rust Arrow ingestion: keep digits and `X`/`x`, require exactly 16 ORCID
characters, uppercase the check digit, and format as `0000-0000-0000-0000`. Blank or invalid values are ignored.
This subblocking policy is independent from same-ORCID hard-link distance constraints.

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

When Arrow paths are available, the graph fallback is Arrow-backed. Before fallback calls, `prepare(...)` receives the
actual fallback-signature groups and loads the union of required `signatures`, `paper_authors`, and selected embedding rows
into memory through the raw-planner batch lookup indexes. Each fallback call then slices that in-memory evidence for
its group. Arrow graph subblocking refuses filtered full scans: `signatures_batch_index`,
`paper_authors_batch_index`, and the selected embedding index (`specter_batch_index` or `specter2_batch_index`) must be
present, and malformed Arrow schemas or declared missing artifacts raise before graph clustering starts. Without Arrow
paths, the graph fallback uses the active `ANDData` object.

For Python/`ANDData` fallback, graph fallback is intentionally wrapped with the old Python SPECTER fallback. Graph
failures are not swallowed: warnings are logged, telemetry records the failure, and then
`cluster_with_specter(...)` runs for the affected group. Arrow-backed Rust production graph fallback does not use this
legacy recovery path; Arrow graph read/prepare/call failures propagate.

## Python and Rust routing

Subblocking has two supported routes:

- **Pure Python.** Direct calls to `make_subblocks(...)` and `make_subblocks_with_telemetry(...)` use the original
  Python implementation.
- **Arrow-native Rust.** `Clusterer.predict(...)` can route oversized blocks to Rust when
  the call's resolved backend is Rust and indexed Arrow signature rows are available.

For `Clusterer.predict(...)` with Arrow paths and a Rust-resolved backend, Rust Arrow subblocking is strict and always
uses the native graph fallback. It is used only when all of these are true:

- `Clusterer.predict(..., backend="rust")` is used, or its `runtime_context` resolves to Rust.
- `arrow_paths["signatures"]` is present.
- `arrow_paths["signatures_batch_index"]` is present.

That path calls `_make_subblocks_with_telemetry_arrow_rust(...)`, so Rust loads the name and ORCID rows needed for subblocking from
Arrow. If an Arrow-backed Rust production prediction is missing either artifact, prediction raises a structured
missing-artifact error instead of silently using `ANDData` partitioning. Python subblocking orchestration remains
available through explicit Python/compatibility routes and direct `make_subblocks(...)` calls.

In Python orchestration, oversized fallback groups call the Python graph fallback callable. In Arrow-native Rust
orchestration, oversized fallback groups are handled inside Rust; Rust does not call back into Python.

## Telemetry

`make_subblocks_with_telemetry(...)` returns the final subblocks plus telemetry for the partition process, including:

- input and single-letter/multi-letter signature counts
- first-name dead-end counts
- fallback candidate and invocation counts
- pre-merge and final subblock counts
- ORCID repair capacity skips
- final SPECTER-labeled subblock counts

`Clusterer.predict(...)` also records graph hook telemetry on `_last_graph_subblocking_telemetry` and
`_last_arrow_graph_subblocking_telemetry`:

- `enabled`, `mode`, `source`, and `candidate_signature_count`
- `arrow_load_seconds` and `arrow_load_metrics`
- `fallback_invocation_count` and per-group `fallback_stats`
- `legacy_fallback_invocation_count`
- `graph_prepare_failed`, `graph_prepare_error`, and `graph_fallback_errors`

`fallback_stats` includes per-group graph details such as candidate edge count, raw and packed component counts,
maximum component size, edge-build seconds, and total fallback seconds.

## Incremental routing

Incremental prediction has two supported routes:

- **Promoted Rust linker.** `Clusterer.predict_incremental_from_arrow_paths` requires a Rust runtime context and a
  validated Arrow artifact bundle. Retrieval and scoring run directly against those Arrow tables using the pinned
  native ABI.
- **Python helper.** `Clusterer.predict_incremental` operates on `ANDData` with a Python runtime context. It covers
  partition coverage but does not implement batched incremental routing.

The APIs do not inspect native capabilities or fall back between implementations. A missing or mismatched pinned
Rust extension is an error on the Arrow route.

`batching_threshold` has two separate entry points. On full-block `Clusterer.predict`, it caps subblock size. On
`Clusterer.predict_incremental_from_arrow_paths`, it caps the number of unassigned query signatures per linker batch.
The standalone Python `Clusterer.predict_incremental` API does not take a batching parameter.

See [production_inference.md](production_inference.md#large-blocks-and-incremental-inference) for the full
caller-facing contract.
