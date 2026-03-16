# Cluster Summary Retrieval Experiment

**Status:** Wave 1 complete (2026-03-16)

## Motivation

S2AND's incremental pipeline has a scaling problem. When a new signature arrives, the current logic compares it against every signature in its name block. For common names, blocks can contain hundreds or thousands of signatures across many clusters, making incremental updates expensive.

This experiment asks: **can we replace "compare against everything in the block" with "retrieve the right few clusters, then compare only against those"?**

The idea is to store a compact *summary* for each cluster (coauthor sets, embedding centroids, etc.), score the new signature against these summaries, and send only the top-K clusters into S2AND's full pairwise pipeline. This is especially important for initial-only queries (first initial + last name), where blocks are largest and metadata is sparsest.

## Key Questions

1. Which cluster summary representation best supports within-block top-K retrieval?
2. How much does retrieval quality degrade when the query has only a first initial (no full name)?
3. For a fixed recall target, how many clusters/signatures do we need to materialize?
4. How prevalent are truly degenerate cases where whole-block fallback is unavoidable?
5. Is exact scan over cluster summaries fast enough, or will we eventually need ANN?

## Experimental Design

### Datasets

Nine cluster-labeled datasets from `data/`:

`aminer`, `arnetminer`, `inspire`, `inventors_s2and`, `kisti`, `orcid`, `pubmed`, `qian`, `zbmath`

Excluded: `medline` and `augmented` (pairwise-only, no cluster labels), `inventors` (non-standard layout), `s2and_mini` (convenience subset).

### Ground-Truth Unit

Retrieval is always *within a single name block*. The ground-truth unit is a **block-cluster component**: a `(block_key, ground_truth_cluster_id)` pair. This matters because some clusters span multiple blocks; we only evaluate the within-block slice that an online request would actually target.

### Evaluation Protocol

Each evaluation example is one **held-out signature**:

1. Pick a block-cluster component with `n >= 2` members.
2. Hold out 1 signature as the query; build the cluster summary from the remaining `n - 1`.
3. Score the query against all cluster summaries in its block.
4. Check whether the true cluster appears in the top K.

This simulates a single new signature arriving after an initial full-block clustering.

A **secondary multi-holdout protocol** (hold out up to 2 signatures from components with `n >= 3`) is planned but not yet executed.

### Query Views

Each held-out signature is evaluated under multiple views that progressively strip information from the *query only* (cluster summaries always use full original data):

| View | What's available | Purpose |
| --- | --- | --- |
| `full` | All fields as-is | Best-case ceiling |
| `initial_only` | First initial only, no middle name, all other metadata | Main production pain point |
| `initial_only_no_specter` | Same but no SPECTER embedding | Tests embedding dependence |
| `initial_only_sparse_metadata` | Initial only, no SPECTER/coauthors/affiliations | Stress test for weak metadata |
| `initial_only_nearly_empty` | Initial only, nothing else | Measures degenerate-case prevalence |

### Methods Tested (Wave 1)

| Method | Summary representation | Scoring |
| --- | --- | --- |
| `size_prior` | Cluster size only | Rank by cluster size (sanity baseline) |
| `coauthor_sparse` | Coauthor token sets | Weighted Jaccard overlap |
| `specter_centroid` | Single SPECTER centroid | Cosine similarity |
| `hybrid_centroid` | Sparse features + SPECTER centroid | Weighted combination of cosine, coauthor/affiliation/venue overlap, year compatibility |
| `hybrid_exemplar_4` | Sparse features + 4 exemplar signatures | Max similarity over exemplars + sparse features |

Wave 2 candidates (not yet run): `hybrid_multi_centroid_2`, `hybrid_multi_centroid_4`, `hybrid_exemplar_8`, learned reranker.

### Hard Filters

Before scoring, three safe pre-filters are applied:

- **ORCID exact match** (if available)
- **Middle-initial conflict** (only fires when the query view retains middle initials)
- **Year-range impossibility**

### Metrics

- **Recall@K** for K in {1, 5, 10, 20, 50, 100}
- **Payload size**: mean signatures materialized at each K (measures downstream cost)
- **Budgeted recall**: recall under signature-materialization budgets {25, 50, 100, 250, 500, 1000}
- **Latency**: per-query wall time for feature extraction + filtering + ranking
- Sliced by dataset, block size, component size, query view, and feature availability

### Success Criteria

**Full-name routing** is production-ready if: Recall@5 >= 0.98, median materialized-signature fraction <= 0.15.

**Initial-only routing** is production-ready if: Recall@20 >= 0.95 on queries with at least one of {SPECTER, coauthors, affiliations}, median materialized-signature fraction <= 0.25.

**Degenerate cases** (`initial_only_nearly_empty`): success means either good recall under large budget, evidence the cases are rare enough for whole-block fallback, or evidence the service should abstain.

## Wave 1 Results

### Setup

- 9 datasets, up to 2,000 held-out queries each (12,273 total)
- 4 query views: `full`, `initial_only`, `initial_only_no_specter`, `initial_only_sparse_metadata`
- 5 methods: `size_prior`, `coauthor_sparse`, `specter_centroid`, `hybrid_centroid`, `hybrid_exemplar_4`

Per-dataset query counts: aminer 2000, arnetminer 690, inspire 2000, inventors_s2and 2000, kisti 2000, orcid 1052, pubmed 390, qian 747, zbmath 1394.

### Main Results: `initial_only` (primary production scenario)

| Method | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | Mean latency (ms) | p95 latency (ms) | Mean sigs @5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `size_prior` | 0.401 | 0.711 | 0.814 | 0.897 | 0.966 | 0.988 | 0.625 | 2.824 | 65.97 |
| `coauthor_sparse` | 0.765 | 0.916 | 0.950 | 0.973 | 0.989 | 0.996 | 0.624 | 2.817 | 40.54 |
| `specter_centroid` | 0.829 | 0.956 | 0.977 | 0.989 | 0.996 | 0.999 | 0.620 | 2.802 | 40.85 |
| `hybrid_centroid` | 0.900 | 0.979 | 0.988 | 0.993 | 0.998 | 0.999 | 0.623 | 2.772 | 41.83 |
| `hybrid_exemplar_4` | 0.897 | 0.978 | 0.987 | 0.993 | 0.998 | 1.000 | 0.626 | 2.811 | 37.02 |

### Results: `initial_only_no_specter`

| Method | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | Mean latency (ms) | p95 latency (ms) | Mean sigs @5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `size_prior` | 0.401 | 0.711 | 0.814 | 0.897 | 0.966 | 0.988 | 0.220 | 0.716 | 65.97 |
| `coauthor_sparse` | 0.765 | 0.916 | 0.950 | 0.973 | 0.989 | 0.996 | 0.212 | 0.676 | 40.54 |
| `specter_centroid` | 0.345 | 0.657 | 0.760 | 0.852 | 0.941 | 0.979 | 0.209 | 0.665 | 61.66 |
| `hybrid_centroid` | 0.849 | 0.960 | 0.977 | 0.986 | 0.995 | 0.999 | 0.212 | 0.673 | 46.10 |
| `hybrid_exemplar_4` | 0.849 | 0.960 | 0.977 | 0.986 | 0.995 | 0.999 | 0.213 | 0.689 | 40.51 |

Key observations:
- `specter_centroid` collapses to a no-signal baseline without SPECTER (R@1 drops from 0.829 to 0.345).
- The hybrid methods degrade gracefully because they still have coauthor/venue/year signals.

### Other view anchors

- `full` + `hybrid_centroid`: R@1=0.919, R@5=0.984, R@10=0.992, R@20=0.996, R@50=0.998, R@100=0.999, mean latency=0.620 ms
- `initial_only_sparse_metadata` + `hybrid_centroid`: R@1=0.837, R@5=0.961, R@10=0.979, R@20=0.989, R@50=0.997, R@100=0.999, mean latency=0.751 ms

### Harder-case slice (blocks with >= 3 candidate components)

Removing trivial 1- and 2-component blocks (20% of queries are single-candidate):

| Method | Queries | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | Mean latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid_centroid` | 8650 | 0.872 | 0.970 | 0.983 | 0.990 | 0.997 | 0.999 | 0.859 |
| `hybrid_exemplar_4` | 8650 | 0.868 | 0.969 | 0.982 | 0.990 | 0.997 | 0.999 | 0.863 |

### Per-Dataset Breakdown (`hybrid_centroid`, `initial_only`)

| Dataset | Queries | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | Mean latency (ms) | p95 latency (ms) | Mean candidates | Median candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aminer` | 2000 | 0.808 | 0.964 | 0.982 | 0.991 | 0.997 | 0.999 | 1.336 | 5.945 | 63.28 | 31 |
| `arnetminer` | 690 | 0.899 | 0.959 | 0.970 | 0.980 | 0.994 | 1.000 | 0.854 | 2.577 | 39.66 | 32 |
| `inspire` | 2000 | 0.871 | 0.972 | 0.984 | 0.991 | 0.999 | 1.000 | 0.401 | 1.631 | 9.12 | 3 |
| `inventors_s2and` | 2000 | 0.916 | 0.969 | 0.978 | 0.986 | 0.993 | 0.999 | 1.073 | 5.431 | 39.24 | 8 |
| `kisti` | 2000 | 0.906 | 0.986 | 0.995 | 0.998 | 1.000 | 1.000 | 0.438 | 1.482 | 18.14 | 10 |
| `orcid` | 1052 | 0.962 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.111 | 0.197 | 4.17 | 4 |
| `pubmed` | 390 | 0.946 | 0.992 | 0.997 | 1.000 | 1.000 | 1.000 | 0.551 | 1.083 | 15.88 | 15 |
| `qian` | 747 | 0.961 | 0.992 | 0.997 | 0.999 | 1.000 | 1.000 | 0.202 | 0.652 | 6.70 | 3 |
| `zbmath` | 1394 | 0.950 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.054 | 0.116 | 1.53 | 1 |

### Retrieval Search Size

- Mean candidate clusters per query: 24.82 (median: 6)
- 20% of queries are trivially single-candidate
- 70.5% of queries have >= 3 candidates

Signatures materialized (mean) for `hybrid_centroid`, `initial_only`:
- Top 1: 11.38, Top 5: 41.83, Top 10: 60.89

For `hybrid_exemplar_4`: Top 1: 10.61, Top 5: 37.02, Top 10: 55.36

### Hard-Filter Behavior

| View | ORCID exact | Middle-initial conflict | Year-range |
| --- | ---: | ---: | ---: |
| `full` | 0.000 | 0.123 | 0.011 |
| `initial_only` | 0.000 | 0.000 | 0.012 |

Middle-initial filtering only fires on views that retain middle initials (as expected).

### Dataset Wall Time (Offline)

This is total experiment time per dataset, not online ranking latency:

inspire 855s, inventors_s2and 131s, aminer 96s, kisti 26s, arnetminer 12s, orcid 9s, pubmed 7s, qian 5s, zbmath 5s

The expensive part is offline dataset loading and summary construction (especially inspire). Online per-query latency is comfortably sub-3ms at p95.

### Dataset Quirks

- `inventors_s2and`: coauthor names with leading/blank whitespace (evaluator normalizes before `compute_block`)
- `orcid`: 76 cluster-member signature IDs absent from the loaded signature map (evaluator skips and logs)

## Recommendations

**Primary finding:** `hybrid_centroid` is the best default. It combines sparse symbolic features (coauthors, affiliations, venues, years, middle names) with a single SPECTER centroid, achieving strong recall across all views while degrading gracefully when signals are missing.

`hybrid_exemplar_4` is extremely close and produces slightly smaller payloads, but requires storing exemplars rather than one centroid per cluster.

**Proposed production plan:**

1. Persist a `hybrid_centroid` summary per cluster within each block.
2. Use exact within-block scan for retrieval.
3. Keep `hybrid_exemplar_4` as a follow-up comparison if smaller payloads matter.
4. Defer ANN until exact scan is shown to be too slow on a production-shaped index.

## Not Yet Executed

- `initial_only_nearly_empty` view
- Multi-holdout secondary protocol
- `all_clusters` and `name_only_router` baselines
- Multi-centroid variants (`hybrid_multi_centroid_2`, `hybrid_multi_centroid_4`)
- Larger Phase 2/3 runs (5,000+ queries, inventors_s2and stress test)

## Phased Execution Plan

### Phase 0: Dataset Census

Compute per-dataset statistics (blocks, components, size distributions, feature availability) before any retrieval runs.

### Phase 1: Small Stratified Sample (N=500/dataset)

Stratify by block size ({2-9, 10-49, 50-199, 200+}), component size ({2, 3-5, 6-10, 11-20, 21+}), and query-information bucket.

### Phase 2: Broader Exact-Scan Run (N=5,000/dataset)

After narrowing the method set based on Phase 1.

### Phase 3: Large-Scale Stress Test

Run the winning methods on `inventors_s2and` to check retrieval quality, exact-scan wall time, and summary memory footprint under a larger, noisier distribution.

## Cluster Summary Families

For reference, the full design space of summary representations considered:

1. **Sparse symbolic** (Family 1): coauthor/affiliation/venue token sets, year stats, name variants. Scored by weighted Jaccard. Cheap baseline.
2. **Single-vector embedding** (Family 2): SPECTER centroid per cluster. Scored by cosine similarity.
3. **Exemplar** (Family 3): M exemplar signatures (M in {2, 4, 8}), chosen by medoid or farthest-point. Scored by max similarity over exemplars.
4. **Multi-centroid** (Family 4): C local centroids (C in {2, 4}). Scored by max cosine to any centroid. For heterogeneous clusters.
5. **Hybrid** (Family 5): sparse features + centroid or exemplars. Weighted linear combination of all signal types. This is what Wave 1 actually tested.

## Scoring Details

For hybrid methods, the candidate score is a weighted sum over:
- `specter_score` (cosine to centroid or max over exemplars)
- `coauthor_score` (token overlap)
- `affiliation_score` (token overlap)
- `venue_score` (token overlap)
- `year_score` (year compatibility)
- `middle_name_score` (middle-name/initial compatibility)

Weights are hand-set in Wave 1; tuning on a held-out subset is planned for Wave 2.

## Output Artifacts

Results are written to dated scratch directories:

- [summary.json](scratch/cluster_retrieval_20260316_full_2k_rerun/summary.json) — dataset/view/method metrics, config, seed
- [per_query.csv](scratch/cluster_retrieval_20260316_full_2k_rerun/per_query.csv) — one row per query with ranks, scores, timings, feature flags
- [failures_topk.csv](scratch/cluster_retrieval_20260316_full_2k_rerun/failures_topk.csv) — queries where true cluster was not in top 20, with score breakdowns
- [diagnostics.json](scratch/cluster_retrieval_20260316_full_2k_rerun/diagnostics.json) — dataset quirks, skipped signatures
- [dataset_census.json](scratch/cluster_retrieval_20260316_full_2k_rerun/dataset_census.json) — feature availability and size distributions

## Run Command

```powershell
$env:S2AND_BACKEND='rust'
uv run python scripts/eval_cluster_retrieval.py `
  --datasets aminer arnetminer inspire inventors_s2and kisti orcid pubmed qian zbmath `
  --limit-queries 2000 `
  --query-views full initial_only initial_only_no_specter initial_only_sparse_metadata `
  --methods size_prior coauthor_sparse specter_centroid hybrid_centroid hybrid_exemplar_4 `
  --sampling-query-view initial_only_sparse_metadata `
  --signature-budgets 25 50 100 250 500 1000 `
  --output-dir scratch/cluster_retrieval_20260316_full_2k_rerun
```

## Non-Goals

- Changing S2AND's clustering semantics.
- Building an ANN index (deferred until exact scan is proven insufficient).
- Building a learned reranker (deferred until simple retrieval headroom is understood).
- Measuring only top-K recall without payload-size metrics.

## Decision Expected

At the end of this work, we want one concrete production answer covering:

- What summary artifact to persist per block
- What retrieval rule to use for full-name queries
- What retrieval rule to use for initial-only queries
- When to fall back to whole-block search or abstain
- Whether exact scan is enough or ANN is required
