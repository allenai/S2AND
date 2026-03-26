# Cluster Summary Retrieval Experiment

**Status:** Wave 1 complete (2026-03-16)

## What This Doc Answers

This document covers the **fallback retrieval step** for incremental clustering.

The intended online decision rule is:

1. **ORCID shortcut**: if the new signature's ORCID maps to an extant cluster or leaf inside the block, route directly there.
2. **Trie/name routing**: if the new signature has an informative first name, use a persisted name router to pick the subblock or fallback leaf.
3. **Cluster retrieval fallback**: if the query is initial-only, the trie misses, or the routed leaf is still too large, rank **clusters** inside the block using persisted cluster summaries. This experiment evaluates this step.
4. **Whole-block fallback or abstain**: if the query is nearly empty or retrieval is low-confidence, send a larger context or leave it unresolved.

The important terminology point is that step 3 is **cluster retrieval**, not subblock retrieval. For low-information signatures, the useful online primitive is "find the right few clusters in the block," then run `predict_incremental` only on those.

## Executive Summary

The main conclusion from the fixed Wave 1 run is:

- Persist **`hybrid_centroid`** summaries per cluster.
- Use **exact within-block scan** over those summaries; ANN is not needed yet.
- For the main production pain point, `initial_only`, `hybrid_centroid` reaches:
  - `R@1 = 0.897`
  - `R@5 = 0.976`
  - `R@20 = 0.992`
  - `R@100 = 0.999`
  - `0.522 ms` mean ranking latency
  - `2.350 ms` p95 ranking latency
- On the harder non-trivial slice with `candidate_components >= 3`, `hybrid_centroid` still reaches:
  - `R@1 = 0.862`
  - `R@5 = 0.965`
  - `R@20 = 0.988`
  - `0.745 ms` mean ranking latency

`hybrid_exemplar_4` is nearly tied and gives slightly smaller payloads, but `hybrid_centroid` is the simpler artifact to persist.

## Motivation

S2AND's incremental pipeline has a scaling problem. When a new signature arrives, the current logic compares it against every signature in its name block. For common names, blocks can contain hundreds or thousands of signatures across many clusters, making incremental updates expensive.

This experiment asks: **can we replace "compare against everything in the block" with "retrieve the right few clusters, then compare only against those"?**

The idea is to store a compact summary for each cluster, score the new signature against those summaries, and materialize only the top-K clusters into S2AND's full incremental path. This matters most for initial-only queries (`J. Smith`) where name routing is weak and blocks are large.

## Key Questions

1. Which cluster summary representation best supports within-block top-K retrieval?
2. How much does retrieval quality degrade when the query has only a first initial?
3. For a fixed recall target, how many clusters and signatures do we need to materialize?
4. How prevalent are truly degenerate cases where whole-block fallback is unavoidable?
5. Is exact scan over cluster summaries fast enough, or will we eventually need ANN?

## Scope Boundary

This experiment does **not** decide how ORCID mappings or tries are stored. It evaluates the fallback retrieval rule for cases where those mechanisms do not already isolate a small candidate set.

## Experimental Design

### Datasets

Nine cluster-labeled datasets from `data/`:

`aminer`, `arnetminer`, `inspire`, `inventors_s2and`, `kisti`, `orcid`, `pubmed`, `qian`, `zbmath`

Excluded:

- `medline` and `augmented` because they are pairwise-only and do not contain cluster labels.
- `inventors` because it uses a non-standard layout.
- `s2and_mini` because it is only a convenience subset.

### Ground-Truth Unit

Retrieval is always evaluated *within a single name block*. The ground-truth unit is a **block-cluster component**: a `(block_key, ground_truth_cluster_id)` pair. Some real clusters span multiple blocks, but an online request only ever needs the within-block slice.

### Evaluation Protocol

Each evaluation example is one held-out signature:

1. Pick a block-cluster component with `n >= 2` members.
2. Hold out 1 signature as the query.
3. Build the cluster summary from the remaining `n - 1` signatures.
4. Score the query against all cluster summaries in its block.
5. Check whether the true cluster appears in the top K.

This simulates a single new signature arriving after an initial full-block clustering.

A secondary multi-holdout protocol is still planned, but was not executed in Wave 1.

### Query Views

Each held-out signature is evaluated under multiple views that progressively strip information from the **query only**. Cluster summaries always use the full original data.

| View | What's available | Purpose |
| --- | --- | --- |
| `full` | All fields as-is | Best-case ceiling |
| `initial_only` | First initial only, no middle name, all other metadata | Main production pain point |
| `initial_only_no_specter` | Same but no SPECTER embedding | Tests embedding dependence |
| `initial_only_sparse_metadata` | Initial only, no SPECTER/coauthors/affiliations | Stress test for weak metadata |
| `initial_only_nearly_empty` | Initial only, nothing else | Reserved for degenerate-case measurement |

### Methods Tested

| Method | Summary representation | Scoring |
| --- | --- | --- |
| `size_prior` | Cluster size only | Rank by cluster size |
| `coauthor_sparse` | Coauthor token sets | Weighted Jaccard overlap |
| `specter_centroid` | Single SPECTER centroid | Cosine similarity |
| `hybrid_centroid` | Sparse features + SPECTER centroid | Weighted combination of embedding and sparse metadata |
| `hybrid_exemplar_4` | Sparse features + 4 exemplar signatures | Max similarity over exemplars + sparse metadata |

Wave 2 candidates not yet run:

- `hybrid_multi_centroid_2`
- `hybrid_multi_centroid_4`
- `hybrid_exemplar_8`
- a learned reranker

### Hard Filters

Before scoring, the evaluator applies three safe pre-filters:

- **ORCID exact match** when the query carries ORCID and at least one candidate does too.
- **Middle-initial conflict** when the query view retains middle initials.
- **Year-range impossibility** for obvious temporal mismatch.

### Metrics

- **Recall@K** for `K in {1, 5, 10, 20, 50, 100}`
- **Payload size** as mean signatures materialized at each K
- **Budgeted recall** under signature-materialization budgets `{25, 50, 100, 250, 500, 1000}`
- **Latency** as per-query wall time for feature extraction, filtering, and ranking
- Slices by dataset, block size, component size, query view, and feature availability

### Success Criteria

**Full-name routing** is production-ready if `Recall@5 >= 0.98` and median materialized-signature fraction is `<= 0.15`.

**Initial-only routing** is production-ready if `Recall@20 >= 0.95` on queries with at least one of `{SPECTER, coauthors, affiliations}` and median materialized-signature fraction is `<= 0.25`.

**Degenerate cases** (`initial_only_nearly_empty`) are acceptable if either recall remains usable under a large budget, the cases are rare enough for whole-block fallback, or the service can safely abstain.

## Wave 1 Setup

- 9 datasets
- up to `2,000` held-out queries per dataset
- `12,273` held-out queries total
- 4 query views:
  - `full`
  - `initial_only`
  - `initial_only_no_specter`
  - `initial_only_sparse_metadata`
- 5 methods:
  - `size_prior`
  - `coauthor_sparse`
  - `specter_centroid`
  - `hybrid_centroid`
  - `hybrid_exemplar_4`
- Rust backend throughout

Per-dataset query counts:

- `aminer`: `2000`
- `arnetminer`: `690`
- `inspire`: `2000`
- `inventors_s2and`: `2000`
- `kisti`: `2000`
- `orcid`: `1052`
- `pubmed`: `390`
- `qian`: `747`
- `zbmath`: `1394`

## Main Results

### `initial_only` (primary production scenario)

| Method | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | Mean latency (ms) | p95 latency (ms) | Mean sigs @5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `size_prior` | 0.415 | 0.717 | 0.820 | 0.896 | 0.963 | 0.988 | 0.529 | 2.373 | 70.97 |
| `coauthor_sparse` | 0.807 | 0.931 | 0.956 | 0.973 | 0.988 | 0.995 | 0.524 | 2.357 | 56.85 |
| `specter_centroid` | 0.810 | 0.951 | 0.974 | 0.987 | 0.995 | 0.999 | 0.521 | 2.326 | 45.37 |
| `hybrid_centroid` | 0.897 | 0.976 | 0.985 | 0.992 | 0.997 | 0.999 | 0.522 | 2.350 | 39.17 |
| `hybrid_exemplar_4` | 0.893 | 0.976 | 0.986 | 0.992 | 0.997 | 1.000 | 0.518 | 2.293 | 34.95 |

Interpretation:

- `hybrid_centroid` is the best default because it has the strongest top-1 performance with excellent top-K recall and a simpler persisted representation.
- `hybrid_exemplar_4` is a close second and wins slightly on payload size.

### `initial_only_no_specter`

| Method | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | Mean latency (ms) | p95 latency (ms) | Mean sigs @5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `size_prior` | 0.415 | 0.717 | 0.820 | 0.896 | 0.963 | 0.988 | 0.093 | 0.359 | 70.97 |
| `coauthor_sparse` | 0.807 | 0.931 | 0.956 | 0.973 | 0.988 | 0.995 | 0.086 | 0.337 | 56.85 |
| `specter_centroid` | 0.415 | 0.717 | 0.820 | 0.896 | 0.963 | 0.988 | 0.084 | 0.334 | 70.97 |
| `hybrid_centroid` | 0.830 | 0.947 | 0.967 | 0.980 | 0.992 | 0.997 | 0.086 | 0.346 | 44.49 |
| `hybrid_exemplar_4` | 0.832 | 0.947 | 0.968 | 0.980 | 0.992 | 0.998 | 0.086 | 0.343 | 40.90 |

Interpretation:

- `specter_centroid` collapses to the `size_prior` ordering when SPECTER is absent.
- The hybrid methods degrade gracefully because they still retain coauthor, venue, affiliation, and year signals.

### Other anchor views

- `full` + `hybrid_centroid`: `R@1 = 0.918`, `R@5 = 0.984`, `R@20 = 0.996`, `R@100 = 0.999`, mean latency `0.513 ms`
- `initial_only_sparse_metadata` + `hybrid_centroid`: `R@1 = 0.840`, `R@5 = 0.960`, `R@20 = 0.989`, `R@100 = 0.999`, mean latency `0.485 ms`

## Harder-Case Slice

The aggregate includes many easy cases. A better view of the real fallback regime is to restrict to queries with `candidate_components >= 3`.

| Method | Queries | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | Mean latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid_centroid` | 8424 | 0.862 | 0.965 | 0.979 | 0.988 | 0.995 | 0.999 | 0.745 |
| `hybrid_exemplar_4` | 8424 | 0.858 | 0.965 | 0.979 | 0.988 | 0.996 | 0.999 | 0.738 |

This is the slice to use when reasoning about "real" cluster retrieval difficulty rather than trivial one- or two-candidate blocks.

## Search Size and Payload

For `hybrid_centroid` on `initial_only`:

- mean candidate clusters per query: `23.51`
- median candidate clusters per query: `5`
- single-candidate queries: `22.0%`
- queries with `>= 3` candidate clusters: `68.6%`
- queries with `>= 6` candidate clusters: `49.0%`

Mean signatures materialized:

- `top 1`: `11.39`
- `top 5`: `39.17`
- `top 10`: `56.93`

For `hybrid_exemplar_4`, mean signatures materialized are:

- `top 1`: `10.65`
- `top 5`: `34.95`
- `top 10`: `51.80`

## Hard-Filter Behavior

| View | ORCID exact | Middle-initial conflict | Year-range |
| --- | ---: | ---: | ---: |
| `full` | 0.000 | 0.119 | 0.010 |
| `initial_only` | 0.000 | 0.000 | 0.011 |

Middle-initial filtering only fires on views that retain middle initials, which is expected.

## Per-Dataset Breakdown (`hybrid_centroid`, `initial_only`)

| Dataset | Queries | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | Mean latency (ms) | p95 latency (ms) | Mean candidates | Median candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aminer` | 2000 | 0.784 | 0.952 | 0.973 | 0.990 | 0.999 | 1.000 | 1.176 | 4.769 | 58.23 | 24 |
| `arnetminer` | 690 | 0.899 | 0.959 | 0.970 | 0.980 | 0.994 | 1.000 | 0.769 | 1.968 | 39.66 | 32 |
| `inspire` | 2000 | 0.893 | 0.979 | 0.989 | 0.994 | 0.999 | 1.000 | 0.249 | 1.090 | 7.36 | 2 |
| `inventors_s2and` | 2000 | 0.898 | 0.954 | 0.966 | 0.975 | 0.986 | 0.997 | 0.906 | 4.790 | 38.15 | 7 |
| `kisti` | 2000 | 0.907 | 0.986 | 0.996 | 0.999 | 1.000 | 1.000 | 0.389 | 1.273 | 18.01 | 10 |
| `orcid` | 1052 | 0.962 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.087 | 0.166 | 4.17 | 4 |
| `pubmed` | 390 | 0.946 | 0.992 | 0.997 | 1.000 | 1.000 | 1.000 | 0.476 | 1.022 | 15.88 | 15 |
| `qian` | 747 | 0.961 | 0.992 | 0.997 | 0.999 | 1.000 | 1.000 | 0.158 | 0.616 | 6.70 | 3 |
| `zbmath` | 1394 | 0.950 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.033 | 0.082 | 1.53 | 1 |

## Offline Dataset Wall Time

This is total experiment time per dataset, not online retrieval latency:

- `inspire`: `569 s`
- `aminer`: `88 s`
- `inventors_s2and`: `61 s`
- `kisti`: `26 s`
- `arnetminer`: `12 s`
- `orcid`: `7 s`
- `pubmed`: `6 s`
- `qian`: `4 s`
- `zbmath`: `4 s`

The expensive part is offline dataset loading and summary construction. Online ranking latency is already comfortably sub-3ms at p95.

## Recommendation

**Primary recommendation:** use `hybrid_centroid` as the default fallback retrieval artifact.

Why:

- It is the strongest simple method on the main production slice.
- It remains strong when SPECTER is missing.
- It uses one centroid per cluster rather than multiple exemplars, which makes persistence and serving simpler.
- Exact within-block scan is already fast enough.

`hybrid_exemplar_4` remains the best follow-up if payload size turns out to matter more than artifact simplicity.

## Concrete Production Rule

Use this online decision rule for incremental clustering:

1. **ORCID available and matched**: route directly to the mapped cluster or leaf.
2. **Informative first name available**: use the trie to pick the subblock or fallback leaf.
3. **Otherwise**: run exact within-block cluster retrieval using persisted `hybrid_centroid` summaries.
4. **If retrieval still looks bad**: fall back to whole-block search or abstain.

This experiment supports step 3. It does not replace steps 1 and 2.

## Next Experiment: Coauthor-Gated Candidate Selection

The next question is whether a cheap coauthor-overlap gate can shrink the candidate set before full cluster scoring.

The evaluator now supports a separate **candidate selector** dimension:

- `all`: the current baseline, score every cluster summary in the block
- `coauthor_overlap_or_all`: if the query has coauthors and any cluster has at least one coauthor-block overlap, rank only those overlapping clusters; otherwise fall back to `all`

This is intentionally a **safe accelerator**, not a hard failure mode for single-author papers.

The metrics to compare are:

- `Recall@K` and budgeted recall against the `all` baseline
- `selector_candidate_components_mean` and `selector_candidate_signatures_mean`
- `candidate_selector_behavior.used_overlap_subset_rate`
- `candidate_selector_behavior.true_component_retained_when_used_overlap`
- latency delta, especially `ranking_latency_ms_mean`

Recommended first comparison:

- query view: `initial_only`
- scorer: `hybrid_centroid`
- candidate selectors: `all` vs `coauthor_overlap_or_all`

Recommended full-run command:

```powershell
$env:S2AND_BACKEND='rust'
uv run python scripts/eval_cluster_retrieval.py `
  --datasets aminer arnetminer inspire inventors_s2and kisti orcid pubmed qian zbmath `
  --limit-queries 2000 `
  --query-views initial_only initial_only_no_specter `
  --methods hybrid_centroid hybrid_exemplar_4 `
  --candidate-selectors all coauthor_overlap_or_all `
  --sampling-query-view initial_only_sparse_metadata `
  --signature-budgets 25 50 100 250 500 1000 `
  --output-dir scratch/cluster_retrieval_coauthor_selector
```

### Executed Result: Full 9-Dataset Run

This run has now been executed at full scale with output in:

- [summary.json](../scratch/cluster_retrieval_coauthor_selector_full_20260316/summary.json)
- [per_query.csv](../scratch/cluster_retrieval_coauthor_selector_full_20260316/per_query.csv)
- [diagnostics.json](../scratch/cluster_retrieval_coauthor_selector_full_20260316/diagnostics.json)

Main takeaway:

- The coauthor-overlap selector is a **good accelerator**.
- It is **not good enough as the only fallback rule**.

For `hybrid_centroid` on `initial_only`:

| Selector | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | Mean latency (ms) | Mean selector candidates | Mean selector signatures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` | 0.900 | 0.979 | 0.988 | 0.993 | 0.998 | 0.999 | 0.718 | 24.82 | 150.19 |
| `coauthor_overlap_or_all` | 0.903 | 0.966 | 0.971 | 0.974 | 0.976 | 0.977 | 0.235 | 5.37 | 55.52 |

Key selector-behavior numbers for `coauthor_overlap_or_all`:

- overlap subset used on `58.9%` of queries
- fallback for missing coauthors on `14.4%`
- fallback for no overlap on `12.4%`
- true-cluster retained on `97.7%` of queries overall
- true-cluster retained on `96.1%` of queries where the overlap subset was actually used

Interpretation:

- Latency drops by about `3x`, and the candidate set shrinks heavily.
- But recall drops too much for a production fallback:
  - `R@20` falls from `0.993` to `0.974`
  - `R@100` falls from `0.999` to `0.977`

The harder non-trivial slice (`candidate_components >= 3`) shows the same pattern:

- `all::hybrid_centroid::initial_only`: `R@20 = 0.990`, mean latency `0.988 ms`
- `coauthor_overlap_or_all::hybrid_centroid::initial_only`: `R@20 = 0.965`, mean latency `0.309 ms`

Conclusion:

- **Do not use any-coauthor-overlap as the only retrieval fallback.**
- It is still a useful **candidate-generation feature** inside a broader fallback, for example:
  - coauthor overlap postings
  - plus another backstop generator such as embedding or affiliation similarity
  - then rank the union with `hybrid_centroid`

## Not Yet Executed

- `initial_only_nearly_empty`
- the secondary multi-holdout protocol
- `all_clusters` and `name_only_router` baselines
- multi-centroid variants
- larger Phase 2 and Phase 3 stress runs

## Cluster Summary Families Considered

1. **Sparse symbolic**: coauthor, affiliation, venue token sets, year stats, name variants
2. **Single-vector embedding**: one SPECTER centroid per cluster
3. **Exemplar**: a small set of representative signatures per cluster
4. **Multi-centroid**: several local centroids for heterogeneous clusters
5. **Hybrid**: sparse features plus centroid or exemplars

Wave 1 evaluated only the first, second, and fifth families directly.

## Scoring Details

For hybrid methods, the score is a weighted sum of:

- `specter_score`
- `coauthor_score`
- `affiliation_score`
- `venue_score`
- `year_score`
- `middle_name_score`

Weights are hand-set in Wave 1. Learned tuning is deferred to Wave 2.

## Output Artifacts

Wave 1 fixed-run artifacts live under:

- [summary.json](../scratch/cluster_retrieval_20260316_full_2k_fixed/summary.json)
- [per_query.csv](../scratch/cluster_retrieval_20260316_full_2k_fixed/per_query.csv)
- [diagnostics.json](../scratch/cluster_retrieval_20260316_full_2k_fixed/diagnostics.json)
- [dataset_census.json](../scratch/cluster_retrieval_20260316_full_2k_fixed/dataset_census.json)
- [summary.json](../scratch/cluster_retrieval_coauthor_selector_full_20260316/summary.json)
- [per_query.csv](../scratch/cluster_retrieval_coauthor_selector_full_20260316/per_query.csv)
- [diagnostics.json](../scratch/cluster_retrieval_coauthor_selector_full_20260316/diagnostics.json)

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
  --output-dir scratch/cluster_retrieval_20260316_full_2k_fixed
```

## Non-Goals

- changing S2AND's clustering semantics
- building an ANN index before exact scan is shown to be insufficient
- building a learned reranker before the simple retrieval headroom is understood
- reporting only top-K recall without payload-size metrics

## Decision Expected

At the end of this work, we want one concrete production answer covering:

- what summary artifact to persist per block
- what retrieval rule to use for full-name queries
- what retrieval rule to use for initial-only queries
- when to fall back to whole-block search or abstain
- whether exact scan is enough or ANN is required
