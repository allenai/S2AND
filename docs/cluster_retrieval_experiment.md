# Cluster Summary Retrieval Experiment Plan

Status date: 2026-03-16

## Execution Status

Wave 1 exact-scan retrieval has now been rerun with the current evaluator.

This snapshot reflects the corrected harness after:

- view-correct feature flags and sampling buckets
- signature-level feature census instead of one-heldout-per-component prevalence
- dropping empty coauthor blocks instead of treating `""` as a real token
- making documented baselines behave like their documented semantics
- reporting richer payload and latency metrics needed for the stated success gates

Completed:

- Rust-backed exact within-block retrieval over `9` datasets
- up to `2000` held-out queries per dataset
- realized query count: `12273` total
- per-dataset realized queries:
  - `aminer`: `2000`
  - `arnetminer`: `690`
  - `inspire`: `2000`
  - `inventors_s2and`: `2000`
  - `kisti`: `2000`
  - `orcid`: `1052`
  - `pubmed`: `390`
  - `qian`: `747`
  - `zbmath`: `1394`
- query views:
  - `full`
  - `initial_only`
  - `initial_only_no_specter`
  - `initial_only_sparse_metadata`
- methods:
  - `size_prior`
  - `coauthor_sparse`
  - `specter_centroid`
  - `hybrid_centroid`
  - `hybrid_exemplar_4`

Not yet executed in this wave:

- `initial_only_nearly_empty`
- multi-heldout secondary protocol
- `all_clusters`
- `name_only_router`
- multi-centroid variants

Artifacts:

- `scratch/cluster_retrieval_20260316_full_2k_rerun/summary.json`
- `scratch/cluster_retrieval_20260316_full_2k_rerun/diagnostics.json`
- `scratch/cluster_retrieval_20260316_full_2k_rerun/per_query.csv`
- `scratch/cluster_retrieval_20260316_full_2k_rerun/failures_topk.csv`
- `scratch/cluster_retrieval_20260316_full_2k_rerun/dataset_census.json`

Run command:

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

## Results Snapshot

Main production takeaway:

- `hybrid_centroid` is the best default starting point.
- `hybrid_exemplar_4` is extremely close and slightly better on some initial-only top-`K` metrics, but requires storing exemplars rather than one centroid.
- Exact scan already looks cheap enough that ANN should remain out of scope for now.

Overall exact-scan results across all `12273` held-out queries:

### `initial_only`

| method | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | mean latency ms | p95 latency ms | mean signatures @5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `size_prior` | 0.401 | 0.711 | 0.814 | 0.897 | 0.966 | 0.988 | 0.625 | 2.824 | 65.97 |
| `coauthor_sparse` | 0.765 | 0.916 | 0.950 | 0.973 | 0.989 | 0.996 | 0.624 | 2.817 | 40.54 |
| `specter_centroid` | 0.829 | 0.956 | 0.977 | 0.989 | 0.996 | 0.999 | 0.620 | 2.802 | 40.85 |
| `hybrid_centroid` | 0.900 | 0.979 | 0.988 | 0.993 | 0.998 | 0.999 | 0.623 | 2.772 | 41.83 |
| `hybrid_exemplar_4` | 0.897 | 0.978 | 0.987 | 0.993 | 0.998 | 1.000 | 0.626 | 2.811 | 37.02 |

### `initial_only_no_specter`

| method | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | mean latency ms | p95 latency ms | mean signatures @5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `size_prior` | 0.401 | 0.711 | 0.814 | 0.897 | 0.966 | 0.988 | 0.220 | 0.716 | 65.97 |
| `coauthor_sparse` | 0.765 | 0.916 | 0.950 | 0.973 | 0.989 | 0.996 | 0.212 | 0.676 | 40.54 |
| `specter_centroid` | 0.345 | 0.657 | 0.760 | 0.852 | 0.941 | 0.979 | 0.209 | 0.665 | 61.66 |
| `hybrid_centroid` | 0.849 | 0.960 | 0.977 | 0.986 | 0.995 | 0.999 | 0.212 | 0.673 | 46.10 |
| `hybrid_exemplar_4` | 0.849 | 0.960 | 0.977 | 0.986 | 0.995 | 0.999 | 0.213 | 0.689 | 40.51 |

Interpretation:

- `specter_centroid` becomes a true no-signal baseline when no SPECTER embedding is available.
- `coauthor_sparse` likewise degrades sharply on sparse-metadata views because it is now a pure coauthor-only baseline.
- the hybrid methods remain robust because they still have multiple non-name signals available.

Other useful anchors:

- `full`, `hybrid_centroid`: `R@1=0.919`, `R@5=0.984`, `R@10=0.992`, `R@20=0.996`, `R@50=0.998`, `R@100=0.999`, `mean latency=0.620 ms`
- `initial_only_sparse_metadata`, `hybrid_centroid`: `R@1=0.837`, `R@5=0.961`, `R@10=0.979`, `R@20=0.989`, `R@50=0.997`, `R@100=0.999`, `mean latency=0.751 ms`

Average retrieval search size:

- mean candidate clusters per query: `24.82`
- median candidate clusters per query: `6`
- fraction of trivially single-candidate queries: `0.200`
- fraction of harder queries with `candidate_components >= 3`: `0.705`
- `hybrid_centroid`, `initial_only`: mean signatures materialized
  - top `1`: `11.38`
  - top `5`: `41.83`
  - top `10`: `60.89`
- `hybrid_exemplar_4`, `initial_only`: mean signatures materialized
  - top `1`: `10.61`
  - top `5`: `37.02`
  - top `10`: `55.36`

### Harder-case slice: raw `candidate_components >= 3`

This removes the trivial single-component and two-component cases from the aggregate:

| method | queries | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | mean latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid_centroid` | 8650 | 0.872 | 0.970 | 0.983 | 0.990 | 0.997 | 0.999 | 0.859 |
| `hybrid_exemplar_4` | 8650 | 0.868 | 0.969 | 0.982 | 0.990 | 0.997 | 0.999 | 0.863 |

This is the more honest slice for nontrivial routing difficulty.

### Per-dataset spread for `hybrid_centroid`, `initial_only`

| dataset | queries | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | mean latency ms | p95 latency ms | mean candidate clusters | median candidate clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aminer` | 2000 | 0.808 | 0.964 | 0.982 | 0.991 | 0.997 | 0.999 | 1.336 | 5.945 | 63.28 | 31 |
| `arnetminer` | 690 | 0.899 | 0.959 | 0.970 | 0.980 | 0.994 | 1.000 | 0.854 | 2.577 | 39.66 | 32 |
| `inspire` | 2000 | 0.871 | 0.972 | 0.984 | 0.991 | 0.999 | 1.000 | 0.401 | 1.631 | 9.12 | 3 |
| `inventors_s2and` | 2000 | 0.916 | 0.969 | 0.978 | 0.986 | 0.993 | 0.999 | 1.073 | 5.431 | 39.24 | 8 |
| `kisti` | 2000 | 0.906 | 0.986 | 0.995 | 0.998 | 1.000 | 1.000 | 0.438 | 1.482 | 18.14 | 10 |
| `orcid` | 1052 | 0.962 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.111 | 0.197 | 4.17 | 4 |
| `pubmed` | 390 | 0.946 | 0.992 | 0.997 | 1.000 | 1.000 | 1.000 | 0.551 | 1.083 | 15.88 | 15 |
| `qian` | 747 | 0.961 | 0.992 | 0.997 | 0.999 | 1.000 | 1.000 | 0.202 | 0.652 | 6.70 | 3 |
| `zbmath` | 1394 | 0.950 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.054 | 0.116 | 1.53 | 1 |

### Dataset wall time

This is offline experiment wall time, not online ranking latency:

- `inspire`: `855.44s`
- `inventors_s2and`: `130.88s`
- `aminer`: `95.91s`
- `kisti`: `25.98s`
- `arnetminer`: `12.00s`
- `orcid`: `8.67s`
- `pubmed`: `6.71s`
- `qian`: `5.31s`
- `zbmath`: `5.26s`

Interpretation:

- reported `latency_ms` now includes query feature extraction, query-view masking, hard filtering, and ranking
- the main views are still comfortably sub-`3ms` p95 in this harness, but this excludes persisted-index I/O and offline summary construction
- the expensive part remains offline dataset loading and summary construction, especially for `inspire` and `inventors_s2and`
- for production, this still strengthens the case for persisting block-local cluster summaries and doing exact scans first
- hard filters change the full-name path more than the initial-only path because `initial_only` intentionally drops middle initials

### Hard-filter behavior

Observed filter application rates:

- `hybrid_centroid`, `full`
  - ORCID exact: `0.000`
  - middle-initial conflict: `0.123`
  - year-range: `0.011`
- `hybrid_centroid`, `initial_only`
  - ORCID exact: `0.000`
  - middle-initial conflict: `0.000`
  - year-range: `0.012`

Interpretation:

- middle-initial filtering now works, but only on views that actually retain middle initials
- the main initial-only results were not substantially inflated by the previous middle-initial clamp bug, because that query view intentionally blanked middle-name information

## Notes From Execution

Observed dataset quirks handled in the evaluator:

- `inventors_s2and` contained coauthor names with leading or blank whitespace; the evaluator normalizes before `compute_block`
- `orcid` had `76` cluster-member signature ids that were absent from the loaded signature map; the evaluator skips them and records the count in diagnostics

Current recommendation after Wave 1:

1. Persist a `hybrid_centroid` summary per extant cluster within a block.
2. Use exact within-block cluster retrieval for `initial_only` cases.
3. Keep `hybrid_exemplar_4` as the main follow-up comparison if we want to trade a bit more storage for slightly smaller payloads.
4. Defer ANN until exact scan is shown to be too slow on a persisted production-shaped index.

## Purpose

This doc proposes a concrete experiment for production incremental routing.

Production target:

1. Run one-time full-block clustering once.
2. Persist per-block artifacts.
3. For later incremental calls, send only a reduced candidate set back into S2AND rather than the whole block.

This plan focuses on the candidate-retrieval problem:

- For a new signature, rank the clusters in its block.
- Keep the top `K` clusters, or keep clusters until a payload budget is reached.
- Measure whether the true cluster is recovered.

This is intentionally separate from ANN/indexing implementation details. The first experiment should use exact within-block scoring over cluster summaries so we can answer the main question first:

- "What cluster summary and ranking function works well enough to justify a production index?"

Only after that should we optimize retrieval infrastructure.

## Scope

Primary use cases:

- Full-name queries with normal metadata.
- Low-information queries where only a first initial is available.

The main production pain point is the second case. Current incremental logic can become expensive there because it effectively compares the new signature against a very large seed set inside the block.

This experiment will test whether we can replace "compare against everything" with "retrieve the right few clusters."

## Main questions

1. Which cluster summary representation best supports within-block top-`K` retrieval?
2. How much does retrieval quality degrade when the query is reduced to first-initial-only?
3. For a fixed recall target, how many clusters or signatures do we need to materialize?
4. How often are the truly degenerate cases so information-poor that a whole-block fallback is still required?
5. Does a simple exact scan over cluster summaries already suffice, or do we later need ANN?

## Datasets

Use the standard cluster-labeled datasets already present under `data/`:

- `aminer`
- `arnetminer`
- `inspire`
- `inventors_s2and`
- `kisti`
- `orcid`
- `pubmed`
- `qian`
- `zbmath`

Exclude from the primary experiment:

- `medline`: pairwise-only, no cluster JSON
- `augmented`: pairwise-only, no cluster JSON
- `inventors`: raw non-standard layout
- `s2and_mini`: non-primary convenience subset

## Ground-truth unit

The retrieval task is "find the correct cluster within a block."

Use the truth unit:

- `block_cluster_component = ground_truth_cluster_id x block_key`

Reason:

- Production retrieval is within-block.
- Some datasets may contain cluster identities that span more than one block.
- We only want to evaluate the within-block component that the online request would actually target.

For every signature:

- Determine its block key from the signature data.
- Map it to its ground-truth cluster id.
- Group signatures by `(block_key, cluster_id)`.

All summaries and queries should operate on these block-local components.

## Evaluation unit

Each evaluation example is one held-out signature:

- Query: one held-out signature.
- Candidate universe: all remaining block-cluster components in the same block.
- Positive target: the residual component that originally contained the held-out signature.

Success definition:

- Retrieval succeeds at top `K` if the true residual component appears in the top `K` ranked candidates.

## Query views to test

The held-out signature should be evaluated under multiple query views. This lets us separate "cluster summary quality" from "information availability."

### View A: `full`

Use the held-out signature as-is.

Purpose:

- Measures the best-case retrieval ceiling.

### View B: `initial_only`

Modify the query view only:

- Replace first name with first initial.
- Blank middle name.
- Keep coauthors, affiliations, paper metadata, and SPECTER if available.

Purpose:

- Directly matches the main production pain point.

### View C: `initial_only_no_specter`

Same as `initial_only`, but remove SPECTER from the query.

Purpose:

- Tests how much the method depends on embeddings.

### View D: `initial_only_sparse_metadata`

Same as `initial_only`, but additionally remove:

- coauthors
- affiliations

Keep paper year and venue if present.

Purpose:

- Stress test for weak-but-not-hopeless cases.

### View E: `initial_only_nearly_empty`

Same as `initial_only`, but remove:

- coauthors
- affiliations
- SPECTER
- middle name

Purpose:

- Measures prevalence and behavior of cases that may need a fallback or abstain policy.

Important:

- Only mutate the query view.
- Candidate cluster summaries must always be built from the retained cluster members with their original fields.

## Holdout protocol

### Primary protocol

For each block-cluster component with size `n >= 2`:

- Hold out exactly 1 signature.
- Build the cluster summary from the remaining `n - 1` signatures.
- Rank all remaining components in the same block.

This is the cleanest simulation of "one new signature arrives later."

### Secondary protocol

Also test a smaller "multi-new-signature" stress case:

- Hold out `h = min(2, floor(n / 3))` signatures for components where `n >= 3`.
- Build the summary from the remaining signatures.
- Evaluate each held-out signature independently against that same residual summary set.

Purpose:

- Simulates later incremental periods where several new signatures from the same author appear after the initial full clustering snapshot.

### Reproducibility

- Use a fixed random seed for holdout selection.
- Write the sampled held-out signature ids to disk so reruns are exact.

## Sampling plan

Do not start with a full sweep over everything. Follow the repo's small-sample-first rule.

### Phase 0: dataset census

Before any ranking experiment, compute and store:

- number of blocks
- number of block-cluster components
- block size distribution
- component size distribution
- fraction of signatures with:
  - full first name
  - middle name / middle initial
  - at least one coauthor
  - at least one affiliation token
  - SPECTER embedding

This tells us how often the hard cases actually occur.

### Phase 1: small stratified sample

Run on a stratified subset:

- up to `N=500` held-out queries per dataset
- stratified by:
  - block size bucket
  - component size bucket
  - query-information bucket

Suggested block size buckets:

- `2..9`
- `10..49`
- `50..199`
- `200+`

Suggested component size buckets:

- `2`
- `3..5`
- `6..10`
- `11..20`
- `21+`

Suggested query-information buckets:

- has full first name
- initial-only after masking but has SPECTER
- initial-only after masking and no SPECTER
- initial-only after masking and sparse metadata

### Phase 2: broader exact-scan run

After narrowing the configuration set, run a much larger sample:

- all datasets above
- up to `N=5,000` held-out queries per dataset, or all if smaller

### Phase 3: large-scale stress

Run the winning few methods on `inventors_s2and` specifically to check:

- retrieval quality on a larger and noisier distribution
- exact-scan wall time
- memory footprint of summaries

Do not build ANN yet unless exact scan on the winning summary is shown to be too slow.

## Cluster summary ideas to test

The experiment should compare a small, deliberate family of summary representations.

### Family 1: sparse symbolic summaries

Per cluster, store:

- set or weighted multiset of normalized first names
- set of middle initials / middle tokens
- top coauthor blocks
- top affiliation tokens or n-grams
- year statistics
- venue tokens

Candidate score:

- weighted Jaccard or overlap-based scoring
- plus simple year penalty / bonus

Purpose:

- Very cheap baseline
- Easy to reason about

### Family 2: single-vector embedding summaries

Per cluster, store:

- SPECTER centroid
- optional centroid norm and variance

Candidate score:

- cosine similarity between query SPECTER and cluster centroid

Purpose:

- Tests whether one embedding vector per cluster is already enough

### Family 3: medoid / exemplar summaries

Per cluster, store:

- `M` exemplar signatures, where `M in {2, 4, 8}`
- exemplars chosen by medoid or farthest-point diversity

Candidate score:

- max similarity over exemplars
- or average of top-2 exemplar similarities

Purpose:

- Handles clusters with multiple sub-regions better than a single centroid

### Family 4: multi-centroid summaries

Per cluster, store:

- `C` local centroids, where `C in {2, 4}`
- optionally a centroid weight per local centroid

Candidate score:

- max cosine to any centroid
- or weighted max / softmax

Purpose:

- Tests whether heterogeneous clusters need more than one embedding anchor

### Family 5: hybrid summaries

Per cluster, store:

- sparse symbolic features from Family 1
- one centroid or exemplar set from Family 2/3/4

Candidate score:

- weighted linear score over:
  - SPECTER similarity
  - coauthor overlap
  - affiliation overlap
  - venue overlap
  - year compatibility
  - middle-name compatibility

Purpose:

- Most realistic production candidate

## Recommended experiment matrix

Keep Wave 1 small.

### Wave 1 baselines

Test these first:

1. `all_clusters`
   - Rank every cluster equally.
   - Sanity baseline for recall vs payload.

2. `name_only_router`
   - Full-name view only.
   - Route by first-name / middle-name compatibility, no metadata ranking.

3. `coauthor_sparse`
   - Sparse symbolic summary with coauthor blocks only.

4. `specter_centroid`
   - Single SPECTER centroid only.

5. `hybrid_centroid`
   - Sparse symbolic summary + one centroid.

6. `hybrid_exemplar_4`
   - Sparse symbolic summary + 4 exemplars.

### Wave 2 refinements

Only if Wave 1 is promising:

1. `hybrid_multi_centroid_2`
2. `hybrid_multi_centroid_4`
3. `hybrid_exemplar_8`
4. learned reranker on top of retrieved candidates

The learned reranker should be delayed until we know simple hand-built retrieval already has enough headroom.

## Retrieval and ranking functions

### Hard filters

Before scoring, apply only very safe filters:

- ORCID exact match if available
- middle-initial conflict
- impossible year range if a rule is clearly justified
- obvious missing-last-name / block mismatch guards

Avoid aggressive filters early; they can hide good summaries behind bad rules.

### Candidate scoring

Start with exact within-block scoring.

For each query and each candidate cluster summary, compute:

- `specter_score`
- `coauthor_score`
- `affiliation_score`
- `venue_score`
- `year_score`
- `middle_name_score`

Then combine with a simple weighted sum:

`score = w1*specter + w2*coauthor + w3*affiliation + w4*venue + w5*year + w6*middle`

Weights can be:

- hand-set in Wave 1
- tuned on a held-out subset in Wave 2

Important:

- Keep the exact same retrieval API across summary families.
- Swap only the representation and score terms needed for that family.

## Evaluation metrics

Primary metrics:

- Recall@`K` for `K in {1, 5, 10, 20, 50}`
- MRR

Payload metrics:

- cumulative signatures materialized when taking top `K`
- cumulative clusters materialized when taking top `K`
- fraction of block signatures materialized

Budgeted metrics:

- Recall under signature-materialization budget `B`
- suggested budgets: `B in {25, 50, 100, 250, 500, 1000}`

Operational metrics:

- exact-scan wall time per query
- summary build time per block
- summary storage size per block

Coverage / abstain metrics:

- if a score threshold is used, report:
  - coverage
  - Recall@K on covered queries
  - uncovered-query rate

Failure analysis slices:

- by dataset
- by block size bucket
- by component size bucket
- by query view
- by feature-availability bucket

## Success criteria

This experiment should end with a deployment recommendation, not just a leaderboard.

Suggested gates:

### Full-name routing

A full-name method is strong enough for production routing if it achieves:

- Recall@5 at or above `0.98`
- with median materialized-signature fraction at or below `0.15`

### Initial-only routing

An initial-only method is strong enough for production candidate retrieval if it achieves:

- Recall@20 at or above `0.95`
- on `initial_only` queries that still have at least one of:
  - SPECTER
  - coauthors
  - affiliations
- with median materialized-signature fraction at or below `0.25`

### Degenerate initial-only cases

For `initial_only_nearly_empty`, success may be:

- good recall under a large budget, or
- evidence that these cases are rare enough to justify a whole-block fallback, or
- evidence that the service should abstain rather than guess

## Output artifacts

Write outputs to a dated scratch directory, for example:

- `scratch/cluster_retrieval_YYYYMMDD/summary.json`
- `scratch/cluster_retrieval_YYYYMMDD/per_query.csv`
- `scratch/cluster_retrieval_YYYYMMDD/failures_topk.csv`
- `scratch/cluster_retrieval_YYYYMMDD/dataset_census.json`

Recommended contents:

### `summary.json`

- dataset-level metrics
- overall metrics
- per-view metrics
- per-bucket metrics
- config metadata
- seed

### `per_query.csv`

One row per held-out query with:

- dataset
- block key
- component id
- held-out signature id
- query view
- summary config
- top-`K` hit flags
- rank of true cluster
- materialized cluster count
- materialized signature count
- feature availability flags
- timing

### `failures_topk.csv`

Focused slice for debugging:

- rows where true cluster was not in top 20
- top predicted cluster ids
- score breakdown
- block size
- component size

## Implementation plan

Implement in small stages.

### Step 1: census + sampler

Build a script that:

- loads a dataset
- forms block-cluster components
- computes feature-availability stats
- samples held-out queries reproducibly

### Step 2: exact retrieval harness

Build one retrieval harness that:

- accepts a summary builder
- accepts a scoring function
- ranks candidate components exactly within block
- emits per-query results

This exact harness is the main experimental backbone.

### Step 3: Wave 1 summary builders

Implement the 6 Wave 1 methods only.

### Step 4: broader run + failure analysis

Use the exact harness to find:

- which methods are clearly bad
- which methods win on initial-only queries
- where failures come from

### Step 5: production recommendation

Choose one of:

1. Full-name trie router + initial-only cluster retrieval
2. Full-name trie router + initial-only whole-block fallback
3. Cluster retrieval for both full-name and initial-only cases

### Step 6: ANN only if needed

Only after choosing the best summary representation:

- profile exact scan on `inventors_s2and`
- if exact scan is too slow, add ANN for the winning embedding-based summary only

## Current CLI shape

The experiment harness now exists at `scripts/eval_cluster_retrieval.py`. Its interface looks like:

```powershell
uv run python scripts/eval_cluster_retrieval.py `
  --datasets aminer arnetminer inspire inventors_s2and kisti orcid pubmed qian zbmath `
  --query-views full initial_only initial_only_no_specter `
  --methods size_prior coauthor_sparse specter_centroid hybrid_centroid hybrid_exemplar_4 `
  --limit-queries 500 `
  --seed 13 `
  --output-dir scratch/cluster_retrieval_20260315
```

And a broader run:

```powershell
uv run python scripts/eval_cluster_retrieval.py `
  --datasets aminer arnetminer inspire inventors_s2and kisti orcid pubmed qian zbmath `
  --query-views full initial_only initial_only_no_specter initial_only_sparse_metadata `
  --methods hybrid_centroid hybrid_exemplar_4 hybrid_multi_centroid_2 `
  --limit-queries 5000 `
  --seed 13 `
  --output-dir scratch/cluster_retrieval_full_20260315
```

## Notes and non-goals

- This experiment is about retrieval quality, not about changing S2AND clustering semantics yet.
- Do not start with a learned ANN stack.
- Do not start with a learned reranker.
- Do not measure only top-`K` recall; always include payload-size metrics.
- Do not use the full cluster to summarize itself when one of its members is the query; that leaks future information.

## Decision expected from this experiment

At the end of this work we want one concrete production answer:

- what summary artifact to persist per block
- what retrieval rule to use for full-name queries
- what retrieval rule to use for initial-only queries
- when to fall back to whole-block search or abstain
- whether exact scan is enough or ANN is actually required
