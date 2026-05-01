# Single-Letter Signatures in Giant Blocks: How It Works Today

This document describes what actually happens when a signature has a single-letter
first name (e.g., "H Wang") during inference in a giant block. It traces the full
pipeline from subblocking through featurization and clustering, identifies where
signal is lost, and catalogs the resulting strengths and weaknesses.

## Background

A giant block is a block (same last name + first initial) containing hundreds of
thousands of signatures. Example: "h wang" with ~466K signatures. A large fraction
of these signatures have only an initial as their first name — no full given name
is available in the metadata.

## Pipeline overview

```
giant block
  |
  v
subblocking (name prefix, then middle, then SPECTER)
  |
  +---> multi-letter subblocks --> predict_helper (pairwise clustering)
  |
  +---> single-letter subblocks --> NOT CLUSTERED (in giant block pipeline)
                                     |
                                     v
                               incremental inference
                               (compare against multi-letter seed clusters)
```

## Step 1: Subblocking

**File**: `s2and/subblocking.py`, function `make_subblocks_with_telemetry`

The first thing that happens is a split: signatures whose normalized first name
has length <= 1 are separated from those with longer first names.

**Multi-letter path**: Recursively subdivide by first-name prefix (first 1 letter,
then 2, etc.) until each subblock fits under `maximum_size` (default 15,000). If
first-name prefixes bottom out (everyone shares the same first name), fall back to
middle-name subdivision, then to SPECTER embedding clustering.

**Single-letter path**: All single-letter-first-name signatures go into a single
subblock keyed by the initial (e.g., `"h"`). If that cohort exceeds `maximum_size`,
middle names are used to subdivide it (e.g., `"h|middle=m"`, `"h|middle=x"`).

**Key consequence**: Single-letter signatures are never mixed with multi-letter
signatures during subblocking. A "J Wang" can never end up in the same subblock
as a "Jing Wang" at this stage.

## Step 2: Giant-block clustering (multi-letter only)

**File**: `scripts/giant_block_cluster_retrieval_task.py`

After subblocking, the script classifies each subblock as multi-letter or
single-letter by checking the first signature's normalized first name length.

Only multi-letter subblocks are passed to `predict_helper` for pairwise
clustering. Single-letter subblocks are logged in the manifest but **receive no
clustering**. They produce no predicted clusters and no `signature_to_cluster_id`
entries.

## Step 3: Incremental inference for single-letter signatures

**File**: `s2and/model.py`, method `_predict_incremental_helper`

The intended path for single-letter signatures is incremental inference: treat the
multi-letter clusters from step 2 as "seed clusters" and try to assign each
single-letter signature to the closest seed.

The incremental path works in phases:

1. **Phase A — pairwise scoring**: For each unassigned (single-letter) signature,
   compute pairwise features and distances against every assigned (seed) signature.
   Average the distances per seed cluster.

2. **Precluster-average broadcast**: Before assignment, unassigned signatures are
   first clustered among themselves via `predict_helper`. Then the per-signature
   seed distances are averaged across each precluster and broadcast back to all
   members. This means an individual single-letter signature's distance to a seed
   cluster is replaced by the average distance of its precluster peers.

3. **Phase C — assignment**: Each unassigned signature is assigned to the closest
   seed cluster if the average distance is below `eps` (the clustering threshold).
   A name-compatibility guard prevents merging signatures whose full first names
   would conflict within the seed cluster.

4. **Phase D — leftover reclustering**: Unassigned signatures that were not
   assigned to any seed cluster are clustered among themselves.

## Precluster-average broadcast: smoothing and its side effects

During incremental inference, unassigned signatures are first pre-clustered among
themselves. Then each pre-cluster's per-member seed distances are averaged and
broadcast back. This is equivalent to computing an average distance between the
pre-cluster centroid and each seed cluster.

**Upside**: Reduces noise for individual pairs. If one "H Wang" has a weak SPECTER
match to a seed cluster but its pre-cluster neighbors have strong matches, the
average pulls it closer.

**Downside**: If a pre-cluster is heterogeneous (several different real people
lumped together because they all look like "H Wang"), the averaging blurs the
signal. A signature that would have been close to its true seed cluster on its own
might get pulled away by unrelated pre-cluster members.

The magnitude of this effect in the `h_wang` block is not yet measured.

## Evaluation implication

The precluster-average broadcast has an important consequence for evaluation:
the final cluster for one single-letter query can depend on which other
single-letter queries are present in the same run.

That happens for two reasons:

1. Before seed assignment, unassigned single-letter signatures are clustered
   together and their seed distances are averaged at the precluster level.
2. After seed assignment, any leftover unassigned signatures are clustered
   together again.

So the system does not define a unique "true target cluster" for a query until
we decide what evaluation regime we mean:

- **Joint target**: evaluate a batch of single-letter queries together and allow
  them to influence each other via preclustering and leftover reclustering.
- **Per-query target**: evaluate each single-letter query alone against the
  fixed multi-letter seed clusters, with no interaction from other held-out
  single-letter queries.

This is why retrieval evaluation has a target-definition question in addition to
the usual candidate-recall question.

## Summary of strengths

1. **Single-letter names are explicitly recognized** as a special case with a
   dedicated boolean feature and separate subblocking path.
2. **Paper-metadata features carry the full signal** — SPECTER, references, title
   similarity, journal overlap, and affiliations are unaffected.
3. **Constraints are permissive but not vacuous** — middle-name conflicts and
   last-name mismatches are still enforced.
4. **The incremental pipeline avoids the O(n^2) all-pairs problem** without doing
   one monolithic all-pairs clustering pass over both the single-letter and
   multi-letter population together.

## Summary of weaknesses

1. **The precluster-average broadcast can blur assignment signal** — heterogeneous
   pre-clusters (mixed real identities) dilute individual per-signature distances.
2. **Single-letter subblocks that exceed `maximum_size` are subdivided by middle
   name**, which is often missing or itself a single initial, making the
   subdivision less effective.
3. **In the current giant-block pipeline, single-letter subblocks are not clustered
   at all in the main pass** — they are deferred entirely to incremental inference,
   so their quality depends entirely on the seed clusters being correct and the
   incremental assignment being accurate.

## Historical Retrieval Alternative

The main alternative explored for this problem was cluster-summary retrieval:
instead of comparing a new low-information signature against every signature in
the block, persist one summary per cluster, retrieve the top few clusters inside
the block, and then run the more expensive chooser or incremental logic only on
that shortlist.

This was evaluated by [scripts/eval_cluster_retrieval.py](../scripts/eval_cluster_retrieval.py)
on labeled datasets as a fallback retrieval primitive. It is historical
background, not the current production path.

### Main conclusions

- `hybrid_centroid` was the best default summary representation.
- Exact within-block scan over persisted cluster summaries was already fast
  enough; ANN was not needed for the tested regime.
- On the main `initial_only` slice, `hybrid_centroid` reached:
  - `R@1 = 0.897`
  - `R@5 = 0.976`
  - `R@20 = 0.992`
  - `R@100 = 0.999`
- On the harder non-trivial slice with `candidate_components >= 3`, it still
  reached:
  - `R@1 = 0.862`
  - `R@5 = 0.965`
  - `R@20 = 0.988`

Interpretation:

- retrieval looked strong enough to support a shortlist-first architecture for
  low-information giant-block queries
- the difficult part was no longer "can we retrieve the right cluster family at
  all?", but what to do with the shortlist afterward

### Coauthor-overlap selector result

A coauthor-overlap candidate selector was also tested as a safe accelerator. It
reduced latency and candidate-set size sharply, but recall dropped too much to
use it as the only fallback rule. It remained a useful candidate-generation
feature, not the full answer.

### Relationship to the current work

- this is why `all__hybrid_centroid` became the retrieval baseline
- it does not answer the current any-input `h_wang` question directly
- the current work in [../TODO.md](../TODO.md) and [../TASK.md](../TASK.md) is
  using that retrieval baseline while testing the frozen `title_fast_v1`
  chooser and the first-pass reject-all logic on real `h_wang` data
