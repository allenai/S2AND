# Joint Safe-Link Recent Failure Report

This report summarizes the main issues in the recent joint safe-link work, the
evidence we now have, the major attempted fixes, and the reasons those fixes
did not solve the underlying problem.

The purpose is not blame. The purpose is to stop repeating the same mistake
pattern.

## Executive Summary

The main recent mistake was repeatedly treating the observed failures as a
training-data or packet-composition problem when the evidence was already
pointing somewhere narrower:

1. S-Lee is not a generic data-shortage problem.
2. S-Lee is not primarily a "correct family missing from the packet" problem.
3. A naive family-level retriever is much worse than the current cluster-level
   retriever.
4. Several interventions looked promising only because they changed train or
   calibration distribution, not because they actually fixed the target failure.

The highest-signal conclusions now are:

- The current cluster retriever already surfaces the correct S-Lee family very
  early for almost all initial-only positive queries.
- The dominant S-Lee failure is more about overconfident within-family scoring
  and abstain behavior than about missing the right family entirely.
- Collapsing a whole family/subblock into one raw aggregated summary destroys
  useful signal and should not be promoted.
- Reweighting, duplicating, or dropping rows can move metrics slightly, but so
  far those moves have not fixed the real failure mode cleanly.

## Current Reference Point

Active official bundle at the time of this report:

- `joint_safe_link_official_stack_20260428e`

Current classic metrics copied from that historical snapshot's `bundle.json`:

| Slice | w5 BA | w25 BA |
| --- | ---: | ---: |
| S2AND | 0.7613 | 0.7611 |
| H-Wang clean | 0.8730 | 0.8758 |
| S-Park | 0.9014 | 0.8875 |
| S-Lee | 0.8064 | 0.7552 |

## What The Error Evidence Says

Primary artifact:

- [scratch/s_lee_w5_manual_analysis_20260413/summary.json](scratch/s_lee_w5_manual_analysis_20260413/summary.json)

Key S-Lee facts at `w5`:

- `4,174` eval queries
- `742` errors
- error breakdown:
  - `560` positive abstains
  - `168` negative wrong links
  - `14` positive wrong links
- query-view breakdown:
  - `4,090` full
  - `84` initial-only
- error-view breakdown:
  - `662` full
  - `80` initial-only

Implications:

- S-Lee is not only an initial-only problem. The initial-only slice is extreme,
  but most raw S-Lee errors are still in full-view queries.
- The dominant error type is positive abstain, not wrong-family linking. That
  already argues against "just add more negatives" or "just prune the packet"
  as the primary fix.
- The initial-only slice is small (`84` queries) but catastrophically weak, so
  it creates a strong temptation to overfit the wrong mechanism.

Qualitatively, the manual packet reviews repeatedly showed two things:

1. Many S-Lee positive abstains already had the correct candidate at rank `1`
   or `2`.
2. Many bad cases lived in a few coarse S* families (`shi`, the broad `so...`
   family, and one smaller `sang...` family).

Those facts were easy to misread. They mean:

- family structure matters
- but "family structure matters" does **not** automatically mean "build a
  simple family retriever"

## Main Wrong Assumptions

### 1. "This is mostly a training-data shortage problem"

This was too broad.

Why it looked plausible:

- S-Lee errors were concentrated in recognizable families.
- The model often abstained, which can look like underexposure.
- Manual review kept producing additional possible cases.

Why this turned out to be incomplete:

- The error pattern is not diffuse across S-Lee. It is structured.
- The current retriever often already finds the right region.
- Reweighting and duplication barely moved the target slice.

### 2. "This is mostly a packet-composition problem"

This was also too broad.

Why it looked plausible:

- The initial-only errors were concentrated in a few giant coarse families.
- Post-hoc inspection made the packet look noisy and implausibly large.

Why this turned out to be incomplete:

- Shrinking the packet after cluster retrieval changed almost nothing in the
  actual decisions when the right candidates were preserved.
- The current cluster-level packet already contains the right family early.

### 3. "If family structure is the issue, use one aggregated summary per family"

This was the clearest wrong move.

Why it looked plausible:

- It is a clean, upstream, production-faithful abstraction.
- It preserves the 15k step-2 clustering and only changes stage-1 retrieval.

Why it failed:

- Aggregating an entire S-Lee family/subblock into one centroid/counter object
  destroyed the discriminative structure.
- The family retriever collapsed onto junk family summaries like `s|middle=c`.

## Attempted Fixes And Why They Failed

### A. Targeted S-Lee Data Duplication / Reweighting

Artifact:

- [scratch/s_lee_targeted_data_fix_experiment_20260413/summary.csv](scratch/s_lee_targeted_data_fix_experiment_20260413/summary.csv)

Representative results:

| Variant | S-Lee w5 | S-Lee w25 | S-Lee initial-only w5 |
| --- | ---: | ---: | ---: |
| baseline | 0.8064 | 0.7552 | 0.5122 |
| `slee_repair_light` | 0.8032 | 0.7547 | 0.5122 |
| `slee_repair_balanced` | 0.8049 | 0.7512 | 0.5061 |
| `slee_repair_negative_heavy` | 0.8071 | 0.7527 | 0.5122 |

Why this failed:

- The best variant only moved S-Lee `w5` by about `+0.0008` and made `w25`
  worse.
- The initial-only slice did not move at all.
- This means the issue is not primarily that the model has not *seen enough*
  of these cases.

Conclusion:

- Adding more of the same row type does not address the actual decision
  boundary.

### B. Candidate / Row Dropping On Known Bad S-Lee Keys

Artifact:

- [scratch/s_lee_candidate_fix_experiment_20260413/summary.csv](scratch/s_lee_candidate_fix_experiment_20260413/summary.csv)

Representative results:

| Variant | S-Lee w5 | S-Lee w25 | S-Lee initial-only w5 | Comment |
| --- | ---: | ---: | ---: | --- |
| baseline | 0.8064 | 0.7552 | 0.5122 | reference |
| `drop_train_shi_only` | 0.8071 | 0.7584 | 0.5122 | tiny move |
| `drop_train_nonshi_plus_shi` | 0.8091 | 0.7607 | 0.5122 | better S-Lee, but by deleting `525` train rows |
| `drop_nonshi_all_splits_plus_train_shi` | 0.8094 | 0.7629 | 0.4841 | better overall S-Lee, worse initial-only |

Why this failed:

- The gains were either tiny or came from deleting rows across train / eval /
  calibration in ways that changed the problem rather than solving it.
- The initial-only slice mostly did not improve.
- Several variants improved S-Lee while damaging other slices or by changing
  calibration thresholds materially.

Conclusion:

- Simple key-based pruning is not robust. It is too blunt and mostly acts by
  changing data composition.

### C. Post-Hoc Family-First Packet Filtering

Artifact:

- [scratch/s_lee_family_first_packet_experiment_20260413/retrain_summary.csv](scratch/s_lee_family_first_packet_experiment_20260413/retrain_summary.csv)

The most informative config was `f3_m3_g5`.

What it did:

- reduced S-Lee initial-only eval packet size from `250` to about `10.5`
  candidates/query
- retained all `82` positive initial-only queries in the retained packet

But the result was:

| Variant | S-Lee w5 | S-Lee w25 | S-Lee initial-only w5 |
| --- | ---: | ---: | ---: |
| baseline | 0.8064 | 0.7552 | 0.5122 |
| `f3_m3_g5` | 0.8050 | 0.7568 | 0.5244 |

Why this failed:

- In inference-only mode, the "safe" packet filters changed `0 / 84`
  initial-only decisions while shrinking the packet drastically.
- That means the current model was already making the same decision from a much
  smaller subset of candidates.
- The only way the end-to-end version moved the metric was by changing the
  train and calibration distribution, not by fixing the retrieval geometry.

Conclusion:

- Post-hoc family pruning is too late. It does not fix the actual decision
  problem.

### D. True Upstream Raw Family Chooser Using Aggregated Family Summaries

Artifact:

- [scratch/s_lee_stage1_family_recall_diagnostic_20260413/summary.json](scratch/s_lee_stage1_family_recall_diagnostic_20260413/summary.json)

This test ranked actual S-Lee step-2 subblocks directly from raw aggregated
family summaries and compared that against the family order already induced by
the current cluster packet.

Results on the `82` positive S-Lee initial-only queries:

| Method | top-1 | top-3 | top-5 | mean positive-family rank |
| --- | ---: | ---: | ---: | ---: |
| direct raw family retriever | 0.0000 | 0.0366 | 0.1220 | 14.83 |
| current cluster-packet family order | 0.0000 | 0.9512 | 1.0000 | 2.33 |

Additional evidence:

- direct family chooser better on `3` queries
- current packet better on `79` queries
- the direct family chooser picked `s|middle=c` as the top family for `81 / 84`
  queries

Why this failed:

- Whole-family aggregation erased the useful cluster-level structure.
- The cluster retriever is already surfacing the right family early.
- The bad abstraction was not "cluster retrieval" but "family = one giant
  centroid/counter object."

Conclusion:

- We should **not** promote a naive family-level retriever.

## What We Actually Learned

### 1. The current retriever is not missing the right S-Lee family

This is now the strongest result in the report.

The current cluster packet already puts the positive family in the top few
family positions almost all the time.

That means:

- building a family chooser may still be worthwhile
- but only if it is based on member-cluster evidence, not a single merged
  family summary

### 2. The biggest S-Lee problem is downstream of "did we see the right family?"

The evidence now points more strongly to:

- overconfident within-family ranking
- overly conservative abstain behavior on positive queries
- feature or objective mismatch inside giant ambiguous families

### 3. We over-focused on the extreme initial-only slice

The initial-only slice is real and severe, but most raw S-Lee errors are still
in full-view queries (`662 / 742`).

That means:

- any theory of S-Lee that only explains initial-only is incomplete
- a clean fix should improve the initial-only slice without pretending the full
  slice is already solved

### 4. Slight metric gains were often false positives

Several variants produced small gains by:

- deleting train rows
- changing calibration composition
- changing thresholds
- trading off other datasets

Those moves are not robust fixes. They are distribution edits.

## Why We Kept Making These Wrong Moves

The repeated pattern was:

1. observe a structured failure
2. choose the easiest surface to perturb
3. get a small metric movement
4. mistake that movement for progress on the true mechanism

Concretely:

- weights and duplication were easy to try
- row drops were easy to try
- post-hoc packet filtering was easy to try
- aggregated family retrieval was conceptually clean

But none of those was tightly matched to the strongest evidence.

## What Not To Do Next

Do not:

- add more S-Lee duplicates or weights without a different supervision unit
- promote row-dropping heuristics as a "solution"
- promote post-hoc family packet filtering
- build a family retriever from one aggregated family summary per subblock
- generalize S-Lee-specific retrieval work to H-Wang or S-Park before the
  S-Lee-specific mechanism actually works

## What To Do Next

The next move should be much narrower:

1. Treat the family as a set of clusters, not as one merged summary.
2. Score families using member-cluster evidence:
   - best cluster score in the family
   - top-2 / top-3 cluster scores
   - family score margin
   - count of plausible clusters in family
   - family-level contradiction counts
3. Keep the current cluster retriever as the primitive.
4. Only add a family stage if it improves over the family order already implied
   by the cluster packet.

If that family-from-member-clusters scorer cannot beat the current packet on
positive-family recall and ambiguity handling, then the problem is not family
selection and we should stop working there.

## Practical Stop / Go Rules

Before promoting another retrieval-stage idea, require:

- positive-family top-3 recall at least as good as the current cluster packet
- clear changed-query evidence, not just packet shrinkage
- no reliance on deleting positives from train or calibration
- no large regressions on S2AND, H-Wang, or S-Park

If a proposal cannot clear those checks in scratch, it should not become a
pipeline change.
