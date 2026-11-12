# Experiments that did not help

Check this list before you propose an experiment.
Each row records an idea that was measured and then rejected, reverted, or shelved.

Read the results as dated priors, not as proofs.
Most were measured on the 1.3 model and the seven benchmark datasets, between September and November 2026.
The raw evidence lives in maintainer archives, not in this repository.
Measure again when the model, the data, or the hardware changes materially.

"F1" means B-cubed (B3) F1 unless a row says otherwise.
A "point" is 0.01 of F1.
"Giants" are the seven giant blocks named in the [EPS review](eps_calibration.md).

## Where the benchmark loss comes from

Measured 2026-11-11 on the 1.3 held-out test blocks with plain average linkage.

| Clustering rule | Precision | Recall | F1 |
| --- | --- | --- | --- |
| One global cut at 0.53 | .9733 | .9569 | .9648 |
| Best single cut per block, chosen with labels | .9842 | .9616 | .9727 |
| Best pruning of the same tree, chosen with labels | .9939 | .9644 | .9789 |

- No cut rule can gain more than 1.4 points; the tree order holds the rest.
- The 50 worst blocks hold 64% of the loss.
- AMiner holds 38% of the loss with 17% of the mentions.
- Blind judging of the 40 costliest AMiner over-merges called 58% of the cost real and 28% a label split.

The one change that did help is the [close-pair bonus](eps_calibration.md#close-pair-bonus).

## Cut selection and score calibration

| Idea | Date | Result | Why it failed |
| --- | --- | --- | --- |
| Learned cut height replacing the global EPS | 2026-10-12 | Plain EPS won 6 of 7 leave-one-dataset-out folds; macro gain +0.012 points | No single tree-only direction improves every dataset |
| Recalibrated pair scores by name commonness and block size, then rebuilt trees | 2026-10-13 | Every variant lost 0.3 to 4.6 points macro | Block size means prolific authors in INSPIRE and homonyms in AMiner |
| Per-block score shift | 2026-11-11 | Oracle shift with all labels: +0.20 validation, +0.23 test; shift from 20% of people revealed: −0.37, −0.26 | Better pair calibration does not improve clusters |
| Name veto and two-model conflict correction for the main and nameless scores | 2026-10-13 | Passed gates on 300-tree scratch models; +0.001 points on the real 1.3 model at fixed EPS | The scratch gains came from the weak models |
| Two cutoffs (AND or OR) on the two models' cluster averages | 2026-11-11 | Validation F1 .96118 (best AND), .96126 (best OR), .96055 (main only) against .96206 for the 50/50 average | No rule beats the average |
| Optimized monotone distance transform with 17 knots | 2026-11-11 | Test F1 .96809, equal to the two-number close-pair bonus; unstable across rounds | The validation surface is flat at the top |

## Embeddings and features

| Idea | Date | Result | Why it failed |
| --- | --- | --- | --- |
| Replacement embeddings for SPECTER2: an experimental in-house checkpoint, SciNCL, jina v5 small, EmbeddingGemma, Qwen3-Embedding-0.6B | 2026-10-21 | Cosine-only pair AUROC: SPECTER2 ahead by 0.0075 (Qian) and 0.0117 (KISTI) over the best alternative | The SPECTER2 proximity adapter is worth +0.016; it was trained on a same-author task |
| The same experimental checkpoint inside the full model | 2026-10-21 | F1 +0.8 points macro at one global EPS; −0.09 points at each dataset's best EPS | The gain is EPS transfer, not better author separation |
| Citation-weighted name counts | 2026-10-09 | Failed the leave-one-dataset-out gates | See the [archived results](archive/name_count_experiment_20261009_results.md) |
| Corrected labels as training data | 2026-10-04 | AMiner-only pair AUROC +1.0; about ±0.2 in the combined training set | Corrections matter for measurement (AMiner F1 74.1 to 79.3), not for the fit |

## Name rules, subblocking, and attachment

| Idea | Date | Result | Why it failed |
| --- | --- | --- | --- |
| Relaxed name-conflict rules to recover INSPIRE recall | 2026-11-11 | Validation blocks only. No rules: INSPIRE precision −0.014, recall +0.030; ignoring once-seen forms lowered precision in four datasets | The conflicts come from wrong names on single source records, which no general rule separates |
| Attaching one rule-blocked mention to a close cluster | 2026-11-11 | Correct joins 16 of 16 on test at cutoff 0.2; +0.02 to +0.05 points | Too few mentions to justify scoring rule-blocked pairs |
| Holding out the rarest 1% of first names, then attaching them | 2026-11-10 | −0.21 to +0.08 points on the giants | Typo splits are rare in giant blocks |
| Attaching one-letter initials in a separate second pass | 2026-11-11 | +2.38 points on `j_smith` fell to −0.10 on clean inputs | Synthetic `visibleq:` query clones inflated the metric; drop them before scoring giants |
| ORCID repair during subblocking (`use_orcid_subblocking`) | 2026-10-14 | 0 merges on all seven giants | The repair skips any ORCID-linked component over the size cap |
| Stricter linker during the no-middle reunite step | 2026-10-20 | Link costs ×2: F1 +0.05, interval includes 0; ×4: F1 −0.13 | The wrong merges are confident same-name people |
| Size caps on the no-middle reunite step | 2026-10-20 | Best cap recovered 0.3 points of pair precision | Caps trade precision for recall about one to one |
| First-name filter on incremental retrieval | 2026-10-22 | Blind review: about 196 regressions against 147 improvements in 438 changed decisions | Removing competitors makes noisy profiles look like clear winners to the gate |
| Keeping the hard veto when exclusions exceed a share of a profile | 2026-11-10 | The cutoff fired 26 times in simulation; plain eviction was right 26 times | Removed in favor of plain eviction |
| Letting other mentions follow an evicted group | 2026-11-10 | Share of a profile's own papers kept fell from 0.939 to 0.823 | Rejected |

Two results in this area are open, not negative.
A guarded ORCID join after clustering measured +0.60 points on hidden-ORCID signatures and is not implemented.
A cluster-pair merge model reached held-out AUC 0.96 against 0.73 for average distance, and was shelved to avoid a third model.

## Evaluation traps

| Trap | Date | What happened |
| --- | --- | --- |
| Benchmark comparison of model versions 1.2 and 1.3 | 2026-10-16 | +0.06 to +0.23 points on corrected metadata, −0.24 on original metadata; treat as a tie |
| Per-dataset gain and loss calls on small datasets | 2026-10-16 | 25 gains and 23 losses; the Qian interval is about 2.7 points wide |
| Giant-block proxy labels built from 1.2-model clusters | 2026-10-16 | They favor 1.2; an ORCID-hidden sweep cut the apparent 3 to 8 point regression to under 0.7 |
| EPS sweeps that score bulk clustering only | 2026-10-23 | Production output scored 0.5 to 1.0 points higher at most EPS values; the best EPS barely moved |
| Per-dataset metrics rounded to three decimals before weighting | 2026-11-11 | INSPIRE moved the weighted F1 in steps of 0.0007, the size of the selection tolerance; the report now sums at full precision |

## Speed

Dates in this table are accurate to within two days.

| Idea | Date | Result | Why it failed |
| --- | --- | --- | --- |
| Cached linkage across EPS trials | 2026-09-23 | About 1% of release EPS calibration time | Linkage is a small share of the run |
| Early stopping in the pairwise hyperparameter search | 2026-09-24 | 1.06× at a safe patience; 1.24× lost AUC on 3 of 20 configurations | The earlier 1.9× estimate was extrapolated, not measured |
| Background-thread release of planner and featurizer state | 2026-10-21 | The next phase slowed from 2.3 s to 8.9 s; peak memory rose 0.9 GiB | The cost moved instead of disappearing |
| Fat link-time optimization with one codegen unit | 2026-10-22 | One-query `s_lee` slowed from 31.8 s to 36.3 s | n/a |
| Generic 64-bit and 128-bit QuickScorer | 2026-10-21 | 2× to 3× slower per row than the lockstep scorer | n/a |
| Exact top-k pruning in retrieval | 2026-10-22 | Retrieval ran 3× slower | n/a |
| Name-sorted scoring tiles | 2026-10-23 | CPU time fell, wall time rose | Heavy tiles serialized on one thread |
| Lock removal alone in the vendored language detector | 2026-10-22 | No incremental gain; slower on full blocks and on Windows | Two global flags bounced between cores; the shipped version uses thread-local state |
| Integer dot products for embeddings | 2026-10-25 | No gain on the test machine | n/a |
| Deferred block featurizer | 2026-10-25 | Slower, and peak memory rose 1.6 GB | n/a |
| Small `batching_threshold` on incremental requests | 2026-10-20 | A threshold of 100 ran 3.5× slower than the default | The default (`None`) is fastest |
| Block state cached across requests | 2026-10-21 | Not built | It needs a caller protocol change, shared storage, and invalidation |
