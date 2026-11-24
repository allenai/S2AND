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
| Split clusters containing multiple ORCIDs | 2026-11 | Audit of 51,757 clusters with all ORCIDs visible and 26,000 mentions per giant. Sixteen held two ORCIDs with at least three mentions each, and four held two with at least five each. | Too few cases to establish precision. Some pairs shared a full name, and others had conflicting middle names. |
| Cluster-pair merge model | 2026-11 | Held-out area under the curve (AUC) was 0.96 versus 0.73 for average distance | Shelved to avoid a third production model |
| Korean name-key subblocking rule | 2026-11-23 | +0.5 to +0.6 ORCID-B3 F1 on `s_park`, but blind judging found roughly 16,700 wrong merges on unlabeled mentions with conflicting first names | ORCID gold cannot see unlabeled records; the rule over-merged unlabeled Korean homonyms |
| Initial-only retrieval depth 50 and looser gate `false_abstain` weight (0.32–0.38 vs 0.25) | 2026-11-23 | Failed the pre-registered blind-judging bar (`s_gupta` added-link consensus wrong rate 22.0% vs 20% cap; `a_silva` 7.5%, `a_khan` 3.8%) | Loosening the miscalibrated 1.4 gate admitted mixed-seed homonyms (BARC `S. K. Gupta`) and one-sided middle-initial errors (`query_has_middle` into zero-middle clusters: 50% consensus wrong) |

The guarded cross-subblock join shipped in `1.0.0b14`. See [ORCID joins](production_inference.md#orcid-joins).

## Pairwise training recipe and linker gate

Measured 2026-11-20 on the `1.3` training data with fixed `1.3.2` hyperparameters. Linker-gate rows compare with the plain refit's linker on the same pairwise model. Other rows compare with `1.3.2`.
Best-EPS F1 uses each held-out dataset's best EPS with the bonus. SPECTER2 giant F1 uses four full giants and 25% of `h_wang`, `s_lee`, and `s_park`. Replay covers 18,162 masked-ORCID queries.
A plain refit moves F1 by up to 0.4 points per dataset and 0.07 pooled across giants, replay accuracy by 0.14 points, and 17 of 10,837 linker test decisions.

| Idea | Date | Result | Why it failed |
| --- | --- | --- | --- |
| Drop name-vetoed pairs from training, main head only | 2026-11-20 | Best-EPS F1 .97345 against .97380 for the plain refit | Vetoed pairs still teach the main head about compatible but different names |
| Drop name-vetoed pairs from both heads | 2026-11-20 | Best-EPS F1 .97257 | The nameless head needs them as "same topic, different person" negatives |
| Sample fewer pairs from big blocks (block weight n or n^1.5 instead of n²) | 2026-11-20 | Best-EPS F1 .97379 and .97319; calibrated EPS fell to 0.27; KISTI and PubMed −0.72 at that EPS | Big blocks hold the hard negatives |
| Year difference capped at 100 instead of 50 | 2026-11-20 | Bit-identical boosters | 0.3% of training pairs span more than 50 years, below the main head's minimum leaf size; old labeled pairs are needed |
| SPECTER2 for non-English papers (language gate removed) | 2026-11-20 | With 25% SPECTER2 omission in augmented data: `a_silva` precision −7.7, pooled giants F1 −0.74, replay accuracy −0.74 | SPECTER2 similarity between Portuguese papers pulls different people together |
| SPECTER2 language gate removed without missing vectors in training | 2026-11-20 | `j_smith` split into 22,876 clusters instead of 7,122. Giant blocks lack vectors for 20–31% of papers, while benchmark blocks have full coverage. | LightGBM learned no missing-value branch and scored missing vectors as cosine −1. The gate supplies missing values for 2.7% of training rows, over half from non-English ZbMATH pairs. |
| SPECTER2 omission in augmented data (25% of variants), language gate kept | 2026-11-20 | Held-out −0.05; sampled pooled giants F1 −0.19 (precision +0.66, recall −1.0); replay accuracy −0.44 (false abstains 641 to 751) | It moves the operating point toward splitting and abstaining |
| Coauthor counts, containment, window and IDF features; affiliation IDF | 2026-11-20 | Pairwise AUC on non-vetoed pairs +0.0001 to +0.0003 over three seeds, below seed noise; KISTI fell in every subset | The existing coauthor features already carry the signal (Python approximation, AUC only) |
| Clip the veto distance (10,000) in the linker's logistic gate | 2026-11-20 | Replay accuracy −0.41; linker test weighted error +2% | The linear gate already discounts those features; clipping shifts it toward abstaining |
| Keep missing pair values out of the linker's pw_* aggregates | 2026-11-20 | Replay accuracy −0.17 (false abstains +75), weighted error −2%; linker test weighted error +1% | Same shift toward abstaining; with clipping too, replay accuracy −0.36 |
| Hard-pair 2.5× weighting in pairwise training (239k rows, 4.6k label flips) | 2026-11-22 | Giant-block pooled F1 −0.41 points (0.9197 vs 0.9238); benchmark overlap +0.25 [−0.17, +0.72] | Over-weighted noisy benchmark boundary pairs and degraded giant clustering |
| Adding 141k giant-block ORCID pairs (`h_wang`/`s_lee`) to pairwise training | 2026-11-22 | Giant-block pooled F1 +0.16 points (short of +0.50 bar), benchmark overlap −0.21 [−0.45, +0.04] | Unstratified giant pairs shifted the decision boundary without clearing refit noise |
| Training the giant linker on fragmented historical production clusters (Stage 4d) | 2026-11-21 | Regressed `h_wang` production-route B3 until rebuilt on 1.4 batch-clustered profiles; rebuilding invalidated 2,185 of 2,187 prior LLM review overrides | Production runs batch clustering before incremental linking, so historical fragmented profiles were out-of-distribution |
| Treating unreviewed once-seen-ORCID queries as all-negative (`negative_singleton_orcid`, Rule A) | 2026-11-22 | Poisoned gate calibration and `h_wang` recall across Stages 4d–4f; removing them closed the `h_wang` gap | 51–70% of once-seen-ORCID queries match an existing un-ORCIDed profile; keep them `unlabeled_singleton_orcid` until reviewed |
| Exposing symmetric `pw_mean_middle_one_missing` to the linker gate on `visibleq:` clones | 2026-11-23 | Offline `v3s` `initial_only` weighted error improved (0.00981 → 0.00846) via a **positive link weight (+0.114)** | 100% of `v3s` `initial_only` queries were synthetic `visibleq:` clones with `author_middle` wiped to `""`, flipping the sign on real production queries with middle initials |
| Adding 15 asymmetric-middle, career-year, and author-count gate features on top of real giant `initial_only` gate recalibration | 2026-11-23 | Leave-one-giant-out (`a_silva`, `s_gupta`, `a_khan`): +0.013 Gold gate AUC (0.8393 → 0.8521) and −67% `q_mid → no_mid` links, but matched-link B3 F1 dipped −0.066 at base links and −0.031 at match links | Almost all the real lift (+0.077 AUC, +3.2 pp precision, +0.170 F1) came from fixing the gate's training/calibration data (real `initial_only` queries + whole-cluster holdouts); new features traded away true large-cluster links on `a_silva` |
| Retraining the 53-feature linker ranker with those 15 new features | 2026-11-23 | Leave-one-giant-out matched-link AUC 0.8345 (vs 0.8393 for gate-only recalibration) and F1 0.93048 (vs 0.93211) | The candidate ranker already orders top-1 well (99.67% forced-choice); the bottleneck is the accept/abstain gate |

## Evaluation traps

| Trap | Date | What happened |
| --- | --- | --- |
| Benchmark comparison of model versions 1.2 and 1.3 | 2026-10-16 | +0.06 to +0.23 points on corrected metadata, −0.24 on original metadata; treat as a tie |
| Per-dataset gain and loss calls on small datasets | 2026-10-16 | 25 gains and 23 losses; the Qian interval is about 2.7 points wide |
| Giant-block proxy labels built from 1.2-model clusters | 2026-10-16 | They favor 1.2; an ORCID-hidden sweep cut the apparent 3 to 8 point regression to under 0.7 |
| Safe-action pair labels for EPS selection | 2026-11 | Abstentions can reflect insufficient evidence to link a pair. Use [ORCID B3](eps_calibration.md#freeze-populations) for identity scoring. |
| EPS sweeps that score bulk clustering only | 2026-10-23 | Production output scored 0.5 to 1.0 points higher at most EPS values; the best EPS barely moved |
| Refit with unchanged hyperparameters | 2026-11-20 | Up to ±0.4 points per small dataset (ArnetMiner, KISTI, PubMed); the calibrated EPS moves within the flat top of the validation curve |
| Benchmark held-out blocks for SPECTER2 changes | 2026-11-20 | Held-out blocks have full SPECTER2 coverage and missed a giant-block collapse; check giant blocks |
| Per-dataset metrics rounded to three decimals before weighting | 2026-11-11 | INSPIRE moved the weighted F1 in steps of 0.0007, the size of the selection tolerance; the report now sums at full precision |
| Stripping final-test ORCID groups from the same-block giant EPS reference | 2026-11-23 | Removed 103,391 of 152,819 ORCID signatures (67.7%) and 552 of 582 authors with ≥50 records (94.8%), shifting dev EPS (0.40) away from all-author (0.38) and prolific-author (0.48) optima |
| Query-level linker splits inside giant blocks with zero giant `no_safe` negatives | 2026-11-23 | 86–97% of giant test queries targeted clusters also positive in training, and 0 of 8,471 giant test queries were `no_safe`, hiding both unseen-author overfitting and the giant false-link rate |
| Including predicted-cluster summaries in blind pair judging packets | 2026-11-23 | Leaked whether the model merged or split every pair in Batch 1 (invalidating 100 error judgments); paper-only packets restored 20/20 ORCID control accuracy |
| Dropping an entire reviewed query on a single `abstain` verdict | 2026-11-23 | Lost 683 of 717 reviewed Stage 4h queries (95.3%) when any candidate in a multi-candidate query received `abstain` (~33% of candidate verdicts were `abstain`); judge only targeted candidates or resolve abstains without discarding the query |
| Under-sizing raw paper-only model-disagreement judging packets | 2026-11-23 | 103 of 200 blind disagreement pairs in Batches 2–4 (51.5%) were "can't tell"; sample roughly 2× the desired decisive count |

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
