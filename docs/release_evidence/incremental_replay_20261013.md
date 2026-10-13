# Bounded version 1.3 incremental replay

This record describes the first bounded audit. The [input treatment record](incremental_treatments_20261013.md) contains the corrected replay and applied decisions.
The later replay accepts valid empty author lists. Its counts replace the initial replay counts for release preparation.

The replay completes. No incremental model fit or source label change occurs.
The current pairwise bundle is `stages/calibrated_no14/production_model_v1.3` beneath `D:/local-unsynced/s2and_v1.3_train_20261012`.
Its manifest SHA-256 is `8bf6de9a470af91ea008823bca153967400f74aaf0ce48d051bddbc6543c5aa0`.
EPS remains 0.55. Retrieval uses the current top-25 default.

| Item | Result |
| --- | --- |
| Query views | 256 across 13 datasets; seed 1111 |
| Sample | All 120 flagged mixed-positive views plus 136 controls |
| Fresh complete-block partitions | 40; each block has at most 400 signatures |
| Partition changes on shared members | 29 of 40 |
| Additional seed coverage | 15 cases; 226 signatures |
| Changed fixed-catalog windows | 157 of 256 |
| Actual work | 181,994 cluster pairs; 86,097 score pairs; 340 seconds |
| Limits | 18 cores; 128 GiB; 900 seconds; 3.2 million cluster pairs; 2 million score pairs |

The replay removes each query from its seeds. It restricts seed retrieval to the current name block.
Fresh partitions use complete public blocks. Component comparisons use member hashes.
Giant blocks retain legacy seed catalogs. This targeted sample does not estimate population error rates.

## Target review subset

The durable evidence root is `D:/local-unsynced/s2and_v1.3_train_20261012/incremental_audit_20261013`.
The copy contains scripts, commands, logs, input sidecars, and reports. All 530 copied files pass SHA-256 comparison.
`replay/subset_summary.json` and `replay/candidate_relations_active.csv` use the actual source-priority labels.
The earlier raw replay summaries do not resolve shadowed source rows.

| Queue beneath the evidence root | Distinct relations | Required action |
| --- | ---: | --- |
| `replay/manual_review_relations.csv` | 620 across 87 queries | Review new or changed relations without a reusable target or complete gold evidence |
| `replay/gold_refresh_relations.csv` | 655 | Derive targets from complete gold |
| `replay/mixed_component_policy_relations.csv` | 111 | Resolve the target policy for mixed components |
| `replay/existing_target_transfers.csv` | 4 | Reuse consistent targets across equivalent member sets |

These queues do not establish incorrect old labels. All 782 sampled rows with an existing target and complete gold agree.
This agreement checks the current any-match rule. It does not establish preservation of earlier manual safe-action judgments.
The active targets have no conflicts. Sixty label rows have different effective member scope.
The current positive target means any candidate member matches the query author. It does not establish component purity.
All 24 lost old-positive relations lack seed eligibility. None is an eligible candidate below rank 25.

## Split overlap

The proposed linker train and calibration inputs contain 231 reserved external-release query signatures: 88 train, 67 calibration-fit, and 76 calibration-check.
They cover 293 views and 1,146 label rows. The existing linker holdout filter removes none.
`replay/exclude_external_holdout_queries.csv` lists the exact identities. Exclude these identities from fit and calibration before the full fit.
The wider same-paper exposure includes 359 fit/calibration queries. Candidate-context exposure remains separate from query-label exposure.

The old linker test contains 787 actual pairwise-fit query endpoints and 79 additional pairwise-validation query endpoints.
Thus, 866 test queries have upstream model exposure. This test does not provide independent whole-pipeline evaluation.
Linker train/test query-signature overlap is zero; their query papers overlap in 249 cases.
These counts use complete lifecycle inputs before feature-stage filters. Cross-source DOI or title identity checks remain outside this audit.
`overlap.json` reproduces all 804,716 saved train pairs and 60,064 validation pairs.
`external_holdout_overlap.json` separates exact identities, same-paper identities, and candidate context.

## Input limits and next steps

Missing or empty author-roster rows prevent native replay for 94 legacy components with 6,174 members; 48 dataset/paper identities lack roster rows.
Sixteen complete-block cases skip fresh partitions for this reason. Resolve the input/runtime contract before the full fit.
The first diagnostic has incorrect cross-block seed scope. Its preserved `unscoped-seeds` files do not support these conclusions.
The corrected run passes all member-scope, query-exclusion, hash, and output-count checks.

Preserve the external holdout. Refresh gold targets. Review the manual subset and mixed-component target policy.
Resolve empty author-roster behavior. Then prepare the incremental fit inputs; full fit remains outside this request.

## Label provenance clarification

Original public benchmark candidate profiles derive from gold author clusters, with the query removed. They do not derive from 1.2 predictions.
Large name-block profiles derive from stored predicted partitions. Their automatic targets use repeated ORCID evidence, with later manual corrections.
Public evaluation also contains a complete 1,842-query manual agent review, plus later initial-only reviews.
The automatic seed-13 allocation reserves approximately 25% for calibration fit, 25% for calibration check, and 50% for test.
The model train pool is separate from this allocation.

The current gold refresh overrides prior weak-evidence abstentions in 19 candidate rows across 12 query views.
`gold_refresh_prior_weak_abstains.csv` beneath the durable evidence root records these matched changes and prior review notes.
Resolve their target meaning before fit. Same-author identity does not establish sufficient visible evidence for a safe action.
The 620 new or changed replay relations require manual review only if they enter the labeled task set.
An unchanged candidate relation can retain its target when its member set and target meaning remain the same.
