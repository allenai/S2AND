# Version 1.3 incremental input treatments

All requested label and feature treatments complete. No incremental model fit occurs in this task.
The durable root is `D:/local-unsynced/s2and_v1.3_train_20261012/incremental_treatments_20261013`.
The portable dataset root is `D:/local-unsynced/s2and_v1.3_upload_20261013/linker_source`.
Use this root for upload and future fits. Preserve the original source bundle.

| Treatment | Result |
| --- | --- |
| Empty author lists | Native candidate planners accept valid papers without author child rows. Missing referenced papers still fail. |
| Corrected replay | 256 query views; 40 fresh complete blocks; all 94 previously excluded components enter replay. |
| Gold targets | Refresh 655 replay relations and retain identity targets separately from safe-action targets. |
| Whole-profile safety | Known wrong members prove an unsafe attachment. Complete same-author evidence alone does not replace manual weak-evidence abstentions. |
| Manual review | 740 view-specific judgments: 62 links, 678 abstentions, zero unresolved judgments. Review covers 18,164 member-view records. |
| Applied review | 728 relations apply in 798 rows. Ten protected relations and two alternate scopes remain in the ledger. |
| Exact reuse | 32 prior abstentions have identical members and visible evidence. Of these, 31 enter the final source. |
| External holdout | Remove 231 query identities, 293 views, and 1,146 rows from fit and calibration. Preserve assigned regression test views. |
| Final source | 2,035,853 rows across ten tables; 2,869 additional components; no open requested targets. |
| Feature context | Additional components change retrieval frequencies. A matched probe finds 19 changed values across 11 unchanged-window rows. |
| Final features | All ten tables complete: 2,035,841 rows and 53 features. Retained targets match source labels. No query group disappears. |
| Portable dataset | 180 physical files; 8.64 GB; internal count index; 14 updated manifests. Relocated native loaders accept all 13 datasets. |

Initial-only review packets mask query first and middle names, ORCID, and the query author row in paper rosters.
Each source table retains its original fields, order, types, and Pandas metadata. Source tables contain 15, 16, or 17 fields.

Each retained query keeps its lifecycle split. The source loader accepts all three evaluation splits.
Remove 117 assignments without usable source candidates. Preserve 660 calibration-only test views outside the scored test allocation.
Loader view counts are 5,608 for calibration fit, 5,615 for calibration check, and 10,837 for test.
Of retained table views, 72 lose a positive candidate and eight gain one.

Existing feature filters omit seven positive self-component rows and five rows outside the current retrieval window.
All 235 active selected windows and their 3,931 source rows have features.
Final work uses 68,795,671 pair computations, 37.85 minutes, and 11.49 GiB peak RAM. CPU capacity adds a 4.81-minute wait.

The portable export preserves all 166 payload hashes and 2,035,853 source rows. It has no filesystem links or external operational paths.
`scripts/production/model/export_linker_source_bundle.py` derives dataset names from the manifest. The assembler enforces the same portability checks.
All 16 export tests pass, including patents, source removal, relocation, and Windows junction cases.
Export commands and checks are in `D:/local-unsynced/s2and_v1.3_upload_20261013/run_evidence`.

`review_evidence` preserves complete packets, rosters, decisions, the corrected replay, and copy checks.
`automatic_labels` preserves gold compilers, accepted ledgers, and target proofs. Source application reports remain in `source_bundle_treated`.

`final_features` preserves the command, logs, table checkpoints, output bundle, and coverage report.
Compute uses at most 20 concurrent local cores and 128 GiB. Paid API cost is zero.
The native fix passes 65 focused tests. The support producer passes nine focused tests. Ruff, ty, and Rust format checks pass.

The existing pairwise boosters remain unchanged. EPS remains 0.55.
Inherited linker tests remain regression evidence. Upstream pairwise exposure affects 866 test query identities before current structural filters.
Inherited train and test query papers overlap in 249 cases before current filters.
Cross-source DOI and title identity checks remain outside this audit.

Next, fit the incremental booster and calibration gate from these treated inputs. Then evaluate the reserved test population once.
The existing production trainer recomputes features. It does not consume this prepared feature cache.
