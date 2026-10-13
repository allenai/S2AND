# ORCID prefix exclusion, October 13

The producer excludes 14 ORCID groups from the September 25 manual review.
These groups contain distinct identities under one identifier. Correct row owners remain unresolved, so the exclusion applies to each complete group.
The [producer](../../scripts/production/counts/generate_orcid_name_prefix_counts.py) contains the exact identifiers in `EXCLUDED_ORCID_GROUPS`.
It preserves raw sources, the surname rule, EVENT counts, thresholds, curated aliases, and artifact formats.
The separate runtime same-ORCID rule remains unchanged.

| Item | Result |
| --- | --- |
| Complete source | 41,899,086 rows; 9,425,662 ORCID groups |
| Exclusion | 14 groups; 4,617 rows; 3,018 surname-matched EVENT votes |
| Retained source | 41,894,469 rows; 9,425,648 groups; 500,286 EVENT votes |
| Prefix pairs | 6,134 before; 6,079 after; 55 removed; 152 retained counts changed; zero added |
| Thresholds | ORCID EVENT 10; curated alias 2 |
| Data SHA-256 | `10c3b6aef4eb4cb61497db4afe942c93263b19115d00c3121626eebe27022a66` |
| Tuple SHA-256 | `0bcb523ebf8e6f7020883d96cc2822f402f212069214fa23e56b6e7bc0b01f0c` |

The rebuild uses the saved disjoint counters at `D:/local-unsynced/name_count_experiment_20261009_run/prefix/counts/contributions.parquet`.
Their SHA-256 is `e7374b6fcde8b512aa720834e67852c93859324712cd7d6ce1bc5990ab88a5ae`.
The counters separate the reviewed groups before thresholds apply. The rebuild does not scan the complete raw source again.
The 4,617 source rows independently reproduce the excluded counter exactly. The all-group counter also reproduces the previous packaged artifact exactly.
The rebuild takes 8.32 seconds with four CPU workers and no paid APIs.

Command: `uv run --no-sync python scratch/orcid_exclusion_20261013/rebuild.py`.
The run record and publication are at `D:/local-unsynced/s2and_v1.3_prep_20261012/prefix/quarantine_20261013`.
The scratch directory contains the command source and stdout/stderr logs.
The original manual review remains in Git at `fa8f13e:docs/release_evidence/orcid_quarantine_proposal_20260925.json`.

The current pairwise bundle is `D:/local-unsynced/s2and_v1.3_train_20261012/stages/calibrated_no14/production_model_v1.3`.
Only its runtime prefix hash and manifest change. Both booster files and all fit records remain byte-identical.
Both models produce identical probabilities on 512 saved feature rows. EPS remains 0.55 because calibration does not use subblocking.
Candidate binding compares fit provenance for name counts and name tuples. Its runtime check still requires all current artifact hashes.
The original bundle and reports remain intact. Final subblocking evaluation, linker fit, and release acceptance remain pending.
