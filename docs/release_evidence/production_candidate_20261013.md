# Patent-free version 1.3 production candidate

Release `1.3` excludes patent data from pairwise and incremental supervision.
This candidate replaces the patent-inclusive package from commit `7e183c1`.
The repository model is `artifacts/common/v1.3/production_model_v1.3`.
Its 11 files total 25,958,235 bytes and match the evaluated package.
Manifest SHA-256: `c7199d52e3ccdfb7fe24d745bd9f7bbbf86df453bc89b783015a623db99eed9f`.

| Component | Features | Saved trees |
| --- | ---: | ---: |
| Main pairwise | 33 | 1,600 |
| Nameless pairwise | 17 | 238 |
| Incremental booster | 53 | 675 |

EPS is 0.55. The gate uses 240 derived features.
EPS 0.55 is the benchmark-validation choice.
[Giant-block EPS review](../eps_calibration.md) remains pending before final acceptance.
Pairwise uses seven benchmarks and the existing non-patent augmented dataset.
The verified reused fit has 804,716 train pairs, 60,064 validation pairs, and 50 trials per head.
All 33 source-role hashes and eight Arrow provenance records match the selected inputs.
Both boosters retain their original bytes and fit records.

Incremental uses 13 non-patent datasets, 1,454,422 train rows, and 99,545 positive rows.
Dev retains 4,946 fit queries and 4,941 check queries. Both dev groups fit or select the gate.
The source snapshot preserves 4,051 initial-query origins.
The trainer excludes each origin record from its derived query's candidate profiles.
Two matched native feature checks pass within `1e-5`; they do not establish universal parity.

| Evaluation population | Result |
| --- | --- |
| Fixed pairwise test, 86,142 pairs | AUROC 0.989868; macro F1 0.954902 |
| Seven benchmark test sets, 2,157 blocks | Signature-weighted B3 F1 about 0.961 |
| Inherited test, 10,837 queries | Accuracy 0.965304; link precision 0.991275; link recall 0.969210 |
| Inherited errors | 289 false abstentions; 62 false links; 25 wrong-candidate links |
| Retained identity regression, 152 cases | 90/104 correct links; 47/48 correct abstentions; no wrong-candidate links |

The standard cluster report uses dataset metrics rounded to three decimal places.
Cluster inference retains 1,566 unknown signatures; scores use 72,853 known signatures.
Some block splits share papers. They do not establish whole-paper isolation.
Prior development and score inspection include these incremental test populations.
These tests supply regression evidence rather than independent release confirmation.
Identity gold differs from manual safe-action labels.
The previous recorded inherited accuracy is 0.968165, with an older query-feature treatment and patent-inclusive pairwise scores.
These two recorded rates do not isolate the effect of patent removal.

The corrected snapshot is `D:/local-unsynced/s2and_incremental_fresh_20261014/ready/complete_datasets_no_patents`.
It has 613 files and 40,268,461,989 logical bytes, including one shared count index.
The S3 research payload excludes that index: 608 files and 37,875,435,573 bytes.
Large immutable files share physical bytes with the preserved preparation snapshot through hard links.
All operational paths resolve inside the package after relocation.
The retained external gold file contains 152 cases with valid block, query, and member references.

The fit exits zero in 73 minutes 40 seconds. Peak trainer RSS is 11.784 GiB.
The approved bounds are 20 cores and 128 GiB; no paid API is used.
The complete package reloads and passes held-out evaluation and installed bulk/incremental Arrow inference.
Installed inference uses the existing matching Python and Rust runtime outside the checkout.
These small runtime checks do not establish latency for large blocks.

Run root: `D:/local-unsynced/s2and_incremental_fresh_20261014`.
Exact commands: `nonpat_commands.json`. Reports and process records remain under `reports`, `stages`, and `logs`.
The original patent research outputs remain outside the release payload.
Final release acceptance and S3/PyPI publication remain pending.
