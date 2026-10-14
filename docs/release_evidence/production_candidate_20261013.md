# Patent-inclusive production candidate

The complete candidate is available in PR 80. Final release acceptance remains open.
Run root: `D:/local-unsynced/s2and_v1.3_patents_train_20261013`.
Resources: 20 jobs, 128 GiB, no paid API.
Source commits: pairwise `5806868`; incremental/evaluation `92718a8`; smoke fix `1ffe94f`.

The repository model is `artifacts/common/v1.3/production_model_v1.3`.
Its 11 files match the evaluated model byte for byte: 38,879,902 bytes.
Manifest SHA-256: `d60b4fea7d480e4a2e7f2cd6d8d82f5c96261f613fe2ba663a49fe466f039e97`.

| Component | Features | Saved trees |
| --- | ---: | ---: |
| Main pairwise | 33 | 708 |
| Nameless pairwise | 17 | 922 |
| Incremental booster | 53 | 675 |

The gate selects 240 derived features. EPS is 0.62.
Pairwise uses 904,716 train pairs and 70,064 validation pairs across nine sources, with 50 trials per head.
Incremental uses 1,454,422 rows, with 99,545 positives. It has no patent incremental train or dev tasks.

Use `D:/local-unsynced/s2and_release_inputs_20261013/ready/complete_datasets_clean_calibration` for the current corrected snapshot.
Its 661 files total 40,442,140,813 bytes. S3 research excludes the common index: 656 files, 38,049,114,397 bytes.
The dev treatment excludes 662 fit and 674 check views through verified query-paper exposure.
The remaining 4,946 fit and 4,941 check views both select or fit the gate.
Excluded views remain outside booster fit. All labels and memberships remain unchanged.

The fresh identity test freezes before candidate scores: 396 positive and 303 author-absent cases.
Patents supply 547 contexts from 368 unique queries. Profiles use corrected gold and original names.
Identity gold differs from manual safe-action targets. Arnetminer and PubMed have no eligible fresh cases.

| Fresh source | Correct links / positives | False links / author-absent cases |
| --- | ---: | ---: |
| AMiner | No positive cases | 0 / 13 |
| INSPIRE | 61 / 70 | 0 / 6 |
| Patents | 286 / 292 | 20 / 255 |
| KISTI | 5 / 8 | 2 / 12 |
| QIAN | 14 / 14 | 0 / 6 |
| zbMATH | 11 / 12 | 2 / 11 |

All 699 cases complete: 19 false abstentions, 24 false links, no wrong-candidate links.
Keep these cases as tests. Use separate patent dev tasks for further changes.
The screen covers declared model/dev papers, known authors, and exact titles; other aliases and common-asset influence remain unverified.

| Other population | Result |
| --- | --- |
| Validation AUROC, main / nameless | 0.993110 / 0.965091 |
| Validation signature-weighted B3 F1 | 0.957430 |
| Outer pairwise AUROC / macro F1 | 0.989617 / 0.955113 |
| Outer B3 F1, signature-weighted / dataset-macro | 0.959267 / 0.955125 |
| Inherited accuracy / link precision / recall | 0.968165 / 0.992191 / 0.971857 |
| Inherited errors, 10,837 queries | 267 false abstentions; 58 false links; 20 wrong-candidate links |

Outer evaluation uses 96,142 pairs, 2,407 blocks, and 78,970 signatures; 1,566 unknown signatures remain in inference context.
These block splits share some papers. Inherited incremental scores provide regression evidence because prior development includes that test population.

The main booster uses the general native score path. Installed candidate wheels pass bulk and incremental Arrow inference outside the checkout.
The smoke budget is 8 GiB; its former 2 GB budget fails below actual RSS.
All nine diagnostic runtime requests preserve profiles and signatures.
Default constraint checks pass for 64 queries, 400 profiles, and 4,726 seeds: 7.518 seconds, 3.77 GiB peak RSS.
Times exclude model load and Arrow open. They do not establish an improvement or whole-block performance.
Formal acceptance and S3/PyPI publication remain pending.
