# Version 1.3 data preparation

The accepted benchmark source is `D:/local-unsynced/s2and_contact_fix_20261009/datasets_10_09_2026`.
The output root is `D:/local-unsynced/s2and_v1.3_prep_20261012`.
The root contains logs, run commands, input hashes, and validation reports.

## Decisions and counts

| Item | Accepted value |
| --- | --- |
| CPU limit | 20 concurrent cores; no paid APIs |
| Public benchmarks | 8 datasets; 769,871 signatures; 487,567 papers |
| Production benchmarks | 7 datasets; AMiner remains included; internal ORCID remains excluded |
| Reviewed label corrections | 102 review rows; 318 moved signatures; 124 additional uncertain signatures |
| Uncertainty masks | 19,741 signatures across 7 clustered benchmarks |
| External evaluation | Seed-1111 outer test blocks; 74,419 signatures; 70,000 known-label pairs |
| Remaining benchmark population | 691,714 signatures; the trainer applies its internal split |
| Augmented data | 121,537 signature-specific variants; 134,184 train, 16,611 validation, and 16,142 external test pairs |
| Cached translations | 3,123 accepted; 286 rejected because raw source titles differ |
| ORCID prefix table | 6,079 pairs after the [14-group exclusion](orcid_prefix_exclusion_20261013.md); tuple SHA-256 `0bcb523ebf8e6f7020883d96cc2822f402f212069214fa23e56b6e7bc0b01f0c` |
| Global count index | 89,695,371 source rows; 699,054,083 eligible mentions; 45,474,003 keys |
| Linker support | 2,036,340 rows; 23,010 active lifecycle assignments; 19 reviewed erroneous ORCID queries excluded |
| Linker replay | 13 datasets; 1,449,889 signatures |

Preserve accepted raw name parts and signature order. The runtime owns normalization. Uncertainty masks retain inference membership but exclude uncertain supervised labels.

Reuse the current benchmark SPECTER2 vectors. Refresh only augmented text changes and informative source gaps. Preserve absent vectors for blank text and legacy replay gaps.

Local SPECTER2 uses base revision `3447645e1def9117997203454fa4495937bfbd83` and adapter revision `2081559630a80fc5851d8f798a05ba81e9468089`.
The Qian command replay matches the accepted preparation outputs. The real 51-text encoder comparison has zero vector difference.

## Durable commands and output status

[Preparation commands](../../scripts/production/data/README.md) cover benchmark preparation, augmentation, vector refresh, linker support, replay, Arrow conversion, and portable packages.
[Global count evidence](name_counts_global_20261012.json) records the complete production index.

All exports and portable packages are complete. Source hashes, native loaders, split boundaries, and uncertainty masks pass validation.
The portable roots are `public_data_v1.3`, `linker_source_bundle`, and `pairwise_inputs_v1.3` beneath the output root.
The public root contains 8 benchmarks and 13 replay datasets. The private package contains 8 fit and 8 evaluation datasets.
The linker source preflight accepts all 2,036,340 support rows. Both package jobs finish with exit code zero.

The public package preserves 81 absent MEDLINE vectors and 270,668 absent legacy replay vectors. It does not manufacture replacement vectors.
The corrected augmented Arrow vectors match all 121,537 source vectors exactly. The package uses the corrected `datasets/<name>` replay layout.

Final evidence: `logs/assembly_run.json`, `logs/package_private.run.json`, `private_arrow_validation.json`, and `logs/arrow_benchmark_evidence/assembly_companion_checks.json` beneath the output root.

## Next release stage

The subsequent [pairwise fit and EPS selection](pairwise_training_20261013.md) are complete.
Set the release evaluation policy. Fit and evaluate the linker with the reserved lifecycle splits.
Apply the release quality and parity gates before S3 publication. This data preparation run does not fit production models or upload artifacts.
