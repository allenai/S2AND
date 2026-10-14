# Corrected release snapshot

This record describes the original preparation snapshot.
Its patent-inclusive population is historical; release `1.3` now excludes patents.
The selected local snapshot path is `D:/local-unsynced/s2and_incremental_fresh_20261014/ready/complete_datasets_no_patents`.
The selected snapshot passes relocation checks.
The [candidate record](production_candidate_20261013.md) gives its current counts, model hashes, and evaluation results.

This record retains the original snapshot's counts and hashes.
No production model fit or S3 upload occurs in this preparation task.

The original snapshot path is `D:/local-unsynced/s2and_release_inputs_20261013/ready/complete_datasets`.
The ready plan is `pairwise/model_plan.json` within that root.
The production trainer accepts that relative plan after relocation.

| Item | Count |
| --- | ---: |
| Corrected benchmark datasets | 9 |
| Pairwise fit datasets | 9 |
| Pairwise evaluation datasets | 9 |
| Incremental runtime datasets | 13 |
| Incremental source label rows | 2,035,853 |
| Current corrected source files | 92 |
| Complete local files | 658 |
| Complete local bytes | 40,441,498,308 |
| Shared count-index bytes | 2,393,026,416 |
| S3 research files | 653 |
| S3 research bytes | 38,048,471,892 |

Patents use `inventors_s2and`, the existing 2,500-block subset.
It supplies 41,468 signatures, 41,316 papers, 6,078 clusters, and complete SPECTER2 vectors.
The outer test holds 250 blocks and 4,551 signatures.
Block splits share some papers; they do not establish whole-paper isolation.
Existing augmented records and labels remain unchanged.

The snapshot retains full corrected benchmark records and current uncertainty masks.
It retains treated incremental labels, memberships, query views, and active split assignments.
The seven giant replay sources lack full abstracts; their current records and cached vectors remain available.

The local root contains one physical count index.
Git LFS stores its four binary files under `artifacts/common/v1.3/name_counts_index`.
The JSON manifest remains in normal Git.
Exclude `name_counts_index/*` from the S3 research upload.

| Identity | SHA-256 |
| --- | --- |
| Public manifest | `55ade74674214747afa78ffd19169bbd3ad11310b02093f49b2bc030c7f60a90` |
| Corrected source inventory | `735dce71f7cbebc9366c65637522981a03b4f9e16f03e1504278b7c8cc658ca7` |
| Pairwise model plan | `eb4619339fabbcd567486f5b7c19b983187540a11527673c9dc4fbd4b1eb8c17` |
| Common count manifest | `85cffc00421f6b8a7ebab6b6bac7da162b4bccde88b83e348c9554e26c7153e7` |

The actual relocated root passes public, source, artifact, and production pairwise preflight checks.
Run evidence, commands, and logs remain in `D:/local-unsynced/s2and_release_inputs_20261013/run_evidence`.
The bounded build preserves its failed license check; the producer fix adds the public license before validation.
Release `1.3` uses the [patent-free pairwise fit](pairwise_training_20261013.md) and the verified [complete candidate](production_candidate_20261013.md).
Final release acceptance and S3/PyPI publication remain pending.
