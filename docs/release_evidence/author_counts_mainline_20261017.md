# Author-count 1.3 release candidate

The release owner selects fresh author-table counts, A, on 2026-10-17.
A replaces paper-author mention counts, M, as the common count authority.
The candidate retains the current 1.3 architecture and promoted incremental linker.
Final release acceptance and publication remain separate steps.

## Inputs and model selection

The inference runtime matches `a41b3bf` with its current tuple, ORCID prefix, and Rust assets.
The frozen launch record also preserves the production-script hashes.

The count build uses the verified October 15 raw export and the current canonical name rules.
Its 84,128,144 raw name groups represent 192,718,689 author rows.
The build takes 756 seconds.
The count manifest SHA256 is `87459dc14e41fd676730f56b0857d3eb61015cdc42bd73ec9b8b01e2f2b8f4b3`.
A measures source author rows. It does not establish a count of distinct people.

Eight feature sources supply 804,716 train pairs and 60,064 validation pairs.
The ORCID audit reviews 28 flags in 24 cases.
None of its 57 proposed relations occurs in the sampled fit matrices.
One sampled negative pair with equal sidecar ORCIDs remains unresolved because source affiliations conflict with the native match.
The fit retains its labels and adds no ORCID pairs.
This audit does not establish correctness for every label.

Both 50-trial hyperparameter searches complete in 56 minutes, with the pilot, at 1.13 GB peak RAM.
MAIN selects trial 26: validation AUC 0.994227, versus H13 at 0.993518.
Nameless retains H13: validation AUC 0.974620.
The release policy selects the highest validation AUC among candidates that pass the existing native fast-plan gate.
Both raw search winners meet that gate; no slower fallback replaces them.
Native scores match fitted LightGBM exactly on 1,000 validation rows per head.

Fresh-distance calibration selects EPS `0.57` and validation B3 F1 `0.95978133` on 72,649 scored signatures.
Production ORCID retention and constraints remain enabled, with `suppress_orcid=False`.
The 74,476 context signatures contain 30 ORCID signatures and 25 within-block same-ID pairs.
All selected JSON and Arrow ORCID values agree.
This sparse coverage gives limited evidence about ORCID-heavy production blocks.

## Candidate status

The complete model passes strict reload and public Arrow bulk and incremental inference.
Its manifest SHA256 is `9e68853b4562ede310ee65a464b2cf32cef279e5b133896e3b64cb62b060a08a`.

The seven-dataset external test reports signature-weighted B3 F1 of 96.35% on 72,853 scored signatures.
Dataset-macro B3 F1 is 95.67%; INSPIRE supplies 73.7% of scored signatures.
The inherited linker test reports 96.715% accuracy on 10,837 queries.

The local `artifacts/common/v1.3` path contains the matching A index and complete model.
Full project checks pass, with 2,128 Python tests and 162 Rust tests.

On 2026-10-19 the release owner applied EPS `0.53` from the final giant-block review.
Only `clusterer.json` and the manifest change; the boosters stay the same.
The new manifest SHA256 is `1d21148553bac71b4c3561857c2d415586a215b3084c963798326699d7bee232`.
At `0.53`, the external test reports signature-weighted B3 F1 of 96.44%.
The identity-gold result stays at 134 of 152 cases.
Python distribution assets match the source.
The installed Python and Rust wheels pass bulk and incremental inference.
The jobs use at most 12 compute threads, with a 128 GiB process-tree limit within the approved 160 GB budget.

Previous common M artifacts remain in a verified external archive.
The Athena producer has focused tests and an offline plan. Live Athena execution remains untested.

## Evidence and limits

The run root is `D:/local-unsynced/s2and_ca_mainline_20261017`.
Its `commands/`, `bounds/`, `orcid_audit/`, and `checks/` directories preserve commands, decisions, resource records, and verification outputs.
The previous common artifacts remain in `archive/previous_common_v1.3_M` under that root.
The [count guide](../../scripts/production/counts/README.md) defines the Athena source and canonical key contract.
The [archived M report](../archive/name_count_experiment_20261009_results.md) preserves the earlier count study.

Earlier giant-block wins and the 96.0540% versus 96.0548% benchmark comparison belong to runtime `2a6563c` and its exact artifacts.
They do not establish scores for this new candidate.
The frozen ORCID reference has measured linker-supervision overlap.
Giant-block acceptance tests remain the next step.
