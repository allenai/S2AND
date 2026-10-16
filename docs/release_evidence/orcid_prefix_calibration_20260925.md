# ORCID prefix-count preparation and S2AND-label comparison, 2026-09-25

**Historical policy:** this run preceded the subsequently requested surname gate.
The generator now requires matching nonempty canonical surnames as well as the
same ORCID. See [the surname-gate change](orcid_surname_gate_20260925.md). The
results and recommendations below describe the earlier generator and have not
been revalidated under the new rule. Its cache discarded surnames and cannot be
used to regenerate surname-matched counts; the full input and downstream count
comparisons must be rebuilt before using the new policy for release.

Full ORCID count preparation and the bounded S2AND-label comparison are complete.
The conservative next-stage candidate excludes the 14 reviewed groups and keeps
the existing event cutoff of 10; this is not a calibrated production optimum.
These are local candidate inputs, not a
production release. No production code, raw export, model asset, or S3 object was
changed. The separate 699-million-mention name-frequency rebuild was not run.

## Input and semantics

The fixed input is `orcid_names_20260925.csv.gz`, 956,536,611 bytes, SHA-256
`bfcdc22eddceead61602e2dcb063f1fb5e8b9d29027ddcd2195ec16d29250915`, from
`s3://s2-atalyaa/disambiguation/author-names/name_counts/s2and_exports_20260925/`.
All refreshed sources are retained. Citation weights do not enter ORCID prefix
counts. A mention repeated across papers does not give that name extra weight
within an ORCID: the producer works on distinct canonical first names per ORCID.

The preserved contract is:

1. Normalize ORCID and first name using the current producer's rules; exclude
   empty canonical first names from name-pair generation.
2. Within each ORCID, deduplicate names and consider unordered name pairs sharing
   the same initial. Generate lengths 2, 3, 4, and 5; canonicalize pair direction;
   omit equal and token-prefix-compatible pairs. Deduplicate emitted prefix pairs
   within each full-name pair.
3. Count these events across ORCIDs. Different full-name pairs from one ORCID can
   contribute repeatedly to a prefix pair. The cutoff is an event threshold,
   **not a minimum number of independent ORCIDs**.
4. Retain event counts at least 5, 10, 20, or 50 in the respective candidates.
   Add alias prefix counts at least 2 only when an ORCID pair is absent after
   thresholding. Alias counts are neither added to nor maximized with retained
   ORCID counts. Changing the cutoff can therefore change a surviving pair's
   score to or from its alias fallback.

The existing 100-name limit is an execution guard before quadratic expansion,
not a proven data-quality rule. The complete source contains a maximum of 1,242
canonical first names per ORCID. This scratch run used an explicit 2,000-name
guard and processed every group. Production defaults were not edited.
All eight groups exceeding 100 occur in the independently reviewed set of 14.
After excluding those 14, the maximum is exactly 100, so the existing guard would
pass that derived input. This observation does not validate 100 as a quality rule.

## Efficient preparation and verification

The source is read with the previously validated sequential PyArrow CSV reader.
The pipeline aggregates raw `(ORCID, first_name)` rows with mention totals,
normalizes each distinct raw first name once, joins that mapping back, and
deduplicates canonical `(ORCID, first_name)` rows. Empty-name mentions stay in
the cache for accounting. Only groups with at least two nonempty canonical first
names enter the pair counter. Bucketing names by their first character avoids
enumerating name pairs that the unchanged producer necessarily rejects.

| Stage/count | Result |
| --- | ---: |
| Source mentions reconciled | 41,899,086 |
| Canonical ORCIDs reconciled | 9,425,662 |
| Distinct canonical-ORCID/raw-first-name rows | 10,689,331 |
| Distinct raw first names normalized | 603,544 |
| Canonical ORCID/first-name cache rows, including empty names | 10,484,891 |
| Empty canonical first-name mentions | 443 |
| Empty canonical first-name cache rows | 344 |
| ORCIDs with at least two nonempty canonical first names | 907,809 |
| Unthresholded prefix-pair keys | 437,780 |
| Unthresholded prefix-pair events | 1,416,921 |
| Cache preparation wall time | 56.67 seconds |
| Counting and eight candidate tables wall time | 54.95 seconds |
| Canonical cache size | 76,697,730 bytes |

Both full stages had a 900-second limit, two DuckDB threads, an 8 GB DuckDB
memory limit, and explicit spill limits. Python counters have separate cardinality
bounds; 8 GB is not a whole-process memory cap. Each stage completed with exit
code zero. Forty-six partial count files preserve completed work, but the scratch
driver does not implement automatic resume. Validation copies contribute most of
the roughly 2 GB local working outputs, not the count cache.

Before the full run, a complete-group hash pilot retained 9,418 ORCIDs and 41,642
mentions. Its 794 unthresholded keys and all four alias-merged candidates matched
the actual production generator. A separate diagnostic fixture of 200 complete
groups / 36,486 rows, including eight groups over the old guard, matched the
generator group by group. Four count-core regression tests, Ruff, and ty passed.
The first count-driver pilot exposed a quoted-SQL-identifier error; it failed
before candidate publication. Its failed record remains preserved, and the
corrected second pilot passed before the full run.
Measured counting-only speedups were 1.7–2.0x on that diagnostic fixture; this is
not a full-corpus before/after speed claim.

An independent output check recounted all 14 reviewed groups with the unchanged
current production generator, verified their complete canonical-name sets and 4,617 mentions against
the full cache, checked all 46 partial hashes, independently reduced every
statistics column, and reconstructed all eight candidate maps from threshold and
alias rules. All checks passed. Clean-group maxima were verified through partial
reduction and the all/clean/flagged maximum identity, not a second full-source
clean recount.

## Reviewed-group sensitivity

The [existing review](orcid_input_policy_20260925.md) identified 14 specific
mixed-name groups. Their raw source rows remain untouched. We saved both an
all-group candidate family and a family excluding those groups solely from
derived prefix evidence.

Those 14 groups account for 611,037 of 1,416,921 events (43.12%), despite comprising
4,617 source mentions (0.0110%). Excluding them leaves 258,900 unthresholded keys
and 805,884 events. This is an exact contribution calculation, not a population
estimate of erroneous ORCIDs.

| Event cutoff | All groups: ORCID keys | All groups: keys after alias fallback | Exclude 14: ORCID keys | Exclude 14: keys after alias fallback |
| --- | ---: | ---: | ---: | ---: |
| 5 | 40,226 | 40,896 | 18,121 | 18,815 |
| 10 | 17,596 | 18,631 | 7,766 | 8,858 |
| 20 | 7,315 | 8,811 | 3,448 | 5,038 |
| 50 | 2,374 | 4,378 | 1,345 | 3,406 |

At cutoff 10, the all-group ORCID table contains 3,322 keys supported by only one
ORCID and 9,979 keys receiving at least half their events from a single ORCID.
After excluding the reviewed 14, those counts are 0 and 9 respectively. The
maximum one-ORCID contribution falls from 1,344 to 12. Distinct-ORCID support and
maximum individual contributions are diagnostics; they do not replace the
producer's event scores in these candidates.

Maximum aggregate event scores are 3,790 with all groups and 3,789 after exclusion,
below the consumer's 100,000-scale prefix-compatible merge priority.

## Validation population and reference

The comparison uses S2AND gold cluster labels, not same-ORCID agreement as the
truth criterion. It freezes 50 complete blocks containing 16,446 signatures from
the existing baseline Arrow datasets. Membership is reconstructed with the
current `ANDData.split_blocks_helper` policy (seed 1111; 80/10/10) in preserved
backup JSON insertion order. An original release split-seed record was unavailable;
this is not a claim to have recovered the exact historical release split.
Known legacy Aminer test blocks were additionally excluded in their entirety.
The reconstructed test split was not scored.

All seven datasets' backup JSON IDs and block assignments matched their Arrow
tables exactly before selection. The largest eligible whole blocks were selected,
without signature subsampling. The selected data span Aminer, Arnetminer, Inspire,
and KISTI. Current scratch manifests reference byte-identical copies of the
existing Arrow tables and indexes; all 32 copies passed size and SHA checks.
No refreshed dataset conversion or augmentation was performed.

The current canonical packaged ORCID prior is absent locally. The available
historical artifact is a directional map with conflicting mirrored values. The
reference therefore means **historical counts evaluated by the current Rust
consumer**, not a replay of historical Python or an identified current production
release. It preserves the current Rust lookup exactly: lexicographic direction
first, then reverse if absent, never summing or maximizing the two. All 10,838
directional lookup checks and raw-versus-mapped pilot partition checks passed.

Graph configuration is fixed at current defaults, with seed 42 from the saved
v1.21 clusterer configuration; direct ORCID subblock repair stays enabled. Each
candidate uses the actual current Arrow/Rust subblocking path. Gold same-author
pair retention is computed from cluster-by-subblock contingency counts, and work
is measured as the sum of `n*(n-1)/2` over subblocks. No all-pairs table or model
fit is required.

The largest eligible block has 1,472 signatures. A maximum subblock size of 4,000
bypasses every selected block, so that control cannot distinguish the priors.
The 100/250/500 comparisons deliberately stress the algorithm on available gold
labels; they cannot establish giant-block production performance. Very large
gold clusters are themselves forced to split at these small capacities.

## Validation results and next-stage candidate

All nine tables (historical reference plus eight candidates) completed all four
capacities on the same 50 blocks. The source labels contain 1,481,908 within-block
same-author pairs. The following compares the conservative candidate,
`exclude_reviewed_14_min10.json`, with the historical-count/current-Rust reference:

| Maximum subblock size | Reference gold pairs retained | Candidate gold pairs retained | Candidate comparison pairs | Comparison-pair change vs reference |
| --- | ---: | ---: | ---: | ---: |
| 100 | 454,745 | 455,180 | 710,106 | +2,140 |
| 250 | 996,385 | 996,385 | 1,755,231 | +202 |
| 500 | 1,339,376 | 1,339,376 | 2,694,291 | +460 |
| 4,000 control | 1,481,908 | 1,481,908 | 3,818,930 | 0 |

This candidate preserves at least as many gold pairs at each tested capacity,
with a small comparison-cost increase versus the reference. This is subblocking
recall/cost evidence, not model F1 or production-scale held-out performance.

The low absolute retention at small capacities partly reflects forced splitting:
capacity-only upper bounds are 537,608, 1,111,408, and 1,364,408 retained gold pairs
at sizes 100, 250, and 500 respectively. For a gold cluster with `n = q*C + r`,
the bound is `q*choose2(C) + choose2(r)`, summed over clusters within blocks. It is
an upper bound; it does not promise simultaneous feasible packing of all clusters.

There is no uniform threshold winner in this bounded screen:

- In the exclusion family, cutoff 5 retains another 68 gold pairs at size 100 and
  another 3 at size 250 relative to cutoff 10, at higher comparison cost.
- Cutoffs 20 and 50 lose 23 gold pairs at size 250 relative to cutoff 10.
- At size 500 every candidate ties the reference's gold-pair retention, while
  exclusion/cutoff 50 saves 52,317 comparisons versus the reference.
- Every table ties at size 4,000 because all selected blocks bypass subblocking.

The reviewed-group policy has a measurable effect separate from the cutoff.
At cutoff 10/size 100, exclusion versus retaining all groups preserves 435 more
gold pairs while saving 343 comparisons. This comes from two Inspire blocks:
`j ellis` gains 435 gold pairs and 1,020 comparisons, while `z liu` changes no
retained gold pairs and saves 1,363 comparisons. At cutoff 10/sizes 250 and 500,
the all-group and exclusion partitions are identical. This does not imply that
the policies are identical at every other cutoff.

A fresh actual-path rerun of the exclusion/cutoff-10 candidate reproduced all
150 block/capacity partitions, gold-pair metrics, comparison counts, and subblock
sizes exactly. Total bounded validation execution, including this repeat, was
579.09 seconds; peak working set across the sweep was about 1.06 GB.

Keep cutoff 10 as the conservative next-stage setting: it preserves the existing
threshold, removes the specific reviewed contamination from derived evidence,
and has no gold-retention regression versus the available reference in this
screen. These data do not establish that 10 is optimal for giant production
blocks. All alternatives and unthresholded statistics remain available. No
candidate has been copied into the package or published; final release validation
still needs an identified production baseline and the later model-level checks.

The [compact machine-readable evidence](orcid_prefix_calibration_20260925.json)
preserves all 36 aggregate comparisons, candidate hashes, contribution statistics,
reference checks, and the final repeatability result without copying gold labels.

## Local reproduction and artifacts

The working directory is the S2AND repository. Commands, logs, exit records,
source/cache/code hashes, and intermediate evidence are preserved under
`scratch/dataset_upgrade_20260925/orcid_calibration/` (gitignored). The code
revision for the full runs was `ff9f61c6a160e7a9bf6a66bca6151219d3ebcccb`;
relevant file hashes are also recorded to identify the exact executed code.

These commands also require local audit inputs outside this calibration directory:
`orcid_deeper/orcid_groups.parquet`, `orcid_deeper/selected_complete_groups.parquet`,
`orcid_policy_review/stratified_complete_groups.parquet`, and
`orcid_policy_review/quarantine_manifest.json`, all under the same
`scratch/dataset_upgrade_20260925/` root. The packaged canonical alias artifact
must be available; this run used 5,027 alias pairs with data SHA-256
`b21638351149389c57eca547b0f79c80084e56ad273f31e778cb1db1866945a8`.
The scripts and prerequisite audit inputs are local and gitignored: a fresh Git
checkout plus these two commands alone cannot reproduce this run. The record
documents the reusable process and this machine's concrete reproduction path.

```powershell
uv --cache-dir scratch/uv-cache run --no-sync --offline --with duckdb==1.4.1 python -m scratch.dataset_upgrade_20260925.orcid_calibration.build_cache --output scratch/dataset_upgrade_20260925/orcid_calibration/full_cache --max-seconds 900 --memory 8GB --threads 2
uv --cache-dir scratch/uv-cache run --no-sync --offline --with duckdb==1.4.1 python -m scratch.dataset_upgrade_20260925.orcid_calibration.build_counts --cache scratch/dataset_upgrade_20260925/orcid_calibration/full_cache/canonical_groups.parquet --output scratch/dataset_upgrade_20260925/orcid_calibration/full_counts --max-seconds 900 --max-groups 2000000
```

These are the executed commands, not overwrite commands: choose a fresh output
directory to repeat a run. The cache script binds this exact source path/hash and
uses the prior full lexical-ORCID audit; a future refresh must repeat that audit
and explicitly update the input identity. It is a one-off scratch implementation,
not a newly supported production CLI.

Key local paths relative to that working directory:

- `full_cache/run.json`: input reconciliation, timing, normalization/cache identity.
- `full_cache/canonical_groups.parquet`: sorted ORCID/first-name cache with mention totals.
- `full_counts/run.json`: exact candidate maps' identities and threshold statistics.
- `full_counts/prefix_statistics.parquet`: all/clean events, independent support,
  and maximum group contributions before any threshold.
- `full_counts/reviewed_group_contributions.json`: exact removable group contributions.
- `full_counts/candidates/`: eight JSON candidate maps; no production manifest or promotion.
- `full_output_verification.json`: independent final artifact verification.
- `validation/frozen_validation.json`: complete selected IDs, labels, and split provenance.
- `validation/`: per-candidate actual-path reports, logs, reference parity, and sweep records.

Paths in this list are under the calibration scratch directory, not the repository
root. Threshold changes and whole-reviewed-group sensitivity can reuse the saved
statistics without rescanning the 41.9-million-row source. A new source filter,
normalizer, or per-group contribution policy requires rebuilding the appropriate
upstream stage.
