# ORCID input review and proposed quarantine, 2026-09-25

Status: **bounded review complete; production quarantine not applied**. The user
subsequently authorized efficient full ORCID count preparation and S2AND-label
validation. Both all-source counts and a 14-group-exclusion sensitivity have now
been prepared; see [the count calibration record](orcid_prefix_calibration_20260925.md).
This work changed no production code, source data, identifiers, model assets,
or S3 objects.

The subsequent [surname-gate change](orcid_surname_gate_20260925.md) is a separately
authorized production-generator change. The mixed-name review remains source
evidence, but the earlier prefix influence and exclusion recommendation predate
that rule. Reassess their effects with surname-matched counts rather than carrying
forward the old candidate choice automatically.

## Decision supported by this review

Quarantine 14 specifically reviewed ORCID groups from a future **derived
first-name-prefix evidence input**. Preserve their original records and every
ingest source. The proposal covers 4,617 of 41,899,086 source rows (about 0.0110%);
41,894,469 would remain if it were applied to this exact export. These are exact
counts of the named groups, not an estimate of all erroneous rows.

The machine-readable [proposal](orcid_quarantine_proposal_20260925.json) records
the input SHA-256, each identifier, evidence, and affected rows. The complete
[53-group manual review](orcid_group_review_20260925.json) preserves the final
decisions. The later count-preparation authorization does not publish this
quarantine policy or modify the raw export.

## What was reviewed

- Eleven previously flagged groups: eight actual producer guard failures and
  three mixed-name groups below the guard. The primary reviewer read 20 full-name
  tuples per group (220 tuples); an independent reviewer read another documented
  20-tuple packet per group. Complete groups contain 4,572 rows; this is not a
  claim that every one of those rows was manually adjudicated.
- Forty-two fresh complete groups: two groups in each of 21 cells crossing seven
  source categories with total mentions 1, 2–10, and 11–50. Source categories are
  Crossref, Medline, Springer, IEEE, SpringerNature, MAG, and pooled OTHER.
  All 425 rows, represented by all 64 full-name tuples and paper-position lists,
  were inspected. Eight selected groups received independent second review.
- Fourteen targeted mentions from five conflicting groups were checked against
  current public Crossref records, ORCID profiles, and accessible publisher pages.
  Raw responses, timestamps, status codes, and hashes were preserved.

Sampling used DuckDB 1.4.1's fixed `hash(orcid) % 1000 = 261` candidate bucket,
then deterministic selection within cells, excluding all 158 earlier diagnostic
groups. Source membership is an anchor, not exclusive provenance: all sources
for each selected ORCID were retained. Two sequential local export scans took
38.58 seconds, bounded to five minutes, 4 GB memory, 4 GB spill, and two threads.
No new dataset download, paid API, production database, or Athena query was used.

This is a balanced diagnostic sample, not a population-proportional sample or
an overall contamination-rate estimate. OTHER does not individually cover every
remaining source. Groups with more than 50 mentions are represented only by the
earlier biased diagnostic cohort, not by this fresh sample.

## Source evidence changes the interpretation

Thirteen of fourteen export associations match the current Crossref name,
position, and ORCID. This establishes that those assignments are present upstream
of the export today; it does not prove their historical payloads or correct owners.

For example, a [publisher author list](https://journals.kmanpub.com/index.php/aftj/article/view/1668)
assigns `0000-0002-5571-8349` separately to four coauthors. Its corresponding
[Crossref record](https://api.crossref.org/works/10.61838/kman.aftj.4.1.5) attaches
that identifier to the first author. This supplies a concrete upstream example
that the previous export-only checks could not reveal.

The fresh sample also exposed conflicts in groups with only two or three first
names:

| ORCID | Complete source rows | Decisive evidence |
| --- | ---: | --- |
| `0000-0003-3895-7409` | 13 | Khadijeh Sharifi and Zahra Sooki appear as separate coauthors, but different papers assign this same identifier to each. |
| `0000-0003-3393-8460` | 27 | Export assigns the ID to Amgad A. Salama at position 0; current Crossref assigns it to coauthor Mahmoud A. Shawky at position 1. |
| `0000-0002-1028-8971` | 5 | The group includes Bruna Urbanski, James R. Garcia-Ayala and Ricardo C. Benine; Crossref reproduces the Bruna and James assignments. |

The Sharifi/Sooki records list both people separately on the same byline, ruling
out a simple name-change explanation for that pair.
[First record](https://api.crossref.org/works/10.1186/s12887-024-05371-0),
[second record](https://api.crossref.org/works/10.1186/s12909-024-05161-3).

The Salama/Shawky disagreement remains a **provenance blocker**: current metadata
alone cannot distinguish a historical source change from a local mapping error.
Do not automatically shift the identifier or label the backfill as its cause.
[Current author array](https://api.crossref.org/works/10.1016/j.iot.2024.101486).

The Bruna/James assignments are present in both exact DOI records; one of the
publisher pages also corroborates the James assignment.
[Bruna record](https://api.crossref.org/works/10.3390/w16101383),
[James record](https://api.crossref.org/works/10.1590/1982-0224-2022-0086).

The first eight targeted Crossref assignments have `authenticated-orcid=false`.
This is recorded provenance, **not a blanket exclusion rule**. Public profile
names and current publisher pages are useful evidence, not complete identity
histories. One publisher denied access and two returned redirect shells; those
responses were not counted as confirming author evidence.

## Manual outcomes and counterexamples

Final group decisions: **14 unsuitable heterogeneous-name groups, 10 compatible
variant groups with no observed conflict, and 29 groups with one canonical first
name**. The latter two labels do not verify ownership or certify release readiness.
Two initially unresolved ordinary groups were adjudicated using the distinct
coauthor evidence above. Every assigned group has one final decision; there are
no missing, duplicate, or invalid annotations.

The label `mixed_identity_evidence` in the JSON means unsuitable heterogeneous
name evidence for unrestricted within-ORCID alias expansion. It does not declare
every associated mention wrong, identify the owner, or prescribe replacement IDs.
Three source-traced ordinary mentions plausibly match the public profile even
though their groups also contain conflicts. Quarantine intentionally gives up
some potentially valid evidence rather than guessing which rows to repair.

Examples that the policy must preserve as counterexamples to crude filters:

- Vietnamese field-order and segmentation variants can be compatible. Such
  variants occur even inside a group whose other names are clearly incompatible.
- Initial/full-name and transliteration families can be compatible: `S. P.` /
  `Stephen P. Hesselbo`, `Ilya` / `Iliya V. Zyabkin`, and `KM` / `Kristian M. Forbes`.
- `Anna N. Gorodishcheva` / `Gorodischeva A. N.` plausibly describes a field-order
  problem. Identity compatibility does not make the raw first field correct.
  The unchanged producer emits no pair for this group because first initials differ.
- The same producer emits four `km`/`kr...` keys for the Forbes example. That is
  its existing initials behavior, not a new cleanup rule or verified alias claim.

## Proposed policy

1. Preserve the immutable raw export and all refreshed source categories. Apply
   any approved exclusions only when constructing the derived ORCID input for
   prefix evidence, with a separate manifest tied to this export's SHA-256.
2. Exclude the entire 14 named groups from that evidence input. Do not delete
   source mentions, change benchmark ORCIDs, affect unweighted name counts, or
   select the most frequent name as the owner.
3. Treat high name diversity, source category, example-like identifiers,
   unauthenticated associations, and surname diversity as review signals only.
   They are insufficient standalone exclusion rules. The 100-name guard remains
   unchanged and is not a semantic-quality filter.
4. Repair individual assignments only with corroborated source and provenance
   evidence. Record before/after identifiers and reasons in a separate approved
   repair manifest; this review proposes no row-level repairs.
5. Preserve the existing prefix-event counting semantics and alias fallback.
   Counting each ORCID only once per prefix pair would be a separate behavior
   change, outside this no-production-code-change task.

## Initial diagnostic effect and remaining limitations

During the initial review, the unchanged producer was exercised on 200 complete diagnostic
groups (36,486 rows), with alias fallback disabled to isolate ORCID evidence.
The unfiltered input failed the existing guard; that initial review did not
raise it. Comparing the 192 groups after removing only eight guard failures
against the 186 groups after the 14 proposed quarantines reduces threshold-10
prefix-pair keys from **59 to 36**. This is a diagnostic-cohort result; global
counts and final merged artifacts had not yet been generated. The later
[authorized full count comparison](orcid_prefix_calibration_20260925.md) now
provides global counts, with an explicit scratch execution bound and unchanged
production defaults.

The three newly found small conflicting groups emit zero prefix events under
the current first-initial restriction. Only two of the 42 fresh groups emit any
events: ten distinct keys, each count one, none reaching threshold ten in this
sample. Thus the small-group findings are real assignment-quality concerns,
but they add no demonstrated prefix-count damage in this cohort.

The completed candidate preparation does not resolve these provenance and
population-quality limitations:

- Corpus paper `275166515`, position 0, has an unresolved export/Crossref
  discrepancy. Resolving it would require saved source payload and backfill mapping.
- The manual sample contains too few contributing ordinary groups to estimate
  their error rate. Full contribution statistics and S2AND-label comparisons now
  measure prefix influence, but do not identify every bad group.
- These 14 exclusions do not certify the remaining 9.4 million identifiers.
- The earlier three-hour preparation proposal remains unapproved and unexecuted.
  The later bounded, efficient ORCID preparation was separately authorized and
  completed; production publication was not part of it.

## Evidence and reproduction

Scratch root: `scratch/dataset_upgrade_20260925/orcid_policy_review/`.
`CONTRACT.md` defines the units, labels, and bounded extension. Full packets,
initial decisions, source adjudications, independent reviews, raw public
responses, and initial 11-group proposal remain preserved. The final manifest
supersedes the preliminary proposal without overwriting it.

Key files: `STRATIFIED_PACKET.md`, `stratified_producer_metrics_verified.json`,
`policy_challenge.md`, `ordinary_second_review.md`, `source_trace/REVIEW.md`,
`source_trace/ordinary_extension/REVIEW.md`, `final_review_validation.json`, and
`final_cohort_impact.json`. Helpers passed Ruff and ty. Final annotation validation
and actual-producer comparisons passed; no full production test run was warranted
because no production behavior changed.

Packet builders and finalizers refuse existing output paths. Use a fresh scratch
output location when reproducing; do not delete preserved evidence just to rerun.
