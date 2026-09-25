# September 2026 dataset and count-input audit

Status: full structural scans and bounded semantic diagnostics completed.
The original large preparation was held pending deeper checking; the user later
authorized efficient full ORCID prefix-count preparation and S2AND-label
validation. See [the count calibration record](orcid_prefix_calibration_20260925.md).
The separate unweighted name-frequency rebuild remains unrun. Dataset conversion
issues below remain unresolved. Follow-up source review and a production-unapplied
14-group quarantine proposal are recorded in
[the ORCID input policy review](orcid_input_policy_20260925.md).
This record covers snapshot validation and count-input preparation only. It does
not approve Arrow conversion, augmentation, model training, or publication.
No production code was changed by this work.

Subsequent scoped authorization changed the ORCID generator to require matching
surnames; see [the change and bounded verification](orcid_surname_gate_20260925.md).
The count adapters and calibration results recorded here predate that change.
The earlier four-column ORCID adapter drops surnames and must be updated before
reuse; the current producer requires `raw_orcid,orcid,first_name,middle,last_name`.

## Inputs and reproducibility

- Dataset: `s3://ai2-s2-research/s2and/datasets_09_24_2026.tar.gz`.
  Size: 8,270,714,312 bytes; S3 last modified `2026-09-25T08:04:04Z`;
  version `nMU2VBvNS948X5hIIuxKUSwUC9kqbvcZ`;
  ETag `b3423926eb279fb416c2c6626ccecf8d-986`.
  SHA-256: `8fd48a5f945d05410861f2c47ee696a33323ff2aacb1e2cd28a2f060f14e7056`.
  The local archive's multipart ETag matches S3. All 32 extracted files match
  their archive entries by size and SHA-256, with no missing or extra files.
- Count prefix:
  `s3://s2-atalyaa/disambiguation/author-names/name_counts/s2and_exports_20260925/`.
  The downloaded README and `run_name_counts.sh` are retained as source evidence,
  not executed. No production database or Athena query was run.
- Local original dataset: `D:/s2and-v1.3-inputs/datasets_09_24_2026/`.
  Downloaded count files: `D:/s2and-v1.3-inputs/audit_20260925/`.
- Scripts, commands, logs, samples and detailed JSON reports:
  `scratch/dataset_upgrade_20260925/` (intentionally gitignored).
- Initial repository HEAD: `080c3da07ba9c4c75371700a13a277c0d184c3a4`.
  Unrelated concurrent working-tree edits appeared during the audit. They were
  neither made nor reverted by this task. The decisive loader requirement was
  rechecked afterward.
- Execution: `uv --cache-dir scratch/uv-cache run --no-sync --offline`, with an
  isolated overlay containing `duckdb==1.4.1` and `ijson==3.4.0`. Existing project
  PyArrow and NumPy were reused; project dependencies and lockfiles were untouched.

## Dataset results

Fully scanned 24 JSON files and numerical contents/IDs of eight SPECTER2 files.
All 769,871 original IDs exactly match the eight local baseline Arrow signature
tables. Seven existing cluster files cover their 766,133 originals exactly;
the 3,738 MEDLINE originals have no locally available pair labels to verify.
All original and added records refer to an existing paper. No duplicate JSON
keys, signature-key/ID mismatches, or duplicate paper-author positions were found.

| Dataset | Original signatures | Added records | Papers | Referenced papers lacking embeddings | Originals lacking current author position |
| --- | ---: | ---: | ---: | ---: | ---: |
| aminer | 157,448 | 84,536 | 1,162,808 | 1,998 | 82 |
| arnetminer | 7,144 | 4,247 | 53,959 | 395 | 9 |
| inspire | 536,564 | 504,330 | 459,315 | 17,002 | 33,682 |
| kisti | 40,383 | 15,548 | 231,579 | 2,833 | 77 |
| medline | 3,738 | 15,413 | 61,792 | 81 | 10 |
| pubmed | 2,871 | 13,327 | 134,623 | 45 | 0 |
| qian | 6,542 | 5,388 | 59,545 | 285 | 4 |
| zbmath | 15,181 | 585 | 23,406 | 423 | 5 |

Counts are dataset rows, not globally deduplicated papers.

### Issues before conversion or training

1. **Email schema incompatibility is reproduced.** All 1,413,245 original and
   added rows have `emails` lists and omit `email`. A tiny real `ANDData` load
   raises `KeyError: 'email'`. Among originals, 19,776 contain email evidence and
   364 contain two emails. A scalar projection needs an explicit policy; silently
   discarding additional emails has not been approved. `source_ids` is already
   supported by the loader and is not a defect.
2. **Refreshed papers and retained signature positions can disagree.** 33,869
   originals lack their current paper-author position; the same check found zero
   in the baseline. There are 9,728 duplicate original paper/position keys, versus
   9,746 in the baseline, so most duplication is not new. A second pass identified
   710 newly duplicated positions. For example, zbmath paper `3078379` moves the
   refreshed Silverberg signature `7929` to position 1, which collides with the
   retained `AUTHOR_MISSING` Zarhin signature `7930`. Keep signature identities;
   do not silently deduplicate by current position.
3. **Added records are not labeled augmentation.** The 643,374 records use
   `paper_position` keys and have no supplied labels or `signature_id`. Of their
   positions, 8,206 already occur in originals; 5,992 have different normalized
   name strings, including aliases. They cannot simply become extra labeled data.
4. **Embedding coverage is incomplete.** All 2,070,192 vectors supplied are
   finite, nonzero, unique within dataset, float64 and 768-dimensional, with valid
   paper IDs. However, 23,062 of 487,567 original referenced paper rows lack a
   vector. The absent set includes every `AMBIGUOUS`/`NOT_FOUND` paper and 39
   `FOUND` papers across the complete paper inventory. Added-record papers all
   have vectors. Pickle opcodes/raw array spans were inspected without executing
   the pickles. Text-to-vector provenance remains unavailable.

### Snapshot consistency

The archive filename does not establish a September 24 cutoff: record update
timestamps extend through `2026-09-25T04:14:03.886846Z`. Matching 2,039
deterministic dataset anchors against the entire refreshed ORCID export found
439 exact paper/position matches. Every one matched both canonical name and
ORCID. Offset -1/+1 comparisons matched neither names nor ORCIDs.

This supports alignment for those 439 ORCID-bearing anchors, not the full
population. Extraction and backfill completion watermarks are still needed to
prove the snapshot includes the complete backfill. The anchor sample is not
random or representative. See `cross_snapshot_report.json` and the dataset
sub-audit's `REVIEW.md`, per-dataset JSONs, and second-pass reports.

## Complete count-export scan

Both compressed files were fully scanned with explicit types and sequential
PyArrow CSV batches with quoted newlines enabled, feeding DuckDB aggregates.
The successful scan completed in about 249 seconds with a 4 GB DuckDB limit.
An earlier direct DuckDB CSV scan hit that memory cap; its failure evidence was
preserved before switching readers. No source rows were ignored on parse errors.

| Check | Name-count export | ORCID export |
| --- | ---: | ---: |
| Logical rows | 89,695,372 | 41,899,086 |
| Sum of mention counts | 699,054,473 | — |
| Expected raw mentions | 699,054,139 | — |
| Distinct raw ORCID IDs | — | 9,430,173 |
| URL-form ORCID rows | — | 10,721 |
| Blank surname rows | 16 | not part of this full-scan check |

The full source distributions contain 45 paper-source values. All sources are
to be retained for the prepared ORCID input, as explicitly selected by the user.
The current producer's old Crossref query is not the selection policy for this
refresh; moreover, its source field and the new paper-source field differ.

**The supplied weighted-name export is ineligible as the exact count source.**
Its own README documents a CSV-reader defect: parallel parsing of quoted
multiline names introduced 334 mentions and fragmented name rows. The complete
scan independently confirms the +334 total. Numeric mean/max/weight/Jensen
invariants all pass, demonstrating why those checks alone are insufficient.
Do not accept the inflated total or subtract 334 from an arbitrary row.

Verified downloaded SHA-256 identities:

```text
name_counts_weighted_20260925.csv.gz
2392d185818a575d09385b054af145803a14d504988e49423ef55766aab66d7f
orcid_names_20260925.csv.gz
bfcdc22eddceead61602e2dcb063f1fb5e8b9d29027ddcd2195ec16d29250915
```

## ORCID semantic diagnostics: not ready for production preparation

Following the user's request for stronger checks, local diagnostics used at most
4 GB DuckDB memory, 4 GB spill, and two threads. The complete-group summary took
about 84 seconds, extraction/check-digit checks about 61 seconds, and a separate
full mention-uniqueness scan about 176 seconds. All read the existing local
export; the raw mention input was not downloaded and no full sorted input or
production artifact was generated. Logs preserve the first summary attempt's
view-binding error; the completed summary was reused rather than rescanned.

The complete export has 9,425,662 normalized ORCIDs. All pass the check-digit
test. All 41,899,086 `(paper_id,position)` keys are unique, with exactly one
ORCID each. These are structural passes, **not evidence of correct identity
assignment**.

The full scan identified eight groups with more than 100 distinct raw first
names. All eight still exceed 100 after the existing producer's exact
normalization; each independently raises its existing 100-name-limit error.
Canonical first depends only on raw first, so groups with at most 100 raw first
names cannot exceed this particular bound through middle-name variation.

| ORCID | Rows | Canonical first names | Crossref-only producer passes 100-name guard |
| --- | ---: | ---: | --- |
| `0000-0000-0000-0001` | 132 | 115 | yes |
| `0000-0001-2345-6789` | 277 | 200 | no |
| `0000-0001-7511-2910` | 184 | 150 | no |
| `0000-0002-1825-0097` | 2,467 | 1,242 | no |
| `0000-0002-3752-9683` | 619 | 147 | no |
| `0000-0002-4011-3590` | 336 | 128 | no |
| `0000-0002-5024-6186` | 175 | 158 | no |
| `0000-0002-5982-8983` | 169 | 126 | no |

The largest group includes Crossref rows for `Editorial Office` (93),
`Ius Canonicum` (64), `Scripta Theologica` (21), `Josiah Carberry` (15), and
many other personal names. That ORCID is also used in
[ORCID's official publication-display examples](https://info.orcid.org/guidelines-on-the-display-of-orcid-ids-in-publications/).
Example use alone is not a deletion criterion; the observed mixed identities
are the relevant concern. Restricting to Crossref does not resolve seven of the
eight guard failures, and the user's all-source policy remains unchanged.

Removing only these eight groups would be inadequate:

- `0000-0002-5571-8349` has 71 Crossref rows, 64 canonical first names and 67 raw
  surnames, including Homa Moghimi, Salman Javanbakht and Ramin Delfanazari.
  It passes the guard and alone produces 18 events for `("ma","moz")`.
- `0009-0004-4499-3929` has 43 Crossref rows and 41 canonical first names.
  Four `ai*` and five `ay*` names alone produce 20 events for `("ai","ay")`,
  exceeding the default threshold of ten. The threshold counts name-pair
  events, not independently supporting ORCIDs.

These results use 158 complete groups (36,061 rows), deliberately enriched for
large/outlier groups plus a deterministic hash subset. Of the 150 groups that
pass the guard, 59 prefix-pair keys reach ten events; 15 have only one supporting
ORCID within this cohort. These are **not population contamination estimates**,
nor proof that every affected prefix pair is false. Alias fallback was disabled
for this diagnostic; no production artifact was built.

Second-pass checks reproduced the largest failure and both below-limit examples
using the unchanged producer. All 36,061 selected rows matched its lexical ORCID
normalizer; an independent iterative check-digit implementation accepted all
158 selected IDs and rejected their deliberately changed check digits. No
selected ORCID appears at multiple author positions on the same paper, so these
checks do not establish a coauthor-copying mechanism. Upstream misassignment,
field interpretation, and the population error rate remain unresolved.

Evidence: `orcid_deeper/group_audit.json`, `mention_audit.json`,
`complete_group_diagnostics.json`, `parent_second_pass.json`, the complete-group
Parquet, and diagnostic scripts/logs under the scratch root above.

The subsequent [source review](orcid_input_policy_20260925.md) documented a
14-group quarantine proposal and inspected a source-stratified ordinary sample.
The later [authorized full comparison](orcid_prefix_calibration_20260925.md)
verified producer parity, quantified full-corpus prefix influence, and prepared
both all-group and exclusion-sensitivity candidates under explicit scratch
execution bounds. No raw identities or production semantics were changed.

## Count preparation policy and tested implementation

- Prepare **unweighted mention counts** from the saved raw mention export.
  Group raw `(first_name,last_name)` and count rows. Middle-name variants may
  collapse here because the existing producer uses only first/last for its four
  mappings. Keep canonical normalization and singleton pruning in that producer.
  Require exactly 699,054,139 source rows and the identical output count sum.
- Citation weights remain separate experiments. Actual formulas are sums of
  `ln(c+1)` and `ln(c+2)`, so they differ on cited papers too. Neither sum can be
  reconstructed from just mention count and total citations. Reweighting requires
  mention-level citations from the saved join inputs. Missing citation matches
  were treated as zero upstream, which differs from known uncited papers.
- Prepare ORCID rows from **all refreshed sources**. Preserve raw ORCID, apply
  the existing lexical normalizer, retain first/middle, and sort by canonical
  ORCID. Check digits are audited, not used as a new exclusion policy. Preserve
  the existing prefix name-pair-event semantics rather than counting raw mentions
  or claiming each prefix pair counts distinct ORCIDs.
- At the time of this audit, output CSV headers matched the then-current readers:
  `first_name,last_name,count` and `raw_orcid,orcid,first_name,middle`.
  The later surname-gate change adds required `last_name` to the ORCID header;
  the old ORCID adapter is not compatible with that contract.
- `scratch/dataset_upgrade_20260925/counts_review/adapt_exports.py` has eight
  passing regression tests, including actual producer readers, a native index
  round trip, external sorting, multiline CSV, source retention, exact totals,
  bounds, and refusal to overwrite outputs. A 100,000-row all-source ORCID sample
  was successfully normalized, sorted and reread by the existing producer reader.
- The scratch raw-mention rebuilder passes four regression tests, including
  multiline/empty-name fixtures. Its 100,000-row bounded sample exactly matches
  an independent Python Counter: 86,314 raw first/last groups, 80,559 singletons,
  1,274 blank-first rows and no blank-last rows. Failed attempts retain staging
  evidence and do not publish final inputs.

Full preparation is separate from building the canonical binary count index or
packaged ORCID prefix-count assets. No production artifacts have been replaced.
The proposed full preparation budget (three hours, 64 GB DuckDB memory, 80 GB
additional disk, and an 8.93 GB raw-input download) was **not approved or run**.
