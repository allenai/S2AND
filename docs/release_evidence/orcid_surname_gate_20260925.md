# ORCID prefix generator surname gate, 2026-09-25

The user requested this scoped generator change after observing that prefix
evidence is used inside surname blocks. This supersedes the previous
no-production-code-change constraint for the generator, its input contract,
tests, and documentation. No generated package assets or S3 objects were changed.

## Behavior

An ORCID-derived first-name pair now contributes only if both names occur with a
shared, nonempty canonical surname within that ORCID. Surname normalization uses
the existing canonical text pipeline and the space-insensitive equivalence used
by within-block constraints: `García`/`Garcia`, `O’Neil`/`O'Neil`, and
`Ou Yang`/`Ouyang` match; `Smith`/`Jones` and `Smith`/`Smithson` do not.

The producer retains the set of surname keys for each distinct canonical first
name in the current ORCID. It considers each first-name pair once and requires
the two surname sets to intersect before generating prefix events. This means:

- Al Smith plus Amy Jones supplies no ORCID alias evidence.
- Al Smith plus Amy Smith supplies the existing `al`/`am` and `al`/`amy` events.
- Seeing that same pair under both Smith and Jones does not count it twice within
  one ORCID.
- If Al/Amy share Smith and Amy/Ava share Jones, Al/Ava do not match transitively.
- Different qualifying first-name pairs can still contribute separately to the
  same prefix pair, preserving event-count semantics.

Empty canonical surnames are skipped with `rejected_empty_canonical_last` in the
metrics. Direct builder calls without a `last_name` field fail explicitly. The CSV
header is now exactly `raw_orcid,orcid,first_name,middle,last_name`; the old
four-column file is rejected. The refreshed source already includes `last_name`.
The production command reference documents its mapping, retains all refreshed
sources, and adds surname projection to the historical warehouse query.

The existing first-initial gate, prefix lengths, prefix compatibility exclusions,
event threshold, and curated alias fallback remain unchanged. The configurable
100-name default still bounds distinct accepted first names across the complete
ORCID. It was not changed into a per-surname guard or a quality filter. Artifact
JSON and its tuple-dependency manifest retain their existing formats.

## Verification

The focused generator and tuple-artifact suites passed: **43 tests**. Coverage
includes normalized surname equality, differing/empty surnames, absent fields,
overlapping surname sets, repeated evidence across surnames/ORCIDs, preservation
of event counts and alias precedence, the existing expansion bounds, actual CSV
publication, and refusal to publish cross-surname ORCID evidence. Ruff, format,
and ty checks passed for the changed Python files and sample verifier. An
independent code review found no remaining correctness issue.

A bounded run used all **200 complete diagnostic groups / 36,486 mentions** from
the earlier saved fixtures, with no new source scan. The pre-change generator
was saved and run on the same rows. An independent reference kept spaced
canonical surnames and applied `canonical_lasts_equivalent` to each candidate
pair, rather than using the new generator's precomputed comparison keys.

| Diagnostic sample metric | Before | With surname gate |
| --- | ---: | ---: |
| Unthresholded prefix-pair keys | 213,623 | 1,848 |
| Prefix-pair events | 626,152 | 3,190 |
| Events from the 14 reviewed mixed-name groups | 611,037 | 3,018 |

All 200 groups matched the independent reference exactly; 36 groups changed.
Summed per-group results matched a combined sorted-stream run. Cutoffs 5, 10, 20,
and 50 also matched independent thresholding of the unthresholded result.
The run took 4.59 seconds within a 120-second bound, with the diagnostic
2,000-name limit explicitly supplied to both versions. There were no empty
canonical surnames in this sample. This cohort is enriched for previously flagged
groups: these changes are **not full-corpus estimates or model-quality results**.
Sharing a surname does not establish identity, so the remaining events are not
automatically valid aliases.

The evidence record is
`scratch/dataset_upgrade_20260925/orcid_last_name_gate/sample_verification.json`.
It contains fixture and generator hashes, command, limits, per-group counts,
and reconciliation results. The preserved pre-change generator is
`generator_before.py` in the same directory. Reproduction uses a fresh evidence
output path, since the verifier refuses to overwrite its result:

```powershell
uv --cache-dir scratch/uv-cache run --no-sync --offline python -m scratch.dataset_upgrade_20260925.orcid_last_name_gate.verify_sample --max-seconds 120 --output scratch/dataset_upgrade_20260925/orcid_last_name_gate/sample_verification_repeat.json
uv --cache-dir scratch/uv-cache run --no-sync --offline pytest -q tests/test_generate_orcid_name_prefix_counts.py tests/test_name_tuple_artifact.py
```

The scratch verifier and its input fixtures are local and gitignored. Normal
repository development requires the dev extra and matching native extension;
`--no-sync` here reuses the already prepared environment.

## Existing count artifacts

The earlier [full-count calibration](orcid_prefix_calibration_20260925.md) and its
JSON record remain historical evidence for the surname-agnostic rule. They do
not validate this changed producer. Their canonical cache has no last names,
and their four-column CSV adapter omits the newly required field. Reusing that
cache or adapter would lose the information needed by the gate.

This change has only the bounded sample verification above. Full count rebuilding
and new label-based comparisons have not been run under the new policy. Start
from the immutable refreshed source, retain surnames through preparation, and
reconsider the earlier 14-group exclusion and threshold recommendation using the
new results. No automatic blacklist is introduced by this generator change.
