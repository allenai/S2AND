# Canonical-v2 Normalization Migration

Status date: 2026-07-25

## Status

The single-mode `canonical_v2` **code cutover has landed** on
`canonical-v2-migration`, but the release is blocked on canonical artifacts and
the v1.3 retrain. Code, models, name counts, ORCID prefix counts, name tuples,
Arrow datasets, and benchmark names must be validated as one release unit.

The v1.0-v1.2 pickles have been removed. The v1.21 bundle remains only as an
explicit historical comparison artifact; it is not packaged or loadable by
canonical-v2. No production model or default declaration is distributed during
the cutover. The v1.3 model will be an explicit immutable external bundle, not
a packaged default.

The active v1.3 implementation ledger is
[1_3_release_blockers.md](1_3_release_blockers.md). The broader historical
remediation ledger remains [work_plan.md](work_plan.md), but work in that file
is not automatically part of the v1.3 release.

The executable order, approvals, test-reveal protocol, and publication sequence
are in [1_3_release_todo.md](1_3_release_todo.md). This document retains
migration history; `release_spec.json` is the machine-readable authority for
the release's frozen thresholds and decisions.

For v1.3, retain the already implemented persisted formats:
`s2and_name_tuples_v3`, `name_counts_index_v2` with
`name_counts_provenance_v3`, `orcid_prefix_counts_v2`, and
`s2and_production_model_bundle_v5`. Format cleanup is not a cutover
prerequisite and is outside the v1.3 critical path.

## Cutover Readiness Checklist

1. **Pending:** regenerate canonical name-count mappings once on internal
   infrastructure with required source/version/checksum provenance. The
   publisher writes the complete `name_counts_index/` from resident mappings
   under a temporary sibling, then renames it once into a previously absent
   target. A failed build leaves that target absent. Regeneration uses a new
   output directory and the enclosing release manifest selects it; the
   publisher does not reuse or replace an existing index.
2. **Pending:** validate the canonical `name_counts_index/` cardinalities,
   digests, exact lookup parity, Python preprocessing latency, and peak RSS.
   Python and Rust runtime paths consume only this index.
3. **Runtime validation and bundle binding complete; generation pending:**
   regenerate `first_k_letter_counts_from_orcid.json` and its adjacent
   `.manifest.json`. Runtime loading is lazy and requires the matching schema,
   normalization version, pair semantics, producer provenance, and exact data
   SHA-256; a data file without its manifest is rejected. Production training
   records the exact data and manifest SHA-256 values in `feature_contract`.
   Runtime compatibility is keyed by the behavior-defining data hash; the
   manifest hash remains release provenance and part of the linker’s pairwise
   bundle binding.
4. **Artifact validation and bundle binding complete:** deterministically
   regenerate and package `s2and/data/s2and_name_tuples_canonical.txt` with its
   strict source/data checksum, cardinality, normalization, and semantics
   metadata. Python validates the artifact once and supplies its frozen pairs
   explicitly to Rust-backed flows. Production training records the exact
   tuple-data SHA-256, and export/load compare it to the packaged aliases.
   Future crash-atomic regeneration requires an approved generation-pointer
   layout rather than same-path files.
5. **Pending:** re-export benchmark training names by signature-ID join, report
   join/divergence metrics, and retrain the production v1.3 pairwise and
   incremental-linker bundle.
6. **Pending:** pass quality, subblocking, runtime, peak-RSS, parity, installed
   wheel, and release-integrity gates on the complete model.

The checklist is complete only when the data and model manifests prove that
every component came from the same normalization/feature contract and expected
generation. A `canonical_v2` string alone is insufficient when generations,
source digests, or model semantics differ.

## Frozen Canonical Name Contract

`s2and.text.canonicalize_name_parts` and the corresponding Rust implementation
are the canonical authorities.

1. Inputs are raw `first`, `middle`, and `last`; `None` is missing/empty.
2. Normalize Unicode spacing. Delete soft hyphen (`U+00AD`) and zero-width
   joiner (`U+200D`) as invisible formatting controls.
3. Normalize apostrophe-like characters, transliterate, and delete apostrophes
   and backticks rather than turning them into token boundaries.
4. Treat `-`, `U+2010`, `U+2011`, `U+2012`, `U+2013`, `U+2014`, `U+2212`,
   `U+FE58`, `U+FE63`, and `U+FF0D` as the same dash separator.
5. Transliterate with `text-unidecode`, lowercase, replace remaining nonletter
   and nonspace characters with spaces, and collapse whitespace.
6. Drop at most one leading title-prefix token from the first field. Keep `md`
   because it is also a common South Asian given-name abbreviation.
7. If the normalized first field starts with a dash-bound group, retain that
   group as the canonical first name. Otherwise retain the first token. Spill
   remaining space-separated tokens into middle before the supplied middle
   field.
8. Normalize the last field independently and preserve its normalized spaces.
   Surname particles remain present. Suffix stripping is outside canonical-v2.

Examples:

- `Anne-Marie Claire` -> first `anne marie`, middle `claire`.
- `O'Connor` and apostrophe-like variants -> `oconnor`.
- `Ou-Yang` and `Ou Yang` -> canonical last `ou yang`.

## Count-Key Contract

`s2and.text.canonical_name_count_keys` derives all count keys from canonical
fields after missing/informativeness gating:

- `first`: canonical first only when `len(first) > 1`, otherwise null.
- `last`: canonical last when present, otherwise null.
- `first_last`: `<first> <last>` only when first is informative and last exists.
- `last_first_initial`: `<last> <first[0]>` only when both exist.

Missing components produce null keys, never sentinel lookups. There is no
runtime `legacy_full_first_token` mode in canonical-v2.

## Compare-Time Name Contracts

### First names

`same_prefix_tokens` is a symmetric comparison over already-canonical first
names. Every aligned token in the shorter token list must be an exact prefix of
its counterpart; extra tokens in the longer value are allowed. Empty values are
missing evidence, never a match.

Representative cases:

| A | B | Compatible |
|---|---|---|
| `jo` | `joann` | yes |
| `jon` | `john` | no |
| `j p` | `jean pierre` | yes |
| `john david` | `j f` | no |
| empty | `alice` | no |

Packaged and user-provided alias tuples are unordered pairs. Python and Rust
canonicalize pair order and deduplicate at the runtime boundary, so behavior
does not depend on duplicated rows or file/input order.

### Last names

Canonical storage preserves spaces. Block-key/count projections may compact
spaces only at their explicitly documented boundary. The last-name constraint
continues to treat hyphen/space variants equivalently through
`canonical_lasts_equivalent`; this is deliberate compare-time policy because
upstream blocks can contain both forms.

## Required Artifact Provenance

Each normalization-sensitive artifact records only the facts needed to bind its
content and semantics. The shared minimum is an artifact schema, the
`canonical_v2` normalization version, and a content digest. Richer source,
generation, cardinality, or command provenance belongs only to artifacts whose
lineage is part of the model contract, such as name counts, Arrow datasets, and
trained model bundles. The ORCID prior uses its direct data file and one
adjacent `.manifest.json`; that manifest is the authority for schema,
normalization, unordered-pair semantics, source provenance, canonical tuple
binding, generator parameters, cardinalities, and the data SHA-256.

Release provenance must be copied from independently verifiable sources rather
than inferred from the currently imported code or trusted caller labels. The
current warehouse producers do not yet meet that gate: B27 must bind snapshot
IDs to query-result evidence, and B28 must replace the retired `pys2` route and
record the replacement tool/source identity. Full warehouse generation remains
blocked until both close. Each writer uses its own publication contract; there
is no universal generation-pointer or fsync protocol.

The release validator must compare, not merely parse, normalization and
generation contracts across:

- each Arrow dataset manifest and batch index;
- the name-count source identity and `name_counts_index/`;
- ORCID prefix counts;
- canonical name tuples;
- pairwise main/nameless boosters and feature contract;
- freshly trained incremental linker and embedded replay target;
- the explicit complete production bundle manifest.

## Benchmark Name Re-export

Production training benchmarks must be re-exported with upstream canonical
names joined by signature ID. The joined canonical names must reach the exact
fields consumed by `ANDData`, Arrow conversion, and pairwise featurization.
Merely adding side-by-side canonical columns is not sufficient because current
loaders ignore them. Either replace the consumed `first`/`middle`/`last` fields
and preserve raw historical names in a separate immutable audit artifact, or
first change every loader/converter/trainer to select named canonical fields
explicitly. Record:

- source and target row counts;
- duplicate and missing signature IDs;
- joined/unjoined counts and rate;
- per-field raw-to-canonical divergence;
- representative differences, especially language-aware compounds;
- output digest and source snapshot.

Run a tiny fixed sample first. The full internal job requires explicit approval
and reproducible logs.

## Retrain and Acceptance Gates

The v1.3 pairwise and incremental-linker models train from the exact released
data. After validation-only EPS selection freezes the pairwise bundle, the
direct linker finalizer rematerializes features, fits once, writes a complete
bundle, reloads it, and evaluates. Its metadata binds the normalization and
feature contracts, both pairwise booster digests, linker digest, and embedded
replay-target digest. A linker bound to another pairwise manifest is invalid.

Required evaluation results:

- Pairwise no-op/alignment comparisons: aggregate AUROC drop `<= 0.001` and
  macro-F1 drop `<= 0.005` on the exact frozen comparable pairs, plus the
  predeclared per-dataset gates.
- Clustering no-op/alignment comparisons: signature-weighted B3 F1 drop
  `<= 0.005`, plus reported macro and per-dataset results.
- The intentional feature-changing retrain must show end-metric non-regression
  versus the shipped production release on identical evaluation sets.
- Subblocking checks include size distributions, merge behavior, ORCID
  co-location, and dash/name-alias cases.
- Runtime must regress by no more than 10% under repeated, interleaved runs of
  the pinned workload. Peak RSS is a diagnostic comparison and each run must
  remain below the absolute byte ceiling frozen before measurement; there is no
  relative RSS gate or post-result owner waiver.

Metrics must be present, finite, and passing. Pairwise and clustering test
scores remain sealed until the complete model is serialized and reloaded.
The one evaluation report applies the release-spec gates. A failed gate aborts
instead of becoming another tuning iteration, and the failed run is retained
for diagnosis.

## Release and Rollback

The release flow must:

1. build once into immutable staging and validate every checksum and contract;
2. fit the linker once, write the complete bundle, reload it, then evaluate;
3. clean-install the exact Python and Rust wheels and run the real-model smoke;
4. verify the release spec, data manifest, complete-model manifest, and one
   evaluation report before writing `SHA256SUMS`;
5. publish those exact approved bytes, Rust first; and
6. run one public probe after index and data verification.

Rollback is deployment of the previous package together with its complete
legacy artifact set. There is no dual runtime normalization mode and no mixing
of old code with canonical artifacts or canonical code with legacy artifacts.

## Current Authorities

- Python canonicalization: `s2and.text.canonicalize_name_parts`.
- Count keys: `s2and.text.canonical_name_count_keys`.
- Last-name compare policy: `s2and.text.canonical_lasts_equivalent`.
- Version constant: `s2and.consts.NORMALIZATION_VERSION`.
- Production Arrow validation: `s2and.arrow_inputs`.
- Count-index format/loading: `s2and.name_counts_index` and
  `s2and.name_counts_manifest`.
- Count-index production writer:
  `scripts/production/counts/generate_name_counts.py`.
- Canonical tuple artifact:
  `s2and/data/s2and_name_tuples_canonical.txt` and its metadata.
  The adjacent `<artifact>.meta.json` uses
  `schema_version = "s2and_name_tuples_v3"` and binds the exact data filename,
  SHA-256, byte size, unordered-pair cardinality, canonical-v2 normalization,
  one lexicographically canonical row per unordered pair, and source
  filename/SHA-256/size. Its generation audit separately counts empty, identity,
  prefix-compatible, and duplicate canonical rows and requires those counts
  plus the output cardinality to equal the input cardinality. Python rejects a
  missing, mismatched, or semantically invalid sidecar before accepting aliases
  and passes validated pairs explicitly to Rust-backed flows. An explicit custom
  text path must carry the same adjacent strict sidecar; pass an explicit set
  (including an empty set) when the aliases are intentionally caller-owned
  instead of an artifact.
- Tuple generation publishes fsynced data first and metadata last, and is only
  safe as an offline staging operation. A process crash between the two
  same-path replacements can leave a fail-closed mixed pair; rerun generation
  in staging and validate before promoting the package. A crash-preserving live
  rollback would require a generation-directory/pointer layout and therefore a
  separately approved artifact schema/layout change.
- The checked-in source includes 1,343 legacy-only pairs accepted by the
  record-level review in
  `docs/release_evidence/name_tuple_legacy_adjudication_v1.md`. The resulting
  artifact has 5,027 pairs and retains all 3,684 pre-review pairs. The 906
  rejected and 17 uncertain candidates are not runtime aliases.
- Tuple binding: `s2and.name_tuple_artifact.load_name_tuple_artifact` retains
  the validated pairs and their data SHA-256. Rust consumes those explicit
  pairs without maintaining a second artifact loader or identity-inspection
  API. Production `feature_contract` contains `name_tuples_data_sha256` and
  `orcid_prefix_counts_data_sha256`; bundle export and load require both to
  match the installed canonical data.
- Frozen examples: `tests/fixtures/canonical_name_examples.json` and
  `tests/test_canonical_name_examples.py`.
- Version-contract tests: `tests/test_normalization_version_contract.py`.

## Current Small Verification Gate

```powershell
uv run pytest -q tests/test_canonical_name_examples.py tests/test_normalization_version_contract.py tests/test_name_counts_binding.py tests/test_generate_orcid_name_prefix_counts.py tests/test_subblocking_merge_candidates.py tests/test_production_model.py -ra
```

This gate checks code behavior only. It does not replace regenerated artifacts,
the v1.3 retrain, installed-wheel smoke, or the release evaluation.
