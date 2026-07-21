# Canonical-v2 Normalization Migration

Status date: 2026-07-10

## Status

The single-mode `canonical_v2` **code cutover has landed** on
`canonical-v2-migration`, but the release is blocked on canonical artifacts and
the v1.3 retrain. Code, models, name counts, ORCID prefix counts, name tuples,
Arrow datasets, and benchmark names must be validated as one release unit.

The v1.0-v1.2 pickles have been removed. The v1.21 bundle remains only as an
explicit historical source/parity artifact; it is not packaged or loadable by
canonical-v2. No production model or default declaration is distributed during
the cutover. This is a migration state, not a releasable package state.

The active engineering remediation ledger is
[work_plan.md](work_plan.md). It includes provenance, cross-artifact binding,
batching, cache, schema, parity, resource, packaging, and documentation defects
found in the 2026-07-09 audit. Those items are part of release readiness even
when they do not directly alter normalization.

## Cutover Readiness Checklist

1. **Pending:** regenerate canonical name-count mappings once on internal
   infrastructure with required source/version/checksum provenance. The v1
   publisher records the source pickle identity for model-lineage compatibility
   and writes `name_counts_index/` directly from the resident mappings, without
   reloading the pickle. Publication is deliberately not transactional across
   the counts and index manifests (owner decision 2026-07-20): each publish is
   individually atomic, a crash between them leaves a torn state that every
   consumer rejects at name-count binding time (the index manifest embeds the
   counts generation provenance and models carry the four-field binding), and
   rerunning the generation command with `--overwrite` repairs it. The
   crash-window behavior is regression-tested in
   `tests/test_generate_name_counts_script.py`.
2. **Pending:** validate the canonical `name_counts_index/` cardinalities,
   digests, exact lookup parity, Python preprocessing latency, and peak RSS.
   Python and Rust runtime paths consume only this index.
3. **Runtime validation and bundle binding complete; generation pending:**
   regenerate the canonical versioned ORCID prefix-count generation and publish
   `first_k_letter_counts_from_orcid.manifest.json`. Runtime loading is lazy but
   requires the manifest, immutable generation, matching normalization/pair
   semantics, source digest, metadata/data checksums, and exact cardinalities;
   the unversioned JSON is no longer a fallback. Production training records
   its exact data SHA-256 in `feature_contract`; export and load require that
   hash to match the packaged priors used by default subblocking.
4. **Artifact validation and bundle binding complete:** deterministically
   regenerate and package `s2and/data/s2and_name_tuples_canonical.txt` with its
   strict source/data checksum, cardinality, normalization, and semantics
   metadata. Python and Rust enforce the same validation. Production training
   records the exact tuple-data SHA-256, and export/load compare it to the
   packaged aliases. Future crash-atomic regeneration requires an approved
   generation-pointer layout rather than same-path files.
5. **Pending:** re-export benchmark training names by signature-ID join, report
   join/divergence metrics, and retrain the production v1.3 pairwise and
   incremental-linker bundle.
6. **Pending:** pass quality, subblocking, runtime, peak-RSS, parity, installed
   wheel, and release-integrity gates on the immutable release candidate.

The checklist is complete only when the release manifest proves that every
component came from the same normalization/feature contract and expected
generation. A valid `canonical_v2` string on each artifact is not sufficient if
their generations, source digests, or model semantics differ.

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

Every normalization-sensitive artifact must record and validate:

- artifact schema version;
- `normalization_version = "canonical_v2"`;
- immutable generation ID;
- source snapshot/query/config digest;
- content SHA-256 and byte size;
- relevant row/key/cardinality/total-mass counts;
- canonical tuple digest when tuple expansion affected generation;
- producing git commit and dirty-state flag;
- generation command and bounded/full-run mode.

Provenance is copied from verified sources; writers must never infer it from the
currently imported code. Data and metadata are staged, validated, fsynced, and
published as one immutable generation with the pointer manifest replaced last.

The release validator must compare, not merely parse, normalization and
generation contracts across:

- each Arrow dataset manifest and batch index;
- the name-count source identity and `name_counts_index/`;
- ORCID prefix counts;
- canonical name tuples;
- pairwise main/nameless boosters and feature contract;
- promoted incremental linker and replay target;
- the explicit complete production bundle manifest.

## Benchmark Name Re-export

Production training benchmarks must be re-exported with upstream canonical
names joined by signature ID. Add canonical columns alongside raw historical
columns rather than overwriting them. Record:

- source and target row counts;
- duplicate and missing signature IDs;
- joined/unjoined counts and rate;
- per-field raw-to-canonical divergence;
- representative differences, especially language-aware compounds;
- output digest and source snapshot.

Run a tiny fixed sample first. The full internal job requires explicit approval
and reproducible logs.

## Retrain and Acceptance Gates

The v1.3 pairwise and incremental-linker models must train from the exact
release-candidate artifacts. Their metadata must bind normalization version,
ordered feature contract, featurizer version, both pairwise booster digests,
linker digest, and replay-target digest.

Required quality evidence:

- Pairwise no-op/alignment comparisons: `AUC delta <= 0.001` and
  `F1 delta <= 0.005` on unchanged inputs.
- Clustering no-op/alignment comparisons: `B3 delta <= 0.005`.
- The intentional feature-changing retrain must show end-metric non-regression
  versus the shipped production release on identical evaluation sets.
- Report per-dataset effects of removing Sinonym, fastText, and reference
  features; do not infer their safety from implementation parity.
- Subblocking checks include size distributions, merge behavior, ORCID
  co-location, and dash/name-alias cases.
- Runtime and peak RSS must be within 10% of the pinned protocol unless the
  repository owner explicitly accepts a measured tradeoff.

Metrics must be present and finite and must pass before any artifact is promoted.
Hyperparameter search does not bypass the same gate.

## Release and Rollback

The release flow must:

1. build into immutable staging locations;
2. validate full checksums, containment, schemas, cross-artifact contracts, and
   strict Arrow batch-index fingerprints;
3. clean-install the exact Python and Rust wheels in an empty `uv` environment;
4. load the explicit complete candidate bundle and run real embedded
   pairwise/incremental fixtures;
5. publish Rust first and publish Python only after the exact Rust version is
   installable;
6. promote only the already-validated release unit.

Rollback is deployment of the previous package together with its complete
legacy artifact set. There is no dual runtime normalization mode and no mixing
of old code with canonical artifacts or canonical code with legacy artifacts.

## Current Authorities

- Python canonicalization: `s2and.text.canonicalize_name_parts`.
- Count keys: `s2and.text.canonical_name_count_keys`.
- Last-name compare policy: `s2and.text.canonical_lasts_equivalent`.
- Version constant: `s2and.consts.NORMALIZATION_VERSION`.
- Production Arrow validation: `s2and.arrow_inputs`.
- Count-index format/publication:
  `s2and.incremental_linking.feature_block_arrow`.
- Canonical tuple artifact:
  `s2and/data/s2and_name_tuples_canonical.txt` and its metadata.
  The adjacent `<artifact>.meta.json` uses
  `schema_version = "s2and_name_tuples_v3"` and binds the exact data filename,
  SHA-256, byte size, unordered-pair cardinality, canonical-v2 normalization,
  one lexicographically canonical row per unordered pair, and source
  filename/SHA-256/size. Its generation audit separately counts empty, identity,
  prefix-compatible, and duplicate canonical rows and requires those counts
  plus the output cardinality to equal the input cardinality. Both Python and
  Rust reject a missing, mismatched, or semantically invalid sidecar before
  accepting aliases. An explicit custom text path must carry the same adjacent
  strict sidecar; pass an explicit set (including an empty set) when the aliases
  are intentionally caller-owned instead of an artifact.
- Tuple generation publishes fsynced data first and metadata last, and is only
  safe as an offline staging operation. A process crash between the two
  same-path replacements can leave a fail-closed mixed pair; rerun generation
  in staging and validate before promoting the package. A crash-preserving live
  rollback would require a generation-directory/pointer layout and therefore a
  separately approved artifact schema/layout change.
- Tuple binding: `s2and.name_tuple_artifact.load_name_tuple_artifact` retains
  the validated pairs and their data SHA-256. Rust applies the same strict
  sidecar and row validation while loading aliases, without maintaining a
  second identity-inspection API. Production `feature_contract` contains
  `name_tuples_data_sha256` and `orcid_prefix_counts_data_sha256`; bundle export
  and load require both to match the installed canonical data.
- Frozen examples: `tests/fixtures/canonical_name_examples.json` and
  `tests/test_canonical_name_examples.py`.
- Version-contract tests: `tests/test_normalization_version_contract.py`.

## Current Small Verification Gate

```powershell
uv run pytest -q tests/test_canonical_name_examples.py tests/test_normalization_version_contract.py tests/test_subblocking_merge_candidates.py tests/test_production_model.py -ra
```

This gate checks code behavior only. It does not replace regenerated artifacts,
the v1.3 retrain, installed-wheel smoke, or release-candidate quality/runtime/RSS
evidence.
