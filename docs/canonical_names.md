# Canonical names

This document is the durable contract for how S2AND canonicalizes raw
author names, builds name-count keys, and compares canonical names.

## Canonical name contract

`s2and.text.canonicalize_name_parts` and its Rust equivalent are the
authorities. Given raw `first`, `middle`, and `last` values, they:

1. normalize Unicode spacing and remove soft hyphen and zero-width joiner;
2. normalize apostrophe-like marks, transliterate, lowercase, and delete
   apostrophes rather than making token boundaries;
3. treat supported Unicode dash variants as one separator;
4. replace remaining nonletters with spaces and collapse whitespace;
5. drop at most one leading title from first name, except `md`;
6. retain a dash-bound first-name group, otherwise keep the first token and
   spill remaining first-name tokens into middle; and
7. normalize last name independently, preserving spaces and surname particles.

Examples: `Anne-Marie Claire` becomes first `anne marie`, middle `claire`;
apostrophe variants of `O'Connor` become `oconnor`; and `Ou-Yang` and
`Ou Yang` both become last `ou yang`.

`s2and.text.canonical_name_count_keys` emits no sentinel keys. It emits
`first` only when first has more than one character, `last` when present,
`first_last` when both are informative, and `last_first_initial` when both
components exist.

At comparison time, `same_prefix_tokens` checks aligned token pairs up to the
shorter token list. In each pair, either token must prefix the other; the
direction can differ between pairs (`ann marie` matches `anna m`). Empty input
is missing evidence. Alias tuples are unordered. Canonical last names retain
spaces in storage, but `canonical_lasts_equivalent` ignores spaces when
comparing them: `ou yang` equals `ouyang`. Dash variants become spaces during
canonicalization, before this comparison.

The live authorities are `s2and.text`, its Rust implementation, and
[the frozen examples](../tests/fixtures/canonical_name_examples.json).
Tuple and ORCID runtime rules live in
[production_inference.md](production_inference.md); Arrow and name-count
formats live in [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md).
The retained manual
tuple review is [release_evidence/name_tuple_legacy_adjudication_v1.md](release_evidence/name_tuple_legacy_adjudication_v1.md).
