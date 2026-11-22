# Canonical names

This document is the durable contract for how S2AND canonicalizes raw
author names, builds name-count keys, and compares canonical names.

## Access the current rules and artifacts

> **Update at v1.3 release:** Runtime `1.0.0` remains unpublished.
> Use PR 80's `canonical-v2-migration` branch and its local Rust build for the current rules.

Clone the branch and build the environment:

```bash
git clone --branch canonical-v2-migration https://github.com/allenai/S2AND.git
cd S2AND
uv sync
```

Name tuples and ORCID prefix counts are ordinary Git files and wheel assets.
The large count index downloads separately, as shown below.
They do not require the count-index download.
See [installation prerequisites](install.md) before the native build.

Use the Python helper on raw name parts:

```python
from s2and.text import canonical_name_count_keys, canonicalize_name_parts

parts = canonicalize_name_parts("Anne-Marie Claire", None, "Ou-Yang")
assert tuple(parts) == ("anne marie", "claire", "ou yang")
keys = canonical_name_count_keys(parts)
```

Use the Rust extension's matching helpers from Python:

```python
from s2and.runtime import load_s2and_rust_extension

s2and_rust = load_s2and_rust_extension()
rust_parts = s2and_rust.canonicalize_name_parts("Anne-Marie Claire", None, "Ou-Yang")
assert rust_parts == tuple(parts)
assert s2and_rust.canonical_name_count_keys(rust_parts) == keys
```

A native Rust application can depend on the same checkout without the extension-module feature:

```toml
[dependencies]
s2and_names = { package = "s2and_rust", path = "../S2AND/s2and_rust", default-features = false }
```

Set `path` relative to the application's `Cargo.toml`.
Use the public `name_normalization` module:

```rust
use s2and_names::name_normalization::{canonical_name_count_keys, canonicalize_name_parts};

fn main() {
    let parts = canonicalize_name_parts("Anne-Marie Claire", "", "Ou-Yang");
    assert_eq!(parts.first, "anne marie");
    assert_eq!(parts.middle, "claire");
    assert_eq!(parts.last, "ou yang");
    let keys = canonical_name_count_keys(&parts);
    assert_eq!(keys.first_last.as_deref(), Some("anne marie ou yang"));
}
```

Use an empty string for a missing raw Rust name part.
The Python helpers accept `None` for that same case.
`canonicalize_name_text` is also available in both modules for whole-name text without the first/middle split.
It does not remove a first-name title.

Load the packaged alias tuples and ORCID prefix counts:

```python
from s2and.name_tuple_artifact import load_packaged_name_tuple_artifact
from s2and.orcid_prefix_counts import (
    DEFAULT_ORCID_PREFIX_COUNTS_DATA_SHA256,
    load_packaged_orcid_prefix_counts,
)

aliases = load_packaged_name_tuple_artifact()
orcid = load_packaged_orcid_prefix_counts(DEFAULT_ORCID_PREFIX_COUNTS_DATA_SHA256)
assert orcid.name_tuples_sha256 == aliases.data_sha256
```

`aliases.pairs` contains canonical unordered first-name alias pairs.
`orcid.counts` contains first-name prefix-pair evidence; it is not the global name-frequency index.

Download the common count index only when you need global name frequencies:

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and/1.3.2/inference/name_counts_index/ s2and/data/name_counts_index/
```

Use the canonical keys with the validated index:

```python
from s2and.name_counts_index import NameCountsIndex

index = NameCountsIndex.open("s2and/data/name_counts_index")
first, last, first_last, last_first_initial = index.lookup_many(
    [keys["first"]],
    [keys["last"]],
    [keys["first_last"]],
    [keys["last_first_initial"]],
)
```

The model contract binds the SHA-256 of the index manifest, so keep its bytes unchanged.
The index requires about 2.42 GB of local space.
`None` keys yield `NaN`; informative keys absent from the index yield the default count `1.0`.
Use [the data guide](data.md) for model assets and the data bundles.

## Canonical name contract

`s2and.text.canonicalize_name_parts` and its Rust equivalent are the
authorities. Given raw `first`, `middle`, and `last` values, they:

1. normalize Unicode spacing and remove soft hyphen and zero-width joiner;
2. normalize apostrophe-like marks, romanize Georgian and Armenian with
   scholar's tables (decision D10), transliterate, lowercase, and delete
   apostrophes rather than making token boundaries;
3. treat supported Unicode dash variants as one separator;
4. replace remaining nonletters with spaces and collapse whitespace;
5. drop at most one leading title from first name, except `md`;
6. split eligible joined capitals and short mixed-case first names into initials
   when the last name passes the case check (decision D9);
7. retain a dash-bound first-name group, otherwise keep the first token and
   spill remaining first-name tokens into middle; and
8. normalize last name independently, preserving spaces and surname particles.

Examples: `Anne-Marie Claire` becomes first `anne marie`, middle `claire`;
apostrophe variants of `O'Connor` become `oconnor`; and `Ou-Yang` and
`Ou Yang` both become last `ou yang`.

### D10. Georgian and Armenian romanization

Before transliteration, D10 applies the tables planned for Scholar's block keys
(`BlockKeyNormalizers`). See [`s2and/text.py`](../s2and/text.py) for the full tables.

| Script | Rule | Examples |
| --- | --- | --- |
| Georgian | Use the national system without apostrophes. Romanize a Mtavruli capital with the uppercase form of its lowercase letter's mapping, preserving the case evidence for D9. | `ბერიძე` → `beridze` |
| Armenian | Expand `և` and U+FB13 through U+FB17 into their letters. Read `ու` as `u`, word-initial `ե` as `ye`, and word-initial `ո` as `vo` unless `վ` follows. | `Արթուր` → `artur`, `Տեր-Եղիազարյան` → `ter eghiazaryan` |

Armenian context skips punctuation and Unicode categories `Mn`, `Cf`, `Cc`, `Co`, and `Cs`.
A word starts at the beginning or after a remaining character outside the letter category `L`.
Python's `unicodedata` and Rust's `unicode-properties` can differ on newly added letters.
Pre-D10 indexes retain the older keys. Until a rebuild, lookups use counts for the new Latin spelling.

### D9. Joined capitals and mixed-case initials

| Branch | Pattern in the raw first name | Exclusions |
| --- | --- | --- |
| Joined capitals | Two or three ASCII capitals after removing whitespace and periods. | `_JOINED_INITIALS_GIVEN_NAMES` (`ED`, `MD`), `NAME_PREFIXES`, and a two-capital entry in `_DOTTED_GIVEN_NAME_ABBREVIATIONS` with a final period, such as `PH.` |
| Mixed case | One ASCII capital and one or two ASCII lowercase letters after removing whitespace. A period prevents a match. | Any `a`, `e`, `i`, `o`, `u`, or `y`, plus `NAME_PREFIXES` and `_MIXED_CASE_NON_INITIALS`, such as `Md`, `Kh`, and `Jr` |

See [`s2and/text.py`](../s2and/text.py) for the full exclusion lists.
Both branches require `last_name_allows_joined_initials` to pass. The canonical
transliteration of the raw last name must contain at least one ASCII lowercase
letter and at least as many lowercase letters as capitals. For example, `á`
counts as lowercase, and `HG` with last `WELLS` stays `hg`.
The first initial becomes the canonical first name. Later initials go before
any existing middle tokens, so `JHM` with last `Wells` gives `j` and `h m`, and
`Jm` with middle `Kevin` and last `Smith` gives `j` and `m kevin`.

Supply the raw last name for the case check. When canonicalizing first and last names
in separate maps, compute `last_name_allows_joined_initials` once per distinct raw
last name and pass it as `last_allows_initials` to `canonicalize_name_parts`.

### Name-count keys

`s2and.text.canonical_name_count_keys` emits no sentinel keys. It emits
`first` only when first has more than one character, `last` when present,
`first_last` when both are informative, and `last_first_initial` when both
components exist.

## Hard name comparison rules

`same_prefix_tokens` checks aligned tokens up to the shorter list. Either token
can prefix the other in each pair, so `ann marie` matches `anna m`. Empty input
is missing evidence. First-name alias tuples are unordered.

### Surnames

`canonical_lasts_equivalent` ignores spaces, so `ou yang` equals `ouyang`.
Dash variants already became spaces during canonicalization.

The comparison also treats two kinds of surname variants as equal.

- Listed variants equal their canonical spellings, such as `mueller` and
  `muller`. The list is
  [s2and_surname_variants_canonical.txt](../s2and/data/s2and_surname_variants_canonical.txt),
  279 `variant,canonical` lines from people who write their name both ways.
- Slavic endings `-sky`, `-skiy`, `-skyi`, `-skyy`, `-skii`, `-skij`, and
  `-skyj` are equal after at least two letters, so `tartakovsky` equals
  `tartakovskiy`. `sky` and `osky` keep their own spelling.

The rule changes only the last-name veto. Canonical names, count keys, and
features keep the stored spelling. Python `canonical_surname_family` and Rust
`surname_family` apply it, and
[surname_family_examples.json](../tests/fixtures/surname_family_examples.json)
keeps the two equal.

### CJK given names

The Chinese, Japanese, and Korean (CJK) rule requires at least one Han, kana,
or hangul character in the raw first and middle names. Any other character must
transliterate to no letters. `cjk_given_name_for_constraint` joins the raw fields,
removes whitespace, normalizes kana width and Unicode composition (NFC), expands
`々` after Han, and transliterates the result. Thus `拓海` matches `拓海`, `奈々`
matches `奈奈`, `张伟` matches `張偉`, and `민수` matches `민 수`.

The script tag is `G` for hangul, otherwise `K` for kana, otherwise `H` for Han.
When both transliterations are complete and have the same tag, compare the
whole given names in place of the first- and middle-name checks. Different
values veto the pair, as in `重喜` and `重郷`, `孝` and `孝徳`, or `ひろ` and
`ひろし`. Use ordinary rules for romanized names (`Xiao` and `孝徳`), different
script tags (`민수` and `敏洙`), and incomplete transliteration (such as
`ピーター` or `𠮷秀`). Feature and count keys still use canonical_v2.

### Joined-initials and spelled-middle readings

With an empty raw middle name, `joined_initials_for_constraint` reads a
title-case first name of two or three ASCII letters as initials (`Ka` → `Jka`).
It excludes `_MIXED_CASE_NON_INITIALS`, `NAME_PREFIXES`, and
`_JOINED_INITIALS_GIVEN_NAMES` (`Ed`, `Md`). An ASCII capital followed by a
capitalized word of at least three letters gives an initial and a word
(`MArfan` → `Wm arfan`).

A reading can clear an ordinary given-name veto if exactly one side has a
joined-initials reading and the other side has a full first name and spells it
exactly. For `J`, every later token must be an initial or at least four letters.
For `W`, there must be exactly two tokens, with the first initial and second
word matching the reading. `Ka` matches `Kristan Allan` or `Kristian A.`, `Mal`
matches `Marco Aurelio Lisboa`, and `MArfan` matches `Mohammad Arfan`. `Li` and
`Lei`, `Ji` and `Jong In`, and `Ka` and `Kristian` still conflict.

`spelled_middle_initials_for_constraint` tests the last word of the raw middle
field, or the last word of the raw first field when the middle field is empty.
The word must pass D9's initials test and fall outside `_MIDDLE_NON_INITIALS`
(for example `JR`, `III`, and `Mc`). The raw first name must pass D9's case
check. Only the final middle token is split, so `Johanna A IP` agrees with
`Johanna A. Inge Petra`, while `IP A` leaves `IP` unspelled. `Johanna IP` gives
`Sip`, agrees with `Johanna Inge Petra`, and spells `Jip` through `Johanna I P`.
These readings affect the constraint only. Canonical names and count keys
keep the original tokenization.

### Middle names

Full middle names also use `same_prefix_tokens` before a full-token conflict
causes a veto. For example, `meg` matches `megan`, and `ch` matches `charles`.
The rule compares aligned tokens up to the shorter list, as for first names.
Middle initial conflicts still cause a veto. Seed, ORCID, surname, and first-name
rules retain their precedence. Alias tuples apply to first names only.

### Dash-bound first-name tails

A dash-bound first name can have more tokens than the other first name. When
the first names match by prefix or one is missing, the middle comparison starts
with the first-name tokens that the other first name does not cover. Thus `jean daniel`
agrees with first `jean` and middle `daniel`, and conflicts with first `jean`
and middle `paul`. A joined token covers the tokens that it contains, so
`xiaoqing` covers `xiao qing`. Two `jean pierre` records compare only their
middle names because all first-name tokens align. When the match comes only from an
alias, it covers the full first names, so no first-name token moves.

The live authorities are `s2and.text`, its Rust implementation, and
[the frozen examples](../tests/fixtures/canonical_name_examples.json).
Tuple and ORCID runtime rules live in
[production_inference.md](production_inference.md); Arrow and name-count
formats live in [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md).
`tests/test_name_tuple_artifact.py` keeps the reviewed tuple decisions in
`tests/fixtures/name_tuples/` consistent with the artifact.
