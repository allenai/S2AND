# Canonical names

This document is the durable contract for how S2AND canonicalizes raw
author names, builds name-count keys, and compares canonical names.

## Access the current rules and artifacts

Runtime `1.0.0` remains unpublished.
Use PR 80's `canonical-v2-migration` branch and its local Rust build for the current rules.
Skip automatic Git LFS downloads when you need only normalization and small artifacts:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch canonical-v2-migration https://github.com/allenai/S2AND.git
cd S2AND
uv sync --extra dev
```

The clone retains LFS pointers for the large count index, model weights, and test fixtures.
Name tuples and ORCID prefix counts are ordinary Git files and wheel assets.
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
import s2and_rust

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
from pathlib import Path

import s2and
from s2and.name_tuple_artifact import load_packaged_name_tuple_artifact
from s2and.orcid_prefix_counts import load_canonical_orcid_prefix_counts

aliases = load_packaged_name_tuple_artifact()
orcid = load_canonical_orcid_prefix_counts(Path(s2and.__file__).parent / "data")
assert orcid.name_tuples_sha256 == aliases.data_sha256
```

`aliases.pairs` contains canonical unordered first-name alias pairs.
`orcid.counts` contains first-name prefix-pair evidence; it is not the global name-frequency index.

Fetch the common count index only when you need global name frequencies:

```bash
git lfs install --local --skip-smudge
git lfs pull --include="artifacts/common/v1.3/name_counts_index/*.bin"
```

Use the canonical keys with the validated index:

```python
from s2and.name_counts_index import NameCountsIndex

index = NameCountsIndex.open("artifacts/common/v1.3/name_counts_index")
first, last, first_last, last_first_initial = index.lookup_many(
    [keys["first"]],
    [keys["last"]],
    [keys["first_last"]],
    [keys["last_first_initial"]],
)
```

The index's JSON manifest is ordinary Git; its four binary files use Git LFS.
Git preserves the manifest's exact bytes because dataset and model contracts bind its SHA-256.
The index requires about 2.39 GB of local space.
`None` keys yield `NaN`; informative keys absent from the index yield the default count `1.0`.
Use [the data guide](data.md) for model assets and the research snapshot.

## Canonical name contract

`s2and.text.canonicalize_name_parts` and its Rust equivalent are the
authorities. Given raw `first`, `middle`, and `last` values, they:

1. normalize Unicode spacing and remove soft hyphen and zero-width joiner;
2. normalize apostrophe-like marks, transliterate, lowercase, and delete
   apostrophes rather than making token boundaries;
3. treat supported Unicode dash variants as one separator;
4. replace remaining nonletters with spaces and collapse whitespace;
5. drop at most one leading title from first name, except `md`;
6. split 2-3 joined capitals, or a short mixed-case first name with no vowel,
   into initials when the last name is not in capitals (decision D9);
7. retain a dash-bound first-name group, otherwise keep the first token and
   spill remaining first-name tokens into middle; and
8. normalize last name independently, preserving spaces and surname particles.

Examples: `Anne-Marie Claire` becomes first `anne marie`, middle `claire`;
apostrophe variants of `O'Connor` become `oconnor`; and `Ou-Yang` and
`Ou Yang` both become last `ou yang`.

Decision D9 reads joined capitals as initials.
The rule removes whitespace and periods from the raw first name.
If 2-3 ASCII capitals remain, the first capital becomes first.
Each other capital becomes a middle token before the existing middle tokens.
For example, `HG` with last `Wells` becomes first `h`, middle `g`.
`JHM` becomes first `j`, middle `h m`.

The rule needs case evidence from the raw last name.
`last_name_allows_joined_initials` transliterates the raw last name with the
canonical table and then counts ASCII letters.
The rule applies only when the result has at least one lowercase letter and no
more capitals than lowercase letters.
Thus `á` counts as lowercase, and `HG` with last `WELLS` stays first `hg`.
The pinned transliteration tables make this definition identical in Python
and Rust.

The rule does not apply to these first names:

- `ED`, a frequent given name, and `MD`, which abbreviates Muhammad (D7);
- a title prefix such as `DR`, which D7 drops;
- a two-capital abbreviation of one given name with a final period: `CH`,
  `SH`, `ZH`, `KH`, `TS`, `YU`, `YA`, `YE`, `GY`, `NY`, `SZ`, `ZS`, `CS`, `PH`,
  or `TH`, for example `PH.` (Philippe), `GY.` (György), and `YU.` (Yuri).

A second branch of D9 reads short mixed-case first names as initials.
It removes whitespace from the raw first name, but it keeps periods.
If one ASCII capital and 1-2 lowercase ASCII letters remain, the branch reads
them as initials.
The letters must not include a vowel: `a`, `e`, `i`, `o`, `u`, or `y`.
For example, `Jh` with last `Smith` becomes first `j`, middle `h`.
`Jm` with middle `Kevin` becomes first `j`, middle `m kevin`.
The last-name rule is the same as for joined capitals.

The mixed-case branch does not apply to these first names:

- a first name with a period, such as `Jh.` or `Kh.`;
- a title prefix such as `Dr`, which D7 drops;
- `Ty` and other strings with a vowel or `y`;
- abbreviations of one given name: `ch`, `chr`, `cs`, `dzh`, `fr`, `gh`,
  `kh`, `khr`, `lj`, `ll`, `md`, `mhd`, `ph`, `sh`, `sk`, `st`, `sw`, `sz`,
  `th`, `vl`, `wm`, `zh`, and `zs`, for example `Md` (Muhammad) and `Kh`
  (Khristo);
- titles before the given name: `ds`, `hjh`, `lt`, `mst`, `rr`, and `smt`;
- the suffixes `jr` and `sr`;
- the real names `lv`, `ng`, and `tsz`; and
- the company word `ltd`.

A review of mention counts and gold clusters on 2026-10-16 selected this list.
Python `_MIXED_CASE_NON_INITIALS` and Rust `MIXED_CASE_NON_INITIALS` hold it.
A Rust test keeps the two lists equal.

Every caller must supply the raw last name.
A count generator can canonicalize first and last names in separate maps.
Such a generator computes `last_name_allows_joined_initials` once for each
distinct last name.
It then supplies that result to `canonicalize_name_parts` as
`last_allows_initials`.

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

Full middle names also use `same_prefix_tokens` before a full-token conflict
causes a veto. For example, `meg` matches `megan`, and `ch` matches `charles`.
The rule compares aligned tokens up to the shorter list, as for first names.
Middle initial conflicts still cause a veto. Seed, ORCID, surname, and first-name
rules retain their precedence. Middle names do not use the alias tuple file.

A dash-bound first name can have more tokens than the other first name. When
the first names match by prefix, the middle comparison starts with the
first-name tokens that the other first name does not cover. Thus `jean daniel`
agrees with first `jean` and middle `daniel`, and conflicts with first `jean`
and middle `paul`. A joined token covers the tokens that it contains, so
`xiaoqing` covers `xiao qing`. Aligned tokens do not move: two `jean pierre`
records compare only their middle names. An alias match covers the full first
names, so no first-name token moves.

The live authorities are `s2and.text`, its Rust implementation, and
[the frozen examples](../tests/fixtures/canonical_name_examples.json).
Tuple and ORCID runtime rules live in
[production_inference.md](production_inference.md); Arrow and name-count
formats live in [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md).
`tests/test_name_tuple_artifact.py` keeps the reviewed tuple decisions in
`tests/fixtures/name_tuples/` consistent with the artifact.
