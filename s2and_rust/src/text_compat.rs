use pyo3::prelude::PyResult;
use std::collections::{HashMap, HashSet};

pub(crate) fn ensure_unidecode_for_text(
    text: &str,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    if text.is_ascii() {
        return Ok(());
    }
    for ch in text.chars() {
        if ch.is_ascii() || unidecode_char_map.contains_key(&ch) {
            continue;
        }
        unidecode_char_map.insert(ch, text_unidecode_char(ch).to_string());
    }
    Ok(())
}

fn text_unidecode_char(ch: char) -> &'static str {
    match ch as u32 {
        0x0080 | 0x0082 | 0x0083 | 0x0084 | 0x0085 | 0x0086 | 0x0087 | 0x0088 | 0x0089 | 0x008A
        | 0x008B | 0x008C | 0x008E | 0x0091 | 0x0092 | 0x0093 | 0x0094 | 0x0095 | 0x0096
        | 0x0097 | 0x0098 | 0x0099 | 0x009A | 0x009B | 0x009C | 0x009E | 0x009F | 0x02E5
        | 0x02E6 | 0x02E7 | 0x02E8 | 0x02E9 | 0x02EA | 0x02EB | 0xFDF0 | 0xFDF1 | 0xFDF2
        | 0xFDF3 | 0xFDF4 | 0xFDF5 | 0xFDF6 | 0xFDF7 | 0xFDF8 | 0xFDF9 | 0xFDFA | 0xFDFB => "",
        0x02FF | 0x03FF | 0x04FF | 0x05FF | 0x06FF | 0x07FF | 0x09FF | 0x0AFF | 0x0BFF | 0x0CFF
        | 0x0DFF | 0x0EFF | 0x0FFF | 0x10FF | 0x11FF | 0x13FF | 0x16FF | 0x17FF | 0x18FF
        | 0x1EFF | 0x1FFF | 0x20FF | 0x21FF | 0x22FF | 0x23FF | 0x24FF | 0x25FF | 0x26FF
        | 0x27FF | 0x2EFF | 0x2FFF | 0x30FF | 0x31FF | 0x32FF | 0x33FF | 0x4DFF | 0x9FFF
        | 0xA4FF | 0xD7FF | 0xFAFF | 0xFDFF => "[?] ",
        0x25F4 | 0x25F5 | 0x25F6 | 0x25F7 => "#",
        0x02EF | 0x02F0 | 0x02F1 | 0x02F2 | 0x02F3 | 0x02F4 | 0x02F5 | 0x02F6 | 0x02F7 | 0x02F8
        | 0x02F9 | 0x02FA | 0x02FB | 0x02FC | 0x02FD | 0x02FE | 0x03F4 | 0x03F5 | 0x03F6
        | 0x03F7 | 0x03F8 | 0x03F9 | 0x03FC | 0x03FD | 0x03FE | 0x0AF0 | 0x0AF1 | 0x0AF9
        | 0x13F5 | 0x13F8 | 0x13F9 | 0x13FA | 0x13FB | 0x13FC | 0x13FD | 0x1EFA | 0x1EFB
        | 0x1EFC | 0x1EFD | 0x1EFE | 0x25F8 | 0x25F9 | 0x25FA | 0x25FB | 0x25FC | 0x25FD
        | 0x25FE | 0xFDFC | 0xFDFD => "[?]",
        _ => unidecode::unidecode_char(ch),
    }
}

pub(crate) fn normalize_ascii_text_compat(text: &str, special_case_apostrophes: bool) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut prev_space = true;
    for byte in text.bytes() {
        let lowered = byte.to_ascii_lowercase();
        if lowered.is_ascii_alphabetic() {
            normalized.push(lowered as char);
            prev_space = false;
        } else if special_case_apostrophes && lowered == b'\'' {
            continue;
        } else if !prev_space {
            normalized.push(' ');
            prev_space = true;
        }
    }
    while normalized.ends_with(' ') {
        normalized.pop();
    }
    normalized
}

pub(crate) fn normalize_ascii_title_compat(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut prev_space = true;
    for byte in text.bytes() {
        let lowered = byte.to_ascii_lowercase();
        if lowered.is_ascii_alphanumeric() {
            normalized.push(lowered as char);
            prev_space = false;
        } else if !prev_space {
            normalized.push(' ');
            prev_space = true;
        }
    }
    while normalized.ends_with(' ') {
        normalized.pop();
    }
    normalized
}

#[cfg(test)]
pub(crate) fn normalize_text_compat_native(text: &str, special_case_apostrophes: bool) -> String {
    normalize_text_compat_with_map(text, special_case_apostrophes, None)
}

pub(crate) fn normalize_text_compat_from_map(
    text: &str,
    special_case_apostrophes: bool,
    unidecode_char_map: &HashMap<char, String>,
) -> String {
    normalize_text_compat_with_map(text, special_case_apostrophes, Some(unidecode_char_map))
}

pub(crate) fn normalize_title_compat_from_map(
    text: &str,
    unidecode_char_map: &HashMap<char, String>,
) -> String {
    if text.is_empty() {
        return String::new();
    }
    if text.is_ascii() {
        return normalize_ascii_title_compat(text);
    }

    let mut transliterated = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            transliterated.push(ch.to_ascii_lowercase());
            continue;
        }
        let mapped = unidecode_char_map
            .get(&ch)
            .map(String::as_str)
            .unwrap_or_else(|| text_unidecode_char(ch));
        for mapped_ch in mapped.chars() {
            transliterated.push(mapped_ch.to_ascii_lowercase());
        }
    }
    normalize_ascii_title_compat(&transliterated)
}

fn normalize_text_compat_with_map(
    text: &str,
    special_case_apostrophes: bool,
    unidecode_char_map: Option<&HashMap<char, String>>,
) -> String {
    if text.is_empty() {
        return String::new();
    }
    if text.is_ascii() {
        return normalize_ascii_text_compat(text, special_case_apostrophes);
    }

    let mut transliterated = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            transliterated.push(ch.to_ascii_lowercase());
            continue;
        }
        let mapped = unidecode_char_map
            .and_then(|char_map| char_map.get(&ch).map(String::as_str))
            .unwrap_or_else(|| text_unidecode_char(ch));
        for mapped_ch in mapped.chars() {
            transliterated.push(mapped_ch.to_ascii_lowercase());
        }
    }

    let source = if special_case_apostrophes {
        transliterated.replace('\'', "")
    } else {
        transliterated
    };
    let mut normalized = String::with_capacity(source.len());
    let mut prev_space = true;
    for ch in source.chars() {
        if ch.is_ascii_alphabetic() {
            normalized.push(ch);
            prev_space = false;
        } else if !prev_space {
            normalized.push(' ');
            prev_space = true;
        }
    }
    while normalized.ends_with(' ') {
        normalized.pop();
    }
    normalized
}

fn is_name_dash(ch: char) -> bool {
    matches!(
        ch,
        '-' | '\u{2010}'
            | '\u{2011}'
            | '\u{2012}'
            | '\u{2013}'
            | '\u{2014}'
            | '\u{2212}'
            | '\u{FE58}'
            | '\u{FE63}'
            | '\u{FF0D}'
    )
}

// D3 apostrophe-like marks (mirrors Python `NAME_APOSTROPHE_LIKE_CHARS` in
// s2and/text.py): ASCII apostrophe, backtick, spacing acute, curly quotes,
// modifier letters (okina/apostrophe), primes, saltillo, U+FE4D (classified with
// apostrophe-like marks by issue #39 despite its Unicode name), fullwidth
// apostrophe.
fn is_name_apostrophe_like(ch: char) -> bool {
    matches!(
        ch,
        '\'' | '`'
            | '\u{00B4}'
            | '\u{2018}'
            | '\u{2019}'
            | '\u{02BB}'
            | '\u{02BC}'
            | '\u{2032}'
            | '\u{2035}'
            | '\u{A78C}'
            | '\u{FE4D}'
            | '\u{FF07}'
    )
}

// Invisible formatting controls deleted before tokenization (mirrors Python
// `_NAME_INVISIBLE_FORMAT_CHARS`): soft hyphen (not a dash separator) and
// zero-width joiner.
fn is_name_invisible_format(ch: char) -> bool {
    matches!(ch, '\u{00AD}' | '\u{200D}')
}

/// canonical_v2 pre-translation on raw code points, before transliteration
/// (mirrors Python `_canonical_name_pretranslate`): delete invisible format
/// controls, unify apostrophe-like marks to ASCII apostrophe, and unify
/// dash-like characters to ASCII hyphen.
fn canonical_name_pretranslate(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if is_name_invisible_format(ch) {
            continue;
        }
        if is_name_apostrophe_like(ch) {
            out.push('\'');
        } else if is_name_dash(ch) {
            out.push('-');
        } else {
            out.push(ch);
        }
    }
    out
}

/// canonical_v2 token normalization of a pre-translated string (mirrors Python
/// `_canonical_name_tokens`): transliterate, lowercase, delete apostrophes and
/// backticks post-transliteration (no token boundary), and split on everything
/// that is not an ASCII letter. Dash binding is decided by the caller before
/// this runs.
fn canonical_name_tokens(
    pretranslated: &str,
    unidecode_char_map: Option<&HashMap<char, String>>,
) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let consume = |mapped_ch: char, current: &mut String, tokens: &mut Vec<String>| {
        let lowered = mapped_ch.to_ascii_lowercase();
        if lowered.is_ascii_lowercase() {
            current.push(lowered);
        } else if lowered == '\'' || lowered == '`' {
            // Deleted, not a separator: O'Brien -> obrien.
        } else if !current.is_empty() {
            tokens.push(std::mem::take(current));
        }
    };
    for ch in pretranslated.chars() {
        if ch.is_ascii() {
            consume(ch, &mut current, &mut tokens);
            continue;
        }
        let mapped = unidecode_char_map
            .and_then(|char_map| char_map.get(&ch).map(String::as_str))
            .unwrap_or_else(|| text_unidecode_char(ch));
        for mapped_ch in mapped.chars() {
            consume(mapped_ch, &mut current, &mut tokens);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Canonicalize a whole name string to spaced canonical_v2 tokens (mirrors
/// Python `canonicalize_name_text`). This is the normalization for canonical
/// middle and last fields and for whole-string artifact keys.
pub(crate) fn canonicalize_name_text_compat(
    raw: &str,
    unidecode_char_map: Option<&HashMap<char, String>>,
) -> String {
    canonical_name_tokens(&canonical_name_pretranslate(raw), unidecode_char_map).join(" ")
}

/// Canonicalize raw first/middle/last per the canonical_v2 pipeline (mirrors
/// Python `canonicalize_name_parts`). Returns (first, middle, last).
///
/// - At most one leading title-prefix token is dropped from first (D7).
/// - First/middle split (D1): a leading dash-bound group stays together in
///   first as spaced tokens; otherwise the first token stays and later tokens
///   spill into middle ahead of existing middle tokens. Space tokens after a
///   dash-bound group still spill.
pub(crate) fn canonicalize_name_parts_compat(
    first_raw: &str,
    middle_raw: &str,
    last_raw: &str,
    name_prefixes: &HashSet<String>,
    unidecode_char_map: Option<&HashMap<char, String>>,
) -> (String, String, String) {
    let first_clean = canonical_name_pretranslate(first_raw);
    let middle_text = canonicalize_name_text_compat(middle_raw, unidecode_char_map);
    let last = canonicalize_name_text_compat(last_raw, unidecode_char_map);

    // Whitespace chunks of the raw first field, each normalized to tokens and
    // tagged with whether a dash bound it together.
    let mut flattened: Vec<(String, usize)> = Vec::new();
    let mut dash_bound: Vec<bool> = Vec::new();
    for (group_index, chunk) in first_clean.split_whitespace().enumerate() {
        dash_bound.push(chunk.contains('-'));
        for token in canonical_name_tokens(&chunk.replace('-', " "), unidecode_char_map) {
            flattened.push((token, group_index));
        }
    }

    if let Some((token, _group)) = flattened.first() {
        if name_prefixes.contains(token) {
            flattened.remove(0);
        }
    }

    let (first_field_tokens, spilled_tokens): (Vec<String>, Vec<String>) = if flattened.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        let lead_group = flattened[0].1;
        if dash_bound[lead_group] {
            // partition preserves order within both halves.
            let (kept, spilled): (Vec<_>, Vec<_>) = flattened
                .into_iter()
                .partition(|(_token, group)| *group == lead_group);
            (
                kept.into_iter().map(|(token, _)| token).collect(),
                spilled.into_iter().map(|(token, _)| token).collect(),
            )
        } else {
            let mut iter = flattened.into_iter().map(|(token, _)| token);
            let head = iter.next().expect("flattened is non-empty");
            (vec![head], iter.collect())
        }
    };

    let first = first_field_tokens.join(" ");
    let middle = if spilled_tokens.is_empty() {
        middle_text
    } else if middle_text.is_empty() {
        spilled_tokens.join(" ")
    } else {
        format!("{} {}", spilled_tokens.join(" "), middle_text)
    };
    (first, middle, last)
}

/// Canonical_v2 count keys built from canonical fields after gating (mirrors
/// Python `canonical_name_count_keys`, D6/D8). A `None` key means no lookup
/// (NaN feature), never a sentinel count. `first` and `first_last` require an
/// informative first (string length > 1); `last_first_initial` requires first
/// and last present and stays initial-char semantics.
pub(crate) struct CanonicalNameCountKeys {
    pub(crate) first: Option<String>,
    pub(crate) last: Option<String>,
    pub(crate) first_last: Option<String>,
    pub(crate) last_first_initial: Option<String>,
}

pub(crate) fn canonical_name_count_keys_compat(first: &str, last: &str) -> CanonicalNameCountKeys {
    let first_informative = crate::py_len(first) > 1;
    let last_present = !last.is_empty();
    CanonicalNameCountKeys {
        first: first_informative.then(|| first.to_string()),
        last: last_present.then(|| last.to_string()),
        first_last: (first_informative && last_present).then(|| format!("{first} {last}")),
        last_first_initial: (!first.is_empty() && last_present).then(|| {
            let first_initial = first.chars().next().expect("first is non-empty");
            format!("{last} {first_initial}")
        }),
    }
}

pub(crate) fn compute_block_compat(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let name_parts: Vec<&str> = name.split(' ').collect();
    if name_parts.len() == 1 {
        return name_parts[0].to_string();
    }
    let Some(first_initial) = name_parts[0].chars().next() else {
        return String::new();
    };
    format!("{} {}", first_initial, name_parts[name_parts.len() - 1])
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_name_count_keys_compat, canonicalize_name_parts_compat,
        normalize_text_compat_native, text_unidecode_char,
    };
    use std::collections::HashSet;
    use std::path::Path;

    #[test]
    fn text_unidecode_char_matches_python_text_unidecode_compat_overrides() {
        assert_eq!(text_unidecode_char('\u{0080}'), "");
        assert_eq!(text_unidecode_char('\u{02EF}'), "[?]");
        assert_eq!(text_unidecode_char('\u{02FF}'), "[?] ");
        assert_eq!(text_unidecode_char('\u{25F4}'), "#");
        assert_eq!(text_unidecode_char('\u{FDFD}'), "[?]");
    }

    // Mirrors `NAME_PREFIXES` in s2and/text.py. Note "md" is deliberately NOT a
    // prefix (D7): it is a common South Asian given-name abbreviation.
    fn python_name_prefixes() -> HashSet<String> {
        [
            "dr",
            "prof",
            "professor",
            "mr",
            "miss",
            "mrs",
            "ms",
            "mx",
            "sir",
            "phd",
            "doctor",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    }

    fn fixture_string(value: &serde_json::Value) -> String {
        // Fixture inputs use null for missing fields; canonicalization treats
        // null and empty identically.
        value.as_str().unwrap_or("").to_string()
    }

    #[test]
    fn canonical_name_fixture_matches_python_reference_for_all_cases() {
        let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests")
            .join("fixtures")
            .join("canonical_name_examples.json");
        let fixture_text =
            std::fs::read_to_string(&fixture_path).expect("read canonical_name_examples.json");
        let fixture: serde_json::Value =
            serde_json::from_str(&fixture_text).expect("parse canonical_name_examples.json");
        assert_eq!(
            fixture["normalization_version"].as_str(),
            Some("canonical_v2")
        );
        let cases = fixture["cases"].as_array().expect("fixture cases array");
        assert!(!cases.is_empty(), "fixture has no cases");

        let name_prefixes = python_name_prefixes();
        for case in cases {
            let case_id = case["id"].as_str().expect("case id");
            let input = &case["input"];
            let canonical = &case["canonical"];
            // Native transliteration table (no Python-supplied unidecode map):
            // the crate's `text_unidecode_char` carries the python parity
            // overrides, so the fixture must pass without a supplied map.
            let (first, middle, last) = canonicalize_name_parts_compat(
                &fixture_string(&input["first"]),
                &fixture_string(&input["middle"]),
                &fixture_string(&input["last"]),
                &name_prefixes,
                None,
            );
            assert_eq!(
                first,
                canonical["first"].as_str().expect("canonical first"),
                "case {case_id}: first"
            );
            assert_eq!(
                middle,
                canonical["middle"].as_str().expect("canonical middle"),
                "case {case_id}: middle"
            );
            assert_eq!(
                last,
                canonical["last"].as_str().expect("canonical last"),
                "case {case_id}: last"
            );

            let expected_keys = &canonical["count_keys"];
            let keys = canonical_name_count_keys_compat(&first, &last);
            for (name, actual) in [
                ("first", &keys.first),
                ("last", &keys.last),
                ("first_last", &keys.first_last),
                ("last_first_initial", &keys.last_first_initial),
            ] {
                let expected = expected_keys[name].as_str();
                assert_eq!(
                    actual.as_deref(),
                    expected,
                    "case {case_id}: count key {name}"
                );
            }
        }
    }

    #[test]
    fn normalize_text_compat_preserves_python_boundaries_for_overrides() {
        assert_eq!(
            normalize_text_compat_native("a\u{0080}b \u{03F4}c \u{02FF}d", false),
            "ab c d",
        );
    }
}
