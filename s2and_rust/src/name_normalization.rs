//! Public canonical_v2 name helpers share the runtime normalization rules.

use crate::HashMap;
use std::sync::OnceLock;

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::text_compat::{
    canonical_name_count_keys_compat, canonicalize_name_parts_compat, canonicalize_name_text_compat,
};
use crate::HashSet;

/// These title tokens match Python `s2and.text.NAME_PREFIXES`.
pub const NAME_PREFIXES: [&str; 11] = [
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
];

/// Canonical first, middle, and last fields preserve name token boundaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalNameParts {
    pub first: String,
    pub middle: String,
    pub last: String,
}

/// Absent keys suppress count lookups for missing or uninformative fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalNameCountKeys {
    pub first: Option<String>,
    pub last: Option<String>,
    pub first_last: Option<String>,
    pub last_first_initial: Option<String>,
}

fn name_prefixes() -> &'static HashSet<String> {
    static PREFIXES: OnceLock<HashSet<String>> = OnceLock::new();
    PREFIXES.get_or_init(|| NAME_PREFIXES.into_iter().map(str::to_string).collect())
}

/// Normalize raw fields with the canonical_v2 rules. Empty strings represent missing fields.
pub fn canonicalize_name_parts(
    first_raw: &str,
    middle_raw: &str,
    last_raw: &str,
) -> CanonicalNameParts {
    let (first, middle, last) =
        canonicalize_name_parts_compat(first_raw, middle_raw, last_raw, name_prefixes(), None);
    CanonicalNameParts {
        first,
        middle,
        last,
    }
}

/// Normalize whole name text without the first/middle split or title removal.
pub fn canonicalize_name_text(raw: &str) -> String {
    canonicalize_name_text_compat(raw, None)
}

/// Build index keys from canonical fields. Normalize raw fields before this call.
pub fn canonical_name_count_keys(parts: &CanonicalNameParts) -> CanonicalNameCountKeys {
    let keys = canonical_name_count_keys_compat(&parts.first, &parts.last);
    CanonicalNameCountKeys {
        first: keys.first,
        last: keys.last,
        first_last: keys.first_last,
        last_first_initial: keys.last_first_initial,
    }
}

#[pyfunction(name = "canonicalize_name_parts", signature = (first_raw, middle_raw, last_raw))]
fn py_canonicalize_name_parts(
    first_raw: Option<&str>,
    middle_raw: Option<&str>,
    last_raw: Option<&str>,
) -> (String, String, String) {
    let parts = canonicalize_name_parts(
        first_raw.unwrap_or(""),
        middle_raw.unwrap_or(""),
        last_raw.unwrap_or(""),
    );
    (parts.first, parts.middle, parts.last)
}

#[pyfunction(name = "canonicalize_name_text", signature = (raw))]
fn py_canonicalize_name_text(raw: Option<&str>) -> String {
    canonicalize_name_text(raw.unwrap_or(""))
}

#[pyfunction(name = "canonical_name_count_keys")]
fn py_canonical_name_count_keys(
    parts: (String, String, String),
) -> HashMap<String, Option<String>> {
    let keys = canonical_name_count_keys(&CanonicalNameParts {
        first: parts.0,
        middle: parts.1,
        last: parts.2,
    });
    HashMap::from_iter([
        ("first".to_string(), keys.first),
        ("last".to_string(), keys.last),
        ("first_last".to_string(), keys.first_last),
        ("last_first_initial".to_string(), keys.last_first_initial),
    ])
}

pub(crate) fn add_to_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_canonicalize_name_parts, module)?)?;
    module.add_function(wrap_pyfunction!(py_canonicalize_name_text, module)?)?;
    module.add_function(wrap_pyfunction!(py_canonical_name_count_keys, module)?)?;
    module.add("NAME_PREFIXES", NAME_PREFIXES.to_vec())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_helpers_match_reviewed_canonical_name_cases() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/canonical_name_examples.json"
        ))
        .expect("parse canonical name fixture");
        let cases = fixture["cases"].as_array().expect("fixture cases");
        assert!(!cases.is_empty());
        for case in cases {
            let id = case["id"].as_str().expect("case ID");
            let input = &case["input"];
            let expected = &case["canonical"];
            let parts = canonicalize_name_parts(
                input["first"].as_str().unwrap_or(""),
                input["middle"].as_str().unwrap_or(""),
                input["last"].as_str().unwrap_or(""),
            );
            assert_eq!(parts.first, expected["first"].as_str().unwrap(), "{id}");
            assert_eq!(parts.middle, expected["middle"].as_str().unwrap(), "{id}");
            assert_eq!(parts.last, expected["last"].as_str().unwrap(), "{id}");
            assert_eq!(
                canonicalize_name_text(input["last"].as_str().unwrap_or("")),
                parts.last,
                "{id}: whole surname"
            );
            let keys = canonical_name_count_keys(&parts);
            for (kind, key) in [
                ("first", keys.first),
                ("last", keys.last),
                ("first_last", keys.first_last),
                ("last_first_initial", keys.last_first_initial),
            ] {
                assert_eq!(
                    key.as_deref(),
                    expected["count_keys"][kind].as_str(),
                    "{id}: {kind}"
                );
            }
        }
    }

    #[test]
    fn public_prefixes_match_python_source() {
        let python_source = include_str!("../../s2and/text.py");
        let prefix_literal = python_source
            .split_once("NAME_PREFIXES = {")
            .expect("Python prefix set")
            .1
            .split_once('}')
            .expect("Python prefix set end")
            .0;
        let python_prefixes: HashSet<String> = prefix_literal
            .split(',')
            .map(|token| token.trim().trim_matches(['\'', '"']).to_string())
            .collect();
        assert_eq!(name_prefixes(), &python_prefixes);
        assert!(!name_prefixes().contains("md"));
    }

    #[test]
    fn optional_python_inputs_match_empty_rust_fields() {
        assert_eq!(
            py_canonicalize_name_parts(None, None, None),
            (String::new(), String::new(), String::new())
        );
        assert_eq!(py_canonicalize_name_text(None), canonicalize_name_text(""));
        let parts = py_canonicalize_name_parts(Some("Dr Anne–Marie Louise"), None, Some("O’Brien"));
        assert_eq!(
            parts,
            ("anne marie".into(), "louise".into(), "obrien".into())
        );
        let keys = py_canonical_name_count_keys(parts);
        assert_eq!(keys["first_last"].as_deref(), Some("anne marie obrien"));
        assert_eq!(keys["last_first_initial"].as_deref(), Some("obrien a"));
        let empty_keys = py_canonical_name_count_keys(("".into(), "".into(), "".into()));
        assert!(empty_keys.values().all(Option::is_none));
    }
}
