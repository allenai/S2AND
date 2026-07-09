use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule, PyTuple};
use pyo3::Bound;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

mod arrow_batch_lookup;
mod constraints;
mod features;
mod ingest_dataset;
mod language_detection;
mod lightgbm_booster;
mod name_counts;
mod orcid;
mod pair_indexing;
mod promoted_linker;
mod raw_arrow;
mod raw_arrow_features;
mod raw_candidate_planner;
mod rayon_pool;
mod retrieval;
mod rust_featurizer;
mod subblocking;
mod text_compat;

use arrow_batch_lookup::IndexedArrowReadStats;
use constraints::{
    first_names_name_compatible, lasts_equivalent_for_constraint, same_prefix_tokens,
};
use features::*;
pub(crate) use ingest_dataset::*;
use language_detection::LanguageDetectorCompat;
use name_counts::{
    NameCountsData, NameCountsLastFirstInitialSemantics, RawNameCountKind, RawNameCountMaps,
};
use orcid::{normalize_orcid_compact_owned, normalize_orcid_owned};
use pair_indexing::upper_triangle_pairs_for_range;
use raw_arrow::paths::{
    extract_name_counts_index_path, extract_path_mapping_string,
    raw_arrow_feature_paths_from_py_dict, required_path_from_py_dict, RawArrowPlannerPaths,
};
use raw_arrow::readers::{
    read_raw_arrow_cluster_seed_disallows, read_raw_arrow_cluster_seeds,
    read_raw_arrow_paper_authors_with_optional_index, read_raw_arrow_papers_with_optional_index,
    read_raw_arrow_query_signatures, read_raw_arrow_signatures_with_optional_index,
    read_raw_arrow_specter_with_optional_index, read_raw_name_counts_index,
    RawArrowAuthorSignalData, RawArrowFeature, RawArrowPaper, RawArrowQuerySignatureRequest,
    RawArrowSignature, RawArrowSummarySignalData,
};
use raw_arrow_features::{
    build_raw_arrow_author_signal_data, build_raw_arrow_feature, build_raw_arrow_summary,
    build_raw_arrow_summary_signals, mask_raw_arrow_query, raw_arrow_name_count_rarity_row,
    raw_arrow_paper_evidence_row, round_six,
};
#[cfg(test)]
use raw_candidate_planner::raw_arrow_summary_signals_for_members_cached;
use raw_candidate_planner::{raw_arrow_labeled_candidate_plan, RawBlockQueryCandidatePlanner};
use rayon_pool::install_with_optional_rayon_pool;
pub(crate) use retrieval::*;
pub(crate) use rust_featurizer::RustFeaturizer;
use subblocking::*;
use text_compat::{
    compute_block_compat, contains_name_dash, ensure_unidecode_for_text,
    normalize_text_compat_from_map, split_first_middle_hyphen_aware_compat,
};

fn py_len(s: &str) -> usize {
    // Python len() semantics for str: count of Unicode scalar values
    s.chars().count()
}

const DROPPED_AFFIXES: [&str; 48] = [
    "ab", "am", "ap", "abu", "al", "auf", "aus", "bar", "bath", "bat", "bet", "bint", "dall",
    "dalla", "das", "de", "degli", "del", "dell", "della", "dem", "den", "der", "di", "do", "dos",
    "ds", "du", "el", "ibn", "im", "jr", "la", "las", "le", "los", "mac", "mc", "mhic", "mic",
    "ter", "und", "van", "vom", "von", "zu", "zum", "zur",
];

const FNV_OFFSET: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;
fn is_dropped_affix(token: &str) -> bool {
    DROPPED_AFFIXES.contains(&token)
}

#[derive(Clone, Serialize, Deserialize)]
struct CounterData {
    // FNV-1a 64-bit hashes of original string keys, sorted ascending.
    // Values are f32 counts. Binary search replaces HashMap lookup.
    // Trade-off: collisions are possible but rare at expected key counts.
    // If they occur, colliding keys merge silently in this representation.
    entries: Vec<(u64, f32)>,
    sum: f32,
}

#[inline(always)]
fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[inline(always)]
fn fnv64_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[inline]
pub(crate) fn vector_norm_f32(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| {
            let val = *value as f64;
            val * val
        })
        .sum::<f64>()
        .sqrt()
}

#[inline(always)]
fn read_u64_le_unchecked(bytes: &[u8], offset: usize) -> u64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(raw)
}

#[inline(always)]
fn read_u32_le_unchecked(bytes: &[u8], offset: usize) -> u32 {
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(raw)
}

#[inline(always)]
fn read_f64_le_unchecked(bytes: &[u8], offset: usize) -> f64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[offset..offset + 8]);
    f64::from_le_bytes(raw)
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
enum ClusterId {
    Int(i64),
    Str(String),
}

type PaperId = String;

#[derive(Clone, Serialize, Deserialize)]
struct SignatureData {
    // Python author_info_first_normalized_without_apostrophe.
    first: Option<String>,
    middle: Option<String>,
    last_normalized: Option<String>,
    orcid: Option<String>,
    email: Option<String>,
    affiliations: Option<CounterData>,
    coauthor_blocks: Option<HashSet<String>>,
    coauthor_ngrams: Option<CounterData>,
    coauthors: Option<HashSet<String>>,
    position: i64,
    paper_id: PaperId,
    name_counts: Option<NameCountsData>,
    // Same canonical first-name field used by Python name_text_features.
    adv_name: Option<String>,
}

impl SignatureData {
    fn first_without_apostrophe(&self) -> Option<&str> {
        self.first.as_deref()
    }

    fn adv_name_for_features(&self) -> Option<&str> {
        self.adv_name.as_deref()
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct PaperData {
    venue_ngrams: Option<CounterData>,
    title_words: Option<CounterData>,
    title_chars: Option<CounterData>,
    year: Option<i64>,
    has_abstract: bool,
    predicted_language: Option<String>,
    is_reliable: bool,
    journal_ngrams: Option<CounterData>,
    specter: Option<Vec<f32>>,
    #[serde(default)]
    specter_norm: Option<f64>,
}

#[derive(Clone)]
struct StageSignatureInput {
    sig_id: String,
    paper_id: PaperId,
    raw_first: String,
    raw_middle: String,
    raw_last: String,
    email: Option<String>,
    position: i64,
    affiliation_values: Vec<String>,
    orcid: Option<String>,
}

#[derive(Clone)]
struct StagePaperInput {
    paper_id: PaperId,
    raw_title: String,
    raw_venue: String,
    raw_journal: String,
    raw_authors: Vec<(i64, String)>,
    year: Option<i64>,
    has_abstract: bool,
    predicted_language: Option<String>,
    is_reliable: bool,
}

#[derive(Clone)]
struct StagePaperPreprocessed {
    authors: Vec<(i64, String)>,
    year: Option<i64>,
    has_abstract: bool,
    predicted_language: Option<String>,
    is_reliable: bool,
    title_words: Option<CounterData>,
    title_chars: Option<CounterData>,
    venue_ngrams: Option<CounterData>,
    journal_ngrams: Option<CounterData>,
}

#[pyfunction]
fn get_build_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let build_info = PyDict::new(py);
    build_info.set_item("crate_version", env!("CARGO_PKG_VERSION"))?;
    build_info.set_item("profile", option_env!("PROFILE").unwrap_or("unknown"))?;
    build_info.set_item("debug_assertions", cfg!(debug_assertions))?;
    build_info.set_item("debug", option_env!("DEBUG").unwrap_or("unknown"))?;
    build_info.set_item("opt_level", option_env!("OPT_LEVEL").unwrap_or("unknown"))?;
    build_info.set_item("target", option_env!("TARGET").unwrap_or("unknown"))?;
    build_info.set_item("host", option_env!("HOST").unwrap_or("unknown"))?;
    build_info.set_item("rustc", option_env!("RUSTC").unwrap_or("unknown"))?;
    build_info.set_item(
        "incremental_linking_pair_plan_row_signals",
        INCREMENTAL_LINKING_PAIR_PLAN_ROW_SIGNALS.to_vec(),
    )?;
    build_info.set_item(
        "incremental_linking_pair_plan_supported_kwargs",
        INCREMENTAL_LINKING_PAIR_PLAN_SUPPORTED_KWARGS.to_vec(),
    )?;
    build_info.set_item(
        "raw_arrow_query_signature_planner_methods",
        RAW_ARROW_QUERY_SIGNATURE_PLANNER_METHODS.to_vec(),
    )?;
    Ok(build_info.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARROW_SCHEMA_CONTRACT_COLUMNS: &[(&str, &str, &str, bool)] = &[
        ("altered_cluster_signatures", "signature_id", "string", true),
        ("cluster_seed_disallows", "signature_id_1", "string", true),
        ("cluster_seed_disallows", "signature_id_2", "string", true),
        ("cluster_seeds", "signature_id", "string", true),
        ("cluster_seeds", "cluster_id", "string", true),
        ("paper_authors", "paper_id", "string", true),
        ("paper_authors", "position", "int64", true),
        ("paper_authors", "author_name", "string", true),
        ("papers", "paper_id", "string", true),
        ("papers", "title", "string", true),
        ("papers", "abstract", "string", false),
        ("papers", "venue", "string", true),
        ("papers", "journal_name", "string", true),
        ("papers", "year", "int64", false),
        ("papers", "predicted_language", "string", false),
        ("papers", "is_reliable", "bool", false),
        ("signatures", "signature_id", "string", true),
        ("signatures", "paper_id", "string", true),
        ("signatures", "author_first", "string", true),
        ("signatures", "author_middle", "string", true),
        ("signatures", "author_last", "string", true),
        ("signatures", "author_suffix", "string", true),
        ("signatures", "author_affiliations", "list<string>", true),
        ("signatures", "author_orcid", "string", true),
        ("signatures", "author_position", "int64", true),
        ("signatures", "author_block", "string", false),
        ("signatures", "author_email", "string", false),
        ("signatures", "source_author_ids", "list<string>", false),
        ("specter", "paper_id", "string", true),
        ("specter", "embedding", "fixed_size_list<float32>", true),
    ];

    fn raw_arrow_feature_for_test(first: &str, year: Option<i64>) -> RawArrowFeature {
        RawArrowFeature {
            query: RetrievalQueryData {
                first: first.to_string(),
                has_full_first: py_len(first) > 1,
                middle_initial_hashes: Vec::new(),
                coauthor_hashes: Vec::new(),
                coauthor_terms: Vec::new(),
                affiliation_hashes: Vec::new(),
                affiliation_terms: Vec::new(),
                venue_hashes: Vec::new(),
                title_hashes: Vec::new(),
                year,
                orcid_hash: None,
                specter: None,
                specter_norm: None,
            },
            name_counts: None,
            paper_author_count: 0,
            query_author: first.to_string(),
        }
    }

    fn raw_arrow_signature_for_test(paper_id: &str) -> RawArrowSignature {
        RawArrowSignature {
            paper_id: paper_id.to_string(),
            author_first: String::new(),
            author_middle: String::new(),
            author_last: String::new(),
            author_suffix: String::new(),
            author_block: None,
            affiliations: Vec::new(),
            email: None,
            orcid: None,
            position: None,
        }
    }

    fn prepare_python_for_test() {
        #[cfg(windows)]
        if let Some(python_home) = option_env!("S2AND_RUST_PYTHONHOME") {
            std::env::set_var("PYTHONHOME", python_home);
        }
        pyo3::prepare_freethreaded_python();
    }

    fn py_err_message(err: PyErr) -> String {
        prepare_python_for_test();
        Python::with_gil(|py| {
            err.value(py)
                .str()
                .expect("PyErr value should stringify")
                .to_str()
                .expect("test error messages are ASCII")
                .to_string()
        })
    }

    #[test]
    fn arrow_schema_contract_json_matches_rust_column_contract() {
        let payload: serde_json::Value =
            serde_json::from_str(include_str!("../../s2and/arrow_schema_contract.json"))
                .expect("schema contract JSON should parse");
        assert_eq!(
            payload
                .get("schema_version")
                .and_then(serde_json::Value::as_str),
            Some("s2and_arrow_schema_contract_v1")
        );
        let tables = payload
            .get("tables")
            .and_then(serde_json::Value::as_object)
            .expect("schema contract should contain a tables object");
        let mut observed = Vec::new();
        for (table_name, columns) in tables {
            for column in columns
                .as_array()
                .expect("schema contract table columns should be arrays")
            {
                observed.push((
                    table_name.as_str(),
                    column
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .expect("schema contract column should contain name"),
                    column
                        .get("datatype")
                        .and_then(serde_json::Value::as_str)
                        .expect("schema contract column should contain datatype"),
                    column
                        .get("required")
                        .and_then(serde_json::Value::as_bool)
                        .expect("schema contract column should contain required"),
                ));
            }
        }
        observed.sort_unstable();
        let mut expected = ARROW_SCHEMA_CONTRACT_COLUMNS.to_vec();
        expected.sort_unstable();
        assert_eq!(observed, expected);
    }

    #[test]
    fn validate_retrieval_top_k_rejects_uint16_rank_overflow() {
        let error = validate_retrieval_rank_top_k((u16::MAX as usize) + 1).unwrap_err();
        assert!(py_err_message(error).contains("retrieval_ranks are stored as uint16"));
    }

    #[test]
    fn retrieval_rank_from_zero_based_offset_rejects_uint16_overflow() {
        assert_eq!(
            retrieval_rank_from_zero_based_offset(0, "test").expect("first rank fits"),
            1
        );
        let error = retrieval_rank_from_zero_based_offset(u16::MAX as usize, "test").unwrap_err();
        assert!(error.contains("retrieval_ranks are stored as uint16"));
    }

    #[test]
    fn feature_index_resolution_preserves_order_and_duplicates() {
        assert_eq!(
            resolve_feature_indices("selected_indices", Some(vec![2, 2, 3]), 5)
                .expect("indices are in range"),
            vec![2, 2, 3]
        );
        assert_eq!(
            resolve_feature_indices("selected_indices", None, 3).expect("default indices"),
            vec![0, 1, 2]
        );

        let result = resolve_feature_indices("selected_indices", Some(vec![0, 3]), 3);
        assert!(result.is_err());
        let message = py_err_message(result.err().expect("error was asserted"));
        assert!(message.contains("selected_indices contains out-of-range index 3 for 3 columns"));
    }

    #[test]
    fn aggregate_matrix_positions_preserve_aggregate_order_and_duplicate_mapping() {
        assert_eq!(
            matrix_positions_for_feature_indices(&[2, 2, 3], &[2, 3, 2])
                .expect("aggregate features are present"),
            vec![0, 2, 0]
        );

        let result = resolve_matrix_aggregate_indices(Some(vec![2, 2]), Some(vec![3]), 4);
        assert!(result.is_err());
        let message = py_err_message(result.err().expect("error was asserted"));
        assert!(message.contains("aggregate index 3 is not present in matrix_indices"));
    }

    #[test]
    fn sorted_subblock_merge_candidates_allows_nan_scores() {
        let mut output = OrderedSubblocks::default();
        output.insert("alice".to_string(), vec!["s1".to_string()]);
        output.insert("bob".to_string(), vec!["s2".to_string()]);
        let mut counts = HashMap::<String, HashMap<String, f64>>::new();
        counts.insert(
            "alice".to_string(),
            HashMap::from([("bob".to_string(), f64::NAN)]),
        );

        let result = sorted_subblock_merge_candidates(&output, 3, &counts);

        assert!(result.is_ok());
        let candidates = result.expect("NaN scores should sort without raising");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, ("alice".to_string(), "bob".to_string()));
        assert!(candidates[0].1.is_nan());
    }

    #[test]
    fn year_signal_value_rejects_out_of_range_and_reserved_sentinel() {
        assert_eq!(
            year_signal_value(None, "query year").expect("missing year"),
            (i32::MIN, 1)
        );
        assert_eq!(
            year_signal_value(Some(2024), "query year").expect("valid year"),
            (2024, 0)
        );
        assert!(year_signal_value(Some(i64::from(i32::MAX) + 1), "query year").is_err());
        assert!(year_signal_value(Some(i64::from(i32::MIN)), "query year").is_err());
    }

    #[test]
    fn sorted_subblock_merge_candidates_keeps_exact_maximum_size_pair() {
        let mut output = OrderedSubblocks::default();
        output.insert(
            "alex|middle=a".to_string(),
            vec!["s1".to_string(), "s2".to_string()],
        );
        output.insert(
            "alex|middle=b".to_string(),
            vec!["s3".to_string(), "s4".to_string(), "s5".to_string()],
        );

        let candidates = sorted_subblock_merge_candidates(&output, 5, &HashMap::new())
            .expect("merge candidates");

        assert_eq!(
            candidates,
            vec![(
                ("alex|middle=a".to_string(), "alex|middle=b".to_string()),
                1e10
            )]
        );
    }

    #[test]
    fn orcid_subblocking_skips_oversized_connected_component_without_partial_merge() {
        let mut output = OrderedSubblocks::default();
        output.insert("a".to_string(), vec!["s1".to_string()]);
        output.insert("b".to_string(), vec!["s2".to_string(), "s3".to_string()]);
        output.insert("c".to_string(), vec!["s4".to_string()]);
        let rows = HashMap::from([
            (
                "s1".to_string(),
                SubblockingSignatureRow {
                    signature_id: "s1".to_string(),
                    paper_id: "p1".to_string(),
                    first: "aa".to_string(),
                    middle: String::new(),
                    affiliations: Vec::new(),
                    orcid: Some("0000-0000-0000-0001".to_string()),
                    position: None,
                },
            ),
            (
                "s2".to_string(),
                SubblockingSignatureRow {
                    signature_id: "s2".to_string(),
                    paper_id: "p2".to_string(),
                    first: "bb".to_string(),
                    middle: String::new(),
                    affiliations: Vec::new(),
                    orcid: Some("0000-0000-0000-0001".to_string()),
                    position: None,
                },
            ),
            (
                "s3".to_string(),
                SubblockingSignatureRow {
                    signature_id: "s3".to_string(),
                    paper_id: "p3".to_string(),
                    first: "bb".to_string(),
                    middle: String::new(),
                    affiliations: Vec::new(),
                    orcid: Some("0000-0000-0000-0002".to_string()),
                    position: None,
                },
            ),
            (
                "s4".to_string(),
                SubblockingSignatureRow {
                    signature_id: "s4".to_string(),
                    paper_id: "p4".to_string(),
                    first: "cc".to_string(),
                    middle: String::new(),
                    affiliations: Vec::new(),
                    orcid: Some("0000-0000-0000-0002".to_string()),
                    position: None,
                },
            ),
        ]);
        let mut telemetry = SubblockingTelemetry::default();

        apply_orcid_subblocking(&mut output, &rows, 3, &mut telemetry);

        assert_eq!(
            output.to_hashmap(),
            HashMap::from([
                ("a".to_string(), vec!["s1".to_string()]),
                ("b".to_string(), vec!["s2".to_string(), "s3".to_string()]),
                ("c".to_string(), vec!["s4".to_string()]),
            ])
        );
        assert_eq!(telemetry.orcid_merge_skipped_due_to_capacity_count, 2);
        assert_eq!(
            telemetry.orcid_merge_skipped_due_to_capacity_signature_count,
            4
        );
    }

    #[test]
    fn i64_author_position_distance_handles_extreme_values() {
        assert!(i64::MIN.abs_diff(0) > 10);
        assert_eq!(10_i64.abs_diff(0), 10);
        assert_eq!(position_diff(i64::MIN, 0), 50.0);
        assert_eq!(position_diff(i64::MIN, i64::MAX), 50.0);
    }

    #[test]
    fn subblocking_arrow_rows_normalize_first_and_middle_names() {
        let mut rows = vec![
            SubblockingSignatureRow {
                signature_id: "s1".to_string(),
                paper_id: "p1".to_string(),
                first: "Alice".to_string(),
                middle: String::new(),
                affiliations: Vec::new(),
                orcid: None,
                position: None,
            },
            SubblockingSignatureRow {
                signature_id: "s2".to_string(),
                paper_id: "p2".to_string(),
                first: "alice".to_string(),
                middle: String::new(),
                affiliations: Vec::new(),
                orcid: None,
                position: None,
            },
            SubblockingSignatureRow {
                signature_id: "s3".to_string(),
                paper_id: "p3".to_string(),
                first: "Qi-Xin".to_string(),
                middle: "A.".to_string(),
                affiliations: Vec::new(),
                orcid: None,
                position: None,
            },
            SubblockingSignatureRow {
                signature_id: "s4".to_string(),
                paper_id: "p4".to_string(),
                first: "Arif\u{2010}ullah".to_string(),
                middle: String::new(),
                affiliations: Vec::new(),
                orcid: None,
                position: None,
            },
        ];
        let prefixes = HashSet::new();
        let unidecode_char_map = HashMap::from([('\u{2010}', "-".to_string())]);

        normalize_subblocking_signature_rows(&mut rows, &prefixes, &unidecode_char_map);

        assert_eq!(rows[0].first, "alice");
        assert_eq!(rows[1].first, "alice");
        assert_eq!(rows[2].first, "qi xin");
        assert_eq!(rows[2].middle, "a");
        assert_eq!(rows[3].first, "arif");
        assert_eq!(rows[3].middle, "ullah");
    }

    #[test]
    fn normalize_text_compat_drops_digits_like_python_reference() {
        let unidecode_char_map = HashMap::new();

        assert_eq!(
            normalize_text_compat_from_map("A1 B-2", false, &unidecode_char_map),
            "a b"
        );
        assert_eq!(
            normalize_text_compat_from_map("O'Neil2", true, &unidecode_char_map),
            "oneil"
        );
    }

    #[test]
    fn normalize_text_compat_missing_mapping_does_not_panic() {
        let unidecode_char_map = HashMap::new();
        assert_eq!(
            normalize_text_compat_from_map("\u{00C9}lodie", false, &unidecode_char_map),
            "elodie",
        );
    }

    #[test]
    fn normalize_text_compat_uses_native_unidecode() {
        assert_eq!(
            text_compat::normalize_text_compat_native("te'\u{6F22}\u{5B57}xt", false),
            "te han zi xt",
        );
        assert_eq!(
            text_compat::normalize_text_compat_native("O\u{2019}Neil", false),
            "o neil",
        );
        assert_eq!(
            text_compat::normalize_text_compat_native("O\u{2019}Neil", true),
            "oneil",
        );
    }

    #[test]
    fn stage_papers_normalize_title_and_authors_without_full_preprocess() {
        let input = StagePaperInput {
            paper_id: "p1".to_string(),
            raw_title: "Some Title".to_string(),
            raw_venue: "My Venue".to_string(),
            raw_journal: "My Journal".to_string(),
            raw_authors: vec![(0, "ALICE-1".to_string()), (1, "Bob O'Neil".to_string())],
            year: Some(2024),
            has_abstract: false,
            predicted_language: None,
            is_reliable: false,
        };

        let papers = preprocess_stage_papers(
            &[input],
            false,
            &HashMap::new(),
            &HashSet::new(),
            &HashSet::new(),
        );

        let paper = &papers[0].1;
        assert_eq!(
            paper.authors,
            vec![(0, "alice".to_string()), (1, "bob o neil".to_string())]
        );
        assert!(paper.title_words.is_some());
        assert!(paper.title_chars.is_none());
        assert!(paper.venue_ngrams.is_none());
        assert!(paper.journal_ngrams.is_none());
    }

    #[test]
    fn raw_arrow_summary_rejects_out_of_range_year_before_mean_sum() {
        let features = HashMap::from([(
            "s1".to_string(),
            raw_arrow_feature_for_test("alice", Some(i64::MAX)),
        )]);

        let err = match build_raw_arrow_summary("c1", &["s1".to_string()], &features, 0) {
            Ok(_) => panic!("out-of-range year should be rejected"),
            Err(err) => err,
        };

        assert!(err.contains("raw Arrow summary year is outside the supported i32 range"));
    }

    #[test]
    fn raw_arrow_summary_year_mean_uses_wide_sum() {
        assert_eq!(
            raw_arrow_year_mean(&[i64::MAX, i64::MAX]),
            Some(i64::MAX as f64)
        );
    }

    #[test]
    fn row_named_signature_count_sums_integer_counts_without_f32_rounding() {
        let first_name_counts = vec![("alice".to_string(), 16_777_217), ("bob".to_string(), 1)];

        assert_eq!(
            row_named_signature_count(&first_name_counts).expect("valid counts"),
            16_777_218
        );
    }

    #[test]
    fn residual_summary_signal_cache_uses_tuple_key_for_nul_collisions() {
        let mut cache = HashMap::<(String, String), RawArrowSummarySignalData>::new();
        let features = HashMap::from([
            (
                "m1".to_string(),
                raw_arrow_feature_for_test("alice", Some(2020)),
            ),
            (
                "m2".to_string(),
                raw_arrow_feature_for_test("bob", Some(2021)),
            ),
        ]);
        let signatures = HashMap::from([
            ("m1".to_string(), raw_arrow_signature_for_test("p1")),
            ("m2".to_string(), raw_arrow_signature_for_test("p2")),
        ]);
        let paper_authors = HashMap::<String, Vec<(i64, String)>>::new();
        let unidecode_char_map = HashMap::new();

        let first_member_ids = {
            let signals = raw_arrow_summary_signals_for_members_cached(
                &mut cache,
                "a\0b",
                "c",
                &["m1".to_string()],
                &features,
                &signatures,
                &paper_authors,
                &unidecode_char_map,
            )
            .expect("first cache lookup");
            signals.member_signature_ids.clone()
        };
        let second_member_ids = {
            let signals = raw_arrow_summary_signals_for_members_cached(
                &mut cache,
                "a",
                "b\0c",
                &["m2".to_string()],
                &features,
                &signatures,
                &paper_authors,
                &unidecode_char_map,
            )
            .expect("second cache lookup");
            signals.member_signature_ids.clone()
        };

        assert_eq!(cache.len(), 2);
        assert_eq!(first_member_ids, vec!["m1".to_string()]);
        assert_eq!(second_member_ids, vec!["m2".to_string()]);
    }

    #[test]
    fn subblock_token_fallback_matches_python_case_preserving_parse() {
        assert_eq!(
            subblock_tokens_from_key("Ali|3,bob|2,a|1"),
            vec!["Ali".to_string(), "bob".to_string()]
        );
    }
}

#[pymodule]
fn _s2and_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("RETRIEVAL_FEATURE_ORDER", RETRIEVAL_FEATURE_ORDER.to_vec())?;
    m.add(
        "DEFAULT_HYBRID_CENTROID_POLICY_NAME",
        DEFAULT_HYBRID_CENTROID_POLICY_NAME,
    )?;
    m.add(
        "DEFAULT_HYBRID_CENTROID_WEIGHTS",
        DEFAULT_HYBRID_CENTROID_WEIGHTS.to_vec(),
    )?;
    m.add(
        "DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS",
        DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS.to_vec(),
    )?;
    m.add(
        "DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS",
        DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS.to_vec(),
    )?;
    m.add(
        "RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE",
        RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE,
    )?;
    m.add(
        "RETRIEVAL_YEAR_SCORE_DECAY_YEARS",
        RETRIEVAL_YEAR_SCORE_DECAY_YEARS,
    )?;
    m.add(
        "RETRIEVAL_YEAR_SCORE_RANGE_GAP",
        RETRIEVAL_YEAR_SCORE_RANGE_GAP,
    )?;
    m.add(
        "RETRIEVAL_YEAR_SCORE_RANGE_PENALTY",
        RETRIEVAL_YEAR_SCORE_RANGE_PENALTY,
    )?;
    m.add(
        "RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP",
        RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP,
    )?;
    m.add(
        "INCREMENTAL_LINKING_PAIR_PLAN_ROW_SIGNALS",
        INCREMENTAL_LINKING_PAIR_PLAN_ROW_SIGNALS.to_vec(),
    )?;
    m.add(
        "RAW_ARROW_QUERY_SIGNATURE_PLANNER_METHODS",
        RAW_ARROW_QUERY_SIGNATURE_PLANNER_METHODS.to_vec(),
    )?;
    m.add_function(wrap_pyfunction!(get_build_info, m)?)?;
    m.add_function(wrap_pyfunction!(raw_arrow_labeled_candidate_plan, m)?)?;
    promoted_linker::add_to_module(m)?;
    lightgbm_booster::add_to_module(m)?;
    m.add_function(wrap_pyfunction!(
        make_subblocks_with_telemetry_arrow_native_graph,
        m
    )?)?;
    m.add_class::<RustFeaturizer>()?;
    m.add_class::<RustHybridCentroidRetriever>()?;
    m.add_class::<RawBlockQueryCandidatePlanner>()?;
    Ok(())
}
