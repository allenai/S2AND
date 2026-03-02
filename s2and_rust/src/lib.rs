use cld2::{detect_language as cld2_detect_language, Format as Cld2Format};
use fasttext::FastText;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule, PyTuple};
use pyo3::Bound;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

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
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[derive(Clone, Serialize, Deserialize)]
struct NameCountsData {
    first: f64,
    first_last: f64,
    last: f64,
    last_first_initial: f64,
}

#[derive(Default)]
struct RawNameCountMaps {
    first: HashMap<String, f64>,
    last: HashMap<String, f64>,
    first_last: HashMap<String, f64>,
    last_first_initial: HashMap<String, f64>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
enum ClusterId {
    Int(i64),
    Str(String),
}

type PaperId = String;

#[derive(Clone, Serialize, Deserialize)]
struct SignatureData {
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
    adv_name: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
struct PaperData {
    venue_ngrams: Option<CounterData>,
    title_words: Option<CounterData>,
    title_chars: Option<CounterData>,
    ref_authors: Option<CounterData>,
    ref_titles: Option<CounterData>,
    ref_venues: Option<CounterData>,
    ref_blocks: Option<CounterData>,
    ref_details_present: bool,
    references: HashSet<PaperId>,
    year: Option<i64>,
    has_abstract: bool,
    predicted_language: Option<String>,
    is_reliable: bool,
    journal_ngrams: Option<CounterData>,
    specter: Option<Vec<f32>>,
    #[serde(default)]
    specter_norm: Option<f64>,
}

#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
struct RustFeaturizer {
    signatures: HashMap<String, SignatureData>,
    #[serde(default)]
    signature_ids: Vec<String>,
    papers: HashMap<PaperId, PaperData>,
    name_tuples: HashMap<String, HashSet<String>>,
    cluster_seeds_disallow: HashSet<(String, String)>,
    cluster_seeds_require: HashMap<String, ClusterId>,
    compute_reference_features: bool,
    cluster_seed_require_value: f64,
    cluster_seed_disallow_value: f64,
    #[serde(skip)]
    cached_signature_id_order: OnceLock<Vec<String>>,
    #[serde(skip)]
    cluster_seeds_disallow_index: OnceLock<HashMap<String, HashSet<String>>>,
}

#[derive(Clone, Default)]
struct JsonIngestTelemetry {
    json_parse_seconds: f64,
    paper_preprocess_seconds: f64,
    reference_counter_seconds: f64,
    signature_preprocess_seconds: f64,
    cluster_seed_seconds: f64,
}

static LAST_JSON_INGEST_TELEMETRY: OnceLock<Mutex<Option<JsonIngestTelemetry>>> = OnceLock::new();

fn json_ingest_telemetry_slot() -> &'static Mutex<Option<JsonIngestTelemetry>> {
    LAST_JSON_INGEST_TELEMETRY.get_or_init(|| Mutex::new(None))
}

fn set_last_json_ingest_telemetry(telemetry: JsonIngestTelemetry) {
    if let Ok(mut slot) = json_ingest_telemetry_slot().lock() {
        *slot = Some(telemetry);
    }
}

fn reset_last_json_ingest_telemetry_internal() {
    if let Ok(mut slot) = json_ingest_telemetry_slot().lock() {
        *slot = None;
    }
}

static RAYON_POOL_CACHE: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();

fn rayon_pool_cache() -> &'static Mutex<HashMap<usize, Arc<rayon::ThreadPool>>> {
    RAYON_POOL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cached_rayon_pool(thread_count: usize) -> Option<Arc<rayon::ThreadPool>> {
    if thread_count == 0 {
        return None;
    }
    if let Ok(cache) = rayon_pool_cache().lock() {
        if let Some(pool) = cache.get(&thread_count) {
            return Some(Arc::clone(pool));
        }
    }

    let built_pool = ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .ok()?;
    let built_pool = Arc::new(built_pool);
    if let Ok(mut cache) = rayon_pool_cache().lock() {
        let pooled = cache
            .entry(thread_count)
            .or_insert_with(|| Arc::clone(&built_pool));
        return Some(Arc::clone(pooled));
    }
    Some(built_pool)
}

fn install_with_optional_rayon_pool<T, F>(num_threads: Option<usize>, compute: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    if let Some(thread_count) = num_threads {
        let threads = thread_count.max(1);
        if let Some(pool) = cached_rayon_pool(threads) {
            return pool.install(compute);
        }
    }
    compute()
}

fn upper_triangle_total_pairs(block_size: usize) -> usize {
    block_size.saturating_mul(block_size.saturating_sub(1)) / 2
}

fn upper_triangle_pairs_for_range(
    block_size: usize,
    start_offset: usize,
    max_pairs: Option<usize>,
) -> PyResult<Vec<(usize, usize)>> {
    let total_pairs = upper_triangle_total_pairs(block_size);
    if start_offset > total_pairs {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "start_offset out of range: start_offset={} total_pairs={}",
            start_offset, total_pairs
        )));
    }
    let remaining = total_pairs.saturating_sub(start_offset);
    let pair_count = max_pairs.unwrap_or(remaining).min(remaining);
    if pair_count == 0 {
        return Ok(Vec::new());
    }

    let mut local_i = 0usize;
    let mut offset_in_row = start_offset;
    while local_i + 1 < block_size {
        let row_pairs = block_size - local_i - 1;
        if offset_in_row < row_pairs {
            break;
        }
        offset_in_row -= row_pairs;
        local_i += 1;
    }
    if local_i + 1 >= block_size {
        return Ok(Vec::new());
    }
    let mut local_j = local_i + 1 + offset_in_row;
    let mut pairs = Vec::with_capacity(pair_count);
    for _ in 0..pair_count {
        pairs.push((local_i, local_j));
        local_j += 1;
        if local_j >= block_size {
            local_i += 1;
            local_j = local_i + 1;
        }
    }
    Ok(pairs)
}

#[pyfunction]
fn get_last_json_ingest_telemetry(py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
    let slot = json_ingest_telemetry_slot().lock().map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err("json ingest telemetry lock poisoned")
    })?;
    let Some(telemetry) = slot.as_ref() else {
        return Ok(None);
    };

    let stage_seconds = PyDict::new(py);
    stage_seconds.set_item("json_parse_seconds", telemetry.json_parse_seconds)?;
    stage_seconds.set_item(
        "paper_preprocess_seconds",
        telemetry.paper_preprocess_seconds,
    )?;
    stage_seconds.set_item(
        "reference_counter_seconds",
        telemetry.reference_counter_seconds,
    )?;
    stage_seconds.set_item(
        "signature_preprocess_seconds",
        telemetry.signature_preprocess_seconds,
    )?;
    stage_seconds.set_item("cluster_seed_seconds", telemetry.cluster_seed_seconds)?;

    let telemetry_dict = PyDict::new(py);
    telemetry_dict.set_item("stage_seconds", stage_seconds)?;
    Ok(Some(telemetry_dict.unbind()))
}

#[pyfunction]
fn reset_last_json_ingest_telemetry() -> PyResult<()> {
    reset_last_json_ingest_telemetry_internal();
    Ok(())
}

fn extract_counter(obj: &Bound<'_, PyAny>) -> PyResult<Option<CounterData>> {
    if obj.is_none() {
        return Ok(None);
    }
    let dict = obj.downcast::<PyDict>()?;
    if dict.len() == 0 {
        return Ok(None);
    }
    let mut entries: Vec<(u64, f32)> = Vec::with_capacity(dict.len());
    let mut sum = 0.0f32;
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let val: f64 = v.extract()?;
        let val32 = val as f32;
        sum += val32;
        entries.push((fnv64(key.as_bytes()), val32));
    }
    entries.sort_unstable_by_key(|e| e.0);
    Ok(Some(CounterData { entries, sum }))
}

fn extract_reference_details_counters(
    py: Python<'_>,
    ref_details_obj: &Bound<'_, PyAny>,
) -> PyResult<(
    Option<CounterData>,
    Option<CounterData>,
    Option<CounterData>,
    Option<CounterData>,
)> {
    let tuple = ref_details_obj.extract::<(PyObject, PyObject, PyObject, PyObject)>()?;
    Ok((
        extract_counter(&tuple.0.bind(py))?,
        extract_counter(&tuple.1.bind(py))?,
        extract_counter(&tuple.2.bind(py))?,
        extract_counter(&tuple.3.bind(py))?,
    ))
}

fn extract_optional_string_set(obj: &Bound<'_, PyAny>) -> PyResult<Option<HashSet<String>>> {
    if obj.is_none() {
        return Ok(None);
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        let v: String = item?.extract()?;
        out.insert(v);
    }
    if out.is_empty() {
        Ok(None)
    } else {
        Ok(Some(out))
    }
}

fn canonical_signature_pair_ref<'a>(a: &'a str, b: &'a str) -> (&'a str, &'a str) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn canonical_signature_pair_owned(a: String, b: String) -> (String, String) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn canonical_signature_pair_cloned(a: &str, b: &str) -> (String, String) {
    let (left, right) = canonical_signature_pair_ref(a, b);
    (left.to_string(), right.to_string())
}

fn extract_pair_set(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<(String, String)>> {
    if obj.is_none() {
        return Ok(HashSet::new());
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        let tuple = item?;
        let (a, b): (String, String) = tuple.extract()?;
        out.insert(canonical_signature_pair_owned(a, b));
    }
    Ok(out)
}

fn extract_name_tuples_map(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, HashSet<String>>> {
    if obj.is_none() {
        return Ok(HashMap::new());
    }
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for item in PyIterator::from_object(obj)? {
        let tuple = item?;
        let (a, b): (String, String) = tuple.extract()?;
        out.entry(a).or_insert_with(HashSet::new).insert(b);
    }
    Ok(out)
}

fn extract_cluster_seeds_require(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, ClusterId>> {
    if obj.is_none() {
        return Ok(HashMap::new());
    }
    let dict = obj.downcast::<PyDict>()?;
    let mut out = HashMap::with_capacity(dict.len());
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let val: ClusterId = if let Ok(i) = v.extract::<i64>() {
            ClusterId::Int(i)
        } else if let Ok(s) = v.extract::<String>() {
            ClusterId::Str(s)
        } else if let Ok(u) = v.extract::<u64>() {
            ClusterId::Int(u as i64)
        } else {
            ClusterId::Str(v.str()?.to_string())
        };
        out.insert(key, val);
    }
    Ok(out)
}

fn extract_id_string(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(i.to_string());
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(u.to_string());
    }
    Ok(obj.str()?.to_string())
}

fn extract_set_id_string(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<PaperId>> {
    if obj.is_none() {
        return Ok(HashSet::new());
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        let v = extract_id_string(&item?)?;
        out.insert(v);
    }
    Ok(out)
}

fn extract_string_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in PyIterator::from_object(obj)? {
        out.push(item?.extract()?);
    }
    Ok(out)
}

fn get_namedtuple_item_or_attr<'py>(
    obj: &Bound<'py, PyAny>,
    allow_tuple_fastpath: bool,
    tuple_index: usize,
    attr_name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if allow_tuple_fastpath {
        return obj.get_item(tuple_index).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "NamedTuple fast-path index out of range or object not indexable: index={} attr={}",
                tuple_index, attr_name
            ))
        });
    }
    obj.getattr(attr_name)
}

fn validate_namedtuple_fastpath_contract(
    sample_obj: &Bound<'_, PyAny>,
    required_fields: &[(usize, &str)],
    tuple_label: &str,
) -> PyResult<bool> {
    let fields_obj = match sample_obj.getattr("_fields") {
        Ok(fields_obj) => fields_obj,
        Err(_) => return Ok(false),
    };

    let field_names: Vec<String> = fields_obj.extract().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{} fast-path expected _fields to be a tuple/list of field names",
            tuple_label
        ))
    })?;

    let max_required_index = required_fields
        .iter()
        .map(|(index, _)| *index)
        .max()
        .unwrap_or(0);
    if sample_obj.get_item(max_required_index).is_err() || field_names.len() <= max_required_index {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} fast-path contract mismatch: object is not indexable to required max index {} or _fields len={}",
            tuple_label,
            max_required_index,
            field_names.len()
        )));
    }

    for (field_index, expected_name) in required_fields {
        let actual_name = field_names
            .get(*field_index)
            .map(|s| s.as_str())
            .unwrap_or("<missing>");
        if actual_name != *expected_name {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} fast-path contract mismatch at index {}: expected '{}' got '{}'",
                tuple_label, field_index, expected_name, actual_name
            )));
        }
    }

    Ok(true)
}

fn validate_dict_namedtuple_fastpath_contract(
    rows: &Bound<'_, PyDict>,
    required_fields: &[(usize, &str)],
    tuple_label: &str,
) -> PyResult<bool> {
    if let Some((_, sample_obj)) = rows.iter().next() {
        return validate_namedtuple_fastpath_contract(&sample_obj, required_fields, tuple_label);
    }
    Ok(false)
}

fn extract_paper_authors_with_positions(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(i64, String)>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in PyIterator::from_object(obj)? {
        let author_obj = item?;
        let (position, author_name) = if let Ok(author_tuple) = author_obj.downcast::<PyTuple>() {
            if author_tuple.len() >= 2 {
                let author_name: String = author_tuple.get_item(0)?.extract()?;
                let position: i64 = author_tuple.get_item(1)?.extract()?;
                (position, author_name)
            } else {
                let position: i64 = author_obj.getattr("position")?.extract()?;
                let author_name: String = author_obj.getattr("author_name")?.extract()?;
                (position, author_name)
            }
        } else {
            let position: i64 = author_obj.getattr("position")?.extract()?;
            let author_name: String = author_obj.getattr("author_name")?.extract()?;
            (position, author_name)
        };
        out.push((position, author_name));
    }
    Ok(out)
}

fn extract_required_string_set(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<String>> {
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        out.insert(item?.extract()?);
    }
    Ok(out)
}

fn extract_affiliation_stopwords(py: Python<'_>) -> PyResult<HashSet<String>> {
    let text_module = py.import("s2and.text")?;
    let stopwords_obj = text_module.getattr("AFFILIATIONS_STOP_WORDS")?;
    extract_required_string_set(&stopwords_obj)
}

fn prefilter_affiliation_text(affiliations: &[String], stopwords: &HashSet<String>) -> String {
    if affiliations.is_empty() {
        return String::new();
    }
    let mut tokens: Vec<&str> = Vec::new();
    for word in affiliations
        .iter()
        .flat_map(|affiliation| affiliation.split_whitespace())
    {
        if !stopwords.contains(word) && py_len(word) > 1 {
            tokens.push(word);
        }
    }
    tokens.join(" ")
}

fn ensure_unidecode_for_text(
    unidecode_fn: &Bound<'_, PyAny>,
    text: &str,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    for ch in text.chars() {
        if ch.is_ascii() || unidecode_char_map.contains_key(&ch) {
            continue;
        }
        let mapped: String = unidecode_fn.call1((ch.to_string(),))?.extract()?;
        unidecode_char_map.insert(ch, mapped);
    }
    Ok(())
}

fn normalize_text_compat_from_map(
    text: &str,
    special_case_apostrophes: bool,
    unidecode_char_map: &HashMap<char, String>,
) -> String {
    if text.is_empty() {
        return String::new();
    }

    let mut transliterated = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            transliterated.push(ch.to_ascii_lowercase());
            continue;
        }
        if let Some(mapped) = unidecode_char_map.get(&ch) {
            for mapped_ch in mapped.chars() {
                for lowered in mapped_ch.to_lowercase() {
                    transliterated.push(lowered);
                }
            }
            continue;
        }
        for lowered in ch.to_lowercase() {
            transliterated.push(lowered);
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

fn split_first_middle_hyphen_aware_compat(
    first_raw: &str,
    middle_raw: &str,
    name_prefixes: &HashSet<String>,
    unidecode_char_map: &HashMap<char, String>,
) -> (String, String) {
    let has_dash_in_first = first_raw.contains('-');
    let first_noapos = normalize_text_compat_from_map(first_raw, true, unidecode_char_map);
    let middle_norm = normalize_text_compat_from_map(middle_raw, false, unidecode_char_map);

    let mut f_parts: Vec<String> = first_noapos
        .split_whitespace()
        .map(|token| token.to_string())
        .collect();
    let m_parts: Vec<String> = middle_norm
        .split_whitespace()
        .map(|token| token.to_string())
        .collect();
    if let Some(prefix) = f_parts.first() {
        if name_prefixes.contains(prefix) {
            f_parts.remove(0);
        }
    }

    if f_parts.is_empty() {
        return (String::new(), m_parts.join(" "));
    }
    if has_dash_in_first {
        return (f_parts.join(" "), m_parts.join(" "));
    }
    let first = f_parts[0].clone();
    let middle = f_parts[1..]
        .iter()
        .chain(m_parts.iter())
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    (first, middle)
}

fn compute_block_compat(name: &str) -> String {
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

fn env_flag_true(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            let lower = value.to_ascii_lowercase();
            lower == "1" || lower == "true" || lower == "yes"
        })
        .unwrap_or(false)
}

fn parse_fasttext_label(label: &str) -> String {
    label.rsplit("__").next().unwrap_or(label).to_string()
}

fn resolve_fasttext_model_path(py: Python<'_>) -> Option<String> {
    let consts = py.import("s2and.consts").ok()?;
    let fasttext_path: String = consts.getattr("FASTTEXT_PATH").ok()?.extract().ok()?;
    let file_cache = py.import("s2and.file_cache").ok()?;
    let cached_path = file_cache.getattr("cached_path").ok()?;
    cached_path.call1((fasttext_path,)).ok()?.extract().ok()
}

fn emit_runtime_warning(py: Python<'_>, message: &str) {
    if let Ok(warnings) = py.import("warnings") {
        let _ = warnings.call_method1("warn", (message.to_string(),));
    }
}

struct LanguageDetectorCompat {
    fasttext: Option<FastText>,
}

impl LanguageDetectorCompat {
    fn new(py: Python<'_>) -> Self {
        if env_flag_true("S2AND_SKIP_FASTTEXT") {
            return Self { fasttext: None };
        }
        let fasttext = resolve_fasttext_model_path(py).and_then(|model_path| {
            let mut model = FastText::new();
            match model.load_model(&model_path) {
                Ok(()) => Some(model),
                Err(err) => {
                    let warning = format!(
                        "s2and_rust: failed to load fastText model at '{}' ({}); falling back to CLD2-only language detection.",
                        model_path, err
                    );
                    emit_runtime_warning(py, &warning);
                    eprintln!("{warning}");
                    None
                }
            }
        });
        Self { fasttext }
    }

    fn detect(&self, text: &str) -> (bool, bool, String) {
        if text.split_whitespace().count() <= 1 {
            return (false, false, "un".to_string());
        }

        let mut alpha_count = 0usize;
        let mut uppercase_count = 0usize;
        for ch in text.chars() {
            if ch.is_alphabetic() {
                alpha_count += 1;
                if ch.is_uppercase() {
                    uppercase_count += 1;
                }
            }
        }
        if alpha_count == 0 {
            return (false, false, "un".to_string());
        }

        let predicted_language_ft = if let Some(fasttext_model) = &self.fasttext {
            let uppercase_ratio = uppercase_count as f64 / alpha_count as f64;
            let mut fasttext_input = text.replace('\n', " ");
            if uppercase_ratio > 0.9 {
                fasttext_input = fasttext_input.to_lowercase();
            }
            match fasttext_model.predict(&fasttext_input, 1, 0.0) {
                Ok(predictions) => predictions
                    .first()
                    .map(|prediction| parse_fasttext_label(&prediction.label))
                    .unwrap_or_else(|| "un_ft".to_string()),
                Err(_) => "un_ft".to_string(),
            }
        } else {
            "un_ft".to_string()
        };

        let mut predicted_language_2 = match cld2_detect_language(text, Cld2Format::Text).0 {
            Some(lang) => lang.0.to_string(),
            None => "un_2".to_string(),
        };
        if predicted_language_2 == "un" {
            predicted_language_2 = "un_2".to_string();
        }

        let (predicted_language, is_reliable) =
            if predicted_language_ft == "un_ft" && predicted_language_2 == "un_2" {
                ("un".to_string(), false)
            } else if predicted_language_ft == "un_ft" {
                (predicted_language_2, true)
            } else if predicted_language_2 == "un_2" {
                (predicted_language_ft, true)
            } else if predicted_language_2 != predicted_language_ft {
                ("un".to_string(), false)
            } else {
                (predicted_language_2, true)
            };

        let is_english = predicted_language == "en";
        (is_reliable, is_english, predicted_language)
    }
}

fn extract_orcid_from_source_id(source_id: &str) -> Option<String> {
    let chars: Vec<char> = source_id.chars().collect();
    if chars.len() < 16 {
        return None;
    }
    for start in 0..chars.len() {
        let mut idx = start;
        let mut compact = String::with_capacity(16);
        let mut valid = true;

        for (group_idx, group_len) in [4usize, 4, 4, 3].iter().enumerate() {
            for _ in 0..*group_len {
                if idx >= chars.len() || !chars[idx].is_ascii_digit() {
                    valid = false;
                    break;
                }
                compact.push(chars[idx]);
                idx += 1;
            }
            if !valid {
                break;
            }
            if group_idx < 3 && idx < chars.len() && chars[idx] == '-' {
                idx += 1;
            }
        }
        if !valid {
            continue;
        }
        if idx >= chars.len() {
            continue;
        }
        let last = chars[idx];
        if !(last.is_ascii_digit() || last == 'X') {
            continue;
        }
        compact.push(last);
        return Some(compact);
    }
    None
}

fn counter_data_from_usize_map(counter_map: HashMap<String, usize>) -> Option<CounterData> {
    if counter_map.is_empty() {
        return None;
    }
    let mut entries: Vec<(u64, f32)> = counter_map
        .iter()
        .map(|(k, v)| (fnv64(k.as_bytes()), *v as f32))
        .collect();
    entries.sort_unstable_by_key(|e| e.0);
    let sum: f32 = entries.iter().map(|e| e.1).sum();
    Some(CounterData { entries, sum })
}

fn extract_specter_for_paper_id(
    spec_dict: &Bound<'_, PyDict>,
    paper_id: &str,
) -> PyResult<Option<Vec<f32>>> {
    if let Ok(Some(val)) = spec_dict.get_item(paper_id) {
        return extract_specter_vec(&val);
    }
    if let Ok(i) = paper_id.parse::<i64>() {
        if let Ok(Some(val)) = spec_dict.get_item(i) {
            return extract_specter_vec(&val);
        }
    }
    if let Ok(u) = paper_id.parse::<u64>() {
        if let Ok(Some(val)) = spec_dict.get_item(u) {
            return extract_specter_vec(&val);
        }
    }
    Ok(None)
}

fn extract_string_opt(obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if obj.is_none() {
        Ok(None)
    } else {
        Ok(Some(obj.extract()?))
    }
}

fn extract_name_counts_data(obj: &Bound<'_, PyAny>) -> PyResult<Option<NameCountsData>> {
    if obj.is_none() {
        return Ok(None);
    }
    let first: Option<f64> = obj.getattr("first")?.extract()?;
    let first_last: Option<f64> = obj.getattr("first_last")?.extract()?;
    let last: Option<f64> = obj.getattr("last")?.extract()?;
    let last_first_initial: Option<f64> = obj.getattr("last_first_initial")?.extract()?;
    Ok(Some(NameCountsData {
        first: first.unwrap_or(f64::NAN),
        first_last: first_last.unwrap_or(f64::NAN),
        last: last.unwrap_or(f64::NAN),
        last_first_initial: last_first_initial.unwrap_or(f64::NAN),
    }))
}

fn extract_specter_vec(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<f32>>> {
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(arr) = obj.downcast::<PyArray1<f32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        let all_zero = slice.iter().all(|v| *v == 0.0);
        if all_zero {
            return Ok(None);
        }
        return Ok(Some(slice.to_vec()));
    }
    if let Ok(arr) = obj.downcast::<PyArray1<f64>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        let all_zero = slice.iter().all(|v| *v == 0.0);
        if all_zero {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(slice.len());
        for v in slice {
            out.push(*v as f32);
        }
        return Ok(Some(out));
    }
    // Fallback: try to extract as Vec<f64>
    let vec_f64: Vec<f64> = obj.extract()?;
    let all_zero = vec_f64.iter().all(|v| *v == 0.0);
    if all_zero {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(vec_f64.len());
    for v in vec_f64 {
        out.push(v as f32);
    }
    Ok(Some(out))
}

fn specter_payload_to_dict<'py>(
    py: Python<'py>,
    payload: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    if let Ok(dict) = payload.downcast::<PyDict>() {
        return Ok(dict.clone());
    }

    if let Ok(tuple_payload) = payload.downcast::<PyTuple>() {
        if tuple_payload.len() != 2 {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported specter pickle tuple payload; expected (X, keys), got tuple length {}",
                tuple_payload.len()
            )));
        }
        let matrix = tuple_payload.get_item(0)?;
        let keys = tuple_payload.get_item(1)?;
        let out = PyDict::new(py);
        for (idx, key_item) in PyIterator::from_object(&keys)?.enumerate() {
            let key = key_item?;
            let row = matrix.get_item(idx)?;
            out.set_item(key, row)?;
        }
        return Ok(out);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Unsupported specter pickle payload; expected dict or (X, keys) tuple",
    ))
}

fn load_pickle_dict<'py>(py: Python<'py>, path: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
    let builtins = py.import("builtins")?;
    let pickle = py.import("pickle")?;
    let file_obj = builtins.call_method1("open", (path, "rb"))?;
    let loaded = pickle.call_method1("load", (&file_obj,));
    let _ = file_obj.call_method0("close");
    match loaded {
        Ok(value) => {
            if value.is_none() {
                Ok(None)
            } else {
                Ok(Some(specter_payload_to_dict(py, &value)?))
            }
        }
        Err(err) => Err(err),
    }
}

type JsonValue = serde_json::Value;
type JsonObject = serde_json::Map<String, JsonValue>;

fn load_json_value(path: &str) -> PyResult<JsonValue> {
    let file = File::open(path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!("failed to open JSON path {}: {}", path, err))
    })?;
    let reader = BufReader::new(file);
    serde_json::from_reader::<_, JsonValue>(reader).map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "failed to parse JSON path {}: {}",
            path, err
        ))
    })
}

fn json_as_object<'a>(value: &'a JsonValue, context: &str) -> PyResult<&'a JsonObject> {
    value.as_object().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("{} must be a JSON object", context))
    })
}

fn json_value_to_string(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(v) => Some(v.clone()),
        JsonValue::Number(v) => Some(v.to_string()),
        JsonValue::Bool(v) => Some(v.to_string()),
        _ => None,
    }
}

fn json_value_to_id(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(v) => Some(v.clone()),
        JsonValue::Number(v) => Some(v.to_string()),
        _ => None,
    }
}

fn json_value_to_i64(value: &JsonValue) -> Option<i64> {
    match value {
        JsonValue::Number(v) => {
            if let Some(i) = v.as_i64() {
                Some(i)
            } else if let Some(u) = v.as_u64() {
                Some(u as i64)
            } else {
                v.as_f64().map(|f| f as i64)
            }
        }
        _ => None,
    }
}

fn json_get_required<'a>(obj: &'a JsonObject, key: &str, context: &str) -> PyResult<&'a JsonValue> {
    obj.get(key).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "missing required key '{}' in {}",
            key, context
        ))
    })
}

fn json_get_string(obj: &JsonObject, key: &str, default: &str) -> String {
    match obj.get(key) {
        None | Some(JsonValue::Null) => default.to_string(),
        Some(value) => json_value_to_string(value).unwrap_or_else(|| default.to_string()),
    }
}

fn json_get_optional_string(obj: &JsonObject, key: &str) -> Option<String> {
    obj.get(key).and_then(|value| match value {
        JsonValue::Null => None,
        _ => json_value_to_string(value),
    })
}

fn json_get_i64_optional(obj: &JsonObject, key: &str) -> Option<i64> {
    obj.get(key).and_then(json_value_to_i64)
}

fn json_get_string_list(value: Option<&JsonValue>) -> Vec<String> {
    let Some(array) = value.and_then(JsonValue::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        if let Some(text) = json_value_to_string(item) {
            out.push(text);
        }
    }
    out
}

fn json_get_id_set(value: Option<&JsonValue>) -> HashSet<PaperId> {
    let Some(array) = value.and_then(JsonValue::as_array) else {
        return HashSet::new();
    };
    let mut out = HashSet::with_capacity(array.len());
    for item in array {
        if let Some(id) = json_value_to_id(item) {
            out.insert(id);
        }
    }
    out
}

fn json_extract_string_f64_map(value: Option<&JsonValue>) -> HashMap<String, f64> {
    let Some(object) = value.and_then(JsonValue::as_object) else {
        return HashMap::new();
    };
    let mut out = HashMap::with_capacity(object.len());
    for (key, val) in object {
        if let Some(f) = val.as_f64() {
            out.insert(key.clone(), f);
        } else if let Some(i) = val.as_i64() {
            out.insert(key.clone(), i as f64);
        } else if let Some(u) = val.as_u64() {
            out.insert(key.clone(), u as f64);
        }
    }
    out
}

fn load_raw_name_counts_from_json_path(
    path: Option<&str>,
    expected_normalization_version: Option<&str>,
    allow_normalization_version_mismatch: bool,
) -> PyResult<RawNameCountMaps> {
    let Some(path_value) = path else {
        return Ok(RawNameCountMaps::default());
    };
    let counts_json = load_json_value(path_value)?;
    let counts_obj = json_as_object(&counts_json, "name counts payload")?;

    // Validate normalization_version if an expected version was provided.
    if let Some(expected) = expected_normalization_version {
        match counts_obj.get("normalization_version") {
            None => {
                let msg = format!(
                    "Missing normalization_version in name counts artifact; fail-fast by default. \
                     path={} expected={} override=S2AND_ALLOW_NORMALIZATION_VERSION_MISMATCH=1",
                    path_value, expected,
                );
                if !allow_normalization_version_mismatch {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(msg));
                }
                eprintln!("WARNING: {}", msg);
            }
            Some(artifact_val) => {
                let artifact_version = artifact_val.as_str().unwrap_or("");
                if artifact_version != expected {
                    let msg = format!(
                        "Normalization version mismatch between runtime and name-count artifact; \
                         fail-fast by default. path={} expected={} artifact={} \
                         override=S2AND_ALLOW_NORMALIZATION_VERSION_MISMATCH=1",
                        path_value, expected, artifact_version,
                    );
                    if !allow_normalization_version_mismatch {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(msg));
                    }
                    eprintln!("WARNING: {}", msg);
                }
            }
        }
    }

    let first = json_extract_string_f64_map(counts_obj.get("first_dict"));
    let last = json_extract_string_f64_map(counts_obj.get("last_dict"));
    let first_last = json_extract_string_f64_map(counts_obj.get("first_last_dict"));
    let last_first_initial = json_extract_string_f64_map(counts_obj.get("last_first_initial_dict"));
    Ok(RawNameCountMaps {
        first,
        last,
        first_last,
        last_first_initial,
    })
}

fn default_name_tuples_path(py: Python<'_>) -> PyResult<String> {
    let consts = py.import("s2and.consts")?;
    let project_root: String = consts.getattr("PROJECT_ROOT_PATH")?.extract()?;
    let pathlib = py.import("pathlib")?;
    let path_obj = pathlib
        .getattr("Path")?
        .call1((project_root,))?
        .call_method1("joinpath", ("data", "s2and_name_tuples_filtered.txt"))?;
    path_obj.call_method0("as_posix")?.extract()
}

fn load_name_tuples_from_text_path(
    py: Python<'_>,
    path: Option<&str>,
) -> PyResult<HashMap<String, HashSet<String>>> {
    let effective_path = match path {
        Some(value) => value.to_string(),
        None => default_name_tuples_path(py)?,
    };
    if !Path::new(&effective_path).exists() {
        return Ok(HashMap::new());
    }
    let text = fs::read_to_string(&effective_path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to read name tuples path {}: {}",
            effective_path, err
        ))
    })?;
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((a, b)) = trimmed.split_once(',') {
            out.entry(a.to_string())
                .or_insert_with(HashSet::new)
                .insert(b.to_string());
        }
    }
    Ok(out)
}

fn has_name_counts_artifact(raw_name_counts: &RawNameCountMaps) -> bool {
    !raw_name_counts.first.is_empty()
        || !raw_name_counts.last.is_empty()
        || !raw_name_counts.first_last.is_empty()
        || !raw_name_counts.last_first_initial.is_empty()
}

fn canonical_last_for_counts(raw_last: &str, normalized_last: &str) -> String {
    if raw_last.contains('-') || normalized_last.contains(' ') {
        normalized_last.replace(' ', "")
    } else {
        normalized_last.to_string()
    }
}

fn build_name_counts_data_from_artifact(
    raw_name_counts: &RawNameCountMaps,
    raw_first: &str,
    first_normalized_token: &str,
    first_without_apostrophe: &str,
    raw_last: &str,
    last_normalized: &str,
) -> Option<NameCountsData> {
    if !has_name_counts_artifact(raw_name_counts) {
        return None;
    }

    let mut first_for_counts = first_normalized_token.to_string();
    if first_for_counts.is_empty() {
        first_for_counts = first_without_apostrophe
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();
    }
    if raw_first.contains('-') {
        let joined = first_without_apostrophe.replace(' ', "");
        if !joined.is_empty() {
            first_for_counts = joined;
        }
    }

    let last_for_counts = canonical_last_for_counts(raw_last, last_normalized);
    let first_last_key = format!("{} {}", first_for_counts, last_for_counts)
        .trim()
        .to_string();
    let first_initial = first_for_counts
        .chars()
        .next()
        .map(|ch| ch.to_string())
        .unwrap_or_default();
    let last_first_initial_key = format!("{} {}", last_for_counts, first_initial)
        .trim()
        .to_string();

    let first = if py_len(&first_for_counts) > 1 {
        *raw_name_counts.first.get(&first_for_counts).unwrap_or(&1.0)
    } else {
        f64::NAN
    };
    let first_last = if py_len(&first_for_counts) > 1 {
        *raw_name_counts
            .first_last
            .get(&first_last_key)
            .unwrap_or(&1.0)
    } else {
        f64::NAN
    };
    let last = *raw_name_counts.last.get(&last_for_counts).unwrap_or(&1.0);
    let last_first_initial = *raw_name_counts
        .last_first_initial
        .get(&last_first_initial_key)
        .unwrap_or(&1.0);

    Some(NameCountsData {
        first,
        first_last,
        last,
        last_first_initial,
    })
}

fn count_initials(s: &str) -> HashMap<char, usize> {
    let mut counts = HashMap::new();
    for part in s.split(' ') {
        if !part.is_empty() {
            if let Some(ch) = part.chars().next() {
                *counts.entry(ch).or_insert(0) += 1;
            }
        }
    }
    counts
}

fn lasts_equivalent_for_constraint(l1: &str, l2: &str) -> bool {
    if l1 == l2 {
        return true;
    }
    l1.replace(' ', "") == l2.replace(' ', "")
}

fn same_prefix_tokens(a: &str, b: &str) -> bool {
    let mut ita = a.split_whitespace();
    let mut itb = b.split_whitespace();
    loop {
        match (ita.next(), itb.next()) {
            (Some(x), Some(y)) => {
                if !(x.starts_with(y) || y.starts_with(x)) {
                    return false;
                }
            }
            _ => return true,
        }
    }
}

fn name_tuple_contains(map: &HashMap<String, HashSet<String>>, a: &str, b: &str) -> bool {
    map.get(a).map_or(false, |vals| vals.contains(b))
}

fn filter_text_for_char_ngrams(text: &str, stopwords: Option<&HashSet<String>>) -> String {
    let Some(stopwords_set) = stopwords else {
        return text.to_string();
    };
    text.split(' ')
        .filter(|word| !stopwords_set.contains(*word) && py_len(word) > 2)
        .collect::<Vec<_>>()
        .join(" ")
}

fn char_ngrams_counter_python_compat(
    text: &str,
    use_unigrams: bool,
    use_bigrams: bool,
    stopwords: Option<&HashSet<String>>,
) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::new();
    }
    let filtered_text = filter_text_for_char_ngrams(text, stopwords);
    if filtered_text.is_empty() {
        return HashMap::new();
    }
    let chars: Vec<char> = filtered_text.chars().collect();
    if chars.is_empty() {
        return HashMap::new();
    }

    let mut out: HashMap<String, usize> = HashMap::new();
    if use_unigrams {
        for ch in chars.iter() {
            if *ch != ' ' {
                *out.entry(ch.to_string()).or_insert(0) += 1;
            }
        }
    }
    if use_bigrams && chars.len() >= 2 {
        for idx in 0..(chars.len() - 1) {
            if chars[idx] == ' ' || chars[idx + 1] == ' ' {
                continue;
            }
            let gram = format!("{}{}", chars[idx], chars[idx + 1]);
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    if chars.len() >= 3 {
        for idx in 0..(chars.len() - 2) {
            if chars[idx] == ' ' || chars[idx + 1] == ' ' || chars[idx + 2] == ' ' {
                continue;
            }
            let gram = format!("{}{}{}", chars[idx], chars[idx + 1], chars[idx + 2]);
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    if chars.len() >= 4 {
        for idx in 0..(chars.len() - 3) {
            if chars[idx] == ' '
                || chars[idx + 1] == ' '
                || chars[idx + 2] == ' '
                || chars[idx + 3] == ' '
            {
                continue;
            }
            let gram = format!(
                "{}{}{}{}",
                chars[idx],
                chars[idx + 1],
                chars[idx + 2],
                chars[idx + 3]
            );
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    out
}

fn word_ngrams_counter_python_compat(
    text: &str,
    stopwords: &HashSet<String>,
) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::new();
    }
    let text_split: Vec<&str> = text
        .split_whitespace()
        .filter(|word| !stopwords.contains(*word) && py_len(word) > 1)
        .collect();
    if text_split.is_empty() {
        return HashMap::new();
    }
    let mut out: HashMap<String, usize> = HashMap::new();
    for token in text_split.iter() {
        *out.entry((*token).to_string()).or_insert(0) += 1;
    }
    if text_split.len() >= 2 {
        for pair in text_split.windows(2) {
            let gram = format!("{} {}", pair[0], pair[1]);
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    if text_split.len() >= 3 {
        for tri in text_split.windows(3) {
            let gram = format!("{} {} {}", tri[0], tri[1], tri[2]);
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    out
}

fn char_ngrams_counter(text: &str) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::new();
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return HashMap::new();
    }

    let mut out: HashMap<String, usize> = HashMap::new();
    for width in [2usize, 3usize, 4usize] {
        if chars.len() < width {
            continue;
        }
        for idx in 0..=(chars.len() - width) {
            let window = &chars[idx..idx + width];
            if window.iter().any(|ch| *ch == ' ') {
                continue;
            }
            let gram: String = window.iter().collect();
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    out
}

fn word_ngrams_counter(text: &str) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::new();
    }

    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.is_empty() {
        return HashMap::new();
    }

    let mut out: HashMap<String, usize> = HashMap::new();
    for tok in tokens.iter() {
        *out.entry((*tok).to_string()).or_insert(0) += 1;
    }
    if tokens.len() >= 2 {
        for pair in tokens.windows(2) {
            let gram = format!("{} {}", pair[0], pair[1]);
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    if tokens.len() >= 3 {
        for tri in tokens.windows(3) {
            let gram = format!("{} {} {}", tri[0], tri[1], tri[2]);
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    out
}

#[pyfunction]
#[pyo3(signature = (coauthor_texts, affiliation_texts, num_threads = None))]
fn signature_ngrams_batch(
    py: Python<'_>,
    coauthor_texts: Vec<String>,
    affiliation_texts: Vec<String>,
    num_threads: Option<usize>,
) -> PyResult<(Vec<HashMap<String, usize>>, Vec<HashMap<String, usize>>)> {
    if coauthor_texts.len() != affiliation_texts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coauthor_texts and affiliation_texts must have equal length",
        ));
    }
    let n = coauthor_texts.len();

    let pairs = py.allow_threads(|| {
        let compute = || {
            (0..n)
                .into_par_iter()
                .map(|idx| {
                    let coauthors = char_ngrams_counter(&coauthor_texts[idx]);
                    let affiliations = word_ngrams_counter(&affiliation_texts[idx]);
                    (coauthors, affiliations)
                })
                .collect::<Vec<_>>()
        };
        install_with_optional_rayon_pool(num_threads, compute)
    });

    let mut coauthor_out: Vec<HashMap<String, usize>> = Vec::with_capacity(n);
    let mut affiliation_out: Vec<HashMap<String, usize>> = Vec::with_capacity(n);
    for (coauthors, affiliations) in pairs {
        coauthor_out.push(coauthors);
        affiliation_out.push(affiliations);
    }
    Ok((coauthor_out, affiliation_out))
}

fn counter_jaccard_data(
    counter1: &Option<CounterData>,
    counter2: &Option<CounterData>,
    denom_max: f64,
) -> f64 {
    let (Some(c1), Some(c2)) = (counter1.as_ref(), counter2.as_ref()) else {
        return f64::NAN;
    };
    if c1.entries.is_empty() || c2.entries.is_empty() {
        return f64::NAN;
    }
    let (small, large) = if c1.entries.len() <= c2.entries.len() {
        (c1, c2)
    } else {
        (c2, c1)
    };
    let mut intersection = 0.0f64;
    if large.entries.len() < small.entries.len().saturating_mul(4) {
        let mut small_idx = 0usize;
        let mut large_idx = 0usize;
        while small_idx < small.entries.len() && large_idx < large.entries.len() {
            let (small_hash, small_value) = small.entries[small_idx];
            let (large_hash, large_value) = large.entries[large_idx];
            if small_hash == large_hash {
                intersection += small_value.min(large_value) as f64;
                small_idx += 1;
                large_idx += 1;
            } else if small_hash < large_hash {
                small_idx += 1;
            } else {
                large_idx += 1;
            }
        }
    } else {
        for (h, v1) in small.entries.iter() {
            if let Ok(idx) = large.entries.binary_search_by_key(h, |e| e.0) {
                intersection += (*v1).min(large.entries[idx].1) as f64;
            }
        }
    }
    let union = c1.sum as f64 + c2.sum as f64 - intersection;
    if union == 0.0 {
        return f64::NAN;
    }
    let denom = if denom_max.is_infinite() {
        union
    } else {
        union.min(denom_max)
    };
    let score = intersection / denom;
    if score > 1.0 {
        1.0
    } else {
        score
    }
}

fn set_jaccard_data<T: Eq + std::hash::Hash>(
    set1: &Option<HashSet<T>>,
    set2: &Option<HashSet<T>>,
) -> f64 {
    let (Some(s1), Some(s2)) = (set1.as_ref(), set2.as_ref()) else {
        return f64::NAN;
    };
    if s1.is_empty() || s2.is_empty() {
        return f64::NAN;
    }
    let intersection = s1.intersection(s2).count();
    let union = s1.len() + s2.len() - intersection;
    if union == 0 {
        return f64::NAN;
    }
    (intersection as f64) / (union as f64)
}

fn refs_jaccard<T: Eq + std::hash::Hash>(set1: &HashSet<T>, set2: &HashSet<T>) -> f64 {
    if set1.is_empty() || set2.is_empty() {
        return f64::NAN;
    }
    let intersection = set1.intersection(set2).count();
    let union = set1.len() + set2.len() - intersection;
    if union == 0 {
        return f64::NAN;
    }
    (intersection as f64) / (union as f64)
}

fn nanmin(a: f64, b: f64) -> f64 {
    if a.is_nan() && b.is_nan() {
        f64::NAN
    } else if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else {
        a.min(b)
    }
}

fn max_propagate_nan(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

fn compute_name_counts_data(
    counts1: &Option<NameCountsData>,
    counts2: &Option<NameCountsData>,
) -> [f64; 6] {
    let (Some(c1), Some(c2)) = (counts1.as_ref(), counts2.as_ref()) else {
        return [f64::NAN; 6];
    };
    let min_first = nanmin(c1.first, c2.first);
    let min_first_last = nanmin(c1.first_last, c2.first_last);
    let min_last = nanmin(c1.last, c2.last);
    let min_last_first_initial = nanmin(c1.last_first_initial, c2.last_first_initial);
    let max_first = max_propagate_nan(c1.first, c2.first);
    let max_first_last = max_propagate_nan(c1.first_last, c2.first_last);
    [
        min_first,
        min_first_last,
        min_last,
        min_last_first_initial,
        max_first,
        max_first_last,
    ]
}

fn first_names_equal(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let (Some(n1), Some(n2)) = (name1, name2) else {
        return f64::NAN;
    };
    if py_len(n1) == 0 || py_len(n2) == 0 {
        return f64::NAN;
    }
    if n1 == "-" || n2 == "-" {
        return f64::NAN;
    }
    let n1_norm = n1.trim().to_lowercase();
    let n2_norm = n2.trim().to_lowercase();
    if n1_norm == n2_norm {
        1.0
    } else {
        0.0
    }
}

fn middle_initials_overlap(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let s1 = name1.unwrap_or("");
    let s2 = name2.unwrap_or("");
    let c1 = count_initials(s1);
    let c2 = count_initials(s2);
    if c1.is_empty() || c2.is_empty() {
        return f64::NAN;
    }
    let mut intersection_sum: usize = 0;
    for (k, v1) in c1.iter() {
        if let Some(v2) = c2.get(k) {
            intersection_sum += std::cmp::min(*v1, *v2);
        }
    }
    let sum1: usize = c1.values().sum();
    let sum2: usize = c2.values().sum();
    let union_sum = sum1 + sum2 - intersection_sum;
    if union_sum == 0 {
        return f64::NAN;
    }
    let score = (intersection_sum as f64) / (union_sum as f64);
    if score > 1.0 { 1.0 } else { score }
}

fn middle_names_equal(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let (Some(n1), Some(n2)) = (name1, name2) else {
        return f64::NAN;
    };
    if py_len(n1) == 0 || py_len(n2) == 0 {
        return f64::NAN;
    }
    if py_len(n1) == 1 || py_len(n2) == 1 {
        let (Some(c1), Some(c2)) = (n1.chars().next(), n2.chars().next()) else {
            return f64::NAN;
        };
        return if c1 == c2 { 1.0 } else { 0.0 };
    }
    if n1 == n2 {
        1.0
    } else {
        0.0
    }
}

fn middle_one_missing(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let n1 = name1.unwrap_or("");
    let n2 = name2.unwrap_or("");
    let len1 = py_len(n1);
    let len2 = py_len(n2);
    let val = (len1 == 0 && len2 != 0) || (len2 == 0 && len1 != 0);
    if val { 1.0 } else { 0.0 }
}

fn single_char_first(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let n1 = name1.unwrap_or("");
    let n2 = name2.unwrap_or("");
    let val = py_len(n1) == 1 || py_len(n2) == 1;
    if val { 1.0 } else { 0.0 }
}

fn single_char_middle(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let n1 = name1.unwrap_or("");
    let n2 = name2.unwrap_or("");
    let mut val = false;
    for part in n1.split(' ') {
        if py_len(part) == 1 {
            val = true;
            break;
        }
    }
    if !val {
        for part in n2.split(' ') {
            if py_len(part) == 1 {
                val = true;
                break;
            }
        }
    }
    if val { 1.0 } else { 0.0 }
}

fn email_parts(email: &str) -> (String, String) {
    let (prefix_raw, suffix_raw) = if let Some((before_last, after_last)) = email.rsplit_once('@') {
        let mut merged_prefix = String::with_capacity(before_last.len());
        for ch in before_last.chars() {
            if ch != '@' {
                merged_prefix.push(ch);
            }
        }
        (merged_prefix, after_last.to_string())
    } else {
        (email.to_string(), "MISSING".to_string())
    };
    let prefix = prefix_raw.trim_matches('.').to_lowercase();
    let suffix = suffix_raw.trim_matches('.').to_lowercase();
    (prefix, suffix)
}

fn email_pair_parts(
    email1: Option<&str>,
    email2: Option<&str>,
) -> Option<((String, String), (String, String))> {
    let (Some(e1), Some(e2)) = (email1, email2) else {
        return None;
    };
    if py_len(e1) == 0 || py_len(e2) == 0 {
        return None;
    }
    Some((email_parts(e1), email_parts(e2)))
}

fn year_diff(year1: Option<i64>, year2: Option<i64>) -> f64 {
    let (Some(y1_raw), Some(y2_raw)) = (year1, year2) else {
        return f64::NAN;
    };
    let y1 = y1_raw as f64;
    let y2 = y2_raw as f64;
    let diff = (y1 - y2).abs();
    diff.min(50.0)
}

fn position_diff(p1: i64, p2: i64) -> f64 {
    let diff = (p1 - p2).abs() as f64;
    diff.min(50.0)
}

fn cosine_sim_vec_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    let len = a.len().min(b.len());
    for i in 0..len {
        let av = a[i] as f64;
        let bv = b[i] as f64;
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

fn cosine_sim_with_norms(a: &[f32], norm_a: f64, b: &[f32], norm_b: f64) -> f64 {
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let len = a.len().min(b.len());
    for i in 0..len {
        dot += (a[i] as f64) * (b[i] as f64);
    }
    dot / (norm_a * norm_b)
}

fn levenshtein_distance(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }

    if a.is_ascii() && b.is_ascii() {
        return levenshtein_distance_bytes(a.as_bytes(), b.as_bytes());
    }

    let b_chars: Vec<char> = b.chars().collect();
    let len_b = b_chars.len();
    if a.is_empty() {
        return len_b;
    }
    if len_b == 0 {
        return a.chars().count();
    }
    let mut prev: Vec<usize> = (0..=len_b).collect();
    let mut cur: Vec<usize> = vec![0; len_b + 1];
    for (i, a_char) in a.chars().enumerate() {
        cur[0] = i + 1;
        for j in 1..=len_b {
            let deletion = prev[j] + 1;
            let insertion = cur[j - 1] + 1;
            let edit = prev[j - 1] + if a_char == b_chars[j - 1] { 0 } else { 1 };
            cur[j] = deletion.min(insertion).min(edit);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[len_b]
}

fn levenshtein_distance_bytes(a: &[u8], b: &[u8]) -> usize {
    let len_a = a.len();
    let len_b = b.len();
    if len_a == 0 {
        return len_b;
    }
    if len_b == 0 {
        return len_a;
    }
    let mut prev: Vec<usize> = (0..=len_b).collect();
    let mut cur: Vec<usize> = vec![0; len_b + 1];
    for i in 1..=len_a {
        cur[0] = i;
        for j in 1..=len_b {
            let deletion = prev[j] + 1;
            let insertion = cur[j - 1] + 1;
            let edit = prev[j - 1] + if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = deletion.min(insertion).min(edit);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[len_b]
}

fn prefix_dist(a: &str, b: &str) -> f64 {
    if a == b {
        return 0.0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (min_chars, max_chars) = if a_chars.len() < b_chars.len() {
        (&a_chars, &b_chars)
    } else {
        (&b_chars, &a_chars)
    };
    let min_len = min_chars.len();
    for i in (1..=min_len).rev() {
        if min_chars[..i] == max_chars[..i] {
            return 1.0 - (i as f64 / min_len as f64);
        }
    }
    1.0
}

fn lcs_length(a: &str, b: &str) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    if a.is_ascii() && b.is_ascii() {
        return lcs_length_bytes(a.as_bytes(), b.as_bytes());
    }

    let b_chars: Vec<char> = b.chars().collect();
    let len_b = b_chars.len();
    let mut prev = vec![0usize; len_b + 1];
    let mut cur = vec![0usize; len_b + 1];
    for a_char in a.chars() {
        cur[0] = 0;
        for j in 1..=len_b {
            if a_char == b_chars[j - 1] {
                cur[j] = prev[j - 1] + 1;
            } else {
                cur[j] = cur[j - 1].max(prev[j]);
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[len_b]
}

fn lcs_length_bytes(a: &[u8], b: &[u8]) -> usize {
    let len_a = a.len();
    let len_b = b.len();
    if len_a == 0 || len_b == 0 {
        return 0;
    }
    let mut prev = vec![0usize; len_b + 1];
    let mut cur = vec![0usize; len_b + 1];
    for i in 1..=len_a {
        cur[0] = 0;
        for j in 1..=len_b {
            if a[i - 1] == b[j - 1] {
                cur[j] = prev[j - 1] + 1;
            } else {
                cur[j] = cur[j - 1].max(prev[j]);
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[len_b]
}

fn metric_lcs_distance(a: &str, b: &str) -> f64 {
    if a == b {
        return 0.0;
    }
    let len_a = py_len(a);
    let len_b = py_len(b);
    let max_len = len_a.max(len_b);
    if max_len == 0 {
        return 0.0;
    }
    let lcs = lcs_length(a, b);
    1.0 - (lcs as f64 / max_len as f64)
}

fn jaro_winkler_similarity(a: &str, b: &str, long_tolerance: bool) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();
    if a_len == 0 || b_len == 0 {
        return 0.0;
    }
    let min_len = a_len.min(b_len);
    let mut search_range = a_len.max(b_len) / 2;
    if search_range > 0 {
        search_range -= 1;
    }
    let mut a_flags = vec![false; a_len];
    let mut b_flags = vec![false; b_len];
    let mut common_chars = 0usize;
    for i in 0..a_len {
        let low = if i > search_range {
            i - search_range
        } else {
            0
        };
        let mut hi = i + search_range;
        if hi >= b_len {
            hi = b_len.saturating_sub(1);
        }
        for j in low..=hi {
            if !b_flags[j] && a_chars[i] == b_chars[j] {
                a_flags[i] = true;
                b_flags[j] = true;
                common_chars += 1;
                break;
            }
        }
    }
    if common_chars == 0 {
        return 0.0;
    }
    let mut k = 0usize;
    let mut trans_count = 0usize;
    for i in 0..a_len {
        if a_flags[i] {
            while k < b_len {
                if b_flags[k] {
                    break;
                }
                k += 1;
            }
            if k < b_len && a_chars[i] != b_chars[k] {
                trans_count += 1;
            }
            k += 1;
        }
    }
    trans_count /= 2;
    let common_f = common_chars as f64;
    let weight = (common_f / a_len as f64
        + common_f / b_len as f64
        + (common_f - trans_count as f64) / common_f)
        / 3.0;
    let mut weight = weight;
    if weight > 0.7 {
        let j = min_len.min(4);
        let mut i = 0usize;
        while i < j && a_chars[i] == b_chars[i] {
            i += 1;
        }
        if i > 0 {
            weight += (i as f64) * 0.1 * (1.0 - weight);
        }
        if long_tolerance && min_len > 4 && common_chars > i + 1 && 2 * common_chars >= min_len + i
        {
            weight += (1.0 - weight)
                * ((common_chars - i - 1) as f64 / (a_len + b_len - i * 2 + 2) as f64);
        }
    }
    weight
}

fn name_text_features(name1: Option<&str>, name2: Option<&str>) -> [f64; 4] {
    let (Some(n1), Some(n2)) = (name1, name2) else {
        return [f64::NAN; 4];
    };
    if py_len(n1) <= 1 || py_len(n2) <= 1 {
        return [f64::NAN; 4];
    }
    let lev = levenshtein_distance(n1, n2) as f64 / (py_len(n1).max(py_len(n2)) as f64);
    let pref = prefix_dist(n1, n2);
    let lcs = metric_lcs_distance(n1, n2);
    let jaro = jaro_winkler_similarity(n1, n2, false);
    [lev, pref, lcs, jaro]
}

const PAPER_IDX_TITLE: usize = 0;
const PAPER_IDX_HAS_ABSTRACT: usize = 1;
const PAPER_IDX_IN_SIGNATURES: usize = 2;
const PAPER_IDX_IS_RELIABLE: usize = 4;
const PAPER_IDX_PREDICTED_LANGUAGE: usize = 5;
const PAPER_IDX_TITLE_NGRAMS_WORDS: usize = 6;
const PAPER_IDX_AUTHORS: usize = 7;
const PAPER_IDX_VENUE: usize = 8;
const PAPER_IDX_JOURNAL_NAME: usize = 9;
const PAPER_IDX_TITLE_NGRAMS_CHARS: usize = 10;
const PAPER_IDX_VENUE_NGRAMS: usize = 11;
const PAPER_IDX_JOURNAL_NGRAMS: usize = 12;
const PAPER_IDX_REFERENCE_DETAILS: usize = 13;
const PAPER_IDX_YEAR: usize = 14;
const PAPER_IDX_REFERENCES: usize = 15;
const PAPER_IDX_PAPER_ID: usize = 16;
const FROM_DATASET_PAPER_PREPROCESS_CHUNK_SIZE: usize = 4096;
const PAPER_FASTPATH_REQUIRED_FIELDS: [(usize, &str); 16] = [
    (PAPER_IDX_TITLE, "title"),
    (PAPER_IDX_HAS_ABSTRACT, "has_abstract"),
    (PAPER_IDX_IN_SIGNATURES, "in_signatures"),
    (PAPER_IDX_IS_RELIABLE, "is_reliable"),
    (PAPER_IDX_PREDICTED_LANGUAGE, "predicted_language"),
    (PAPER_IDX_TITLE_NGRAMS_WORDS, "title_ngrams_words"),
    (PAPER_IDX_AUTHORS, "authors"),
    (PAPER_IDX_VENUE, "venue"),
    (PAPER_IDX_JOURNAL_NAME, "journal_name"),
    (PAPER_IDX_TITLE_NGRAMS_CHARS, "title_ngrams_chars"),
    (PAPER_IDX_VENUE_NGRAMS, "venue_ngrams"),
    (PAPER_IDX_JOURNAL_NGRAMS, "journal_ngrams"),
    (PAPER_IDX_REFERENCE_DETAILS, "reference_details"),
    (PAPER_IDX_YEAR, "year"),
    (PAPER_IDX_REFERENCES, "references"),
    (PAPER_IDX_PAPER_ID, "paper_id"),
];

const SIG_IDX_FIRST_RAW: usize = 0;
const SIG_IDX_FIRST_NORMALIZED_NO_APOSTROPHE: usize = 1;
const SIG_IDX_MIDDLE_RAW: usize = 2;
const SIG_IDX_MIDDLE_NORMALIZED_NO_APOSTROPHE: usize = 3;
const SIG_IDX_LAST_NORMALIZED: usize = 4;
const SIG_IDX_LAST_RAW: usize = 5;
const SIG_IDX_COAUTHORS: usize = 9;
const SIG_IDX_COAUTHOR_BLOCKS: usize = 10;
const SIG_IDX_AFFILIATIONS: usize = 12;
const SIG_IDX_AFFILIATIONS_NGRAMS: usize = 13;
const SIG_IDX_COAUTHOR_NGRAMS: usize = 14;
const SIG_IDX_EMAIL: usize = 15;
const SIG_IDX_ORCID: usize = 16;
const SIG_IDX_NAME_COUNTS: usize = 17;
const SIG_IDX_POSITION: usize = 18;
const SIG_IDX_PAPER_ID: usize = 23;
const FULL_FEATURE_COUNT: usize = 39;
const SIGNATURE_FASTPATH_REQUIRED_FIELDS: [(usize, &str); 13] = [
    (
        SIG_IDX_FIRST_NORMALIZED_NO_APOSTROPHE,
        "author_info_first_normalized_without_apostrophe",
    ),
    (
        SIG_IDX_MIDDLE_NORMALIZED_NO_APOSTROPHE,
        "author_info_middle_normalized_without_apostrophe",
    ),
    (SIG_IDX_LAST_NORMALIZED, "author_info_last_normalized"),
    (SIG_IDX_COAUTHORS, "author_info_coauthors"),
    (SIG_IDX_COAUTHOR_BLOCKS, "author_info_coauthor_blocks"),
    (SIG_IDX_AFFILIATIONS, "author_info_affiliations"),
    (
        SIG_IDX_AFFILIATIONS_NGRAMS,
        "author_info_affiliations_n_grams",
    ),
    (SIG_IDX_COAUTHOR_NGRAMS, "author_info_coauthor_n_grams"),
    (SIG_IDX_EMAIL, "author_info_email"),
    (SIG_IDX_ORCID, "author_info_orcid"),
    (SIG_IDX_NAME_COUNTS, "author_info_name_counts"),
    (SIG_IDX_POSITION, "author_info_position"),
    (SIG_IDX_PAPER_ID, "paper_id"),
];

impl RustFeaturizer {
    fn cluster_seeds_disallow_index(&self) -> &HashMap<String, HashSet<String>> {
        self.cluster_seeds_disallow_index.get_or_init(|| {
            let mut index: HashMap<String, HashSet<String>> = HashMap::new();
            for (left, right) in self.cluster_seeds_disallow.iter() {
                index
                    .entry(left.clone())
                    .or_insert_with(HashSet::new)
                    .insert(right.clone());
            }
            index
        })
    }

    fn cluster_seeds_disallow_contains(&self, sig_id1: &str, sig_id2: &str) -> bool {
        let (left, right) = canonical_signature_pair_ref(sig_id1, sig_id2);
        self.cluster_seeds_disallow_index()
            .get(left)
            .is_some_and(|rights| rights.contains(right))
    }

    fn featurize_pair_data(
        &self,
        s1: &SignatureData,
        s2: &SignatureData,
        p1: &PaperData,
        p2: &PaperData,
    ) -> [f64; FULL_FEATURE_COUNT] {
        let mut feats = [f64::NAN; FULL_FEATURE_COUNT];
        let mut feat_i: usize = 0;
        macro_rules! push_feat {
            ($value:expr) => {{
                feats[feat_i] = $value;
                feat_i += 1;
            }};
        }

        let first1 = s1.first.as_deref();
        let first2 = s2.first.as_deref();
        let middle1 = s1.middle.as_deref();
        let middle2 = s2.middle.as_deref();

        push_feat!(first_names_equal(first1, first2));
        push_feat!(middle_initials_overlap(middle1, middle2));
        push_feat!(middle_names_equal(middle1, middle2));
        push_feat!(middle_one_missing(middle1, middle2));
        push_feat!(single_char_first(first1, first2));
        push_feat!(single_char_middle(middle1, middle2));

        push_feat!(counter_jaccard_data(
            &s1.affiliations,
            &s2.affiliations,
            f64::INFINITY,
        ));

        let (email_prefix, email_suffix) =
            match email_pair_parts(s1.email.as_deref(), s2.email.as_deref()) {
                Some(((p1, sfx1), (p2, sfx2))) => (
                    if p1 == p2 { 1.0 } else { 0.0 },
                    if sfx1 == sfx2 { 1.0 } else { 0.0 },
                ),
                None => (f64::NAN, f64::NAN),
            };
        push_feat!(email_prefix);
        push_feat!(email_suffix);

        push_feat!(set_jaccard_data(&s1.coauthor_blocks, &s2.coauthor_blocks));
        push_feat!(counter_jaccard_data(
            &s1.coauthor_ngrams,
            &s2.coauthor_ngrams,
            5000.0,
        ));
        push_feat!(set_jaccard_data(&s1.coauthors, &s2.coauthors));

        push_feat!(counter_jaccard_data(
            &p1.venue_ngrams,
            &p2.venue_ngrams,
            f64::INFINITY,
        ));
        push_feat!(year_diff(p1.year, p2.year));

        push_feat!(counter_jaccard_data(
            &p1.title_words,
            &p2.title_words,
            f64::INFINITY,
        ));
        push_feat!(counter_jaccard_data(
            &p1.title_chars,
            &p2.title_chars,
            f64::INFINITY,
        ));

        if self.compute_reference_features && p1.ref_details_present && p2.ref_details_present {
            push_feat!(counter_jaccard_data(
                &p1.ref_authors,
                &p2.ref_authors,
                5000.0,
            ));
            push_feat!(counter_jaccard_data(
                &p1.ref_titles,
                &p2.ref_titles,
                f64::INFINITY,
            ));
            push_feat!(counter_jaccard_data(
                &p1.ref_venues,
                &p2.ref_venues,
                f64::INFINITY,
            ));
            push_feat!(counter_jaccard_data(
                &p1.ref_blocks,
                &p2.ref_blocks,
                f64::INFINITY,
            ));
            let self_cite =
                if p1.references.contains(&s2.paper_id) || p2.references.contains(&s1.paper_id) {
                    1.0
                } else {
                    0.0
                };
            push_feat!(self_cite);
            push_feat!(refs_jaccard(&p1.references, &p2.references));
        } else {
            for _ in 0..6 {
                push_feat!(f64::NAN);
            }
        }

        let english_or_unknown_count = {
            let mut count = 0i64;
            if let Some(l1) = p1.predicted_language.as_deref() {
                if l1 == "en" || l1 == "un" {
                    count += 1;
                }
            }
            if let Some(l2) = p2.predicted_language.as_deref() {
                if l2 == "en" || l2 == "un" {
                    count += 1;
                }
            }
            count
        };

        push_feat!(position_diff(s1.position, s2.position));
        push_feat!((p1.has_abstract as i64 + p2.has_abstract as i64) as f64);
        push_feat!(english_or_unknown_count as f64);
        let same_lang = match (
            p1.predicted_language.as_deref(),
            p2.predicted_language.as_deref(),
        ) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b,
            _ => false,
        };
        push_feat!(if same_lang { 1.0 } else { 0.0 });
        push_feat!((p1.is_reliable as i64 + p2.is_reliable as i64) as f64);

        let counts = compute_name_counts_data(&s1.name_counts, &s2.name_counts);
        for value in counts.iter() {
            push_feat!(*value);
        }

        let specter_sim = if english_or_unknown_count == 2 {
            if let (Some(specter_a), Some(specter_b)) = (p1.specter.as_ref(), p2.specter.as_ref()) {
                let score = match (p1.specter_norm, p2.specter_norm) {
                    (Some(norm_a), Some(norm_b)) if specter_a.len() == specter_b.len() => {
                        cosine_sim_with_norms(specter_a, norm_a, specter_b, norm_b)
                    }
                    _ => cosine_sim_vec_f32(specter_a, specter_b),
                };
                score + 1.0
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        push_feat!(specter_sim);

        push_feat!(counter_jaccard_data(
            &p1.journal_ngrams,
            &p2.journal_ngrams,
            f64::INFINITY,
        ));

        let advanced = name_text_features(s1.adv_name.as_deref(), s2.adv_name.as_deref());
        for value in advanced.iter() {
            push_feat!(*value);
        }

        debug_assert_eq!(feat_i, FULL_FEATURE_COUNT);
        feats
    }

    fn constraint_value_from_records(
        &self,
        sig_id1: &str,
        sig_id2: &str,
        s1: &SignatureData,
        s2: &SignatureData,
        p1: &PaperData,
        p2: &PaperData,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
    ) -> Option<f64> {
        if self.cluster_seeds_disallow_contains(sig_id1, sig_id2) {
            return Some(self.cluster_seed_disallow_value);
        }

        if !incremental_dont_use_cluster_seeds {
            if let (Some(c1), Some(c2)) = (
                self.cluster_seeds_require.get(sig_id1),
                self.cluster_seeds_require.get(sig_id2),
            ) {
                if c1 == c2 {
                    return Some(self.cluster_seed_require_value);
                }
            }
        }

        if dont_merge_cluster_seeds {
            if let (Some(c1), Some(c2)) = (
                self.cluster_seeds_require.get(sig_id1),
                self.cluster_seeds_require.get(sig_id2),
            ) {
                if c1 != c2 {
                    return Some(self.cluster_seed_disallow_value);
                }
            }
        }

        if let (Some(o1), Some(o2)) = (s1.orcid.as_deref(), s2.orcid.as_deref()) {
            if o1 == o2 {
                return Some(low_value);
            }
        }

        let last1 = s1.last_normalized.as_deref().unwrap_or("");
        let last2 = s2.last_normalized.as_deref().unwrap_or("");
        if !lasts_equivalent_for_constraint(last1, last2) {
            return Some(high_value);
        }

        let first1 = s1.first.as_deref().unwrap_or("");
        let first2 = s2.first.as_deref().unwrap_or("");
        if !first1.is_empty() && !first2.is_empty() {
            if let (Some(c1), Some(c2)) = (first1.chars().next(), first2.chars().next()) {
                if c1 != c2 {
                    return Some(high_value);
                }
            }
        }

        if p1.is_reliable && p2.is_reliable {
            let l1 = p1.predicted_language.as_deref();
            let l2 = p2.predicted_language.as_deref();
            if l1 != l2 {
                return Some(high_value);
            }
        }

        let f1_join: String = first1.split_whitespace().collect();
        let f2_join: String = first2.split_whitespace().collect();
        let f1_tok = first1.split_whitespace().next().unwrap_or(first1);
        let f2_tok = first2.split_whitespace().next().unwrap_or(first2);
        let known_alias = name_tuple_contains(&self.name_tuples, first1, first2)
            || name_tuple_contains(&self.name_tuples, &f1_join, &f2_join)
            || name_tuple_contains(&self.name_tuples, f1_tok, f2_tok);

        let prefix = same_prefix_tokens(first1, first2);
        if !prefix && !known_alias {
            return Some(high_value);
        }

        let middle1_str = s1.middle.as_deref().unwrap_or("");
        let middle1_tokens: Vec<&str> = middle1_str.split_whitespace().collect();
        if !middle1_tokens.is_empty() {
            let middle2_str = s2.middle.as_deref().unwrap_or("");
            let middle2_tokens: Vec<&str> = middle2_str.split_whitespace().collect();
            if !middle2_tokens.is_empty() {
                let middle1_set: HashSet<&str> = middle1_tokens.iter().copied().collect();
                let middle2_set: HashSet<&str> = middle2_tokens.iter().copied().collect();
                let mut overlapping_affixes: HashSet<&str> = HashSet::new();
                for token in middle1_set.intersection(&middle2_set) {
                    if is_dropped_affix(token) {
                        overlapping_affixes.insert(*token);
                    }
                }

                let middle_1_all: Vec<&str> = middle1_tokens
                    .iter()
                    .copied()
                    .filter(|w| !w.is_empty() && !overlapping_affixes.contains(w))
                    .collect();
                let middle_2_all: Vec<&str> = middle2_tokens
                    .iter()
                    .copied()
                    .filter(|w| !w.is_empty() && !overlapping_affixes.contains(w))
                    .collect();

                let middle_1_words: HashSet<&str> = middle_1_all
                    .iter()
                    .copied()
                    .filter(|w| py_len(w) > 1)
                    .collect();
                let middle_2_words: HashSet<&str> = middle_2_all
                    .iter()
                    .copied()
                    .filter(|w| py_len(w) > 1)
                    .collect();

                let mut middle_1_firsts: HashSet<char> = HashSet::new();
                for word in middle_1_all.iter() {
                    if let Some(ch) = word.chars().next() {
                        middle_1_firsts.insert(ch);
                    }
                }
                let mut middle_2_firsts: HashSet<char> = HashSet::new();
                for word in middle_2_all.iter() {
                    if let Some(ch) = word.chars().next() {
                        middle_2_firsts.insert(ch);
                    }
                }

                let conflicting_initials = !middle_1_firsts.is_empty()
                    && !middle_2_firsts.is_empty()
                    && middle_1_firsts.is_disjoint(&middle_2_firsts);

                let mut middle_1_chars: HashSet<char> = HashSet::new();
                for word in middle_1_words.iter() {
                    for ch in word.chars() {
                        middle_1_chars.insert(ch);
                    }
                }
                let mut middle_2_chars: HashSet<char> = HashSet::new();
                for word in middle_2_words.iter() {
                    for ch in word.chars() {
                        middle_2_chars.insert(ch);
                    }
                }

                let conflicting_full_names = !middle_1_words.is_empty()
                    && !middle_2_words.is_empty()
                    && middle_1_words.is_disjoint(&middle_2_words)
                    && middle_1_chars != middle_2_chars;

                if conflicting_initials || conflicting_full_names {
                    return Some(high_value);
                }
            }
        }

        None
    }

    fn get_constraint_value_for_pair(
        &self,
        sig_id1: &str,
        sig_id2: &str,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
    ) -> PyResult<Option<f64>> {
        let s1 = self
            .signatures
            .get(sig_id1)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string()))?;
        let s2 = self
            .signatures
            .get(sig_id2)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string()))?;
        let p1 = self
            .papers
            .get(&s1.paper_id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(s1.paper_id.to_string()))?;
        let p2 = self
            .papers
            .get(&s2.paper_id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(s2.paper_id.to_string()))?;
        Ok(self.constraint_value_from_records(
            sig_id1,
            sig_id2,
            s1,
            s2,
            p1,
            p2,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
        ))
    }

    fn validate_constraint_pair_inputs(&self, sig_id1: &str, sig_id2: &str) -> PyResult<()> {
        let s1 = self
            .signatures
            .get(sig_id1)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string()))?;
        let s2 = self
            .signatures
            .get(sig_id2)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string()))?;
        if self.papers.get(&s1.paper_id).is_none() {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                s1.paper_id.to_string(),
            ));
        }
        if self.papers.get(&s2.paper_id).is_none() {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                s2.paper_id.to_string(),
            ));
        }
        Ok(())
    }

    fn signature_id_order(&self) -> &[String] {
        if !self.signature_ids.is_empty() {
            return self.signature_ids.as_slice();
        }
        self.cached_signature_id_order
            .get_or_init(|| {
                let mut ids: Vec<String> = self.signatures.keys().cloned().collect();
                ids.sort_unstable();
                ids
            })
            .as_slice()
    }

    fn full_feature_count(&self) -> usize {
        FULL_FEATURE_COUNT
    }
}

#[pymethods]
impl RustFeaturizer {
    #[classattr]
    const SUPPORTS_FROM_DATASET_PAPER_PREPROCESS: bool = true;

    #[staticmethod]
    #[pyo3(signature = (dataset, cluster_seed_require_value = 0.0, cluster_seed_disallow_value = 10000.0, num_threads = None))]
    fn from_dataset(
        py: Python<'_>,
        dataset: &Bound<'_, PyAny>,
        cluster_seed_require_value: f64,
        cluster_seed_disallow_value: f64,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let compute_reference_features: bool = dataset
            .getattr("compute_reference_features")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let preprocess: bool = dataset
            .getattr("preprocess")
            .and_then(|v| v.extract())
            .unwrap_or(true);

        let text_module = py.import("s2and.text")?;
        let unidecode = text_module.getattr("unidecode")?;
        let stop_words = extract_required_string_set(&text_module.getattr("STOPWORDS")?)?;
        let venue_stop_words = extract_required_string_set(&text_module.getattr("VENUE_STOP_WORDS")?)?;
        let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
        let affiliation_stopwords = extract_affiliation_stopwords(py)?;
        let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
        let mut language_detector: Option<LanguageDetectorCompat> = None;

        #[derive(Clone)]
        struct PaperInput {
            paper_id: PaperId,
            raw_title: String,
            raw_venue: String,
            raw_journal_name: String,
            raw_authors: Vec<(i64, String)>,
            need_title_words: bool,
            need_title_chars: bool,
            need_venue_ngrams: bool,
            need_journal_ngrams: bool,
            need_author_normalization: bool,
        }

        #[derive(Clone)]
        struct PaperComputed {
            paper_id: PaperId,
            normalized_authors: Vec<(i64, String)>,
            title_words: Option<CounterData>,
            title_chars: Option<CounterData>,
            venue_ngrams: Option<CounterData>,
            journal_ngrams: Option<CounterData>,
        }

        let papers_obj = dataset.getattr("papers")?;
        let papers_dict = papers_obj.downcast::<PyDict>()?;
        let use_paper_tuple_fastpath = validate_dict_namedtuple_fastpath_contract(
            &papers_dict,
            &PAPER_FASTPATH_REQUIRED_FIELDS,
            "Paper",
        )?;
        let specter_obj = dataset.getattr("specter_embeddings").ok();
        let specter_dict = specter_obj
            .as_ref()
            .and_then(|v| v.downcast::<PyDict>().ok());

        let mut papers = HashMap::with_capacity(papers_dict.len());
        let mut paper_authors_by_id: HashMap<PaperId, Vec<(i64, String)>> =
            HashMap::with_capacity(papers_dict.len());
        let mut paper_inputs: Vec<PaperInput> = Vec::new();
        for (_paper_id_obj, paper_obj) in papers_dict.iter() {
            let paper_id = extract_id_string(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_PAPER_ID,
                "paper_id",
            )?)?;
            let raw_title = extract_string_opt(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_TITLE,
                "title",
            )?)?
            .unwrap_or_default();
            let raw_venue = extract_string_opt(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_VENUE,
                "venue",
            )?)?
            .unwrap_or_default();
            let raw_journal_name = extract_string_opt(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_JOURNAL_NAME,
                "journal_name",
            )?)?
            .unwrap_or_default();
            let in_signatures: Option<bool> = get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_IN_SIGNATURES,
                "in_signatures",
            )?
            .extract()?;
            let in_signatures = in_signatures.unwrap_or(false);
            let venue_ngrams = extract_counter(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_VENUE_NGRAMS,
                "venue_ngrams",
            )?)?;
            let title_words = extract_counter(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_TITLE_NGRAMS_WORDS,
                "title_ngrams_words",
            )?)?;
            let title_chars = extract_counter(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_TITLE_NGRAMS_CHARS,
                "title_ngrams_chars",
            )?)?;
            let journal_ngrams = extract_counter(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_JOURNAL_NGRAMS,
                "journal_ngrams",
            )?)?;
            let paper_authors =
                extract_paper_authors_with_positions(&get_namedtuple_item_or_attr(
                    &paper_obj,
                    use_paper_tuple_fastpath,
                    PAPER_IDX_AUTHORS,
                    "authors",
                )?)?;

            let ref_details_present;
            let mut ref_authors = None;
            let mut ref_titles = None;
            let mut ref_venues = None;
            let mut ref_blocks = None;
            if compute_reference_features {
                let ref_details_obj = get_namedtuple_item_or_attr(
                    &paper_obj,
                    use_paper_tuple_fastpath,
                    PAPER_IDX_REFERENCE_DETAILS,
                    "reference_details",
                )?;
                ref_details_present = !ref_details_obj.is_none();
                if ref_details_present {
                    (ref_authors, ref_titles, ref_venues, ref_blocks) =
                        extract_reference_details_counters(py, &ref_details_obj)?;
                }
            } else {
                ref_details_present = false;
            }

            let references = extract_set_id_string(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_REFERENCES,
                "references",
            )?)?;
            let year: Option<i64> = get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_YEAR,
                "year",
            )?
            .extract()?;
            let year = match year {
                Some(v) if v > 0 => Some(v),
                _ => None,
            };
            let has_abstract: bool = get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_HAS_ABSTRACT,
                "has_abstract",
            )?
            .extract()?;
            let mut predicted_language = extract_string_opt(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_PREDICTED_LANGUAGE,
                "predicted_language",
            )?)?;
            let is_reliable_raw: Option<bool> = get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_IS_RELIABLE,
                "is_reliable",
            )?
            .extract()?;
            let mut is_reliable = is_reliable_raw.unwrap_or(false);

            let need_title_words = title_words.is_none();
            let need_title_chars = preprocess && in_signatures && title_chars.is_none();
            let need_venue_ngrams = preprocess && in_signatures && venue_ngrams.is_none();
            let need_journal_ngrams = preprocess && in_signatures && journal_ngrams.is_none();
            let need_language = in_signatures && predicted_language.is_none();
            if need_language {
                if language_detector.is_none() {
                    language_detector = Some(LanguageDetectorCompat::new(py));
                }
                let detector = language_detector.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("missing language detector")
                })?;
                let (reliable, _is_english, language) = detector.detect(&raw_title);
                predicted_language = Some(language);
                is_reliable = reliable;
            }

            let need_author_normalization = need_title_words
                || need_title_chars
                || need_venue_ngrams
                || need_journal_ngrams
                || need_language;
            if need_author_normalization {
                ensure_unidecode_for_text(&unidecode, &raw_title, &mut unidecode_char_map)?;
                if preprocess {
                    ensure_unidecode_for_text(&unidecode, &raw_venue, &mut unidecode_char_map)?;
                    ensure_unidecode_for_text(
                        &unidecode,
                        &raw_journal_name,
                        &mut unidecode_char_map,
                    )?;
                }
                for (_, author_name) in paper_authors.iter() {
                    ensure_unidecode_for_text(&unidecode, author_name, &mut unidecode_char_map)?;
                }
                paper_inputs.push(PaperInput {
                    paper_id: paper_id.clone(),
                    raw_title,
                    raw_venue,
                    raw_journal_name,
                    raw_authors: paper_authors.clone(),
                    need_title_words,
                    need_title_chars,
                    need_venue_ngrams,
                    need_journal_ngrams,
                    need_author_normalization,
                });
            }
            paper_authors_by_id.insert(paper_id.clone(), paper_authors);

            let specter = if let Some(spec_dict) = &specter_dict {
                extract_specter_for_paper_id(spec_dict, &paper_id)?
            } else {
                None
            };
            let specter_norm = specter.as_ref().map(|values| {
                values
                    .iter()
                    .map(|value| {
                        let value_f64 = *value as f64;
                        value_f64 * value_f64
                    })
                    .sum::<f64>()
                    .sqrt()
            });

            papers.insert(
                paper_id,
                PaperData {
                    venue_ngrams,
                    title_words,
                    title_chars,
                    ref_authors,
                    ref_titles,
                    ref_venues,
                    ref_blocks,
                    references,
                    year,
                    has_abstract,
                    predicted_language,
                    is_reliable,
                    journal_ngrams,
                    specter,
                    specter_norm,
                    ref_details_present,
                },
            );
        }

        for paper_input_chunk in paper_inputs.chunks(FROM_DATASET_PAPER_PREPROCESS_CHUNK_SIZE) {
            let computed_chunk: Vec<PaperComputed> = py.allow_threads(|| {
                let compute = || {
                    paper_input_chunk
                        .par_iter()
                        .map(|paper_input| {
                            let normalized_title = normalize_text_compat_from_map(
                                &paper_input.raw_title,
                                false,
                                &unidecode_char_map,
                            );
                            let normalized_venue = if paper_input.need_venue_ngrams {
                                Some(normalize_text_compat_from_map(
                                    &paper_input.raw_venue,
                                    false,
                                    &unidecode_char_map,
                                ))
                            } else {
                                None
                            };
                            let normalized_journal = if paper_input.need_journal_ngrams {
                                Some(normalize_text_compat_from_map(
                                    &paper_input.raw_journal_name,
                                    false,
                                    &unidecode_char_map,
                                ))
                            } else {
                                None
                            };
                            let normalized_authors = if paper_input.need_author_normalization {
                                paper_input
                                    .raw_authors
                                    .iter()
                                    .map(|(position, raw_name)| {
                                        (
                                            *position,
                                            normalize_text_compat_from_map(
                                                raw_name,
                                                false,
                                                &unidecode_char_map,
                                            ),
                                        )
                                    })
                                    .collect()
                            } else {
                                paper_input.raw_authors.clone()
                            };
                            let title_words = if paper_input.need_title_words {
                                counter_data_from_usize_map(word_ngrams_counter_python_compat(
                                    &normalized_title,
                                    &stop_words,
                                ))
                            } else {
                                None
                            };
                            let title_chars = if paper_input.need_title_chars {
                                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                    &normalized_title,
                                    false,
                                    true,
                                    Some(&stop_words),
                                ))
                            } else {
                                None
                            };
                            let venue_ngrams = if paper_input.need_venue_ngrams {
                                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                    normalized_venue.as_deref().unwrap_or(""),
                                    false,
                                    true,
                                    Some(&venue_stop_words),
                                ))
                            } else {
                                None
                            };
                            let journal_ngrams = if paper_input.need_journal_ngrams {
                                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                    normalized_journal.as_deref().unwrap_or(""),
                                    false,
                                    true,
                                    Some(&venue_stop_words),
                                ))
                            } else {
                                None
                            };
                            PaperComputed {
                                paper_id: paper_input.paper_id.clone(),
                                normalized_authors,
                                title_words,
                                title_chars,
                                venue_ngrams,
                                journal_ngrams,
                            }
                        })
                        .collect::<Vec<_>>()
                };
                install_with_optional_rayon_pool(num_threads, compute)
            });
            for computed in computed_chunk {
                let PaperComputed {
                    paper_id,
                    normalized_authors,
                    title_words,
                    title_chars,
                    venue_ngrams,
                    journal_ngrams,
                } = computed;
                if let Some(paper) = papers.get_mut(&paper_id) {
                    if let Some(counter) = title_words {
                        paper.title_words = Some(counter);
                    }
                    if let Some(counter) = title_chars {
                        paper.title_chars = Some(counter);
                    }
                    if let Some(counter) = venue_ngrams {
                        paper.venue_ngrams = Some(counter);
                    }
                    if let Some(counter) = journal_ngrams {
                        paper.journal_ngrams = Some(counter);
                    }
                }
                paper_authors_by_id.insert(paper_id, normalized_authors);
            }
        }

        let signatures_obj = dataset.getattr("signatures")?;
        let signatures_dict = signatures_obj.downcast::<PyDict>()?;
        let use_signature_tuple_fastpath = validate_dict_namedtuple_fastpath_contract(
            &signatures_dict,
            &SIGNATURE_FASTPATH_REQUIRED_FIELDS,
            "Signature",
        )?;
        let mut signature_rows: Vec<(String, SignatureData, Option<String>, Option<String>)> =
            Vec::with_capacity(signatures_dict.len());
        for (sig_id_obj, sig_obj) in signatures_dict.iter() {
            let sig_id: String = sig_id_obj.extract()?;
            let raw_first = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_FIRST_RAW,
                "author_info_first",
            )?)?
            .unwrap_or_default();
            let raw_middle = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_MIDDLE_RAW,
                "author_info_middle",
            )?)?
            .unwrap_or_default();
            let raw_last = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_LAST_RAW,
                "author_info_last",
            )?)?
            .unwrap_or_default();

            let mut first = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_FIRST_NORMALIZED_NO_APOSTROPHE,
                "author_info_first_normalized_without_apostrophe",
            )?)?;
            let mut middle = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_MIDDLE_NORMALIZED_NO_APOSTROPHE,
                "author_info_middle_normalized_without_apostrophe",
            )?)?;
            let mut last_normalized = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_LAST_NORMALIZED,
                "author_info_last_normalized",
            )?)?;
            if first.is_none() || middle.is_none() || last_normalized.is_none() {
                ensure_unidecode_for_text(&unidecode, &raw_first, &mut unidecode_char_map)?;
                ensure_unidecode_for_text(&unidecode, &raw_middle, &mut unidecode_char_map)?;
                ensure_unidecode_for_text(&unidecode, &raw_last, &mut unidecode_char_map)?;

                let (first_without_apostrophe, middle_without_apostrophe) =
                    split_first_middle_hyphen_aware_compat(
                        &raw_first,
                        &raw_middle,
                        &name_prefixes,
                        &unidecode_char_map,
                    );
                if first.is_none() {
                    first = Some(first_without_apostrophe);
                }
                if middle.is_none() {
                    middle = Some(middle_without_apostrophe);
                }
                if last_normalized.is_none() {
                    last_normalized = Some(normalize_text_compat_from_map(
                        &raw_last,
                        false,
                        &unidecode_char_map,
                    ));
                }
            }

            let raw_orcid = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_ORCID,
                "author_info_orcid",
            )?)?;
            let orcid = if let Some(value) = raw_orcid {
                let upper_value = value.to_ascii_uppercase();
                extract_orcid_from_source_id(&value)
                    .or_else(|| extract_orcid_from_source_id(&upper_value))
                    .or(Some(value))
            } else {
                None
            };
            let email = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_EMAIL,
                "author_info_email",
            )?)?;
            let affiliations = extract_counter(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_AFFILIATIONS_NGRAMS,
                "author_info_affiliations_n_grams",
            )?)?;
            let mut coauthor_blocks = extract_optional_string_set(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_COAUTHOR_BLOCKS,
                "author_info_coauthor_blocks",
            )?)?;
            let coauthor_ngrams = extract_counter(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_COAUTHOR_NGRAMS,
                "author_info_coauthor_n_grams",
            )?)?;
            let mut coauthors = extract_optional_string_set(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_COAUTHORS,
                "author_info_coauthors",
            )?)?;
            let position: i64 = get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_POSITION,
                "author_info_position",
            )?
            .extract()?;
            let paper_id = extract_id_string(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_PAPER_ID,
                "paper_id",
            )?)?;
            let name_counts = extract_name_counts_data(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_NAME_COUNTS,
                "author_info_name_counts",
            )?)?;

            let mut coauthor_text_for_compute: Option<String> = None;
            let mut affiliation_text_for_compute: Option<String> = None;

            let need_coauthor_from_papers =
                coauthor_ngrams.is_none() || coauthors.is_none() || coauthor_blocks.is_none();
            if need_coauthor_from_papers {
                if let Some(paper_authors) = paper_authors_by_id.get(&paper_id) {
                    let coauthor_names: Vec<String> = paper_authors
                        .iter()
                        .filter(|(author_position, _)| *author_position != position)
                        .map(|(_, author_name)| author_name.clone())
                        .collect();
                    if !coauthor_names.is_empty() {
                        if coauthor_ngrams.is_none() {
                            coauthor_text_for_compute = Some(coauthor_names.join(" "));
                        }
                        if coauthors.is_none() {
                            coauthors = Some(coauthor_names.iter().cloned().collect());
                        }
                        if coauthor_blocks.is_none() {
                            let mut coauthor_blocks_set: HashSet<String> =
                                HashSet::with_capacity(coauthor_names.len());
                            for coauthor in coauthor_names.iter() {
                                coauthor_blocks_set.insert(compute_block_compat(coauthor));
                            }
                            if !coauthor_blocks_set.is_empty() {
                                coauthor_blocks = Some(coauthor_blocks_set);
                            }
                        }
                    }
                }
            }

            if affiliations.is_none() {
                let affiliation_list = extract_string_list(&get_namedtuple_item_or_attr(
                    &sig_obj,
                    use_signature_tuple_fastpath,
                    SIG_IDX_AFFILIATIONS,
                    "author_info_affiliations",
                )?)?;
                let mut normalized_affiliation_list: Vec<String> =
                    Vec::with_capacity(affiliation_list.len());
                for affiliation in affiliation_list.iter() {
                    ensure_unidecode_for_text(&unidecode, affiliation, &mut unidecode_char_map)?;
                    normalized_affiliation_list.push(normalize_text_compat_from_map(
                        affiliation,
                        false,
                        &unidecode_char_map,
                    ));
                }
                let prefiltered = prefilter_affiliation_text(
                    &normalized_affiliation_list,
                    &affiliation_stopwords,
                );
                if !prefiltered.is_empty() {
                    affiliation_text_for_compute = Some(prefiltered);
                }
            }
            let adv_name = first.clone();

            signature_rows.push((
                sig_id,
                SignatureData {
                    first,
                    middle,
                    last_normalized,
                    orcid,
                    email,
                    affiliations,
                    coauthor_blocks,
                    coauthor_ngrams,
                    coauthors,
                    position,
                    paper_id,
                    name_counts,
                    adv_name,
                },
                coauthor_text_for_compute,
                affiliation_text_for_compute,
            ));
        }

        let computed_rows = py.allow_threads(|| {
            let compute = || {
                signature_rows
                    .into_par_iter()
                    .map(
                        |(
                            sig_id,
                            mut signature,
                            coauthor_text_for_compute,
                            affiliation_text_for_compute,
                        )| {
                            if signature.coauthor_ngrams.is_none() {
                                if let Some(coauthor_text) = coauthor_text_for_compute.as_deref() {
                                    signature.coauthor_ngrams = counter_data_from_usize_map(
                                        char_ngrams_counter(coauthor_text),
                                    );
                                }
                            }
                            if signature.affiliations.is_none() {
                                if let Some(affiliation_text) =
                                    affiliation_text_for_compute.as_deref()
                                {
                                    signature.affiliations = counter_data_from_usize_map(
                                        word_ngrams_counter(affiliation_text),
                                    );
                                }
                            }
                            (sig_id, signature)
                        },
                    )
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let mut signatures = HashMap::with_capacity(computed_rows.len());
        for (sig_id, signature) in computed_rows {
            signatures.insert(sig_id, signature);
        }
        let mut signature_ids: Vec<String> = signatures.keys().cloned().collect();
        signature_ids.sort_unstable();

        let name_tuples = extract_name_tuples_map(&dataset.getattr("name_tuples")?)?;
        let cluster_seeds_disallow = extract_pair_set(&dataset.getattr("cluster_seeds_disallow")?)?;
        let cluster_seeds_require =
            extract_cluster_seeds_require(&dataset.getattr("cluster_seeds_require")?)?;

        Ok(RustFeaturizer {
            signatures,
            signature_ids,
            papers,
            name_tuples,
            cluster_seeds_disallow,
            cluster_seeds_require,
            compute_reference_features,
            cluster_seed_require_value,
            cluster_seed_disallow_value,
            cached_signature_id_order: OnceLock::new(),
            cluster_seeds_disallow_index: OnceLock::new(),
        })
    }

    #[staticmethod]
    #[pyo3(
        signature = (
            signatures_path,
            papers_path,
            cluster_seeds_path = None,
            specter_embeddings = None,
            name_tuples_path = None,
            name_counts_path = None,
            preprocess = true,
            compute_reference_features = false,
            cluster_seed_require_value = 0.0,
            cluster_seed_disallow_value = 10000.0,
            num_threads = None,
            expected_normalization_version = None,
            allow_normalization_version_mismatch = false
        )
    )]
    fn from_json_paths(
        py: Python<'_>,
        signatures_path: &str,
        papers_path: &str,
        cluster_seeds_path: Option<&str>,
        specter_embeddings: Option<&Bound<'_, PyAny>>,
        name_tuples_path: Option<&str>,
        name_counts_path: Option<&str>,
        preprocess: bool,
        compute_reference_features: bool,
        cluster_seed_require_value: f64,
        cluster_seed_disallow_value: f64,
        num_threads: Option<usize>,
        expected_normalization_version: Option<&str>,
        allow_normalization_version_mismatch: bool,
    ) -> PyResult<Self> {
        reset_last_json_ingest_telemetry_internal();

        let text_module = py.import("s2and.text")?;
        let unidecode = text_module.getattr("unidecode")?;
        let stop_words_obj = text_module.getattr("STOPWORDS")?;
        let venue_stop_words_obj = text_module.getattr("VENUE_STOP_WORDS")?;
        let name_prefixes_obj = text_module.getattr("NAME_PREFIXES")?;

        let stop_words = extract_required_string_set(&stop_words_obj)?;
        let venue_stop_words = extract_required_string_set(&venue_stop_words_obj)?;
        let name_prefixes = extract_required_string_set(&name_prefixes_obj)?;

        let language_detector = if preprocess {
            Some(LanguageDetectorCompat::new(py))
        } else {
            None
        };

        let json_parse_start = Instant::now();
        let signatures_json = load_json_value(signatures_path)?;
        let signatures_obj = json_as_object(&signatures_json, "signatures payload")?;
        let raw_name_counts = load_raw_name_counts_from_json_path(
            name_counts_path,
            expected_normalization_version,
            allow_normalization_version_mismatch,
        )?;
        let name_tuples = load_name_tuples_from_text_path(py, name_tuples_path)?;
        let affiliation_stopwords = extract_affiliation_stopwords(py)?;

        #[derive(Clone)]
        struct SignatureInput {
            sig_id: String,
            paper_id: PaperId,
            raw_first: String,
            raw_middle: String,
            raw_last: String,
            email: Option<String>,
            position: i64,
            affiliation_values: Vec<String>,
            source_id_source: Option<String>,
            source_ids: Vec<String>,
        }

        #[derive(Clone)]
        struct PaperInput {
            paper_id: PaperId,
            raw_title: String,
            raw_venue: String,
            raw_journal: String,
            raw_authors: Vec<(i64, String)>,
            references: HashSet<PaperId>,
            year: Option<i64>,
            has_abstract: bool,
            predicted_language: Option<String>,
            is_reliable: bool,
        }

        let paper_preprocess_start = Instant::now();
        let mut needed_paper_ids: HashSet<PaperId> = HashSet::new();
        let mut signature_inputs: Vec<SignatureInput> = Vec::with_capacity(signatures_obj.len());
        let mut paper_inputs: Vec<PaperInput> = Vec::new();
        let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
        for (sig_id, sig_value) in signatures_obj.iter() {
            let sig_dict = json_as_object(sig_value, "signature entry")?;
            let paper_id_value = json_get_required(sig_dict, "paper_id", "signature entry")?;
            let paper_id = json_value_to_id(paper_id_value).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("signature paper_id must be string/int")
            })?;
            needed_paper_ids.insert(paper_id.clone());

            let author_info_value = json_get_required(sig_dict, "author_info", "signature entry")?;
            let author_info = json_as_object(author_info_value, "signature author_info")?;
            let raw_first = json_get_string(author_info, "first", "");
            let raw_middle = json_get_string(author_info, "middle", "");
            let raw_last = json_get_string(author_info, "last", "");
            let email = json_get_optional_string(author_info, "email");
            let position = json_get_i64_optional(author_info, "position").unwrap_or(0);
            let affiliation_values = json_get_string_list(author_info.get("affiliations"));
            let source_id_source = json_get_optional_string(author_info, "source_id_source");
            let source_ids = if source_id_source.as_deref() == Some("ORCID") {
                json_get_string_list(author_info.get("source_ids"))
            } else {
                Vec::new()
            };

            ensure_unidecode_for_text(&unidecode, &raw_first, &mut unidecode_char_map)?;
            ensure_unidecode_for_text(&unidecode, &raw_middle, &mut unidecode_char_map)?;
            ensure_unidecode_for_text(&unidecode, &raw_last, &mut unidecode_char_map)?;
            for affiliation in affiliation_values.iter() {
                ensure_unidecode_for_text(&unidecode, affiliation, &mut unidecode_char_map)?;
            }

            signature_inputs.push(SignatureInput {
                sig_id: sig_id.clone(),
                paper_id,
                raw_first,
                raw_middle,
                raw_last,
                email,
                position,
                affiliation_values,
                source_id_source,
                source_ids,
            });
        }
        drop(signatures_json);

        #[derive(Clone)]
        struct PaperPreprocessed {
            title: String,
            venue: String,
            journal_name: String,
            authors: Vec<(i64, String)>,
            references: HashSet<PaperId>,
            year: Option<i64>,
            has_abstract: bool,
            predicted_language: Option<String>,
            is_reliable: bool,
            title_words: Option<CounterData>,
            title_chars: Option<CounterData>,
            venue_ngrams: Option<CounterData>,
            journal_ngrams: Option<CounterData>,
        }

        let papers_json = load_json_value(papers_path)?;
        let papers_obj = json_as_object(&papers_json, "papers payload")?;
        for (_paper_key, paper_value) in papers_obj.iter() {
            let paper_dict = json_as_object(paper_value, "paper entry")?;
            let paper_id_value = json_get_required(paper_dict, "paper_id", "paper entry")?;
            let paper_id = json_value_to_id(paper_id_value).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("paper paper_id must be string/int")
            })?;
            if !needed_paper_ids.contains(&paper_id) {
                continue;
            }

            let raw_title = json_get_string(paper_dict, "title", "");
            let raw_venue = json_get_string(paper_dict, "venue", "");
            let raw_journal = json_get_string(paper_dict, "journal_name", "");

            if preprocess {
                ensure_unidecode_for_text(&unidecode, &raw_title, &mut unidecode_char_map)?;
                ensure_unidecode_for_text(&unidecode, &raw_venue, &mut unidecode_char_map)?;
                ensure_unidecode_for_text(&unidecode, &raw_journal, &mut unidecode_char_map)?;
            }

            let mut raw_authors: Vec<(i64, String)> = Vec::new();
            if let Some(author_values) = paper_dict.get("authors").and_then(JsonValue::as_array) {
                for author_value in author_values {
                    let Some(author_dict) = author_value.as_object() else {
                        continue;
                    };
                    let position = json_get_i64_optional(author_dict, "position").unwrap_or(0);
                    let raw_author_name = json_get_string(author_dict, "author_name", "");
                    if preprocess {
                        ensure_unidecode_for_text(
                            &unidecode,
                            &raw_author_name,
                            &mut unidecode_char_map,
                        )?;
                    }
                    raw_authors.push((position, raw_author_name));
                }
            }

            let references = json_get_id_set(paper_dict.get("references"));

            let year = match json_get_i64_optional(paper_dict, "year") {
                Some(v) if v > 0 => Some(v),
                _ => None,
            };

            let has_abstract = match paper_dict.get("abstract") {
                None | Some(JsonValue::Null) => false,
                Some(JsonValue::String(s)) => !s.is_empty(),
                Some(_) => true,
            };

            let (is_reliable, predicted_language) = if preprocess {
                let detector = language_detector.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("missing language detector")
                })?;
                let (reliable, _is_english, language) = detector.detect(&raw_title);
                (reliable, Some(language))
            } else {
                (false, None)
            };

            paper_inputs.push(PaperInput {
                paper_id,
                raw_title,
                raw_venue,
                raw_journal,
                raw_authors,
                references,
                year,
                has_abstract,
                predicted_language,
                is_reliable,
            });
        }
        drop(needed_paper_ids);
        drop(papers_json);
        let json_parse_seconds = json_parse_start.elapsed().as_secs_f64();

        let computed_papers = py.allow_threads(|| {
            let compute = || {
                paper_inputs
                    .par_iter()
                    .map(|paper_input| {
                        let title = if preprocess {
                            normalize_text_compat_from_map(
                                &paper_input.raw_title,
                                false,
                                &unidecode_char_map,
                            )
                        } else {
                            paper_input.raw_title.clone()
                        };
                        let venue = if preprocess {
                            normalize_text_compat_from_map(
                                &paper_input.raw_venue,
                                false,
                                &unidecode_char_map,
                            )
                        } else {
                            paper_input.raw_venue.clone()
                        };
                        let journal_name = if preprocess {
                            normalize_text_compat_from_map(
                                &paper_input.raw_journal,
                                false,
                                &unidecode_char_map,
                            )
                        } else {
                            paper_input.raw_journal.clone()
                        };
                        let authors = if preprocess {
                            paper_input
                                .raw_authors
                                .iter()
                                .map(|(position, raw_name)| {
                                    (
                                        *position,
                                        normalize_text_compat_from_map(
                                            raw_name,
                                            false,
                                            &unidecode_char_map,
                                        ),
                                    )
                                })
                                .collect::<Vec<_>>()
                        } else {
                            paper_input.raw_authors.clone()
                        };

                        let title_words = if preprocess {
                            counter_data_from_usize_map(word_ngrams_counter_python_compat(
                                &title,
                                &stop_words,
                            ))
                        } else {
                            None
                        };

                        let title_chars = if preprocess {
                            counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                &title,
                                false,
                                true,
                                Some(&stop_words),
                            ))
                        } else {
                            None
                        };

                        let venue_ngrams = if preprocess {
                            counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                &venue,
                                false,
                                true,
                                Some(&venue_stop_words),
                            ))
                        } else {
                            None
                        };

                        let journal_ngrams = if preprocess {
                            counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                &journal_name,
                                false,
                                true,
                                Some(&venue_stop_words),
                            ))
                        } else {
                            None
                        };

                        (
                            paper_input.paper_id.clone(),
                            PaperPreprocessed {
                                title,
                                venue,
                                journal_name,
                                authors,
                                references: paper_input.references.clone(),
                                year: paper_input.year,
                                has_abstract: paper_input.has_abstract,
                                predicted_language: paper_input.predicted_language.clone(),
                                is_reliable: paper_input.is_reliable,
                                title_words,
                                title_chars,
                                venue_ngrams,
                                journal_ngrams,
                            },
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        drop(paper_inputs);

        let mut preprocessed_papers: HashMap<PaperId, PaperPreprocessed> =
            HashMap::with_capacity(computed_papers.len());
        for (paper_id, preprocessed) in computed_papers {
            preprocessed_papers.insert(paper_id, preprocessed);
        }
        let paper_preprocess_seconds = paper_preprocess_start.elapsed().as_secs_f64();

        let signature_preprocess_start = Instant::now();
        let signature_inputs: Vec<SignatureInput> = signature_inputs
            .into_iter()
            .filter(|entry| preprocessed_papers.contains_key(&entry.paper_id))
            .collect();
        let computed_signatures = py.allow_threads(|| {
            let compute = || {
                signature_inputs
                    .par_iter()
                    .map(|entry| {
                        let middle_normalized = normalize_text_compat_from_map(
                            &entry.raw_middle,
                            false,
                            &unidecode_char_map,
                        );
                        let first_normalized = normalize_text_compat_from_map(
                            &entry.raw_first,
                            false,
                            &unidecode_char_map,
                        );
                        let mut first_middle_split: Vec<String> =
                            format!("{} {}", first_normalized, middle_normalized)
                                .split_whitespace()
                                .map(|token| token.to_string())
                                .collect();
                        if let Some(prefix) = first_middle_split.first() {
                            if name_prefixes.contains(prefix) {
                                first_middle_split.remove(0);
                            }
                        }
                        let first_normalized_token =
                            first_middle_split.get(0).cloned().unwrap_or_default();
                        let (first_without_apostrophe, middle_without_apostrophe) =
                            split_first_middle_hyphen_aware_compat(
                                &entry.raw_first,
                                &entry.raw_middle,
                                &name_prefixes,
                                &unidecode_char_map,
                            );
                        let last_normalized = normalize_text_compat_from_map(
                            &entry.raw_last,
                            false,
                            &unidecode_char_map,
                        );

                        let mut coauthor_list: Vec<String> = Vec::new();
                        if let Some(preprocessed_paper) = preprocessed_papers.get(&entry.paper_id) {
                            for (author_position, author_name) in preprocessed_paper.authors.iter()
                            {
                                if *author_position != entry.position {
                                    coauthor_list.push(author_name.clone());
                                }
                            }
                        }
                        let coauthors = if coauthor_list.is_empty() {
                            None
                        } else {
                            Some(coauthor_list.iter().cloned().collect::<HashSet<String>>())
                        };

                        let mut coauthor_blocks_set: HashSet<String> = HashSet::new();
                        for coauthor in coauthor_list.iter() {
                            coauthor_blocks_set.insert(compute_block_compat(coauthor));
                        }
                        let coauthor_blocks = if coauthor_blocks_set.is_empty() {
                            None
                        } else {
                            Some(coauthor_blocks_set)
                        };

                        let normalized_affiliations: Vec<String> = if preprocess {
                            entry
                                .affiliation_values
                                .iter()
                                .map(|affiliation| {
                                    normalize_text_compat_from_map(
                                        affiliation,
                                        false,
                                        &unidecode_char_map,
                                    )
                                })
                                .collect()
                        } else {
                            entry.affiliation_values.clone()
                        };

                        let affiliation_text = if preprocess {
                            prefilter_affiliation_text(
                                &normalized_affiliations,
                                &affiliation_stopwords,
                            )
                        } else {
                            String::new()
                        };
                        let coauthor_text = if preprocess {
                            coauthor_list.join(" ")
                        } else {
                            String::new()
                        };

                        let affiliations = if preprocess && !affiliation_text.is_empty() {
                            counter_data_from_usize_map(word_ngrams_counter(&affiliation_text))
                        } else {
                            None
                        };
                        let coauthor_ngrams = if preprocess && !coauthor_text.is_empty() {
                            counter_data_from_usize_map(char_ngrams_counter(&coauthor_text))
                        } else {
                            None
                        };

                        let normalized_orcid = if entry.source_id_source.as_deref() == Some("ORCID")
                        {
                            entry
                                .source_ids
                                .first()
                                .and_then(|source_id| extract_orcid_from_source_id(source_id))
                        } else {
                            None
                        };

                        let name_counts = build_name_counts_data_from_artifact(
                            &raw_name_counts,
                            &entry.raw_first,
                            &first_normalized_token,
                            &first_without_apostrophe,
                            &entry.raw_last,
                            &last_normalized,
                        );

                        (
                            entry.sig_id.clone(),
                            SignatureData {
                                first: Some(first_without_apostrophe.clone()),
                                middle: Some(middle_without_apostrophe),
                                last_normalized: Some(last_normalized),
                                orcid: normalized_orcid,
                                email: entry.email.clone(),
                                affiliations,
                                coauthor_blocks,
                                coauthor_ngrams,
                                coauthors,
                                position: entry.position,
                                paper_id: entry.paper_id.clone(),
                                name_counts,
                                adv_name: Some(first_without_apostrophe),
                            },
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        let mut signatures: HashMap<String, SignatureData> =
            HashMap::with_capacity(computed_signatures.len());
        for (sig_id, signature) in computed_signatures {
            signatures.insert(sig_id, signature);
        }
        drop(signature_inputs);
        drop(unidecode_char_map);
        let mut signature_ids: Vec<String> = signatures.keys().cloned().collect();
        signature_ids.sort_unstable();
        let signature_preprocess_seconds = signature_preprocess_start.elapsed().as_secs_f64();

        let specter_dict = match specter_embeddings {
            Some(obj) => {
                if let Ok(path) = obj.extract::<&str>() {
                    load_pickle_dict(py, path)?
                } else if let Ok(dict) = obj.downcast::<PyDict>() {
                    Some(dict.clone())
                } else if obj.is_none() {
                    None
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "specter_embeddings must be str, dict, or None; got {}",
                        obj.get_type().name()?
                    )));
                }
            }
            None => None,
        };

        let reference_counter_start = Instant::now();
        let mut papers: HashMap<PaperId, PaperData> =
            HashMap::with_capacity(preprocessed_papers.len());
        if compute_reference_features {
            for (paper_id, paper) in preprocessed_papers.iter() {
                let mut ref_authors = None;
                let mut ref_titles = None;
                let mut ref_venues = None;
                let mut ref_blocks = None;
                let ref_details_present = true;

                let mut titles: Vec<String> = Vec::new();
                let mut venues: Vec<String> = Vec::new();
                let mut journals: Vec<String> = Vec::new();
                let mut authors: Vec<String> = Vec::new();
                let mut blocks: Vec<String> = Vec::new();

                for reference_id in paper.references.iter() {
                    if let Some(reference_paper) = preprocessed_papers.get(reference_id) {
                        if !reference_paper.title.is_empty() {
                            titles.push(reference_paper.title.clone());
                        }
                        if !reference_paper.venue.is_empty() {
                            venues.push(reference_paper.venue.clone());
                        }
                        if !reference_paper.journal_name.is_empty() {
                            journals.push(reference_paper.journal_name.clone());
                        }
                        for (_, author_name) in reference_paper.authors.iter() {
                            if author_name.is_empty() {
                                continue;
                            }
                            authors.push(author_name.clone());
                            let block = compute_block_compat(author_name);
                            blocks.push(block);
                        }
                    }
                }

                let author_names = authors.join(" ");
                let reference_titles = titles.join(" ");
                let venues_joined = venues.join(" ");
                let journals_joined = journals.join(" ");
                let reference_venues = if venues_joined == journals_joined {
                    venues_joined
                } else {
                    format!("{} {}", venues_joined, journals_joined)
                        .trim()
                        .to_string()
                };

                if !author_names.is_empty() {
                    ref_authors = counter_data_from_usize_map(char_ngrams_counter_python_compat(
                        &author_names,
                        false,
                        true,
                        Some(&stop_words),
                    ));
                }

                if !reference_titles.is_empty() {
                    ref_titles = counter_data_from_usize_map(char_ngrams_counter_python_compat(
                        &reference_titles,
                        false,
                        true,
                        Some(&stop_words),
                    ));
                }

                if !reference_venues.is_empty() {
                    ref_venues = counter_data_from_usize_map(char_ngrams_counter_python_compat(
                        &reference_venues,
                        false,
                        true,
                        Some(&venue_stop_words),
                    ));
                }

                if !blocks.is_empty() {
                    let mut block_counter: HashMap<String, usize> = HashMap::new();
                    for block in blocks {
                        *block_counter.entry(block).or_insert(0) += 1;
                    }
                    ref_blocks = counter_data_from_usize_map(block_counter);
                }
                let specter = if let Some(spec_dict) = &specter_dict {
                    extract_specter_for_paper_id(spec_dict, paper_id)?
                } else {
                    None
                };
                let specter_norm = specter.as_ref().map(|values| {
                    values
                        .iter()
                        .map(|value| {
                            let value_f64 = *value as f64;
                            value_f64 * value_f64
                        })
                        .sum::<f64>()
                        .sqrt()
                });

                papers.insert(
                    paper_id.clone(),
                    PaperData {
                        venue_ngrams: paper.venue_ngrams.clone(),
                        title_words: paper.title_words.clone(),
                        title_chars: paper.title_chars.clone(),
                        ref_authors,
                        ref_titles,
                        ref_venues,
                        ref_blocks,
                        ref_details_present,
                        references: paper.references.clone(),
                        year: paper.year,
                        has_abstract: paper.has_abstract,
                        predicted_language: paper.predicted_language.clone(),
                        is_reliable: paper.is_reliable,
                        journal_ngrams: paper.journal_ngrams.clone(),
                        specter,
                        specter_norm,
                    },
                );
            }
        } else {
            for (paper_id, paper) in preprocessed_papers.into_iter() {
                let specter = if let Some(spec_dict) = &specter_dict {
                    extract_specter_for_paper_id(spec_dict, &paper_id)?
                } else {
                    None
                };
                let specter_norm = specter.as_ref().map(|values| {
                    values
                        .iter()
                        .map(|value| {
                            let value_f64 = *value as f64;
                            value_f64 * value_f64
                        })
                        .sum::<f64>()
                        .sqrt()
                });
                papers.insert(
                    paper_id,
                    PaperData {
                        venue_ngrams: paper.venue_ngrams,
                        title_words: paper.title_words,
                        title_chars: paper.title_chars,
                        ref_authors: None,
                        ref_titles: None,
                        ref_venues: None,
                        ref_blocks: None,
                        ref_details_present: false,
                        references: paper.references,
                        year: paper.year,
                        has_abstract: paper.has_abstract,
                        predicted_language: paper.predicted_language,
                        is_reliable: paper.is_reliable,
                        journal_ngrams: paper.journal_ngrams,
                        specter,
                        specter_norm,
                    },
                );
            }
        }
        let reference_counter_seconds = reference_counter_start.elapsed().as_secs_f64();

        let cluster_seed_start = Instant::now();
        let mut cluster_seeds_disallow: HashSet<(String, String)> = HashSet::new();
        let mut cluster_seeds_require: HashMap<String, ClusterId> = HashMap::new();
        if let Some(path) = cluster_seeds_path {
            let cluster_seeds_json = load_json_value(path)?;
            let cluster_seeds_obj = json_as_object(&cluster_seeds_json, "cluster_seeds payload")?;
            let mut cluster_num = 0_i64;
            for (signature_id_a, values_value) in cluster_seeds_obj.iter() {
                let values_obj = json_as_object(values_value, "cluster seed entry")?;
                let mut root_added = false;
                for (signature_id_b, constraint_value) in values_obj.iter() {
                    let Some(constraint) = json_value_to_string(constraint_value) else {
                        continue;
                    };
                    if constraint == "disallow" {
                        cluster_seeds_disallow.insert(canonical_signature_pair_cloned(
                            signature_id_a,
                            signature_id_b,
                        ));
                    } else if constraint == "require" {
                        if !root_added {
                            cluster_seeds_require
                                .insert(signature_id_a.clone(), ClusterId::Int(cluster_num));
                            root_added = true;
                        }
                        cluster_seeds_require
                            .insert(signature_id_b.clone(), ClusterId::Int(cluster_num));
                    }
                }
                cluster_num += 1;
            }
        }
        let cluster_seed_seconds = cluster_seed_start.elapsed().as_secs_f64();

        set_last_json_ingest_telemetry(JsonIngestTelemetry {
            json_parse_seconds,
            paper_preprocess_seconds,
            reference_counter_seconds,
            signature_preprocess_seconds,
            cluster_seed_seconds,
        });

        Ok(RustFeaturizer {
            signatures,
            signature_ids,
            papers,
            name_tuples,
            cluster_seeds_disallow,
            cluster_seeds_require,
            compute_reference_features,
            cluster_seed_require_value,
            cluster_seed_disallow_value,
            cached_signature_id_order: OnceLock::new(),
            cluster_seeds_disallow_index: OnceLock::new(),
        })
    }

    fn update_cluster_seeds(
        &mut self,
        cluster_seeds_require: &Bound<'_, PyAny>,
        cluster_seeds_disallow: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.cluster_seeds_require = extract_cluster_seeds_require(cluster_seeds_require)?;
        self.cluster_seeds_disallow = extract_pair_set(cluster_seeds_disallow)?;
        self.cluster_seeds_disallow_index = OnceLock::new();
        Ok(())
    }

    #[pyo3(
        signature = (
            sig_id1,
            sig_id2,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false
        )
    )]
    fn get_constraint(
        &self,
        sig_id1: &str,
        sig_id2: &str,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
    ) -> PyResult<Option<f64>> {
        self.get_constraint_value_for_pair(
            sig_id1,
            sig_id2,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
        )
    }

    #[pyo3(
        signature = (
            pairs,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None
        )
    )]
    fn get_constraints_matrix(
        &self,
        py: Python<'_>,
        pairs: Vec<(String, String)>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<Option<f64>>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        for (sig_id1, sig_id2) in pairs.iter() {
            self.validate_constraint_pair_inputs(sig_id1, sig_id2)?;
        }

        let values = py.allow_threads(|| {
            let compute = || {
                pairs
                    .par_iter()
                    .map(|(sig_id1, sig_id2)| -> PyResult<Option<f64>> {
                        let s1 = self.signatures.get(sig_id1).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string())
                        })?;
                        let s2 = self.signatures.get(sig_id2).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string())
                        })?;
                        let p1 = self.papers.get(&s1.paper_id).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(s1.paper_id.to_string())
                        })?;
                        let p2 = self.papers.get(&s2.paper_id).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(s2.paper_id.to_string())
                        })?;
                        Ok(self.constraint_value_from_records(
                            sig_id1,
                            sig_id2,
                            s1,
                            s2,
                            p1,
                            p2,
                            low_value,
                            high_value,
                            dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds,
                        ))
                    })
                    .collect::<PyResult<Vec<_>>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        values
    }

    #[pyo3(
        signature = (
            pairs,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None
        )
    )]
    fn get_constraints_matrix_indexed(
        &self,
        py: Python<'_>,
        pairs: Vec<(u32, u32)>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<Option<f64>>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        for (left_idx, right_idx) in pairs.iter() {
            let left = *left_idx as usize;
            let right = *right_idx as usize;
            if left >= signature_count || right >= signature_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "pair index out of range: left={} right={} signature_count={}",
                    left, right, signature_count
                )));
            }
        }

        let mut lookup: Vec<(&String, &SignatureData, &PaperData)> =
            Vec::with_capacity(signature_count);
        for signature_id in signature_ids.iter() {
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            lookup.push((signature_id, signature, paper));
        }

        let values = py.allow_threads(|| {
            let compute = || {
                pairs
                    .par_iter()
                    .map(|(left_idx, right_idx)| {
                        let (left_id, s1, p1) = lookup[*left_idx as usize];
                        let (right_id, s2, p2) = lookup[*right_idx as usize];
                        self.constraint_value_from_records(
                            left_id,
                            right_id,
                            s1,
                            s2,
                            p1,
                            p2,
                            low_value,
                            high_value,
                            dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        Ok(values)
    }

    #[pyo3(
        signature = (
            block_signature_indices,
            start_offset = 0,
            max_pairs = None,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None
        )
    )]
    fn get_constraints_block_upper_triangle_indexed(
        &self,
        py: Python<'_>,
        block_signature_indices: Vec<u32>,
        start_offset: usize,
        max_pairs: Option<usize>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<Option<f64>>)> {
        if block_signature_indices.len() <= 1 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            if global_idx >= signature_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "block signature index out of range: index={} signature_count={}",
                    global_idx, signature_count
                )));
            }
        }

        let mut block_lookup: Vec<(&String, &SignatureData, &PaperData)> =
            Vec::with_capacity(block_signature_indices.len());
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            let signature_id = &signature_ids[global_idx];
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            block_lookup.push((signature_id, signature, paper));
        }

        let local_pairs =
            upper_triangle_pairs_for_range(block_lookup.len(), start_offset, max_pairs)?;
        if local_pairs.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let left_indices: Vec<u32> = local_pairs.iter().map(|(left, _)| *left as u32).collect();
        let right_indices: Vec<u32> = local_pairs.iter().map(|(_, right)| *right as u32).collect();
        let values = py.allow_threads(|| {
            let compute = || {
                local_pairs
                    .par_iter()
                    .map(|(left_idx, right_idx)| {
                        let (left_id, s1, p1) = block_lookup[*left_idx];
                        let (right_id, s2, p2) = block_lookup[*right_idx];
                        self.constraint_value_from_records(
                            left_id,
                            right_id,
                            s1,
                            s2,
                            p1,
                            p2,
                            low_value,
                            high_value,
                            dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        Ok((left_indices, right_indices, values))
    }

    fn featurize_pair(&self, sig_id1: &str, sig_id2: &str) -> PyResult<Vec<f64>> {
        let s1 = self
            .signatures
            .get(sig_id1)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string()))?;
        let s2 = self
            .signatures
            .get(sig_id2)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string()))?;
        let p1 = self
            .papers
            .get(&s1.paper_id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(s1.paper_id.to_string()))?;
        let p2 = self
            .papers
            .get(&s2.paper_id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(s2.paper_id.to_string()))?;
        Ok(self.featurize_pair_data(s1, s2, p1, p2).to_vec())
    }

    #[pyo3(signature = (pairs, num_threads = None))]
    fn featurize_pairs(
        &self,
        py: Python<'_>,
        pairs: Vec<(String, String)>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<Vec<f64>>> {
        for (sig_id1, sig_id2) in pairs.iter() {
            let s1 = self
                .signatures
                .get(sig_id1)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string()))?;
            let s2 = self
                .signatures
                .get(sig_id2)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string()))?;
            if self.papers.get(&s1.paper_id).is_none() {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    s1.paper_id.to_string(),
                ));
            }
            if self.papers.get(&s2.paper_id).is_none() {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    s2.paper_id.to_string(),
                ));
            }
        }
        let feats = py.allow_threads(|| {
            let compute = || {
                pairs
                    .par_iter()
                    .map(|(sig_id1, sig_id2)| -> PyResult<Vec<f64>> {
                        let s1 = self.signatures.get(sig_id1).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string())
                        })?;
                        let s2 = self.signatures.get(sig_id2).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string())
                        })?;
                        let p1 = self.papers.get(&s1.paper_id).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(s1.paper_id.to_string())
                        })?;
                        let p2 = self.papers.get(&s2.paper_id).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(s2.paper_id.to_string())
                        })?;
                        Ok(self.featurize_pair_data(s1, s2, p1, p2).to_vec())
                    })
                    .collect::<PyResult<Vec<_>>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        feats
    }

    fn signature_ids(&self) -> Vec<String> {
        self.signature_id_order().to_vec()
    }

    fn update_signature_name_counts(&mut self, signatures: &Bound<'_, PyAny>) -> PyResult<usize> {
        let signatures_dict = signatures.downcast::<PyDict>()?;
        let mut updated = 0usize;
        for (sig_id_obj, sig_obj) in signatures_dict.iter() {
            let sig_id: String = sig_id_obj.extract()?;
            let Some(signature) = self.signatures.get_mut(&sig_id) else {
                continue;
            };
            let counts_obj = sig_obj.getattr("author_info_name_counts")?;
            let counts = extract_name_counts_data(&counts_obj)?;
            if counts.is_some() {
                signature.name_counts = counts;
                updated += 1;
            }
        }
        Ok(updated)
    }

    #[pyo3(signature = (pairs, selected_indices = None, num_threads = None, nan_value = f64::NAN))]
    fn featurize_pairs_matrix<'py>(
        &self,
        py: Python<'py>,
        pairs: Vec<(String, String)>,
        selected_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let row_count = pairs.len();
        if row_count == 0 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

        for (sig_id1, sig_id2) in pairs.iter() {
            let s1 = self
                .signatures
                .get(sig_id1)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string()))?;
            let s2 = self
                .signatures
                .get(sig_id2)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string()))?;
            if self.papers.get(&s1.paper_id).is_none() {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    s1.paper_id.to_string(),
                ));
            }
            if self.papers.get(&s2.paper_id).is_none() {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    s2.paper_id.to_string(),
                ));
            }
        }

        let mut id_to_lookup_idx: HashMap<&str, usize> =
            HashMap::with_capacity(row_count.saturating_mul(2));
        let mut lookup: Vec<(&SignatureData, &PaperData)> = Vec::new();
        for (sig_id1, sig_id2) in pairs.iter() {
            for sig_id in [sig_id1.as_str(), sig_id2.as_str()] {
                if id_to_lookup_idx.contains_key(sig_id) {
                    continue;
                }
                let signature = self
                    .signatures
                    .get(sig_id)
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id.to_string()))?;
                let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
                })?;
                let lookup_idx = lookup.len();
                lookup.push((signature, paper));
                id_to_lookup_idx.insert(sig_id, lookup_idx);
            }
        }

        let mut indexed_pairs: Vec<(usize, usize)> = Vec::with_capacity(row_count);
        for (sig_id1, sig_id2) in pairs.iter() {
            let left_idx = *id_to_lookup_idx
                .get(sig_id1.as_str())
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id1.clone()))?;
            let right_idx = *id_to_lookup_idx
                .get(sig_id2.as_str())
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id2.clone()))?;
            indexed_pairs.push((left_idx, right_idx));
        }

        let full_cols = self.full_feature_count();
        let indices: Vec<usize> = selected_indices.unwrap_or_else(|| (0..full_cols).collect());
        for idx in indices.iter() {
            if *idx >= full_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "selected_indices contains out-of-range index {} for {} columns",
                    idx, full_cols
                )));
            }
        }
        let out_cols = indices.len();
        if out_cols == 0 {
            let empty_cols = numpy::ndarray::Array2::<f64>::zeros((row_count, 0));
            return Ok(empty_cols.to_pyarray(py));
        }
        let out = py.allow_threads(|| {
            let compute = || {
                let mut buffer = vec![0.0_f64; row_count * out_cols];
                buffer
                    .par_chunks_mut(out_cols)
                    .zip(indexed_pairs.par_iter())
                    .for_each(|(out_row, (left_idx, right_idx))| {
                        let (s1, p1) = lookup[*left_idx];
                        let (s2, p2) = lookup[*right_idx];
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (dest, idx) in out_row.iter_mut().zip(indices.iter()) {
                            let mut value = row[*idx];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            *dest = value;
                        }
                    });
                buffer
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let array =
            numpy::ndarray::Array2::from_shape_vec((row_count, out_cols), out).map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build output matrix: {}",
                    err
                ))
            })?;
        Ok(array.to_pyarray(py))
    }

    #[pyo3(signature = (pairs, selected_indices = None, num_threads = None, nan_value = f64::NAN))]
    fn featurize_pairs_matrix_indexed<'py>(
        &self,
        py: Python<'py>,
        pairs: Vec<(u32, u32)>,
        selected_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let row_count = pairs.len();
        if row_count == 0 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

        let signature_ids = self.signature_id_order();
        for (left_idx, right_idx) in pairs.iter() {
            let left = *left_idx as usize;
            let right = *right_idx as usize;
            if left >= signature_ids.len() || right >= signature_ids.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "pair index out of range: left={} right={} signature_count={}",
                    left,
                    right,
                    signature_ids.len()
                )));
            }
        }

        let mut lookup: Vec<(&SignatureData, &PaperData)> = Vec::with_capacity(signature_ids.len());
        for signature_id in signature_ids.iter() {
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            lookup.push((signature, paper));
        }

        let full_cols = self.full_feature_count();
        let indices: Vec<usize> = selected_indices.unwrap_or_else(|| (0..full_cols).collect());
        for idx in indices.iter() {
            if *idx >= full_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "selected_indices contains out-of-range index {} for {} columns",
                    idx, full_cols
                )));
            }
        }
        let out_cols = indices.len();
        if out_cols == 0 {
            let empty_cols = numpy::ndarray::Array2::<f64>::zeros((row_count, 0));
            return Ok(empty_cols.to_pyarray(py));
        }

        let out = py.allow_threads(|| {
            let compute = || {
                let mut buffer = vec![0.0_f64; row_count * out_cols];
                buffer
                    .par_chunks_mut(out_cols)
                    .zip(pairs.par_iter())
                    .for_each(|(out_row, (left_idx, right_idx))| {
                        let (s1, p1) = lookup[*left_idx as usize];
                        let (s2, p2) = lookup[*right_idx as usize];
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (dest, idx) in out_row.iter_mut().zip(indices.iter()) {
                            let mut value = row[*idx];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            *dest = value;
                        }
                    });
                buffer
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let array =
            numpy::ndarray::Array2::from_shape_vec((row_count, out_cols), out).map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build output matrix: {}",
                    err
                ))
            })?;
        Ok(array.to_pyarray(py))
    }

    #[pyo3(
        signature = (
            block_signature_indices,
            start_offset = 0,
            max_pairs = None,
            selected_indices = None,
            num_threads = None,
            nan_value = f64::NAN
        )
    )]
    fn featurize_block_upper_triangle_matrix_indexed<'py>(
        &self,
        py: Python<'py>,
        block_signature_indices: Vec<u32>,
        start_offset: usize,
        max_pairs: Option<usize>,
        selected_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if block_signature_indices.len() <= 1 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            if global_idx >= signature_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "block signature index out of range: index={} signature_count={}",
                    global_idx, signature_count
                )));
            }
        }

        let mut block_lookup: Vec<(&SignatureData, &PaperData)> =
            Vec::with_capacity(block_signature_indices.len());
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            let signature_id = &signature_ids[global_idx];
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            block_lookup.push((signature, paper));
        }

        let local_pairs =
            upper_triangle_pairs_for_range(block_lookup.len(), start_offset, max_pairs)?;
        let row_count = local_pairs.len();
        if row_count == 0 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

        let full_cols = self.full_feature_count();
        let indices: Vec<usize> = selected_indices.unwrap_or_else(|| (0..full_cols).collect());
        for idx in indices.iter() {
            if *idx >= full_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "selected_indices contains out-of-range index {} for {} columns",
                    idx, full_cols
                )));
            }
        }
        let out_cols = indices.len();
        if out_cols == 0 {
            let empty_cols = numpy::ndarray::Array2::<f64>::zeros((row_count, 0));
            return Ok(empty_cols.to_pyarray(py));
        }

        let out = py.allow_threads(|| {
            let compute = || {
                let mut buffer = vec![0.0_f64; row_count * out_cols];
                buffer
                    .par_chunks_mut(out_cols)
                    .zip(local_pairs.par_iter())
                    .for_each(|(out_row, (left_idx, right_idx))| {
                        let (s1, p1) = block_lookup[*left_idx];
                        let (s2, p2) = block_lookup[*right_idx];
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (dest, idx) in out_row.iter_mut().zip(indices.iter()) {
                            let mut value = row[*idx];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            *dest = value;
                        }
                    });
                buffer
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let array =
            numpy::ndarray::Array2::from_shape_vec((row_count, out_cols), out).map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build output matrix: {}",
                    err
                ))
            })?;
        Ok(array.to_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let file =
            File::create(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let file =
            File::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let reader = BufReader::new(file);
        let featurizer: RustFeaturizer = bincode::deserialize_from(reader)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(featurizer)
    }
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
    Ok(build_info.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyString;

    #[test]
    fn reference_details_extraction_errors_are_not_silenced() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let non_tuple = PyString::new(py, "not-a-tuple");
            let result = extract_reference_details_counters(py, non_tuple.as_any());
            assert!(result.is_err(), "non-tuple reference_details should raise");
            let err = result
                .err()
                .unwrap_or_else(|| unreachable!("assert above guarantees error"));
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
        });
    }
}

#[pymodule]
fn _s2and_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(get_build_info, m)?)?;
    m.add_function(wrap_pyfunction!(signature_ngrams_batch, m)?)?;
    m.add_function(wrap_pyfunction!(get_last_json_ingest_telemetry, m)?)?;
    m.add_function(wrap_pyfunction!(reset_last_json_ingest_telemetry, m)?)?;
    m.add_class::<RustFeaturizer>()?;
    Ok(())
}
