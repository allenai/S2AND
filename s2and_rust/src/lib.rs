use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::f64;
use std::fs::File;
use std::io::{BufReader, BufWriter};

fn py_len(s: &str) -> usize {
    // Python len() semantics for str: count of Unicode scalar values
    s.chars().count()
}

const DROPPED_AFFIXES: [&str; 48] = [
    "ab", "am", "ap", "abu", "al", "auf", "aus", "bar", "bath", "bat", "bet", "bint", "dall", "dalla", "das", "de",
    "degli", "del", "dell", "della", "dem", "den", "der", "di", "do", "dos", "ds", "du", "el", "ibn", "im", "jr", "la",
    "las", "le", "los", "mac", "mc", "mhic", "mic", "ter", "und", "van", "vom", "von", "zu", "zum", "zur",
];

fn is_dropped_affix(token: &str) -> bool {
    DROPPED_AFFIXES.contains(&token)
}

#[derive(Clone, Serialize, Deserialize)]
struct CounterData {
    map: HashMap<String, f64>,
    sum: f64,
}

#[derive(Clone, Serialize, Deserialize)]
struct NameCountsData {
    first: f64,
    first_last: f64,
    last: f64,
    last_first_initial: f64,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
enum ClusterId {
    Int(i64),
    Str(String),
}

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
    paper_id: i64,
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
    references: HashSet<i64>,
    year: Option<i64>,
    has_abstract: bool,
    predicted_language: Option<String>,
    is_reliable: bool,
    journal_ngrams: Option<CounterData>,
    specter: Option<Vec<f32>>,
}

#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
struct RustFeaturizer {
    signatures: HashMap<String, SignatureData>,
    papers: HashMap<i64, PaperData>,
    name_tuples: HashMap<String, HashSet<String>>,
    cluster_seeds_disallow: HashSet<(String, String)>,
    cluster_seeds_require: HashMap<String, ClusterId>,
    compute_reference_features: bool,
    cluster_seed_require_value: f64,
    cluster_seed_disallow_value: f64,
}

fn extract_counter(obj: &Bound<'_, PyAny>) -> PyResult<Option<CounterData>> {
    if obj.is_none() {
        return Ok(None);
    }
    let dict = obj.downcast::<PyDict>()?;
    if dict.len() == 0 {
        return Ok(None);
    }
    let mut map = HashMap::with_capacity(dict.len());
    let mut sum = 0.0;
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let val: f64 = v.extract()?;
        sum += val;
        map.insert(key, val);
    }
    Ok(Some(CounterData { map, sum }))
}

fn extract_set_str(obj: &Bound<'_, PyAny>) -> PyResult<Option<HashSet<String>>> {
    if obj.is_none() {
        return Ok(None);
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_bound_object(obj)? {
        let v: String = item?.extract()?;
        out.insert(v);
    }
    if out.is_empty() {
        Ok(None)
    } else {
        Ok(Some(out))
    }
}

fn extract_pair_set(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<(String, String)>> {
    if obj.is_none() {
        return Ok(HashSet::new());
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_bound_object(obj)? {
        let tuple = item?;
        let (a, b): (String, String) = tuple.extract()?;
        out.insert((a, b));
    }
    Ok(out)
}

fn extract_name_tuples_map(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, HashSet<String>>> {
    if obj.is_none() {
        return Ok(HashMap::new());
    }
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for item in PyIterator::from_bound_object(obj)? {
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

fn extract_set_i64(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<i64>> {
    if obj.is_none() {
        return Ok(HashSet::new());
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_bound_object(obj)? {
        let v: i64 = item?.extract()?;
        out.insert(v);
    }
    Ok(out)
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
        let slice = unsafe { arr.as_slice()? };
        let all_zero = slice.iter().all(|v| *v == 0.0);
        if all_zero {
            return Ok(None);
        }
        return Ok(Some(slice.to_vec()));
    }
    if let Ok(arr) = obj.downcast::<PyArray1<f64>>() {
        let slice = unsafe { arr.as_slice()? };
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

fn counter_jaccard(counter1: Option<&Bound<'_, PyAny>>, counter2: Option<&Bound<'_, PyAny>>, denom_max: f64) -> PyResult<f64> {
    if counter1.is_none() || counter2.is_none() {
        return Ok(f64::NAN);
    }
    let c1 = counter1.unwrap();
    let c2 = counter2.unwrap();
    let len1 = c1.len()?;
    let len2 = c2.len()?;
    if len1 == 0 || len2 == 0 {
        return Ok(f64::NAN);
    }

    let mut sum1: f64 = 0.0;
    let mut sum2: f64 = 0.0;
    let mut intersection: f64 = 0.0;

    let items1 = c1.call_method0("items")?;
    for item in PyIterator::from_bound_object(&items1)? {
        let tuple = item?;
        let key: String = tuple.get_item(0)?.extract()?;
        let v1_any = tuple.get_item(1)?;
        let v1_f: f64 = v1_any.extract()?;
        sum1 += v1_f;
        let v2 = c2.call_method1("get", (key, 0.0))?;
        let v2_f: f64 = v2.extract()?;
        intersection += v1_f.min(v2_f);
    }

    let values2 = c2.call_method0("values")?;
    for v in PyIterator::from_bound_object(&values2)? {
        let v_any = v?;
        let v_f: f64 = v_any.extract()?;
        sum2 += v_f;
    }

    let union = sum1 + sum2 - intersection;
    if union == 0.0 {
        return Ok(f64::NAN);
    }
    let denom = if denom_max.is_infinite() {
        union
    } else {
        union.min(denom_max)
    };
    let score = intersection / denom;
    Ok(if score > 1.0 { 1.0 } else { score })
}

fn set_jaccard(set1: Option<&Bound<'_, PyAny>>, set2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    if set1.is_none() || set2.is_none() {
        return Ok(f64::NAN);
    }
    let s1 = set1.unwrap();
    let s2 = set2.unwrap();
    let len1 = s1.len()? as i64;
    let len2 = s2.len()? as i64;
    if len1 == 0 || len2 == 0 {
        return Ok(f64::NAN);
    }
    let mut intersection: i64 = 0;
    for item in PyIterator::from_bound_object(s1)? {
        let obj = item?;
        if s2.contains(obj)? {
            intersection += 1;
        }
    }
    let union = len1 + len2 - intersection;
    if union == 0 {
        return Ok(f64::NAN);
    }
    Ok((intersection as f64) / (union as f64))
}

fn counter_jaccard_data(counter1: &Option<CounterData>, counter2: &Option<CounterData>, denom_max: f64) -> f64 {
    if counter1.is_none() || counter2.is_none() {
        return f64::NAN;
    }
    let c1 = counter1.as_ref().unwrap();
    let c2 = counter2.as_ref().unwrap();
    if c1.map.is_empty() || c2.map.is_empty() {
        return f64::NAN;
    }
    let (small, large) = if c1.map.len() <= c2.map.len() { (c1, c2) } else { (c2, c1) };
    let mut intersection = 0.0;
    for (k, v1) in small.map.iter() {
        if let Some(v2) = large.map.get(k) {
            intersection += v1.min(*v2);
        }
    }
    let union = c1.sum + c2.sum - intersection;
    if union == 0.0 {
        return f64::NAN;
    }
    let denom = if denom_max.is_infinite() {
        union
    } else {
        union.min(denom_max)
    };
    let score = intersection / denom;
    if score > 1.0 { 1.0 } else { score }
}

fn set_jaccard_data<T: Eq + std::hash::Hash>(set1: &Option<HashSet<T>>, set2: &Option<HashSet<T>>) -> f64 {
    if set1.is_none() || set2.is_none() {
        return f64::NAN;
    }
    let s1 = set1.as_ref().unwrap();
    let s2 = set2.as_ref().unwrap();
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

fn refs_jaccard(set1: &HashSet<i64>, set2: &HashSet<i64>) -> f64 {
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

fn extract_name_counts(counts: &Bound<'_, PyAny>) -> PyResult<[f64; 4]> {
    let first: Option<f64> = counts.getattr("first")?.extract()?;
    let first_last: Option<f64> = counts.getattr("first_last")?.extract()?;
    let last: Option<f64> = counts.getattr("last")?.extract()?;
    let last_first_initial: Option<f64> = counts.getattr("last_first_initial")?.extract()?;
    Ok([
        first.unwrap_or(f64::NAN),
        first_last.unwrap_or(f64::NAN),
        last.unwrap_or(f64::NAN),
        last_first_initial.unwrap_or(f64::NAN),
    ])
}

fn compute_name_counts(counts1: Option<&Bound<'_, PyAny>>, counts2: Option<&Bound<'_, PyAny>>) -> PyResult<[f64; 6]> {
    if counts1.is_none() || counts2.is_none() {
        return Ok([f64::NAN; 6]);
    }
    let c1 = extract_name_counts(counts1.unwrap())?;
    let c2 = extract_name_counts(counts2.unwrap())?;
    let min_first = nanmin(c1[0], c2[0]);
    let min_first_last = nanmin(c1[1], c2[1]);
    let min_last = nanmin(c1[2], c2[2]);
    let min_last_first_initial = nanmin(c1[3], c2[3]);
    let max_first = max_propagate_nan(c1[0], c2[0]);
    let max_first_last = max_propagate_nan(c1[1], c2[1]);
    Ok([
        min_first,
        min_first_last,
        min_last,
        min_last_first_initial,
        max_first,
        max_first_last,
    ])
}

fn compute_name_counts_data(counts1: &Option<NameCountsData>, counts2: &Option<NameCountsData>) -> [f64; 6] {
    if counts1.is_none() || counts2.is_none() {
        return [f64::NAN; 6];
    }
    let c1 = counts1.as_ref().unwrap();
    let c2 = counts2.as_ref().unwrap();
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

#[pyfunction]
fn first_names_equal(name1: Option<&str>, name2: Option<&str>) -> PyResult<f64> {
    if name1.is_none() || name2.is_none() {
        return Ok(f64::NAN);
    }
    let n1 = name1.unwrap();
    let n2 = name2.unwrap();
    if py_len(n1) == 0 || py_len(n2) == 0 {
        return Ok(f64::NAN);
    }
    if n1 == "-" || n2 == "-" {
        return Ok(f64::NAN);
    }
    let n1_norm = n1.trim().to_lowercase();
    let n2_norm = n2.trim().to_lowercase();
    if n1_norm == n2_norm {
        Ok(1.0)
    } else {
        Ok(0.0)
    }
}

#[pyfunction]
fn middle_initials_overlap(name1: Option<&str>, name2: Option<&str>) -> PyResult<f64> {
    let s1 = name1.unwrap_or("");
    let s2 = name2.unwrap_or("");
    let c1 = count_initials(s1);
    let c2 = count_initials(s2);
    if c1.is_empty() || c2.is_empty() {
        return Ok(f64::NAN);
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
        return Ok(f64::NAN);
    }
    let score = (intersection_sum as f64) / (union_sum as f64);
    Ok(if score > 1.0 { 1.0 } else { score })
}

#[pyfunction]
fn middle_names_equal(name1: Option<&str>, name2: Option<&str>) -> PyResult<f64> {
    if name1.is_none() || name2.is_none() {
        return Ok(f64::NAN);
    }
    let n1 = name1.unwrap();
    let n2 = name2.unwrap();
    if py_len(n1) == 0 || py_len(n2) == 0 {
        return Ok(f64::NAN);
    }
    if py_len(n1) == 1 || py_len(n2) == 1 {
        let c1 = n1.chars().next().unwrap();
        let c2 = n2.chars().next().unwrap();
        return Ok(if c1 == c2 { 1.0 } else { 0.0 });
    }
    if n1 == n2 {
        Ok(1.0)
    } else {
        Ok(0.0)
    }
}

#[pyfunction]
fn middle_one_missing(name1: Option<&str>, name2: Option<&str>) -> PyResult<f64> {
    let n1 = name1.unwrap_or("");
    let n2 = name2.unwrap_or("");
    let len1 = py_len(n1);
    let len2 = py_len(n2);
    let val = (len1 == 0 && len2 != 0) || (len2 == 0 && len1 != 0);
    Ok(if val { 1.0 } else { 0.0 })
}

#[pyfunction]
fn single_char_first(name1: Option<&str>, name2: Option<&str>) -> PyResult<f64> {
    let n1 = name1.unwrap_or("");
    let n2 = name2.unwrap_or("");
    let val = py_len(n1) == 1 || py_len(n2) == 1;
    Ok(if val { 1.0 } else { 0.0 })
}

#[pyfunction]
fn single_char_middle(name1: Option<&str>, name2: Option<&str>) -> PyResult<f64> {
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
    Ok(if val { 1.0 } else { 0.0 })
}

#[pyfunction]
fn affiliation_overlap(counter1: Option<&Bound<'_, PyAny>>, counter2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    counter_jaccard(counter1, counter2, f64::INFINITY)
}

fn email_parts(email: &str) -> (String, String) {
    let mut e = email.to_string();
    if !e.contains('@') {
        e.push_str("@MISSING");
    }
    let parts: Vec<&str> = e.split('@').collect();
    let last = parts.len() - 1;
    let prefix_raw = parts[..last].join("");
    let suffix_raw = parts[last];
    let prefix = prefix_raw.trim_matches('.').to_lowercase();
    let suffix = suffix_raw.trim_matches('.').to_lowercase();
    (prefix, suffix)
}

#[pyfunction]
fn email_prefix_equal(email1: Option<&str>, email2: Option<&str>) -> PyResult<f64> {
    if email1.is_none() || email2.is_none() {
        return Ok(f64::NAN);
    }
    let e1 = email1.unwrap();
    let e2 = email2.unwrap();
    if py_len(e1) == 0 || py_len(e2) == 0 {
        return Ok(f64::NAN);
    }
    let (p1, _) = email_parts(e1);
    let (p2, _) = email_parts(e2);
    Ok(if p1 == p2 { 1.0 } else { 0.0 })
}

#[pyfunction]
fn email_suffix_equal(email1: Option<&str>, email2: Option<&str>) -> PyResult<f64> {
    if email1.is_none() || email2.is_none() {
        return Ok(f64::NAN);
    }
    let e1 = email1.unwrap();
    let e2 = email2.unwrap();
    if py_len(e1) == 0 || py_len(e2) == 0 {
        return Ok(f64::NAN);
    }
    let (_, s1) = email_parts(e1);
    let (_, s2) = email_parts(e2);
    Ok(if s1 == s2 { 1.0 } else { 0.0 })
}

#[pyfunction]
fn coauthor_overlap(set1: Option<&Bound<'_, PyAny>>, set2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    set_jaccard(set1, set2)
}

#[pyfunction]
fn coauthor_similarity(counter1: Option<&Bound<'_, PyAny>>, counter2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    counter_jaccard(counter1, counter2, 5000.0)
}

#[pyfunction]
fn coauthor_match(set1: Option<&Bound<'_, PyAny>>, set2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    set_jaccard(set1, set2)
}

#[pyfunction]
fn venue_overlap(counter1: Option<&Bound<'_, PyAny>>, counter2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    counter_jaccard(counter1, counter2, f64::INFINITY)
}

#[pyfunction]
fn year_diff(year1: Option<i64>, year2: Option<i64>) -> PyResult<f64> {
    if year1.is_none() || year2.is_none() {
        return Ok(f64::NAN);
    }
    let y1 = year1.unwrap() as f64;
    let y2 = year2.unwrap() as f64;
    let diff = (y1 - y2).abs();
    Ok(diff.min(50.0))
}

#[pyfunction]
fn title_overlap_words(counter1: Option<&Bound<'_, PyAny>>, counter2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    counter_jaccard(counter1, counter2, f64::INFINITY)
}

#[pyfunction]
fn title_overlap_chars(counter1: Option<&Bound<'_, PyAny>>, counter2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    counter_jaccard(counter1, counter2, f64::INFINITY)
}

#[pyfunction]
fn references_authors_overlap(counter1: Option<&Bound<'_, PyAny>>, counter2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    counter_jaccard(counter1, counter2, 5000.0)
}

#[pyfunction]
fn references_titles_overlap(counter1: Option<&Bound<'_, PyAny>>, counter2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    counter_jaccard(counter1, counter2, f64::INFINITY)
}

#[pyfunction]
fn references_venues_overlap(counter1: Option<&Bound<'_, PyAny>>, counter2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    counter_jaccard(counter1, counter2, f64::INFINITY)
}

#[pyfunction]
fn references_author_blocks_jaccard(counter1: Option<&Bound<'_, PyAny>>, counter2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    counter_jaccard(counter1, counter2, f64::INFINITY)
}

#[pyfunction]
fn references_self_citation(refs1: &Bound<'_, PyAny>, refs2: &Bound<'_, PyAny>, pid1: i64, pid2: i64) -> PyResult<f64> {
    let r1 = refs1;
    let r2 = refs2;
    let in1 = r1.contains(pid2)?;
    let in2 = r2.contains(pid1)?;
    Ok(if in1 || in2 { 1.0 } else { 0.0 })
}

#[pyfunction]
fn references_overlap(set1: Option<&Bound<'_, PyAny>>, set2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    set_jaccard(set1, set2)
}

#[pyfunction]
fn position_diff(p1: i64, p2: i64) -> PyResult<f64> {
    let diff = (p1 - p2).abs() as f64;
    Ok(diff.min(50.0))
}

#[pyfunction]
fn abstract_count(has1: bool, has2: bool) -> PyResult<f64> {
    let count = (has1 as i64 + has2 as i64) as f64;
    Ok(count)
}

#[pyfunction]
fn english_count(lang1: Option<&str>, lang2: Option<&str>) -> PyResult<f64> {
    let mut count: i64 = 0;
    if let Some(l1) = lang1 {
        if l1 == "en" || l1 == "un" {
            count += 1;
        }
    }
    if let Some(l2) = lang2 {
        if l2 == "en" || l2 == "un" {
            count += 1;
        }
    }
    Ok(count as f64)
}

#[pyfunction]
fn same_language(lang1: Option<&str>, lang2: Option<&str>) -> PyResult<f64> {
    let eq = match (lang1, lang2) {
        (None, None) => true,
        (Some(a), Some(b)) => a == b,
        _ => false,
    };
    Ok(if eq { 1.0 } else { 0.0 })
}

#[pyfunction]
fn language_reliability_count(reliable1: bool, reliable2: bool) -> PyResult<f64> {
    Ok((reliable1 as i64 + reliable2 as i64) as f64)
}

#[pyfunction]
fn first_name_count_min(counts1: Option<&Bound<'_, PyAny>>, counts2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    Ok(compute_name_counts(counts1, counts2)?[0])
}

#[pyfunction]
fn last_first_name_count_min(counts1: Option<&Bound<'_, PyAny>>, counts2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    Ok(compute_name_counts(counts1, counts2)?[1])
}

#[pyfunction]
fn last_name_count_min(counts1: Option<&Bound<'_, PyAny>>, counts2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    Ok(compute_name_counts(counts1, counts2)?[2])
}

#[pyfunction]
fn last_first_initial_count_min(counts1: Option<&Bound<'_, PyAny>>, counts2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    Ok(compute_name_counts(counts1, counts2)?[3])
}

#[pyfunction]
fn first_name_count_max(counts1: Option<&Bound<'_, PyAny>>, counts2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    Ok(compute_name_counts(counts1, counts2)?[4])
}

#[pyfunction]
fn last_first_name_count_max(counts1: Option<&Bound<'_, PyAny>>, counts2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    Ok(compute_name_counts(counts1, counts2)?[5])
}

fn extract_vec_f64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>()
}

fn cosine_sim_vec(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    let len = a.len().min(b.len());
    for i in 0..len {
        let av = a[i];
        let bv = b[i];
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

fn cosine_sim_any(vec1: &Bound<'_, PyAny>, vec2: &Bound<'_, PyAny>) -> PyResult<(f64, bool)> {
    if let Ok(a) = vec1.downcast::<PyArray1<f32>>() {
        if let Ok(b) = vec2.downcast::<PyArray1<f32>>() {
            let a_slice = unsafe { a.as_slice()? };
            let b_slice = unsafe { b.as_slice()? };
            let all_zero_a = a_slice.iter().all(|v| *v == 0.0);
            let all_zero_b = b_slice.iter().all(|v| *v == 0.0);
            if all_zero_a || all_zero_b {
                return Ok((0.0, false));
            }
            let mut dot: f64 = 0.0;
            let mut norm_a: f64 = 0.0;
            let mut norm_b: f64 = 0.0;
            let len = a_slice.len().min(b_slice.len());
            for i in 0..len {
                let av = a_slice[i] as f64;
                let bv = b_slice[i] as f64;
                dot += av * bv;
                norm_a += av * av;
                norm_b += bv * bv;
            }
            if norm_a == 0.0 || norm_b == 0.0 {
                return Ok((0.0, true));
            }
            return Ok((dot / (norm_a.sqrt() * norm_b.sqrt()), true));
        }
    }
    if let Ok(a) = vec1.downcast::<PyArray1<f64>>() {
        if let Ok(b) = vec2.downcast::<PyArray1<f64>>() {
            let a_slice = unsafe { a.as_slice()? };
            let b_slice = unsafe { b.as_slice()? };
            let all_zero_a = a_slice.iter().all(|v| *v == 0.0);
            let all_zero_b = b_slice.iter().all(|v| *v == 0.0);
            if all_zero_a || all_zero_b {
                return Ok((0.0, false));
            }
            let mut dot: f64 = 0.0;
            let mut norm_a: f64 = 0.0;
            let mut norm_b: f64 = 0.0;
            let len = a_slice.len().min(b_slice.len());
            for i in 0..len {
                let av = a_slice[i];
                let bv = b_slice[i];
                dot += av * bv;
                norm_a += av * av;
                norm_b += bv * bv;
            }
            if norm_a == 0.0 || norm_b == 0.0 {
                return Ok((0.0, true));
            }
            return Ok((dot / (norm_a.sqrt() * norm_b.sqrt()), true));
        }
    }
    let a_vec = extract_vec_f64(vec1)?;
    let b_vec = extract_vec_f64(vec2)?;
    let all_zero_a = a_vec.iter().all(|v| *v == 0.0);
    let all_zero_b = b_vec.iter().all(|v| *v == 0.0);
    if all_zero_a || all_zero_b {
        return Ok((0.0, false));
    }
    Ok((cosine_sim_vec(&a_vec, &b_vec), true))
}

#[pyfunction]
fn specter_cosine_sim(_py: Python<'_>, vec1: Option<&Bound<'_, PyAny>>, vec2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    if vec1.is_none() || vec2.is_none() {
        return Ok(f64::NAN);
    }
    let (score, valid) = cosine_sim_any(vec1.unwrap(), vec2.unwrap())?;
    if !valid {
        return Ok(f64::NAN);
    }
    Ok(score + 1.0)
}

#[pyfunction]
fn journal_overlap(counter1: Option<&Bound<'_, PyAny>>, counter2: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    counter_jaccard(counter1, counter2, f64::INFINITY)
}

fn levenshtein_distance(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let len_a = a_chars.len();
    let len_b = b_chars.len();
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
            let edit = prev[j - 1] + if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            cur[j] = deletion.min(insertion).min(edit);
        }
        prev.clone_from_slice(&cur);
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
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let len_a = a_chars.len();
    let len_b = b_chars.len();
    if len_a == 0 || len_b == 0 {
        return 0;
    }
    let mut prev = vec![0usize; len_b + 1];
    let mut cur = vec![0usize; len_b + 1];
    for i in 1..=len_a {
        for j in 1..=len_b {
            if a_chars[i - 1] == b_chars[j - 1] {
                cur[j] = prev[j - 1] + 1;
            } else {
                cur[j] = cur[j - 1].max(prev[j]);
            }
        }
        prev.clone_from_slice(&cur);
        for v in cur.iter_mut() {
            *v = 0;
        }
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
        let low = if i > search_range { i - search_range } else { 0 };
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
        if long_tolerance && min_len > 4 && common_chars > i + 1 && 2 * common_chars >= min_len + i {
            weight += (1.0 - weight)
                * ((common_chars - i - 1) as f64
                    / (a_len + b_len - i * 2 + 2) as f64);
        }
    }
    weight
}

fn name_text_features(name1: Option<&str>, name2: Option<&str>) -> [f64; 4] {
    if name1.is_none() || name2.is_none() {
        return [f64::NAN; 4];
    }
    let n1 = name1.unwrap();
    let n2 = name2.unwrap();
    if py_len(n1) <= 1 || py_len(n2) <= 1 {
        return [f64::NAN; 4];
    }
    let lev = levenshtein_distance(n1, n2) as f64 / (py_len(n1).max(py_len(n2)) as f64);
    let pref = prefix_dist(n1, n2);
    let lcs = metric_lcs_distance(n1, n2);
    let jaro = jaro_winkler_similarity(n1, n2, false);
    [lev, pref, lcs, jaro]
}

fn name_text_feature(name1: Option<&str>, name2: Option<&str>, idx: usize) -> PyResult<f64> {
    let scores = name_text_features(name1, name2);
    Ok(scores.get(idx).copied().unwrap_or(f64::NAN))
}

#[pyfunction]
fn name_text_levenshtein(_py: Python<'_>, name1: Option<&str>, name2: Option<&str>) -> PyResult<f64> {
    name_text_feature(name1, name2, 0)
}

#[pyfunction]
fn name_text_prefix(_py: Python<'_>, name1: Option<&str>, name2: Option<&str>) -> PyResult<f64> {
    name_text_feature(name1, name2, 1)
}

#[pyfunction]
fn name_text_lcs(_py: Python<'_>, name1: Option<&str>, name2: Option<&str>) -> PyResult<f64> {
    name_text_feature(name1, name2, 2)
}

#[pyfunction]
fn name_text_jaro(_py: Python<'_>, name1: Option<&str>, name2: Option<&str>) -> PyResult<f64> {
    name_text_feature(name1, name2, 3)
}

impl RustFeaturizer {
    fn featurize_pair_data(&self, s1: &SignatureData, s2: &SignatureData, p1: &PaperData, p2: &PaperData) -> Vec<f64> {
        let mut feats: Vec<f64> = Vec::with_capacity(39);
        let first1 = s1.first.as_deref();
        let first2 = s2.first.as_deref();
        let middle1 = s1.middle.as_deref();
        let middle2 = s2.middle.as_deref();

        feats.push(first_names_equal(first1, first2).unwrap_or(f64::NAN));
        feats.push(middle_initials_overlap(middle1, middle2).unwrap_or(f64::NAN));
        feats.push(middle_names_equal(middle1, middle2).unwrap_or(f64::NAN));
        feats.push(middle_one_missing(middle1, middle2).unwrap_or(f64::NAN));
        feats.push(single_char_first(first1, first2).unwrap_or(f64::NAN));
        feats.push(single_char_middle(middle1, middle2).unwrap_or(f64::NAN));

        feats.push(counter_jaccard_data(&s1.affiliations, &s2.affiliations, f64::INFINITY));

        feats.push(email_prefix_equal(s1.email.as_deref(), s2.email.as_deref()).unwrap_or(f64::NAN));
        feats.push(email_suffix_equal(s1.email.as_deref(), s2.email.as_deref()).unwrap_or(f64::NAN));

        feats.push(set_jaccard_data(&s1.coauthor_blocks, &s2.coauthor_blocks));
        feats.push(counter_jaccard_data(&s1.coauthor_ngrams, &s2.coauthor_ngrams, 5000.0));
        feats.push(set_jaccard_data(&s1.coauthors, &s2.coauthors));

        feats.push(counter_jaccard_data(&p1.venue_ngrams, &p2.venue_ngrams, f64::INFINITY));
        feats.push(year_diff(p1.year, p2.year).unwrap_or(f64::NAN));

        feats.push(counter_jaccard_data(&p1.title_words, &p2.title_words, f64::INFINITY));
        feats.push(counter_jaccard_data(&p1.title_chars, &p2.title_chars, f64::INFINITY));

        if self.compute_reference_features && p1.ref_details_present && p2.ref_details_present
        {
            feats.push(counter_jaccard_data(&p1.ref_authors, &p2.ref_authors, 5000.0));
            feats.push(counter_jaccard_data(&p1.ref_titles, &p2.ref_titles, f64::INFINITY));
            feats.push(counter_jaccard_data(&p1.ref_venues, &p2.ref_venues, f64::INFINITY));
            feats.push(counter_jaccard_data(&p1.ref_blocks, &p2.ref_blocks, f64::INFINITY));
            let self_cite = if p1.references.contains(&s2.paper_id) || p2.references.contains(&s1.paper_id) {
                1.0
            } else {
                0.0
            };
            feats.push(self_cite);
            feats.push(refs_jaccard(&p1.references, &p2.references));
        } else {
            feats.extend_from_slice(&[
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ]);
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

        feats.push(position_diff(s1.position, s2.position).unwrap_or(f64::NAN));
        feats.push((p1.has_abstract as i64 + p2.has_abstract as i64) as f64);
        feats.push(english_or_unknown_count as f64);
        let same_lang = match (p1.predicted_language.as_deref(), p2.predicted_language.as_deref()) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b,
            _ => false,
        };
        feats.push(if same_lang { 1.0 } else { 0.0 });
        feats.push((p1.is_reliable as i64 + p2.is_reliable as i64) as f64);

        let counts = compute_name_counts_data(&s1.name_counts, &s2.name_counts);
        feats.extend_from_slice(&counts);

        let specter_sim = if english_or_unknown_count == 2 && p1.specter.is_some() && p2.specter.is_some() {
            let score = cosine_sim_vec_f32(p1.specter.as_ref().unwrap(), p2.specter.as_ref().unwrap());
            score + 1.0
        } else {
            f64::NAN
        };
        feats.push(specter_sim);

        feats.push(counter_jaccard_data(&p1.journal_ngrams, &p2.journal_ngrams, f64::INFINITY));

        let advanced = name_text_features(s1.adv_name.as_deref(), s2.adv_name.as_deref());
        feats.extend_from_slice(&advanced);

        feats
    }
}

#[pymethods]
impl RustFeaturizer {
    #[staticmethod]
    #[pyo3(signature = (dataset, cluster_seed_require_value = 0.0, cluster_seed_disallow_value = 10000.0))]
    fn from_dataset(
        dataset: &Bound<'_, PyAny>,
        cluster_seed_require_value: f64,
        cluster_seed_disallow_value: f64,
    ) -> PyResult<Self> {
        let compute_reference_features: bool = dataset
            .getattr("compute_reference_features")
            .and_then(|v| v.extract())
            .unwrap_or(false);

        let signatures_obj = dataset.getattr("signatures")?;
        let signatures_dict = signatures_obj.downcast::<PyDict>()?;
        let mut signatures = HashMap::with_capacity(signatures_dict.len());
        for (sig_id_obj, sig_obj) in signatures_dict.iter() {
            let sig_id: String = sig_id_obj.extract()?;
            let first =
                extract_string_opt(&sig_obj.getattr("author_info_first_normalized_without_apostrophe")?)?;
            let middle =
                extract_string_opt(&sig_obj.getattr("author_info_middle_normalized_without_apostrophe")?)?;
            let last_normalized = extract_string_opt(&sig_obj.getattr("author_info_last_normalized")?)?;
            let orcid = extract_string_opt(&sig_obj.getattr("author_info_orcid")?)?;
            let email = extract_string_opt(&sig_obj.getattr("author_info_email")?)?;
            let affiliations = extract_counter(&sig_obj.getattr("author_info_affiliations_n_grams")?)?;
            let coauthor_blocks = extract_set_str(&sig_obj.getattr("author_info_coauthor_blocks")?)?;
            let coauthor_ngrams = extract_counter(&sig_obj.getattr("author_info_coauthor_n_grams")?)?;
            let coauthors = extract_set_str(&sig_obj.getattr("author_info_coauthors")?)?;
            let position: i64 = sig_obj.getattr("author_info_position")?.extract()?;
            let paper_id: i64 = sig_obj.getattr("paper_id")?.extract()?;
            let name_counts = extract_name_counts_data(&sig_obj.getattr("author_info_name_counts")?)?;
            let adv_name = first.clone();
            signatures.insert(
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
            );
        }

        let papers_obj = dataset.getattr("papers")?;
        let papers_dict = papers_obj.downcast::<PyDict>()?;
        let specter_obj = dataset.getattr("specter_embeddings").ok();
        let specter_dict = specter_obj
            .as_ref()
            .and_then(|v| v.downcast::<PyDict>().ok());

        let mut papers = HashMap::with_capacity(papers_dict.len());
        for (_paper_id_obj, paper_obj) in papers_dict.iter() {
            let paper_id: i64 = paper_obj.getattr("paper_id")?.extract()?;
            let venue_ngrams = extract_counter(&paper_obj.getattr("venue_ngrams")?)?;
            let title_words = extract_counter(&paper_obj.getattr("title_ngrams_words")?)?;
            let title_chars = extract_counter(&paper_obj.getattr("title_ngrams_chars")?)?;
            let journal_ngrams = extract_counter(&paper_obj.getattr("journal_ngrams")?)?;

        let ref_details_obj = paper_obj.getattr("reference_details")?;
        let ref_details_present = !ref_details_obj.is_none();
        let mut ref_authors = None;
        let mut ref_titles = None;
        let mut ref_venues = None;
        let mut ref_blocks = None;
        if !ref_details_obj.is_none() {
                if let Ok(tuple) = ref_details_obj.extract::<(PyObject, PyObject, PyObject, PyObject)>() {
                    Python::with_gil(|py| {
                        ref_authors = extract_counter(&tuple.0.bind(py)).ok().flatten();
                        ref_titles = extract_counter(&tuple.1.bind(py)).ok().flatten();
                        ref_venues = extract_counter(&tuple.2.bind(py)).ok().flatten();
                        ref_blocks = extract_counter(&tuple.3.bind(py)).ok().flatten();
                    });
                }
            }

            let references = extract_set_i64(&paper_obj.getattr("references")?)?;
            let year: Option<i64> = paper_obj.getattr("year")?.extract()?;
            let year = match year {
                Some(v) if v > 0 => Some(v),
                _ => None,
            };
            let has_abstract: bool = paper_obj.getattr("has_abstract")?.extract()?;
            let predicted_language = extract_string_opt(&paper_obj.getattr("predicted_language")?)?;
            let is_reliable: Option<bool> = paper_obj.getattr("is_reliable")?.extract()?;
            let is_reliable = is_reliable.unwrap_or(false);

            let specter = if let Some(spec_dict) = &specter_dict {
                let key_str = paper_id.to_string();
                if let Ok(Some(val)) = spec_dict.get_item(key_str) {
                    extract_specter_vec(&val)?
                } else if let Ok(Some(val)) = spec_dict.get_item(paper_id) {
                    extract_specter_vec(&val)?
                } else {
                    None
                }
            } else {
                None
            };

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
                    ref_details_present,
                },
            );
        }

        let name_tuples = extract_name_tuples_map(&dataset.getattr("name_tuples")?)?;
        let cluster_seeds_disallow = extract_pair_set(&dataset.getattr("cluster_seeds_disallow")?)?;
        let cluster_seeds_require = extract_cluster_seeds_require(&dataset.getattr("cluster_seeds_require")?)?;

        Ok(RustFeaturizer {
            signatures,
            papers,
            name_tuples,
            cluster_seeds_disallow,
            cluster_seeds_require,
            compute_reference_features,
            cluster_seed_require_value,
            cluster_seed_disallow_value,
        })
    }

    fn update_cluster_seeds(
        &mut self,
        cluster_seeds_require: &Bound<'_, PyAny>,
        cluster_seeds_disallow: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.cluster_seeds_require = extract_cluster_seeds_require(cluster_seeds_require)?;
        self.cluster_seeds_disallow = extract_pair_set(cluster_seeds_disallow)?;
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

        let sig1 = sig_id1.to_string();
        let sig2 = sig_id2.to_string();
        if self.cluster_seeds_disallow.contains(&(sig1.clone(), sig2.clone()))
            || self.cluster_seeds_disallow.contains(&(sig2.clone(), sig1.clone()))
        {
            return Ok(Some(self.cluster_seed_disallow_value));
        }

        if !incremental_dont_use_cluster_seeds {
            if let (Some(c1), Some(c2)) = (
                self.cluster_seeds_require.get(sig_id1),
                self.cluster_seeds_require.get(sig_id2),
            ) {
                if c1 == c2 {
                    return Ok(Some(self.cluster_seed_require_value));
                }
            }
        }

        if dont_merge_cluster_seeds {
            if let (Some(c1), Some(c2)) = (
                self.cluster_seeds_require.get(sig_id1),
                self.cluster_seeds_require.get(sig_id2),
            ) {
                if c1 != c2 {
                    return Ok(Some(self.cluster_seed_disallow_value));
                }
            }
        }

        if let (Some(o1), Some(o2)) = (s1.orcid.as_deref(), s2.orcid.as_deref()) {
            if o1 == o2 {
                return Ok(Some(low_value));
            }
        }

        let last1 = s1.last_normalized.as_deref().unwrap_or("");
        let last2 = s2.last_normalized.as_deref().unwrap_or("");
        if !lasts_equivalent_for_constraint(last1, last2) {
            return Ok(Some(high_value));
        }

        let first1 = s1.first.as_deref().unwrap_or("");
        let first2 = s2.first.as_deref().unwrap_or("");
        if !first1.is_empty() && !first2.is_empty() {
            if let (Some(c1), Some(c2)) = (first1.chars().next(), first2.chars().next()) {
                if c1 != c2 {
                    return Ok(Some(high_value));
                }
            }
        }

        if p1.is_reliable && p2.is_reliable {
            let l1 = p1.predicted_language.as_deref();
            let l2 = p2.predicted_language.as_deref();
            if l1 != l2 {
                return Ok(Some(high_value));
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
            return Ok(Some(high_value));
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

                let middle_1_words: HashSet<&str> =
                    middle_1_all.iter().copied().filter(|w| py_len(w) > 1).collect();
                let middle_2_words: HashSet<&str> =
                    middle_2_all.iter().copied().filter(|w| py_len(w) > 1).collect();

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
                    return Ok(Some(high_value));
                }
            }
        }

        Ok(None)
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
        Ok(self.featurize_pair_data(s1, s2, p1, p2))
    }

    fn featurize_pairs(&self, py: Python<'_>, pairs: Vec<(String, String)>) -> PyResult<Vec<Vec<f64>>> {
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
                return Err(pyo3::exceptions::PyKeyError::new_err(s1.paper_id.to_string()));
            }
            if self.papers.get(&s2.paper_id).is_none() {
                return Err(pyo3::exceptions::PyKeyError::new_err(s2.paper_id.to_string()));
            }
        }
        let feats = py.allow_threads(|| {
            pairs
                .par_iter()
                .map(|(sig_id1, sig_id2)| {
                    let s1 = self.signatures.get(sig_id1).unwrap();
                    let s2 = self.signatures.get(sig_id2).unwrap();
                    let p1 = self.papers.get(&s1.paper_id).unwrap();
                    let p2 = self.papers.get(&s2.paper_id).unwrap();
                    self.featurize_pair_data(s1, s2, p1, p2)
                })
                .collect::<Vec<_>>()
        });
        Ok(feats)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let file = File::create(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let file = File::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let reader = BufReader::new(file);
        let featurizer: RustFeaturizer =
            bincode::deserialize_from(reader).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(featurizer)
    }
}

#[pyfunction]
fn featurize_pair(
    first1: &Bound<'_, PyAny>,
    middle1: &Bound<'_, PyAny>,
    first2: &Bound<'_, PyAny>,
    middle2: &Bound<'_, PyAny>,
    affiliation1: &Bound<'_, PyAny>,
    affiliation2: &Bound<'_, PyAny>,
    email1: &Bound<'_, PyAny>,
    email2: &Bound<'_, PyAny>,
    coauthor_blocks1: &Bound<'_, PyAny>,
    coauthor_blocks2: &Bound<'_, PyAny>,
    coauthor_ngrams1: &Bound<'_, PyAny>,
    coauthor_ngrams2: &Bound<'_, PyAny>,
    coauthors1: &Bound<'_, PyAny>,
    coauthors2: &Bound<'_, PyAny>,
    venue_ngrams1: &Bound<'_, PyAny>,
    venue_ngrams2: &Bound<'_, PyAny>,
    year1: &Bound<'_, PyAny>,
    year2: &Bound<'_, PyAny>,
    title_words1: &Bound<'_, PyAny>,
    title_words2: &Bound<'_, PyAny>,
    title_chars1: &Bound<'_, PyAny>,
    title_chars2: &Bound<'_, PyAny>,
    compute_reference_features: bool,
    ref_authors1: &Bound<'_, PyAny>,
    ref_authors2: &Bound<'_, PyAny>,
    ref_titles1: &Bound<'_, PyAny>,
    ref_titles2: &Bound<'_, PyAny>,
    ref_venues1: &Bound<'_, PyAny>,
    ref_venues2: &Bound<'_, PyAny>,
    ref_blocks1: &Bound<'_, PyAny>,
    ref_blocks2: &Bound<'_, PyAny>,
    references1: &Bound<'_, PyAny>,
    references2: &Bound<'_, PyAny>,
    paper_id1: &Bound<'_, PyAny>,
    paper_id2: &Bound<'_, PyAny>,
    position1: i64,
    position2: i64,
    has_abstract1: bool,
    has_abstract2: bool,
    lang1: &Bound<'_, PyAny>,
    lang2: &Bound<'_, PyAny>,
    reliable1: bool,
    reliable2: bool,
    name_counts1: &Bound<'_, PyAny>,
    name_counts2: &Bound<'_, PyAny>,
    specter1: &Bound<'_, PyAny>,
    specter2: &Bound<'_, PyAny>,
    journal1: &Bound<'_, PyAny>,
    journal2: &Bound<'_, PyAny>,
    adv_name1: &Bound<'_, PyAny>,
    adv_name2: &Bound<'_, PyAny>,
) -> PyResult<Vec<f64>> {
    let mut feats: Vec<f64> = Vec::with_capacity(39);

    let first1_opt: Option<&str> = if first1.is_none() { None } else { Some(first1.extract()?) };
    let first2_opt: Option<&str> = if first2.is_none() { None } else { Some(first2.extract()?) };
    let middle1_opt: Option<&str> = if middle1.is_none() { None } else { Some(middle1.extract()?) };
    let middle2_opt: Option<&str> = if middle2.is_none() { None } else { Some(middle2.extract()?) };

    feats.push(first_names_equal(first1_opt, first2_opt)?);
    feats.push(middle_initials_overlap(middle1_opt, middle2_opt)?);
    feats.push(middle_names_equal(middle1_opt, middle2_opt)?);
    feats.push(middle_one_missing(middle1_opt, middle2_opt)?);
    feats.push(single_char_first(first1_opt, first2_opt)?);
    feats.push(single_char_middle(middle1_opt, middle2_opt)?);

    let affiliation1_opt = if affiliation1.is_none() { None } else { Some(affiliation1) };
    let affiliation2_opt = if affiliation2.is_none() { None } else { Some(affiliation2) };
    feats.push(affiliation_overlap(affiliation1_opt, affiliation2_opt)?);

    let email1_opt: Option<&str> = if email1.is_none() { None } else { Some(email1.extract()?) };
    let email2_opt: Option<&str> = if email2.is_none() { None } else { Some(email2.extract()?) };
    feats.push(email_prefix_equal(email1_opt, email2_opt)?);
    feats.push(email_suffix_equal(email1_opt, email2_opt)?);

    let coauthor_blocks1_opt = if coauthor_blocks1.is_none() { None } else { Some(coauthor_blocks1) };
    let coauthor_blocks2_opt = if coauthor_blocks2.is_none() { None } else { Some(coauthor_blocks2) };
    let coauthor_ngrams1_opt = if coauthor_ngrams1.is_none() { None } else { Some(coauthor_ngrams1) };
    let coauthor_ngrams2_opt = if coauthor_ngrams2.is_none() { None } else { Some(coauthor_ngrams2) };
    let coauthors1_opt = if coauthors1.is_none() { None } else { Some(coauthors1) };
    let coauthors2_opt = if coauthors2.is_none() { None } else { Some(coauthors2) };
    feats.push(coauthor_overlap(coauthor_blocks1_opt, coauthor_blocks2_opt)?);
    feats.push(coauthor_similarity(coauthor_ngrams1_opt, coauthor_ngrams2_opt)?);
    feats.push(coauthor_match(coauthors1_opt, coauthors2_opt)?);

    let venue_ngrams1_opt = if venue_ngrams1.is_none() { None } else { Some(venue_ngrams1) };
    let venue_ngrams2_opt = if venue_ngrams2.is_none() { None } else { Some(venue_ngrams2) };
    feats.push(venue_overlap(venue_ngrams1_opt, venue_ngrams2_opt)?);

    let year1_opt: Option<i64> = if year1.is_none() { None } else { Some(year1.extract()?) };
    let year2_opt: Option<i64> = if year2.is_none() { None } else { Some(year2.extract()?) };
    feats.push(year_diff(year1_opt, year2_opt)?);

    let title_words1_opt = if title_words1.is_none() { None } else { Some(title_words1) };
    let title_words2_opt = if title_words2.is_none() { None } else { Some(title_words2) };
    let title_chars1_opt = if title_chars1.is_none() { None } else { Some(title_chars1) };
    let title_chars2_opt = if title_chars2.is_none() { None } else { Some(title_chars2) };
    feats.push(title_overlap_words(title_words1_opt, title_words2_opt)?);
    feats.push(title_overlap_chars(title_chars1_opt, title_chars2_opt)?);

    if compute_reference_features
        && !ref_authors1.is_none()
        && !ref_authors2.is_none()
        && !ref_titles1.is_none()
        && !ref_titles2.is_none()
        && !ref_venues1.is_none()
        && !ref_venues2.is_none()
        && !ref_blocks1.is_none()
        && !ref_blocks2.is_none()
    {
        feats.push(references_authors_overlap(Some(ref_authors1), Some(ref_authors2))?);
        feats.push(references_titles_overlap(Some(ref_titles1), Some(ref_titles2))?);
        feats.push(references_venues_overlap(Some(ref_venues1), Some(ref_venues2))?);
        feats.push(references_author_blocks_jaccard(Some(ref_blocks1), Some(ref_blocks2))?);

        let pid1_opt: Option<i64> = if paper_id1.is_none() { None } else { Some(paper_id1.extract()?) };
        let pid2_opt: Option<i64> = if paper_id2.is_none() { None } else { Some(paper_id2.extract()?) };
        let self_cite = if !references1.is_none()
            && !references2.is_none()
            && pid1_opt.is_some()
            && pid2_opt.is_some()
        {
            references_self_citation(references1, references2, pid1_opt.unwrap(), pid2_opt.unwrap())?
        } else {
            f64::NAN
        };
        feats.push(self_cite);
        let references1_opt = if references1.is_none() { None } else { Some(references1) };
        let references2_opt = if references2.is_none() { None } else { Some(references2) };
        feats.push(references_overlap(references1_opt, references2_opt)?);
    } else {
        feats.extend_from_slice(&[
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ]);
    }

    let english_or_unknown_count = {
        let mut count = 0i64;
        let lang1_opt: Option<&str> = if lang1.is_none() { None } else { Some(lang1.extract()?) };
        let lang2_opt: Option<&str> = if lang2.is_none() { None } else { Some(lang2.extract()?) };
        if let Some(l1) = lang1_opt {
            if l1 == "en" || l1 == "un" {
                count += 1;
            }
        }
        if let Some(l2) = lang2_opt {
            if l2 == "en" || l2 == "un" {
                count += 1;
            }
        }
        count
    };

    feats.push(position_diff(position1, position2)?);
    feats.push(abstract_count(has_abstract1, has_abstract2)?);
    feats.push(english_or_unknown_count as f64);
    let lang1_opt: Option<&str> = if lang1.is_none() { None } else { Some(lang1.extract()?) };
    let lang2_opt: Option<&str> = if lang2.is_none() { None } else { Some(lang2.extract()?) };
    feats.push(same_language(lang1_opt, lang2_opt)?);
    feats.push(language_reliability_count(reliable1, reliable2)?);

    let name_counts1_opt = if name_counts1.is_none() { None } else { Some(name_counts1) };
    let name_counts2_opt = if name_counts2.is_none() { None } else { Some(name_counts2) };
    let name_counts = compute_name_counts(name_counts1_opt, name_counts2_opt)?;
    feats.extend_from_slice(&name_counts);

    let specter_sim = if english_or_unknown_count == 2 && !specter1.is_none() && !specter2.is_none() {
        let (score, valid) = cosine_sim_any(specter1, specter2)?;
        if valid {
            score + 1.0
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };
    feats.push(specter_sim);

    let journal1_opt = if journal1.is_none() { None } else { Some(journal1) };
    let journal2_opt = if journal2.is_none() { None } else { Some(journal2) };
    feats.push(journal_overlap(journal1_opt, journal2_opt)?);

    let adv_name1_opt: Option<&str> = if adv_name1.is_none() { None } else { Some(adv_name1.extract()?) };
    let adv_name2_opt: Option<&str> = if adv_name2.is_none() { None } else { Some(adv_name2.extract()?) };
    let advanced = name_text_features(adv_name1_opt, adv_name2_opt);
    feats.extend_from_slice(&advanced);

    Ok(feats)
}

#[pymodule]
fn s2and_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(first_names_equal, m)?)?;
    m.add_function(wrap_pyfunction!(middle_initials_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(middle_names_equal, m)?)?;
    m.add_function(wrap_pyfunction!(middle_one_missing, m)?)?;
    m.add_function(wrap_pyfunction!(single_char_first, m)?)?;
    m.add_function(wrap_pyfunction!(single_char_middle, m)?)?;
    m.add_function(wrap_pyfunction!(affiliation_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(email_prefix_equal, m)?)?;
    m.add_function(wrap_pyfunction!(email_suffix_equal, m)?)?;
    m.add_function(wrap_pyfunction!(coauthor_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(coauthor_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(coauthor_match, m)?)?;
    m.add_function(wrap_pyfunction!(venue_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(year_diff, m)?)?;
    m.add_function(wrap_pyfunction!(title_overlap_words, m)?)?;
    m.add_function(wrap_pyfunction!(title_overlap_chars, m)?)?;
    m.add_function(wrap_pyfunction!(references_authors_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(references_titles_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(references_venues_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(references_author_blocks_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(references_self_citation, m)?)?;
    m.add_function(wrap_pyfunction!(references_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(position_diff, m)?)?;
    m.add_function(wrap_pyfunction!(abstract_count, m)?)?;
    m.add_function(wrap_pyfunction!(english_count, m)?)?;
    m.add_function(wrap_pyfunction!(same_language, m)?)?;
    m.add_function(wrap_pyfunction!(language_reliability_count, m)?)?;
    m.add_function(wrap_pyfunction!(first_name_count_min, m)?)?;
    m.add_function(wrap_pyfunction!(last_first_name_count_min, m)?)?;
    m.add_function(wrap_pyfunction!(last_name_count_min, m)?)?;
    m.add_function(wrap_pyfunction!(last_first_initial_count_min, m)?)?;
    m.add_function(wrap_pyfunction!(first_name_count_max, m)?)?;
    m.add_function(wrap_pyfunction!(last_first_name_count_max, m)?)?;
    m.add_function(wrap_pyfunction!(specter_cosine_sim, m)?)?;
    m.add_function(wrap_pyfunction!(journal_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(name_text_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(name_text_prefix, m)?)?;
    m.add_function(wrap_pyfunction!(name_text_lcs, m)?)?;
    m.add_function(wrap_pyfunction!(name_text_jaro, m)?)?;
    m.add_function(wrap_pyfunction!(featurize_pair, m)?)?;
    m.add_class::<RustFeaturizer>()?;
    Ok(())
}
