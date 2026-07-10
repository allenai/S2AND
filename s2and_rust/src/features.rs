use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator};
use pyo3::Bound;
use std::collections::{HashMap, HashSet};

use crate::constraints::count_initials;
use crate::name_counts::NameCountsData;
use crate::{py_len, CounterData};

pub(crate) fn extract_string_string_map(
    obj: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, String>> {
    let dict = obj.downcast::<PyDict>()?;
    let mut out = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        out.insert(key.extract()?, value.extract()?);
    }
    Ok(out)
}

pub(crate) fn extract_string_vec_map(
    obj: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, Vec<String>>> {
    let dict = obj.downcast::<PyDict>()?;
    let mut out = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key_text: String = key.extract()?;
        let mut values = Vec::new();
        for item in PyIterator::from_object(&value)? {
            values.push(item?.extract()?);
        }
        out.insert(key_text, values);
    }
    Ok(out)
}

pub(crate) fn filter_text_for_char_ngrams(
    text: &str,
    stopwords: Option<&HashSet<String>>,
    drop_short_tokens: bool,
) -> String {
    text.split(' ')
        .filter(|word| {
            (!drop_short_tokens || py_len(word) > 2)
                && stopwords.map_or(true, |stopwords_set| !stopwords_set.contains(*word))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn char_ngrams_counter_python_compat(
    text: &str,
    use_unigrams: bool,
    use_bigrams: bool,
    stopwords: Option<&HashSet<String>>,
    drop_short_tokens: bool,
) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::new();
    }
    let filtered_text = filter_text_for_char_ngrams(text, stopwords, drop_short_tokens);
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

pub(crate) fn word_ngrams_counter_python_compat(
    text: &str,
    stopwords: &HashSet<String>,
    drop_short_tokens: bool,
) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::new();
    }
    let text_split: Vec<&str> = text
        .split_whitespace()
        .filter(|word| !stopwords.contains(*word) && (!drop_short_tokens || py_len(word) > 1))
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

pub(crate) fn char_ngrams_counter(text: &str) -> HashMap<String, usize> {
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

pub(crate) fn word_ngrams_counter(text: &str) -> HashMap<String, usize> {
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

#[cfg(test)]
mod char_ngram_tests {
    use super::char_ngrams_counter_python_compat;

    #[test]
    fn explicit_short_token_filter_applies_without_stopwords() {
        let filtered = char_ngrams_counter_python_compat("li wu abcd", false, true, None, true);
        assert!(!filtered.contains_key("li"));
        assert!(filtered.contains_key("ab"));

        let unfiltered = char_ngrams_counter_python_compat("li wu abcd", false, true, None, false);
        assert!(unfiltered.contains_key("li"));
        assert!(unfiltered.contains_key("wu"));
        assert!(unfiltered.contains_key("ab"));
    }
}

pub(crate) fn counter_jaccard_data(
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

pub(crate) fn set_jaccard_data<T: Eq + std::hash::Hash>(
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

pub(crate) fn nanmin(a: f64, b: f64) -> f64 {
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

pub(crate) fn max_propagate_nan(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

pub(crate) fn compute_name_counts_data(
    counts1: Option<&NameCountsData>,
    counts2: Option<&NameCountsData>,
) -> [f64; 6] {
    let (Some(c1), Some(c2)) = (counts1, counts2) else {
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

pub(crate) fn first_names_equal(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let (Some(n1), Some(n2)) = (name1, name2) else {
        return f64::NAN;
    };
    // Trim/lowercase first, then test emptiness, so whitespace-only inputs are
    // treated as empty (NaN) rather than comparing equal as "".
    let n1_norm = n1.trim().to_lowercase();
    let n2_norm = n2.trim().to_lowercase();
    if n1_norm.is_empty() || n2_norm.is_empty() {
        return f64::NAN;
    }
    if n1_norm == "-" || n2_norm == "-" {
        return f64::NAN;
    }
    if n1_norm == n2_norm {
        1.0
    } else {
        0.0
    }
}

pub(crate) fn middle_initials_overlap(name1: Option<&str>, name2: Option<&str>) -> f64 {
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
    if score > 1.0 {
        1.0
    } else {
        score
    }
}

pub(crate) fn middle_names_equal(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let (Some(n1), Some(n2)) = (name1, name2) else {
        return f64::NAN;
    };
    if py_len(n1) == 0 || py_len(n2) == 0 {
        return f64::NAN;
    }
    // When either side is a single-character initial, compare the sets of token
    // initials so a joined multi-token middle ("james lee") matches the other
    // side's initial for ANY of its tokens, not just the first one.
    if py_len(n1) == 1 || py_len(n2) == 1 {
        for initial_1 in n1.split(' ').filter_map(|t| t.chars().next()) {
            for initial_2 in n2.split(' ').filter_map(|t| t.chars().next()) {
                if initial_1 == initial_2 {
                    return 1.0;
                }
            }
        }
        return 0.0;
    }
    if n1 == n2 {
        1.0
    } else {
        0.0
    }
}

pub(crate) fn middle_one_missing(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let n1 = name1.unwrap_or("");
    let n2 = name2.unwrap_or("");
    let len1 = py_len(n1);
    let len2 = py_len(n2);
    let val = (len1 == 0 && len2 != 0) || (len2 == 0 && len1 != 0);
    if val {
        1.0
    } else {
        0.0
    }
}

pub(crate) fn single_char_first(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let n1 = name1.unwrap_or("");
    let n2 = name2.unwrap_or("");
    let val = py_len(n1) == 1 || py_len(n2) == 1;
    if val {
        1.0
    } else {
        0.0
    }
}

pub(crate) fn single_char_middle(name1: Option<&str>, name2: Option<&str>) -> f64 {
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
    if val {
        1.0
    } else {
        0.0
    }
}

pub(crate) fn email_parts(email: &str) -> Option<(String, String)> {
    let (prefix_raw, suffix_raw) = email.split_once('@')?;
    if prefix_raw.is_empty()
        || suffix_raw.is_empty()
        || suffix_raw.contains('@')
        || email.chars().any(char::is_whitespace)
    {
        return None;
    }
    let prefix = prefix_raw.trim_matches('.').to_lowercase();
    let suffix = suffix_raw.trim_matches('.').to_lowercase();
    if prefix.is_empty() || suffix.is_empty() {
        return None;
    }
    Some((prefix, suffix))
}

pub(crate) fn email_pair_parts(
    email1: Option<&str>,
    email2: Option<&str>,
) -> Option<((String, String), (String, String))> {
    let (Some(e1), Some(e2)) = (email1, email2) else {
        return None;
    };
    if py_len(e1) == 0 || py_len(e2) == 0 {
        return None;
    }
    Some((email_parts(e1)?, email_parts(e2)?))
}

pub(crate) fn year_diff(year1: Option<i64>, year2: Option<i64>) -> f64 {
    let (Some(y1_raw), Some(y2_raw)) = (year1, year2) else {
        return f64::NAN;
    };
    let y1 = y1_raw as f64;
    let y2 = y2_raw as f64;
    let diff = (y1 - y2).abs();
    diff.min(50.0)
}

pub(crate) fn position_diff(p1: i64, p2: i64) -> f64 {
    p1.abs_diff(p2).min(50) as f64
}

pub(crate) fn cosine_sim_vec_f32(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::NAN;
    }
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..a.len() {
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

pub(crate) fn cosine_sim_with_norms(a: &[f32], norm_a: f64, b: &[f32], norm_b: f64) -> f64 {
    if a.len() != b.len() {
        return f64::NAN;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    let mut dot = 0.0;
    for i in 0..a.len() {
        dot += (a[i] as f64) * (b[i] as f64);
    }
    dot / (norm_a * norm_b)
}

pub(crate) fn levenshtein_distance(a: &str, b: &str) -> usize {
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

pub(crate) fn levenshtein_distance_bytes(a: &[u8], b: &[u8]) -> usize {
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

pub(crate) fn prefix_dist(a: &str, b: &str) -> f64 {
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

pub(crate) fn lcs_length(a: &str, b: &str) -> usize {
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

pub(crate) fn lcs_length_bytes(a: &[u8], b: &[u8]) -> usize {
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

pub(crate) fn metric_lcs_distance(a: &str, b: &str) -> f64 {
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

pub(crate) fn jaro_winkler_similarity(a: &str, b: &str, long_tolerance: bool) -> f64 {
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

pub(crate) fn name_text_features(name1: Option<&str>, name2: Option<&str>) -> [f64; 4] {
    let (Some(n1), Some(n2)) = (name1, name2) else {
        return [f64::NAN; 4];
    };
    if py_len(n1) == 0 || py_len(n2) == 0 {
        return [f64::NAN; 4];
    }
    let lev = levenshtein_distance(n1, n2) as f64 / (py_len(n1).max(py_len(n2)) as f64);
    let pref = prefix_dist(n1, n2);
    let lcs = metric_lcs_distance(n1, n2);
    let jaro = jaro_winkler_similarity(n1, n2, false);
    [lev, pref, lcs, jaro]
}

pub(crate) const FULL_FEATURE_COUNT: usize = 33;

pub(crate) struct MatrixAggregateIndexSelection {
    pub(crate) matrix_indices: Vec<usize>,
    pub(crate) aggregate_indices: Vec<usize>,
    pub(crate) aggregate_matrix_positions: Vec<usize>,
}

/// Resolve optional feature indices to concrete feature columns.
///
/// Contract: `None` expands to `0..full_cols`; explicit lists are returned
/// unchanged, including caller order and duplicate indices; every index must be
/// strictly less than `full_cols`.
pub(crate) fn resolve_feature_indices(
    argument_name: &str,
    indices: Option<Vec<usize>>,
    full_cols: usize,
) -> PyResult<Vec<usize>> {
    let resolved = indices.unwrap_or_else(|| (0..full_cols).collect());
    validate_feature_indices(argument_name, &resolved, full_cols)?;
    Ok(resolved)
}

pub(crate) fn validate_feature_indices(
    argument_name: &str,
    indices: &[usize],
    full_cols: usize,
) -> PyResult<()> {
    for idx in indices.iter() {
        if *idx >= full_cols {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{argument_name} contains out-of-range index {} for {} columns",
                idx, full_cols
            )));
        }
    }
    Ok(())
}

/// Map aggregate feature indices onto already materialized matrix columns.
///
/// Contract: one output position is produced for each aggregate index, in
/// aggregate-index order. Matrix indices keep caller order and duplicates;
/// duplicate matrix features map to their first matrix position because all
/// duplicate columns contain the same feature value for a row.
pub(crate) fn matrix_positions_for_feature_indices(
    matrix_indices: &[usize],
    aggregate_indices: &[usize],
) -> PyResult<Vec<usize>> {
    let mut first_position_by_feature: HashMap<usize, usize> =
        HashMap::with_capacity(matrix_indices.len());
    for (position, feature_index) in matrix_indices.iter().enumerate() {
        first_position_by_feature
            .entry(*feature_index)
            .or_insert(position);
    }

    let mut positions = Vec::with_capacity(aggregate_indices.len());
    for feature_index in aggregate_indices.iter() {
        let Some(matrix_position) = first_position_by_feature.get(feature_index) else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "aggregate index {} is not present in matrix_indices; include it to avoid recomputation",
                feature_index
            )));
        };
        positions.push(*matrix_position);
    }
    Ok(positions)
}

/// Resolve paired matrix and aggregate feature-index contracts.
///
/// Contract: `aggregate_indices=None` mirrors the resolved matrix columns; when
/// explicit, aggregates must be valid feature indices and already present in
/// `matrix_indices` so downstream aggregation reads the computed matrix instead
/// of recomputing feature rows.
pub(crate) fn resolve_matrix_aggregate_indices(
    matrix_indices: Option<Vec<usize>>,
    aggregate_indices: Option<Vec<usize>>,
    full_cols: usize,
) -> PyResult<MatrixAggregateIndexSelection> {
    let resolved_matrix_indices =
        resolve_feature_indices("matrix_indices", matrix_indices, full_cols)?;
    let resolved_aggregate_indices = match aggregate_indices {
        Some(indices) => {
            validate_feature_indices("aggregate_indices", &indices, full_cols)?;
            indices
        }
        None => resolved_matrix_indices.clone(),
    };
    let aggregate_matrix_positions = matrix_positions_for_feature_indices(
        &resolved_matrix_indices,
        &resolved_aggregate_indices,
    )?;
    Ok(MatrixAggregateIndexSelection {
        matrix_indices: resolved_matrix_indices,
        aggregate_indices: resolved_aggregate_indices,
        aggregate_matrix_positions,
    })
}

#[cfg(test)]
mod email_tests {
    use super::{email_pair_parts, email_parts};

    #[test]
    fn malformed_emails_are_missing_evidence() {
        for malformed in [
            "jsmith",
            "a@b@c",
            "@",
            "a@",
            "@b",
            "a b@c",
            "a@b c",
            " a@b",
            "a@b ",
            "a\u{00a0}@b",
            ".@b",
        ] {
            assert_eq!(email_parts(malformed), None, "{malformed:?}");
        }
        assert_eq!(
            email_parts("J.Smith@MIT.EDU."),
            Some(("j.smith".into(), "mit.edu".into()))
        );
        assert_eq!(email_pair_parts(Some("a@b@c"), Some("ab@c")), None);
    }
}
