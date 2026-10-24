use crate::{HashMap, HashSet};
use pyo3::prelude::*;

use crate::constraints::count_initials;
use crate::name_counts::NameCountsData;
use crate::text_compat::is_python_whitespace_compat;
use crate::{fnv64_update, py_len, py_len_above_one, py_len_is_one, CounterData, FNV_OFFSET};

pub(crate) fn filter_text_for_char_ngrams(
    text: &str,
    stopwords: Option<&HashSet<String>>,
    drop_short_tokens: bool,
) -> String {
    text.split(' ')
        .filter(|word| {
            (!drop_short_tokens || word.chars().nth(2).is_some())
                && stopwords.map_or(true, |stopwords_set| !stopwords_set.contains(*word))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The map form of `char_ngram_counter_data`; its reference in tests.
#[cfg(test)]
pub(crate) fn char_ngrams_counter_python_compat(
    text: &str,
    use_unigrams: bool,
    use_bigrams: bool,
    stopwords: Option<&HashSet<String>>,
    drop_short_tokens: bool,
) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::default();
    }
    let filtered_text = filter_text_for_char_ngrams(text, stopwords, drop_short_tokens);
    if filtered_text.is_empty() {
        return HashMap::default();
    }
    let chars: Vec<char> = filtered_text.chars().collect();
    if chars.is_empty() {
        return HashMap::default();
    }

    let mut out: HashMap<String, usize> = HashMap::default();
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

/// `counter_data_from_usize_map(char_ngrams_counter_python_compat(..))` without
/// a String and a map entry per gram.
///
/// Featurizer builds count grams of every title and venue, and the formatted
/// keys and hashed map inserts dominated that work. A gram's hash is taken over
/// its characters' UTF-8 bytes, which equal the bytes of the formatted gram.
/// Grams are counted by sorting on hash, then length and characters, so distinct
/// grams with colliding hashes stay separate entries. Entries and their f32 sum
/// follow hash order, as before.
pub(crate) fn char_ngram_counter_data(
    text: &str,
    use_unigrams: bool,
    use_bigrams: bool,
    stopwords: Option<&HashSet<String>>,
    drop_short_tokens: bool,
) -> Option<CounterData> {
    if text.is_empty() {
        return None;
    }
    let filtered_text = filter_text_for_char_ngrams(text, stopwords, drop_short_tokens);
    let chars: Vec<char> = filtered_text.chars().collect();
    // (hash, length, characters); unused character slots hold '\0'.
    let mut grams: Vec<(u64, u8, [char; 4])> = Vec::new();
    let mut push = |gram: &[char]| {
        let mut hash = FNV_OFFSET;
        let mut buffer = [0u8; 4];
        for ch in gram {
            hash = fnv64_update(hash, ch.encode_utf8(&mut buffer).as_bytes());
        }
        let mut key = ['\0'; 4];
        key[..gram.len()].copy_from_slice(gram);
        grams.push((hash, gram.len() as u8, key));
    };
    if use_unigrams {
        for ch in chars.iter().filter(|ch| **ch != ' ') {
            push(std::slice::from_ref(ch));
        }
    }
    for (enabled, width) in [(use_bigrams, 2), (true, 3), (true, 4)] {
        if enabled {
            for window in chars.windows(width).filter(|window| !window.contains(&' ')) {
                push(window);
            }
        }
    }
    if grams.is_empty() {
        return None;
    }
    grams.sort_unstable();
    let mut entries: Vec<(u64, f32)> = Vec::new();
    for run in grams.chunk_by(|left, right| left == right) {
        entries.push((run[0].0, run.len() as f32));
    }
    let sum: f32 = entries.iter().map(|entry| entry.1).sum();
    Some(CounterData::new(entries, sum))
}

/// The words `word_ngrams_counter_python_compat` counts.
fn filtered_words<'a>(
    text: &'a str,
    stopwords: &HashSet<String>,
    drop_short_tokens: bool,
) -> Vec<&'a str> {
    text.split_whitespace()
        .filter(|word| !stopwords.contains(*word) && (!drop_short_tokens || py_len_above_one(word)))
        .collect()
}

/// Counter data of the word unigrams, bigrams and trigrams of `words`, equal to
/// `counter_data_from_usize_map` over the counts the map counters build.
///
/// Grams join words with one space and words hold no whitespace, so a gram's
/// hash is its words' bytes with a space byte between them. Counting by sorting
/// on hash, then length and words, avoids a String and a map entry per gram and
/// keeps distinct grams with colliding hashes as separate entries.
fn word_gram_counter_data(words: &[&str]) -> Option<CounterData> {
    let mut grams: Vec<(u64, u8, [&str; 3])> = Vec::with_capacity(3 * words.len());
    for width in 1..=3 {
        for window in words.windows(width) {
            let mut hash = fnv64_update(FNV_OFFSET, window[0].as_bytes());
            for word in &window[1..] {
                hash = fnv64_update(fnv64_update(hash, b" "), word.as_bytes());
            }
            let mut key = [""; 3];
            key[..width].copy_from_slice(window);
            grams.push((hash, width as u8, key));
        }
    }
    if grams.is_empty() {
        return None;
    }
    grams.sort_unstable();
    let mut entries: Vec<(u64, f32)> = Vec::new();
    for run in grams.chunk_by(|left, right| left == right) {
        entries.push((run[0].0, run.len() as f32));
    }
    let sum: f32 = entries.iter().map(|entry| entry.1).sum();
    Some(CounterData::new(entries, sum))
}

/// `counter_data_from_usize_map(word_ngrams_counter_python_compat(..))`.
pub(crate) fn title_word_counter_data(
    text: &str,
    stopwords: &HashSet<String>,
    drop_short_tokens: bool,
) -> Option<CounterData> {
    word_gram_counter_data(&filtered_words(text, stopwords, drop_short_tokens))
}

/// `counter_data_from_usize_map(word_ngrams_counter(..))`.
pub(crate) fn word_counter_data(text: &str) -> Option<CounterData> {
    word_gram_counter_data(&text.split_whitespace().collect::<Vec<_>>())
}

pub(crate) fn word_ngrams_counter_python_compat(
    text: &str,
    stopwords: &HashSet<String>,
    drop_short_tokens: bool,
) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::default();
    }
    let text_split = filtered_words(text, stopwords, drop_short_tokens);
    if text_split.is_empty() {
        return HashMap::default();
    }
    let mut out: HashMap<String, usize> = HashMap::default();
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

pub(crate) fn word_ngrams_counter(text: &str) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::default();
    }

    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.is_empty() {
        return HashMap::default();
    }

    let mut out: HashMap<String, usize> = HashMap::default();
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
    use super::intersection_min_sum_avx2;

    #[test]
    fn avx2_intersection_matches_the_scalar_merge() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 99u64;
        let mut next = move |bound: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) % bound
        };
        let mut checked = 0usize;
        for _ in 0..20_000 {
            let len_a = 1 + next(600) as usize;
            let len_b = 1 + next(600) as usize;
            let universe = 1 + next(4000);
            let mut make = |len: usize| {
                let mut keys: Vec<u64> = (0..len)
                    .map(|_| next(universe).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    .collect();
                keys.sort_unstable();
                keys.dedup();
                keys.into_iter()
                    .map(|key| (key, (1 + next(6)) as f32))
                    .collect::<Vec<_>>()
            };
            let a = make(len_a);
            let b = make(len_b);
            let sum = |entries: &[(u64, f32)]| entries.iter().map(|entry| entry.1).sum::<f32>();
            let counter_a = crate::CounterData::new(a.clone(), sum(&a));
            let counter_b = crate::CounterData::new(b.clone(), sum(&b));
            assert!(counter_a.simd && counter_b.simd);
            let mut expected = 0.0f64;
            let (mut i, mut j) = (0, 0);
            while i < a.len() && j < b.len() {
                if a[i].0 == b[j].0 {
                    expected += a[i].1.min(b[j].1) as f64;
                }
                let (x, y) = (a[i].0, b[j].0);
                i += usize::from(x <= y);
                j += usize::from(y <= x);
            }
            // SAFETY: AVX2 was detected above.
            let observed = unsafe { intersection_min_sum_avx2(&counter_a, &counter_b) } as f64;
            assert_eq!(
                observed.to_bits(),
                expected.to_bits(),
                "a={} b={}",
                a.len(),
                b.len()
            );
            checked += usize::from(expected > 0.0);
        }
        assert!(checked > 10_000);
    }

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

    #[test]
    fn counter_data_matches_the_map_form() {
        use super::char_ngram_counter_data;
        use crate::counter_data_from_usize_map;
        use crate::HashSet;

        let alphabet: Vec<char> = "ab c  déé文字ßxyz\u{1F600}\u{0}".chars().collect();
        let stopwords: HashSet<String> = ["ab", "the", "c"]
            .iter()
            .map(|word| word.to_string())
            .collect();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % bound as u64) as usize
        };
        let mut texts: Vec<String> = vec![String::new(), " ".into(), "a".into(), "the ab c".into()];
        for _ in 0..3000 {
            let length = next(40);
            texts.push(
                (0..length)
                    .map(|_| alphabet[next(alphabet.len())])
                    .collect(),
            );
        }
        for text in &texts {
            for flags in 0..8 {
                let (unigrams, bigrams, drop_short) =
                    (flags & 1 != 0, flags & 2 != 0, flags & 4 != 0);
                for words in [None, Some(&stopwords)] {
                    let expected = counter_data_from_usize_map(char_ngrams_counter_python_compat(
                        text, unigrams, bigrams, words, drop_short,
                    ));
                    let actual =
                        char_ngram_counter_data(text, unigrams, bigrams, words, drop_short);
                    match (expected, actual) {
                        (None, None) => {}
                        (Some(expected), Some(actual)) => {
                            assert_eq!(actual.keys, expected.keys, "{text:?} {flags}");
                            assert_eq!(actual.values, expected.values, "{text:?} {flags}");
                            assert_eq!(actual.sum.to_bits(), expected.sum.to_bits(), "{text:?}");
                        }
                        (expected, actual) => panic!(
                            "{text:?} {flags}: {} vs {}",
                            expected.is_some(),
                            actual.is_some()
                        ),
                    }
                }
            }
            let word_cases = [
                (
                    counter_data_from_usize_map(super::word_ngrams_counter(text)),
                    super::word_counter_data(text),
                ),
                (
                    counter_data_from_usize_map(super::word_ngrams_counter_python_compat(
                        text, &stopwords, true,
                    )),
                    super::title_word_counter_data(text, &stopwords, true),
                ),
                (
                    counter_data_from_usize_map(super::word_ngrams_counter_python_compat(
                        text, &stopwords, false,
                    )),
                    super::title_word_counter_data(text, &stopwords, false),
                ),
            ];
            for (expected, actual) in word_cases {
                assert_eq!(
                    expected.map(|data| (data.keys, data.values, data.sum.to_bits())),
                    actual.map(|data| (data.keys, data.values, data.sum.to_bits())),
                    "words of {text:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod middle_initial_tests {
    use super::{middle_initial_counts, middle_initials_overlap, middle_initials_overlap_counts};

    #[test]
    fn counts_overlap_matches_the_per_pair_form() {
        let parts = ["a", "b", "ab", "anne", "", " ", "é", "b c", "a a", "de"];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % bound as u64) as usize
        };
        let mut names: Vec<String> = vec![String::new(), " ".into(), "a".into()];
        for _ in 0..300 {
            let count = next(5);
            names.push(
                (0..count)
                    .map(|_| parts[next(parts.len())])
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
        let (mut nan, mut finite) = (0, 0);
        for name1 in &names {
            let counts1 = middle_initial_counts(name1);
            for name2 in &names {
                let counts2 = middle_initial_counts(name2);
                let expected = middle_initials_overlap(Some(name1), Some(name2));
                let actual = middle_initials_overlap_counts(&counts1, &counts2);
                assert_eq!(actual.to_bits(), expected.to_bits(), "{name1:?} {name2:?}");
                if actual.is_nan() {
                    nan += 1;
                } else {
                    finite += 1;
                }
            }
        }
        assert!(nan > 0 && finite > 0);
    }
}

/// Sum of `min(value_a, value_b)` over the keys two counters share, for
/// counters whose `simd` flag holds: four keys per side are compared all
/// against all per step, and integer counts sum exactly in any order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn intersection_min_sum_avx2(a: &CounterData, b: &CounterData) -> u64 {
    use std::arch::x86_64::*;
    let (na, nb) = (a.len(), b.len());
    let (mut i, mut j) = (0usize, 0usize);
    let mut acc = _mm_setzero_si128();
    // Low 32-bit lanes of the four 64-bit compare lanes.
    let pack_lows = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    while i + 4 <= na && j + 4 <= nb {
        let ak = _mm256_loadu_si256(a.keys.as_ptr().add(i) as *const __m256i);
        let bk = _mm256_loadu_si256(b.keys.as_ptr().add(j) as *const __m256i);
        let av = _mm_cvttps_epi32(_mm_loadu_ps(a.values.as_ptr().add(i)));
        let bv = _mm_cvttps_epi32(_mm_loadu_ps(b.values.as_ptr().add(j)));
        macro_rules! rotation {
            ($keys:expr, $values:expr) => {{
                let equal = _mm256_cmpeq_epi64(ak, $keys);
                let equal32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(equal, pack_lows));
                acc = _mm_add_epi32(acc, _mm_and_si128(_mm_min_epi32(av, $values), equal32));
            }};
        }
        rotation!(bk, bv);
        rotation!(
            _mm256_permute4x64_epi64(bk, 0b00_11_10_01),
            _mm_shuffle_epi32(bv, 0b00_11_10_01)
        );
        rotation!(
            _mm256_permute4x64_epi64(bk, 0b01_00_11_10),
            _mm_shuffle_epi32(bv, 0b01_00_11_10)
        );
        rotation!(
            _mm256_permute4x64_epi64(bk, 0b10_01_00_11),
            _mm_shuffle_epi32(bv, 0b10_01_00_11)
        );
        let a_last = *a.keys.get_unchecked(i + 3);
        let b_last = *b.keys.get_unchecked(j + 3);
        i += 4 * usize::from(a_last <= b_last);
        j += 4 * usize::from(b_last <= a_last);
    }
    let mut lanes = [0i32; 4];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, acc);
    let mut total: u64 = lanes.iter().map(|lane| *lane as u64).sum();
    while i < na && j < nb {
        let (ka, kb) = (*a.keys.get_unchecked(i), *b.keys.get_unchecked(j));
        if ka == kb {
            total += a.values.get_unchecked(i).min(*b.values.get_unchecked(j)) as u64;
        }
        i += usize::from(ka <= kb);
        j += usize::from(kb <= ka);
    }
    total
}

pub(crate) fn counter_jaccard_data(
    counter1: &Option<CounterData>,
    counter2: &Option<CounterData>,
    denom_max: f64,
) -> f64 {
    let (Some(c1), Some(c2)) = (counter1.as_ref(), counter2.as_ref()) else {
        return f64::NAN;
    };
    if c1.is_empty() || c2.is_empty() {
        return f64::NAN;
    }
    let (small, large) = if c1.len() <= c2.len() {
        (c1, c2)
    } else {
        (c2, c1)
    };
    let mut intersection = 0.0f64;
    if c1.simd
        && c2.simd
        && large.len() < small.len().saturating_mul(4)
        && std::arch::is_x86_feature_detected!("avx2")
    {
        // Integer counts: the exact sum equals the merge's f64 sum.
        // SAFETY: AVX2 was detected above.
        intersection = unsafe { intersection_min_sum_avx2(c1, c2) } as f64;
    } else if large.len() < small.len().saturating_mul(4) {
        let mut small_idx = 0usize;
        let mut large_idx = 0usize;
        while small_idx < small.len() && large_idx < large.len() {
            let (small_hash, small_value) = (small.keys[small_idx], small.values[small_idx]);
            let (large_hash, large_value) = (large.keys[large_idx], large.values[large_idx]);
            if small_hash == large_hash {
                intersection += small_value.min(large_value) as f64;
            }
            // Advance without a hash-dependent branch; equal hashes move both,
            // so the sum keeps its order.
            small_idx += usize::from(small_hash <= large_hash);
            large_idx += usize::from(large_hash <= small_hash);
        }
    } else {
        for (h, v1) in small.iter() {
            if let Ok(idx) = large.keys.binary_search(h) {
                intersection += (*v1).min(large.values[idx]) as f64;
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

/// Counts of a middle name's initials, sorted by initial.
pub(crate) fn middle_initial_counts(middle: &str) -> Vec<(char, u32)> {
    let mut counts: Vec<(char, u32)> = count_initials(middle)
        .into_iter()
        .map(|(initial, count)| (initial, count as u32))
        .collect();
    counts.sort_unstable();
    counts
}

/// `middle_initials_overlap` on two signatures' sorted initial counts.
///
/// Building two hash maps per pair took 2.6% of a full block's CPU. The sums
/// are integers, so the merge order does not change the value.
pub(crate) fn middle_initials_overlap_counts(
    counts1: &[(char, u32)],
    counts2: &[(char, u32)],
) -> f64 {
    if counts1.is_empty() || counts2.is_empty() {
        return f64::NAN;
    }
    let mut intersection_sum = 0u64;
    let (mut i, mut j) = (0, 0);
    while i < counts1.len() && j < counts2.len() {
        let (initial1, value1) = counts1[i];
        let (initial2, value2) = counts2[j];
        if initial1 == initial2 {
            intersection_sum += u64::from(value1.min(value2));
        }
        i += usize::from(initial1 <= initial2);
        j += usize::from(initial2 <= initial1);
    }
    let sum1: u64 = counts1.iter().map(|(_, value)| u64::from(*value)).sum();
    let sum2: u64 = counts2.iter().map(|(_, value)| u64::from(*value)).sum();
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

/// The per-pair form of `middle_initials_overlap_counts`; its reference in tests.
#[cfg(test)]
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
    if n1.is_empty() || n2.is_empty() {
        return f64::NAN;
    }
    // When either side is a single-character initial, compare the sets of token
    // initials so a joined multi-token middle ("james lee") matches the other
    // side's initial for ANY of its tokens, not just the first one.
    if py_len_is_one(n1) || py_len_is_one(n2) {
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
    let val = n1.is_empty() != n2.is_empty();
    if val {
        1.0
    } else {
        0.0
    }
}

pub(crate) fn single_char_first(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let n1 = name1.unwrap_or("");
    let n2 = name2.unwrap_or("");
    let val = py_len_is_one(n1) || py_len_is_one(n2);
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
        if py_len_is_one(part) {
            val = true;
            break;
        }
    }
    if !val {
        for part in n2.split(' ') {
            if py_len_is_one(part) {
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
        || email.chars().any(is_python_whitespace_compat)
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
    if e1.is_empty() || e2.is_empty() {
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
    if n1.is_empty() || n2.is_empty() {
        return [f64::NAN; 4];
    }
    let lev = levenshtein_distance(n1, n2) as f64 / (py_len(n1).max(py_len(n2)) as f64);
    let pref = prefix_dist(n1, n2);
    let lcs = metric_lcs_distance(n1, n2);
    let jaro = jaro_winkler_similarity(n1, n2, false);
    [lev, pref, lcs, jaro]
}

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
        HashMap::with_capacity_and_hasher(matrix_indices.len(), Default::default());
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

    #[test]
    fn python_whitespace_in_emails_is_missing_evidence() {
        for separator in ['\u{001c}', '\u{001d}', '\u{001e}', '\u{001f}'] {
            let malformed = format!("a{separator}@b");
            assert_eq!(
                email_parts(&malformed),
                None,
                "separator U+{:04X}",
                separator as u32
            );
        }
        for malformed in ["a\u{001c}\u{001d}\u{001e}\u{001f}@b", "a\u{00a0}@b"] {
            assert_eq!(email_parts(malformed), None, "{malformed:?}");
        }
        assert_eq!(email_parts("A@B"), Some(("a".into(), "b".into())));
    }
}
