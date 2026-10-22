use crate::{HashMap, HashSet};

pub(crate) fn count_initials(s: &str) -> HashMap<char, usize> {
    let mut counts = HashMap::default();
    for part in s.split(' ') {
        if !part.is_empty() {
            if let Some(ch) = part.chars().next() {
                *counts.entry(ch).or_insert(0) += 1;
            }
        }
    }
    counts
}

#[cfg(test)]
// Compare-time equivalence for canonical last names: joined and spaced
// spellings of one surname are equivalent ("ou yang" == "ouyang"). Deliberate
// canonical_v2 compare-time policy (not a legacy artifact shim); mirrors
// s2and.text.canonical_lasts_equivalent.
pub(crate) fn lasts_equivalent_for_constraint(l1: &str, l2: &str) -> bool {
    if l1 == l2 {
        return true;
    }
    l1.replace(' ', "") == l2.replace(' ', "")
}

pub(crate) fn same_prefix_tokens(a: &str, b: &str) -> bool {
    same_prefix_token_iters(a.split_whitespace(), b.split_whitespace())
}

/// Name inputs of the hard constraint for one signature, split once.
///
/// Blocks compare every pair, so splitting per signature instead of per pair
/// removes most of the constraint's string work.
pub(crate) struct ConstraintNameKey<'a> {
    last: &'a str,
    /// The last name without spaces; it borrows `last` when there are none.
    last_compact: std::borrow::Cow<'a, str>,
    first_initial: Option<char>,
    first_tokens: Vec<&'a str>,
    middle_tokens: Vec<&'a str>,
}

impl<'a> ConstraintNameKey<'a> {
    pub(crate) fn new(first: &'a str, middle: &'a str, last: &'a str) -> Self {
        let last_compact = if last.contains(' ') {
            std::borrow::Cow::Owned(last.replace(' ', ""))
        } else {
            std::borrow::Cow::Borrowed(last)
        };
        Self {
            last,
            last_compact,
            first_initial: if first.is_empty() {
                None
            } else {
                first.chars().next()
            },
            first_tokens: first.split_whitespace().collect(),
            middle_tokens: middle.split_whitespace().collect(),
        }
    }
}

/// Return whether two signatures' names violate the hard constraint.
///
/// Matches the name steps of `RustFeaturizer::constraint_value_from_records`:
/// last names, first initials, first-name compatibility, then middle names.
/// `first_aliases` answers the name-tuple lookup, which only runs for first
/// names that share no prefix.
pub(crate) fn constraint_names_conflict(
    key_1: &ConstraintNameKey<'_>,
    key_2: &ConstraintNameKey<'_>,
    first_aliases: impl FnOnce() -> bool,
) -> bool {
    if key_1.last != key_2.last && key_1.last_compact != key_2.last_compact {
        return true;
    }
    if let (Some(c1), Some(c2)) = (key_1.first_initial, key_2.first_initial) {
        if c1 != c2 {
            return true;
        }
    }
    let has_first_1 = !key_1.first_tokens.is_empty();
    let has_first_2 = !key_2.first_tokens.is_empty();
    let same_prefix = has_first_1
        && has_first_2
        && same_prefix_token_iters(
            key_1.first_tokens.iter().copied(),
            key_2.first_tokens.iter().copied(),
        );
    if has_first_1 && has_first_2 && !same_prefix && !first_aliases() {
        return true;
    }
    let has_tail = key_1.first_tokens.len() > 1 || key_2.first_tokens.len() > 1;
    if has_tail && (!has_first_1 || !has_first_2 || same_prefix) {
        let mut tokens_1 = key_1.first_tokens
            [uncovered_first_tail_start(&key_1.first_tokens, &key_2.first_tokens)..]
            .to_vec();
        let mut tokens_2 = key_2.first_tokens
            [uncovered_first_tail_start(&key_2.first_tokens, &key_1.first_tokens)..]
            .to_vec();
        tokens_1.extend_from_slice(&key_1.middle_tokens);
        tokens_2.extend_from_slice(&key_2.middle_tokens);
        return middle_names_conflict_small(&tokens_1, &tokens_2);
    }
    middle_names_conflict_small(&key_1.middle_tokens, &key_2.middle_tokens)
}

/// Token and character capacities of the stack summaries in
/// `middle_names_conflict_small`; longer names use `middle_names_conflict`.
const SMALL_MIDDLE_TOKENS: usize = 16;
const SMALL_MIDDLE_CHARS: usize = 96;

/// The sorted, deduplicated summary of one side's middle tokens.
struct SmallMiddleSummary<'a> {
    words: [&'a str; SMALL_MIDDLE_TOKENS],
    word_count: usize,
    initials: [char; SMALL_MIDDLE_TOKENS],
    initial_count: usize,
    chars: [char; SMALL_MIDDLE_CHARS],
    char_count: usize,
}

impl<'a> SmallMiddleSummary<'a> {
    /// Summarize `tokens` like `middle_token_summary`, skipping dropped
    /// affixes that both sides contain. Return None past the capacities.
    fn new(tokens: &[&'a str], other: &[&str]) -> Option<Self> {
        let mut summary = Self {
            words: [""; SMALL_MIDDLE_TOKENS],
            word_count: 0,
            initials: ['\0'; SMALL_MIDDLE_TOKENS],
            initial_count: 0,
            chars: ['\0'; SMALL_MIDDLE_CHARS],
            char_count: 0,
        };
        for token in tokens.iter().copied() {
            if token.is_empty() || (crate::is_dropped_affix(token) && other.contains(&token)) {
                continue;
            }
            let mut chars = token.chars();
            if let Some(initial) = chars.next() {
                summary.initials[summary.initial_count] = initial;
                summary.initial_count += 1;
            }
            if chars.next().is_some() {
                summary.words[summary.word_count] = token;
                summary.word_count += 1;
                for ch in token.chars() {
                    if summary.char_count == SMALL_MIDDLE_CHARS {
                        return None;
                    }
                    summary.chars[summary.char_count] = ch;
                    summary.char_count += 1;
                }
            }
        }
        summary.word_count = sort_dedup(&mut summary.words[..summary.word_count]);
        summary.initial_count = sort_dedup(&mut summary.initials[..summary.initial_count]);
        summary.char_count = sort_dedup(&mut summary.chars[..summary.char_count]);
        Some(summary)
    }

    fn words(&self) -> &[&'a str] {
        &self.words[..self.word_count]
    }

    fn initials(&self) -> &[char] {
        &self.initials[..self.initial_count]
    }

    fn chars(&self) -> &[char] {
        &self.chars[..self.char_count]
    }
}

/// Sort `values` and move its distinct values to the front; return their count.
fn sort_dedup<T: Ord + Copy>(values: &mut [T]) -> usize {
    values.sort_unstable();
    let mut count = 0;
    for index in 0..values.len() {
        if count == 0 || values[index] != values[count - 1] {
            values[count] = values[index];
            count += 1;
        }
    }
    count
}

/// Return whether two sorted, deduplicated slices share no value.
fn sorted_disjoint<T: Ord>(left: &[T], right: &[T]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return false,
        }
    }
    true
}

/// `middle_names_conflict` on stack arrays instead of hash sets.
///
/// Pairs dominate constraint time, and their middle names are short, so the
/// summaries fit on the stack. Longer names take the hash-set version.
pub(crate) fn middle_names_conflict_small(middle_1: &[&str], middle_2: &[&str]) -> bool {
    if middle_1.is_empty() || middle_2.is_empty() {
        return false;
    }
    if middle_1.len() > SMALL_MIDDLE_TOKENS || middle_2.len() > SMALL_MIDDLE_TOKENS {
        return middle_names_conflict(middle_1, middle_2);
    }
    let (Some(summary_1), Some(summary_2)) = (
        SmallMiddleSummary::new(middle_1, middle_2),
        SmallMiddleSummary::new(middle_2, middle_1),
    ) else {
        return middle_names_conflict(middle_1, middle_2);
    };
    let conflicting_initials = !summary_1.initials().is_empty()
        && !summary_2.initials().is_empty()
        && sorted_disjoint(summary_1.initials(), summary_2.initials());
    let conflicting_full_names = !summary_1.words().is_empty()
        && !summary_2.words().is_empty()
        && sorted_disjoint(summary_1.words(), summary_2.words())
        && summary_1.chars() != summary_2.chars()
        && !same_prefix_token_iters(middle_1.iter().copied(), middle_2.iter().copied());
    conflicting_initials || conflicting_full_names
}

fn same_prefix_token_iters<'a>(
    mut ita: impl Iterator<Item = &'a str>,
    mut itb: impl Iterator<Item = &'a str>,
) -> bool {
    let mut saw_pair = false;
    loop {
        match (ita.next(), itb.next()) {
            (Some(x), Some(y)) => {
                saw_pair = true;
                if !(x.starts_with(y) || y.starts_with(x)) {
                    return false;
                }
            }
            _ => return saw_pair,
        }
    }
}

/// Return the index where the uncovered later first-name tokens start.
///
/// Mirrors `s2and.text._uncovered_first_tail`: the other first name covers its
/// aligned positions, and the remainder of its last token covers the next
/// tokens ("xiaoqing" covers "xiao qing"). The first token never spills.
fn uncovered_first_tail_start(tokens: &[&str], other: &[&str]) -> usize {
    let mut start = other.len().max(1);
    if start >= tokens.len() {
        return tokens.len();
    }
    if let Some(last) = other.last() {
        let aligned = tokens[other.len() - 1];
        let mut rest = last.strip_prefix(aligned).unwrap_or("");
        while !rest.is_empty() && start < tokens.len() {
            let token = tokens[start];
            if let Some(remaining) = rest.strip_prefix(token) {
                rest = remaining;
            } else if token.starts_with(rest) {
                rest = "";
            } else {
                break;
            }
            start += 1;
        }
    }
    start
}

#[cfg(test)]
/// Return the middle-name tokens that the hard name constraint compares.
///
/// Mirrors `s2and.text.constraint_middle_tokens`. Call it only for compatible
/// first names.
pub(crate) fn constraint_middle_tokens<'a>(
    first_1: &'a str,
    middle_1: &'a str,
    first_2: &'a str,
    middle_2: &'a str,
) -> (Vec<&'a str>, Vec<&'a str>) {
    let mut tokens_1: Vec<&'a str> = Vec::new();
    let mut tokens_2: Vec<&'a str> = Vec::new();
    // A dash-bound first group ("jean daniel") matches by prefix only on its
    // aligned tokens. Its uncovered later tokens must also pass the middle
    // rule, as for "jean" + "daniel". An alias match covers the full first
    // name. Single-token first names have no tail, so most pairs skip this.
    let has_tail =
        first_1.split_whitespace().nth(1).is_some() || first_2.split_whitespace().nth(1).is_some();
    if has_tail
        && (first_1.split_whitespace().next().is_none()
            || first_2.split_whitespace().next().is_none()
            || same_prefix_tokens(first_1, first_2))
    {
        let first_tokens_1: Vec<&'a str> = first_1.split_whitespace().collect();
        let first_tokens_2: Vec<&'a str> = first_2.split_whitespace().collect();
        tokens_1.extend_from_slice(
            &first_tokens_1[uncovered_first_tail_start(&first_tokens_1, &first_tokens_2)..],
        );
        tokens_2.extend_from_slice(
            &first_tokens_2[uncovered_first_tail_start(&first_tokens_2, &first_tokens_1)..],
        );
    }
    tokens_1.extend(middle_1.split_whitespace());
    tokens_2.extend(middle_2.split_whitespace());
    (tokens_1, tokens_2)
}

/// Return the full words, initials and full-word letters of middle tokens.
fn middle_token_summary<'a>(
    tokens: &[&'a str],
    overlapping_affixes: &HashSet<&str>,
) -> (HashSet<&'a str>, HashSet<char>, HashSet<char>) {
    let mut words = HashSet::default();
    let mut initials = HashSet::default();
    let mut word_chars = HashSet::default();
    for token in tokens.iter().copied() {
        if token.is_empty() || overlapping_affixes.contains(token) {
            continue;
        }
        if let Some(initial) = token.chars().next() {
            initials.insert(initial);
        }
        if crate::py_len(token) > 1 {
            words.insert(token);
            word_chars.extend(token.chars());
        }
    }
    (words, initials, word_chars)
}

/// Return whether two middle-name token lists violate the hard name rule.
///
/// Mirrors the middle-name step of `ANDData._get_constraint`.
pub(crate) fn middle_names_conflict(middle_1: &[&str], middle_2: &[&str]) -> bool {
    if middle_1.is_empty() || middle_2.is_empty() {
        return false;
    }
    let middle_1_set: HashSet<&str> = middle_1.iter().copied().collect();
    let middle_2_set: HashSet<&str> = middle_2.iter().copied().collect();
    let overlapping_affixes: HashSet<&str> = middle_1_set
        .intersection(&middle_2_set)
        .copied()
        .filter(|token| crate::is_dropped_affix(token))
        .collect();
    let (words_1, initials_1, chars_1) = middle_token_summary(middle_1, &overlapping_affixes);
    let (words_2, initials_2, chars_2) = middle_token_summary(middle_2, &overlapping_affixes);
    let conflicting_initials =
        !initials_1.is_empty() && !initials_2.is_empty() && initials_1.is_disjoint(&initials_2);
    let conflicting_full_names = !words_1.is_empty()
        && !words_2.is_empty()
        && words_1.is_disjoint(&words_2)
        && chars_1 != chars_2
        && !same_prefix_token_iters(middle_1.iter().copied(), middle_2.iter().copied());
    conflicting_initials || conflicting_full_names
}

#[cfg(test)]
pub(crate) fn name_tuple_contains(
    map: &HashMap<String, HashSet<String>>,
    a: &str,
    b: &str,
) -> bool {
    // Alias ingestion stores both directions, so the hot constraint path needs
    // one hash lookup while remaining order-independent.
    map.get(a).map_or(false, |vals| vals.contains(b))
}

#[cfg(test)]
pub(crate) fn first_names_name_compatible(
    first_1: &str,
    first_2: &str,
    name_tuples: &HashMap<String, HashSet<String>>,
) -> bool {
    if first_1.split_whitespace().next().is_none() || first_2.split_whitespace().next().is_none() {
        // Missing first-name evidence is unknown, not an incompatibility signal.
        return true;
    }
    if same_prefix_tokens(first_1, first_2) {
        return true;
    }
    // canonical_v2: the regenerated name-tuple artifact is keyed on full
    // canonical first names, so the fields are probed exactly as given.
    name_tuple_contains(name_tuples, first_1, first_2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_tuple_compatibility_does_not_apply_extra_case_normalization() {
        let mut name_tuples = HashMap::default();
        crate::insert_name_tuple_alias(&mut name_tuples, "Bill".to_string(), "William".to_string());

        assert!(first_names_name_compatible("Bill", "William", &name_tuples));
        assert!(first_names_name_compatible("William", "Bill", &name_tuples));
        assert!(!first_names_name_compatible(
            "bill",
            "william",
            &name_tuples
        ));
    }

    #[test]
    fn empty_prefix_tokens_are_not_positive_evidence() {
        assert!(!same_prefix_tokens("", "alice"));
        assert!(!same_prefix_tokens("", ""));
        assert!(first_names_name_compatible(
            "",
            "alice",
            &HashMap::default()
        ));
    }

    fn conflict(first_1: &str, middle_1: &str, first_2: &str, middle_2: &str) -> bool {
        let (tokens_1, tokens_2) = constraint_middle_tokens(first_1, middle_1, first_2, middle_2);
        middle_names_conflict(&tokens_1, &tokens_2)
    }

    #[test]
    fn dash_bound_first_tail_joins_the_middle_comparison() {
        assert!(!conflict("j d", "t", "j", "d"));
        assert!(conflict("chi hung", "", "c", "m"));
        assert!(!conflict("jean daniel", "", "jean", "daniel"));
        assert!(conflict("jean daniel", "", "jean", "paul"));
        assert!(!conflict("j d", "t", "jean daniel", ""));
        assert!(!conflict("", "d", "j d", ""));
    }

    #[test]
    fn covered_first_tokens_stay_out_of_the_middle_comparison() {
        assert!(conflict("jean pierre", "a", "jean pierre", "b"));
        assert!(conflict("j d", "t", "j d", "s"));
        assert!(!conflict("xiao qing", "", "xiaoqing", "frank"));
        assert!(conflict("j d", "t", "jd", "s"));
        assert_eq!(
            constraint_middle_tokens("j d", "t", "jean", "d"),
            (vec!["d", "t"], vec!["d"])
        );
    }

    #[test]
    fn alias_matched_first_tail_stays_out_of_the_middle_comparison() {
        assert_eq!(
            constraint_middle_tokens("abol ghasem", "", "ali", "r"),
            (Vec::<&str>::new(), vec!["r"])
        );
        assert!(!conflict("abol ghasem", "", "ali", "r"));
        assert!(conflict("abol ghasem", "m", "ali", "r"));
    }
}
