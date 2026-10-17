use std::collections::{HashMap, HashSet};

pub(crate) fn count_initials(s: &str) -> HashMap<char, usize> {
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
    let mut words = HashSet::new();
    let mut initials = HashSet::new();
    let mut word_chars = HashSet::new();
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

pub(crate) fn name_tuple_contains(
    map: &HashMap<String, HashSet<String>>,
    a: &str,
    b: &str,
) -> bool {
    // Alias ingestion stores both directions, so the hot constraint path needs
    // one hash lookup while remaining order-independent.
    map.get(a).map_or(false, |vals| vals.contains(b))
}

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
        let mut name_tuples = HashMap::new();
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
        assert!(first_names_name_compatible("", "alice", &HashMap::new()));
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
