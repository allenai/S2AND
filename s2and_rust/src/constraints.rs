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
    let mut ita = a.split_whitespace();
    let mut itb = b.split_whitespace();
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
    // canonical first names, so the fields are probed exactly as given. The
    // legacy joined/first-token probes were shims for pre-canonical multi-token
    // fields and are retired.
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
}
