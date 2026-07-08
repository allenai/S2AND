use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::constraints::same_prefix_tokens;
use crate::features::{compute_name_counts_data, word_ngrams_counter};
use crate::name_counts::{NameCountsData, NameCountsLastFirstInitialSemantics, RawNameCountMaps};
use crate::orcid::normalize_orcid_owned;
use crate::raw_arrow::readers::{
    RawArrowAuthorSignalData, RawArrowFeature, RawArrowNameCountRarityRow, RawArrowPaper,
    RawArrowPaperEvidenceRow, RawArrowSignature, RawArrowSummarySignalData,
};
use crate::text_compat::{
    compute_block_compat, normalize_text_compat_from_map, split_first_middle_hyphen_aware_compat,
};
use crate::{
    build_name_counts_data_from_artifact, counter_data_from_hash_count_map,
    fnv64, hash_string_values, prefilter_affiliation_text, py_len, query_terms_from_values,
    raw_arrow_year_mean, term_set_from_normalized_text, validate_row_signal_year, vector_norm_f32,
    RetrievalQueryData, RetrievalSummaryData, RETRIEVAL_MEGA_AUTHOR_THRESHOLD,
};
pub(crate) fn build_raw_arrow_feature(
    signature: &RawArrowSignature,
    paper: Option<&RawArrowPaper>,
    paper_authors: Option<&Vec<(i64, String)>>,
    specter_by_paper_id: Option<&HashMap<String, Arc<Vec<f32>>>>,
    raw_name_counts: &RawNameCountMaps,
    name_prefixes: &HashSet<String>,
    affiliation_stopwords: &HashSet<String>,
    unidecode_char_map: &HashMap<char, String>,
    orcid_enabled: bool,
) -> RawArrowFeature {
    let (first, middle) = split_first_middle_hyphen_aware_compat(
        &signature.author_first,
        &signature.author_middle,
        name_prefixes,
        unidecode_char_map,
    );
    let last_normalized =
        normalize_text_compat_from_map(&signature.author_last, false, unidecode_char_map);
    let name_counts = build_name_counts_data_from_artifact(
        raw_name_counts,
        &signature.author_first,
        &first,
        &signature.author_last,
        &last_normalized,
        // Arrow datasets always use the current `<last> <first-initial>` lookup-key form;
        // legacy `<last> <full-first-token>` semantics only apply to Python ANDData ingest.
        NameCountsLastFirstInitialSemantics::InitialChar,
    );
    let middle_initials: HashSet<char> = middle
        .split_whitespace()
        .filter_map(|token| token.chars().next())
        .collect();

    let mut coauthor_blocks = HashSet::new();
    let mut paper_author_count = 0usize;
    if let Some(authors) = paper_authors {
        paper_author_count = authors.len();
        if let Some(signature_position) = signature.position {
            for (position, author_name) in authors.iter() {
                let normalized =
                    normalize_text_compat_from_map(author_name, false, unidecode_char_map);
                if *position == signature_position {
                    continue;
                }
                if normalized.is_empty() {
                    continue;
                }
                let block = compute_block_compat(&normalized);
                if !block.is_empty() {
                    coauthor_blocks.insert(block);
                }
            }
        }
    }

    let mut normalized_affiliations = Vec::with_capacity(signature.affiliations.len());
    for affiliation in signature.affiliations.iter() {
        let normalized = normalize_text_compat_from_map(affiliation, false, unidecode_char_map);
        if !normalized.is_empty() {
            normalized_affiliations.push(normalized);
        }
    }
    let affiliation_text =
        prefilter_affiliation_text(&normalized_affiliations, affiliation_stopwords);
    let affiliation_terms: HashSet<String> =
        word_ngrams_counter(&affiliation_text).into_keys().collect();

    let (venue_terms, title_terms, year) = if let Some(paper_data) = paper {
        let venue_text = [paper_data.venue.as_str(), paper_data.journal_name.as_str()]
            .iter()
            .filter(|part| !part.trim().is_empty())
            .copied()
            .collect::<Vec<_>>()
            .join(" ");
        let normalized_venue =
            normalize_text_compat_from_map(&venue_text, false, unidecode_char_map);
        let normalized_title =
            normalize_text_compat_from_map(&paper_data.title, false, unidecode_char_map);
        (
            term_set_from_normalized_text(&normalized_venue),
            term_set_from_normalized_text(&normalized_title),
            paper_data.year,
        )
    } else {
        (HashSet::new(), HashSet::new(), None)
    };

    let coauthor_terms = query_terms_from_values(&coauthor_blocks);
    let coauthor_hashes = coauthor_terms.iter().map(|term| term.hash).collect();
    let affiliation_query_terms = query_terms_from_values(&affiliation_terms);
    let affiliation_hashes = affiliation_query_terms
        .iter()
        .map(|term| term.hash)
        .collect();
    let venue_hashes = hash_string_values(&venue_terms);
    let title_hashes = hash_string_values(&title_terms);
    let middle_initial_hashes = {
        let mut hashes: Vec<u64> = middle_initials
            .iter()
            .map(|ch| {
                let mut buf = [0u8; 4];
                fnv64(ch.encode_utf8(&mut buf).as_bytes())
            })
            .collect();
        hashes.sort_unstable();
        hashes.dedup();
        hashes
    };
    // ORCID enablement has two equivalent control surfaces and they should always agree:
    //   1. Per-request: `orcid_enabled` here, derived from `not clusterer.suppress_orcid`
    //      in [s2and/incremental_linking/production.py] and threaded through the planner.
    //   2. Per-ingest: Python `ANDData(use_orcid_id=False)` strips ORCIDs at signature
    //      build time (see s2and/data.py author_info_orcid handling), used by offline
    //      training data prep only.
    // When `orcid_enabled=false`, the kernel suppresses ORCID at hash time so the
    // signature.orcid value is irrelevant. When `orcid_enabled=true`, the kernel honors
    // whatever the ingest layer produced — a None signature.orcid is "no ORCID for this
    // signature", not an error.
    let orcid_hash = if orcid_enabled {
        signature
            .orcid
            .as_ref()
            .and_then(|value| normalize_orcid_owned(value).map(|orcid| fnv64(orcid.as_bytes())))
    } else {
        None
    };
    let specter = specter_by_paper_id
        .and_then(|values| values.get(&signature.paper_id))
        .map(Arc::clone);
    let specter_norm = specter.as_ref().map(|values| vector_norm_f32(values));
    let query_author = [
        signature.author_first.as_str(),
        signature.author_middle.as_str(),
        signature.author_last.as_str(),
        signature.author_suffix.as_str(),
    ]
    .iter()
    .filter(|value| !value.trim().is_empty())
    .copied()
    .collect::<Vec<_>>()
    .join(" ");

    RawArrowFeature {
        query: RetrievalQueryData {
            has_full_first: py_len(&first) > 1,
            first,
            middle_initial_hashes,
            coauthor_hashes,
            coauthor_terms,
            affiliation_hashes,
            affiliation_terms: affiliation_query_terms,
            venue_hashes,
            title_hashes,
            year,
            orcid_hash,
            specter,
            specter_norm,
        },
        name_counts,
        paper_author_count,
        query_author,
    }
}

pub(crate) fn build_raw_arrow_author_signal_data(
    signature: &RawArrowSignature,
    paper_authors: Option<&Vec<(i64, String)>>,
    unidecode_char_map: &HashMap<char, String>,
) -> RawArrowAuthorSignalData {
    let mut paper_author_names = HashSet::new();
    let mut local10_author_names = HashSet::new();
    if let Some(authors) = paper_authors {
        for (position, author_name) in authors.iter() {
            let normalized = normalize_text_compat_from_map(author_name, false, unidecode_char_map);
            if normalized.is_empty() {
                continue;
            }
            paper_author_names.insert(normalized.clone());
            if let Some(author_position) = signature.position {
                if *position != author_position && (*position).abs_diff(author_position) <= 10 {
                    local10_author_names.insert(normalized);
                }
            }
        }
    }
    RawArrowAuthorSignalData {
        paper_author_names,
        local10_author_names,
    }
}

pub(crate) fn mask_raw_arrow_query(
    base: &RetrievalQueryData,
    requested_view: &str,
) -> Result<(RetrievalQueryData, String), String> {
    let resolved_view = if requested_view == "auto" {
        if base.has_full_first {
            "full"
        } else {
            "initial_only"
        }
    } else if requested_view == "full" || requested_view == "initial_only" {
        requested_view
    } else {
        return Err(format!("Unknown query view: {requested_view}"));
    };
    if resolved_view == "full" {
        return Ok((base.clone(), resolved_view.to_string()));
    }

    let masked = RetrievalQueryData {
        first: base
            .first
            .chars()
            .next()
            .map(|ch| ch.to_string())
            .unwrap_or_default(),
        has_full_first: false,
        middle_initial_hashes: Vec::new(),
        coauthor_hashes: base.coauthor_hashes.clone(),
        coauthor_terms: base.coauthor_terms.clone(),
        affiliation_hashes: base.affiliation_hashes.clone(),
        affiliation_terms: base.affiliation_terms.clone(),
        venue_hashes: base.venue_hashes.clone(),
        title_hashes: base.title_hashes.clone(),
        year: base.year,
        orcid_hash: base.orcid_hash,
        specter: base.specter.clone(),
        specter_norm: base.specter_norm,
    };
    Ok((masked, resolved_view.to_string()))
}

pub(crate) fn euclidean_distance_f32(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| {
            let diff = (*left_value as f64) - (*right_value as f64);
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

pub(crate) fn validate_raw_arrow_specter_dimensions(
    component_key: &str,
    vectors: &[&[f32]],
) -> Result<(), String> {
    let Some(first) = vectors.first() else {
        return Ok(());
    };
    let expected_dim = first.len();
    for vector in vectors.iter().skip(1) {
        if vector.len() != expected_dim {
            return Err(format!(
                "component_key '{}' has mixed SPECTER dimensions: expected {}, got {}",
                component_key,
                expected_dim,
                vector.len()
            ));
        }
    }
    Ok(())
}

pub(crate) fn select_raw_arrow_exemplars(
    vectors: &[&[f32]],
    max_exemplars: usize,
) -> Vec<Vec<f32>> {
    if max_exemplars == 0 || vectors.is_empty() {
        return Vec::new();
    }
    if vectors.len() <= max_exemplars {
        return vectors.iter().map(|vector| vector.to_vec()).collect();
    }
    let dim = vectors[0].len();
    let mut centroid = vec![0.0f32; dim];
    for vector in vectors.iter() {
        for (idx, value) in vector.iter().enumerate() {
            centroid[idx] += *value;
        }
    }
    for value in centroid.iter_mut() {
        *value /= vectors.len() as f32;
    }

    let mut selected = Vec::<usize>::new();
    let mut best_index = 0usize;
    let mut best_distance = f64::NEG_INFINITY;
    for (idx, vector) in vectors.iter().enumerate() {
        let distance = euclidean_distance_f32(vector, &centroid);
        if distance > best_distance {
            best_distance = distance;
            best_index = idx;
        }
    }
    selected.push(best_index);
    let mut selected_flags = vec![false; vectors.len()];
    selected_flags[best_index] = true;
    let mut min_distances = vec![f64::INFINITY; vectors.len()];
    for (idx, vector) in vectors.iter().enumerate() {
        if idx != best_index {
            min_distances[idx] = euclidean_distance_f32(vector, vectors[best_index]);
        }
    }

    while selected.len() < max_exemplars {
        let mut next_index = None;
        let mut next_distance = f64::NEG_INFINITY;
        for (idx, distance) in min_distances.iter().enumerate() {
            if selected_flags[idx] {
                continue;
            }
            if *distance > next_distance {
                next_distance = *distance;
                next_index = Some(idx);
            }
        }
        let Some(idx) = next_index else {
            break;
        };
        selected.push(idx);
        selected_flags[idx] = true;
        for (candidate_idx, vector) in vectors.iter().enumerate() {
            if selected_flags[candidate_idx] {
                continue;
            }
            let distance = euclidean_distance_f32(vector, vectors[idx]);
            if distance < min_distances[candidate_idx] {
                min_distances[candidate_idx] = distance;
            }
        }
    }
    selected
        .into_iter()
        .map(|idx| vectors[idx].to_vec())
        .collect()
}

pub(crate) fn build_raw_arrow_summary(
    component_key: &str,
    signature_ids: &[String],
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    max_exemplars: usize,
) -> Result<RetrievalSummaryData, String> {
    let mut first_name_counts: HashMap<String, usize> = HashMap::new();
    let mut middle_initial_counts: HashMap<u64, usize> = HashMap::new();
    let mut coauthor_counts: HashMap<u64, usize> = HashMap::new();
    let mut non_mega_coauthor_counts: HashMap<u64, usize> = HashMap::new();
    let mut affiliation_counts: HashMap<u64, usize> = HashMap::new();
    let mut venue_counts: HashMap<u64, usize> = HashMap::new();
    let mut title_counts: HashMap<u64, usize> = HashMap::new();
    let mut years = Vec::<i64>::new();
    let mut orcid_hashes = Vec::<u64>::new();
    let mut specter_vectors = Vec::<&[f32]>::new();
    let mut max_paper_author_count = 0usize;

    for signature_id in signature_ids {
        let feature = features_by_signature_id.get(signature_id).ok_or_else(|| {
            format!(
                "cluster seed signature_id '{}' is missing from computed raw Arrow features",
                signature_id
            )
        })?;
        if py_len(&feature.query.first) > 1 {
            *first_name_counts
                .entry(feature.query.first.clone())
                .or_insert(0) += 1;
        }
        for hash in feature.query.middle_initial_hashes.iter() {
            *middle_initial_counts.entry(*hash).or_insert(0) += 1;
        }
        for hash in feature.query.coauthor_hashes.iter() {
            *coauthor_counts.entry(*hash).or_insert(0) += 1;
            if feature.paper_author_count < RETRIEVAL_MEGA_AUTHOR_THRESHOLD {
                *non_mega_coauthor_counts.entry(*hash).or_insert(0) += 1;
            }
        }
        for hash in feature.query.affiliation_hashes.iter() {
            *affiliation_counts.entry(*hash).or_insert(0) += 1;
        }
        for hash in feature.query.venue_hashes.iter() {
            *venue_counts.entry(*hash).or_insert(0) += 1;
        }
        for hash in feature.query.title_hashes.iter() {
            *title_counts.entry(*hash).or_insert(0) += 1;
        }
        if let Some(year) = feature.query.year {
            years.push(year);
        }
        if let Some(orcid_hash) = feature.query.orcid_hash {
            orcid_hashes.push(orcid_hash);
        }
        if let Some(specter) = feature.query.specter.as_ref() {
            specter_vectors.push(specter.as_slice());
        }
        max_paper_author_count = max_paper_author_count.max(feature.paper_author_count);
    }
    validate_raw_arrow_specter_dimensions(component_key, &specter_vectors)?;

    let mut first_name_pairs: Vec<(String, u64)> = first_name_counts
        .into_iter()
        .map(|(name, count)| (name, count as u64))
        .collect();
    first_name_pairs.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    for year in years.iter() {
        validate_row_signal_year(*year, "raw Arrow summary year")?;
    }
    years.sort_unstable();
    orcid_hashes.sort_unstable();
    orcid_hashes.dedup();
    let year_min = years.first().copied();
    let year_max = years.last().copied();
    let year_mean = raw_arrow_year_mean(&years);

    let specter_centroid = if specter_vectors.is_empty() {
        None
    } else {
        let dim = specter_vectors[0].len();
        let mut centroid = vec![0.0f32; dim];
        for vector in specter_vectors.iter() {
            for (idx, value) in vector.iter().enumerate() {
                centroid[idx] += *value;
            }
        }
        for value in centroid.iter_mut() {
            *value /= specter_vectors.len() as f32;
        }
        Some(centroid)
    };
    let specter_centroid_norm = specter_centroid
        .as_ref()
        .map(|values| vector_norm_f32(values));
    let exemplar_vectors = select_raw_arrow_exemplars(&specter_vectors, max_exemplars);
    let exemplar_norms = exemplar_vectors
        .iter()
        .map(|values| vector_norm_f32(values))
        .collect();

    Ok(RetrievalSummaryData {
        component_key: component_key.to_string(),
        size: signature_ids.len(),
        first_name_counts: first_name_pairs,
        middle_initial_counts: counter_data_from_hash_count_map(middle_initial_counts),
        coauthor_counts: counter_data_from_hash_count_map(coauthor_counts),
        non_mega_coauthor_counts: counter_data_from_hash_count_map(non_mega_coauthor_counts),
        affiliation_counts: counter_data_from_hash_count_map(affiliation_counts),
        venue_counts: counter_data_from_hash_count_map(venue_counts),
        title_counts: counter_data_from_hash_count_map(title_counts),
        max_paper_author_count,
        year_min,
        year_max,
        year_mean,
        orcid_hashes,
        specter_centroid,
        specter_centroid_norm,
        exemplar_vectors,
        exemplar_norms,
    })
}

pub(crate) fn build_raw_arrow_summary_signals(
    signature_ids: &[String],
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    signatures: &HashMap<String, RawArrowSignature>,
    paper_authors: &HashMap<String, Vec<(i64, String)>>,
    unidecode_char_map: &HashMap<char, String>,
) -> Result<RawArrowSummarySignalData, String> {
    let mut name_counts_values = Vec::<NameCountsData>::new();
    let mut member_paper_author_names = Vec::<HashSet<String>>::with_capacity(signature_ids.len());
    let mut member_paper_author_counts = Vec::<usize>::with_capacity(signature_ids.len());
    let mut member_local10_author_names =
        Vec::<HashSet<String>>::with_capacity(signature_ids.len());
    let mut member_signature_ids = Vec::<String>::with_capacity(signature_ids.len());

    for signature_id in signature_ids {
        let feature = features_by_signature_id.get(signature_id).ok_or_else(|| {
            format!(
                "cluster seed signature_id '{}' is missing from computed raw Arrow features",
                signature_id
            )
        })?;
        if let Some(name_counts) = feature.name_counts.as_ref() {
            name_counts_values.push(name_counts.clone());
        }
        let signature = signatures.get(signature_id).ok_or_else(|| {
            format!(
                "cluster seed signature_id '{}' is missing from signatures",
                signature_id
            )
        })?;
        let author_signals = build_raw_arrow_author_signal_data(
            signature,
            paper_authors.get(&signature.paper_id),
            unidecode_char_map,
        );
        member_paper_author_names.push(author_signals.paper_author_names);
        member_paper_author_counts.push(feature.paper_author_count);
        member_local10_author_names.push(author_signals.local10_author_names);
        member_signature_ids.push(signature_id.clone());
    }

    Ok(RawArrowSummarySignalData {
        name_counts_values,
        member_paper_author_names,
        member_paper_author_counts,
        member_local10_author_names,
        member_signature_ids,
    })
}

pub(crate) fn round_six(value: f64) -> f32 {
    ((value * 1_000_000.0).round() / 1_000_000.0) as f32
}

pub(crate) fn valid_positive_finite(value: f64) -> Option<f64> {
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

pub(crate) fn update_minimum(target: &mut Option<f64>, value: f64) {
    let Some(valid_value) = valid_positive_finite(value) else {
        return;
    };
    *target = Some(match target {
        Some(current) => current.min(valid_value),
        None => valid_value,
    });
}

pub(crate) fn name_count_rarity(value: Option<f64>) -> f32 {
    match value {
        Some(count) if count.is_finite() && count > 0.0 => round_six(1.0 / count.sqrt()),
        _ => 0.0,
    }
}

pub(crate) fn raw_arrow_name_count_rarity_row(
    query: &RetrievalQueryData,
    query_name_counts: &Option<NameCountsData>,
    summary: &RetrievalSummaryData,
    summary_signals: &RawArrowSummarySignalData,
) -> RawArrowNameCountRarityRow {
    let mut candidate_first_last_min = None;
    let mut candidate_last_min = None;
    for candidate_counts in summary_signals.name_counts_values.iter() {
        update_minimum(&mut candidate_first_last_min, candidate_counts.first_last);
        update_minimum(&mut candidate_last_min, candidate_counts.last);
    }
    let candidate_last_name_count_min_rarity = name_count_rarity(candidate_last_min);
    let candidate_last_first_name_count_min_rarity = name_count_rarity(candidate_first_last_min);

    let mut last_name_count_min_rarity = 0.0f32;
    let mut last_first_name_count_min_rarity = 0.0f32;
    if let Some(query_counts) = query_name_counts.as_ref() {
        let mut observed_minima: [Option<f64>; 6] = [None, None, None, None, None, None];
        for candidate_counts in summary_signals.name_counts_values.iter() {
            let values = compute_name_counts_data(Some(query_counts), Some(candidate_counts));
            for (index, value) in values.iter().enumerate() {
                update_minimum(&mut observed_minima[index], *value);
            }
        }
        last_name_count_min_rarity = name_count_rarity(observed_minima[2]);
        if query.has_full_first {
            last_first_name_count_min_rarity = name_count_rarity(observed_minima[1]);
        }
    }

    let mut first_prefix_match = 0.0f64;
    if py_len(&query.first) > 1 && summary.size > 0 {
        for (candidate_first, count) in summary.first_name_counts.iter() {
            if py_len(candidate_first) > 1 && same_prefix_tokens(&query.first, candidate_first) {
                first_prefix_match =
                    first_prefix_match.max((*count as f64) / (summary.size as f64));
            }
        }
    }

    RawArrowNameCountRarityRow {
        last_name_count_min_rarity,
        candidate_last_name_count_min_rarity,
        candidate_last_first_name_count_min_rarity,
        last_first_name_count_min_rarity,
        first_prefix_x_last_first_name_count_min_rarity: round_six(
            first_prefix_match * (last_first_name_count_min_rarity as f64),
        ),
    }
}

pub(crate) fn set_intersection_count(left: &HashSet<String>, right: &HashSet<String>) -> usize {
    if left.len() <= right.len() {
        left.iter().filter(|value| right.contains(*value)).count()
    } else {
        right.iter().filter(|value| left.contains(*value)).count()
    }
}

pub(crate) fn raw_arrow_paper_evidence_row(
    query_signature_id: &str,
    query_paper_author_count: usize,
    query_author_signals: &RawArrowAuthorSignalData,
    summary_signals: &RawArrowSummarySignalData,
) -> RawArrowPaperEvidenceRow {
    let mut best_author_jaccard = 0.0f64;
    let mut best_author_containment = 0.0f64;
    let mut best_author_overlap = 0.0f64;
    let mut best_local10_jaccard = 0.0f64;
    let mut best_local10_overlap_count = 0.0f64;
    let mut best_author_count_log_absdiff: Option<f64> = None;

    for (((candidate_names, candidate_count), candidate_local10_names), candidate_signature_id) in
        summary_signals
            .member_paper_author_names
            .iter()
            .zip(summary_signals.member_paper_author_counts.iter())
            .zip(summary_signals.member_local10_author_names.iter())
            .zip(summary_signals.member_signature_ids.iter())
    {
        if query_signature_id == candidate_signature_id {
            continue;
        }
        let intersection =
            set_intersection_count(&query_author_signals.paper_author_names, candidate_names);
        let union =
            query_author_signals.paper_author_names.len() + candidate_names.len() - intersection;
        if union > 0 {
            best_author_jaccard = best_author_jaccard.max((intersection as f64) / (union as f64));
        }
        let denominator = query_author_signals
            .paper_author_names
            .len()
            .min(candidate_names.len());
        if denominator > 0 {
            best_author_containment =
                best_author_containment.max((intersection as f64) / (denominator as f64));
        }
        best_author_overlap = best_author_overlap.max(intersection as f64);

        let local10_intersection = set_intersection_count(
            &query_author_signals.local10_author_names,
            candidate_local10_names,
        );
        let local10_union = query_author_signals.local10_author_names.len()
            + candidate_local10_names.len()
            - local10_intersection;
        if local10_union > 0 {
            best_local10_jaccard =
                best_local10_jaccard.max((local10_intersection as f64) / (local10_union as f64));
        }
        best_local10_overlap_count = best_local10_overlap_count.max(local10_intersection as f64);

        let count_delta =
            ((query_paper_author_count as f64).ln_1p() - (*candidate_count as f64).ln_1p()).abs();
        best_author_count_log_absdiff = Some(match best_author_count_log_absdiff {
            Some(current) => current.min(count_delta),
            None => count_delta,
        });
    }

    RawArrowPaperEvidenceRow {
        paper_author_list_max_jaccard: round_six(best_author_jaccard),
        paper_author_list_max_containment: round_six(best_author_containment),
        paper_author_list_max_overlap_count: round_six(best_author_overlap),
        local_author_window10_jaccard_max: round_six(best_local10_jaccard),
        local_author_window10_overlap_count_max: round_six(best_local10_overlap_count),
        best_author_count_log_absdiff: round_six(best_author_count_log_absdiff.unwrap_or(0.0)),
    }
}
