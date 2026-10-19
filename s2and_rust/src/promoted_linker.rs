use super::{ensure_unidecode_for_text, normalize_text_compat_from_map, py_len};
use numpy::{PyArray1, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule};
use pyo3::Bound;
use std::collections::HashMap;

pub(super) fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(
        promoted_linker_non_pairwise_features,
        m
    )?)?;
    Ok(())
}
const LINKER_GENERIC_FAMILY_MIN_COUNT: f32 = 3.0;
const LINKER_GENERIC_FAMILY_MIN_RATIO: f32 = 0.6;

fn linker_round(value: f32, scale: f32) -> f32 {
    // Use round-half-to-even (banker's rounding) to match np.round, which the
    // Python reference at tests/linker_row_feature_reference.py applies to every
    // value passed through linker_round.
    (value * scale).round_ties_even() / scale
}

fn linker_clip01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn linker_bool(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn linker_dict_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?.ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!("Missing linker row signal: {key}"))
    })
}

fn linker_extract_f32_vec(
    dict: &Bound<'_, PyDict>,
    key: &str,
    row_count: usize,
) -> PyResult<Vec<f32>> {
    let obj = linker_dict_item(dict, key)?;
    let values: Vec<f32> = if let Ok(arr) = obj.downcast::<PyArray1<f32>>() {
        arr.readonly().as_slice()?.to_vec()
    } else if let Ok(arr) = obj.downcast::<PyArray1<f64>>() {
        arr.readonly()
            .as_slice()?
            .iter()
            .map(|value| *value as f32)
            .collect()
    } else if let Ok(arr) = obj.downcast::<PyArray1<u16>>() {
        arr.readonly()
            .as_slice()?
            .iter()
            .map(|value| *value as f32)
            .collect()
    } else if let Ok(arr) = obj.downcast::<PyArray1<u32>>() {
        arr.readonly()
            .as_slice()?
            .iter()
            .map(|value| *value as f32)
            .collect()
    } else if let Ok(arr) = obj.downcast::<PyArray1<i32>>() {
        arr.readonly()
            .as_slice()?
            .iter()
            .map(|value| *value as f32)
            .collect()
    } else if let Ok(arr) = obj.downcast::<PyArray1<u8>>() {
        arr.readonly()
            .as_slice()?
            .iter()
            .map(|value| *value as f32)
            .collect()
    } else {
        let mut out = Vec::with_capacity(row_count);
        for item in PyIterator::from_object(&obj)? {
            let value = item?.extract::<f64>()? as f32;
            out.push(value);
        }
        out
    };
    if values.len() != row_count {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Signal {key:?} must have row_count={row_count}, got {}",
            values.len()
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Signal {key:?} contains non-finite values"
        )));
    }
    Ok(values)
}

fn linker_rank_order_values(values: &[f32], key: &str) -> PyResult<Vec<i64>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if value.fract() != 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Signal {key:?} contains non-integral rank value {value}"
            )));
        }
        out.push(*value as i64);
    }
    Ok(out)
}

fn linker_extract_string_vec(
    dict: &Bound<'_, PyDict>,
    key: &str,
    row_count: usize,
) -> PyResult<Vec<String>> {
    let obj = linker_dict_item(dict, key)?;
    let mut values = Vec::with_capacity(row_count);
    for item in PyIterator::from_object(&obj)? {
        let current = item?;
        if current.is_none() {
            values.push(String::new());
        } else {
            values.push(current.extract::<String>()?);
        }
    }
    if values.len() != row_count {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Signal {key:?} must have row_count={row_count}, got {}",
            values.len()
        )));
    }
    Ok(values)
}

fn linker_optional_string_vec(
    dict: &Bound<'_, PyDict>,
    key: &str,
    row_count: usize,
) -> PyResult<Option<Vec<String>>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => {
            Ok(Some(linker_extract_string_vec(dict, key, row_count)?))
        }
        _ => Ok(None),
    }
}

fn linker_groups(row_query_signature_indices: &[u32]) -> Vec<Vec<usize>> {
    let mut ordered: Vec<usize> = (0..row_query_signature_indices.len()).collect();
    ordered.sort_by_key(|index| row_query_signature_indices[*index]);
    let mut groups = Vec::new();
    let mut start = 0usize;
    while start < ordered.len() {
        let query_index = row_query_signature_indices[ordered[start]];
        let mut end = start + 1;
        while end < ordered.len() && row_query_signature_indices[ordered[end]] == query_index {
            end += 1;
        }
        groups.push(ordered[start..end].to_vec());
        start = end;
    }
    groups
}

fn linker_retrieval_ordered_groups(
    groups: &[Vec<usize>],
    retrieval_score: &[f32],
    retrieval_rank_order: &[i64],
    component_keys: &[String],
) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(groups.len());
    for group in groups {
        let mut ordered = group.clone();
        ordered.sort_by(|left, right| {
            retrieval_score[*right]
                .total_cmp(&retrieval_score[*left])
                .then_with(|| retrieval_rank_order[*left].cmp(&retrieval_rank_order[*right]))
                .then_with(|| component_keys[*left].cmp(&component_keys[*right]))
        });
        out.push(ordered);
    }
    out
}

fn linker_normalize_alpha(value: &str, unidecode_char_map: &HashMap<char, String>) -> String {
    normalize_text_compat_from_map(value, unidecode_char_map)
        .chars()
        .filter(|character| character.is_ascii_alphabetic())
        .collect()
}

fn linker_normalized_alpha_vec(
    values: &[String],
    unidecode_char_map: &HashMap<char, String>,
) -> Vec<String> {
    let mut cache = HashMap::<&str, String>::new();
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if let Some(normalized) = cache.get(value.as_str()) {
            out.push(normalized.clone());
        } else {
            let normalized = linker_normalize_alpha(value, unidecode_char_map);
            cache.insert(value.as_str(), normalized.clone());
            out.push(normalized);
        }
    }
    out
}

fn linker_family_ids(
    component_keys: &[String],
    dominant_first_names: &[String],
    named_signature_count: &[f32],
    cluster_size: &[f32],
) -> Vec<String> {
    let mut out = component_keys.to_vec();
    for index in 0..component_keys.len() {
        let dominant = dominant_first_names[index].as_str();
        let named_count = named_signature_count[index];
        let dominance_ratio = named_count / cluster_size[index].max(1.0);
        if !dominant.is_empty()
            && named_count >= LINKER_GENERIC_FAMILY_MIN_COUNT
            && dominance_ratio >= LINKER_GENERIC_FAMILY_MIN_RATIO
        {
            out[index] = dominant.to_string();
        }
    }
    out
}

struct LinkerGroupFeatures {
    retrieval_score_gap_vs_best_competitor: Vec<f32>,
    same_family_as_top1: Vec<f32>,
    same_family_as_heuristic_choice: Vec<f32>,
    same_dominant_first_as_best_top5: Vec<f32>,
    current_retrieval_rank: Vec<f32>,
}

fn linker_derive_group_features(
    ordered_groups: &[Vec<usize>],
    retrieval_score: &[f32],
    retrieval_rank_order: &[i64],
    component_keys: &[String],
    family_ids: &[String],
    dominant_first_alpha: &[String],
    top5_mean_distance: &[f32],
) -> LinkerGroupFeatures {
    let row_count = retrieval_score.len();
    let mut retrieval_score_gap_vs_best_competitor = vec![0.0f32; row_count];
    let mut same_family_as_top1 = vec![0.0f32; row_count];
    let mut same_family_as_heuristic_choice = vec![0.0f32; row_count];
    let mut dominant_first_top1_match = vec![0.0f32; row_count];
    let mut same_dominant_first_as_best_top5 = vec![0.0f32; row_count];
    let mut current_retrieval_rank = vec![0.0f32; row_count];

    for ordered in ordered_groups {
        let top1 = ordered[0];
        let runner_up = if ordered.len() > 1 {
            ordered[1]
        } else {
            ordered[0]
        };
        let mut best_top5 = ordered[0];
        for index in ordered.iter().copied().skip(1) {
            let current_key = (
                top5_mean_distance[index],
                retrieval_rank_order[index],
                component_keys[index].as_str(),
            );
            let best_key = (
                top5_mean_distance[best_top5],
                retrieval_rank_order[best_top5],
                component_keys[best_top5].as_str(),
            );
            if current_key < best_key {
                best_top5 = index;
            }
        }
        for (current_rank, index) in ordered.iter().enumerate() {
            let competitor = if *index == top1 { runner_up } else { top1 };
            current_retrieval_rank[*index] = (current_rank + 1) as f32;
            retrieval_score_gap_vs_best_competitor[*index] = linker_round(
                retrieval_score[*index] - retrieval_score[competitor],
                1_000_000.0,
            );
            same_family_as_top1[*index] = linker_bool(
                !family_ids[*index].is_empty() && family_ids[*index] == family_ids[top1],
            );
            dominant_first_top1_match[*index] = linker_bool(
                !dominant_first_alpha[*index].is_empty()
                    && dominant_first_alpha[*index] == dominant_first_alpha[top1],
            );
            same_dominant_first_as_best_top5[*index] = linker_bool(
                !dominant_first_alpha[*index].is_empty()
                    && dominant_first_alpha[*index] == dominant_first_alpha[best_top5],
            );
            // Match the Python reference in tests/linker_row_feature_reference.py:
            // retrieval_score is used raw inside the product; do not clip to [0,1].
            same_family_as_heuristic_choice[*index] = linker_round(
                dominant_first_top1_match[*index] * retrieval_score[*index]
                    + same_dominant_first_as_best_top5[*index] * (1.0 - top5_mean_distance[*index]),
                1_000_000.0,
            );
        }
    }

    LinkerGroupFeatures {
        retrieval_score_gap_vs_best_competitor,
        same_family_as_top1,
        same_family_as_heuristic_choice,
        same_dominant_first_as_best_top5,
        current_retrieval_rank,
    }
}

fn linker_year_gap_features(
    query_year: &[f32],
    query_year_missing: &[f32],
    candidate_year_min: &[f32],
    candidate_year_max: &[f32],
    candidate_year_range_missing: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut gap = vec![0.0f32; query_year.len()];
    let mut signed_gap = vec![0.0f32; query_year.len()];
    let mut span = vec![0.0f32; query_year.len()];
    for index in 0..query_year.len() {
        if candidate_year_range_missing[index] == 0.0 {
            let raw_span = (candidate_year_max[index] - candidate_year_min[index]).max(0.0);
            // Match the Python reference in tests/linker_row_feature_reference.py,
            // which applies np.round(span, 6) at the end.
            span[index] = linker_round(raw_span, 1_000_000.0);
        }
        if query_year_missing[index] != 0.0 || candidate_year_range_missing[index] != 0.0 {
            continue;
        }
        if query_year[index] < candidate_year_min[index] {
            let current_gap = candidate_year_min[index] - query_year[index];
            gap[index] = linker_round(current_gap, 1_000_000.0);
            signed_gap[index] = linker_round(-current_gap, 1_000_000.0);
        } else if query_year[index] > candidate_year_max[index] {
            let current_gap = query_year[index] - candidate_year_max[index];
            gap[index] = linker_round(current_gap, 1_000_000.0);
            signed_gap[index] = linker_round(current_gap, 1_000_000.0);
        }
    }
    (gap, signed_gap, span)
}

fn linker_alpha_length(value: &str) -> f32 {
    py_len(value) as f32
}

fn linker_set_f32_array<'py>(
    py: Python<'py>,
    payload: &Bound<'py, PyDict>,
    key: &str,
    values: Vec<f32>,
) -> PyResult<()> {
    payload.set_item(key, values.to_pyarray(py))?;
    Ok(())
}

#[pyfunction]
fn promoted_linker_non_pairwise_features<'py>(
    py: Python<'py>,
    signals: &Bound<'py, PyDict>,
) -> PyResult<Py<PyDict>> {
    let row_query_signature_indices_obj = linker_dict_item(signals, "row_query_signature_indices")?;
    let row_query_signature_indices =
        if let Ok(arr) = row_query_signature_indices_obj.downcast::<PyArray1<u32>>() {
            arr.readonly().as_slice()?.to_vec()
        } else {
            let mut out = Vec::new();
            for item in PyIterator::from_object(&row_query_signature_indices_obj)? {
                out.push(item?.extract::<u32>()?);
            }
            out
        };
    let row_count = row_query_signature_indices.len();

    let retrieval_score = linker_extract_f32_vec(signals, "retrieval_score", row_count)?;
    let retrieval_rank = linker_extract_f32_vec(signals, "retrieval_rank", row_count)?;
    let retrieval_rank_order = linker_rank_order_values(&retrieval_rank, "retrieval_rank")?;
    let component_keys = linker_extract_string_vec(signals, "candidate_component_key", row_count)?;
    let cluster_size = linker_extract_f32_vec(signals, "cluster_size", row_count)?;
    let named_signature_count =
        linker_extract_f32_vec(signals, "named_signature_count", row_count)?;
    let dominant_first_name = linker_extract_string_vec(signals, "dominant_first_name", row_count)?;
    let candidate_year_min = linker_extract_f32_vec(signals, "candidate_year_min", row_count)?;
    let candidate_year_max = linker_extract_f32_vec(signals, "candidate_year_max", row_count)?;
    let candidate_year_range_missing =
        linker_extract_f32_vec(signals, "candidate_year_range_missing", row_count)?;
    let query_first_token = linker_extract_string_vec(signals, "query_first_token", row_count)?;
    let query_year = linker_extract_f32_vec(signals, "query_year", row_count)?;
    let query_year_missing = linker_extract_f32_vec(signals, "query_year_missing", row_count)?;
    let query_has_affiliations =
        linker_extract_f32_vec(signals, "query_has_affiliations", row_count)?;
    let affiliation_overlap = linker_extract_f32_vec(signals, "affiliation_overlap", row_count)?;
    let coauthor_overlap = linker_extract_f32_vec(signals, "coauthor_overlap", row_count)?;
    let year_compatibility = linker_extract_f32_vec(signals, "year_compatibility", row_count)?;
    let specter_exemplar_similarity =
        linker_extract_f32_vec(signals, "specter_exemplar_similarity", row_count)?;
    let min_distance = linker_extract_f32_vec(signals, "min_distance", row_count)?;
    let top5_mean_distance = linker_extract_f32_vec(signals, "top5_mean_distance", row_count)?;
    let last_name_count_min_rarity =
        linker_extract_f32_vec(signals, "last_name_count_min_rarity", row_count)?;
    let last_first_name_count_min_rarity =
        linker_extract_f32_vec(signals, "last_first_name_count_min_rarity", row_count)?;
    let candidate_cluster_max_paper_author_count = linker_extract_f32_vec(
        signals,
        "candidate_cluster_max_paper_author_count",
        row_count,
    )?;
    let paper_author_list_max_jaccard =
        linker_extract_f32_vec(signals, "paper_author_list_max_jaccard", row_count)?;
    let paper_author_list_max_containment =
        linker_extract_f32_vec(signals, "paper_author_list_max_containment", row_count)?;
    let paper_author_list_max_overlap_count =
        linker_extract_f32_vec(signals, "paper_author_list_max_overlap_count", row_count)?;
    let local_author_window10_jaccard_max =
        linker_extract_f32_vec(signals, "local_author_window10_jaccard_max", row_count)?;
    let local_author_window10_overlap_count_max = linker_extract_f32_vec(
        signals,
        "local_author_window10_overlap_count_max",
        row_count,
    )?;
    let best_author_count_log_absdiff =
        linker_extract_f32_vec(signals, "best_author_count_log_absdiff", row_count)?;

    let groups = linker_groups(&row_query_signature_indices);
    let ordered_groups = linker_retrieval_ordered_groups(
        &groups,
        &retrieval_score,
        &retrieval_rank_order,
        &component_keys,
    );
    let family_ids_from_signal = linker_optional_string_vec(signals, "family_id", row_count)?;
    let generated_family_id_count = if family_ids_from_signal.is_some() {
        0usize
    } else {
        row_count
    };
    let family_ids = family_ids_from_signal.unwrap_or_else(|| {
        linker_family_ids(
            &component_keys,
            &dominant_first_name,
            &named_signature_count,
            &cluster_size,
        )
    });
    let generic_family_override_count = family_ids
        .iter()
        .zip(component_keys.iter())
        .filter(|(family, component)| !family.is_empty() && *family != *component)
        .count();
    let mut linker_unidecode_char_map = HashMap::<char, String>::new();
    for value in query_first_token.iter().chain(dominant_first_name.iter()) {
        ensure_unidecode_for_text(value, &mut linker_unidecode_char_map)?;
    }
    let query_first_alpha =
        linker_normalized_alpha_vec(&query_first_token, &linker_unidecode_char_map);
    let dominant_first_alpha =
        linker_normalized_alpha_vec(&dominant_first_name, &linker_unidecode_char_map);
    let group_features = linker_derive_group_features(
        &ordered_groups,
        &retrieval_score,
        &retrieval_rank_order,
        &component_keys,
        &family_ids,
        &dominant_first_alpha,
        &top5_mean_distance,
    );
    let (year_gap_to_candidate_range, year_gap_signed_to_candidate_range, candidate_year_span) =
        linker_year_gap_features(
            &query_year,
            &query_year_missing,
            &candidate_year_min,
            &candidate_year_max,
            &candidate_year_range_missing,
        );

    let mut affiliation_contradiction_severity = vec![0.0f32; row_count];
    let mut anchor_evidence_count = vec![0.0f32; row_count];
    let mut strong_positive_anchor_score = vec![0.0f32; row_count];
    let mut weak_residual_anchor_score = vec![0.0f32; row_count];
    let mut sparse_relative_winner_score = vec![0.0f32; row_count];
    let mut query_first_prefix_match_any_length = vec![0.0f32; row_count];
    let mut candidate_dominant_first_name_length = vec![0.0f32; row_count];
    let mut cluster_size_log = vec![0.0f32; row_count];
    let mut retrieval_reciprocal_rank = vec![0.0f32; row_count];

    for index in 0..row_count {
        if query_has_affiliations[index] > 0.0 {
            affiliation_contradiction_severity[index] =
                linker_round((1.0 - affiliation_overlap[index]).max(0.0), 1_000_000.0);
        }
        let query_first = &query_first_alpha[index];
        let dominant_first = &dominant_first_alpha[index];
        let retrieval_gap = group_features.retrieval_score_gap_vs_best_competitor[index];
        anchor_evidence_count[index] =
            linker_bool(min_distance[index] <= 0.15) + linker_bool(retrieval_gap >= 0.02);
        let distance_signal = 1.0 - linker_clip01(min_distance[index]);
        let support_strength = 0.20 * distance_signal;
        let same_top1 = group_features.same_family_as_top1[index];
        strong_positive_anchor_score[index] = linker_round(
            linker_clip01(support_strength) * (0.5 + 0.5 * linker_clip01(same_top1)),
            1_000_000.0,
        );
        let retrieval_gap_scaled = linker_clip01((retrieval_gap.clamp(-0.2, 0.3) + 0.2) / 0.5);
        let residual_support = 0.28 * distance_signal + 0.08 * retrieval_gap_scaled;
        weak_residual_anchor_score[index] =
            linker_round(same_top1 * linker_clip01(residual_support), 1_000_000.0);
        sparse_relative_winner_score[index] = linker_round(
            linker_bool(group_features.current_retrieval_rank[index] <= 1.0)
                * same_top1
                * linker_clip01(retrieval_gap.clamp(0.0, 0.3) / 0.3)
                * linker_clip01(residual_support),
            1_000_000.0,
        );
        query_first_prefix_match_any_length[index] = linker_bool(
            !query_first.is_empty()
                && !dominant_first.is_empty()
                && (query_first.starts_with(dominant_first)
                    || dominant_first.starts_with(query_first)),
        );
        candidate_dominant_first_name_length[index] = linker_alpha_length(dominant_first);
        cluster_size_log[index] = (1.0 + cluster_size[index].max(0.0)).ln();
        retrieval_reciprocal_rank[index] = linker_round(
            1.0 / group_features.current_retrieval_rank[index].max(1.0),
            1_000_000.0,
        );
    }

    let payload = PyDict::new(py);
    linker_set_f32_array(py, &payload, "min_distance", min_distance.clone())?;
    linker_set_f32_array(
        py,
        &payload,
        "affiliation_contradiction_severity",
        affiliation_contradiction_severity,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "same_family_as_heuristic_choice",
        group_features.same_family_as_heuristic_choice,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "same_dominant_first_as_best_top5",
        group_features.same_dominant_first_as_best_top5,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "specter_exemplar_similarity",
        specter_exemplar_similarity,
    )?;
    linker_set_f32_array(py, &payload, "coauthor_overlap", coauthor_overlap)?;
    linker_set_f32_array(py, &payload, "affiliation_overlap", affiliation_overlap)?;
    linker_set_f32_array(py, &payload, "year_compatibility", year_compatibility)?;
    linker_set_f32_array(
        py,
        &payload,
        "retrieval_rank",
        group_features.current_retrieval_rank,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "retrieval_reciprocal_rank",
        retrieval_reciprocal_rank,
    )?;
    linker_set_f32_array(py, &payload, "cluster_size_log", cluster_size_log)?;
    linker_set_f32_array(py, &payload, "candidate_year_span", candidate_year_span)?;
    linker_set_f32_array(
        py,
        &payload,
        "year_gap_to_candidate_range",
        year_gap_to_candidate_range,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "year_gap_signed_to_candidate_range",
        year_gap_signed_to_candidate_range,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "candidate_dominant_first_name_length",
        candidate_dominant_first_name_length,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "query_first_prefix_match_any_length",
        query_first_prefix_match_any_length,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "candidate_cluster_max_paper_author_count",
        candidate_cluster_max_paper_author_count,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "paper_author_list_max_jaccard",
        paper_author_list_max_jaccard,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "paper_author_list_max_containment",
        paper_author_list_max_containment,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "paper_author_list_max_overlap_count",
        paper_author_list_max_overlap_count,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "local_author_window10_jaccard_max",
        local_author_window10_jaccard_max,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "local_author_window10_overlap_count_max",
        local_author_window10_overlap_count_max,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "best_author_count_log_absdiff",
        best_author_count_log_absdiff,
    )?;
    linker_set_f32_array(py, &payload, "anchor_evidence_count", anchor_evidence_count)?;
    linker_set_f32_array(
        py,
        &payload,
        "strong_positive_anchor_score",
        strong_positive_anchor_score,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "weak_residual_anchor_score",
        weak_residual_anchor_score,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "sparse_relative_winner_score",
        sparse_relative_winner_score,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "last_name_count_min_rarity",
        last_name_count_min_rarity,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "last_first_name_count_min_rarity",
        last_first_name_count_min_rarity,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "top5_mean_distance",
        top5_mean_distance.clone(),
    )?;
    let telemetry = PyDict::new(py);
    telemetry.set_item("generated_family_id_count", generated_family_id_count)?;
    telemetry.set_item(
        "generic_family_override_count",
        generic_family_override_count,
    )?;
    payload.set_item("telemetry", telemetry)?;
    Ok(payload.unbind())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn linker_alpha_normalization_uses_text_normalization() {
        let unidecode_char_map = HashMap::from([('\u{00C9}', "E".to_string())]);
        assert_eq!(
            linker_normalize_alpha("\u{00C9}lodie-2", &unidecode_char_map),
            "elodie"
        );
    }
}
