use super::*;
#[derive(Clone)]
pub(crate) struct RetrievalSummaryData {
    pub(crate) component_key: String,
    pub(crate) size: usize,
    pub(crate) first_name_counts: Vec<(String, u64)>,
    pub(crate) middle_initial_counts: Option<CounterData>,
    pub(crate) coauthor_counts: Option<CounterData>,
    pub(crate) non_mega_coauthor_counts: Option<CounterData>,
    pub(crate) affiliation_counts: Option<CounterData>,
    pub(crate) venue_counts: Option<CounterData>,
    pub(crate) title_counts: Option<CounterData>,
    pub(crate) max_paper_author_count: usize,
    pub(crate) year_min: Option<i64>,
    pub(crate) year_max: Option<i64>,
    pub(crate) year_mean: Option<f64>,
    pub(crate) orcid_hashes: Vec<u64>,
    pub(crate) specter_centroid: Option<Vec<f32>>,
    pub(crate) specter_centroid_norm: Option<f64>,
    pub(crate) exemplar_vectors: Vec<Vec<f32>>,
    pub(crate) exemplar_norms: Vec<f64>,
}

#[derive(Clone, Copy)]
pub(crate) struct RetrievalQueryTerm {
    pub(crate) hash: u64,
    pub(crate) token_count: u8,
}

#[derive(Clone)]
pub(crate) struct RetrievalQueryData {
    pub(crate) first: String,
    pub(crate) has_full_first: bool,
    pub(crate) middle_initial_hashes: Vec<u64>,
    pub(crate) coauthor_hashes: Vec<u64>,
    pub(crate) coauthor_terms: Vec<RetrievalQueryTerm>,
    pub(crate) affiliation_hashes: Vec<u64>,
    pub(crate) affiliation_terms: Vec<RetrievalQueryTerm>,
    pub(crate) venue_hashes: Vec<u64>,
    pub(crate) title_hashes: Vec<u64>,
    pub(crate) year: Option<i64>,
    pub(crate) orcid_hash: Option<u64>,
    pub(crate) specter: Option<Arc<Vec<f32>>>,
    pub(crate) specter_norm: Option<f64>,
}

#[derive(Clone, Copy)]
pub(crate) struct RetrievalHybridWeights {
    pub(crate) centroid: f64,
    pub(crate) coauthor: f64,
    pub(crate) affiliation: f64,
    pub(crate) middle: f64,
    pub(crate) first_name: f64,
}

pub(crate) const DEFAULT_HYBRID_CENTROID_POLICY_NAME: &str = "h_wang_any_input_v2";
const DEFAULT_HYBRID_CENTROID_WEIGHTS: RetrievalHybridWeights = RetrievalHybridWeights {
    centroid: 0.527232,
    coauthor: 0.223412,
    affiliation: 0.146909,
    middle: 0.009439,
    first_name: 0.093007,
};
const DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS: RetrievalHybridWeights =
    RetrievalHybridWeights {
        centroid: 0.520012,
        coauthor: 0.220264,
        affiliation: 0.109278,
        middle: 0.150447,
        first_name: 0.0,
    };
pub(crate) const INCREMENTAL_LINKING_PAIR_PLAN_ROW_SIGNALS: [&str; 1] = ["row_orcid_match"];
pub(crate) const INCREMENTAL_LINKING_PAIR_PLAN_SUPPORTED_KWARGS: [&str; 5] = [
    "num_threads",
    "query_signature_ids",
    "retrieval_subblock_index",
    "query_candidate_component_keys_by_signature_id",
    "full_first_global_backfill_count",
];
pub(crate) const RAW_ARROW_QUERY_SIGNATURE_PLANNER_METHODS: [&str; 4] = [
    "from_query_signatures",
    "from_auto_queries",
    "plan_query_signatures",
    "build_telemetry",
];
pub(crate) const RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE: f64 = -0.25;
pub(crate) const RETRIEVAL_YEAR_SCORE_DECAY_YEARS: f64 = 15.0;
pub(crate) const RETRIEVAL_YEAR_SCORE_RANGE_GAP: i64 = 10;
pub(crate) const RETRIEVAL_YEAR_SCORE_RANGE_PENALTY: f64 = 0.15;
const RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP: i64 = 35;
pub(crate) const RETRIEVAL_MEGA_AUTHOR_THRESHOLD: usize = 50;

pub(crate) fn filter_excluded_candidate_indices(
    indices: Vec<usize>,
    excluded_candidate_indices: Option<&HashSet<usize>>,
) -> Vec<usize> {
    let Some(excluded) = excluded_candidate_indices else {
        return indices;
    };
    if excluded.is_empty() {
        return indices;
    }
    indices
        .into_iter()
        .filter(|index| !excluded.contains(index))
        .collect()
}

pub(crate) fn default_candidate_indices(
    summary_count: usize,
    base_candidate_indices: Option<&[usize]>,
    excluded_candidate_indices: Option<&HashSet<usize>>,
) -> Vec<usize> {
    let values = base_candidate_indices.map_or_else(
        || (0..summary_count).collect::<Vec<_>>(),
        |base_values| base_values.to_vec(),
    );
    filter_excluded_candidate_indices(values, excluded_candidate_indices)
}

#[derive(Clone, Copy)]
pub(crate) enum RetrievalFirstNameMode {
    Prefix,
    ExactThenPrefixHalf,
}

#[derive(Clone, Copy)]
pub(crate) struct RetrievalOverlapConfig {
    pub(crate) use_idf: bool,
    pub(crate) per_term_cap: Option<f64>,
    pub(crate) total_cap: Option<f64>,
    pub(crate) min_token_count: u8,
    pub(crate) unigram_weight: f64,
    pub(crate) multi_token_weight: f64,
}

#[pyclass]
pub(crate) struct RustHybridCentroidRetriever {
    pub(crate) summaries: Vec<RetrievalSummaryData>,
    pub(crate) component_index_by_key: HashMap<String, usize>,
    pub(crate) coauthor_cluster_df: HashMap<u64, usize>,
    pub(crate) non_mega_coauthor_cluster_df: HashMap<u64, usize>,
    pub(crate) affiliation_cluster_df: HashMap<u64, usize>,
}

pub(crate) struct RustNameCompatibleSubblockSelector {
    pub(crate) signature_to_subblock: HashMap<String, String>,
    pub(crate) subblock_to_components: HashMap<String, Vec<String>>,
    pub(crate) subblock_tokens_by_subblock: HashMap<String, Vec<String>>,
    pub(crate) name_tuples: HashMap<String, HashSet<String>>,
}

#[derive(Default)]
pub(crate) struct RetrievalPairPlanQueryResult {
    pub(crate) row_query_signature_indices: Vec<u32>,
    pub(crate) row_component_keys: Vec<String>,
    pub(crate) row_retrieval_scores: Vec<f32>,
    pub(crate) row_retrieval_ranks: Vec<u16>,
    pub(crate) row_component_sizes: Vec<u32>,
    pub(crate) row_named_signature_counts: Vec<u32>,
    pub(crate) row_dominant_first_names: Vec<String>,
    pub(crate) row_candidate_year_min: Vec<i32>,
    pub(crate) row_candidate_year_max: Vec<i32>,
    pub(crate) row_candidate_year_range_missing: Vec<u8>,
    pub(crate) row_query_first_tokens: Vec<String>,
    pub(crate) row_query_years: Vec<i32>,
    pub(crate) row_query_year_missing: Vec<u8>,
    pub(crate) row_query_has_affiliations: Vec<u8>,
    pub(crate) row_query_has_coauthors: Vec<u8>,
    pub(crate) row_orcid_match: Vec<u8>,
    pub(crate) row_middle_initial_compatibility: Vec<f32>,
    pub(crate) row_affiliation_overlap: Vec<f32>,
    pub(crate) row_coauthor_overlap: Vec<f32>,
    pub(crate) row_venue_overlap: Vec<f32>,
    pub(crate) row_year_compatibility: Vec<f32>,
    pub(crate) row_title_overlap: Vec<f32>,
    pub(crate) row_specter_centroid_similarity: Vec<f32>,
    pub(crate) row_specter_exemplar_similarity: Vec<f32>,
    pub(crate) right_signature_indices_by_row: Vec<Vec<u32>>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RetrievalError {
    InvalidValue(String),
    MissingKey(String),
    Overflow(String),
}

impl std::fmt::Display for RetrievalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidValue(message) | Self::MissingKey(message) | Self::Overflow(message) => {
                message
            }
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for RetrievalError {}

pub(crate) fn retrieval_error_to_py(error: RetrievalError) -> PyErr {
    match error {
        RetrievalError::InvalidValue(message) => pyo3::exceptions::PyValueError::new_err(message),
        RetrievalError::MissingKey(message) => pyo3::exceptions::PyKeyError::new_err(message),
        RetrievalError::Overflow(message) => pyo3::exceptions::PyOverflowError::new_err(message),
    }
}

pub(crate) fn year_signal_value(
    year: Option<i64>,
    field_name: &str,
) -> Result<(i32, u8), RetrievalError> {
    let Some(value) = year else {
        return Ok((i32::MIN, 1));
    };
    let converted = validate_row_signal_year(value, field_name)?;
    Ok((converted, 0))
}

pub(crate) fn validate_row_signal_year(year: i64, field_name: &str) -> Result<i32, RetrievalError> {
    let converted = i32::try_from(year).map_err(|_| {
        RetrievalError::InvalidValue(format!(
            "{field_name} is outside the supported i32 range: {year}"
        ))
    })?;
    if converted == i32::MIN {
        return Err(RetrievalError::InvalidValue(format!(
            "{field_name} uses reserved missing-year sentinel value: {year}"
        )));
    }
    Ok(converted)
}

pub(crate) fn raw_arrow_year_mean(years: &[i64]) -> Option<f64> {
    if years.is_empty() {
        return None;
    }
    let sum: i128 = years.iter().map(|year| i128::from(*year)).sum();
    Some(sum as f64 / years.len() as f64)
}

pub(crate) fn row_named_signature_count(
    first_name_counts: &[(String, u64)],
) -> Result<u32, RetrievalError> {
    let total = first_name_counts
        .iter()
        .try_fold(0u64, |current, (_first_name, count)| {
            current.checked_add(*count).ok_or_else(|| {
                RetrievalError::Overflow(
                    "named_signature_count exceeds the supported u64 range".to_string(),
                )
            })
        })?;
    Ok(total.min(u32::MAX as u64) as u32)
}

pub(crate) struct RetrievalCandidateSelection {
    pub(crate) indices: Vec<usize>,
    pub(crate) return_all: bool,
}

impl RustHybridCentroidRetriever {
    fn summary_for_candidate_index<'a>(
        &'a self,
        idx: usize,
        override_index: Option<usize>,
        override_summary: Option<&'a RetrievalSummaryData>,
    ) -> &'a RetrievalSummaryData {
        match (override_index, override_summary) {
            (Some(replaced_idx), Some(replaced_summary)) if idx == replaced_idx => replaced_summary,
            _ => &self.summaries[idx],
        }
    }

    fn compare_scored_candidates(
        &self,
        left: &(usize, f32),
        right: &(usize, f32),
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
    ) -> Ordering {
        let left_component_key = self
            .summary_for_candidate_index(left.0, override_index, override_summary)
            .component_key
            .as_str();
        let right_component_key = self
            .summary_for_candidate_index(right.0, override_index, override_summary)
            .component_key
            .as_str();
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left_component_key.cmp(right_component_key))
    }

    fn keep_sorted_top_k_scored_candidates(
        &self,
        scored: &mut Vec<(usize, f32)>,
        top_k: usize,
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
    ) {
        if top_k == 0 {
            scored.clear();
            return;
        }
        let limit = top_k.min(scored.len());
        if limit < scored.len() {
            scored.select_nth_unstable_by(limit - 1, |left, right| {
                self.compare_scored_candidates(left, right, override_index, override_summary)
            });
            scored.truncate(limit);
        }
        scored.sort_unstable_by(|left, right| {
            self.compare_scored_candidates(left, right, override_index, override_summary)
        });
    }

    fn score_top_k_candidate_indices_default_inner(
        &self,
        query_data: &RetrievalQueryData,
        candidate_indices: &[usize],
        top_k: usize,
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
    ) -> Vec<(usize, f32)> {
        let mut scored = candidate_indices
            .iter()
            .map(|idx| {
                let summary =
                    self.summary_for_candidate_index(*idx, override_index, override_summary);
                (
                    *idx,
                    score_hybrid_centroid_query(
                        query_data,
                        summary,
                        &self.coauthor_cluster_df,
                        &self.non_mega_coauthor_cluster_df,
                        &self.affiliation_cluster_df,
                        self.summaries.len(),
                    ),
                )
            })
            .collect::<Vec<_>>();
        self.keep_sorted_top_k_scored_candidates(
            &mut scored,
            top_k,
            override_index,
            override_summary,
        );
        scored
    }

    fn scored_candidates_to_keys_scores(
        &self,
        scored: Vec<(usize, f32)>,
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
    ) -> (Vec<String>, Vec<f32>) {
        let component_keys = scored
            .iter()
            .map(|(idx, _)| {
                self.summary_for_candidate_index(*idx, override_index, override_summary)
                    .component_key
                    .clone()
            })
            .collect();
        let scores = scored.iter().map(|(_, score)| *score).collect();
        (component_keys, scores)
    }

    fn hard_filtered_candidate_indices_for_query(
        &self,
        query_data: &RetrievalQueryData,
        mut candidate_indices: Vec<usize>,
    ) -> RetrievalCandidateSelection {
        if let Some(selection) =
            self.orcid_candidate_selection_for_query(query_data, candidate_indices.iter().copied())
        {
            return selection;
        }

        let middle_filtered: Vec<usize> = candidate_indices
            .iter()
            .copied()
            .filter(|idx| {
                !has_middle_initial_conflict(
                    &query_data.middle_initial_hashes,
                    &self.summaries[*idx].middle_initial_counts,
                )
            })
            .collect();
        if !middle_filtered.is_empty() {
            candidate_indices = middle_filtered;
        }

        let year_filtered: Vec<usize> = candidate_indices
            .iter()
            .copied()
            .filter(|idx| {
                !has_impossible_year_conflict(
                    query_data.year,
                    &self.summaries[*idx],
                    RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP,
                )
            })
            .collect();
        if !year_filtered.is_empty() {
            candidate_indices = year_filtered;
        }

        RetrievalCandidateSelection {
            indices: candidate_indices,
            return_all: false,
        }
    }

    fn orcid_candidate_selection_for_query<I>(
        &self,
        query_data: &RetrievalQueryData,
        candidate_indices: I,
    ) -> Option<RetrievalCandidateSelection>
    where
        I: IntoIterator<Item = usize>,
    {
        let orcid_hash = query_data.orcid_hash?;
        let orcid_matches: Vec<usize> = candidate_indices
            .into_iter()
            .filter(|idx| contains_hashed_value(&self.summaries[*idx].orcid_hashes, orcid_hash))
            .collect();
        if orcid_matches.is_empty() {
            None
        } else {
            Some(RetrievalCandidateSelection {
                indices: orcid_matches,
                return_all: true,
            })
        }
    }

    fn candidate_indices_for_pair_plan_query(
        &self,
        query_data: &RetrievalQueryData,
        base_candidate_indices: Option<&[usize]>,
        excluded_candidate_indices: Option<&HashSet<usize>>,
        query_signature_id: Option<&str>,
        selector: Option<&RustNameCompatibleSubblockSelector>,
        global_backfill_count: usize,
        allow_global_orcid_override: bool,
    ) -> RetrievalCandidateSelection {
        if allow_global_orcid_override {
            let global_candidates =
                (0..self.summaries.len()).filter(|idx| match excluded_candidate_indices {
                    Some(excluded) => !excluded.contains(idx),
                    None => true,
                });
            if let Some(selection) =
                self.orcid_candidate_selection_for_query(query_data, global_candidates)
            {
                return selection;
            }
        }
        let selected = if query_data.has_full_first {
            match (query_signature_id, selector) {
                (Some(signature_id), Some(selector)) => selector
                    .select_candidate_indices_for_summaries(
                        signature_id,
                        &query_data.first,
                        &self.summaries,
                        base_candidate_indices,
                        global_backfill_count,
                    ),
                _ => None,
            }
        } else {
            None
        };
        let candidate_indices = match selected {
            Some(indices) => {
                let filtered =
                    filter_excluded_candidate_indices(indices, excluded_candidate_indices);
                if filtered.is_empty() {
                    default_candidate_indices(
                        self.summaries.len(),
                        base_candidate_indices,
                        excluded_candidate_indices,
                    )
                } else {
                    filtered
                }
            }
            None => default_candidate_indices(
                self.summaries.len(),
                base_candidate_indices,
                excluded_candidate_indices,
            ),
        };
        self.hard_filtered_candidate_indices_for_query(query_data, candidate_indices)
    }

    pub(crate) fn build_pair_plan_query_result(
        &self,
        current_query: &RetrievalQueryData,
        row_query_first_token: &str,
        query_signature_index: u32,
        base_candidate_indices: Option<&[usize]>,
        excluded_candidate_indices: Option<&HashSet<usize>>,
        query_signature_id: Option<&str>,
        component_member_indices: &HashMap<String, Vec<u32>>,
        top_k: usize,
        selector: Option<&RustNameCompatibleSubblockSelector>,
        global_backfill_count: usize,
        allow_global_orcid_override: bool,
    ) -> Result<RetrievalPairPlanQueryResult, RetrievalError> {
        let selection = self.candidate_indices_for_pair_plan_query(
            current_query,
            base_candidate_indices,
            excluded_candidate_indices,
            query_signature_id,
            selector,
            global_backfill_count,
            allow_global_orcid_override,
        );
        if selection.indices.is_empty() {
            return Ok(RetrievalPairPlanQueryResult::default());
        }
        let effective_top_k = if selection.return_all {
            selection.indices.len()
        } else {
            top_k
        };
        let scored = self.score_top_k_candidate_indices_default_inner(
            current_query,
            &selection.indices,
            effective_top_k,
            None,
            None,
        );

        let mut result = RetrievalPairPlanQueryResult::default();
        let (query_year, query_year_missing) = year_signal_value(current_query.year, "query year")?;
        let query_has_affiliations = u8::from(!current_query.affiliation_hashes.is_empty());
        let query_has_coauthors = u8::from(!current_query.coauthor_hashes.is_empty());
        for (rank_offset, (summary_index, score)) in scored.iter().enumerate() {
            let summary = &self.summaries[*summary_index];
            let Some(member_indices) = component_member_indices.get(&summary.component_key) else {
                return Err(RetrievalError::MissingKey(format!(
                    "Missing component members for retrieved component_key: {}",
                    summary.component_key
                )));
            };
            let chooser_features = chooser_summary_features(current_query, summary);
            let mut dominant_first_name = "";
            let mut dominant_first_count = 0u64;
            for (first_name, count) in summary.first_name_counts.iter() {
                if *count > dominant_first_count
                    || (*count == dominant_first_count && first_name.as_str() > dominant_first_name)
                {
                    dominant_first_name = first_name.as_str();
                    dominant_first_count = *count;
                }
            }

            result
                .row_query_signature_indices
                .push(query_signature_index);
            result
                .row_component_keys
                .push(summary.component_key.clone());
            result.row_retrieval_scores.push(*score);
            result
                .row_retrieval_ranks
                .push(retrieval_rank_from_zero_based_offset(
                    rank_offset,
                    "Rust retrieval",
                )?);
            result
                .row_component_sizes
                .push(summary.size.min(u32::MAX as usize) as u32);
            result
                .row_named_signature_counts
                .push(row_named_signature_count(&summary.first_name_counts)?);
            result
                .row_dominant_first_names
                .push(dominant_first_name.to_string());
            let (candidate_year_min, candidate_year_min_missing) =
                year_signal_value(summary.year_min, "candidate year_min")?;
            let (candidate_year_max, candidate_year_max_missing) =
                year_signal_value(summary.year_max, "candidate year_max")?;
            result.row_candidate_year_min.push(candidate_year_min);
            result.row_candidate_year_max.push(candidate_year_max);
            result.row_candidate_year_range_missing.push(u8::from(
                candidate_year_min_missing != 0 || candidate_year_max_missing != 0,
            ));
            result
                .row_query_first_tokens
                .push(row_query_first_token.to_string());
            result.row_query_years.push(query_year);
            result.row_query_year_missing.push(query_year_missing);
            result
                .row_query_has_affiliations
                .push(query_has_affiliations);
            result.row_query_has_coauthors.push(query_has_coauthors);
            result
                .row_orcid_match
                .push(u8::from(current_query.orcid_hash.is_some_and(
                    |orcid_hash| contains_hashed_value(&summary.orcid_hashes, orcid_hash),
                )));
            result
                .row_middle_initial_compatibility
                .push(chooser_features[0]);
            result.row_affiliation_overlap.push(chooser_features[1]);
            result.row_coauthor_overlap.push(chooser_features[2]);
            result.row_venue_overlap.push(chooser_features[3]);
            result.row_year_compatibility.push(chooser_features[4]);
            result.row_title_overlap.push(chooser_features[5]);
            result
                .row_specter_centroid_similarity
                .push(chooser_features[6]);
            result
                .row_specter_exemplar_similarity
                .push(chooser_features[7]);
            result
                .right_signature_indices_by_row
                .push(member_indices.clone());
        }
        Ok(result)
    }

    fn extract_candidate_indices_by_query_signature_id(
        &self,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<HashMap<String, Vec<usize>>> {
        let candidate_keys_by_query = extract_string_vec_map(obj)?;
        let mut out = HashMap::with_capacity(candidate_keys_by_query.len());
        for (query_signature_id, component_keys) in candidate_keys_by_query {
            let mut indices = Vec::with_capacity(component_keys.len());
            for component_key in component_keys {
                let Some(candidate_index) = self.component_index_by_key.get(&component_key) else {
                    return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                        "Unknown component_key for RustHybridCentroidRetriever query window: {component_key}"
                    )));
                };
                indices.push(*candidate_index);
            }
            out.insert(query_signature_id, indices);
        }
        Ok(out)
    }

    fn score_top_k_candidate_indices_default(
        &self,
        py: Python<'_>,
        query_data: &RetrievalQueryData,
        candidate_indices: &[usize],
        top_k: usize,
        num_threads: Option<usize>,
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        if candidate_indices.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut scored: Vec<(usize, f32)> = py.allow_threads(|| {
            let compute = || {
                candidate_indices
                    .par_iter()
                    .map(|idx| {
                        let summary = self.summary_for_candidate_index(
                            *idx,
                            override_index,
                            override_summary,
                        );
                        (
                            *idx,
                            score_hybrid_centroid_query(
                                query_data,
                                summary,
                                &self.coauthor_cluster_df,
                                &self.non_mega_coauthor_cluster_df,
                                &self.affiliation_cluster_df,
                                self.summaries.len(),
                            ),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        self.keep_sorted_top_k_scored_candidates(
            &mut scored,
            top_k,
            override_index,
            override_summary,
        );
        Ok(self.scored_candidates_to_keys_scores(scored, override_index, override_summary))
    }
}

pub(crate) fn extract_string_hashes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut hashes = Vec::new();
    for item in PyIterator::from_object(obj)? {
        let value: String = item?.extract()?;
        hashes.push(fnv64(value.as_bytes()));
    }
    hashes.sort_unstable();
    hashes.dedup();
    Ok(hashes)
}

pub(crate) fn extract_orcid_hashes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut hashes = Vec::new();
    for item in PyIterator::from_object(obj)? {
        let value: String = item?.extract()?;
        if let Some(orcid) = normalize_orcid_owned(&value) {
            hashes.push(fnv64(orcid.as_bytes()));
        }
    }
    hashes.sort_unstable();
    hashes.dedup();
    Ok(hashes)
}

pub(crate) fn extract_query_terms(obj: &Bound<'_, PyAny>) -> PyResult<Vec<RetrievalQueryTerm>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut terms = Vec::new();
    for item in PyIterator::from_object(obj)? {
        let value: String = item?.extract()?;
        terms.push(RetrievalQueryTerm {
            hash: fnv64(value.as_bytes()),
            token_count: term_token_count(&value),
        });
    }
    terms.sort_unstable_by_key(|term| term.hash);
    terms.dedup_by_key(|term| term.hash);
    Ok(terms)
}

pub(crate) fn extract_optional_orcid_hash(obj: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
    if obj.is_none() {
        return Ok(None);
    }
    let value: String = obj.extract()?;
    Ok(normalize_orcid_owned(&value).map(|orcid| fnv64(orcid.as_bytes())))
}

pub(crate) fn exact_name_match_compat(a: &str, b: &str) -> bool {
    !a.is_empty() && a == b
}

pub(crate) fn counter_query_overlap_hashes(
    query_hashes: &[u64],
    counter: &Option<CounterData>,
    size: usize,
) -> f64 {
    let Some(counter_data) = counter.as_ref() else {
        return 0.0;
    };
    if size == 0 || query_hashes.is_empty() || counter_data.entries.is_empty() {
        return 0.0;
    }
    let mut overlap = 0.0f64;
    for query_hash in query_hashes {
        if let Ok(index) = counter_data
            .entries
            .binary_search_by_key(query_hash, |entry| entry.0)
        {
            overlap += (counter_data.entries[index].1 as f64) / (size as f64);
        }
    }
    overlap / (query_hashes.len() as f64)
}

pub(crate) fn overlap_idf_weight(
    df_map: &HashMap<u64, usize>,
    hash: u64,
    total_summary_count: usize,
) -> f64 {
    let df = df_map.get(&hash).copied().unwrap_or(0) as f64;
    (((total_summary_count as f64) + 1.0) / (df + 1.0)).ln() + 1.0
}

pub(crate) fn overlap_query_term_weight(
    term: &RetrievalQueryTerm,
    df_map: &HashMap<u64, usize>,
    total_summary_count: usize,
    config: RetrievalOverlapConfig,
) -> f64 {
    if term.token_count < config.min_token_count {
        return 0.0;
    }
    let mut weight = if term.token_count <= 1 {
        config.unigram_weight
    } else {
        config.multi_token_weight
    };
    if config.use_idf {
        weight *= overlap_idf_weight(df_map, term.hash, total_summary_count);
    }
    weight.max(0.0)
}

pub(crate) fn weighted_counter_query_overlap(
    query_terms: &[RetrievalQueryTerm],
    counter: &Option<CounterData>,
    size: usize,
    df_map: &HashMap<u64, usize>,
    total_summary_count: usize,
    config: RetrievalOverlapConfig,
) -> f64 {
    let Some(counter_data) = counter.as_ref() else {
        return 0.0;
    };
    if size == 0 || query_terms.is_empty() || counter_data.entries.is_empty() {
        return 0.0;
    }
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for term in query_terms {
        let query_weight = overlap_query_term_weight(term, df_map, total_summary_count, config);
        if query_weight <= 0.0 {
            continue;
        }
        denominator += query_weight;
        if let Ok(index) = counter_data
            .entries
            .binary_search_by_key(&term.hash, |entry| entry.0)
        {
            let mut contribution = (counter_data.entries[index].1 as f64) / (size as f64);
            if let Some(cap) = config.per_term_cap {
                contribution = contribution.min(cap);
            }
            numerator += query_weight * contribution;
        }
    }
    if denominator <= 0.0 {
        return 0.0;
    }
    let mut score = numerator / denominator;
    if let Some(cap) = config.total_cap {
        score = score.min(cap);
    }
    score
}

pub(crate) fn middle_initial_score_hashes(
    query_hashes: &[u64],
    counter: &Option<CounterData>,
    size: usize,
) -> f64 {
    let Some(counter_data) = counter.as_ref() else {
        return 0.0;
    };
    if size == 0 || query_hashes.is_empty() || counter_data.entries.is_empty() {
        return 0.0;
    }
    let mut overlap = 0.0f64;
    let mut overlap_found = false;
    for query_hash in query_hashes {
        if let Ok(index) = counter_data
            .entries
            .binary_search_by_key(query_hash, |entry| entry.0)
        {
            overlap += (counter_data.entries[index].1 as f64) / (size as f64);
            overlap_found = true;
        }
    }
    if overlap_found {
        overlap / (query_hashes.len() as f64)
    } else {
        RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE
    }
}

pub(crate) fn first_name_score_mode(
    query_first: &str,
    counts: &[(String, u64)],
    size: usize,
    mode: RetrievalFirstNameMode,
) -> f64 {
    if size == 0 || py_len(query_first) <= 1 || counts.is_empty() {
        return 0.0;
    }
    let mut best = 0.0f64;
    for (first_name, count) in counts.iter() {
        if py_len(first_name) <= 1 {
            continue;
        }
        let share = (*count as f64) / (size as f64);
        let candidate = match mode {
            RetrievalFirstNameMode::Prefix => {
                if same_prefix_tokens(query_first, first_name) {
                    share
                } else {
                    0.0
                }
            }
            RetrievalFirstNameMode::ExactThenPrefixHalf => {
                if exact_name_match_compat(query_first, first_name) {
                    share
                } else if same_prefix_tokens(query_first, first_name) {
                    share * 0.5
                } else {
                    0.0
                }
            }
        };
        best = best.max(candidate);
    }
    best
}

pub(crate) fn year_score(query_year: Option<i64>, summary: &RetrievalSummaryData) -> f64 {
    let Some(query_year_value) = query_year else {
        return 0.0;
    };
    let Some(year_mean) = summary.year_mean else {
        return 0.0;
    };
    let distance = ((query_year_value as f64) - year_mean).abs();
    let mut score = (1.0 - (distance / RETRIEVAL_YEAR_SCORE_DECAY_YEARS)).max(0.0);
    if let (Some(year_min), Some(year_max)) = (summary.year_min, summary.year_max) {
        if query_year_value < year_min.saturating_sub(RETRIEVAL_YEAR_SCORE_RANGE_GAP)
            || query_year_value > year_max.saturating_add(RETRIEVAL_YEAR_SCORE_RANGE_GAP)
        {
            score -= RETRIEVAL_YEAR_SCORE_RANGE_PENALTY;
        }
    }
    score
}

pub(crate) fn contains_hashed_value(sorted_hashes: &[u64], target: u64) -> bool {
    sorted_hashes.binary_search(&target).is_ok()
}

pub(crate) fn has_middle_initial_conflict(
    query_hashes: &[u64],
    counter: &Option<CounterData>,
) -> bool {
    let Some(counter_data) = counter.as_ref() else {
        return false;
    };
    if query_hashes.is_empty() || counter_data.entries.is_empty() {
        return false;
    }
    !query_hashes.iter().any(|query_hash| {
        counter_data
            .entries
            .binary_search_by_key(query_hash, |entry| entry.0)
            .is_ok()
    })
}

pub(crate) fn has_impossible_year_conflict(
    query_year: Option<i64>,
    summary: &RetrievalSummaryData,
    max_year_gap: i64,
) -> bool {
    let Some(query_year_value) = query_year else {
        return false;
    };
    let (Some(year_min), Some(year_max)) = (summary.year_min, summary.year_max) else {
        return false;
    };
    query_year_value < year_min.saturating_sub(max_year_gap)
        || query_year_value > year_max.saturating_add(max_year_gap)
}

pub(crate) fn extract_retrieval_summary(
    obj: &Bound<'_, PyAny>,
    include_exemplars: bool,
) -> PyResult<RetrievalSummaryData> {
    let component_key: String = obj.getattr("component_key")?.extract()?;
    let size: usize = obj.getattr("size")?.extract()?;
    let first_name_counts = extract_string_count_pairs(&obj.getattr("first_name_counts")?)?;
    let middle_initial_counts = extract_counter(&obj.getattr("middle_initial_counts")?)?;
    let coauthor_counts = extract_counter(&obj.getattr("coauthor_counts")?)?;
    let non_mega_coauthor_counts = extract_counter(&obj.getattr("non_mega_coauthor_counts")?)?;
    let affiliation_counts = extract_counter(&obj.getattr("affiliation_counts")?)?;
    let venue_counts = extract_counter(&obj.getattr("venue_counts")?)?;
    let title_counts = extract_counter(&obj.getattr("title_counts")?)?;
    let max_paper_author_count: usize = obj.getattr("max_paper_author_count")?.extract()?;
    let year_min: Option<i64> = obj.getattr("year_min")?.extract()?;
    let year_max: Option<i64> = obj.getattr("year_max")?.extract()?;
    let year_mean: Option<f64> = obj.getattr("year_mean")?.extract()?;
    let orcid_hashes = extract_orcid_hashes(&obj.getattr("orcid_values")?)?;
    let specter_centroid = extract_specter_vec(&obj.getattr("specter_centroid")?)?;
    let specter_centroid_norm = specter_centroid
        .as_ref()
        .map(|values| vector_norm_f32(values));
    let exemplar_vectors = if include_exemplars {
        extract_specter_vec_list(&obj.getattr("exemplar_vectors")?)?
    } else {
        Vec::new()
    };
    let exemplar_norms = exemplar_vectors
        .iter()
        .map(|values| vector_norm_f32(values))
        .collect();

    Ok(RetrievalSummaryData {
        component_key,
        size,
        first_name_counts,
        middle_initial_counts,
        coauthor_counts,
        non_mega_coauthor_counts,
        affiliation_counts,
        venue_counts,
        title_counts,
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

pub(crate) fn extract_retrieval_query(obj: &Bound<'_, PyAny>) -> PyResult<RetrievalQueryData> {
    let first: String = obj.getattr("first")?.extract()?;
    let has_full_first = match obj.getattr("has_full_first") {
        Ok(value) => value.extract()?,
        Err(_) => first.chars().count() > 1,
    };
    let middle_initial_hashes = extract_string_hashes(&obj.getattr("middle_initials")?)?;
    let coauthor_terms = extract_query_terms(&obj.getattr("coauthor_blocks")?)?;
    let coauthor_hashes = coauthor_terms.iter().map(|term| term.hash).collect();
    let affiliation_terms = extract_query_terms(&obj.getattr("affiliation_terms")?)?;
    let affiliation_hashes = affiliation_terms.iter().map(|term| term.hash).collect();
    let venue_hashes = extract_string_hashes(&obj.getattr("venue_terms")?)?;
    let title_hashes = extract_string_hashes(&obj.getattr("title_terms")?)?;
    let year: Option<i64> = obj.getattr("year")?.extract()?;
    let orcid_hash = extract_optional_orcid_hash(&obj.getattr("orcid")?)?;
    let specter = extract_specter_vec(&obj.getattr("specter")?)?;
    let specter_norm = specter.as_ref().map(|values| vector_norm_f32(values));
    let specter = specter.map(Arc::new);

    Ok(RetrievalQueryData {
        first,
        has_full_first,
        middle_initial_hashes,
        coauthor_hashes,
        coauthor_terms,
        affiliation_hashes,
        affiliation_terms,
        venue_hashes,
        title_hashes,
        year,
        orcid_hash,
        specter,
        specter_norm,
    })
}

const COAUTHOR_OVERLAP_CONFIG: RetrievalOverlapConfig = RetrievalOverlapConfig {
    use_idf: true,
    per_term_cap: Some(0.35),
    total_cap: None,
    min_token_count: 1,
    unigram_weight: 1.0,
    multi_token_weight: 1.0,
};

const AFFILIATION_OVERLAP_CONFIG: RetrievalOverlapConfig = RetrievalOverlapConfig {
    use_idf: true,
    per_term_cap: None,
    total_cap: None,
    min_token_count: 1,
    unigram_weight: 1.0,
    multi_token_weight: 1.0,
};

pub(crate) fn specter_exemplar_score(
    query: &RetrievalQueryData,
    summary: &RetrievalSummaryData,
) -> f64 {
    let (Some(query_specter), Some(query_norm)) = (query.specter.as_ref(), query.specter_norm)
    else {
        return 0.0;
    };
    summary
        .exemplar_vectors
        .iter()
        .zip(summary.exemplar_norms.iter())
        .map(|(vector, norm)| cosine_sim_with_norms(query_specter, query_norm, vector, *norm))
        .fold(0.0f64, f64::max)
}

pub(crate) fn query_counter_overlap_count(
    query_terms: &[RetrievalQueryTerm],
    counter: &Option<CounterData>,
) -> usize {
    let Some(counter_data) = counter.as_ref() else {
        return 0;
    };
    query_terms
        .iter()
        .filter(|term| {
            counter_data
                .entries
                .binary_search_by_key(&term.hash, |entry| entry.0)
                .is_ok()
        })
        .count()
}

pub(crate) fn should_rescue_candidate_mega_coauthors(
    query: &RetrievalQueryData,
    summary: &RetrievalSummaryData,
) -> bool {
    if summary.max_paper_author_count < RETRIEVAL_MEGA_AUTHOR_THRESHOLD
        || query.coauthor_terms.is_empty()
    {
        return false;
    }

    let full_overlap = query_counter_overlap_count(&query.coauthor_terms, &summary.coauthor_counts);
    if full_overlap < 3 {
        return false;
    }
    let filtered_overlap =
        query_counter_overlap_count(&query.coauthor_terms, &summary.non_mega_coauthor_counts);
    if full_overlap <= filtered_overlap {
        return false;
    }

    (full_overlap as f64) / (query.coauthor_terms.len() as f64) >= 0.995
}

pub(crate) fn score_hybrid_centroid_query(
    query: &RetrievalQueryData,
    summary: &RetrievalSummaryData,
    coauthor_cluster_df: &HashMap<u64, usize>,
    non_mega_coauthor_cluster_df: &HashMap<u64, usize>,
    affiliation_cluster_df: &HashMap<u64, usize>,
    total_summary_count: usize,
) -> f32 {
    let weights = if query.has_full_first {
        DEFAULT_HYBRID_CENTROID_WEIGHTS
    } else {
        DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS
    };
    let use_non_mega_coauthor_counter = summary.max_paper_author_count
        >= RETRIEVAL_MEGA_AUTHOR_THRESHOLD
        && !should_rescue_candidate_mega_coauthors(query, summary);
    let (coauthor_counts, coauthor_df) = if use_non_mega_coauthor_counter {
        (
            &summary.non_mega_coauthor_counts,
            non_mega_coauthor_cluster_df,
        )
    } else {
        (&summary.coauthor_counts, coauthor_cluster_df)
    };
    let coauthor_score = weighted_counter_query_overlap(
        &query.coauthor_terms,
        coauthor_counts,
        summary.size,
        coauthor_df,
        total_summary_count,
        COAUTHOR_OVERLAP_CONFIG,
    );
    let affiliation_score = weighted_counter_query_overlap(
        &query.affiliation_terms,
        &summary.affiliation_counts,
        summary.size,
        affiliation_cluster_df,
        total_summary_count,
        AFFILIATION_OVERLAP_CONFIG,
    );
    let middle_score = middle_initial_score_hashes(
        &query.middle_initial_hashes,
        &summary.middle_initial_counts,
        summary.size,
    );
    let first_name_score = first_name_score_mode(
        &query.first,
        &summary.first_name_counts,
        summary.size,
        if query.has_full_first {
            RetrievalFirstNameMode::ExactThenPrefixHalf
        } else {
            RetrievalFirstNameMode::Prefix
        },
    );
    let centroid_score = match (
        query.specter.as_ref(),
        query.specter_norm,
        summary.specter_centroid.as_ref(),
        summary.specter_centroid_norm,
    ) {
        (Some(query_specter), Some(query_norm), Some(summary_specter), Some(summary_norm)) => {
            cosine_sim_with_norms(query_specter, query_norm, summary_specter, summary_norm)
        }
        _ => 0.0,
    };
    let exemplar_score = specter_exemplar_score(query, summary);
    let specter_score = centroid_score.max(exemplar_score);
    (weights.centroid * specter_score
        + weights.coauthor * coauthor_score
        + weights.affiliation * affiliation_score
        + weights.middle * middle_score
        + weights.first_name * first_name_score) as f32
}

pub(crate) fn chooser_summary_features(
    query: &RetrievalQueryData,
    summary: &RetrievalSummaryData,
) -> [f32; 8] {
    let middle_score = middle_initial_score_hashes(
        &query.middle_initial_hashes,
        &summary.middle_initial_counts,
        summary.size,
    ) as f32;
    let affiliation_score = counter_query_overlap_hashes(
        &query.affiliation_hashes,
        &summary.affiliation_counts,
        summary.size,
    ) as f32;
    let coauthor_score = counter_query_overlap_hashes(
        &query.coauthor_hashes,
        &summary.coauthor_counts,
        summary.size,
    ) as f32;
    let venue_score =
        counter_query_overlap_hashes(&query.venue_hashes, &summary.venue_counts, summary.size)
            as f32;
    let year_score_value = year_score(query.year, summary) as f32;
    let title_score =
        counter_query_overlap_hashes(&query.title_hashes, &summary.title_counts, summary.size)
            as f32;
    let specter_centroid_score = match (
        query.specter.as_ref(),
        query.specter_norm,
        summary.specter_centroid.as_ref(),
        summary.specter_centroid_norm,
    ) {
        (Some(query_specter), Some(query_norm), Some(summary_specter), Some(summary_norm)) => {
            cosine_sim_with_norms(query_specter, query_norm, summary_specter, summary_norm) as f32
        }
        _ => 0.0,
    };
    let specter_exemplar_score = specter_exemplar_score(query, summary) as f32;
    [
        middle_score,
        affiliation_score,
        coauthor_score,
        venue_score,
        year_score_value,
        title_score,
        specter_centroid_score,
        specter_exemplar_score,
    ]
}

pub(crate) fn update_cluster_df_from_counter(
    obj: &Bound<'_, PyAny>,
    df_map: &mut HashMap<u64, usize>,
) -> PyResult<()> {
    if obj.is_none() {
        return Ok(());
    }
    let dict = obj.downcast::<PyDict>()?;
    for (key_obj, _value_obj) in dict.iter() {
        let key: String = key_obj.extract()?;
        let hash = fnv64(key.as_bytes());
        *df_map.entry(hash).or_insert(0) += 1;
    }
    Ok(())
}

impl RustNameCompatibleSubblockSelector {
    fn from_py(
        retrieval_subblock_index: &Bound<'_, PyAny>,
        name_tuples: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let signature_to_subblock = extract_string_string_map(
            &retrieval_subblock_index.get_item("signature_to_subblock")?,
        )?;
        let subblock_to_components =
            extract_string_vec_map(&retrieval_subblock_index.get_item("subblock_to_components")?)?;
        let subblock_tokens_by_subblock =
            match retrieval_subblock_index.get_item("subblock_tokens_by_subblock") {
                Ok(tokens_obj) => extract_string_vec_map(&tokens_obj)?,
                Err(_) => subblock_to_components
                    .keys()
                    .map(|subblock| (subblock.clone(), subblock_tokens_from_key(subblock)))
                    .collect(),
            };
        let name_tuples = extract_name_tuples_map(name_tuples)?;
        Ok(Self {
            signature_to_subblock,
            subblock_to_components,
            subblock_tokens_by_subblock,
            name_tuples,
        })
    }

    fn allowed_component_keys(
        &self,
        query_signature_id: &str,
        query_first: &str,
    ) -> Option<HashSet<String>> {
        let query_subblock = self.signature_to_subblock.get(query_signature_id)?;
        let mut allowed_components: HashSet<String> = HashSet::new();
        if let Some(components) = self.subblock_to_components.get(query_subblock) {
            allowed_components.extend(components.iter().cloned());
        }
        for (subblock, tokens) in self.subblock_tokens_by_subblock.iter() {
            if tokens
                .iter()
                .any(|token| first_names_name_compatible(query_first, token, &self.name_tuples))
            {
                if let Some(components) = self.subblock_to_components.get(subblock) {
                    allowed_components.extend(components.iter().cloned());
                }
            }
        }
        Some(allowed_components)
    }

    fn select_candidate_indices_for_summaries(
        &self,
        query_signature_id: &str,
        query_first: &str,
        summaries: &[RetrievalSummaryData],
        base_candidate_indices: Option<&[usize]>,
        global_backfill_count: usize,
    ) -> Option<Vec<usize>> {
        let allowed_components = self.allowed_component_keys(query_signature_id, query_first)?;
        let ordered_indices: Vec<usize> = base_candidate_indices
            .map_or_else(|| (0..summaries.len()).collect(), |values| values.to_vec());
        let mut selected: Vec<usize> = ordered_indices
            .iter()
            .copied()
            .filter(|index| allowed_components.contains(&summaries[*index].component_key))
            .collect();
        if selected.is_empty() {
            return None;
        }
        if global_backfill_count > 0 {
            let mut selected_set: HashSet<String> = selected
                .iter()
                .map(|index| summaries[*index].component_key.clone())
                .collect();
            let mut remaining = global_backfill_count;
            for index in ordered_indices {
                if remaining == 0 {
                    break;
                }
                if selected_set.insert(summaries[index].component_key.clone()) {
                    selected.push(index);
                    remaining -= 1;
                }
            }
        }
        Some(selected)
    }
}

#[pymethods]
impl RustHybridCentroidRetriever {
    #[new]
    #[pyo3(signature = (summaries, include_exemplars = false))]
    fn new(summaries: &Bound<'_, PyAny>, include_exemplars: bool) -> PyResult<Self> {
        let mut packed_summaries = Vec::new();
        let mut component_index_by_key = HashMap::new();
        let mut coauthor_cluster_df = HashMap::new();
        let mut non_mega_coauthor_cluster_df = HashMap::new();
        let mut affiliation_cluster_df = HashMap::new();
        for item in PyIterator::from_object(summaries)? {
            let summary_obj = item?;
            update_cluster_df_from_counter(
                &summary_obj.getattr("coauthor_counts")?,
                &mut coauthor_cluster_df,
            )?;
            update_cluster_df_from_counter(
                &summary_obj.getattr("non_mega_coauthor_counts")?,
                &mut non_mega_coauthor_cluster_df,
            )?;
            update_cluster_df_from_counter(
                &summary_obj.getattr("affiliation_counts")?,
                &mut affiliation_cluster_df,
            )?;
            let summary = extract_retrieval_summary(&summary_obj, include_exemplars)?;
            component_index_by_key.insert(summary.component_key.clone(), packed_summaries.len());
            packed_summaries.push(summary);
        }
        Ok(Self {
            summaries: packed_summaries,
            component_index_by_key,
            coauthor_cluster_df,
            non_mega_coauthor_cluster_df,
            affiliation_cluster_df,
        })
    }

    #[pyo3(signature = (
        queries,
        query_signature_indices,
        component_member_indices_by_key,
        top_k,
        num_threads = None,
        query_signature_ids = None,
        retrieval_subblock_index = None,
        query_candidate_component_keys_by_signature_id = None,
        full_first_global_backfill_count = 0,
        name_tuples = None
    ))]
    fn top_k_hybrid_centroid_pair_plan<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        query_signature_indices: PyReadonlyArray1<'py, u32>,
        component_member_indices_by_key: &Bound<'py, PyAny>,
        top_k: usize,
        num_threads: Option<usize>,
        query_signature_ids: Option<&Bound<'py, PyAny>>,
        retrieval_subblock_index: Option<&Bound<'py, PyAny>>,
        query_candidate_component_keys_by_signature_id: Option<&Bound<'py, PyAny>>,
        full_first_global_backfill_count: usize,
        name_tuples: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        validate_retrieval_rank_top_k(top_k)?;
        let mut query_data = Vec::new();
        for item in PyIterator::from_object(queries)? {
            query_data.push(extract_retrieval_query(&item?)?);
        }
        let query_indices_slice = query_signature_indices.as_slice()?;
        if query_data.len() != query_indices_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "queries and query_signature_indices must have equal length: {} != {}",
                query_data.len(),
                query_indices_slice.len()
            )));
        }
        let query_indices = query_indices_slice.to_vec();
        let query_signature_ids = query_signature_ids
            .map(|values| {
                PyIterator::from_object(values)?
                    .map(|item| item.and_then(|value| value.extract::<String>()))
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        if let Some(values) = query_signature_ids.as_ref() {
            if values.len() != query_data.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "queries and query_signature_ids must have equal length: {} != {}",
                    query_data.len(),
                    values.len()
                )));
            }
        }
        if retrieval_subblock_index.is_some() && query_signature_ids.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "query_signature_ids are required when retrieval_subblock_index is provided",
            ));
        }
        if query_candidate_component_keys_by_signature_id.is_some() && query_signature_ids.is_none()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "query_signature_ids are required when query candidate component keys are provided",
            ));
        }
        let selector = match retrieval_subblock_index {
            Some(index) => {
                let pairs = name_tuples.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "name_tuples are required when retrieval_subblock_index is provided",
                    )
                })?;
                Some(RustNameCompatibleSubblockSelector::from_py(index, pairs)?)
            }
            None => None,
        };
        let query_candidate_indices_by_signature_id =
            query_candidate_component_keys_by_signature_id
                .map(|mapping| self.extract_candidate_indices_by_query_signature_id(mapping))
                .transpose()?;
        let component_member_indices =
            extract_component_member_indices(component_member_indices_by_key)?;

        let mut row_query_signature_indices = Vec::<u32>::new();
        let mut row_component_keys = Vec::<String>::new();
        let mut row_retrieval_scores = Vec::<f32>::new();
        let mut row_retrieval_ranks = Vec::<u16>::new();
        let mut row_component_sizes = Vec::<u32>::new();
        let mut row_named_signature_counts = Vec::<u32>::new();
        let mut row_dominant_first_names = Vec::<String>::new();
        let mut row_candidate_year_min = Vec::<i32>::new();
        let mut row_candidate_year_max = Vec::<i32>::new();
        let mut row_candidate_year_range_missing = Vec::<u8>::new();
        let mut row_query_first_tokens = Vec::<String>::new();
        let mut row_query_years = Vec::<i32>::new();
        let mut row_query_year_missing = Vec::<u8>::new();
        let mut row_query_has_affiliations = Vec::<u8>::new();
        let mut row_query_has_coauthors = Vec::<u8>::new();
        let mut row_orcid_match = Vec::<u8>::new();
        let mut row_middle_initial_compatibility = Vec::<f32>::new();
        let mut row_affiliation_overlap = Vec::<f32>::new();
        let mut row_coauthor_overlap = Vec::<f32>::new();
        let mut row_venue_overlap = Vec::<f32>::new();
        let mut row_year_compatibility = Vec::<f32>::new();
        let mut row_title_overlap = Vec::<f32>::new();
        let mut row_specter_centroid_similarity = Vec::<f32>::new();
        let mut row_specter_exemplar_similarity = Vec::<f32>::new();
        let mut left_signature_indices = Vec::<u32>::new();
        let mut right_signature_indices = Vec::<u32>::new();
        let mut pair_row_indices = Vec::<u32>::new();

        let query_results: Vec<Result<RetrievalPairPlanQueryResult, RetrievalError>> = py
            .allow_threads(|| {
                let compute = || {
                    query_data
                        .par_iter()
                        .enumerate()
                        .map(|(query_offset, current_query)| {
                            let query_signature_id = query_signature_ids
                                .as_ref()
                                .map(|values| values[query_offset].as_str());
                            let base_candidate_indices =
                                query_signature_id.and_then(|signature_id| {
                                    query_candidate_indices_by_signature_id.as_ref().and_then(
                                        |mapping| mapping.get(signature_id).map(Vec::as_slice),
                                    )
                                });
                            self.build_pair_plan_query_result(
                                current_query,
                                current_query.first.as_str(),
                                query_indices[query_offset],
                                base_candidate_indices,
                                None,
                                query_signature_id,
                                &component_member_indices,
                                top_k,
                                selector.as_ref(),
                                full_first_global_backfill_count,
                                true,
                            )
                        })
                        .collect::<Vec<_>>()
                };
                install_with_optional_rayon_pool(num_threads, compute)
            });

        for query_result in query_results {
            let mut query_result = query_result.map_err(retrieval_error_to_py)?;
            let base_row_index = u32::try_from(row_component_keys.len()).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(
                    "retrieved candidate row count exceeds u32",
                )
            })?;
            for (local_row_index, member_indices) in query_result
                .right_signature_indices_by_row
                .iter()
                .enumerate()
            {
                let row_index = checked_retrieved_row_index(base_row_index, local_row_index)?;
                let query_signature_index =
                    query_result.row_query_signature_indices[local_row_index];
                for member_index in member_indices.iter() {
                    left_signature_indices.push(query_signature_index);
                    right_signature_indices.push(*member_index);
                    pair_row_indices.push(row_index);
                }
            }
            row_query_signature_indices.append(&mut query_result.row_query_signature_indices);
            row_component_keys.append(&mut query_result.row_component_keys);
            row_retrieval_scores.append(&mut query_result.row_retrieval_scores);
            row_retrieval_ranks.append(&mut query_result.row_retrieval_ranks);
            row_component_sizes.append(&mut query_result.row_component_sizes);
            row_named_signature_counts.append(&mut query_result.row_named_signature_counts);
            row_dominant_first_names.append(&mut query_result.row_dominant_first_names);
            row_candidate_year_min.append(&mut query_result.row_candidate_year_min);
            row_candidate_year_max.append(&mut query_result.row_candidate_year_max);
            row_candidate_year_range_missing
                .append(&mut query_result.row_candidate_year_range_missing);
            row_query_first_tokens.append(&mut query_result.row_query_first_tokens);
            row_query_years.append(&mut query_result.row_query_years);
            row_query_year_missing.append(&mut query_result.row_query_year_missing);
            row_query_has_affiliations.append(&mut query_result.row_query_has_affiliations);
            row_query_has_coauthors.append(&mut query_result.row_query_has_coauthors);
            row_orcid_match.append(&mut query_result.row_orcid_match);
            row_middle_initial_compatibility
                .append(&mut query_result.row_middle_initial_compatibility);
            row_affiliation_overlap.append(&mut query_result.row_affiliation_overlap);
            row_coauthor_overlap.append(&mut query_result.row_coauthor_overlap);
            row_venue_overlap.append(&mut query_result.row_venue_overlap);
            row_year_compatibility.append(&mut query_result.row_year_compatibility);
            row_title_overlap.append(&mut query_result.row_title_overlap);
            row_specter_centroid_similarity
                .append(&mut query_result.row_specter_centroid_similarity);
            row_specter_exemplar_similarity
                .append(&mut query_result.row_specter_exemplar_similarity);
        }

        let payload = PyDict::new(py);
        payload.set_item("row_count", row_component_keys.len())?;
        payload.set_item(
            "left_signature_indices",
            left_signature_indices.to_pyarray(py),
        )?;
        payload.set_item(
            "right_signature_indices",
            right_signature_indices.to_pyarray(py),
        )?;
        payload.set_item("pair_row_indices", pair_row_indices.to_pyarray(py))?;
        payload.set_item(
            "row_query_signature_indices",
            row_query_signature_indices.to_pyarray(py),
        )?;
        payload.set_item("row_component_keys", row_component_keys)?;
        payload.set_item("retrieval_scores", row_retrieval_scores.to_pyarray(py))?;
        payload.set_item("retrieval_ranks", row_retrieval_ranks.to_pyarray(py))?;
        payload.set_item("row_component_sizes", row_component_sizes.to_pyarray(py))?;
        payload.set_item(
            "row_named_signature_counts",
            row_named_signature_counts.to_pyarray(py),
        )?;
        payload.set_item("row_dominant_first_names", row_dominant_first_names)?;
        payload.set_item(
            "row_candidate_year_min",
            row_candidate_year_min.to_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_year_max",
            row_candidate_year_max.to_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_year_range_missing",
            row_candidate_year_range_missing.to_pyarray(py),
        )?;
        payload.set_item("row_query_first_tokens", row_query_first_tokens)?;
        payload.set_item("row_query_years", row_query_years.to_pyarray(py))?;
        payload.set_item(
            "row_query_year_missing",
            row_query_year_missing.to_pyarray(py),
        )?;
        payload.set_item(
            "row_query_has_affiliations",
            row_query_has_affiliations.to_pyarray(py),
        )?;
        payload.set_item(
            "row_query_has_coauthors",
            row_query_has_coauthors.to_pyarray(py),
        )?;
        payload.set_item("row_orcid_match", row_orcid_match.to_pyarray(py))?;
        payload.set_item(
            "middle_initial_compatibility",
            row_middle_initial_compatibility.to_pyarray(py),
        )?;
        payload.set_item(
            "affiliation_overlap",
            row_affiliation_overlap.to_pyarray(py),
        )?;
        payload.set_item("coauthor_overlap", row_coauthor_overlap.to_pyarray(py))?;
        payload.set_item("venue_overlap", row_venue_overlap.to_pyarray(py))?;
        payload.set_item("year_compatibility", row_year_compatibility.to_pyarray(py))?;
        payload.set_item("title_overlap", row_title_overlap.to_pyarray(py))?;
        payload.set_item(
            "specter_centroid_similarity",
            row_specter_centroid_similarity.to_pyarray(py),
        )?;
        payload.set_item(
            "specter_exemplar_similarity",
            row_specter_exemplar_similarity.to_pyarray(py),
        )?;
        Ok(payload.unbind())
    }

    #[pyo3(signature = (query, component_keys, top_k, num_threads = None, override_summary = None))]
    fn top_k_hybrid_centroid_subset(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        component_keys: &Bound<'_, PyAny>,
        top_k: usize,
        num_threads: Option<usize>,
        override_summary: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        validate_positive_top_k(top_k)?;

        let query_data = extract_retrieval_query(query)?;
        let mut candidate_indices = Vec::new();
        for item in PyIterator::from_object(component_keys)? {
            let component_key: String = item?.extract()?;
            let Some(candidate_index) = self.component_index_by_key.get(&component_key) else {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "Unknown component_key for RustHybridCentroidRetriever: {component_key}"
                )));
            };
            candidate_indices.push(*candidate_index);
        }

        let override_data = override_summary
            .map(|value| extract_retrieval_summary(value, true))
            .transpose()?;
        let override_index = if let Some(override_summary_data) = override_data.as_ref() {
            let Some(candidate_index) = self
                .component_index_by_key
                .get(&override_summary_data.component_key)
            else {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "Unknown override component_key for RustHybridCentroidRetriever: {}",
                    override_summary_data.component_key
                )));
            };
            if !candidate_indices.contains(candidate_index) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "override_summary component_key {} was not present in component_keys",
                    override_summary_data.component_key
                )));
            }
            Some(*candidate_index)
        } else {
            None
        };

        self.score_top_k_candidate_indices_default(
            py,
            &query_data,
            &candidate_indices,
            top_k,
            num_threads,
            override_index,
            override_data.as_ref(),
        )
    }
}

pub(crate) fn checked_retrieved_row_index(
    base_row_index: u32,
    local_row_index: usize,
) -> PyResult<u32> {
    let local_row_index = u32::try_from(local_row_index).map_err(|_| {
        pyo3::exceptions::PyOverflowError::new_err("retrieved candidate row count exceeds u32")
    })?;
    base_row_index.checked_add(local_row_index).ok_or_else(|| {
        pyo3::exceptions::PyOverflowError::new_err("retrieved candidate row count exceeds u32")
    })
}

pub(crate) fn validate_positive_top_k(top_k: usize) -> PyResult<()> {
    if top_k == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "top_k must be positive",
        ));
    }
    Ok(())
}

pub(crate) fn validate_retrieval_rank_top_k(top_k: usize) -> PyResult<()> {
    validate_positive_top_k(top_k)?;
    if top_k > u16::MAX as usize {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "top_k must be <= {} because retrieval_ranks are stored as uint16",
            u16::MAX
        )));
    }
    Ok(())
}

pub(crate) fn retrieval_rank_from_zero_based_offset(
    rank_offset: usize,
    context: &str,
) -> Result<u16, RetrievalError> {
    let one_based_rank = rank_offset.checked_add(1).ok_or_else(|| {
        RetrievalError::Overflow(format!("{context} retrieval rank overflowed usize"))
    })?;
    u16::try_from(one_based_rank).map_err(|_| {
        RetrievalError::Overflow(format!(
            "{context} produced retrieval rank {one_based_rank}, but retrieval_ranks are stored as uint16 and support ranks in [1, {}]",
            u16::MAX
        ))
    })
}
