use super::*;
use crate::retrieval_overlap::{
    component_overlap, overlap_scores, ComponentOverlap, OverlapIndex, PreparedQuery,
};
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
pub(crate) const RAW_ARROW_QUERY_SIGNATURE_PLANNER_METHODS: [&str; 6] = [
    "from_query_signatures",
    "from_auto_queries",
    "plan_query_signatures",
    "plan",
    "name_counts_index",
    "build_telemetry",
];
pub(crate) const RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE: f64 = -0.25;
pub(crate) const RETRIEVAL_YEAR_SCORE_DECAY_YEARS: f64 = 15.0;
pub(crate) const RETRIEVAL_YEAR_SCORE_RANGE_GAP: i64 = 10;
pub(crate) const RETRIEVAL_YEAR_SCORE_RANGE_PENALTY: f64 = 0.15;
const RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP: i64 = 35;
pub(crate) const RETRIEVAL_MEGA_AUTHOR_THRESHOLD: usize = 50;

fn candidate_indices_excluding(
    summary_count: usize,
    excluded_candidate_indices: Option<&HashSet<usize>>,
) -> Vec<usize> {
    let indices = (0..summary_count).collect::<Vec<_>>();
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
    /// Inverted coauthor and affiliation counters of `summaries`. Retrievers
    /// that only score given rows, such as the labeled-row plan, skip it.
    overlap_index: Option<OverlapIndex>,
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
    /// Index `summaries`: component keys, per-term cluster document frequencies
    /// and, when the retriever ranks candidates, the inverted overlap counters.
    pub(crate) fn from_summaries(
        summaries: Vec<RetrievalSummaryData>,
        with_overlap_index: bool,
    ) -> Result<Self, RetrievalError> {
        let mut component_index_by_key = HashMap::with_capacity(summaries.len());
        let mut coauthor_cluster_df = HashMap::new();
        let mut non_mega_coauthor_cluster_df = HashMap::new();
        let mut affiliation_cluster_df = HashMap::new();
        for (index, summary) in summaries.iter().enumerate() {
            component_index_by_key.insert(summary.component_key.clone(), index);
            increment_df_from_counter(&summary.coauthor_counts, &mut coauthor_cluster_df);
            increment_df_from_counter(
                &summary.non_mega_coauthor_counts,
                &mut non_mega_coauthor_cluster_df,
            );
            increment_df_from_counter(&summary.affiliation_counts, &mut affiliation_cluster_df);
        }
        let overlap_index = with_overlap_index
            .then(|| OverlapIndex::build(&summaries))
            .transpose()?;
        Ok(Self {
            summaries,
            component_index_by_key,
            coauthor_cluster_df,
            non_mega_coauthor_cluster_df,
            affiliation_cluster_df,
            overlap_index,
        })
    }

    fn overlap_index(&self) -> &OverlapIndex {
        self.overlap_index
            .as_ref()
            .expect("retrievers that rank candidates are built with an overlap index")
    }

    fn prepare_query(&self, query: &RetrievalQueryData) -> PreparedQuery {
        PreparedQuery::new(
            query,
            &self.coauthor_cluster_df,
            &self.non_mega_coauthor_cluster_df,
            &self.affiliation_cluster_df,
            self.summaries.len(),
        )
    }

    /// Hybrid score of candidate `idx`. Indexed components use the query's
    /// posting walk; an override summary is looked up directly.
    fn score_candidate(
        &self,
        query_data: &RetrievalQueryData,
        prepared: &PreparedQuery,
        overlaps: &[ComponentOverlap],
        idx: usize,
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
        specter_scores: Option<&[f64]>,
    ) -> f32 {
        let summary = self.summary_for_candidate_index(idx, override_index, override_summary);
        let overridden = Some(idx) == override_index && override_summary.is_some();
        let overlap = if overridden {
            component_overlap(query_data, prepared, summary)
        } else {
            overlaps[idx]
        };
        let (coauthor_score, affiliation_score) =
            overlap_scores(query_data, prepared, summary, &overlap);
        // A precomputed row holds this query's SPECTER score of every summary;
        // an override summary is not in it.
        let specter_score = match specter_scores {
            Some(row) if !overridden => row[idx],
            _ => fused_specter_score(query_data, summary),
        };
        hybrid_centroid_score(
            query_data,
            summary,
            coauthor_score,
            affiliation_score,
            specter_score,
        )
    }

    /// SPECTER scores of each query against every summary, one row per query.
    ///
    /// Summaries are the outer loop, so each summary's vectors are read from
    /// memory once per block of queries instead of once per query; at hundreds
    /// of thousands of summaries that traffic, not arithmetic, bounds scoring.
    pub(crate) fn specter_score_rows(&self, queries: &[RetrievalQueryData]) -> Vec<Vec<f64>> {
        let mut rows = vec![vec![0.0f64; self.summaries.len()]; queries.len()];
        for (index, summary) in self.summaries.iter().enumerate() {
            for (row, query) in rows.iter_mut().zip(queries) {
                row[index] = fused_specter_score(query, summary);
            }
        }
        rows
    }

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
        specter_scores: Option<&[f64]>,
    ) -> Vec<(usize, f32)> {
        let prepared = self.prepare_query(query_data);
        let overlaps =
            self.overlap_index()
                .component_overlaps(query_data, &prepared, self.summaries.len());
        let mut scored = candidate_indices
            .iter()
            .map(|idx| {
                (
                    *idx,
                    self.score_candidate(
                        query_data,
                        &prepared,
                        &overlaps,
                        *idx,
                        override_index,
                        override_summary,
                        specter_scores,
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
        excluded_candidate_indices: Option<&HashSet<usize>>,
    ) -> RetrievalCandidateSelection {
        let candidate_indices =
            candidate_indices_excluding(self.summaries.len(), excluded_candidate_indices);
        if let Some(selection) =
            self.orcid_candidate_selection_for_query(query_data, candidate_indices.iter().copied())
        {
            return selection;
        }
        self.hard_filtered_candidate_indices_for_query(query_data, candidate_indices)
    }

    pub(crate) fn build_pair_plan_query_result(
        &self,
        current_query: &RetrievalQueryData,
        row_query_first_token: &str,
        query_signature_index: u32,
        excluded_candidate_indices: Option<&HashSet<usize>>,
        component_member_indices: &HashMap<String, Vec<u32>>,
        top_k: usize,
        specter_scores: Option<&[f64]>,
    ) -> Result<RetrievalPairPlanQueryResult, RetrievalError> {
        let selection =
            self.candidate_indices_for_pair_plan_query(current_query, excluded_candidate_indices);
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
            specter_scores,
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

        let prepared = self.prepare_query(query_data);
        let mut scored: Vec<(usize, f32)> = py.allow_threads(|| {
            let compute = || {
                let overlaps = self.overlap_index().component_overlaps(
                    query_data,
                    &prepared,
                    self.summaries.len(),
                );
                candidate_indices
                    .par_iter()
                    .map(|idx| {
                        (
                            *idx,
                            self.score_candidate(
                                query_data,
                                &prepared,
                                &overlaps,
                                *idx,
                                override_index,
                                override_summary,
                                None,
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

pub(crate) const COAUTHOR_OVERLAP_CONFIG: RetrievalOverlapConfig = RetrievalOverlapConfig {
    use_idf: true,
    per_term_cap: Some(0.35),
    total_cap: None,
    min_token_count: 1,
    unigram_weight: 1.0,
    multi_token_weight: 1.0,
};

pub(crate) const AFFILIATION_OVERLAP_CONFIG: RetrievalOverlapConfig = RetrievalOverlapConfig {
    use_idf: true,
    per_term_cap: None,
    total_cap: None,
    min_token_count: 1,
    unigram_weight: 1.0,
    multi_token_weight: 1.0,
};

/// Independent partial sums per dot product; a multiple of every SIMD width.
const DOT_LANES: usize = 8;

/// Dot products of `query` with N equal-length vectors in one pass, in f64.
///
/// Retrieval scores every component for every query, so these sums are most of
/// incremental planning's CPU time. Each product keeps `DOT_LANES` partial sums
/// that the compiler vectorizes, then folds them in a fixed order, so results
/// are deterministic. The order differs from a sequential sum, so values can
/// differ from `cosine_sim_with_norms` by about 1e-15 relative; scores are
/// rounded to f32 afterwards, and retrieval only ranks them.
fn fused_dot_products<const N: usize>(query: &[f32], vectors: [&[f32]; N]) -> [f64; N] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the running CPU supports AVX2.
            return unsafe { fused_dot_products_avx2(query, vectors) };
        }
    }
    lane_dot_products(query, vectors)
}

/// `lane_dot_products` with AVX2 intrinsics. Lanes 0-3 and 4-7 of each sum live
/// in two 256-bit registers and take the same separate multiply and add, so the
/// results equal the portable code on every CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fused_dot_products_avx2<const N: usize>(query: &[f32], vectors: [&[f32]; N]) -> [f64; N] {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_castps256_ps128, _mm256_cvtps_pd, _mm256_extractf128_ps,
        _mm256_loadu_ps, _mm256_mul_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };
    let len = query.len();
    let vectors = vectors.map(|vector| &vector[..len]);
    let body = len - len % DOT_LANES;
    let mut low = [_mm256_setzero_pd(); N];
    let mut high = [_mm256_setzero_pd(); N];
    let mut start = 0;
    while start < body {
        let q = _mm256_loadu_ps(query.as_ptr().add(start));
        let q_low = _mm256_cvtps_pd(_mm256_castps256_ps128(q));
        let q_high = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(q));
        for k in 0..N {
            let v = _mm256_loadu_ps(vectors[k].as_ptr().add(start));
            low[k] = _mm256_add_pd(
                low[k],
                _mm256_mul_pd(q_low, _mm256_cvtps_pd(_mm256_castps256_ps128(v))),
            );
            high[k] = _mm256_add_pd(
                high[k],
                _mm256_mul_pd(q_high, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v))),
            );
        }
        start += DOT_LANES;
    }
    let mut sums = [0.0f64; N];
    for k in 0..N {
        let mut partial = [0.0f64; DOT_LANES];
        _mm256_storeu_pd(partial.as_mut_ptr(), low[k]);
        _mm256_storeu_pd(partial.as_mut_ptr().add(4), high[k]);
        sums[k] = fold_lanes(&partial, query, vectors[k], body);
    }
    sums
}

/// Fixed-order fold of the lane sums, then the remainder in index order.
#[inline(always)]
fn fold_lanes(partial: &[f64; DOT_LANES], query: &[f32], vector: &[f32], body: usize) -> f64 {
    let mut sum = ((partial[0] + partial[1]) + (partial[2] + partial[3]))
        + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
    for i in body..query.len() {
        sum += f64::from(query[i]) * f64::from(vector[i]);
    }
    sum
}

#[inline(always)]
fn lane_dot_products<const N: usize>(query: &[f32], vectors: [&[f32]; N]) -> [f64; N] {
    let len = query.len();
    let vectors = vectors.map(|vector| &vector[..len]);
    let body = len - len % DOT_LANES;
    let mut lanes = [[0.0f64; DOT_LANES]; N];
    for (chunk_index, query_chunk) in query[..body].chunks_exact(DOT_LANES).enumerate() {
        let start = chunk_index * DOT_LANES;
        let mut q = [0.0f64; DOT_LANES];
        for (lane, value) in q.iter_mut().zip(query_chunk) {
            *lane = f64::from(*value);
        }
        for k in 0..N {
            let vector_chunk = &vectors[k][start..start + DOT_LANES];
            for lane in 0..DOT_LANES {
                lanes[k][lane] += q[lane] * f64::from(vector_chunk[lane]);
            }
        }
    }
    let mut sums = [0.0f64; N];
    for k in 0..N {
        sums[k] = fold_lanes(&lanes[k], query, vectors[k], body);
    }
    sums
}

/// Cosines of `query` with the summary's centroid and exemplars, computed together.
///
/// Returns the same value as `centroid.max(specter_exemplar_score(..))` did in
/// `hybrid_centroid_score`: each cosine follows `cosine_sim_with_norms`
/// (NaN on a length mismatch, 0.0 for a zero norm), and the maxima fold in the
/// same order. Retrieval scores every component for every query, so this loop
/// is the largest CPU cost of incremental planning.
fn fused_specter_score(query: &RetrievalQueryData, summary: &RetrievalSummaryData) -> f64 {
    let (Some(query_specter), Some(query_norm)) = (query.specter.as_ref(), query.specter_norm)
    else {
        return 0.0;
    };
    let query_specter = query_specter.as_slice();
    let centroid = match (
        summary.specter_centroid.as_ref(),
        summary.specter_centroid_norm,
    ) {
        (Some(vector), Some(norm)) => Some((vector.as_slice(), norm)),
        _ => None,
    };
    let centroid_slots = usize::from(centroid.is_some());
    if summary.exemplar_vectors.len() + centroid_slots > 5 {
        // More vectors than the fused kernel holds: use the reference path.
        let centroid_score = centroid
            .map(|(vector, norm)| cosine_sim_with_norms(query_specter, query_norm, vector, norm))
            .unwrap_or(0.0);
        return centroid_score.max(specter_exemplar_score(query, summary));
    }
    // Slot 0 is the centroid when present; the exemplars follow in order.
    let mut vectors: [&[f32]; 5] = [&[]; 5];
    let mut norms = [0.0f64; 5];
    let mut count = 0;
    if let Some((vector, norm)) = centroid {
        vectors[0] = vector;
        norms[0] = norm;
        count = 1;
    }
    let first_exemplar = count;
    for (vector, norm) in summary
        .exemplar_vectors
        .iter()
        .zip(summary.exemplar_norms.iter())
    {
        vectors[count] = vector.as_slice();
        norms[count] = *norm;
        count += 1;
    }
    let len = query_specter.len();
    // Equal-length vectors are summed in the fused pass; others become NaN below.
    let mut fused_slots = [0usize; 5];
    let mut fused_vectors: [&[f32]; 5] = [&[]; 5];
    let mut fused_count = 0;
    for slot in 0..count {
        if vectors[slot].len() == len {
            fused_slots[fused_count] = slot;
            fused_vectors[fused_count] = vectors[slot];
            fused_count += 1;
        }
    }
    let v = fused_vectors;
    let sums: [f64; 5] = match fused_count {
        0 => [0.0; 5],
        1 => pad(fused_dot_products(query_specter, [v[0]])),
        2 => pad(fused_dot_products(query_specter, [v[0], v[1]])),
        3 => pad(fused_dot_products(query_specter, [v[0], v[1], v[2]])),
        4 => pad(fused_dot_products(query_specter, [v[0], v[1], v[2], v[3]])),
        _ => fused_dot_products(query_specter, [v[0], v[1], v[2], v[3], v[4]]),
    };
    let mut dots = [0.0f64; 5];
    for index in 0..fused_count {
        dots[fused_slots[index]] = sums[index];
    }
    let cosine = |slot: usize| -> f64 {
        if vectors[slot].len() != len {
            f64::NAN
        } else if query_norm == 0.0 || norms[slot] == 0.0 {
            0.0
        } else {
            dots[slot] / (query_norm * norms[slot])
        }
    };
    let centroid_score = if centroid.is_some() { cosine(0) } else { 0.0 };
    let exemplar_score = (first_exemplar..count).map(cosine).fold(0.0f64, f64::max);
    centroid_score.max(exemplar_score)
}

fn pad<const N: usize>(values: [f64; N]) -> [f64; 5] {
    let mut padded = [0.0f64; 5];
    padded[..N].copy_from_slice(&values);
    padded
}

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

/// Hybrid centroid score of one summary, looked up directly in its counters.
pub(crate) fn score_hybrid_centroid_query(
    query: &RetrievalQueryData,
    summary: &RetrievalSummaryData,
    coauthor_cluster_df: &HashMap<u64, usize>,
    non_mega_coauthor_cluster_df: &HashMap<u64, usize>,
    affiliation_cluster_df: &HashMap<u64, usize>,
    total_summary_count: usize,
) -> f32 {
    let prepared = PreparedQuery::new(
        query,
        coauthor_cluster_df,
        non_mega_coauthor_cluster_df,
        affiliation_cluster_df,
        total_summary_count,
    );
    let overlap = component_overlap(query, &prepared, summary);
    let (coauthor_score, affiliation_score) = overlap_scores(query, &prepared, summary, &overlap);
    hybrid_centroid_score(
        query,
        summary,
        coauthor_score,
        affiliation_score,
        fused_specter_score(query, summary),
    )
}

/// Weighted sum of the hybrid score terms, given the two overlap scores.
fn hybrid_centroid_score(
    query: &RetrievalQueryData,
    summary: &RetrievalSummaryData,
    coauthor_score: f64,
    affiliation_score: f64,
    specter_score: f64,
) -> f32 {
    let weights = if query.has_full_first {
        DEFAULT_HYBRID_CENTROID_WEIGHTS
    } else {
        DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS
    };
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

#[pymethods]
impl RustHybridCentroidRetriever {
    #[new]
    #[pyo3(signature = (summaries, include_exemplars = false))]
    fn new(summaries: &Bound<'_, PyAny>, include_exemplars: bool) -> PyResult<Self> {
        let mut packed_summaries = Vec::new();
        for item in PyIterator::from_object(summaries)? {
            packed_summaries.push(extract_retrieval_summary(&item?, include_exemplars)?);
        }
        Self::from_summaries(packed_summaries, true).map_err(retrieval_error_to_py)
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

#[cfg(test)]
mod overlap_index_tests {
    //! Indexed and prepared overlap scoring against verbatim copies of the
    //! per-component scoring they replaced.

    use super::*;
    use crate::retrieval_overlap::rescues_mega_coauthors;

    // Verbatim copies of the replaced scoring (2121a38), renamed.
    fn reference_weighted_counter_query_overlap(
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

    fn reference_query_counter_overlap_count(
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

    fn reference_should_rescue_candidate_mega_coauthors(
        query: &RetrievalQueryData,
        summary: &RetrievalSummaryData,
    ) -> bool {
        if summary.max_paper_author_count < RETRIEVAL_MEGA_AUTHOR_THRESHOLD
            || query.coauthor_terms.is_empty()
        {
            return false;
        }

        let full_overlap =
            reference_query_counter_overlap_count(&query.coauthor_terms, &summary.coauthor_counts);
        if full_overlap < 3 {
            return false;
        }
        let filtered_overlap = reference_query_counter_overlap_count(
            &query.coauthor_terms,
            &summary.non_mega_coauthor_counts,
        );
        if full_overlap <= filtered_overlap {
            return false;
        }

        (full_overlap as f64) / (query.coauthor_terms.len() as f64) >= 0.995
    }

    fn reference_score_hybrid_centroid_query(
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
            && !reference_should_rescue_candidate_mega_coauthors(query, summary);
        let (coauthor_counts, coauthor_df) = if use_non_mega_coauthor_counter {
            (
                &summary.non_mega_coauthor_counts,
                non_mega_coauthor_cluster_df,
            )
        } else {
            (&summary.coauthor_counts, coauthor_cluster_df)
        };
        let coauthor_score = reference_weighted_counter_query_overlap(
            &query.coauthor_terms,
            coauthor_counts,
            summary.size,
            coauthor_df,
            total_summary_count,
            COAUTHOR_OVERLAP_CONFIG,
        );
        let affiliation_score = reference_weighted_counter_query_overlap(
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

    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, bound: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % bound
        }
    }

    fn counter(mut entries: Vec<(u64, f32)>) -> Option<CounterData> {
        entries.sort_unstable_by_key(|entry| entry.0);
        entries.dedup_by_key(|entry| entry.0);
        let sum = entries.iter().map(|entry| entry.1).sum();
        Some(CounterData { entries, sum })
    }

    fn random_counter(
        rng: &mut Lcg,
        universe: u64,
        max_terms: u64,
        size: usize,
    ) -> Option<CounterData> {
        match rng.below(10) {
            0 => None,
            1 => Some(CounterData {
                entries: Vec::new(),
                sum: 0.0,
            }),
            _ => counter(
                (0..1 + rng.below(max_terms))
                    .map(|_| {
                        (
                            rng.below(universe),
                            1.0 + rng.below(size.max(1) as u64) as f32,
                        )
                    })
                    .collect(),
            ),
        }
    }

    fn vector(rng: &mut Lcg) -> Vec<f32> {
        (0..8)
            .map(|_| rng.below(1000) as f32 / 500.0 - 1.0)
            .collect()
    }

    fn summary(rng: &mut Lcg, key: String) -> RetrievalSummaryData {
        let size = [0usize, 1, 2, 3, 7, 40][rng.below(6) as usize];
        let coauthor_counts = random_counter(rng, 300, 120, size);
        // Non-mega counters keep a subset of the full counter's terms, as built.
        let non_mega_coauthor_counts = coauthor_counts.as_ref().and_then(|full| {
            let kept: Vec<(u64, f32)> = full
                .entries
                .iter()
                .copied()
                .filter(|_| rng.below(3) != 0)
                .collect();
            if kept.is_empty() {
                None
            } else {
                counter(kept)
            }
        });
        let specter_centroid = (rng.below(3) != 0).then(|| vector(rng));
        let exemplar_vectors: Vec<Vec<f32>> = (0..rng.below(3)).map(|_| vector(rng)).collect();
        RetrievalSummaryData {
            component_key: key,
            size,
            first_name_counts: vec![("anna".to_string(), 1), ("jian".to_string(), 2)],
            middle_initial_counts: random_counter(rng, 4, 2, size),
            coauthor_counts,
            non_mega_coauthor_counts,
            affiliation_counts: random_counter(rng, 60, 20, size),
            venue_counts: None,
            title_counts: None,
            max_paper_author_count: [3usize, 49, 50, 3000][rng.below(4) as usize],
            year_min: None,
            year_max: None,
            year_mean: None,
            orcid_hashes: Vec::new(),
            specter_centroid_norm: specter_centroid
                .as_ref()
                .map(|values| vector_norm_f32(values)),
            specter_centroid,
            exemplar_norms: exemplar_vectors
                .iter()
                .map(|values| vector_norm_f32(values))
                .collect(),
            exemplar_vectors,
        }
    }

    fn terms(hashes: impl IntoIterator<Item = u64>, rng: &mut Lcg) -> Vec<RetrievalQueryTerm> {
        hashes
            .into_iter()
            .map(|hash| RetrievalQueryTerm {
                hash,
                // A token count of 0 has weight 0, which the numerator skips.
                token_count: [0u8, 1, 1, 2][rng.below(4) as usize],
            })
            .collect()
    }

    fn query(rng: &mut Lcg, coauthor_terms: Vec<RetrievalQueryTerm>) -> RetrievalQueryData {
        let affiliation_terms = terms(
            (0..rng.below(8)).map(|_| rng.below(60)).collect::<Vec<_>>(),
            rng,
        );
        let specter = (rng.below(4) != 0).then(|| vector(rng));
        RetrievalQueryData {
            first: ["anna", "a", "jian"][rng.below(3) as usize].to_string(),
            has_full_first: rng.below(2) == 0,
            middle_initial_hashes: Vec::new(),
            coauthor_hashes: coauthor_terms.iter().map(|term| term.hash).collect(),
            coauthor_terms,
            affiliation_hashes: affiliation_terms.iter().map(|term| term.hash).collect(),
            affiliation_terms,
            venue_hashes: Vec::new(),
            title_hashes: Vec::new(),
            year: None,
            orcid_hash: None,
            specter_norm: specter.as_ref().map(|values| vector_norm_f32(values)),
            specter: specter.map(Arc::new),
        }
    }

    /// Random summaries, identical twins whose scores tie and are ordered by
    /// key, and rescue-rule edge cases for a 200-term query over hashes
    /// 1000..1200.
    fn synthetic_summaries() -> Vec<RetrievalSummaryData> {
        let mut rng = Lcg(5);
        let mut summaries: Vec<RetrievalSummaryData> = (0..300)
            .map(|index| summary(&mut rng, format!("c{index:03}")))
            .collect();
        for index in 0..20 {
            let mut twin = summaries[index].clone();
            twin.component_key = format!("b{index:03}");
            summaries.push(twin);
        }
        // (full, non-mega) hits of the 200-term query: 199/0 is a ratio of
        // exactly 0.995, 198/0 is below it, 2/0 has too few hits, 199/199 gains
        // nothing over the non-mega counter.
        for (label, full, filtered) in [
            ("r199", 199u64, 0u64),
            ("r198", 198, 0),
            ("r002", 2, 0),
            ("r199eq", 199, 199),
        ] {
            let mut rescue = summary(&mut rng, label.to_string());
            rescue.size = 5;
            rescue.max_paper_author_count = 1000;
            rescue.coauthor_counts = counter(
                (1000..1000 + full)
                    .map(|hash| (hash, 2.0))
                    .chain([(7, 1.0)])
                    .collect(),
            );
            rescue.non_mega_coauthor_counts = counter(
                (1000..1000 + filtered)
                    .map(|hash| (hash, 1.0))
                    .chain([(7, 1.0)])
                    .collect(),
            );
            summaries.push(rescue);
        }
        summaries
    }

    fn synthetic_queries() -> Vec<RetrievalQueryData> {
        let mut rng = Lcg(17);
        let mut queries = Vec::new();
        for count in [0u64, 1, 3, 6, 40, 250] {
            for _ in 0..4 {
                let mut hashes: Vec<u64> = (0..count).map(|_| rng.below(300)).collect();
                hashes.sort_unstable();
                hashes.dedup();
                let coauthor_terms = terms(hashes, &mut rng);
                queries.push(query(&mut rng, coauthor_terms));
            }
        }
        // Unsorted query terms with a repeated hash.
        let repeated = terms([9, 3, 9, 250, 1], &mut rng);
        queries.push(query(&mut rng, repeated));
        let unit_terms = |hashes: std::ops::Range<u64>| -> Vec<RetrievalQueryTerm> {
            hashes
                .map(|hash| RetrievalQueryTerm {
                    hash,
                    token_count: 1,
                })
                .collect()
        };
        queries.push(query(&mut rng, unit_terms(1000..1200)));
        queries.push(query(&mut rng, unit_terms(1000..1003)));
        queries
    }

    fn reference_ranking(
        retriever: &RustHybridCentroidRetriever,
        query: &RetrievalQueryData,
    ) -> Vec<(String, u32)> {
        let mut scored: Vec<(String, f32)> = retriever
            .summaries
            .iter()
            .map(|summary| {
                (
                    summary.component_key.clone(),
                    reference_score_hybrid_centroid_query(
                        query,
                        summary,
                        &retriever.coauthor_cluster_df,
                        &retriever.non_mega_coauthor_cluster_df,
                        &retriever.affiliation_cluster_df,
                        retriever.summaries.len(),
                    ),
                )
            })
            .collect();
        scored.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        scored
            .into_iter()
            .map(|(key, score)| (key, score.to_bits()))
            .collect()
    }

    #[test]
    fn direct_and_indexed_scores_match_the_replaced_scoring_bit_for_bit() {
        let retriever = RustHybridCentroidRetriever::from_summaries(synthetic_summaries(), true)
            .expect("retriever");
        let (mut rescued, mut non_mega, mut positive) = (0usize, 0usize, 0usize);
        for query in synthetic_queries() {
            let prepared = retriever.prepare_query(&query);
            let overlaps = retriever.overlap_index().component_overlaps(
                &query,
                &prepared,
                retriever.summaries.len(),
            );
            for (index, summary) in retriever.summaries.iter().enumerate() {
                let reference = reference_score_hybrid_centroid_query(
                    &query,
                    summary,
                    &retriever.coauthor_cluster_df,
                    &retriever.non_mega_coauthor_cluster_df,
                    &retriever.affiliation_cluster_df,
                    retriever.summaries.len(),
                );
                let direct = score_hybrid_centroid_query(
                    &query,
                    summary,
                    &retriever.coauthor_cluster_df,
                    &retriever.non_mega_coauthor_cluster_df,
                    &retriever.affiliation_cluster_df,
                    retriever.summaries.len(),
                );
                let indexed = retriever
                    .score_candidate(&query, &prepared, &overlaps, index, None, None, None);
                assert_eq!(
                    direct.to_bits(),
                    reference.to_bits(),
                    "{}",
                    summary.component_key
                );
                assert_eq!(
                    indexed.to_bits(),
                    reference.to_bits(),
                    "{}",
                    summary.component_key
                );

                // The f64 overlap scores match too, before the final f32 rounding.
                let mega = summary.max_paper_author_count >= RETRIEVAL_MEGA_AUTHOR_THRESHOLD;
                let rescue = reference_should_rescue_candidate_mega_coauthors(&query, summary);
                let (coauthor_counts, coauthor_df) = if mega && !rescue {
                    (
                        &summary.non_mega_coauthor_counts,
                        &retriever.non_mega_coauthor_cluster_df,
                    )
                } else {
                    (&summary.coauthor_counts, &retriever.coauthor_cluster_df)
                };
                let reference_scores = (
                    reference_weighted_counter_query_overlap(
                        &query.coauthor_terms,
                        coauthor_counts,
                        summary.size,
                        coauthor_df,
                        retriever.summaries.len(),
                        COAUTHOR_OVERLAP_CONFIG,
                    )
                    .to_bits(),
                    reference_weighted_counter_query_overlap(
                        &query.affiliation_terms,
                        &summary.affiliation_counts,
                        summary.size,
                        &retriever.affiliation_cluster_df,
                        retriever.summaries.len(),
                        AFFILIATION_OVERLAP_CONFIG,
                    )
                    .to_bits(),
                );
                let direct_overlap = component_overlap(&query, &prepared, summary);
                let direct_scores = overlap_scores(&query, &prepared, summary, &direct_overlap);
                let indexed_scores = overlap_scores(&query, &prepared, summary, &overlaps[index]);
                assert_eq!(
                    (direct_scores.0.to_bits(), direct_scores.1.to_bits()),
                    reference_scores
                );
                assert_eq!(
                    (indexed_scores.0.to_bits(), indexed_scores.1.to_bits()),
                    reference_scores
                );
                rescued += usize::from(rescue);
                non_mega += usize::from(mega && !rescue);
                positive += usize::from(direct_scores.0 > 0.0);
            }
        }
        assert!(
            rescued >= 2 && non_mega > 100 && positive > 100,
            "{rescued} {non_mega} {positive}"
        );
    }

    #[test]
    fn indexed_top_k_keeps_the_replaced_ranking_and_tie_order() {
        let retriever = RustHybridCentroidRetriever::from_summaries(synthetic_summaries(), true)
            .expect("retriever");
        let members: HashMap<String, Vec<u32>> = retriever
            .summaries
            .iter()
            .map(|summary| (summary.component_key.clone(), Vec::new()))
            .collect();
        let mut ties = 0;
        for query in synthetic_queries() {
            let expected = reference_ranking(&retriever, &query);
            ties += expected
                .windows(2)
                .filter(|pair| pair[0].1 == pair[1].1)
                .count();
            for top_k in [1usize, 25, expected.len()] {
                let result = retriever
                    .build_pair_plan_query_result(&query, "anna", 0, None, &members, top_k, None)
                    .expect("plan row");
                let observed: Vec<(String, u32)> = result
                    .row_component_keys
                    .into_iter()
                    .zip(
                        result
                            .row_retrieval_scores
                            .iter()
                            .map(|score| score.to_bits()),
                    )
                    .collect();
                assert_eq!(observed, expected[..top_k.min(expected.len())]);
            }
        }
        assert!(ties > 20, "{ties}");
    }

    #[test]
    fn rescue_rule_thresholds() {
        assert!(rescues_mega_coauthors(200, 199, 0));
        assert!(!rescues_mega_coauthors(200, 198, 0));
        assert!(!rescues_mega_coauthors(2, 2, 0));
        assert!(rescues_mega_coauthors(3, 3, 0));
        assert!(!rescues_mega_coauthors(200, 199, 199));
    }

    #[test]
    fn counters_with_repeated_hashes_are_rejected() {
        let mut rng = Lcg(1);
        let mut bad = summary(&mut rng, "bad".to_string());
        bad.affiliation_counts = Some(CounterData {
            entries: vec![(4, 1.0), (4, 2.0)],
            sum: 3.0,
        });
        assert!(RustHybridCentroidRetriever::from_summaries(vec![bad.clone()], true).is_err());
        assert!(RustHybridCentroidRetriever::from_summaries(vec![bad], false).is_ok());
    }

    /// The fused centroid and exemplar cosines must match the one-at-a-time path
    /// to rounding, and exactly for zero norms, length mismatches, and long
    /// exemplar lists (which keep the reference path).
    #[test]
    fn fused_specter_score_matches_reference_to_rounding() {
        let mut rng = Lcg(11);
        for case in 0..4000 {
            let mut summary = summary(&mut rng, format!("c{case}"));
            let mut query = query(&mut rng, Vec::new());
            // Half the cases use long vectors: several lane chunks plus a remainder.
            let length = [8usize, 13, 64, 768][if case % 2 == 0 {
                0
            } else {
                1 + rng.below(3) as usize
            }];
            let long_vector = |rng: &mut Lcg| -> Vec<f32> {
                (0..length)
                    .map(|_| rng.below(1000) as f32 / 500.0 - 1.0)
                    .collect()
            };
            if length != 8 {
                let values = long_vector(&mut rng);
                query.specter_norm = Some(vector_norm_f32(&values));
                query.specter = Some(std::sync::Arc::new(values));
                if summary.specter_centroid.is_some() {
                    let centroid = long_vector(&mut rng);
                    summary.specter_centroid_norm = Some(vector_norm_f32(&centroid));
                    summary.specter_centroid = Some(centroid);
                }
            }
            let exemplar_count = rng.below(7) as usize;
            summary.exemplar_vectors = (0..exemplar_count).map(|_| long_vector(&mut rng)).collect();
            if rng.below(10) == 0 && exemplar_count > 0 {
                // A length mismatch must give the same NaN handling as the reference.
                summary.exemplar_vectors[0].push(0.5);
            }
            summary.exemplar_norms = summary
                .exemplar_vectors
                .iter()
                .map(|values| {
                    if rng.below(15) == 0 {
                        0.0
                    } else {
                        vector_norm_f32(values)
                    }
                })
                .collect();
            if rng.below(20) == 0 {
                query.specter_norm = query.specter_norm.map(|_| 0.0);
            }
            let reference = match (
                query.specter.as_ref(),
                query.specter_norm,
                summary.specter_centroid.as_ref(),
                summary.specter_centroid_norm,
            ) {
                (Some(q), Some(qn), Some(c), Some(cn)) => cosine_sim_with_norms(q, qn, c, cn),
                _ => 0.0,
            }
            .max(specter_exemplar_score(&query, &summary));
            let fused = fused_specter_score(&query, &summary);
            if reference.is_nan() || reference == 0.0 {
                assert_eq!(fused.to_bits(), reference.to_bits(), "case {case}");
            } else {
                let tolerance = 1e-12 * reference.abs().max(1e-3);
                assert!(
                    (fused - reference).abs() <= tolerance,
                    "case {case}: {fused} vs {reference}"
                );
            }
        }
    }

    /// The AVX2 kernel must equal the portable one bit for bit, so scores do not
    /// depend on the CPU.
    #[test]
    fn avx2_dot_products_match_portable_bits() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return;
            }
            let mut rng = Lcg(23);
            for case in 0..500 {
                let length = [1usize, 7, 8, 9, 64, 767, 768, 1031][case % 8];
                let random = |rng: &mut Lcg| -> Vec<f32> {
                    (0..length)
                        .map(|_| rng.below(20001) as f32 / 1000.0 - 10.0)
                        .collect()
                };
                let query = random(&mut rng);
                let vectors: Vec<Vec<f32>> = (0..5).map(|_| random(&mut rng)).collect();
                let slices = [
                    vectors[0].as_slice(),
                    vectors[1].as_slice(),
                    vectors[2].as_slice(),
                    vectors[3].as_slice(),
                    vectors[4].as_slice(),
                ];
                let portable = lane_dot_products(&query, slices);
                // SAFETY: AVX2 was detected above.
                let avx2 = unsafe { fused_dot_products_avx2(&query, slices) };
                for k in 0..5 {
                    assert_eq!(
                        portable[k].to_bits(),
                        avx2[k].to_bits(),
                        "case {case} vector {k}"
                    );
                }
            }
        }
    }

    /// Lane-summed SPECTER scoring versus the sequential reference, 768-dim vectors.
    /// Run with `--ignored --nocapture` in a release build.
    #[test]
    #[ignore = "benchmark"]
    fn fused_specter_score_timing() {
        let mut rng = Lcg(5);
        let dim = 768;
        let random_vector = |rng: &mut Lcg| -> Vec<f32> {
            (0..dim)
                .map(|_| rng.below(2001) as f32 / 1000.0 - 1.0)
                .collect()
        };
        let query_vector = random_vector(&mut rng);
        let query = RetrievalQueryData {
            specter_norm: Some(vector_norm_f32(&query_vector)),
            specter: Some(std::sync::Arc::new(query_vector)),
            ..query(&mut rng, Vec::new())
        };
        let component_count: usize = std::env::var("BENCH_COMPONENTS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(2000);
        let summaries: Vec<RetrievalSummaryData> = (0..component_count)
            .map(|index| {
                let mut summary = summary(&mut rng, format!("c{index}"));
                let centroid = random_vector(&mut rng);
                summary.specter_centroid_norm = Some(vector_norm_f32(&centroid));
                summary.specter_centroid = Some(centroid);
                summary.exemplar_vectors = (0..4).map(|_| random_vector(&mut rng)).collect();
                summary.exemplar_norms = summary
                    .exemplar_vectors
                    .iter()
                    .map(|values| vector_norm_f32(values))
                    .collect();
                summary
            })
            .collect();
        let reference = |summary: &RetrievalSummaryData| -> f64 {
            cosine_sim_with_norms(
                query.specter.as_ref().unwrap(),
                query.specter_norm.unwrap(),
                summary.specter_centroid.as_ref().unwrap(),
                summary.specter_centroid_norm.unwrap(),
            )
            .max(specter_exemplar_score(&query, summary))
        };
        for (label, fast) in [
            ("reference", false),
            ("lanes", true),
            ("reference", false),
            ("lanes", true),
        ] {
            let start = std::time::Instant::now();
            let mut total = 0.0;
            let repeats = 40_000 / summaries.len();
            for _ in 0..repeats {
                for summary in &summaries {
                    total += if fast {
                        fused_specter_score(&query, summary)
                    } else {
                        reference(summary)
                    };
                }
            }
            let per_score =
                start.elapsed().as_secs_f64() * 1e9 / ((repeats * summaries.len()) as f64);
            println!("{label}: {per_score:.0} ns per score (checksum {total:.6})");
        }
    }
}
