use cld2::{detect_language_ext as cld2_detect_language_ext, Format as Cld2Format};
use fasttext::FastText;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule, PyTuple};
use pyo3::Bound;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

fn py_len(s: &str) -> usize {
    // Python len() semantics for str: count of Unicode scalar values
    s.chars().count()
}

const DROPPED_AFFIXES: [&str; 48] = [
    "ab", "am", "ap", "abu", "al", "auf", "aus", "bar", "bath", "bat", "bet", "bint", "dall",
    "dalla", "das", "de", "degli", "del", "dell", "della", "dem", "den", "der", "di", "do", "dos",
    "ds", "du", "el", "ibn", "im", "jr", "la", "las", "le", "los", "mac", "mc", "mhic", "mic",
    "ter", "und", "van", "vom", "von", "zu", "zum", "zur",
];

fn is_dropped_affix(token: &str) -> bool {
    DROPPED_AFFIXES.contains(&token)
}

#[derive(Clone, Serialize, Deserialize)]
struct CounterData {
    // FNV-1a 64-bit hashes of original string keys, sorted ascending.
    // Values are f32 counts. Binary search replaces HashMap lookup.
    // Trade-off: collisions are possible but rare at expected key counts.
    // If they occur, colliding keys merge silently in this representation.
    entries: Vec<(u64, f32)>,
    sum: f32,
}

#[inline(always)]
fn fnv64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[derive(Clone, Serialize, Deserialize)]
struct NameCountsData {
    first: f64,
    first_last: f64,
    last: f64,
    last_first_initial: f64,
}

#[derive(Default)]
struct RawNameCountMaps {
    first: HashMap<String, f64>,
    last: HashMap<String, f64>,
    first_last: HashMap<String, f64>,
    last_first_initial: HashMap<String, f64>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
enum ClusterId {
    Int(i64),
    Str(String),
}

type PaperId = String;

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
    paper_id: PaperId,
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
    references: HashSet<PaperId>,
    year: Option<i64>,
    has_abstract: bool,
    predicted_language: Option<String>,
    is_reliable: bool,
    journal_ngrams: Option<CounterData>,
    specter: Option<Vec<f32>>,
    #[serde(default)]
    specter_norm: Option<f64>,
}

#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
struct RustFeaturizer {
    signatures: HashMap<String, SignatureData>,
    #[serde(default)]
    signature_ids: Vec<String>,
    papers: HashMap<PaperId, PaperData>,
    name_tuples: HashMap<String, HashSet<String>>,
    cluster_seeds_disallow: HashSet<(String, String)>,
    cluster_seeds_require: HashMap<String, ClusterId>,
    compute_reference_features: bool,
    cluster_seed_require_value: f64,
    cluster_seed_disallow_value: f64,
    #[serde(skip)]
    cached_signature_id_order: OnceLock<Vec<String>>,
    #[serde(skip)]
    cluster_seeds_disallow_index: OnceLock<HashMap<String, HashSet<String>>>,
}

#[derive(Clone)]
struct RetrievalSummaryData {
    component_key: String,
    size: usize,
    first_name_counts: Vec<(String, f32)>,
    middle_initial_counts: Option<CounterData>,
    coauthor_counts: Option<CounterData>,
    affiliation_counts: Option<CounterData>,
    venue_counts: Option<CounterData>,
    title_counts: Option<CounterData>,
    year_min: Option<i64>,
    year_max: Option<i64>,
    year_mean: Option<f64>,
    orcid_hashes: Vec<u64>,
    specter_centroid: Option<Vec<f32>>,
    specter_centroid_norm: Option<f64>,
    exemplar_vectors: Vec<Vec<f32>>,
    exemplar_norms: Vec<f64>,
}

#[derive(Clone, Copy)]
struct RetrievalQueryTerm {
    hash: u64,
    token_count: u8,
}

struct RetrievalQueryData {
    first: String,
    has_full_first: bool,
    middle_initial_hashes: Vec<u64>,
    coauthor_hashes: Vec<u64>,
    coauthor_terms: Vec<RetrievalQueryTerm>,
    affiliation_hashes: Vec<u64>,
    affiliation_terms: Vec<RetrievalQueryTerm>,
    venue_hashes: Vec<u64>,
    title_hashes: Vec<u64>,
    year: Option<i64>,
    orcid_hash: Option<u64>,
    specter: Option<Vec<f32>>,
    specter_norm: Option<f64>,
}

#[derive(Clone, Copy)]
struct RetrievalHybridWeights {
    centroid: f64,
    coauthor: f64,
    affiliation: f64,
    middle: f64,
    first_name: f64,
}

const RETRIEVAL_FEATURE_ORDER: [&str; 5] = [
    "centroid",
    "coauthor",
    "affiliation",
    "middle",
    "first_name",
];
const DEFAULT_HYBRID_CENTROID_POLICY_NAME: &str = "h_wang_any_input_v2";
const DEFAULT_HYBRID_CENTROID_WEIGHTS: [f64; 5] =
    [0.527232, 0.223412, 0.146909, 0.009439, 0.093007];
const DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS: [f64; 5] =
    [0.520012, 0.220264, 0.109278, 0.150447, 0.0];
const DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS: [f64; 5] = [0.40, 0.23, 0.12, 0.05, 0.07];
const RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE: f64 = -0.25;
const RETRIEVAL_YEAR_SCORE_DECAY_YEARS: f64 = 15.0;
const RETRIEVAL_YEAR_SCORE_RANGE_GAP: i64 = 10;
const RETRIEVAL_YEAR_SCORE_RANGE_PENALTY: f64 = 0.15;
const RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP: i64 = 35;

impl RetrievalHybridWeights {
    fn from_array(weights: [f64; 5]) -> Self {
        Self {
            centroid: weights[0],
            coauthor: weights[1],
            affiliation: weights[2],
            middle: weights[3],
            first_name: weights[4],
        }
    }
}

#[derive(Clone, Copy)]
enum RetrievalFirstNameMode {
    Prefix,
    ExactOnly,
    ExactThenPrefixHalf,
    PrefixLengthRatio,
    ExactThenPrefixLengthRatio,
}

#[derive(Clone, Copy)]
enum RetrievalSpecterMode {
    Centroid,
    ExemplarMax,
    CentroidExemplar50_50,
    CentroidExemplar25_75,
    CentroidExemplar75_25,
    MaxOfCentroidExemplar,
}

#[derive(Clone, Copy)]
struct RetrievalOverlapConfig {
    use_idf: bool,
    per_term_cap: Option<f64>,
    total_cap: Option<f64>,
    min_token_count: u8,
    unigram_weight: f64,
    multi_token_weight: f64,
}

#[derive(Clone, Copy)]
struct RetrievalExperimentalConfig {
    first_name_mode: RetrievalFirstNameMode,
    specter_mode: RetrievalSpecterMode,
    coauthor: RetrievalOverlapConfig,
    affiliation: RetrievalOverlapConfig,
}

#[pyclass]
struct RustHybridCentroidRetriever {
    summaries: Vec<RetrievalSummaryData>,
    max_block_component_size: usize,
    component_index_by_key: HashMap<String, usize>,
    coauthor_cluster_df: HashMap<u64, usize>,
    affiliation_cluster_df: HashMap<u64, usize>,
}

#[pyclass]
struct RustNameCompatibleSubblockSelector {
    signature_to_subblock: HashMap<String, String>,
    subblock_to_components: HashMap<String, Vec<String>>,
    subblock_tokens_by_subblock: HashMap<String, Vec<String>>,
    name_tuples: HashMap<String, HashSet<String>>,
}

#[derive(Default)]
struct RetrievalPairPlanQueryResult {
    row_query_signature_indices: Vec<u32>,
    row_component_keys: Vec<String>,
    row_retrieval_scores: Vec<f32>,
    row_retrieval_ranks: Vec<u16>,
    row_component_sizes: Vec<u32>,
    row_named_signature_counts: Vec<u32>,
    row_dominant_first_names: Vec<String>,
    row_candidate_year_min: Vec<i32>,
    row_candidate_year_max: Vec<i32>,
    row_candidate_year_range_missing: Vec<u8>,
    row_query_first_tokens: Vec<String>,
    row_query_years: Vec<i32>,
    row_query_year_missing: Vec<u8>,
    row_query_has_affiliations: Vec<u8>,
    row_query_has_coauthors: Vec<u8>,
    row_middle_initial_compatibility: Vec<f32>,
    row_affiliation_overlap: Vec<f32>,
    row_coauthor_overlap: Vec<f32>,
    row_venue_overlap: Vec<f32>,
    row_year_compatibility: Vec<f32>,
    row_title_overlap: Vec<f32>,
    row_specter_centroid_similarity: Vec<f32>,
    row_specter_exemplar_similarity: Vec<f32>,
    right_signature_indices_by_row: Vec<Vec<u32>>,
}

#[derive(Clone, Copy)]
struct PairAggregateRowRange {
    row_offset: usize,
    start: usize,
    stop: usize,
}

struct PairAggregateBuffers {
    counts: Vec<u32>,
    sums: Vec<f64>,
    mins: Vec<f64>,
    maxs: Vec<f64>,
}

impl RustHybridCentroidRetriever {
    fn default_hybrid_weights_for_query(query_data: &RetrievalQueryData) -> RetrievalHybridWeights {
        if query_data.has_full_first {
            RetrievalHybridWeights::from_array(DEFAULT_HYBRID_CENTROID_WEIGHTS)
        } else {
            RetrievalHybridWeights::from_array(DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS)
        }
    }

    fn default_experimental_config_for_query(
        query_data: &RetrievalQueryData,
    ) -> RetrievalExperimentalConfig {
        RetrievalExperimentalConfig {
            first_name_mode: if query_data.has_full_first {
                RetrievalFirstNameMode::ExactThenPrefixHalf
            } else {
                RetrievalFirstNameMode::Prefix
            },
            specter_mode: RetrievalSpecterMode::MaxOfCentroidExemplar,
            coauthor: RetrievalOverlapConfig {
                use_idf: true,
                per_term_cap: Some(0.35),
                ..default_overlap_config()
            },
            affiliation: RetrievalOverlapConfig {
                use_idf: true,
                ..default_overlap_config()
            },
        }
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
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
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
        max_block_component_size: usize,
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
    ) -> Vec<(usize, f32)> {
        let weights = Self::default_hybrid_weights_for_query(query_data);
        let config = Self::default_experimental_config_for_query(query_data);
        let mut scored = candidate_indices
            .iter()
            .map(|idx| {
                let summary =
                    self.summary_for_candidate_index(*idx, override_index, override_summary);
                (
                    *idx,
                    score_experimental_hybrid_centroid_query(
                        query_data,
                        summary,
                        max_block_component_size,
                        weights,
                        config,
                        &self.coauthor_cluster_df,
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
    ) -> Vec<usize> {
        if let Some(orcid_hash) = query_data.orcid_hash {
            let orcid_matches: Vec<usize> = candidate_indices
                .iter()
                .copied()
                .filter(|idx| contains_hashed_value(&self.summaries[*idx].orcid_hashes, orcid_hash))
                .collect();
            if !orcid_matches.is_empty() {
                candidate_indices = orcid_matches;
            }
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

        candidate_indices
    }

    fn candidate_indices_for_pair_plan_query(
        &self,
        query_data: &RetrievalQueryData,
        base_candidate_indices: Option<&[usize]>,
        query_signature_id: Option<&str>,
        selector: Option<&RustNameCompatibleSubblockSelector>,
        global_backfill_count: usize,
    ) -> Vec<usize> {
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
        let candidate_indices = selected.unwrap_or_else(|| {
            base_candidate_indices.map_or_else(
                || (0..self.summaries.len()).collect(),
                |values| values.to_vec(),
            )
        });
        self.hard_filtered_candidate_indices_for_query(query_data, candidate_indices)
    }

    fn build_pair_plan_query_result(
        &self,
        current_query: &RetrievalQueryData,
        query_signature_index: u32,
        base_candidate_indices: Option<&[usize]>,
        query_signature_id: Option<&str>,
        component_member_indices: &HashMap<String, Vec<u32>>,
        top_k: usize,
        selector: Option<&RustNameCompatibleSubblockSelector>,
        global_backfill_count: usize,
    ) -> Result<RetrievalPairPlanQueryResult, String> {
        let candidate_indices = self.candidate_indices_for_pair_plan_query(
            current_query,
            base_candidate_indices,
            query_signature_id,
            selector,
            global_backfill_count,
        );
        if candidate_indices.is_empty() {
            return Ok(RetrievalPairPlanQueryResult::default());
        }
        let scored = self.score_top_k_candidate_indices_default_inner(
            current_query,
            &candidate_indices,
            top_k,
            self.max_block_component_size,
            None,
            None,
        );

        let mut result = RetrievalPairPlanQueryResult::default();
        let query_year_missing = u8::from(current_query.year.is_none());
        let query_year = current_query.year.unwrap_or(i32::MIN as i64) as i32;
        let query_has_affiliations = u8::from(!current_query.affiliation_hashes.is_empty());
        let query_has_coauthors = u8::from(!current_query.coauthor_hashes.is_empty());
        for (rank_offset, (summary_index, score)) in scored.iter().enumerate() {
            let summary = &self.summaries[*summary_index];
            let Some(member_indices) = component_member_indices.get(&summary.component_key) else {
                return Err(format!(
                    "Missing component members for retrieved component_key: {}",
                    summary.component_key
                ));
            };
            let chooser_features = chooser_summary_features(current_query, summary);
            let mut dominant_first_name = "";
            let mut dominant_first_count = 0.0f32;
            let mut named_signature_count = 0.0f32;
            for (first_name, count) in summary.first_name_counts.iter() {
                named_signature_count += *count;
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
                .push((rank_offset + 1).min(u16::MAX as usize) as u16);
            result
                .row_component_sizes
                .push(summary.size.min(u32::MAX as usize) as u32);
            result
                .row_named_signature_counts
                .push(named_signature_count.round().max(0.0) as u32);
            result
                .row_dominant_first_names
                .push(dominant_first_name.to_string());
            result
                .row_candidate_year_min
                .push(summary.year_min.unwrap_or(i32::MIN as i64) as i32);
            result
                .row_candidate_year_max
                .push(summary.year_max.unwrap_or(i32::MIN as i64) as i32);
            result.row_candidate_year_range_missing.push(u8::from(
                summary.year_min.is_none() || summary.year_max.is_none(),
            ));
            result
                .row_query_first_tokens
                .push(current_query.first.clone());
            result.row_query_years.push(query_year);
            result.row_query_year_missing.push(query_year_missing);
            result
                .row_query_has_affiliations
                .push(query_has_affiliations);
            result.row_query_has_coauthors.push(query_has_coauthors);
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

    fn score_top_k_candidate_indices_experimental(
        &self,
        py: Python<'_>,
        query_data: &RetrievalQueryData,
        candidate_indices: &[usize],
        top_k: usize,
        max_block_component_size: usize,
        num_threads: Option<usize>,
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
        weights: RetrievalHybridWeights,
        config: RetrievalExperimentalConfig,
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
                            score_experimental_hybrid_centroid_query(
                                query_data,
                                summary,
                                max_block_component_size,
                                weights,
                                config,
                                &self.coauthor_cluster_df,
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

    fn score_top_k_candidate_indices(
        &self,
        py: Python<'_>,
        query_data: &RetrievalQueryData,
        candidate_indices: &[usize],
        top_k: usize,
        max_block_component_size: usize,
        num_threads: Option<usize>,
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
        weights: RetrievalHybridWeights,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        if candidate_indices.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let scored: Vec<(usize, f32)> = py.allow_threads(|| {
            let compute = || {
                let mut scored = candidate_indices
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
                                max_block_component_size,
                                weights,
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
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        Ok(self.scored_candidates_to_keys_scores(scored, override_index, override_summary))
    }
}

#[derive(Clone, Default)]
struct JsonIngestTelemetry {
    json_parse_seconds: f64,
    paper_preprocess_seconds: f64,
    reference_counter_seconds: f64,
    signature_preprocess_seconds: f64,
    cluster_seed_seconds: f64,
    missing_specter_paper_count: usize,
    defaulted_name_count_signature_count: usize,
    defaulted_name_count_first_count: usize,
    defaulted_name_count_first_last_count: usize,
    defaulted_name_count_last_count: usize,
    defaulted_name_count_last_first_initial_count: usize,
    defaulted_signature_author_position_count: usize,
    defaulted_paper_author_position_count: usize,
}

static LAST_JSON_INGEST_TELEMETRY: OnceLock<Mutex<Option<JsonIngestTelemetry>>> = OnceLock::new();

fn json_ingest_telemetry_slot() -> &'static Mutex<Option<JsonIngestTelemetry>> {
    LAST_JSON_INGEST_TELEMETRY.get_or_init(|| Mutex::new(None))
}

fn set_last_json_ingest_telemetry(telemetry: JsonIngestTelemetry) {
    if let Ok(mut slot) = json_ingest_telemetry_slot().lock() {
        *slot = Some(telemetry);
    }
}

fn reset_last_json_ingest_telemetry_internal() {
    if let Ok(mut slot) = json_ingest_telemetry_slot().lock() {
        *slot = None;
    }
}

static RAYON_POOL_CACHE: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();

fn rayon_pool_cache() -> &'static Mutex<HashMap<usize, Arc<rayon::ThreadPool>>> {
    RAYON_POOL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cached_rayon_pool(thread_count: usize) -> Option<Arc<rayon::ThreadPool>> {
    if thread_count == 0 {
        return None;
    }
    if let Ok(cache) = rayon_pool_cache().lock() {
        if let Some(pool) = cache.get(&thread_count) {
            return Some(Arc::clone(pool));
        }
    }

    let built_pool = ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .ok()?;
    let built_pool = Arc::new(built_pool);
    if let Ok(mut cache) = rayon_pool_cache().lock() {
        let pooled = cache
            .entry(thread_count)
            .or_insert_with(|| Arc::clone(&built_pool));
        return Some(Arc::clone(pooled));
    }
    Some(built_pool)
}

fn install_with_optional_rayon_pool<T, F>(num_threads: Option<usize>, compute: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    if let Some(thread_count) = num_threads {
        let threads = thread_count.max(1);
        if let Some(pool) = cached_rayon_pool(threads) {
            return pool.install(compute);
        }
    }
    compute()
}

fn upper_triangle_total_pairs(block_size: usize) -> usize {
    block_size.saturating_mul(block_size.saturating_sub(1)) / 2
}

fn upper_triangle_pairs_for_range(
    block_size: usize,
    start_offset: usize,
    max_pairs: Option<usize>,
) -> PyResult<Vec<(usize, usize)>> {
    let total_pairs = upper_triangle_total_pairs(block_size);
    if start_offset > total_pairs {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "start_offset out of range: start_offset={} total_pairs={}",
            start_offset, total_pairs
        )));
    }
    let remaining = total_pairs.saturating_sub(start_offset);
    let pair_count = max_pairs.unwrap_or(remaining).min(remaining);
    if pair_count == 0 {
        return Ok(Vec::new());
    }

    let mut local_i = 0usize;
    let mut offset_in_row = start_offset;
    while local_i + 1 < block_size {
        let row_pairs = block_size - local_i - 1;
        if offset_in_row < row_pairs {
            break;
        }
        offset_in_row -= row_pairs;
        local_i += 1;
    }
    if local_i + 1 >= block_size {
        return Ok(Vec::new());
    }
    let mut local_j = local_i + 1 + offset_in_row;
    let mut pairs = Vec::with_capacity(pair_count);
    for _ in 0..pair_count {
        pairs.push((local_i, local_j));
        local_j += 1;
        if local_j >= block_size {
            local_i += 1;
            local_j = local_i + 1;
        }
    }
    Ok(pairs)
}

#[pyfunction]
fn get_last_json_ingest_telemetry(py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
    let slot = json_ingest_telemetry_slot().lock().map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err("json ingest telemetry lock poisoned")
    })?;
    let Some(telemetry) = slot.as_ref() else {
        return Ok(None);
    };

    let stage_seconds = PyDict::new(py);
    stage_seconds.set_item("json_parse_seconds", telemetry.json_parse_seconds)?;
    stage_seconds.set_item(
        "paper_preprocess_seconds",
        telemetry.paper_preprocess_seconds,
    )?;
    stage_seconds.set_item(
        "reference_counter_seconds",
        telemetry.reference_counter_seconds,
    )?;
    stage_seconds.set_item(
        "signature_preprocess_seconds",
        telemetry.signature_preprocess_seconds,
    )?;
    stage_seconds.set_item("cluster_seed_seconds", telemetry.cluster_seed_seconds)?;

    let telemetry_dict = PyDict::new(py);
    telemetry_dict.set_item("stage_seconds", stage_seconds)?;
    let counts = PyDict::new(py);
    counts.set_item(
        "missing_specter_paper_count",
        telemetry.missing_specter_paper_count,
    )?;
    counts.set_item(
        "defaulted_name_count_signature_count",
        telemetry.defaulted_name_count_signature_count,
    )?;
    counts.set_item(
        "defaulted_name_count_first_count",
        telemetry.defaulted_name_count_first_count,
    )?;
    counts.set_item(
        "defaulted_name_count_first_last_count",
        telemetry.defaulted_name_count_first_last_count,
    )?;
    counts.set_item(
        "defaulted_name_count_last_count",
        telemetry.defaulted_name_count_last_count,
    )?;
    counts.set_item(
        "defaulted_name_count_last_first_initial_count",
        telemetry.defaulted_name_count_last_first_initial_count,
    )?;
    counts.set_item(
        "defaulted_signature_author_position_count",
        telemetry.defaulted_signature_author_position_count,
    )?;
    counts.set_item(
        "defaulted_paper_author_position_count",
        telemetry.defaulted_paper_author_position_count,
    )?;
    telemetry_dict.set_item("counts", counts)?;
    Ok(Some(telemetry_dict.unbind()))
}

#[pyfunction]
fn reset_last_json_ingest_telemetry() -> PyResult<()> {
    reset_last_json_ingest_telemetry_internal();
    Ok(())
}

fn extract_counter(obj: &Bound<'_, PyAny>) -> PyResult<Option<CounterData>> {
    if obj.is_none() {
        return Ok(None);
    }
    let dict = obj.downcast::<PyDict>()?;
    if dict.len() == 0 {
        return Ok(None);
    }
    let mut entries: Vec<(u64, f32)> = Vec::with_capacity(dict.len());
    let mut sum = 0.0f32;
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let val: f64 = v.extract()?;
        let val32 = val as f32;
        sum += val32;
        entries.push((fnv64(key.as_bytes()), val32));
    }
    entries.sort_unstable_by_key(|e| e.0);
    Ok(Some(CounterData { entries, sum }))
}

fn extract_reference_details_counters(
    py: Python<'_>,
    ref_details_obj: &Bound<'_, PyAny>,
) -> PyResult<(
    Option<CounterData>,
    Option<CounterData>,
    Option<CounterData>,
    Option<CounterData>,
)> {
    let tuple = ref_details_obj.extract::<(PyObject, PyObject, PyObject, PyObject)>()?;
    Ok((
        extract_counter(&tuple.0.bind(py))?,
        extract_counter(&tuple.1.bind(py))?,
        extract_counter(&tuple.2.bind(py))?,
        extract_counter(&tuple.3.bind(py))?,
    ))
}

fn extract_optional_string_set(obj: &Bound<'_, PyAny>) -> PyResult<Option<HashSet<String>>> {
    if obj.is_none() {
        return Ok(None);
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        let v: String = item?.extract()?;
        out.insert(v);
    }
    if out.is_empty() {
        Ok(None)
    } else {
        Ok(Some(out))
    }
}

fn canonical_signature_pair_ref<'a>(a: &'a str, b: &'a str) -> (&'a str, &'a str) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn canonical_signature_pair_owned(a: String, b: String) -> (String, String) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn canonical_signature_pair_cloned(a: &str, b: &str) -> (String, String) {
    let (left, right) = canonical_signature_pair_ref(a, b);
    (left.to_string(), right.to_string())
}

fn extract_pair_set(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<(String, String)>> {
    if obj.is_none() {
        return Ok(HashSet::new());
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        let tuple = item?;
        let (a, b): (String, String) = tuple.extract()?;
        out.insert(canonical_signature_pair_owned(a, b));
    }
    Ok(out)
}

fn extract_name_tuples_map(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, HashSet<String>>> {
    if obj.is_none() {
        return Ok(HashMap::new());
    }
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for item in PyIterator::from_object(obj)? {
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

fn extract_id_string(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(i.to_string());
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(u.to_string());
    }
    Ok(obj.str()?.to_string())
}

fn extract_set_id_string(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<PaperId>> {
    if obj.is_none() {
        return Ok(HashSet::new());
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        let v = extract_id_string(&item?)?;
        out.insert(v);
    }
    Ok(out)
}

fn extract_string_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in PyIterator::from_object(obj)? {
        out.push(item?.extract()?);
    }
    Ok(out)
}

fn get_namedtuple_item_or_attr<'py>(
    obj: &Bound<'py, PyAny>,
    allow_tuple_fastpath: bool,
    tuple_index: usize,
    attr_name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if allow_tuple_fastpath {
        return obj.get_item(tuple_index).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "NamedTuple fast-path index out of range or object not indexable: index={} attr={}",
                tuple_index, attr_name
            ))
        });
    }
    obj.getattr(attr_name)
}

fn validate_namedtuple_fastpath_contract(
    sample_obj: &Bound<'_, PyAny>,
    required_fields: &[(usize, &str)],
    tuple_label: &str,
) -> PyResult<bool> {
    let fields_obj = match sample_obj.getattr("_fields") {
        Ok(fields_obj) => fields_obj,
        Err(_) => return Ok(false),
    };

    let field_names: Vec<String> = fields_obj.extract().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{} fast-path expected _fields to be a tuple/list of field names",
            tuple_label
        ))
    })?;

    let max_required_index = required_fields
        .iter()
        .map(|(index, _)| *index)
        .max()
        .unwrap_or(0);
    if sample_obj.get_item(max_required_index).is_err() || field_names.len() <= max_required_index {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} fast-path contract mismatch: object is not indexable to required max index {} or _fields len={}",
            tuple_label,
            max_required_index,
            field_names.len()
        )));
    }

    for (field_index, expected_name) in required_fields {
        let actual_name = field_names
            .get(*field_index)
            .map(|s| s.as_str())
            .unwrap_or("<missing>");
        if actual_name != *expected_name {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} fast-path contract mismatch at index {}: expected '{}' got '{}'",
                tuple_label, field_index, expected_name, actual_name
            )));
        }
    }

    Ok(true)
}

fn validate_dict_namedtuple_fastpath_contract(
    rows: &Bound<'_, PyDict>,
    required_fields: &[(usize, &str)],
    tuple_label: &str,
) -> PyResult<bool> {
    if let Some((_, sample_obj)) = rows.iter().next() {
        return validate_namedtuple_fastpath_contract(&sample_obj, required_fields, tuple_label);
    }
    Ok(false)
}

fn extract_paper_authors_with_positions(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(i64, String)>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in PyIterator::from_object(obj)? {
        let author_obj = item?;
        let (position, author_name) = if let Ok(author_tuple) = author_obj.downcast::<PyTuple>() {
            if author_tuple.len() >= 2 {
                let author_name: String = author_tuple.get_item(0)?.extract()?;
                let position: i64 = author_tuple.get_item(1)?.extract()?;
                (position, author_name)
            } else {
                let position: i64 = author_obj.getattr("position")?.extract()?;
                let author_name: String = author_obj.getattr("author_name")?.extract()?;
                (position, author_name)
            }
        } else {
            let position: i64 = author_obj.getattr("position")?.extract()?;
            let author_name: String = author_obj.getattr("author_name")?.extract()?;
            (position, author_name)
        };
        out.push((position, author_name));
    }
    Ok(out)
}

fn extract_required_string_set(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<String>> {
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        out.insert(item?.extract()?);
    }
    Ok(out)
}

fn extract_affiliation_stopwords(py: Python<'_>) -> PyResult<HashSet<String>> {
    let text_module = py.import("s2and.text")?;
    let stopwords_obj = text_module.getattr("AFFILIATIONS_STOP_WORDS")?;
    extract_required_string_set(&stopwords_obj)
}

fn prefilter_affiliation_text(affiliations: &[String], stopwords: &HashSet<String>) -> String {
    if affiliations.is_empty() {
        return String::new();
    }
    let mut tokens: Vec<&str> = Vec::new();
    for word in affiliations
        .iter()
        .flat_map(|affiliation| affiliation.split_whitespace())
    {
        if !stopwords.contains(word) && py_len(word) > 1 {
            tokens.push(word);
        }
    }
    tokens.join(" ")
}

fn ensure_unidecode_for_text(
    unidecode_fn: &Bound<'_, PyAny>,
    text: &str,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    for ch in text.chars() {
        if ch.is_ascii() || unidecode_char_map.contains_key(&ch) {
            continue;
        }
        let mapped: String = unidecode_fn.call1((ch.to_string(),))?.extract()?;
        unidecode_char_map.insert(ch, mapped);
    }
    Ok(())
}

fn normalize_text_compat_from_map(
    text: &str,
    special_case_apostrophes: bool,
    unidecode_char_map: &HashMap<char, String>,
) -> String {
    if text.is_empty() {
        return String::new();
    }

    let mut transliterated = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            transliterated.push(ch.to_ascii_lowercase());
            continue;
        }
        if let Some(mapped) = unidecode_char_map.get(&ch) {
            for mapped_ch in mapped.chars() {
                for lowered in mapped_ch.to_lowercase() {
                    transliterated.push(lowered);
                }
            }
            continue;
        }
        for lowered in ch.to_lowercase() {
            transliterated.push(lowered);
        }
    }

    let source = if special_case_apostrophes {
        transliterated.replace('\'', "")
    } else {
        transliterated
    };
    let mut normalized = String::with_capacity(source.len());
    let mut prev_space = true;
    for ch in source.chars() {
        if ch.is_ascii_alphabetic() {
            normalized.push(ch);
            prev_space = false;
        } else if !prev_space {
            normalized.push(' ');
            prev_space = true;
        }
    }
    while normalized.ends_with(' ') {
        normalized.pop();
    }
    normalized
}

fn split_first_middle_hyphen_aware_compat(
    first_raw: &str,
    middle_raw: &str,
    name_prefixes: &HashSet<String>,
    unidecode_char_map: &HashMap<char, String>,
) -> (String, String) {
    let has_dash_in_first = first_raw.contains('-');
    let first_noapos = normalize_text_compat_from_map(first_raw, true, unidecode_char_map);
    let middle_norm = normalize_text_compat_from_map(middle_raw, false, unidecode_char_map);

    let mut f_parts: Vec<String> = first_noapos
        .split_whitespace()
        .map(|token| token.to_string())
        .collect();
    let m_parts: Vec<String> = middle_norm
        .split_whitespace()
        .map(|token| token.to_string())
        .collect();
    if let Some(prefix) = f_parts.first() {
        if name_prefixes.contains(prefix) {
            f_parts.remove(0);
        }
    }

    if f_parts.is_empty() {
        return (String::new(), m_parts.join(" "));
    }
    if has_dash_in_first {
        return (f_parts.join(" "), m_parts.join(" "));
    }
    let first = f_parts[0].clone();
    let middle = f_parts[1..]
        .iter()
        .chain(m_parts.iter())
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    (first, middle)
}

fn compute_block_compat(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let name_parts: Vec<&str> = name.split(' ').collect();
    if name_parts.len() == 1 {
        return name_parts[0].to_string();
    }
    let Some(first_initial) = name_parts[0].chars().next() else {
        return String::new();
    };
    format!("{} {}", first_initial, name_parts[name_parts.len() - 1])
}

fn env_flag_true(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            let lower = value.to_ascii_lowercase();
            lower == "1" || lower == "true" || lower == "yes"
        })
        .unwrap_or(false)
}

fn parse_fasttext_label(label: &str) -> String {
    label.rsplit("__").next().unwrap_or(label).to_string()
}

fn resolve_fasttext_model_path(py: Python<'_>) -> Option<String> {
    let consts = py.import("s2and.consts").ok()?;
    let fasttext_path: String = consts.getattr("FASTTEXT_PATH").ok()?.extract().ok()?;
    let file_cache = py.import("s2and.file_cache").ok()?;
    let cached_path = file_cache.getattr("cached_path").ok()?;
    cached_path.call1((fasttext_path,)).ok()?.extract().ok()
}

fn emit_runtime_warning(py: Python<'_>, message: &str) {
    if let Ok(warnings) = py.import("warnings") {
        let _ = warnings.call_method1("warn", (message.to_string(),));
    }
}

struct LanguageDetectorCompat {
    fasttext: Option<FastText>,
}

impl LanguageDetectorCompat {
    fn new(py: Python<'_>) -> Self {
        if env_flag_true("S2AND_SKIP_FASTTEXT") {
            return Self { fasttext: None };
        }
        let fasttext = resolve_fasttext_model_path(py).and_then(|model_path| {
            let mut model = FastText::new();
            match model.load_model(&model_path) {
                Ok(()) => Some(model),
                Err(err) => {
                    let warning = format!(
                        "s2and_rust: failed to load fastText model at '{}' ({}); falling back to CLD2-only language detection.",
                        model_path, err
                    );
                    emit_runtime_warning(py, &warning);
                    eprintln!("{warning}");
                    None
                }
            }
        });
        Self { fasttext }
    }

    fn detect(&self, text: &str) -> (bool, bool, String) {
        if text.split_whitespace().count() <= 1 {
            return (false, false, "un".to_string());
        }

        let mut alpha_count = 0usize;
        let mut uppercase_count = 0usize;
        for ch in text.chars() {
            if ch.is_alphabetic() {
                alpha_count += 1;
                if ch.is_uppercase() {
                    uppercase_count += 1;
                }
            }
        }
        if alpha_count == 0 {
            return (false, false, "un".to_string());
        }

        let predicted_language_ft = if let Some(fasttext_model) = &self.fasttext {
            let uppercase_ratio = uppercase_count as f64 / alpha_count as f64;
            let mut fasttext_input = text.replace('\n', " ");
            if uppercase_ratio > 0.9 {
                fasttext_input = fasttext_input.to_lowercase();
            }
            match fasttext_model.predict(&fasttext_input, 1, 0.0) {
                Ok(predictions) => predictions
                    .first()
                    .map(|prediction| parse_fasttext_label(&prediction.label))
                    .unwrap_or_else(|| "un_ft".to_string()),
                Err(_) => "un_ft".to_string(),
            }
        } else {
            "un_ft".to_string()
        };

        let cld2_result = cld2_detect_language_ext(text, Cld2Format::Text, &Default::default());
        let mut predicted_language_2 = match cld2_result.scores[0].language {
            Some(lang) => lang.0.to_string(),
            None => "un_2".to_string(),
        };
        if predicted_language_2 == "un" {
            predicted_language_2 = "un_2".to_string();
        }

        let (predicted_language, is_reliable) =
            if predicted_language_ft == "un_ft" && predicted_language_2 == "un_2" {
                ("un".to_string(), false)
            } else if predicted_language_ft == "un_ft" {
                (predicted_language_2, true)
            } else if predicted_language_2 == "un_2" {
                (predicted_language_ft, true)
            } else if predicted_language_2 != predicted_language_ft {
                ("un".to_string(), false)
            } else {
                (predicted_language_2, true)
            };

        let is_english = predicted_language == "en";
        (is_reliable, is_english, predicted_language)
    }
}

fn extract_orcid_from_source_id(source_id: &str) -> Option<String> {
    let chars: Vec<char> = source_id.chars().collect();
    if chars.len() < 16 {
        return None;
    }
    for start in 0..chars.len() {
        let mut idx = start;
        let mut compact = String::with_capacity(16);
        let mut valid = true;

        for (group_idx, group_len) in [4usize, 4, 4, 3].iter().enumerate() {
            for _ in 0..*group_len {
                if idx >= chars.len() || !chars[idx].is_ascii_digit() {
                    valid = false;
                    break;
                }
                compact.push(chars[idx]);
                idx += 1;
            }
            if !valid {
                break;
            }
            if group_idx < 3 && idx < chars.len() && chars[idx] == '-' {
                idx += 1;
            }
        }
        if !valid {
            continue;
        }
        if idx >= chars.len() {
            continue;
        }
        let last = chars[idx];
        if !(last.is_ascii_digit() || last == 'X') {
            continue;
        }
        compact.push(last);
        return Some(compact);
    }
    None
}

fn counter_data_from_usize_map(counter_map: HashMap<String, usize>) -> Option<CounterData> {
    if counter_map.is_empty() {
        return None;
    }
    let mut entries: Vec<(u64, f32)> = counter_map
        .iter()
        .map(|(k, v)| (fnv64(k.as_bytes()), *v as f32))
        .collect();
    entries.sort_unstable_by_key(|e| e.0);
    let sum: f32 = entries.iter().map(|e| e.1).sum();
    Some(CounterData { entries, sum })
}

fn extract_specter_for_paper_id(
    spec_dict: &Bound<'_, PyDict>,
    paper_id: &str,
) -> PyResult<Option<Vec<f32>>> {
    if let Ok(Some(val)) = spec_dict.get_item(paper_id) {
        return extract_specter_vec(&val);
    }
    if let Ok(i) = paper_id.parse::<i64>() {
        if let Ok(Some(val)) = spec_dict.get_item(i) {
            return extract_specter_vec(&val);
        }
    }
    if let Ok(u) = paper_id.parse::<u64>() {
        if let Ok(Some(val)) = spec_dict.get_item(u) {
            return extract_specter_vec(&val);
        }
    }
    Ok(None)
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
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        let all_zero = slice.iter().all(|v| *v == 0.0);
        if all_zero {
            return Ok(None);
        }
        return Ok(Some(slice.to_vec()));
    }
    if let Ok(arr) = obj.downcast::<PyArray1<f64>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
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

fn extract_u32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(arr) = obj.downcast::<PyArray1<u32>>() {
        let readonly = arr.readonly();
        return Ok(readonly.as_slice()?.to_vec());
    }
    if let Ok(arr) = obj.downcast::<PyArray1<u64>>() {
        let readonly = arr.readonly();
        return readonly
            .as_slice()?
            .iter()
            .map(|value| {
                u32::try_from(*value).map_err(|_| {
                    pyo3::exceptions::PyOverflowError::new_err(format!(
                        "component member signature index exceeds u32: {value}"
                    ))
                })
            })
            .collect();
    }
    let values: Vec<u64> = obj.extract()?;
    values
        .into_iter()
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(format!(
                    "component member signature index exceeds u32: {value}"
                ))
            })
        })
        .collect()
}

fn extract_component_member_indices(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, Vec<u32>>> {
    let mut out = HashMap::new();
    let items = obj.call_method0("items")?;
    for item in PyIterator::from_object(&items)? {
        let tuple = item?.downcast_into::<PyTuple>()?;
        if tuple.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "component_member_indices_by_key.items() yielded a non-pair",
            ));
        }
        let component_key: String = tuple.get_item(0)?.extract()?;
        let members = extract_u32_vec(&tuple.get_item(1)?)?;
        out.insert(component_key, members);
    }
    Ok(out)
}

fn extract_specter_vec_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut vectors = Vec::new();
    for item in PyIterator::from_object(obj)? {
        if let Some(vector) = extract_specter_vec(&item?)? {
            vectors.push(vector);
        }
    }
    Ok(vectors)
}

fn extract_string_count_pairs(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(String, f32)>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let dict = obj.downcast::<PyDict>()?;
    if dict.len() == 0 {
        return Ok(Vec::new());
    }
    let mut entries = Vec::with_capacity(dict.len());
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let val: f64 = v.extract()?;
        entries.push((key, val as f32));
    }
    Ok(entries)
}

fn term_token_count(value: &str) -> u8 {
    value
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .count()
        .min(u8::MAX as usize) as u8
}

fn extract_string_hashes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
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

fn extract_query_terms(obj: &Bound<'_, PyAny>) -> PyResult<Vec<RetrievalQueryTerm>> {
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

fn extract_optional_string_hash(obj: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
    if obj.is_none() {
        return Ok(None);
    }
    let value: String = obj.extract()?;
    Ok(Some(fnv64(value.as_bytes())))
}

fn same_prefix_tokens_compat(a: &str, b: &str) -> bool {
    let ta: Vec<&str> = a.split_whitespace().collect();
    let tb: Vec<&str> = b.split_whitespace().collect();
    for (x, y) in ta.iter().zip(tb.iter()) {
        if !(x.starts_with(y) || y.starts_with(x)) {
            return false;
        }
    }
    true
}

fn exact_name_match_compat(a: &str, b: &str) -> bool {
    !a.is_empty() && a == b
}

fn counter_query_overlap_hashes(
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

fn overlap_idf_weight(df_map: &HashMap<u64, usize>, hash: u64, total_summary_count: usize) -> f64 {
    let df = df_map.get(&hash).copied().unwrap_or(0) as f64;
    (((total_summary_count as f64) + 1.0) / (df + 1.0)).ln() + 1.0
}

fn overlap_query_term_weight(
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

fn weighted_counter_query_overlap(
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

fn middle_initial_score_hashes(
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

fn first_name_score_prefix(query_first: &str, counts: &[(String, f32)], size: usize) -> f64 {
    if size == 0 || py_len(query_first) <= 1 || counts.is_empty() {
        return 0.0;
    }
    let mut best = 0.0f64;
    for (first_name, count) in counts.iter() {
        if py_len(first_name) <= 1 {
            continue;
        }
        if same_prefix_tokens_compat(query_first, first_name) {
            best = best.max((*count as f64) / (size as f64));
        }
    }
    best
}

fn first_name_score_mode(
    query_first: &str,
    counts: &[(String, f32)],
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
                if same_prefix_tokens_compat(query_first, first_name) {
                    share
                } else {
                    0.0
                }
            }
            RetrievalFirstNameMode::ExactOnly => {
                if exact_name_match_compat(query_first, first_name) {
                    share
                } else {
                    0.0
                }
            }
            RetrievalFirstNameMode::ExactThenPrefixHalf => {
                if exact_name_match_compat(query_first, first_name) {
                    share
                } else if same_prefix_tokens_compat(query_first, first_name) {
                    share * 0.5
                } else {
                    0.0
                }
            }
            RetrievalFirstNameMode::PrefixLengthRatio => {
                if same_prefix_tokens_compat(query_first, first_name) {
                    let query_len = py_len(query_first) as f64;
                    let candidate_len = py_len(first_name) as f64;
                    share * (query_len.min(candidate_len) / query_len.max(candidate_len))
                } else {
                    0.0
                }
            }
            RetrievalFirstNameMode::ExactThenPrefixLengthRatio => {
                if exact_name_match_compat(query_first, first_name) {
                    share
                } else if same_prefix_tokens_compat(query_first, first_name) {
                    let query_len = py_len(query_first) as f64;
                    let candidate_len = py_len(first_name) as f64;
                    share * (query_len.min(candidate_len) / query_len.max(candidate_len)) * 0.75
                } else {
                    0.0
                }
            }
        };
        best = best.max(candidate);
    }
    best
}

fn year_score(query_year: Option<i64>, summary: &RetrievalSummaryData) -> f64 {
    let Some(query_year_value) = query_year else {
        return 0.0;
    };
    let Some(year_mean) = summary.year_mean else {
        return 0.0;
    };
    let distance = ((query_year_value as f64) - year_mean).abs();
    let mut score = (1.0 - (distance / RETRIEVAL_YEAR_SCORE_DECAY_YEARS)).max(0.0);
    if let (Some(year_min), Some(year_max)) = (summary.year_min, summary.year_max) {
        if query_year_value < year_min - RETRIEVAL_YEAR_SCORE_RANGE_GAP
            || query_year_value > year_max + RETRIEVAL_YEAR_SCORE_RANGE_GAP
        {
            score -= RETRIEVAL_YEAR_SCORE_RANGE_PENALTY;
        }
    }
    score
}

fn contains_hashed_value(sorted_hashes: &[u64], target: u64) -> bool {
    sorted_hashes.binary_search(&target).is_ok()
}

fn has_middle_initial_conflict(query_hashes: &[u64], counter: &Option<CounterData>) -> bool {
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

fn has_impossible_year_conflict(
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
    query_year_value < year_min - max_year_gap || query_year_value > year_max + max_year_gap
}

fn extract_retrieval_summary(
    obj: &Bound<'_, PyAny>,
    include_exemplars: bool,
) -> PyResult<RetrievalSummaryData> {
    let component_key: String = obj.getattr("component_key")?.extract()?;
    let size: usize = obj.getattr("size")?.extract()?;
    let first_name_counts = extract_string_count_pairs(&obj.getattr("first_name_counts")?)?;
    let middle_initial_counts = extract_counter(&obj.getattr("middle_initial_counts")?)?;
    let coauthor_counts = extract_counter(&obj.getattr("coauthor_counts")?)?;
    let affiliation_counts = extract_counter(&obj.getattr("affiliation_counts")?)?;
    let venue_counts = extract_counter(&obj.getattr("venue_counts")?)?;
    let title_counts = extract_counter(&obj.getattr("title_counts")?)?;
    let year_min: Option<i64> = obj.getattr("year_min")?.extract()?;
    let year_max: Option<i64> = obj.getattr("year_max")?.extract()?;
    let year_mean: Option<f64> = obj.getattr("year_mean")?.extract()?;
    let orcid_hashes = extract_string_hashes(&obj.getattr("orcid_values")?)?;
    let specter_centroid = extract_specter_vec(&obj.getattr("specter_centroid")?)?;
    let specter_centroid_norm = specter_centroid.as_ref().map(|values| {
        values
            .iter()
            .map(|value| {
                let val = *value as f64;
                val * val
            })
            .sum::<f64>()
            .sqrt()
    });
    let exemplar_vectors = if include_exemplars {
        extract_specter_vec_list(&obj.getattr("exemplar_vectors")?)?
    } else {
        Vec::new()
    };
    let exemplar_norms = exemplar_vectors
        .iter()
        .map(|values| {
            values
                .iter()
                .map(|value| {
                    let val = *value as f64;
                    val * val
                })
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    Ok(RetrievalSummaryData {
        component_key,
        size,
        first_name_counts,
        middle_initial_counts,
        coauthor_counts,
        affiliation_counts,
        venue_counts,
        title_counts,
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

fn extract_retrieval_query(obj: &Bound<'_, PyAny>) -> PyResult<RetrievalQueryData> {
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
    let orcid_hash = extract_optional_string_hash(&obj.getattr("orcid")?)?;
    let specter = extract_specter_vec(&obj.getattr("specter")?)?;
    let specter_norm = specter.as_ref().map(|values| {
        values
            .iter()
            .map(|value| {
                let val = *value as f64;
                val * val
            })
            .sum::<f64>()
            .sqrt()
    });

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

fn extract_retrieval_weights(weights: Vec<f64>) -> PyResult<RetrievalHybridWeights> {
    if weights.len() != 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected 5 retrieval weights, got {}",
            weights.len()
        )));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Retrieval weights must all be finite",
        ));
    }
    Ok(RetrievalHybridWeights::from_array([
        weights[0], weights[1], weights[2], weights[3], weights[4],
    ]))
}

fn default_overlap_config() -> RetrievalOverlapConfig {
    RetrievalOverlapConfig {
        use_idf: false,
        per_term_cap: None,
        total_cap: None,
        min_token_count: 1,
        unigram_weight: 1.0,
        multi_token_weight: 1.0,
    }
}

fn parse_first_name_mode(mode: &str) -> PyResult<RetrievalFirstNameMode> {
    match mode {
        "prefix" => Ok(RetrievalFirstNameMode::Prefix),
        "exact_only" => Ok(RetrievalFirstNameMode::ExactOnly),
        "exact_then_prefix_half" => Ok(RetrievalFirstNameMode::ExactThenPrefixHalf),
        "prefix_length_ratio" => Ok(RetrievalFirstNameMode::PrefixLengthRatio),
        "exact_then_prefix_length_ratio" => Ok(RetrievalFirstNameMode::ExactThenPrefixLengthRatio),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown first_name_mode: {mode}"
        ))),
    }
}

fn parse_specter_mode(mode: &str) -> PyResult<RetrievalSpecterMode> {
    match mode {
        "centroid" => Ok(RetrievalSpecterMode::Centroid),
        "exemplar_max" => Ok(RetrievalSpecterMode::ExemplarMax),
        "centroid_exemplar_50_50" => Ok(RetrievalSpecterMode::CentroidExemplar50_50),
        "centroid_exemplar_25_75" => Ok(RetrievalSpecterMode::CentroidExemplar25_75),
        "centroid_exemplar_75_25" => Ok(RetrievalSpecterMode::CentroidExemplar75_25),
        "max_centroid_exemplar" => Ok(RetrievalSpecterMode::MaxOfCentroidExemplar),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown specter_mode: {mode}"
        ))),
    }
}

fn build_experimental_config(
    first_name_mode: &str,
    specter_mode: &str,
    coauthor_use_idf: bool,
    coauthor_per_term_cap: Option<f64>,
    coauthor_total_cap: Option<f64>,
    affiliation_use_idf: bool,
    affiliation_per_term_cap: Option<f64>,
    affiliation_total_cap: Option<f64>,
    affiliation_min_token_count: usize,
    affiliation_unigram_weight: f64,
    affiliation_multi_token_weight: f64,
) -> PyResult<RetrievalExperimentalConfig> {
    if affiliation_min_token_count == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "affiliation_min_token_count must be positive",
        ));
    }
    if !affiliation_unigram_weight.is_finite() || !affiliation_multi_token_weight.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Affiliation structure weights must be finite",
        ));
    }
    Ok(RetrievalExperimentalConfig {
        first_name_mode: parse_first_name_mode(first_name_mode)?,
        specter_mode: parse_specter_mode(specter_mode)?,
        coauthor: RetrievalOverlapConfig {
            use_idf: coauthor_use_idf,
            per_term_cap: coauthor_per_term_cap,
            total_cap: coauthor_total_cap,
            ..default_overlap_config()
        },
        affiliation: RetrievalOverlapConfig {
            use_idf: affiliation_use_idf,
            per_term_cap: affiliation_per_term_cap,
            total_cap: affiliation_total_cap,
            min_token_count: affiliation_min_token_count.min(u8::MAX as usize) as u8,
            unigram_weight: affiliation_unigram_weight,
            multi_token_weight: affiliation_multi_token_weight,
        },
    })
}

fn specter_exemplar_score(query: &RetrievalQueryData, summary: &RetrievalSummaryData) -> f64 {
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

fn score_hybrid_centroid_query(
    query: &RetrievalQueryData,
    summary: &RetrievalSummaryData,
    _max_block_component_size: usize,
    weights: RetrievalHybridWeights,
) -> f32 {
    let coauthor_score = counter_query_overlap_hashes(
        &query.coauthor_hashes,
        &summary.coauthor_counts,
        summary.size,
    );
    let affiliation_score = counter_query_overlap_hashes(
        &query.affiliation_hashes,
        &summary.affiliation_counts,
        summary.size,
    );
    let middle_score = middle_initial_score_hashes(
        &query.middle_initial_hashes,
        &summary.middle_initial_counts,
        summary.size,
    );
    let first_name_score =
        first_name_score_prefix(&query.first, &summary.first_name_counts, summary.size);
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
    (weights.centroid * centroid_score
        + weights.coauthor * coauthor_score
        + weights.affiliation * affiliation_score
        + weights.middle * middle_score
        + weights.first_name * first_name_score) as f32
}

fn score_experimental_hybrid_centroid_query(
    query: &RetrievalQueryData,
    summary: &RetrievalSummaryData,
    _max_block_component_size: usize,
    weights: RetrievalHybridWeights,
    config: RetrievalExperimentalConfig,
    coauthor_cluster_df: &HashMap<u64, usize>,
    affiliation_cluster_df: &HashMap<u64, usize>,
    total_summary_count: usize,
) -> f32 {
    let coauthor_score = weighted_counter_query_overlap(
        &query.coauthor_terms,
        &summary.coauthor_counts,
        summary.size,
        coauthor_cluster_df,
        total_summary_count,
        config.coauthor,
    );
    let affiliation_score = weighted_counter_query_overlap(
        &query.affiliation_terms,
        &summary.affiliation_counts,
        summary.size,
        affiliation_cluster_df,
        total_summary_count,
        config.affiliation,
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
        config.first_name_mode,
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
    let specter_score = match config.specter_mode {
        RetrievalSpecterMode::Centroid => centroid_score,
        RetrievalSpecterMode::ExemplarMax => exemplar_score,
        RetrievalSpecterMode::CentroidExemplar50_50 => 0.5 * centroid_score + 0.5 * exemplar_score,
        RetrievalSpecterMode::CentroidExemplar25_75 => {
            0.25 * centroid_score + 0.75 * exemplar_score
        }
        RetrievalSpecterMode::CentroidExemplar75_25 => {
            0.75 * centroid_score + 0.25 * exemplar_score
        }
        RetrievalSpecterMode::MaxOfCentroidExemplar => centroid_score.max(exemplar_score),
    };
    (weights.centroid * specter_score
        + weights.coauthor * coauthor_score
        + weights.affiliation * affiliation_score
        + weights.middle * middle_score
        + weights.first_name * first_name_score) as f32
}

fn chooser_summary_features(
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

fn update_cluster_df_from_counter(
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

fn specter_payload_to_dict<'py>(
    py: Python<'py>,
    payload: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    if let Ok(dict) = payload.downcast::<PyDict>() {
        return Ok(dict.clone());
    }

    if let Ok(tuple_payload) = payload.downcast::<PyTuple>() {
        if tuple_payload.len() != 2 {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported specter pickle tuple payload; expected (X, keys), got tuple length {}",
                tuple_payload.len()
            )));
        }
        let matrix = tuple_payload.get_item(0)?;
        let keys = tuple_payload.get_item(1)?;
        let out = PyDict::new(py);
        for (idx, key_item) in PyIterator::from_object(&keys)?.enumerate() {
            let key = key_item?;
            let row = matrix.get_item(idx)?;
            out.set_item(key, row)?;
        }
        return Ok(out);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Unsupported specter pickle payload; expected dict or (X, keys) tuple",
    ))
}

fn load_pickle_dict<'py>(py: Python<'py>, path: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
    let builtins = py.import("builtins")?;
    let pickle = py.import("pickle")?;
    let file_obj = builtins.call_method1("open", (path, "rb"))?;
    let loaded = pickle.call_method1("load", (&file_obj,));
    let _ = file_obj.call_method0("close");
    match loaded {
        Ok(value) => {
            if value.is_none() {
                Ok(None)
            } else {
                Ok(Some(specter_payload_to_dict(py, &value)?))
            }
        }
        Err(err) => Err(err),
    }
}

type JsonValue = serde_json::Value;
type JsonObject = serde_json::Map<String, JsonValue>;

fn load_json_value(path: &str) -> PyResult<JsonValue> {
    let file = File::open(path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!("failed to open JSON path {}: {}", path, err))
    })?;
    let reader = BufReader::new(file);
    serde_json::from_reader::<_, JsonValue>(reader).map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "failed to parse JSON path {}: {}",
            path, err
        ))
    })
}

fn json_as_object<'a>(value: &'a JsonValue, context: &str) -> PyResult<&'a JsonObject> {
    value.as_object().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("{} must be a JSON object", context))
    })
}

fn json_value_to_string(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(v) => Some(v.clone()),
        JsonValue::Number(v) => Some(v.to_string()),
        JsonValue::Bool(v) => Some(v.to_string()),
        _ => None,
    }
}

fn json_value_to_id(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(v) => Some(v.clone()),
        JsonValue::Number(v) => Some(v.to_string()),
        _ => None,
    }
}

fn json_value_to_i64(value: &JsonValue) -> Option<i64> {
    match value {
        JsonValue::Number(v) => {
            if let Some(i) = v.as_i64() {
                Some(i)
            } else if let Some(u) = v.as_u64() {
                Some(u as i64)
            } else {
                v.as_f64().map(|f| f as i64)
            }
        }
        _ => None,
    }
}

fn json_get_required<'a>(obj: &'a JsonObject, key: &str, context: &str) -> PyResult<&'a JsonValue> {
    obj.get(key).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "missing required key '{}' in {}",
            key, context
        ))
    })
}

fn json_get_string(obj: &JsonObject, key: &str, default: &str) -> String {
    match obj.get(key) {
        None | Some(JsonValue::Null) => default.to_string(),
        Some(value) => json_value_to_string(value).unwrap_or_else(|| default.to_string()),
    }
}

fn json_get_optional_string(obj: &JsonObject, key: &str) -> Option<String> {
    obj.get(key).and_then(|value| match value {
        JsonValue::Null => None,
        _ => json_value_to_string(value),
    })
}

fn json_get_i64_optional(obj: &JsonObject, key: &str) -> Option<i64> {
    obj.get(key).and_then(json_value_to_i64)
}

fn json_get_string_list(value: Option<&JsonValue>) -> Vec<String> {
    let Some(array) = value.and_then(JsonValue::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        if let Some(text) = json_value_to_string(item) {
            out.push(text);
        }
    }
    out
}

fn json_get_id_set(value: Option<&JsonValue>) -> HashSet<PaperId> {
    let Some(array) = value.and_then(JsonValue::as_array) else {
        return HashSet::new();
    };
    let mut out = HashSet::with_capacity(array.len());
    for item in array {
        if let Some(id) = json_value_to_id(item) {
            out.insert(id);
        }
    }
    out
}

fn json_extract_string_f64_map(value: Option<&JsonValue>) -> HashMap<String, f64> {
    let Some(object) = value.and_then(JsonValue::as_object) else {
        return HashMap::new();
    };
    let mut out = HashMap::with_capacity(object.len());
    for (key, val) in object {
        if let Some(f) = val.as_f64() {
            out.insert(key.clone(), f);
        } else if let Some(i) = val.as_i64() {
            out.insert(key.clone(), i as f64);
        } else if let Some(u) = val.as_u64() {
            out.insert(key.clone(), u as f64);
        }
    }
    out
}

fn load_raw_name_counts_from_json_path(
    path: Option<&str>,
    expected_normalization_version: Option<&str>,
    allow_normalization_version_mismatch: bool,
) -> PyResult<RawNameCountMaps> {
    let Some(path_value) = path else {
        return Ok(RawNameCountMaps::default());
    };
    let counts_json = load_json_value(path_value)?;
    let counts_obj = json_as_object(&counts_json, "name counts payload")?;

    // Validate normalization_version if an expected version was provided.
    if let Some(expected) = expected_normalization_version {
        match counts_obj.get("normalization_version") {
            None => {
                let msg = format!(
                    "Missing normalization_version in name counts artifact; fail-fast by default. \
                     path={} expected={} override=S2AND_ALLOW_NORMALIZATION_VERSION_MISMATCH=1",
                    path_value, expected,
                );
                if !allow_normalization_version_mismatch {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(msg));
                }
                eprintln!("WARNING: {}", msg);
            }
            Some(artifact_val) => {
                let artifact_version = artifact_val.as_str().unwrap_or("");
                if artifact_version != expected {
                    let msg = format!(
                        "Normalization version mismatch between runtime and name-count artifact; \
                         fail-fast by default. path={} expected={} artifact={} \
                         override=S2AND_ALLOW_NORMALIZATION_VERSION_MISMATCH=1",
                        path_value, expected, artifact_version,
                    );
                    if !allow_normalization_version_mismatch {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(msg));
                    }
                    eprintln!("WARNING: {}", msg);
                }
            }
        }
    }

    let first = json_extract_string_f64_map(counts_obj.get("first_dict"));
    let last = json_extract_string_f64_map(counts_obj.get("last_dict"));
    let first_last = json_extract_string_f64_map(counts_obj.get("first_last_dict"));
    let last_first_initial = json_extract_string_f64_map(counts_obj.get("last_first_initial_dict"));
    Ok(RawNameCountMaps {
        first,
        last,
        first_last,
        last_first_initial,
    })
}

fn default_name_tuples_path(py: Python<'_>) -> PyResult<String> {
    let consts = py.import("s2and.consts")?;
    let project_root: String = consts.getattr("PROJECT_ROOT_PATH")?.extract()?;
    let pathlib = py.import("pathlib")?;
    let path_obj = pathlib
        .getattr("Path")?
        .call1((project_root,))?
        .call_method1("joinpath", ("data", "s2and_name_tuples_filtered.txt"))?;
    path_obj.call_method0("as_posix")?.extract()
}

fn load_name_tuples_from_text_path(
    py: Python<'_>,
    path: Option<&str>,
) -> PyResult<HashMap<String, HashSet<String>>> {
    let effective_path = match path {
        Some(value) => value.to_string(),
        None => default_name_tuples_path(py)?,
    };
    if !Path::new(&effective_path).exists() {
        return Ok(HashMap::new());
    }
    let text = fs::read_to_string(&effective_path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to read name tuples path {}: {}",
            effective_path, err
        ))
    })?;
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((a, b)) = trimmed.split_once(',') {
            out.entry(a.to_string())
                .or_insert_with(HashSet::new)
                .insert(b.to_string());
        }
    }
    Ok(out)
}

fn has_name_counts_artifact(raw_name_counts: &RawNameCountMaps) -> bool {
    !raw_name_counts.first.is_empty()
        || !raw_name_counts.last.is_empty()
        || !raw_name_counts.first_last.is_empty()
        || !raw_name_counts.last_first_initial.is_empty()
}

fn canonical_last_for_counts(raw_last: &str, normalized_last: &str) -> String {
    if raw_last.contains('-') || normalized_last.contains(' ') {
        normalized_last.replace(' ', "")
    } else {
        normalized_last.to_string()
    }
}

#[derive(Clone, Copy, Default)]
struct NameCountsDefaultTelemetry {
    first: bool,
    first_last: bool,
    last: bool,
    last_first_initial: bool,
}

impl NameCountsDefaultTelemetry {
    fn any(self) -> bool {
        self.first || self.first_last || self.last || self.last_first_initial
    }
}

struct NameCountsBuildResult {
    data: Option<NameCountsData>,
    telemetry: NameCountsDefaultTelemetry,
}

fn build_name_counts_data_from_artifact(
    raw_name_counts: &RawNameCountMaps,
    raw_first: &str,
    first_normalized_token: &str,
    first_without_apostrophe: &str,
    raw_last: &str,
    last_normalized: &str,
) -> NameCountsBuildResult {
    if !has_name_counts_artifact(raw_name_counts) {
        return NameCountsBuildResult {
            data: None,
            telemetry: NameCountsDefaultTelemetry::default(),
        };
    }

    let mut telemetry = NameCountsDefaultTelemetry::default();
    let mut first_for_counts = first_normalized_token.to_string();
    if first_for_counts.is_empty() {
        first_for_counts = first_without_apostrophe
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();
    }
    if raw_first.contains('-') {
        let joined = first_without_apostrophe.replace(' ', "");
        if !joined.is_empty() {
            first_for_counts = joined;
        }
    }

    let last_for_counts = canonical_last_for_counts(raw_last, last_normalized);
    let first_last_key = format!("{} {}", first_for_counts, last_for_counts)
        .trim()
        .to_string();
    let first_initial = first_for_counts
        .chars()
        .next()
        .map(|ch| ch.to_string())
        .unwrap_or_default();
    let last_first_initial_key = format!("{} {}", last_for_counts, first_initial)
        .trim()
        .to_string();

    let first = if py_len(&first_for_counts) > 1 {
        match raw_name_counts.first.get(&first_for_counts) {
            Some(value) => *value,
            None => {
                telemetry.first = true;
                1.0
            }
        }
    } else {
        f64::NAN
    };
    let first_last = if py_len(&first_for_counts) > 1 {
        match raw_name_counts.first_last.get(&first_last_key) {
            Some(value) => *value,
            None => {
                telemetry.first_last = true;
                1.0
            }
        }
    } else {
        f64::NAN
    };
    let last = match raw_name_counts.last.get(&last_for_counts) {
        Some(value) => *value,
        None => {
            telemetry.last = true;
            1.0
        }
    };
    let last_first_initial = match raw_name_counts
        .last_first_initial
        .get(&last_first_initial_key)
    {
        Some(value) => *value,
        None => {
            telemetry.last_first_initial = true;
            1.0
        }
    };

    NameCountsBuildResult {
        data: Some(NameCountsData {
            first,
            first_last,
            last,
            last_first_initial,
        }),
        telemetry,
    }
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

fn first_name_forms(value: &str) -> (String, String, String) {
    let normalized = value.trim().to_lowercase();
    let parts: Vec<&str> = normalized.split_whitespace().collect();
    let joined = parts.join("");
    let token = parts
        .first()
        .map_or_else(|| normalized.clone(), |part| (*part).to_string());
    (normalized, joined, token)
}

fn first_names_name_compatible(
    first_1: &str,
    first_2: &str,
    name_tuples: &HashMap<String, HashSet<String>>,
) -> bool {
    let first_1 = first_1.trim().to_lowercase();
    let first_2 = first_2.trim().to_lowercase();
    if first_1.is_empty() || first_2.is_empty() {
        return true;
    }
    if first_1.chars().next() != first_2.chars().next() {
        return false;
    }
    if same_prefix_tokens(&first_1, &first_2) {
        return true;
    }
    let forms_1 = first_name_forms(&first_1);
    let forms_2 = first_name_forms(&first_2);
    name_tuple_contains(name_tuples, &forms_1.0, &forms_2.0)
        || name_tuple_contains(name_tuples, &forms_1.1, &forms_2.1)
        || name_tuple_contains(name_tuples, &forms_1.2, &forms_2.2)
}

fn subblock_tokens_from_key(subblock_key: &str) -> Vec<String> {
    let local_key = subblock_key
        .rsplit_once("::")
        .map_or(subblock_key, |(_prefix, suffix)| suffix);
    let mut values = HashSet::new();
    for raw_token in local_key.split(',') {
        let token = raw_token
            .trim()
            .split_once('|')
            .map_or(raw_token.trim(), |(token, _rest)| token.trim())
            .to_lowercase();
        if py_len(&token) > 1 {
            values.insert(token);
        }
    }
    let mut out: Vec<String> = values.into_iter().collect();
    out.sort_unstable();
    out
}

fn extract_string_string_map(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, String>> {
    let dict = obj.downcast::<PyDict>()?;
    let mut out = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        out.insert(key.extract()?, value.extract()?);
    }
    Ok(out)
}

fn extract_string_vec_map(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, Vec<String>>> {
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

fn filter_text_for_char_ngrams(text: &str, stopwords: Option<&HashSet<String>>) -> String {
    let Some(stopwords_set) = stopwords else {
        return text.to_string();
    };
    text.split(' ')
        .filter(|word| !stopwords_set.contains(*word) && py_len(word) > 2)
        .collect::<Vec<_>>()
        .join(" ")
}

fn char_ngrams_counter_python_compat(
    text: &str,
    use_unigrams: bool,
    use_bigrams: bool,
    stopwords: Option<&HashSet<String>>,
) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::new();
    }
    let filtered_text = filter_text_for_char_ngrams(text, stopwords);
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

fn word_ngrams_counter_python_compat(
    text: &str,
    stopwords: &HashSet<String>,
) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::new();
    }
    let text_split: Vec<&str> = text
        .split_whitespace()
        .filter(|word| !stopwords.contains(*word) && py_len(word) > 1)
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

fn char_ngrams_counter(text: &str) -> HashMap<String, usize> {
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

fn word_ngrams_counter(text: &str) -> HashMap<String, usize> {
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

#[pyfunction]
#[pyo3(signature = (coauthor_texts, affiliation_texts, num_threads = None))]
fn signature_ngrams_batch(
    py: Python<'_>,
    coauthor_texts: Vec<String>,
    affiliation_texts: Vec<String>,
    num_threads: Option<usize>,
) -> PyResult<(Vec<HashMap<String, usize>>, Vec<HashMap<String, usize>>)> {
    if coauthor_texts.len() != affiliation_texts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coauthor_texts and affiliation_texts must have equal length",
        ));
    }
    let n = coauthor_texts.len();

    let pairs = py.allow_threads(|| {
        let compute = || {
            (0..n)
                .into_par_iter()
                .map(|idx| {
                    let coauthors = char_ngrams_counter(&coauthor_texts[idx]);
                    let affiliations = word_ngrams_counter(&affiliation_texts[idx]);
                    (coauthors, affiliations)
                })
                .collect::<Vec<_>>()
        };
        install_with_optional_rayon_pool(num_threads, compute)
    });

    let mut coauthor_out: Vec<HashMap<String, usize>> = Vec::with_capacity(n);
    let mut affiliation_out: Vec<HashMap<String, usize>> = Vec::with_capacity(n);
    for (coauthors, affiliations) in pairs {
        coauthor_out.push(coauthors);
        affiliation_out.push(affiliations);
    }
    Ok((coauthor_out, affiliation_out))
}

fn counter_jaccard_data(
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

fn set_jaccard_data<T: Eq + std::hash::Hash>(
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

fn refs_jaccard<T: Eq + std::hash::Hash>(set1: &HashSet<T>, set2: &HashSet<T>) -> f64 {
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

fn compute_name_counts_data(
    counts1: &Option<NameCountsData>,
    counts2: &Option<NameCountsData>,
) -> [f64; 6] {
    let (Some(c1), Some(c2)) = (counts1.as_ref(), counts2.as_ref()) else {
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

fn first_names_equal(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let (Some(n1), Some(n2)) = (name1, name2) else {
        return f64::NAN;
    };
    if py_len(n1) == 0 || py_len(n2) == 0 {
        return f64::NAN;
    }
    if n1 == "-" || n2 == "-" {
        return f64::NAN;
    }
    let n1_norm = n1.trim().to_lowercase();
    let n2_norm = n2.trim().to_lowercase();
    if n1_norm == n2_norm {
        1.0
    } else {
        0.0
    }
}

fn middle_initials_overlap(name1: Option<&str>, name2: Option<&str>) -> f64 {
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

fn middle_names_equal(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let (Some(n1), Some(n2)) = (name1, name2) else {
        return f64::NAN;
    };
    if py_len(n1) == 0 || py_len(n2) == 0 {
        return f64::NAN;
    }
    if py_len(n1) == 1 || py_len(n2) == 1 {
        let (Some(c1), Some(c2)) = (n1.chars().next(), n2.chars().next()) else {
            return f64::NAN;
        };
        return if c1 == c2 { 1.0 } else { 0.0 };
    }
    if n1 == n2 {
        1.0
    } else {
        0.0
    }
}

fn middle_one_missing(name1: Option<&str>, name2: Option<&str>) -> f64 {
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

fn single_char_first(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let n1 = name1.unwrap_or("");
    let n2 = name2.unwrap_or("");
    let val = py_len(n1) == 1 || py_len(n2) == 1;
    if val {
        1.0
    } else {
        0.0
    }
}

fn single_char_middle(name1: Option<&str>, name2: Option<&str>) -> f64 {
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

fn email_parts(email: &str) -> (String, String) {
    let (prefix_raw, suffix_raw) = if let Some((before_last, after_last)) = email.rsplit_once('@') {
        let mut merged_prefix = String::with_capacity(before_last.len());
        for ch in before_last.chars() {
            if ch != '@' {
                merged_prefix.push(ch);
            }
        }
        (merged_prefix, after_last.to_string())
    } else {
        (email.to_string(), "MISSING".to_string())
    };
    let prefix = prefix_raw.trim_matches('.').to_lowercase();
    let suffix = suffix_raw.trim_matches('.').to_lowercase();
    (prefix, suffix)
}

fn email_pair_parts(
    email1: Option<&str>,
    email2: Option<&str>,
) -> Option<((String, String), (String, String))> {
    let (Some(e1), Some(e2)) = (email1, email2) else {
        return None;
    };
    if py_len(e1) == 0 || py_len(e2) == 0 {
        return None;
    }
    Some((email_parts(e1), email_parts(e2)))
}

fn year_diff(year1: Option<i64>, year2: Option<i64>) -> f64 {
    let (Some(y1_raw), Some(y2_raw)) = (year1, year2) else {
        return f64::NAN;
    };
    let y1 = y1_raw as f64;
    let y2 = y2_raw as f64;
    let diff = (y1 - y2).abs();
    diff.min(50.0)
}

fn position_diff(p1: i64, p2: i64) -> f64 {
    let diff = (p1 - p2).abs() as f64;
    diff.min(50.0)
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

fn cosine_sim_with_norms(a: &[f32], norm_a: f64, b: &[f32], norm_b: f64) -> f64 {
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let len = a.len().min(b.len());
    for i in 0..len {
        dot += (a[i] as f64) * (b[i] as f64);
    }
    dot / (norm_a * norm_b)
}

fn levenshtein_distance(a: &str, b: &str) -> usize {
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

fn levenshtein_distance_bytes(a: &[u8], b: &[u8]) -> usize {
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

fn lcs_length_bytes(a: &[u8], b: &[u8]) -> usize {
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

fn name_text_features(name1: Option<&str>, name2: Option<&str>) -> [f64; 4] {
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

const PAPER_IDX_TITLE: usize = 0;
const PAPER_IDX_HAS_ABSTRACT: usize = 1;
const PAPER_IDX_IN_SIGNATURES: usize = 2;
const PAPER_IDX_IS_RELIABLE: usize = 4;
const PAPER_IDX_PREDICTED_LANGUAGE: usize = 5;
const PAPER_IDX_TITLE_NGRAMS_WORDS: usize = 6;
const PAPER_IDX_AUTHORS: usize = 7;
const PAPER_IDX_VENUE: usize = 8;
const PAPER_IDX_JOURNAL_NAME: usize = 9;
const PAPER_IDX_TITLE_NGRAMS_CHARS: usize = 10;
const PAPER_IDX_VENUE_NGRAMS: usize = 11;
const PAPER_IDX_JOURNAL_NGRAMS: usize = 12;
const PAPER_IDX_REFERENCE_DETAILS: usize = 13;
const PAPER_IDX_YEAR: usize = 14;
const PAPER_IDX_REFERENCES: usize = 15;
const PAPER_IDX_PAPER_ID: usize = 16;
const FROM_DATASET_PAPER_PREPROCESS_CHUNK_SIZE: usize = 4096;
const PAPER_FASTPATH_REQUIRED_FIELDS: [(usize, &str); 16] = [
    (PAPER_IDX_TITLE, "title"),
    (PAPER_IDX_HAS_ABSTRACT, "has_abstract"),
    (PAPER_IDX_IN_SIGNATURES, "in_signatures"),
    (PAPER_IDX_IS_RELIABLE, "is_reliable"),
    (PAPER_IDX_PREDICTED_LANGUAGE, "predicted_language"),
    (PAPER_IDX_TITLE_NGRAMS_WORDS, "title_ngrams_words"),
    (PAPER_IDX_AUTHORS, "authors"),
    (PAPER_IDX_VENUE, "venue"),
    (PAPER_IDX_JOURNAL_NAME, "journal_name"),
    (PAPER_IDX_TITLE_NGRAMS_CHARS, "title_ngrams_chars"),
    (PAPER_IDX_VENUE_NGRAMS, "venue_ngrams"),
    (PAPER_IDX_JOURNAL_NGRAMS, "journal_ngrams"),
    (PAPER_IDX_REFERENCE_DETAILS, "reference_details"),
    (PAPER_IDX_YEAR, "year"),
    (PAPER_IDX_REFERENCES, "references"),
    (PAPER_IDX_PAPER_ID, "paper_id"),
];

const SIG_IDX_FIRST_RAW: usize = 0;
const SIG_IDX_FIRST_NORMALIZED_NO_APOSTROPHE: usize = 1;
const SIG_IDX_MIDDLE_RAW: usize = 2;
const SIG_IDX_MIDDLE_NORMALIZED_NO_APOSTROPHE: usize = 3;
const SIG_IDX_LAST_NORMALIZED: usize = 4;
const SIG_IDX_LAST_RAW: usize = 5;
const SIG_IDX_COAUTHORS: usize = 9;
const SIG_IDX_COAUTHOR_BLOCKS: usize = 10;
const SIG_IDX_AFFILIATIONS: usize = 12;
const SIG_IDX_AFFILIATIONS_NGRAMS: usize = 13;
const SIG_IDX_COAUTHOR_NGRAMS: usize = 14;
const SIG_IDX_EMAIL: usize = 15;
const SIG_IDX_ORCID: usize = 16;
const SIG_IDX_NAME_COUNTS: usize = 17;
const SIG_IDX_POSITION: usize = 18;
const SIG_IDX_PAPER_ID: usize = 23;
const FULL_FEATURE_COUNT: usize = 39;
const SIGNATURE_FASTPATH_REQUIRED_FIELDS: [(usize, &str); 13] = [
    (
        SIG_IDX_FIRST_NORMALIZED_NO_APOSTROPHE,
        "author_info_first_normalized_without_apostrophe",
    ),
    (
        SIG_IDX_MIDDLE_NORMALIZED_NO_APOSTROPHE,
        "author_info_middle_normalized_without_apostrophe",
    ),
    (SIG_IDX_LAST_NORMALIZED, "author_info_last_normalized"),
    (SIG_IDX_COAUTHORS, "author_info_coauthors"),
    (SIG_IDX_COAUTHOR_BLOCKS, "author_info_coauthor_blocks"),
    (SIG_IDX_AFFILIATIONS, "author_info_affiliations"),
    (
        SIG_IDX_AFFILIATIONS_NGRAMS,
        "author_info_affiliations_n_grams",
    ),
    (SIG_IDX_COAUTHOR_NGRAMS, "author_info_coauthor_n_grams"),
    (SIG_IDX_EMAIL, "author_info_email"),
    (SIG_IDX_ORCID, "author_info_orcid"),
    (SIG_IDX_NAME_COUNTS, "author_info_name_counts"),
    (SIG_IDX_POSITION, "author_info_position"),
    (SIG_IDX_PAPER_ID, "paper_id"),
];

impl RustFeaturizer {
    fn cluster_seeds_disallow_index(&self) -> &HashMap<String, HashSet<String>> {
        self.cluster_seeds_disallow_index.get_or_init(|| {
            let mut index: HashMap<String, HashSet<String>> = HashMap::new();
            for (left, right) in self.cluster_seeds_disallow.iter() {
                index
                    .entry(left.clone())
                    .or_insert_with(HashSet::new)
                    .insert(right.clone());
            }
            index
        })
    }

    fn cluster_seeds_disallow_contains(&self, sig_id1: &str, sig_id2: &str) -> bool {
        let (left, right) = canonical_signature_pair_ref(sig_id1, sig_id2);
        self.cluster_seeds_disallow_index()
            .get(left)
            .is_some_and(|rights| rights.contains(right))
    }

    fn featurize_pair_data(
        &self,
        s1: &SignatureData,
        s2: &SignatureData,
        p1: &PaperData,
        p2: &PaperData,
    ) -> [f64; FULL_FEATURE_COUNT] {
        let mut feats = [f64::NAN; FULL_FEATURE_COUNT];
        let mut feat_i: usize = 0;
        macro_rules! push_feat {
            ($value:expr) => {{
                feats[feat_i] = $value;
                feat_i += 1;
            }};
        }

        let first1 = s1.first.as_deref();
        let first2 = s2.first.as_deref();
        let middle1 = s1.middle.as_deref();
        let middle2 = s2.middle.as_deref();

        push_feat!(first_names_equal(first1, first2));
        push_feat!(middle_initials_overlap(middle1, middle2));
        push_feat!(middle_names_equal(middle1, middle2));
        push_feat!(middle_one_missing(middle1, middle2));
        push_feat!(single_char_first(first1, first2));
        push_feat!(single_char_middle(middle1, middle2));

        push_feat!(counter_jaccard_data(
            &s1.affiliations,
            &s2.affiliations,
            f64::INFINITY,
        ));

        let (email_prefix, email_suffix) =
            match email_pair_parts(s1.email.as_deref(), s2.email.as_deref()) {
                Some(((p1, sfx1), (p2, sfx2))) => (
                    if p1 == p2 { 1.0 } else { 0.0 },
                    if sfx1 == sfx2 { 1.0 } else { 0.0 },
                ),
                None => (f64::NAN, f64::NAN),
            };
        push_feat!(email_prefix);
        push_feat!(email_suffix);

        push_feat!(set_jaccard_data(&s1.coauthor_blocks, &s2.coauthor_blocks));
        push_feat!(counter_jaccard_data(
            &s1.coauthor_ngrams,
            &s2.coauthor_ngrams,
            5000.0,
        ));
        push_feat!(set_jaccard_data(&s1.coauthors, &s2.coauthors));

        push_feat!(counter_jaccard_data(
            &p1.venue_ngrams,
            &p2.venue_ngrams,
            f64::INFINITY,
        ));
        push_feat!(year_diff(p1.year, p2.year));

        push_feat!(counter_jaccard_data(
            &p1.title_words,
            &p2.title_words,
            f64::INFINITY,
        ));
        push_feat!(counter_jaccard_data(
            &p1.title_chars,
            &p2.title_chars,
            f64::INFINITY,
        ));

        if self.compute_reference_features && p1.ref_details_present && p2.ref_details_present {
            push_feat!(counter_jaccard_data(
                &p1.ref_authors,
                &p2.ref_authors,
                5000.0,
            ));
            push_feat!(counter_jaccard_data(
                &p1.ref_titles,
                &p2.ref_titles,
                f64::INFINITY,
            ));
            push_feat!(counter_jaccard_data(
                &p1.ref_venues,
                &p2.ref_venues,
                f64::INFINITY,
            ));
            push_feat!(counter_jaccard_data(
                &p1.ref_blocks,
                &p2.ref_blocks,
                f64::INFINITY,
            ));
            let self_cite =
                if p1.references.contains(&s2.paper_id) || p2.references.contains(&s1.paper_id) {
                    1.0
                } else {
                    0.0
                };
            push_feat!(self_cite);
            push_feat!(refs_jaccard(&p1.references, &p2.references));
        } else {
            for _ in 0..6 {
                push_feat!(f64::NAN);
            }
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

        push_feat!(position_diff(s1.position, s2.position));
        push_feat!((p1.has_abstract as i64 + p2.has_abstract as i64) as f64);
        push_feat!(english_or_unknown_count as f64);
        let same_lang = match (
            p1.predicted_language.as_deref(),
            p2.predicted_language.as_deref(),
        ) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b,
            _ => false,
        };
        push_feat!(if same_lang { 1.0 } else { 0.0 });
        push_feat!((p1.is_reliable as i64 + p2.is_reliable as i64) as f64);

        let counts = compute_name_counts_data(&s1.name_counts, &s2.name_counts);
        for value in counts.iter() {
            push_feat!(*value);
        }

        let specter_sim = if english_or_unknown_count == 2 {
            if let (Some(specter_a), Some(specter_b)) = (p1.specter.as_ref(), p2.specter.as_ref()) {
                let score = match (p1.specter_norm, p2.specter_norm) {
                    (Some(norm_a), Some(norm_b)) if specter_a.len() == specter_b.len() => {
                        cosine_sim_with_norms(specter_a, norm_a, specter_b, norm_b)
                    }
                    _ => cosine_sim_vec_f32(specter_a, specter_b),
                };
                score + 1.0
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        push_feat!(specter_sim);

        push_feat!(counter_jaccard_data(
            &p1.journal_ngrams,
            &p2.journal_ngrams,
            f64::INFINITY,
        ));

        let advanced = name_text_features(s1.adv_name.as_deref(), s2.adv_name.as_deref());
        for value in advanced.iter() {
            push_feat!(*value);
        }

        debug_assert_eq!(feat_i, FULL_FEATURE_COUNT);
        feats
    }

    fn constraint_value_from_records(
        &self,
        sig_id1: &str,
        sig_id2: &str,
        s1: &SignatureData,
        s2: &SignatureData,
        _p1: &PaperData,
        _p2: &PaperData,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        suppress_orcid: bool,
    ) -> Option<f64> {
        if !incremental_dont_use_cluster_seeds
            && self.cluster_seeds_disallow_contains(sig_id1, sig_id2)
        {
            return Some(self.cluster_seed_disallow_value);
        }

        if !incremental_dont_use_cluster_seeds {
            if let (Some(c1), Some(c2)) = (
                self.cluster_seeds_require.get(sig_id1),
                self.cluster_seeds_require.get(sig_id2),
            ) {
                if c1 == c2 {
                    return Some(self.cluster_seed_require_value);
                }
            }
        }

        if dont_merge_cluster_seeds && !incremental_dont_use_cluster_seeds {
            if let (Some(c1), Some(c2)) = (
                self.cluster_seeds_require.get(sig_id1),
                self.cluster_seeds_require.get(sig_id2),
            ) {
                if c1 != c2 {
                    return Some(self.cluster_seed_disallow_value);
                }
            }
        }

        if !suppress_orcid {
            if let (Some(o1), Some(o2)) = (s1.orcid.as_deref(), s2.orcid.as_deref()) {
                if o1 == o2 {
                    return Some(low_value);
                }
            }
        }

        let last1 = s1.last_normalized.as_deref().unwrap_or("");
        let last2 = s2.last_normalized.as_deref().unwrap_or("");
        if !lasts_equivalent_for_constraint(last1, last2) {
            return Some(high_value);
        }

        let first1 = s1.first.as_deref().unwrap_or("");
        let first2 = s2.first.as_deref().unwrap_or("");
        if !first1.is_empty() && !first2.is_empty() {
            if let (Some(c1), Some(c2)) = (first1.chars().next(), first2.chars().next()) {
                if c1 != c2 {
                    return Some(high_value);
                }
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
            return Some(high_value);
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

                let middle_1_words: HashSet<&str> = middle_1_all
                    .iter()
                    .copied()
                    .filter(|w| py_len(w) > 1)
                    .collect();
                let middle_2_words: HashSet<&str> = middle_2_all
                    .iter()
                    .copied()
                    .filter(|w| py_len(w) > 1)
                    .collect();

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
                    return Some(high_value);
                }
            }
        }
        None
    }

    fn get_constraint_value_for_pair(
        &self,
        sig_id1: &str,
        sig_id2: &str,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        suppress_orcid: bool,
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
        Ok(self.constraint_value_from_records(
            sig_id1,
            sig_id2,
            s1,
            s2,
            p1,
            p2,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
            suppress_orcid,
        ))
    }

    fn validate_constraint_pair_inputs(&self, sig_id1: &str, sig_id2: &str) -> PyResult<()> {
        let s1 = self
            .signatures
            .get(sig_id1)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string()))?;
        let s2 = self
            .signatures
            .get(sig_id2)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string()))?;
        if self.papers.get(&s1.paper_id).is_none() {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                s1.paper_id.to_string(),
            ));
        }
        if self.papers.get(&s2.paper_id).is_none() {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                s2.paper_id.to_string(),
            ));
        }
        Ok(())
    }

    fn signature_id_order(&self) -> &[String] {
        if !self.signature_ids.is_empty() {
            return self.signature_ids.as_slice();
        }
        self.cached_signature_id_order
            .get_or_init(|| {
                let mut ids: Vec<String> = self.signatures.keys().cloned().collect();
                ids.sort_unstable();
                ids
            })
            .as_slice()
    }

    fn full_feature_count(&self) -> usize {
        FULL_FEATURE_COUNT
    }

    fn signature_paper_lookup(&self) -> PyResult<Vec<(&SignatureData, &PaperData)>> {
        let signature_ids = self.signature_id_order();
        let mut lookup: Vec<(&SignatureData, &PaperData)> = Vec::with_capacity(signature_ids.len());
        for signature_id in signature_ids.iter() {
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            lookup.push((signature, paper));
        }
        Ok(lookup)
    }

    fn pair_aggregate_row_ranges(owner_row_indices: &[u32]) -> Option<Vec<PairAggregateRowRange>> {
        if owner_row_indices.is_empty() {
            return Some(Vec::new());
        }
        let mut ranges = Vec::new();
        let mut start = 0usize;
        let mut previous = owner_row_indices[0];
        for (offset, row_index) in owner_row_indices.iter().enumerate().skip(1) {
            if *row_index < previous {
                return None;
            }
            if *row_index != previous {
                ranges.push(PairAggregateRowRange {
                    row_offset: previous as usize,
                    start,
                    stop: offset,
                });
                start = offset;
                previous = *row_index;
            }
        }
        ranges.push(PairAggregateRowRange {
            row_offset: previous as usize,
            start,
            stop: owner_row_indices.len(),
        });
        Some(ranges)
    }

    fn empty_pair_aggregate_buffers(
        row_count: usize,
        aggregate_cols: usize,
    ) -> PairAggregateBuffers {
        PairAggregateBuffers {
            counts: vec![0_u32; row_count],
            sums: vec![0.0_f64; row_count * aggregate_cols],
            mins: vec![f64::INFINITY; row_count * aggregate_cols],
            maxs: vec![f64::NEG_INFINITY; row_count * aggregate_cols],
        }
    }

    fn aggregate_pair_index_arrays_grouped(
        &self,
        left_indices: &[u32],
        right_indices: &[u32],
        row_ranges: &[PairAggregateRowRange],
        row_count: usize,
        aggregate_indices: &[usize],
        nan_value: f64,
        lookup: &[(&SignatureData, &PaperData)],
    ) -> PairAggregateBuffers {
        let aggregate_cols = aggregate_indices.len();
        let mut out = Self::empty_pair_aggregate_buffers(row_count, aggregate_cols);
        if aggregate_cols == 0 {
            for range in row_ranges.iter() {
                out.counts[range.row_offset] =
                    (range.stop - range.start).min(u32::MAX as usize) as u32;
            }
            return out;
        }

        let group_count = row_ranges.len();
        let mut group_counts = vec![0_u32; group_count];
        let mut group_sums = vec![0.0_f64; group_count * aggregate_cols];
        let mut group_mins = vec![f64::INFINITY; group_count * aggregate_cols];
        let mut group_maxs = vec![f64::NEG_INFINITY; group_count * aggregate_cols];
        group_counts
            .par_iter_mut()
            .zip(group_sums.par_chunks_mut(aggregate_cols))
            .zip(group_mins.par_chunks_mut(aggregate_cols))
            .zip(group_maxs.par_chunks_mut(aggregate_cols))
            .zip(row_ranges.par_iter())
            .for_each(|((((count, sums_row), mins_row), maxs_row), range)| {
                for pair_offset in range.start..range.stop {
                    *count = count.saturating_add(1);
                    let (s1, p1) = lookup[left_indices[pair_offset] as usize];
                    let (s2, p2) = lookup[right_indices[pair_offset] as usize];
                    let row = self.featurize_pair_data(s1, s2, p1, p2);
                    for (aggregate_position, feature_index) in aggregate_indices.iter().enumerate()
                    {
                        let mut value = row[*feature_index];
                        if value.is_nan() && !nan_value.is_nan() {
                            value = nan_value;
                        }
                        sums_row[aggregate_position] += value;
                        if value < mins_row[aggregate_position] {
                            mins_row[aggregate_position] = value;
                        }
                        if value > maxs_row[aggregate_position] {
                            maxs_row[aggregate_position] = value;
                        }
                    }
                }
            });

        for (group_offset, range) in row_ranges.iter().enumerate() {
            out.counts[range.row_offset] = group_counts[group_offset];
            let source_start = group_offset * aggregate_cols;
            let target_start = range.row_offset * aggregate_cols;
            out.sums[target_start..target_start + aggregate_cols]
                .copy_from_slice(&group_sums[source_start..source_start + aggregate_cols]);
            out.mins[target_start..target_start + aggregate_cols]
                .copy_from_slice(&group_mins[source_start..source_start + aggregate_cols]);
            out.maxs[target_start..target_start + aggregate_cols]
                .copy_from_slice(&group_maxs[source_start..source_start + aggregate_cols]);
        }
        out
    }

    fn aggregate_pair_index_arrays_sequential(
        &self,
        left_indices: &[u32],
        right_indices: &[u32],
        owner_row_indices: &[u32],
        row_count: usize,
        aggregate_indices: &[usize],
        nan_value: f64,
        lookup: &[(&SignatureData, &PaperData)],
    ) -> PairAggregateBuffers {
        let aggregate_cols = aggregate_indices.len();
        let mut out = Self::empty_pair_aggregate_buffers(row_count, aggregate_cols);
        if aggregate_cols == 0 {
            for row_index in owner_row_indices.iter() {
                out.counts[*row_index as usize] = out.counts[*row_index as usize].saturating_add(1);
            }
            return out;
        }

        for (pair_offset, row_index) in owner_row_indices.iter().enumerate() {
            let row_offset = *row_index as usize;
            out.counts[row_offset] = out.counts[row_offset].saturating_add(1);
            let aggregate_row_start = row_offset * aggregate_cols;
            let (s1, p1) = lookup[left_indices[pair_offset] as usize];
            let (s2, p2) = lookup[right_indices[pair_offset] as usize];
            let row = self.featurize_pair_data(s1, s2, p1, p2);
            for (aggregate_position, feature_index) in aggregate_indices.iter().enumerate() {
                let mut value = row[*feature_index];
                if value.is_nan() && !nan_value.is_nan() {
                    value = nan_value;
                }
                let stats_index = aggregate_row_start + aggregate_position;
                out.sums[stats_index] += value;
                if value < out.mins[stats_index] {
                    out.mins[stats_index] = value;
                }
                if value > out.maxs[stats_index] {
                    out.maxs[stats_index] = value;
                }
            }
        }
        out
    }

    fn update_top5_distance(row: &mut [f64], value: f64) {
        if value >= row[4] {
            return;
        }
        row[4] = value;
        row.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    }
}

#[pymethods]
impl RustFeaturizer {
    #[classattr]
    const SUPPORTS_FROM_DATASET_PAPER_PREPROCESS: bool = true;

    #[staticmethod]
    #[pyo3(signature = (dataset, cluster_seed_require_value = 0.0, cluster_seed_disallow_value = 10000.0, num_threads = None))]
    fn from_dataset(
        py: Python<'_>,
        dataset: &Bound<'_, PyAny>,
        cluster_seed_require_value: f64,
        cluster_seed_disallow_value: f64,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let compute_reference_features: bool = dataset
            .getattr("compute_reference_features")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let preprocess: bool = dataset
            .getattr("preprocess")
            .and_then(|v| v.extract())
            .unwrap_or(true);

        let text_module = py.import("s2and.text")?;
        let unidecode = text_module.getattr("unidecode")?;
        let stop_words = extract_required_string_set(&text_module.getattr("STOPWORDS")?)?;
        let venue_stop_words =
            extract_required_string_set(&text_module.getattr("VENUE_STOP_WORDS")?)?;
        let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
        let affiliation_stopwords = extract_affiliation_stopwords(py)?;
        let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
        let mut language_detector: Option<LanguageDetectorCompat> = None;

        #[derive(Clone)]
        struct PaperInput {
            paper_id: PaperId,
            raw_title: String,
            raw_venue: String,
            raw_journal_name: String,
            raw_authors: Vec<(i64, String)>,
            need_title_words: bool,
            need_title_chars: bool,
            need_venue_ngrams: bool,
            need_journal_ngrams: bool,
            need_author_normalization: bool,
        }

        #[derive(Clone)]
        struct PaperComputed {
            paper_id: PaperId,
            normalized_authors: Vec<(i64, String)>,
            title_words: Option<CounterData>,
            title_chars: Option<CounterData>,
            venue_ngrams: Option<CounterData>,
            journal_ngrams: Option<CounterData>,
        }

        let papers_obj = dataset.getattr("papers")?;
        let papers_dict = papers_obj.downcast::<PyDict>()?;
        let use_paper_tuple_fastpath = validate_dict_namedtuple_fastpath_contract(
            &papers_dict,
            &PAPER_FASTPATH_REQUIRED_FIELDS,
            "Paper",
        )?;
        let specter_obj = dataset.getattr("specter_embeddings").ok();
        let specter_dict = specter_obj
            .as_ref()
            .and_then(|v| v.downcast::<PyDict>().ok());

        let mut papers = HashMap::with_capacity(papers_dict.len());
        let mut paper_authors_by_id: HashMap<PaperId, Vec<(i64, String)>> =
            HashMap::with_capacity(papers_dict.len());
        let mut paper_inputs: Vec<PaperInput> = Vec::new();
        for (_paper_id_obj, paper_obj) in papers_dict.iter() {
            let paper_id = extract_id_string(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_PAPER_ID,
                "paper_id",
            )?)?;
            let raw_title = extract_string_opt(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_TITLE,
                "title",
            )?)?
            .unwrap_or_default();
            let raw_venue = extract_string_opt(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_VENUE,
                "venue",
            )?)?
            .unwrap_or_default();
            let raw_journal_name = extract_string_opt(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_JOURNAL_NAME,
                "journal_name",
            )?)?
            .unwrap_or_default();
            let in_signatures: Option<bool> = get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_IN_SIGNATURES,
                "in_signatures",
            )?
            .extract()?;
            let in_signatures = in_signatures.unwrap_or(false);
            let venue_ngrams = extract_counter(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_VENUE_NGRAMS,
                "venue_ngrams",
            )?)?;
            let title_words = extract_counter(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_TITLE_NGRAMS_WORDS,
                "title_ngrams_words",
            )?)?;
            let title_chars = extract_counter(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_TITLE_NGRAMS_CHARS,
                "title_ngrams_chars",
            )?)?;
            let journal_ngrams = extract_counter(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_JOURNAL_NGRAMS,
                "journal_ngrams",
            )?)?;
            let paper_authors =
                extract_paper_authors_with_positions(&get_namedtuple_item_or_attr(
                    &paper_obj,
                    use_paper_tuple_fastpath,
                    PAPER_IDX_AUTHORS,
                    "authors",
                )?)?;

            let ref_details_present;
            let mut ref_authors = None;
            let mut ref_titles = None;
            let mut ref_venues = None;
            let mut ref_blocks = None;
            if compute_reference_features {
                let ref_details_obj = get_namedtuple_item_or_attr(
                    &paper_obj,
                    use_paper_tuple_fastpath,
                    PAPER_IDX_REFERENCE_DETAILS,
                    "reference_details",
                )?;
                ref_details_present = !ref_details_obj.is_none();
                if ref_details_present {
                    (ref_authors, ref_titles, ref_venues, ref_blocks) =
                        extract_reference_details_counters(py, &ref_details_obj)?;
                }
            } else {
                ref_details_present = false;
            }

            let references = extract_set_id_string(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_REFERENCES,
                "references",
            )?)?;
            let year: Option<i64> = get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_YEAR,
                "year",
            )?
            .extract()?;
            let year = match year {
                Some(v) if v > 0 => Some(v),
                _ => None,
            };
            let has_abstract: bool = get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_HAS_ABSTRACT,
                "has_abstract",
            )?
            .extract()?;
            let mut predicted_language = extract_string_opt(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_PREDICTED_LANGUAGE,
                "predicted_language",
            )?)?;
            let is_reliable_raw: Option<bool> = get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_IS_RELIABLE,
                "is_reliable",
            )?
            .extract()?;
            let mut is_reliable = is_reliable_raw.unwrap_or(false);

            let need_title_words = title_words.is_none();
            let need_title_chars = preprocess && in_signatures && title_chars.is_none();
            let need_venue_ngrams = preprocess && in_signatures && venue_ngrams.is_none();
            let need_journal_ngrams = preprocess && in_signatures && journal_ngrams.is_none();
            let need_language = in_signatures && predicted_language.is_none();
            if need_language {
                if language_detector.is_none() {
                    language_detector = Some(LanguageDetectorCompat::new(py));
                }
                let detector = language_detector.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("missing language detector")
                })?;
                let (reliable, _is_english, language) = detector.detect(&raw_title);
                predicted_language = Some(language);
                is_reliable = reliable;
            }

            let need_author_normalization = need_title_words
                || need_title_chars
                || need_venue_ngrams
                || need_journal_ngrams
                || need_language;
            if need_author_normalization {
                ensure_unidecode_for_text(&unidecode, &raw_title, &mut unidecode_char_map)?;
                if preprocess {
                    ensure_unidecode_for_text(&unidecode, &raw_venue, &mut unidecode_char_map)?;
                    ensure_unidecode_for_text(
                        &unidecode,
                        &raw_journal_name,
                        &mut unidecode_char_map,
                    )?;
                }
                for (_, author_name) in paper_authors.iter() {
                    ensure_unidecode_for_text(&unidecode, author_name, &mut unidecode_char_map)?;
                }
                paper_inputs.push(PaperInput {
                    paper_id: paper_id.clone(),
                    raw_title,
                    raw_venue,
                    raw_journal_name,
                    raw_authors: paper_authors.clone(),
                    need_title_words,
                    need_title_chars,
                    need_venue_ngrams,
                    need_journal_ngrams,
                    need_author_normalization,
                });
            }
            paper_authors_by_id.insert(paper_id.clone(), paper_authors);

            let specter = if let Some(spec_dict) = &specter_dict {
                extract_specter_for_paper_id(spec_dict, &paper_id)?
            } else {
                None
            };
            let specter_norm = specter.as_ref().map(|values| {
                values
                    .iter()
                    .map(|value| {
                        let value_f64 = *value as f64;
                        value_f64 * value_f64
                    })
                    .sum::<f64>()
                    .sqrt()
            });

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
                    specter_norm,
                    ref_details_present,
                },
            );
        }

        for paper_input_chunk in paper_inputs.chunks(FROM_DATASET_PAPER_PREPROCESS_CHUNK_SIZE) {
            let computed_chunk: Vec<PaperComputed> = py.allow_threads(|| {
                let compute = || {
                    paper_input_chunk
                        .par_iter()
                        .map(|paper_input| {
                            let normalized_title = normalize_text_compat_from_map(
                                &paper_input.raw_title,
                                false,
                                &unidecode_char_map,
                            );
                            let normalized_venue = if paper_input.need_venue_ngrams {
                                Some(normalize_text_compat_from_map(
                                    &paper_input.raw_venue,
                                    false,
                                    &unidecode_char_map,
                                ))
                            } else {
                                None
                            };
                            let normalized_journal = if paper_input.need_journal_ngrams {
                                Some(normalize_text_compat_from_map(
                                    &paper_input.raw_journal_name,
                                    false,
                                    &unidecode_char_map,
                                ))
                            } else {
                                None
                            };
                            let normalized_authors = if paper_input.need_author_normalization {
                                paper_input
                                    .raw_authors
                                    .iter()
                                    .map(|(position, raw_name)| {
                                        (
                                            *position,
                                            normalize_text_compat_from_map(
                                                raw_name,
                                                false,
                                                &unidecode_char_map,
                                            ),
                                        )
                                    })
                                    .collect()
                            } else {
                                paper_input.raw_authors.clone()
                            };
                            let title_words = if paper_input.need_title_words {
                                counter_data_from_usize_map(word_ngrams_counter_python_compat(
                                    &normalized_title,
                                    &stop_words,
                                ))
                            } else {
                                None
                            };
                            let title_chars = if paper_input.need_title_chars {
                                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                    &normalized_title,
                                    false,
                                    true,
                                    Some(&stop_words),
                                ))
                            } else {
                                None
                            };
                            let venue_ngrams = if paper_input.need_venue_ngrams {
                                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                    normalized_venue.as_deref().unwrap_or(""),
                                    false,
                                    true,
                                    Some(&venue_stop_words),
                                ))
                            } else {
                                None
                            };
                            let journal_ngrams = if paper_input.need_journal_ngrams {
                                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                    normalized_journal.as_deref().unwrap_or(""),
                                    false,
                                    true,
                                    Some(&venue_stop_words),
                                ))
                            } else {
                                None
                            };
                            PaperComputed {
                                paper_id: paper_input.paper_id.clone(),
                                normalized_authors,
                                title_words,
                                title_chars,
                                venue_ngrams,
                                journal_ngrams,
                            }
                        })
                        .collect::<Vec<_>>()
                };
                install_with_optional_rayon_pool(num_threads, compute)
            });
            for computed in computed_chunk {
                let PaperComputed {
                    paper_id,
                    normalized_authors,
                    title_words,
                    title_chars,
                    venue_ngrams,
                    journal_ngrams,
                } = computed;
                if let Some(paper) = papers.get_mut(&paper_id) {
                    if let Some(counter) = title_words {
                        paper.title_words = Some(counter);
                    }
                    if let Some(counter) = title_chars {
                        paper.title_chars = Some(counter);
                    }
                    if let Some(counter) = venue_ngrams {
                        paper.venue_ngrams = Some(counter);
                    }
                    if let Some(counter) = journal_ngrams {
                        paper.journal_ngrams = Some(counter);
                    }
                }
                paper_authors_by_id.insert(paper_id, normalized_authors);
            }
        }

        let signatures_obj = dataset.getattr("signatures")?;
        let signatures_dict = signatures_obj.downcast::<PyDict>()?;
        let use_signature_tuple_fastpath = validate_dict_namedtuple_fastpath_contract(
            &signatures_dict,
            &SIGNATURE_FASTPATH_REQUIRED_FIELDS,
            "Signature",
        )?;
        let mut signature_rows: Vec<(String, SignatureData, Option<String>, Option<String>)> =
            Vec::with_capacity(signatures_dict.len());
        for (sig_id_obj, sig_obj) in signatures_dict.iter() {
            let sig_id: String = sig_id_obj.extract()?;
            let raw_first = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_FIRST_RAW,
                "author_info_first",
            )?)?
            .unwrap_or_default();
            let raw_middle = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_MIDDLE_RAW,
                "author_info_middle",
            )?)?
            .unwrap_or_default();
            let raw_last = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_LAST_RAW,
                "author_info_last",
            )?)?
            .unwrap_or_default();

            let mut first = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_FIRST_NORMALIZED_NO_APOSTROPHE,
                "author_info_first_normalized_without_apostrophe",
            )?)?;
            let mut middle = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_MIDDLE_NORMALIZED_NO_APOSTROPHE,
                "author_info_middle_normalized_without_apostrophe",
            )?)?;
            let mut last_normalized = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_LAST_NORMALIZED,
                "author_info_last_normalized",
            )?)?;
            if first.is_none() || middle.is_none() || last_normalized.is_none() {
                ensure_unidecode_for_text(&unidecode, &raw_first, &mut unidecode_char_map)?;
                ensure_unidecode_for_text(&unidecode, &raw_middle, &mut unidecode_char_map)?;
                ensure_unidecode_for_text(&unidecode, &raw_last, &mut unidecode_char_map)?;

                let (first_without_apostrophe, middle_without_apostrophe) =
                    split_first_middle_hyphen_aware_compat(
                        &raw_first,
                        &raw_middle,
                        &name_prefixes,
                        &unidecode_char_map,
                    );
                if first.is_none() {
                    first = Some(first_without_apostrophe);
                }
                if middle.is_none() {
                    middle = Some(middle_without_apostrophe);
                }
                if last_normalized.is_none() {
                    last_normalized = Some(normalize_text_compat_from_map(
                        &raw_last,
                        false,
                        &unidecode_char_map,
                    ));
                }
            }

            let raw_orcid = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_ORCID,
                "author_info_orcid",
            )?)?;
            let orcid = if let Some(value) = raw_orcid {
                let upper_value = value.to_ascii_uppercase();
                extract_orcid_from_source_id(&value)
                    .or_else(|| extract_orcid_from_source_id(&upper_value))
                    .or(Some(value))
            } else {
                None
            };
            let email = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_EMAIL,
                "author_info_email",
            )?)?;
            let affiliations = extract_counter(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_AFFILIATIONS_NGRAMS,
                "author_info_affiliations_n_grams",
            )?)?;
            let mut coauthor_blocks = extract_optional_string_set(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_COAUTHOR_BLOCKS,
                "author_info_coauthor_blocks",
            )?)?;
            let coauthor_ngrams = extract_counter(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_COAUTHOR_NGRAMS,
                "author_info_coauthor_n_grams",
            )?)?;
            let mut coauthors = extract_optional_string_set(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_COAUTHORS,
                "author_info_coauthors",
            )?)?;
            let position: i64 = get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_POSITION,
                "author_info_position",
            )?
            .extract()?;
            let paper_id = extract_id_string(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_PAPER_ID,
                "paper_id",
            )?)?;
            let name_counts = extract_name_counts_data(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_NAME_COUNTS,
                "author_info_name_counts",
            )?)?;

            let mut coauthor_text_for_compute: Option<String> = None;
            let mut affiliation_text_for_compute: Option<String> = None;

            let need_coauthor_from_papers =
                coauthor_ngrams.is_none() || coauthors.is_none() || coauthor_blocks.is_none();
            if need_coauthor_from_papers {
                if let Some(paper_authors) = paper_authors_by_id.get(&paper_id) {
                    let coauthor_names: Vec<String> = paper_authors
                        .iter()
                        .filter(|(author_position, _)| *author_position != position)
                        .map(|(_, author_name)| author_name.clone())
                        .collect();
                    if !coauthor_names.is_empty() {
                        if coauthor_ngrams.is_none() {
                            coauthor_text_for_compute = Some(coauthor_names.join(" "));
                        }
                        if coauthors.is_none() {
                            coauthors = Some(coauthor_names.iter().cloned().collect());
                        }
                        if coauthor_blocks.is_none() {
                            let mut coauthor_blocks_set: HashSet<String> =
                                HashSet::with_capacity(coauthor_names.len());
                            for coauthor in coauthor_names.iter() {
                                coauthor_blocks_set.insert(compute_block_compat(coauthor));
                            }
                            if !coauthor_blocks_set.is_empty() {
                                coauthor_blocks = Some(coauthor_blocks_set);
                            }
                        }
                    }
                }
            }

            if affiliations.is_none() {
                let affiliation_list = extract_string_list(&get_namedtuple_item_or_attr(
                    &sig_obj,
                    use_signature_tuple_fastpath,
                    SIG_IDX_AFFILIATIONS,
                    "author_info_affiliations",
                )?)?;
                let mut normalized_affiliation_list: Vec<String> =
                    Vec::with_capacity(affiliation_list.len());
                for affiliation in affiliation_list.iter() {
                    ensure_unidecode_for_text(&unidecode, affiliation, &mut unidecode_char_map)?;
                    normalized_affiliation_list.push(normalize_text_compat_from_map(
                        affiliation,
                        false,
                        &unidecode_char_map,
                    ));
                }
                let prefiltered = prefilter_affiliation_text(
                    &normalized_affiliation_list,
                    &affiliation_stopwords,
                );
                if !prefiltered.is_empty() {
                    affiliation_text_for_compute = Some(prefiltered);
                }
            }
            let adv_name = first.clone();

            signature_rows.push((
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
                coauthor_text_for_compute,
                affiliation_text_for_compute,
            ));
        }

        let computed_rows = py.allow_threads(|| {
            let compute = || {
                signature_rows
                    .into_par_iter()
                    .map(
                        |(
                            sig_id,
                            mut signature,
                            coauthor_text_for_compute,
                            affiliation_text_for_compute,
                        )| {
                            if signature.coauthor_ngrams.is_none() {
                                if let Some(coauthor_text) = coauthor_text_for_compute.as_deref() {
                                    signature.coauthor_ngrams = counter_data_from_usize_map(
                                        char_ngrams_counter(coauthor_text),
                                    );
                                }
                            }
                            if signature.affiliations.is_none() {
                                if let Some(affiliation_text) =
                                    affiliation_text_for_compute.as_deref()
                                {
                                    signature.affiliations = counter_data_from_usize_map(
                                        word_ngrams_counter(affiliation_text),
                                    );
                                }
                            }
                            (sig_id, signature)
                        },
                    )
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let mut signatures = HashMap::with_capacity(computed_rows.len());
        for (sig_id, signature) in computed_rows {
            signatures.insert(sig_id, signature);
        }
        let mut signature_ids: Vec<String> = signatures.keys().cloned().collect();
        signature_ids.sort_unstable();

        let name_tuples = extract_name_tuples_map(&dataset.getattr("name_tuples")?)?;
        let cluster_seeds_disallow = extract_pair_set(&dataset.getattr("cluster_seeds_disallow")?)?;
        let cluster_seeds_require =
            extract_cluster_seeds_require(&dataset.getattr("cluster_seeds_require")?)?;

        Ok(RustFeaturizer {
            signatures,
            signature_ids,
            papers,
            name_tuples,
            cluster_seeds_disallow,
            cluster_seeds_require,
            compute_reference_features,
            cluster_seed_require_value,
            cluster_seed_disallow_value,
            cached_signature_id_order: OnceLock::new(),
            cluster_seeds_disallow_index: OnceLock::new(),
        })
    }

    #[staticmethod]
    #[pyo3(
        signature = (
            signatures_path,
            papers_path,
            cluster_seeds_path = None,
            specter_embeddings = None,
            name_tuples_path = None,
            name_counts_path = None,
            preprocess = true,
            compute_reference_features = false,
            cluster_seed_require_value = 0.0,
            cluster_seed_disallow_value = 10000.0,
            num_threads = None,
            expected_normalization_version = None,
            allow_normalization_version_mismatch = false
        )
    )]
    fn from_json_paths(
        py: Python<'_>,
        signatures_path: &str,
        papers_path: &str,
        cluster_seeds_path: Option<&str>,
        specter_embeddings: Option<&Bound<'_, PyAny>>,
        name_tuples_path: Option<&str>,
        name_counts_path: Option<&str>,
        preprocess: bool,
        compute_reference_features: bool,
        cluster_seed_require_value: f64,
        cluster_seed_disallow_value: f64,
        num_threads: Option<usize>,
        expected_normalization_version: Option<&str>,
        allow_normalization_version_mismatch: bool,
    ) -> PyResult<Self> {
        reset_last_json_ingest_telemetry_internal();

        let text_module = py.import("s2and.text")?;
        let unidecode = text_module.getattr("unidecode")?;
        let stop_words_obj = text_module.getattr("STOPWORDS")?;
        let venue_stop_words_obj = text_module.getattr("VENUE_STOP_WORDS")?;
        let name_prefixes_obj = text_module.getattr("NAME_PREFIXES")?;

        let stop_words = extract_required_string_set(&stop_words_obj)?;
        let venue_stop_words = extract_required_string_set(&venue_stop_words_obj)?;
        let name_prefixes = extract_required_string_set(&name_prefixes_obj)?;

        let language_detector = if preprocess {
            Some(LanguageDetectorCompat::new(py))
        } else {
            None
        };

        let json_parse_start = Instant::now();
        let signatures_json = load_json_value(signatures_path)?;
        let signatures_obj = json_as_object(&signatures_json, "signatures payload")?;
        let raw_name_counts = load_raw_name_counts_from_json_path(
            name_counts_path,
            expected_normalization_version,
            allow_normalization_version_mismatch,
        )?;
        let name_tuples = load_name_tuples_from_text_path(py, name_tuples_path)?;
        let affiliation_stopwords = extract_affiliation_stopwords(py)?;

        #[derive(Clone)]
        struct SignatureInput {
            sig_id: String,
            paper_id: PaperId,
            raw_first: String,
            raw_middle: String,
            raw_last: String,
            email: Option<String>,
            position: i64,
            affiliation_values: Vec<String>,
            source_id_source: Option<String>,
            source_ids: Vec<String>,
        }

        #[derive(Clone)]
        struct PaperInput {
            paper_id: PaperId,
            raw_title: String,
            raw_venue: String,
            raw_journal: String,
            raw_authors: Vec<(i64, String)>,
            references: HashSet<PaperId>,
            year: Option<i64>,
            has_abstract: bool,
            predicted_language: Option<String>,
            is_reliable: bool,
        }

        let paper_preprocess_start = Instant::now();
        let mut needed_paper_ids: HashSet<PaperId> = HashSet::new();
        let mut signature_inputs: Vec<SignatureInput> = Vec::with_capacity(signatures_obj.len());
        let mut paper_inputs: Vec<PaperInput> = Vec::new();
        let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
        let mut defaulted_signature_author_position_count = 0usize;
        for (sig_id, sig_value) in signatures_obj.iter() {
            let sig_dict = json_as_object(sig_value, "signature entry")?;
            let paper_id_value = json_get_required(sig_dict, "paper_id", "signature entry")?;
            let paper_id = json_value_to_id(paper_id_value).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("signature paper_id must be string/int")
            })?;
            needed_paper_ids.insert(paper_id.clone());

            let author_info_value = json_get_required(sig_dict, "author_info", "signature entry")?;
            let author_info = json_as_object(author_info_value, "signature author_info")?;
            let raw_first = json_get_string(author_info, "first", "");
            let raw_middle = json_get_string(author_info, "middle", "");
            let raw_last = json_get_string(author_info, "last", "");
            let email = json_get_optional_string(author_info, "email");
            let position = match json_get_i64_optional(author_info, "position") {
                Some(value) => value,
                None => {
                    defaulted_signature_author_position_count += 1;
                    0
                }
            };
            let affiliation_values = json_get_string_list(author_info.get("affiliations"));
            let source_id_source = json_get_optional_string(author_info, "source_id_source");
            let source_ids = if source_id_source.as_deref() == Some("ORCID") {
                json_get_string_list(author_info.get("source_ids"))
            } else {
                Vec::new()
            };

            ensure_unidecode_for_text(&unidecode, &raw_first, &mut unidecode_char_map)?;
            ensure_unidecode_for_text(&unidecode, &raw_middle, &mut unidecode_char_map)?;
            ensure_unidecode_for_text(&unidecode, &raw_last, &mut unidecode_char_map)?;
            for affiliation in affiliation_values.iter() {
                ensure_unidecode_for_text(&unidecode, affiliation, &mut unidecode_char_map)?;
            }

            signature_inputs.push(SignatureInput {
                sig_id: sig_id.clone(),
                paper_id,
                raw_first,
                raw_middle,
                raw_last,
                email,
                position,
                affiliation_values,
                source_id_source,
                source_ids,
            });
        }
        drop(signatures_json);

        #[derive(Clone)]
        struct PaperPreprocessed {
            title: String,
            venue: String,
            journal_name: String,
            authors: Vec<(i64, String)>,
            references: HashSet<PaperId>,
            year: Option<i64>,
            has_abstract: bool,
            predicted_language: Option<String>,
            is_reliable: bool,
            title_words: Option<CounterData>,
            title_chars: Option<CounterData>,
            venue_ngrams: Option<CounterData>,
            journal_ngrams: Option<CounterData>,
        }

        let papers_json = load_json_value(papers_path)?;
        let papers_obj = json_as_object(&papers_json, "papers payload")?;
        let mut defaulted_paper_author_position_count = 0usize;
        for (_paper_key, paper_value) in papers_obj.iter() {
            let paper_dict = json_as_object(paper_value, "paper entry")?;
            let paper_id_value = json_get_required(paper_dict, "paper_id", "paper entry")?;
            let paper_id = json_value_to_id(paper_id_value).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("paper paper_id must be string/int")
            })?;
            if !needed_paper_ids.contains(&paper_id) {
                continue;
            }

            let raw_title = json_get_string(paper_dict, "title", "");
            let raw_venue = json_get_string(paper_dict, "venue", "");
            let raw_journal = json_get_string(paper_dict, "journal_name", "");

            if preprocess {
                ensure_unidecode_for_text(&unidecode, &raw_title, &mut unidecode_char_map)?;
                ensure_unidecode_for_text(&unidecode, &raw_venue, &mut unidecode_char_map)?;
                ensure_unidecode_for_text(&unidecode, &raw_journal, &mut unidecode_char_map)?;
            }

            let mut raw_authors: Vec<(i64, String)> = Vec::new();
            if let Some(author_values) = paper_dict.get("authors").and_then(JsonValue::as_array) {
                for author_value in author_values {
                    let Some(author_dict) = author_value.as_object() else {
                        continue;
                    };
                    let position = match json_get_i64_optional(author_dict, "position") {
                        Some(value) => value,
                        None => {
                            defaulted_paper_author_position_count += 1;
                            0
                        }
                    };
                    let raw_author_name = json_get_string(author_dict, "author_name", "");
                    if preprocess {
                        ensure_unidecode_for_text(
                            &unidecode,
                            &raw_author_name,
                            &mut unidecode_char_map,
                        )?;
                    }
                    raw_authors.push((position, raw_author_name));
                }
            }

            let references = json_get_id_set(paper_dict.get("references"));

            let year = match json_get_i64_optional(paper_dict, "year") {
                Some(v) if v > 0 => Some(v),
                _ => None,
            };

            let has_abstract = match paper_dict.get("abstract") {
                None | Some(JsonValue::Null) => false,
                Some(JsonValue::String(s)) => !s.is_empty(),
                Some(_) => true,
            };

            let (is_reliable, predicted_language) = if preprocess {
                let detector = language_detector.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("missing language detector")
                })?;
                let (reliable, _is_english, language) = detector.detect(&raw_title);
                (reliable, Some(language))
            } else {
                (false, None)
            };

            paper_inputs.push(PaperInput {
                paper_id,
                raw_title,
                raw_venue,
                raw_journal,
                raw_authors,
                references,
                year,
                has_abstract,
                predicted_language,
                is_reliable,
            });
        }
        drop(needed_paper_ids);
        drop(papers_json);
        let json_parse_seconds = json_parse_start.elapsed().as_secs_f64();

        let computed_papers = py.allow_threads(|| {
            let compute = || {
                paper_inputs
                    .par_iter()
                    .map(|paper_input| {
                        let title = if preprocess {
                            normalize_text_compat_from_map(
                                &paper_input.raw_title,
                                false,
                                &unidecode_char_map,
                            )
                        } else {
                            paper_input.raw_title.clone()
                        };
                        let venue = if preprocess {
                            normalize_text_compat_from_map(
                                &paper_input.raw_venue,
                                false,
                                &unidecode_char_map,
                            )
                        } else {
                            paper_input.raw_venue.clone()
                        };
                        let journal_name = if preprocess {
                            normalize_text_compat_from_map(
                                &paper_input.raw_journal,
                                false,
                                &unidecode_char_map,
                            )
                        } else {
                            paper_input.raw_journal.clone()
                        };
                        let authors = if preprocess {
                            paper_input
                                .raw_authors
                                .iter()
                                .map(|(position, raw_name)| {
                                    (
                                        *position,
                                        normalize_text_compat_from_map(
                                            raw_name,
                                            false,
                                            &unidecode_char_map,
                                        ),
                                    )
                                })
                                .collect::<Vec<_>>()
                        } else {
                            paper_input.raw_authors.clone()
                        };

                        let title_words = if preprocess {
                            counter_data_from_usize_map(word_ngrams_counter_python_compat(
                                &title,
                                &stop_words,
                            ))
                        } else {
                            None
                        };

                        let title_chars = if preprocess {
                            counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                &title,
                                false,
                                true,
                                Some(&stop_words),
                            ))
                        } else {
                            None
                        };

                        let venue_ngrams = if preprocess {
                            counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                &venue,
                                false,
                                true,
                                Some(&venue_stop_words),
                            ))
                        } else {
                            None
                        };

                        let journal_ngrams = if preprocess {
                            counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                &journal_name,
                                false,
                                true,
                                Some(&venue_stop_words),
                            ))
                        } else {
                            None
                        };

                        (
                            paper_input.paper_id.clone(),
                            PaperPreprocessed {
                                title,
                                venue,
                                journal_name,
                                authors,
                                references: paper_input.references.clone(),
                                year: paper_input.year,
                                has_abstract: paper_input.has_abstract,
                                predicted_language: paper_input.predicted_language.clone(),
                                is_reliable: paper_input.is_reliable,
                                title_words,
                                title_chars,
                                venue_ngrams,
                                journal_ngrams,
                            },
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        drop(paper_inputs);

        let mut preprocessed_papers: HashMap<PaperId, PaperPreprocessed> =
            HashMap::with_capacity(computed_papers.len());
        for (paper_id, preprocessed) in computed_papers {
            preprocessed_papers.insert(paper_id, preprocessed);
        }
        let paper_preprocess_seconds = paper_preprocess_start.elapsed().as_secs_f64();

        let signature_preprocess_start = Instant::now();
        let signature_inputs: Vec<SignatureInput> = signature_inputs
            .into_iter()
            .filter(|entry| preprocessed_papers.contains_key(&entry.paper_id))
            .collect();
        let computed_signatures = py.allow_threads(|| {
            let compute = || {
                signature_inputs
                    .par_iter()
                    .map(|entry| {
                        let middle_normalized = normalize_text_compat_from_map(
                            &entry.raw_middle,
                            false,
                            &unidecode_char_map,
                        );
                        let first_normalized = normalize_text_compat_from_map(
                            &entry.raw_first,
                            false,
                            &unidecode_char_map,
                        );
                        let mut first_middle_split: Vec<String> =
                            format!("{} {}", first_normalized, middle_normalized)
                                .split_whitespace()
                                .map(|token| token.to_string())
                                .collect();
                        if let Some(prefix) = first_middle_split.first() {
                            if name_prefixes.contains(prefix) {
                                first_middle_split.remove(0);
                            }
                        }
                        let first_normalized_token =
                            first_middle_split.get(0).cloned().unwrap_or_default();
                        let (first_without_apostrophe, middle_without_apostrophe) =
                            split_first_middle_hyphen_aware_compat(
                                &entry.raw_first,
                                &entry.raw_middle,
                                &name_prefixes,
                                &unidecode_char_map,
                            );
                        let last_normalized = normalize_text_compat_from_map(
                            &entry.raw_last,
                            false,
                            &unidecode_char_map,
                        );

                        let mut coauthor_list: Vec<String> = Vec::new();
                        if let Some(preprocessed_paper) = preprocessed_papers.get(&entry.paper_id) {
                            for (author_position, author_name) in preprocessed_paper.authors.iter()
                            {
                                if *author_position != entry.position {
                                    coauthor_list.push(author_name.clone());
                                }
                            }
                        }
                        let coauthors = if coauthor_list.is_empty() {
                            None
                        } else {
                            Some(coauthor_list.iter().cloned().collect::<HashSet<String>>())
                        };

                        let mut coauthor_blocks_set: HashSet<String> = HashSet::new();
                        for coauthor in coauthor_list.iter() {
                            coauthor_blocks_set.insert(compute_block_compat(coauthor));
                        }
                        let coauthor_blocks = if coauthor_blocks_set.is_empty() {
                            None
                        } else {
                            Some(coauthor_blocks_set)
                        };

                        let normalized_affiliations: Vec<String> = if preprocess {
                            entry
                                .affiliation_values
                                .iter()
                                .map(|affiliation| {
                                    normalize_text_compat_from_map(
                                        affiliation,
                                        false,
                                        &unidecode_char_map,
                                    )
                                })
                                .collect()
                        } else {
                            entry.affiliation_values.clone()
                        };

                        let affiliation_text = if preprocess {
                            prefilter_affiliation_text(
                                &normalized_affiliations,
                                &affiliation_stopwords,
                            )
                        } else {
                            String::new()
                        };
                        let coauthor_text = if preprocess {
                            coauthor_list.join(" ")
                        } else {
                            String::new()
                        };

                        let affiliations = if preprocess && !affiliation_text.is_empty() {
                            counter_data_from_usize_map(word_ngrams_counter(&affiliation_text))
                        } else {
                            None
                        };
                        let coauthor_ngrams = if preprocess && !coauthor_text.is_empty() {
                            counter_data_from_usize_map(char_ngrams_counter(&coauthor_text))
                        } else {
                            None
                        };

                        let normalized_orcid = if entry.source_id_source.as_deref() == Some("ORCID")
                        {
                            entry
                                .source_ids
                                .first()
                                .and_then(|source_id| extract_orcid_from_source_id(source_id))
                        } else {
                            None
                        };

                        let name_counts_result = build_name_counts_data_from_artifact(
                            &raw_name_counts,
                            &entry.raw_first,
                            &first_normalized_token,
                            &first_without_apostrophe,
                            &entry.raw_last,
                            &last_normalized,
                        );

                        (
                            entry.sig_id.clone(),
                            SignatureData {
                                first: Some(first_without_apostrophe.clone()),
                                middle: Some(middle_without_apostrophe),
                                last_normalized: Some(last_normalized),
                                orcid: normalized_orcid,
                                email: entry.email.clone(),
                                affiliations,
                                coauthor_blocks,
                                coauthor_ngrams,
                                coauthors,
                                position: entry.position,
                                paper_id: entry.paper_id.clone(),
                                name_counts: name_counts_result.data,
                                adv_name: Some(first_without_apostrophe),
                            },
                            name_counts_result.telemetry,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        let mut signatures: HashMap<String, SignatureData> =
            HashMap::with_capacity(computed_signatures.len());
        let mut defaulted_name_count_signature_count = 0usize;
        let mut defaulted_name_count_first_count = 0usize;
        let mut defaulted_name_count_first_last_count = 0usize;
        let mut defaulted_name_count_last_count = 0usize;
        let mut defaulted_name_count_last_first_initial_count = 0usize;
        for (sig_id, signature, name_count_telemetry) in computed_signatures {
            if name_count_telemetry.any() {
                defaulted_name_count_signature_count += 1;
            }
            if name_count_telemetry.first {
                defaulted_name_count_first_count += 1;
            }
            if name_count_telemetry.first_last {
                defaulted_name_count_first_last_count += 1;
            }
            if name_count_telemetry.last {
                defaulted_name_count_last_count += 1;
            }
            if name_count_telemetry.last_first_initial {
                defaulted_name_count_last_first_initial_count += 1;
            }
            signatures.insert(sig_id, signature);
        }
        drop(signature_inputs);
        drop(unidecode_char_map);
        let mut signature_ids: Vec<String> = signatures.keys().cloned().collect();
        signature_ids.sort_unstable();
        let signature_preprocess_seconds = signature_preprocess_start.elapsed().as_secs_f64();

        let specter_dict = match specter_embeddings {
            Some(obj) => {
                if let Ok(path) = obj.extract::<&str>() {
                    load_pickle_dict(py, path)?
                } else if let Ok(dict) = obj.downcast::<PyDict>() {
                    Some(dict.clone())
                } else if obj.is_none() {
                    None
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "specter_embeddings must be str, dict, or None; got {}",
                        obj.get_type().name()?
                    )));
                }
            }
            None => None,
        };

        let reference_counter_start = Instant::now();
        let mut missing_specter_paper_count = 0usize;
        let mut papers: HashMap<PaperId, PaperData> =
            HashMap::with_capacity(preprocessed_papers.len());
        if compute_reference_features {
            for (paper_id, paper) in preprocessed_papers.iter() {
                let mut ref_authors = None;
                let mut ref_titles = None;
                let mut ref_venues = None;
                let mut ref_blocks = None;
                let ref_details_present = true;

                let mut titles: Vec<String> = Vec::new();
                let mut venues: Vec<String> = Vec::new();
                let mut journals: Vec<String> = Vec::new();
                let mut authors: Vec<String> = Vec::new();
                let mut blocks: Vec<String> = Vec::new();

                for reference_id in paper.references.iter() {
                    if let Some(reference_paper) = preprocessed_papers.get(reference_id) {
                        if !reference_paper.title.is_empty() {
                            titles.push(reference_paper.title.clone());
                        }
                        if !reference_paper.venue.is_empty() {
                            venues.push(reference_paper.venue.clone());
                        }
                        if !reference_paper.journal_name.is_empty() {
                            journals.push(reference_paper.journal_name.clone());
                        }
                        for (_, author_name) in reference_paper.authors.iter() {
                            if author_name.is_empty() {
                                continue;
                            }
                            authors.push(author_name.clone());
                            let block = compute_block_compat(author_name);
                            blocks.push(block);
                        }
                    }
                }

                let author_names = authors.join(" ");
                let reference_titles = titles.join(" ");
                let venues_joined = venues.join(" ");
                let journals_joined = journals.join(" ");
                let reference_venues = if venues_joined == journals_joined {
                    venues_joined
                } else {
                    format!("{} {}", venues_joined, journals_joined)
                        .trim()
                        .to_string()
                };

                if !author_names.is_empty() {
                    ref_authors = counter_data_from_usize_map(char_ngrams_counter_python_compat(
                        &author_names,
                        false,
                        true,
                        Some(&stop_words),
                    ));
                }

                if !reference_titles.is_empty() {
                    ref_titles = counter_data_from_usize_map(char_ngrams_counter_python_compat(
                        &reference_titles,
                        false,
                        true,
                        Some(&stop_words),
                    ));
                }

                if !reference_venues.is_empty() {
                    ref_venues = counter_data_from_usize_map(char_ngrams_counter_python_compat(
                        &reference_venues,
                        false,
                        true,
                        Some(&venue_stop_words),
                    ));
                }

                if !blocks.is_empty() {
                    let mut block_counter: HashMap<String, usize> = HashMap::new();
                    for block in blocks {
                        *block_counter.entry(block).or_insert(0) += 1;
                    }
                    ref_blocks = counter_data_from_usize_map(block_counter);
                }
                let specter = if let Some(spec_dict) = &specter_dict {
                    extract_specter_for_paper_id(spec_dict, paper_id)?
                } else {
                    None
                };
                if specter.is_none() {
                    missing_specter_paper_count += 1;
                }
                let specter_norm = specter.as_ref().map(|values| {
                    values
                        .iter()
                        .map(|value| {
                            let value_f64 = *value as f64;
                            value_f64 * value_f64
                        })
                        .sum::<f64>()
                        .sqrt()
                });

                papers.insert(
                    paper_id.clone(),
                    PaperData {
                        venue_ngrams: paper.venue_ngrams.clone(),
                        title_words: paper.title_words.clone(),
                        title_chars: paper.title_chars.clone(),
                        ref_authors,
                        ref_titles,
                        ref_venues,
                        ref_blocks,
                        ref_details_present,
                        references: paper.references.clone(),
                        year: paper.year,
                        has_abstract: paper.has_abstract,
                        predicted_language: paper.predicted_language.clone(),
                        is_reliable: paper.is_reliable,
                        journal_ngrams: paper.journal_ngrams.clone(),
                        specter,
                        specter_norm,
                    },
                );
            }
        } else {
            for (paper_id, paper) in preprocessed_papers.into_iter() {
                let specter = if let Some(spec_dict) = &specter_dict {
                    extract_specter_for_paper_id(spec_dict, &paper_id)?
                } else {
                    None
                };
                if specter.is_none() {
                    missing_specter_paper_count += 1;
                }
                let specter_norm = specter.as_ref().map(|values| {
                    values
                        .iter()
                        .map(|value| {
                            let value_f64 = *value as f64;
                            value_f64 * value_f64
                        })
                        .sum::<f64>()
                        .sqrt()
                });
                papers.insert(
                    paper_id,
                    PaperData {
                        venue_ngrams: paper.venue_ngrams,
                        title_words: paper.title_words,
                        title_chars: paper.title_chars,
                        ref_authors: None,
                        ref_titles: None,
                        ref_venues: None,
                        ref_blocks: None,
                        ref_details_present: false,
                        references: paper.references,
                        year: paper.year,
                        has_abstract: paper.has_abstract,
                        predicted_language: paper.predicted_language,
                        is_reliable: paper.is_reliable,
                        journal_ngrams: paper.journal_ngrams,
                        specter,
                        specter_norm,
                    },
                );
            }
        }
        let reference_counter_seconds = reference_counter_start.elapsed().as_secs_f64();

        let cluster_seed_start = Instant::now();
        let mut cluster_seeds_disallow: HashSet<(String, String)> = HashSet::new();
        let mut cluster_seeds_require: HashMap<String, ClusterId> = HashMap::new();
        if let Some(path) = cluster_seeds_path {
            let cluster_seeds_json = load_json_value(path)?;
            let cluster_seeds_obj = json_as_object(&cluster_seeds_json, "cluster_seeds payload")?;
            let mut cluster_num = 0_i64;
            for (signature_id_a, values_value) in cluster_seeds_obj.iter() {
                let values_obj = json_as_object(values_value, "cluster seed entry")?;
                let mut root_added = false;
                for (signature_id_b, constraint_value) in values_obj.iter() {
                    let Some(constraint) = json_value_to_string(constraint_value) else {
                        continue;
                    };
                    if constraint == "disallow" {
                        cluster_seeds_disallow.insert(canonical_signature_pair_cloned(
                            signature_id_a,
                            signature_id_b,
                        ));
                    } else if constraint == "require" {
                        if !root_added {
                            cluster_seeds_require
                                .insert(signature_id_a.clone(), ClusterId::Int(cluster_num));
                            root_added = true;
                        }
                        cluster_seeds_require
                            .insert(signature_id_b.clone(), ClusterId::Int(cluster_num));
                    }
                }
                cluster_num += 1;
            }
        }
        let cluster_seed_seconds = cluster_seed_start.elapsed().as_secs_f64();

        set_last_json_ingest_telemetry(JsonIngestTelemetry {
            json_parse_seconds,
            paper_preprocess_seconds,
            reference_counter_seconds,
            signature_preprocess_seconds,
            cluster_seed_seconds,
            missing_specter_paper_count,
            defaulted_name_count_signature_count,
            defaulted_name_count_first_count,
            defaulted_name_count_first_last_count,
            defaulted_name_count_last_count,
            defaulted_name_count_last_first_initial_count,
            defaulted_signature_author_position_count,
            defaulted_paper_author_position_count,
        });

        Ok(RustFeaturizer {
            signatures,
            signature_ids,
            papers,
            name_tuples,
            cluster_seeds_disallow,
            cluster_seeds_require,
            compute_reference_features,
            cluster_seed_require_value,
            cluster_seed_disallow_value,
            cached_signature_id_order: OnceLock::new(),
            cluster_seeds_disallow_index: OnceLock::new(),
        })
    }

    fn update_cluster_seeds(
        &mut self,
        cluster_seeds_require: &Bound<'_, PyAny>,
        cluster_seeds_disallow: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.cluster_seeds_require = extract_cluster_seeds_require(cluster_seeds_require)?;
        self.cluster_seeds_disallow = extract_pair_set(cluster_seeds_disallow)?;
        self.cluster_seeds_disallow_index = OnceLock::new();
        Ok(())
    }

    #[pyo3(
        signature = (
            sig_id1,
            sig_id2,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            suppress_orcid = false
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
        suppress_orcid: bool,
    ) -> PyResult<Option<f64>> {
        self.get_constraint_value_for_pair(
            sig_id1,
            sig_id2,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
            suppress_orcid,
        )
    }

    #[pyo3(
        signature = (
            pairs,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None,
            suppress_orcid = false
        )
    )]
    fn get_constraints_matrix(
        &self,
        py: Python<'_>,
        pairs: Vec<(String, String)>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
        suppress_orcid: bool,
    ) -> PyResult<Vec<Option<f64>>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        for (sig_id1, sig_id2) in pairs.iter() {
            self.validate_constraint_pair_inputs(sig_id1, sig_id2)?;
        }

        let values = py.allow_threads(|| {
            let compute = || {
                pairs
                    .par_iter()
                    .map(|(sig_id1, sig_id2)| -> PyResult<Option<f64>> {
                        let s1 = self.signatures.get(sig_id1).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string())
                        })?;
                        let s2 = self.signatures.get(sig_id2).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string())
                        })?;
                        let p1 = self.papers.get(&s1.paper_id).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(s1.paper_id.to_string())
                        })?;
                        let p2 = self.papers.get(&s2.paper_id).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(s2.paper_id.to_string())
                        })?;
                        Ok(self.constraint_value_from_records(
                            sig_id1,
                            sig_id2,
                            s1,
                            s2,
                            p1,
                            p2,
                            low_value,
                            high_value,
                            dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds,
                            suppress_orcid,
                        ))
                    })
                    .collect::<PyResult<Vec<_>>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        values
    }

    #[pyo3(
        signature = (
            pairs,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None,
            suppress_orcid = false
        )
    )]
    fn get_constraints_matrix_indexed(
        &self,
        py: Python<'_>,
        pairs: Vec<(u32, u32)>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
        suppress_orcid: bool,
    ) -> PyResult<Vec<Option<f64>>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        for (left_idx, right_idx) in pairs.iter() {
            let left = *left_idx as usize;
            let right = *right_idx as usize;
            if left >= signature_count || right >= signature_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "pair index out of range: left={} right={} signature_count={}",
                    left, right, signature_count
                )));
            }
        }

        let mut lookup: Vec<(&String, &SignatureData, &PaperData)> =
            Vec::with_capacity(signature_count);
        for signature_id in signature_ids.iter() {
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            lookup.push((signature_id, signature, paper));
        }

        let values = py.allow_threads(|| {
            let compute = || {
                pairs
                    .par_iter()
                    .map(|(left_idx, right_idx)| {
                        let (left_id, s1, p1) = lookup[*left_idx as usize];
                        let (right_id, s2, p2) = lookup[*right_idx as usize];
                        self.constraint_value_from_records(
                            left_id,
                            right_id,
                            s1,
                            s2,
                            p1,
                            p2,
                            low_value,
                            high_value,
                            dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds,
                            suppress_orcid,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        Ok(values)
    }

    #[pyo3(
        signature = (
            left_signature_indices,
            right_signature_indices,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None,
            suppress_orcid = false,
            large_integer = 100000.0
        )
    )]
    fn linker_pair_index_arrays_constraint_labels<'py>(
        &self,
        py: Python<'py>,
        left_signature_indices: PyReadonlyArray1<'py, u32>,
        right_signature_indices: PyReadonlyArray1<'py, u32>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
        suppress_orcid: bool,
        large_integer: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let left_indices = left_signature_indices.as_slice()?;
        let right_indices = right_signature_indices.as_slice()?;
        let pair_count = left_indices.len();
        if right_indices.len() != pair_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "left_signature_indices and right_signature_indices must have equal length: left={} right={}",
                left_indices.len(),
                right_indices.len()
            )));
        }

        let signature_ids = self.signature_id_order();
        for (left_idx, right_idx) in left_indices.iter().zip(right_indices.iter()) {
            let left = *left_idx as usize;
            let right = *right_idx as usize;
            if left >= signature_ids.len() || right >= signature_ids.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "pair index out of range: left={} right={} signature_count={}",
                    left,
                    right,
                    signature_ids.len()
                )));
            }
        }

        let lookup = self.signature_paper_lookup()?;
        let labels = py.allow_threads(|| {
            let compute = || {
                left_indices
                    .par_iter()
                    .zip(right_indices.par_iter())
                    .map(|(left_idx, right_idx)| {
                        let left = *left_idx as usize;
                        let right = *right_idx as usize;
                        let sig_id1 = signature_ids[left].as_str();
                        let sig_id2 = signature_ids[right].as_str();
                        let (s1, p1) = lookup[left];
                        let (s2, p2) = lookup[right];
                        match self.constraint_value_from_records(
                            sig_id1,
                            sig_id2,
                            s1,
                            s2,
                            p1,
                            p2,
                            low_value,
                            high_value,
                            dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds,
                            suppress_orcid,
                        ) {
                            Some(value) => value - large_integer,
                            None => f64::NAN,
                        }
                    })
                    .collect::<Vec<f64>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        Ok(numpy::ndarray::Array1::from_vec(labels).to_pyarray(py))
    }

    #[pyo3(
        signature = (
            row_indices,
            row_count,
            pair_distances,
            pair_labels = None,
            num_threads = None,
            large_integer = 100000.0,
            hard_disallow_distance = 10000.0
        )
    )]
    fn linker_pair_distance_accumulators<'py>(
        &self,
        py: Python<'py>,
        row_indices: PyReadonlyArray1<'py, u32>,
        row_count: usize,
        pair_distances: PyReadonlyArray1<'py, f64>,
        pair_labels: Option<PyReadonlyArray1<'py, f64>>,
        num_threads: Option<usize>,
        large_integer: f64,
        hard_disallow_distance: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        u64,
    )> {
        let _ = num_threads;
        let owner_row_indices = row_indices.as_slice()?;
        let model_distances = pair_distances.as_slice()?;
        let pair_count = owner_row_indices.len();
        if model_distances.len() != pair_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "row_indices and pair_distances must have equal length: rows={} distances={}",
                owner_row_indices.len(),
                model_distances.len()
            )));
        }
        let labels = match pair_labels.as_ref() {
            Some(values) => {
                let slice = values.as_slice()?;
                if slice.len() != pair_count {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "pair_labels length must match row_indices length: labels={} rows={}",
                        slice.len(),
                        pair_count
                    )));
                }
                Some(slice)
            }
            None => None,
        };
        for row_index in owner_row_indices.iter() {
            let bounded = *row_index as usize;
            if bounded >= row_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "row index out of range: row_index={} row_count={}",
                    bounded, row_count
                )));
            }
        }

        let mut counts = vec![0_u32; row_count];
        let mut sums = vec![0.0_f64; row_count];
        let mut mins = vec![f64::INFINITY; row_count];
        let mut top_distances = vec![f64::INFINITY; row_count * 5];
        let mut hard_disallow_pair_count = 0_u64;

        for pair_offset in 0..pair_count {
            let label = labels.map(|values| values[pair_offset]).unwrap_or(f64::NAN);
            let value = if label.is_nan() {
                model_distances[pair_offset]
            } else {
                label + large_integer
            };
            if value.is_nan() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "pairwise model returned NaN distance",
                ));
            }
            let row = owner_row_indices[pair_offset] as usize;
            counts[row] = counts[row].saturating_add(1);
            sums[row] += value;
            if value < mins[row] {
                mins[row] = value;
            }
            if value >= hard_disallow_distance {
                hard_disallow_pair_count = hard_disallow_pair_count.saturating_add(1);
            }
            let top_start = row * 5;
            Self::update_top5_distance(&mut top_distances[top_start..top_start + 5], value);
        }

        let top_array = numpy::ndarray::Array2::from_shape_vec((row_count, 5), top_distances)
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build top-distance matrix: {}",
                    err
                ))
            })?;
        Ok((
            numpy::ndarray::Array1::from_vec(counts).to_pyarray(py),
            numpy::ndarray::Array1::from_vec(sums).to_pyarray(py),
            numpy::ndarray::Array1::from_vec(mins).to_pyarray(py),
            top_array.to_pyarray(py),
            hard_disallow_pair_count,
        ))
    }

    #[pyo3(
        signature = (
            block_signature_indices,
            start_offset = 0,
            max_pairs = None,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None,
            suppress_orcid = false
        )
    )]
    fn get_constraints_block_upper_triangle_indexed(
        &self,
        py: Python<'_>,
        block_signature_indices: Vec<u32>,
        start_offset: usize,
        max_pairs: Option<usize>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
        suppress_orcid: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<Option<f64>>)> {
        if block_signature_indices.len() <= 1 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            if global_idx >= signature_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "block signature index out of range: index={} signature_count={}",
                    global_idx, signature_count
                )));
            }
        }

        let mut block_lookup: Vec<(&String, &SignatureData, &PaperData)> =
            Vec::with_capacity(block_signature_indices.len());
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            let signature_id = &signature_ids[global_idx];
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            block_lookup.push((signature_id, signature, paper));
        }

        let local_pairs =
            upper_triangle_pairs_for_range(block_lookup.len(), start_offset, max_pairs)?;
        if local_pairs.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let left_indices: Vec<u32> = local_pairs.iter().map(|(left, _)| *left as u32).collect();
        let right_indices: Vec<u32> = local_pairs.iter().map(|(_, right)| *right as u32).collect();
        let values = py.allow_threads(|| {
            let compute = || {
                local_pairs
                    .par_iter()
                    .map(|(left_idx, right_idx)| {
                        let (left_id, s1, p1) = block_lookup[*left_idx];
                        let (right_id, s2, p2) = block_lookup[*right_idx];
                        self.constraint_value_from_records(
                            left_id,
                            right_id,
                            s1,
                            s2,
                            p1,
                            p2,
                            low_value,
                            high_value,
                            dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds,
                            suppress_orcid,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        Ok((left_indices, right_indices, values))
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
        Ok(self.featurize_pair_data(s1, s2, p1, p2).to_vec())
    }

    #[pyo3(signature = (pairs, num_threads = None))]
    fn featurize_pairs(
        &self,
        py: Python<'_>,
        pairs: Vec<(String, String)>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<Vec<f64>>> {
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
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    s1.paper_id.to_string(),
                ));
            }
            if self.papers.get(&s2.paper_id).is_none() {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    s2.paper_id.to_string(),
                ));
            }
        }
        let feats = py.allow_threads(|| {
            let compute = || {
                pairs
                    .par_iter()
                    .map(|(sig_id1, sig_id2)| -> PyResult<Vec<f64>> {
                        let s1 = self.signatures.get(sig_id1).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string())
                        })?;
                        let s2 = self.signatures.get(sig_id2).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string())
                        })?;
                        let p1 = self.papers.get(&s1.paper_id).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(s1.paper_id.to_string())
                        })?;
                        let p2 = self.papers.get(&s2.paper_id).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(s2.paper_id.to_string())
                        })?;
                        Ok(self.featurize_pair_data(s1, s2, p1, p2).to_vec())
                    })
                    .collect::<PyResult<Vec<_>>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        feats
    }

    fn signature_ids(&self) -> Vec<String> {
        self.signature_id_order().to_vec()
    }

    fn update_signature_name_counts(&mut self, signatures: &Bound<'_, PyAny>) -> PyResult<usize> {
        let signatures_dict = signatures.downcast::<PyDict>()?;
        let mut updated = 0usize;
        for (sig_id_obj, sig_obj) in signatures_dict.iter() {
            let sig_id: String = sig_id_obj.extract()?;
            let Some(signature) = self.signatures.get_mut(&sig_id) else {
                continue;
            };
            let counts_obj = sig_obj.getattr("author_info_name_counts")?;
            let counts = extract_name_counts_data(&counts_obj)?;
            if counts.is_some() {
                signature.name_counts = counts;
                updated += 1;
            }
        }
        Ok(updated)
    }

    #[pyo3(signature = (pairs, selected_indices = None, num_threads = None, nan_value = f64::NAN))]
    fn featurize_pairs_matrix<'py>(
        &self,
        py: Python<'py>,
        pairs: Vec<(String, String)>,
        selected_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let row_count = pairs.len();
        if row_count == 0 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

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
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    s1.paper_id.to_string(),
                ));
            }
            if self.papers.get(&s2.paper_id).is_none() {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    s2.paper_id.to_string(),
                ));
            }
        }

        let mut id_to_lookup_idx: HashMap<&str, usize> =
            HashMap::with_capacity(row_count.saturating_mul(2));
        let mut lookup: Vec<(&SignatureData, &PaperData)> = Vec::new();
        for (sig_id1, sig_id2) in pairs.iter() {
            for sig_id in [sig_id1.as_str(), sig_id2.as_str()] {
                if id_to_lookup_idx.contains_key(sig_id) {
                    continue;
                }
                let signature = self
                    .signatures
                    .get(sig_id)
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id.to_string()))?;
                let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
                })?;
                let lookup_idx = lookup.len();
                lookup.push((signature, paper));
                id_to_lookup_idx.insert(sig_id, lookup_idx);
            }
        }

        let mut indexed_pairs: Vec<(usize, usize)> = Vec::with_capacity(row_count);
        for (sig_id1, sig_id2) in pairs.iter() {
            let left_idx = *id_to_lookup_idx
                .get(sig_id1.as_str())
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id1.clone()))?;
            let right_idx = *id_to_lookup_idx
                .get(sig_id2.as_str())
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id2.clone()))?;
            indexed_pairs.push((left_idx, right_idx));
        }

        let full_cols = self.full_feature_count();
        let indices: Vec<usize> = selected_indices.unwrap_or_else(|| (0..full_cols).collect());
        for idx in indices.iter() {
            if *idx >= full_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "selected_indices contains out-of-range index {} for {} columns",
                    idx, full_cols
                )));
            }
        }
        let out_cols = indices.len();
        if out_cols == 0 {
            let empty_cols = numpy::ndarray::Array2::<f64>::zeros((row_count, 0));
            return Ok(empty_cols.to_pyarray(py));
        }
        let out = py.allow_threads(|| {
            let compute = || {
                let mut buffer = vec![0.0_f64; row_count * out_cols];
                buffer
                    .par_chunks_mut(out_cols)
                    .zip(indexed_pairs.par_iter())
                    .for_each(|(out_row, (left_idx, right_idx))| {
                        let (s1, p1) = lookup[*left_idx];
                        let (s2, p2) = lookup[*right_idx];
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (dest, idx) in out_row.iter_mut().zip(indices.iter()) {
                            let mut value = row[*idx];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            *dest = value;
                        }
                    });
                buffer
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let array =
            numpy::ndarray::Array2::from_shape_vec((row_count, out_cols), out).map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build output matrix: {}",
                    err
                ))
            })?;
        Ok(array.to_pyarray(py))
    }

    #[pyo3(signature = (pairs, selected_indices = None, num_threads = None, nan_value = f64::NAN))]
    fn featurize_pairs_matrix_indexed<'py>(
        &self,
        py: Python<'py>,
        pairs: Vec<(u32, u32)>,
        selected_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let row_count = pairs.len();
        if row_count == 0 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

        let signature_ids = self.signature_id_order();
        for (left_idx, right_idx) in pairs.iter() {
            let left = *left_idx as usize;
            let right = *right_idx as usize;
            if left >= signature_ids.len() || right >= signature_ids.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "pair index out of range: left={} right={} signature_count={}",
                    left,
                    right,
                    signature_ids.len()
                )));
            }
        }

        let mut lookup: Vec<(&SignatureData, &PaperData)> = Vec::with_capacity(signature_ids.len());
        for signature_id in signature_ids.iter() {
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            lookup.push((signature, paper));
        }

        let full_cols = self.full_feature_count();
        let indices: Vec<usize> = selected_indices.unwrap_or_else(|| (0..full_cols).collect());
        for idx in indices.iter() {
            if *idx >= full_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "selected_indices contains out-of-range index {} for {} columns",
                    idx, full_cols
                )));
            }
        }
        let out_cols = indices.len();
        if out_cols == 0 {
            let empty_cols = numpy::ndarray::Array2::<f64>::zeros((row_count, 0));
            return Ok(empty_cols.to_pyarray(py));
        }

        let out = py.allow_threads(|| {
            let compute = || {
                let mut buffer = vec![0.0_f64; row_count * out_cols];
                buffer
                    .par_chunks_mut(out_cols)
                    .zip(pairs.par_iter())
                    .for_each(|(out_row, (left_idx, right_idx))| {
                        let (s1, p1) = lookup[*left_idx as usize];
                        let (s2, p2) = lookup[*right_idx as usize];
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (dest, idx) in out_row.iter_mut().zip(indices.iter()) {
                            let mut value = row[*idx];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            *dest = value;
                        }
                    });
                buffer
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let array =
            numpy::ndarray::Array2::from_shape_vec((row_count, out_cols), out).map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build output matrix: {}",
                    err
                ))
            })?;
        Ok(array.to_pyarray(py))
    }

    #[pyo3(
        signature = (
            pairs,
            row_indices,
            row_count,
            matrix_indices = None,
            aggregate_indices = None,
            num_threads = None,
            nan_value = f64::NAN
        )
    )]
    fn linker_pair_features_and_aggregate_stats_indexed<'py>(
        &self,
        py: Python<'py>,
        pairs: Vec<(u32, u32)>,
        row_indices: Vec<u32>,
        row_count: usize,
        matrix_indices: Option<Vec<usize>>,
        aggregate_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let pair_count = pairs.len();
        if row_indices.len() != pair_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "row_indices length must match pairs length: row_indices={} pairs={}",
                row_indices.len(),
                pair_count
            )));
        }

        let full_cols = self.full_feature_count();
        let resolved_matrix_indices: Vec<usize> =
            matrix_indices.unwrap_or_else(|| (0..full_cols).collect());
        for idx in resolved_matrix_indices.iter() {
            if *idx >= full_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "matrix_indices contains out-of-range index {} for {} columns",
                    idx, full_cols
                )));
            }
        }

        let resolved_aggregate_indices: Vec<usize> =
            aggregate_indices.unwrap_or_else(|| resolved_matrix_indices.clone());
        for idx in resolved_aggregate_indices.iter() {
            if *idx >= full_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "aggregate_indices contains out-of-range index {} for {} columns",
                    idx, full_cols
                )));
            }
        }

        let matrix_position_by_feature: HashMap<usize, usize> = resolved_matrix_indices
            .iter()
            .enumerate()
            .map(|(position, feature_index)| (*feature_index, position))
            .collect();
        let mut aggregate_matrix_positions: Vec<usize> =
            Vec::with_capacity(resolved_aggregate_indices.len());
        for feature_index in resolved_aggregate_indices.iter() {
            let Some(matrix_position) = matrix_position_by_feature.get(feature_index) else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "aggregate index {} is not present in matrix_indices; include it to avoid recomputation",
                    feature_index
                )));
            };
            aggregate_matrix_positions.push(*matrix_position);
        }

        let signature_ids = self.signature_id_order();
        for (left_idx, right_idx) in pairs.iter() {
            let left = *left_idx as usize;
            let right = *right_idx as usize;
            if left >= signature_ids.len() || right >= signature_ids.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "pair index out of range: left={} right={} signature_count={}",
                    left,
                    right,
                    signature_ids.len()
                )));
            }
        }
        for row_index in row_indices.iter() {
            let bounded = *row_index as usize;
            if bounded >= row_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "row index out of range: row_index={} row_count={}",
                    bounded, row_count
                )));
            }
        }

        let mut lookup: Vec<(&SignatureData, &PaperData)> = Vec::with_capacity(signature_ids.len());
        for signature_id in signature_ids.iter() {
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            lookup.push((signature, paper));
        }

        let out_cols = resolved_matrix_indices.len();
        let aggregate_cols = resolved_aggregate_indices.len();
        let matrix_buffer = py.allow_threads(|| {
            let compute = || {
                let mut buffer = vec![0.0_f64; pair_count * out_cols];
                if out_cols == 0 {
                    return buffer;
                }
                buffer
                    .par_chunks_mut(out_cols)
                    .zip(pairs.par_iter())
                    .for_each(|(out_row, (left_idx, right_idx))| {
                        let (s1, p1) = lookup[*left_idx as usize];
                        let (s2, p2) = lookup[*right_idx as usize];
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (dest, idx) in out_row.iter_mut().zip(resolved_matrix_indices.iter()) {
                            let mut value = row[*idx];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            *dest = value;
                        }
                    });
                buffer
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let mut counts = vec![0_u32; row_count];
        let mut sums = vec![0.0_f64; row_count * aggregate_cols];
        let mut mins = vec![f64::INFINITY; row_count * aggregate_cols];
        let mut maxs = vec![f64::NEG_INFINITY; row_count * aggregate_cols];
        if aggregate_cols > 0 {
            for (pair_offset, row_index) in row_indices.iter().enumerate() {
                let row_offset = *row_index as usize;
                counts[row_offset] = counts[row_offset].saturating_add(1);
                let matrix_row_start = pair_offset * out_cols;
                let aggregate_row_start = row_offset * aggregate_cols;
                for (aggregate_position, matrix_position) in
                    aggregate_matrix_positions.iter().enumerate()
                {
                    let value = matrix_buffer[matrix_row_start + *matrix_position];
                    let stats_index = aggregate_row_start + aggregate_position;
                    sums[stats_index] += value;
                    if value < mins[stats_index] {
                        mins[stats_index] = value;
                    }
                    if value > maxs[stats_index] {
                        maxs[stats_index] = value;
                    }
                }
            }
        } else {
            for row_index in row_indices.iter() {
                counts[*row_index as usize] = counts[*row_index as usize].saturating_add(1);
            }
        }

        let matrix_array =
            numpy::ndarray::Array2::from_shape_vec((pair_count, out_cols), matrix_buffer).map_err(
                |err| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to build pair feature matrix: {}",
                        err
                    ))
                },
            )?;
        let sums_array = numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), sums)
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate sums matrix: {}",
                    err
                ))
            })?;
        let mins_array = numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), mins)
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate mins matrix: {}",
                    err
                ))
            })?;
        let maxs_array = numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), maxs)
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate maxs matrix: {}",
                    err
                ))
            })?;
        Ok((
            matrix_array.to_pyarray(py),
            numpy::ndarray::Array1::from_vec(counts).to_pyarray(py),
            sums_array.to_pyarray(py),
            mins_array.to_pyarray(py),
            maxs_array.to_pyarray(py),
        ))
    }

    #[pyo3(
        signature = (
            left_signature_indices,
            right_signature_indices,
            row_indices,
            row_count,
            matrix_indices = None,
            aggregate_indices = None,
            num_threads = None,
            nan_value = f64::NAN,
            aggregate_nan_value = None
        )
    )]
    fn linker_pair_index_arrays_and_aggregate_stats<'py>(
        &self,
        py: Python<'py>,
        left_signature_indices: PyReadonlyArray1<'py, u32>,
        right_signature_indices: PyReadonlyArray1<'py, u32>,
        row_indices: PyReadonlyArray1<'py, u32>,
        row_count: usize,
        matrix_indices: Option<Vec<usize>>,
        aggregate_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
        aggregate_nan_value: Option<f64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let left_indices = left_signature_indices.as_slice()?;
        let right_indices = right_signature_indices.as_slice()?;
        let owner_row_indices = row_indices.as_slice()?;
        let pair_count = left_indices.len();
        if right_indices.len() != pair_count || owner_row_indices.len() != pair_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "left_signature_indices, right_signature_indices, and row_indices must have equal length: left={} right={} rows={}",
                left_indices.len(),
                right_indices.len(),
                owner_row_indices.len()
            )));
        }

        let full_cols = self.full_feature_count();
        let resolved_matrix_indices: Vec<usize> =
            matrix_indices.unwrap_or_else(|| (0..full_cols).collect());
        for idx in resolved_matrix_indices.iter() {
            if *idx >= full_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "matrix_indices contains out-of-range index {} for {} columns",
                    idx, full_cols
                )));
            }
        }

        let resolved_aggregate_indices: Vec<usize> =
            aggregate_indices.unwrap_or_else(|| resolved_matrix_indices.clone());
        for idx in resolved_aggregate_indices.iter() {
            if *idx >= full_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "aggregate_indices contains out-of-range index {} for {} columns",
                    idx, full_cols
                )));
            }
        }

        let matrix_position_by_feature: HashMap<usize, usize> = resolved_matrix_indices
            .iter()
            .enumerate()
            .map(|(position, feature_index)| (*feature_index, position))
            .collect();
        let mut aggregate_matrix_positions: Vec<usize> =
            Vec::with_capacity(resolved_aggregate_indices.len());
        for feature_index in resolved_aggregate_indices.iter() {
            let Some(matrix_position) = matrix_position_by_feature.get(feature_index) else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "aggregate index {} is not present in matrix_indices; include it to avoid recomputation",
                    feature_index
                )));
            };
            aggregate_matrix_positions.push(*matrix_position);
        }

        let signature_ids = self.signature_id_order();
        for (left_idx, right_idx) in left_indices.iter().zip(right_indices.iter()) {
            let left = *left_idx as usize;
            let right = *right_idx as usize;
            if left >= signature_ids.len() || right >= signature_ids.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "pair index out of range: left={} right={} signature_count={}",
                    left,
                    right,
                    signature_ids.len()
                )));
            }
        }
        for row_index in owner_row_indices.iter() {
            let bounded = *row_index as usize;
            if bounded >= row_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "row index out of range: row_index={} row_count={}",
                    bounded, row_count
                )));
            }
        }

        let mut lookup: Vec<(&SignatureData, &PaperData)> = Vec::with_capacity(signature_ids.len());
        for signature_id in signature_ids.iter() {
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            lookup.push((signature, paper));
        }

        let out_cols = resolved_matrix_indices.len();
        let aggregate_cols = resolved_aggregate_indices.len();
        let resolved_aggregate_nan_value = aggregate_nan_value.unwrap_or(nan_value);
        let matrix_buffer = py.allow_threads(|| {
            let compute = || {
                let mut buffer = vec![0.0_f64; pair_count * out_cols];
                if out_cols == 0 {
                    return buffer;
                }
                buffer
                    .par_chunks_mut(out_cols)
                    .zip(left_indices.par_iter().zip(right_indices.par_iter()))
                    .for_each(|(out_row, (left_idx, right_idx))| {
                        let (s1, p1) = lookup[*left_idx as usize];
                        let (s2, p2) = lookup[*right_idx as usize];
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (dest, idx) in out_row.iter_mut().zip(resolved_matrix_indices.iter()) {
                            let mut value = row[*idx];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            *dest = value;
                        }
                    });
                buffer
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let mut counts = vec![0_u32; row_count];
        let mut sums = vec![0.0_f64; row_count * aggregate_cols];
        let mut mins = vec![f64::INFINITY; row_count * aggregate_cols];
        let mut maxs = vec![f64::NEG_INFINITY; row_count * aggregate_cols];
        if aggregate_cols > 0 {
            for (pair_offset, row_index) in owner_row_indices.iter().enumerate() {
                let row_offset = *row_index as usize;
                counts[row_offset] = counts[row_offset].saturating_add(1);
                let matrix_row_start = pair_offset * out_cols;
                let aggregate_row_start = row_offset * aggregate_cols;
                for (aggregate_position, matrix_position) in
                    aggregate_matrix_positions.iter().enumerate()
                {
                    let mut value = matrix_buffer[matrix_row_start + *matrix_position];
                    if value.is_nan() && !resolved_aggregate_nan_value.is_nan() {
                        value = resolved_aggregate_nan_value;
                    }
                    let stats_index = aggregate_row_start + aggregate_position;
                    sums[stats_index] += value;
                    if value < mins[stats_index] {
                        mins[stats_index] = value;
                    }
                    if value > maxs[stats_index] {
                        maxs[stats_index] = value;
                    }
                }
            }
        } else {
            for row_index in owner_row_indices.iter() {
                counts[*row_index as usize] = counts[*row_index as usize].saturating_add(1);
            }
        }

        let matrix_array =
            numpy::ndarray::Array2::from_shape_vec((pair_count, out_cols), matrix_buffer).map_err(
                |err| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to build pair feature matrix: {}",
                        err
                    ))
                },
            )?;
        let sums_array = numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), sums)
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate sums matrix: {}",
                    err
                ))
            })?;
        let mins_array = numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), mins)
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate mins matrix: {}",
                    err
                ))
            })?;
        let maxs_array = numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), maxs)
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate maxs matrix: {}",
                    err
                ))
            })?;
        Ok((
            matrix_array.to_pyarray(py),
            numpy::ndarray::Array1::from_vec(counts).to_pyarray(py),
            sums_array.to_pyarray(py),
            mins_array.to_pyarray(py),
            maxs_array.to_pyarray(py),
        ))
    }

    #[pyo3(
        signature = (
            left_signature_indices,
            right_signature_indices,
            row_indices,
            row_count,
            aggregate_indices = None,
            num_threads = None,
            nan_value = f64::NAN
        )
    )]
    fn linker_pair_index_arrays_aggregate_stats<'py>(
        &self,
        py: Python<'py>,
        left_signature_indices: PyReadonlyArray1<'py, u32>,
        right_signature_indices: PyReadonlyArray1<'py, u32>,
        row_indices: PyReadonlyArray1<'py, u32>,
        row_count: usize,
        aggregate_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let left_indices = left_signature_indices.as_slice()?;
        let right_indices = right_signature_indices.as_slice()?;
        let owner_row_indices = row_indices.as_slice()?;
        let pair_count = left_indices.len();
        if right_indices.len() != pair_count || owner_row_indices.len() != pair_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "left_signature_indices, right_signature_indices, and row_indices must have equal length: left={} right={} rows={}",
                left_indices.len(),
                right_indices.len(),
                owner_row_indices.len()
            )));
        }

        let full_cols = self.full_feature_count();
        let resolved_aggregate_indices: Vec<usize> =
            aggregate_indices.unwrap_or_else(|| (0..full_cols).collect());
        for idx in resolved_aggregate_indices.iter() {
            if *idx >= full_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "aggregate_indices contains out-of-range index {} for {} columns",
                    idx, full_cols
                )));
            }
        }

        let signature_ids = self.signature_id_order();
        for (left_idx, right_idx) in left_indices.iter().zip(right_indices.iter()) {
            let left = *left_idx as usize;
            let right = *right_idx as usize;
            if left >= signature_ids.len() || right >= signature_ids.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "pair index out of range: left={} right={} signature_count={}",
                    left,
                    right,
                    signature_ids.len()
                )));
            }
        }
        for row_index in owner_row_indices.iter() {
            let bounded = *row_index as usize;
            if bounded >= row_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "row index out of range: row_index={} row_count={}",
                    bounded, row_count
                )));
            }
        }

        let lookup = self.signature_paper_lookup()?;
        let row_ranges = Self::pair_aggregate_row_ranges(owner_row_indices);
        let aggregate_cols = resolved_aggregate_indices.len();
        let aggregate_buffers = py.allow_threads(|| {
            let compute = || match row_ranges.as_ref() {
                Some(ranges) => self.aggregate_pair_index_arrays_grouped(
                    left_indices,
                    right_indices,
                    ranges,
                    row_count,
                    &resolved_aggregate_indices,
                    nan_value,
                    &lookup,
                ),
                None => self.aggregate_pair_index_arrays_sequential(
                    left_indices,
                    right_indices,
                    owner_row_indices,
                    row_count,
                    &resolved_aggregate_indices,
                    nan_value,
                    &lookup,
                ),
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let sums_array = numpy::ndarray::Array2::from_shape_vec(
            (row_count, aggregate_cols),
            aggregate_buffers.sums,
        )
        .map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to build aggregate sums matrix: {}",
                err
            ))
        })?;
        let mins_array = numpy::ndarray::Array2::from_shape_vec(
            (row_count, aggregate_cols),
            aggregate_buffers.mins,
        )
        .map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to build aggregate mins matrix: {}",
                err
            ))
        })?;
        let maxs_array = numpy::ndarray::Array2::from_shape_vec(
            (row_count, aggregate_cols),
            aggregate_buffers.maxs,
        )
        .map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to build aggregate maxs matrix: {}",
                err
            ))
        })?;
        Ok((
            numpy::ndarray::Array1::from_vec(aggregate_buffers.counts).to_pyarray(py),
            sums_array.to_pyarray(py),
            mins_array.to_pyarray(py),
            maxs_array.to_pyarray(py),
        ))
    }

    #[pyo3(
        signature = (
            block_signature_indices,
            start_offset = 0,
            max_pairs = None,
            selected_indices = None,
            num_threads = None,
            nan_value = f64::NAN
        )
    )]
    fn featurize_block_upper_triangle_matrix_indexed<'py>(
        &self,
        py: Python<'py>,
        block_signature_indices: Vec<u32>,
        start_offset: usize,
        max_pairs: Option<usize>,
        selected_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if block_signature_indices.len() <= 1 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            if global_idx >= signature_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "block signature index out of range: index={} signature_count={}",
                    global_idx, signature_count
                )));
            }
        }

        let mut block_lookup: Vec<(&SignatureData, &PaperData)> =
            Vec::with_capacity(block_signature_indices.len());
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            let signature_id = &signature_ids[global_idx];
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            block_lookup.push((signature, paper));
        }

        let local_pairs =
            upper_triangle_pairs_for_range(block_lookup.len(), start_offset, max_pairs)?;
        let row_count = local_pairs.len();
        if row_count == 0 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

        let full_cols = self.full_feature_count();
        let indices: Vec<usize> = selected_indices.unwrap_or_else(|| (0..full_cols).collect());
        for idx in indices.iter() {
            if *idx >= full_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "selected_indices contains out-of-range index {} for {} columns",
                    idx, full_cols
                )));
            }
        }
        let out_cols = indices.len();
        if out_cols == 0 {
            let empty_cols = numpy::ndarray::Array2::<f64>::zeros((row_count, 0));
            return Ok(empty_cols.to_pyarray(py));
        }

        let out = py.allow_threads(|| {
            let compute = || {
                let mut buffer = vec![0.0_f64; row_count * out_cols];
                buffer
                    .par_chunks_mut(out_cols)
                    .zip(local_pairs.par_iter())
                    .for_each(|(out_row, (left_idx, right_idx))| {
                        let (s1, p1) = block_lookup[*left_idx];
                        let (s2, p2) = block_lookup[*right_idx];
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (dest, idx) in out_row.iter_mut().zip(indices.iter()) {
                            let mut value = row[*idx];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            *dest = value;
                        }
                    });
                buffer
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let array =
            numpy::ndarray::Array2::from_shape_vec((row_count, out_cols), out).map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build output matrix: {}",
                    err
                ))
            })?;
        Ok(array.to_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let file =
            File::create(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let file =
            File::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let reader = BufReader::new(file);
        let featurizer: RustFeaturizer = bincode::deserialize_from(reader)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(featurizer)
    }
}

impl RustNameCompatibleSubblockSelector {
    fn from_py(
        py: Python<'_>,
        retrieval_subblock_index: &Bound<'_, PyAny>,
        name_tuples_path: Option<String>,
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
        let name_tuples = load_name_tuples_from_text_path(py, name_tuples_path.as_deref())?;
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

    fn select_ordered_component_keys(
        &self,
        query_signature_id: &str,
        query_first: &str,
        ordered_component_keys: Vec<String>,
        global_backfill_count: usize,
    ) -> Option<Vec<String>> {
        let allowed_components = self.allowed_component_keys(query_signature_id, query_first)?;
        let mut selected: Vec<String> = ordered_component_keys
            .iter()
            .filter(|component_key| allowed_components.contains(*component_key))
            .cloned()
            .collect();
        if selected.is_empty() {
            return None;
        }
        if global_backfill_count > 0 {
            let mut selected_set: HashSet<String> = selected.iter().cloned().collect();
            let mut remaining = global_backfill_count;
            for component_key in ordered_component_keys {
                if remaining == 0 {
                    break;
                }
                if selected_set.insert(component_key.clone()) {
                    selected.push(component_key);
                    remaining -= 1;
                }
            }
        }
        Some(selected)
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
impl RustNameCompatibleSubblockSelector {
    #[new]
    #[pyo3(signature = (retrieval_subblock_index, name_tuples_path = None))]
    fn new(
        py: Python<'_>,
        retrieval_subblock_index: &Bound<'_, PyAny>,
        name_tuples_path: Option<String>,
    ) -> PyResult<Self> {
        Self::from_py(py, retrieval_subblock_index, name_tuples_path)
    }

    #[pyo3(signature = (query_signature_id, query_first, component_keys, global_backfill_count = 0))]
    fn select(
        &self,
        query_signature_id: &str,
        query_first: &str,
        component_keys: &Bound<'_, PyAny>,
        global_backfill_count: usize,
    ) -> PyResult<Option<Vec<String>>> {
        let ordered_component_keys: Vec<String> = PyIterator::from_object(component_keys)?
            .map(|item| item.and_then(|value| value.extract()))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(self.select_ordered_component_keys(
            query_signature_id,
            query_first,
            ordered_component_keys,
            global_backfill_count,
        ))
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
        let mut affiliation_cluster_df = HashMap::new();
        for item in PyIterator::from_object(summaries)? {
            let summary_obj = item?;
            update_cluster_df_from_counter(
                &summary_obj.getattr("coauthor_counts")?,
                &mut coauthor_cluster_df,
            )?;
            update_cluster_df_from_counter(
                &summary_obj.getattr("affiliation_counts")?,
                &mut affiliation_cluster_df,
            )?;
            let summary = extract_retrieval_summary(&summary_obj, include_exemplars)?;
            component_index_by_key.insert(summary.component_key.clone(), packed_summaries.len());
            packed_summaries.push(summary);
        }
        let max_block_component_size = packed_summaries
            .iter()
            .map(|summary| summary.size)
            .max()
            .unwrap_or(0);
        Ok(Self {
            summaries: packed_summaries,
            max_block_component_size,
            component_index_by_key,
            coauthor_cluster_df,
            affiliation_cluster_df,
        })
    }

    fn summary_count(&self) -> usize {
        self.summaries.len()
    }

    #[pyo3(signature = (query, top_k, num_threads = None))]
    fn top_k_hybrid_centroid(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        top_k: usize,
        num_threads: Option<usize>,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        if top_k == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "top_k must be positive",
            ));
        }
        let query_data = extract_retrieval_query(query)?;

        let mut candidate_indices: Vec<usize> = (0..self.summaries.len()).collect();

        if let Some(orcid_hash) = query_data.orcid_hash {
            let orcid_matches: Vec<usize> = candidate_indices
                .iter()
                .copied()
                .filter(|idx| contains_hashed_value(&self.summaries[*idx].orcid_hashes, orcid_hash))
                .collect();
            if !orcid_matches.is_empty() {
                candidate_indices = orcid_matches;
            }
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

        if candidate_indices.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        self.score_top_k_candidate_indices_experimental(
            py,
            &query_data,
            &candidate_indices,
            top_k,
            self.max_block_component_size,
            num_threads,
            None,
            None,
            Self::default_hybrid_weights_for_query(&query_data),
            Self::default_experimental_config_for_query(&query_data),
        )
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
        full_first_global_backfill_count = 0
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
    ) -> PyResult<Py<PyDict>> {
        if top_k == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "top_k must be positive",
            ));
        }
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
        let selector = retrieval_subblock_index
            .map(|index| RustNameCompatibleSubblockSelector::from_py(py, index, None))
            .transpose()?;
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

        let query_results: Vec<Result<RetrievalPairPlanQueryResult, String>> =
            py.allow_threads(|| {
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
                                query_indices[query_offset],
                                base_candidate_indices,
                                query_signature_id,
                                &component_member_indices,
                                top_k,
                                selector.as_ref(),
                                full_first_global_backfill_count,
                            )
                        })
                        .collect::<Vec<_>>()
                };
                install_with_optional_rayon_pool(num_threads, compute)
            });

        for query_result in query_results {
            let mut query_result = query_result.map_err(pyo3::exceptions::PyKeyError::new_err)?;
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
                let local_row_index = u32::try_from(local_row_index).map_err(|_| {
                    pyo3::exceptions::PyOverflowError::new_err(
                        "retrieved candidate row count exceeds u32",
                    )
                })?;
                let row_index = base_row_index.checked_add(local_row_index).ok_or_else(|| {
                    pyo3::exceptions::PyOverflowError::new_err(
                        "retrieved candidate row count exceeds u32",
                    )
                })?;
                let query_signature_index =
                    query_result.row_query_signature_indices[local_row_index as usize];
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

    #[pyo3(signature = (query, top_k, weights, num_threads = None))]
    fn top_k_weighted_hybrid_centroid(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        top_k: usize,
        weights: Vec<f64>,
        num_threads: Option<usize>,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        if top_k == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "top_k must be positive",
            ));
        }
        let query_data = extract_retrieval_query(query)?;
        let weights_data = extract_retrieval_weights(weights)?;

        let mut candidate_indices: Vec<usize> = (0..self.summaries.len()).collect();

        if let Some(orcid_hash) = query_data.orcid_hash {
            let orcid_matches: Vec<usize> = candidate_indices
                .iter()
                .copied()
                .filter(|idx| contains_hashed_value(&self.summaries[*idx].orcid_hashes, orcid_hash))
                .collect();
            if !orcid_matches.is_empty() {
                candidate_indices = orcid_matches;
            }
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

        if candidate_indices.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        self.score_top_k_candidate_indices(
            py,
            &query_data,
            &candidate_indices,
            top_k,
            self.max_block_component_size,
            num_threads,
            None,
            None,
            weights_data,
        )
    }

    #[pyo3(signature = (query, component_keys, top_k, max_block_component_size, num_threads = None, override_summary = None))]
    fn top_k_hybrid_centroid_subset(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        component_keys: &Bound<'_, PyAny>,
        top_k: usize,
        max_block_component_size: usize,
        num_threads: Option<usize>,
        override_summary: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        if top_k == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "top_k must be positive",
            ));
        }

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

        self.score_top_k_candidate_indices_experimental(
            py,
            &query_data,
            &candidate_indices,
            top_k,
            max_block_component_size,
            num_threads,
            override_index,
            override_data.as_ref(),
            Self::default_hybrid_weights_for_query(&query_data),
            Self::default_experimental_config_for_query(&query_data),
        )
    }

    #[pyo3(signature = (query, component_keys, top_k, max_block_component_size, weights, num_threads = None, override_summary = None))]
    fn top_k_weighted_hybrid_centroid_subset(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        component_keys: &Bound<'_, PyAny>,
        top_k: usize,
        max_block_component_size: usize,
        weights: Vec<f64>,
        num_threads: Option<usize>,
        override_summary: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        if top_k == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "top_k must be positive",
            ));
        }

        let query_data = extract_retrieval_query(query)?;
        let weights_data = extract_retrieval_weights(weights)?;
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
            .map(|value| extract_retrieval_summary(value, false))
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

        self.score_top_k_candidate_indices(
            py,
            &query_data,
            &candidate_indices,
            top_k,
            max_block_component_size,
            num_threads,
            override_index,
            override_data.as_ref(),
            weights_data,
        )
    }

    #[pyo3(
        signature = (
            query,
            component_keys,
            top_k,
            max_block_component_size,
            weights,
            first_name_mode = "prefix",
            specter_mode = "centroid",
            coauthor_use_idf = false,
            coauthor_per_term_cap = None,
            coauthor_total_cap = None,
            affiliation_use_idf = false,
            affiliation_per_term_cap = None,
            affiliation_total_cap = None,
            affiliation_min_token_count = 1,
            affiliation_unigram_weight = 1.0,
            affiliation_multi_token_weight = 1.0,
            num_threads = None,
            override_summary = None
        )
    )]
    fn top_k_experimental_weighted_hybrid_centroid_subset(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        component_keys: &Bound<'_, PyAny>,
        top_k: usize,
        max_block_component_size: usize,
        weights: Vec<f64>,
        first_name_mode: &str,
        specter_mode: &str,
        coauthor_use_idf: bool,
        coauthor_per_term_cap: Option<f64>,
        coauthor_total_cap: Option<f64>,
        affiliation_use_idf: bool,
        affiliation_per_term_cap: Option<f64>,
        affiliation_total_cap: Option<f64>,
        affiliation_min_token_count: usize,
        affiliation_unigram_weight: f64,
        affiliation_multi_token_weight: f64,
        num_threads: Option<usize>,
        override_summary: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        if top_k == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "top_k must be positive",
            ));
        }

        let query_data = extract_retrieval_query(query)?;
        let weights_data = extract_retrieval_weights(weights)?;
        let config = build_experimental_config(
            first_name_mode,
            specter_mode,
            coauthor_use_idf,
            coauthor_per_term_cap,
            coauthor_total_cap,
            affiliation_use_idf,
            affiliation_per_term_cap,
            affiliation_total_cap,
            affiliation_min_token_count,
            affiliation_unigram_weight,
            affiliation_multi_token_weight,
        )?;
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
            .map(|value| {
                extract_retrieval_summary(
                    value,
                    !matches!(config.specter_mode, RetrievalSpecterMode::Centroid),
                )
            })
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

        self.score_top_k_candidate_indices_experimental(
            py,
            &query_data,
            &candidate_indices,
            top_k,
            max_block_component_size,
            num_threads,
            override_index,
            override_data.as_ref(),
            weights_data,
            config,
        )
    }

    #[pyo3(signature = (query, component_keys, num_threads = None, override_summary = None))]
    fn chooser_feature_rows_subset(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        component_keys: &Bound<'_, PyAny>,
        num_threads: Option<usize>,
        override_summary: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        let _resolved_num_threads = num_threads;
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

        let payload = PyDict::new(py);
        for candidate_index in candidate_indices {
            let summary = if override_index == Some(candidate_index) {
                override_data
                    .as_ref()
                    .unwrap_or_else(|| unreachable!("override_index implies override_data"))
            } else {
                &self.summaries[candidate_index]
            };
            let feature_values = chooser_summary_features(&query_data, summary);
            let feature_dict = PyDict::new(py);
            feature_dict.set_item("middle_initial_compatibility", feature_values[0])?;
            feature_dict.set_item("affiliation_overlap", feature_values[1])?;
            feature_dict.set_item("coauthor_overlap", feature_values[2])?;
            feature_dict.set_item("venue_overlap", feature_values[3])?;
            feature_dict.set_item("year_compatibility", feature_values[4])?;
            feature_dict.set_item("title_overlap", feature_values[5])?;
            feature_dict.set_item("specter_centroid_similarity", feature_values[6])?;
            feature_dict.set_item("specter_exemplar_similarity", feature_values[7])?;
            payload.set_item(summary.component_key.as_str(), feature_dict)?;
        }
        Ok(payload.unbind())
    }
}

const LINKER_CLUSTER_SIZE_LOG_CAPPED_REFERENCE_SIZE: f32 = 192.0;
const LINKER_GENERIC_HEURISTIC_OVERRIDE_MARGIN: f32 = 0.01;
const LINKER_GENERIC_CROSS_FAMILY_EXTRA_MARGIN: f32 = 0.04;
const LINKER_GENERIC_FAMILY_MIN_COUNT: f32 = 3.0;
const LINKER_GENERIC_FAMILY_MIN_RATIO: f32 = 0.6;
const LINKER_EXACT_TITLE_ANCHOR_THRESHOLD: f32 = 0.95;
const LINKER_ANCHOR_SUPPORT_OVERLAP_THRESHOLD: f32 = 0.05;
const LINKER_ANCHOR_YEAR_COMPATIBILITY_THRESHOLD: f32 = 0.85;

fn linker_round(value: f32, scale: f32) -> f32 {
    (value * scale).round() / scale
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
    let values = if let Ok(arr) = obj.downcast::<PyArray1<f32>>() {
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
            out.push(item?.extract::<f64>()? as f32);
        }
        out
    };
    if values.len() != row_count {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Signal {key:?} must have row_count={row_count}, got {}",
            values.len()
        )));
    }
    if values.iter().any(|value| value.is_nan()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Signal {key:?} contains NaN values"
        )));
    }
    Ok(values)
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
    if dict.get_item(key)?.is_none() {
        return Ok(None);
    }
    Ok(Some(linker_extract_string_vec(dict, key, row_count)?))
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
    retrieval_rank: &[f32],
    component_keys: &[String],
) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(groups.len());
    for group in groups {
        let mut ordered = group.clone();
        ordered.sort_by(|left, right| {
            (retrieval_rank[*left] as i64)
                .cmp(&(retrieval_rank[*right] as i64))
                .then_with(|| component_keys[*left].cmp(&component_keys[*right]))
        });
        out.push(ordered);
    }
    out
}

fn linker_best_index_by_group(
    primary: &[f32],
    retrieval_rank: &[f32],
    component_keys: &[String],
    groups: &[Vec<usize>],
    higher_is_better: bool,
) -> Vec<usize> {
    let mut out = vec![0usize; primary.len()];
    for group in groups {
        let mut best = group[0];
        for index in group.iter().copied().skip(1) {
            let current_key = if higher_is_better {
                (
                    -primary[index],
                    retrieval_rank[index] as i64,
                    component_keys[index].as_str(),
                )
            } else {
                (
                    primary[index],
                    retrieval_rank[index] as i64,
                    component_keys[index].as_str(),
                )
            };
            let best_key = if higher_is_better {
                (
                    -primary[best],
                    retrieval_rank[best] as i64,
                    component_keys[best].as_str(),
                )
            } else {
                (
                    primary[best],
                    retrieval_rank[best] as i64,
                    component_keys[best].as_str(),
                )
            };
            if current_key < best_key {
                best = index;
            }
        }
        for index in group {
            out[*index] = best;
        }
    }
    out
}

fn linker_normalize_alpha(value: &str) -> String {
    value
        .chars()
        .flat_map(|character| character.to_lowercase())
        .filter(|character| character.is_ascii_alphabetic())
        .collect()
}

fn linker_normalized_alpha_vec(values: &[String]) -> Vec<String> {
    let mut cache = HashMap::<String, String>::new();
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if let Some(normalized) = cache.get(value) {
            out.push(normalized.clone());
        } else {
            let normalized = linker_normalize_alpha(value);
            cache.insert(value.clone(), normalized.clone());
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

fn linker_confident_family_mask(family_ids: &[String], component_keys: &[String]) -> Vec<bool> {
    family_ids
        .iter()
        .zip(component_keys.iter())
        .map(|(family, component)| !family.is_empty() && family != component)
        .collect()
}

fn linker_coarse_family_keys(
    dominant_first_alpha: &[String],
    family_alpha: &[String],
    component_alpha: &[String],
) -> Vec<String> {
    dominant_first_alpha
        .iter()
        .zip(family_alpha.iter())
        .zip(component_alpha.iter())
        .map(|((dominant, family), component)| {
            let source = if !dominant.is_empty() {
                dominant
            } else if !family.is_empty() {
                family
            } else {
                component
            };
            source.chars().take(3).collect()
        })
        .collect()
}

fn linker_cross_family(
    left: usize,
    right: usize,
    family_ids: &[String],
    confident_family: &[bool],
) -> bool {
    confident_family[left] && confident_family[right] && family_ids[left] != family_ids[right]
}

struct LinkerGroupFeatures {
    retrieval_score_gap_vs_best_competitor: Vec<f32>,
    retrieval_score_best_gap: Vec<f32>,
    same_family_as_top1: Vec<f32>,
    same_family_as_best_top5: Vec<f32>,
    same_family_as_heuristic_choice: Vec<f32>,
    coarse_family_top5_best_gap: Vec<f32>,
    candidate_pair_share_within_coarse_family: Vec<f32>,
}

fn linker_derive_group_features(
    ordered_groups: &[Vec<usize>],
    retrieval_score: &[f32],
    retrieval_rank: &[f32],
    component_keys: &[String],
    family_ids: &[String],
    confident_family: &[bool],
    coarse_family_keys: &[String],
    pair_count: &[f32],
    top5_mean_distance: &[f32],
) -> LinkerGroupFeatures {
    let row_count = retrieval_score.len();
    let mut retrieval_score_gap_vs_best_competitor = vec![0.0f32; row_count];
    let mut retrieval_score_best_gap = vec![0.0f32; row_count];
    let mut same_family_as_top1 = vec![0.0f32; row_count];
    let mut same_family_as_best_top5 = vec![0.0f32; row_count];
    let mut same_family_as_heuristic_choice = vec![0.0f32; row_count];
    let mut coarse_family_top5_best_gap = vec![0.0f32; row_count];
    let mut candidate_pair_share_within_coarse_family = vec![1.0f32; row_count];

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
                retrieval_rank[index] as i64,
                component_keys[index].as_str(),
            );
            let best_key = (
                top5_mean_distance[best_top5],
                retrieval_rank[best_top5] as i64,
                component_keys[best_top5].as_str(),
            );
            if current_key < best_key {
                best_top5 = index;
            }
        }
        let mut heuristic_choice = best_top5;
        if best_top5 != top1 {
            let mut effective_margin = LINKER_GENERIC_HEURISTIC_OVERRIDE_MARGIN;
            if linker_cross_family(top1, best_top5, family_ids, confident_family) {
                effective_margin += LINKER_GENERIC_CROSS_FAMILY_EXTRA_MARGIN;
            }
            if top5_mean_distance[best_top5] + effective_margin >= top5_mean_distance[top1] {
                heuristic_choice = top1;
            }
        }
        let best_score = ordered
            .iter()
            .map(|index| retrieval_score[*index])
            .fold(f32::NEG_INFINITY, f32::max);
        for index in ordered {
            let competitor = if *index == top1 { runner_up } else { top1 };
            retrieval_score_gap_vs_best_competitor[*index] = linker_round(
                retrieval_score[*index] - retrieval_score[competitor],
                1_000_000.0,
            );
            retrieval_score_best_gap[*index] =
                linker_round(best_score - retrieval_score[*index], 1_000_000.0);
            same_family_as_top1[*index] = linker_bool(
                !family_ids[*index].is_empty() && family_ids[*index] == family_ids[top1],
            );
            same_family_as_best_top5[*index] = linker_bool(
                !family_ids[*index].is_empty() && family_ids[*index] == family_ids[best_top5],
            );
            same_family_as_heuristic_choice[*index] = linker_bool(
                !family_ids[*index].is_empty()
                    && family_ids[*index] == family_ids[heuristic_choice],
            );
        }

        let mut coarse_to_rows: HashMap<String, Vec<usize>> = HashMap::new();
        for index in ordered {
            coarse_to_rows
                .entry(coarse_family_keys[*index].clone())
                .or_default()
                .push(*index);
        }
        for coarse_rows in coarse_to_rows.values() {
            let total_pairs = coarse_rows
                .iter()
                .map(|index| pair_count[*index])
                .sum::<f32>()
                .max(1.0);
            let mut best = coarse_rows[0];
            for index in coarse_rows.iter().copied().skip(1) {
                let current_key = (
                    top5_mean_distance[index],
                    retrieval_rank[index] as i64,
                    component_keys[index].as_str(),
                );
                let best_key = (
                    top5_mean_distance[best],
                    retrieval_rank[best] as i64,
                    component_keys[best].as_str(),
                );
                if current_key < best_key {
                    best = index;
                }
            }
            for index in coarse_rows {
                coarse_family_top5_best_gap[*index] = linker_round(
                    top5_mean_distance[*index] - top5_mean_distance[best],
                    1_000_000.0,
                );
                candidate_pair_share_within_coarse_family[*index] =
                    linker_round(pair_count[*index] / total_pairs, 1_000_000.0);
            }
        }
    }

    LinkerGroupFeatures {
        retrieval_score_gap_vs_best_competitor,
        retrieval_score_best_gap,
        same_family_as_top1,
        same_family_as_best_top5,
        same_family_as_heuristic_choice,
        coarse_family_top5_best_gap,
        candidate_pair_share_within_coarse_family,
    }
}

fn linker_year_mismatch_severity(
    query_year: &[f32],
    query_year_missing: &[f32],
    candidate_year_min: &[f32],
    candidate_year_max: &[f32],
    candidate_year_range_missing: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; query_year.len()];
    for index in 0..query_year.len() {
        if query_year_missing[index] != 0.0 || candidate_year_range_missing[index] != 0.0 {
            continue;
        }
        if query_year[index] < candidate_year_min[index] {
            out[index] = linker_round(
                ((candidate_year_min[index] - query_year[index]) / 10.0).min(1.0),
                1_000_000.0,
            );
        } else if query_year[index] > candidate_year_max[index] {
            out[index] = linker_round(
                ((query_year[index] - candidate_year_max[index]) / 10.0).min(1.0),
                1_000_000.0,
            );
        }
    }
    out
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
    let component_keys = linker_extract_string_vec(signals, "candidate_component_key", row_count)?;
    let query_view = linker_extract_string_vec(signals, "query_view", row_count)?;
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
    let query_has_coauthors = linker_extract_f32_vec(signals, "query_has_coauthors", row_count)?;
    let affiliation_overlap = linker_extract_f32_vec(signals, "affiliation_overlap", row_count)?;
    let coauthor_overlap = linker_extract_f32_vec(signals, "coauthor_overlap", row_count)?;
    let venue_overlap = linker_extract_f32_vec(signals, "venue_overlap", row_count)?;
    let year_compatibility = linker_extract_f32_vec(signals, "year_compatibility", row_count)?;
    let title_overlap = linker_extract_f32_vec(signals, "title_overlap", row_count)?;
    let specter_exemplar_similarity =
        linker_extract_f32_vec(signals, "specter_exemplar_similarity", row_count)?;
    let min_distance = linker_extract_f32_vec(signals, "min_distance", row_count)?;
    let top5_mean_distance = linker_extract_f32_vec(signals, "top5_mean_distance", row_count)?;
    let pair_count = linker_extract_f32_vec(signals, "pair_count", row_count)?;
    let last_name_count_min_rarity =
        linker_extract_f32_vec(signals, "last_name_count_min_rarity", row_count)?;
    let candidate_last_first_name_count_min_rarity = linker_extract_f32_vec(
        signals,
        "candidate_last_first_name_count_min_rarity",
        row_count,
    )?;
    let last_first_name_count_min_rarity =
        linker_extract_f32_vec(signals, "last_first_name_count_min_rarity", row_count)?;
    let first_prefix_x_last_first_name_count_min_rarity = linker_extract_f32_vec(
        signals,
        "first_prefix_x_last_first_name_count_min_rarity",
        row_count,
    )?;

    let groups = linker_groups(&row_query_signature_indices);
    let ordered_groups = linker_retrieval_ordered_groups(&groups, &retrieval_rank, &component_keys);
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
    let confident_family = linker_confident_family_mask(&family_ids, &component_keys);
    let query_first_alpha = linker_normalized_alpha_vec(&query_first_token);
    let dominant_first_alpha = linker_normalized_alpha_vec(&dominant_first_name);
    let family_alpha = linker_normalized_alpha_vec(&family_ids);
    let component_alpha = linker_normalized_alpha_vec(&component_keys);
    let coarse_family_keys =
        linker_coarse_family_keys(&dominant_first_alpha, &family_alpha, &component_alpha);
    let best_top5_indices = linker_best_index_by_group(
        &top5_mean_distance,
        &retrieval_rank,
        &component_keys,
        &groups,
        false,
    );
    let group_features = linker_derive_group_features(
        &ordered_groups,
        &retrieval_score,
        &retrieval_rank,
        &component_keys,
        &family_ids,
        &confident_family,
        &coarse_family_keys,
        &pair_count,
        &top5_mean_distance,
    );
    let year_mismatch_severity = linker_year_mismatch_severity(
        &query_year,
        &query_year_missing,
        &candidate_year_min,
        &candidate_year_max,
        &candidate_year_range_missing,
    );

    let mut affiliation_contradiction_severity = vec![0.0f32; row_count];
    let mut first_name_compatibility = vec![0.0f32; row_count];
    let mut coauthor_contradiction = vec![0.0f32; row_count];
    let mut contradiction = vec![0.0f32; row_count];
    let mut exact_anchor_evidence_flag = vec![0.0f32; row_count];
    let mut anchor_evidence_count = vec![0.0f32; row_count];
    let mut strong_positive_anchor_score = vec![0.0f32; row_count];
    let mut weak_residual_anchor_score = vec![0.0f32; row_count];
    let mut sparse_relative_winner_score = vec![0.0f32; row_count];
    let mut query_first_prefix_match = vec![0.0f32; row_count];
    let mut cluster_size_log_capped = vec![0.0f32; row_count];
    let mut distance_spread_top5_minus_min = vec![0.0f32; row_count];
    let mut query_view_initial_only = vec![0.0f32; row_count];
    let mut top5_distance_best_gap = vec![0.0f32; row_count];

    let log_reference = (1.0 + LINKER_CLUSTER_SIZE_LOG_CAPPED_REFERENCE_SIZE).ln();
    for index in 0..row_count {
        if query_has_affiliations[index] > 0.0 {
            affiliation_contradiction_severity[index] =
                linker_round((1.0 - affiliation_overlap[index]).max(0.0), 1_000_000.0);
        }
        let query_first = &query_first_alpha[index];
        let dominant_first = &dominant_first_alpha[index];
        if !dominant_first.is_empty()
            && ((!query_first.is_empty()
                && (query_first == dominant_first
                    || query_first.starts_with(dominant_first)
                    || dominant_first.starts_with(query_first)))
                || (!query_first.is_empty() && dominant_first.starts_with(&query_first[0..1])))
        {
            first_name_compatibility[index] = 1.0;
        }
        if query_has_coauthors[index] > 0.0 {
            coauthor_contradiction[index] = (1.0 - coauthor_overlap[index]).max(0.0);
        }
        let title_anchor = title_overlap[index] >= LINKER_EXACT_TITLE_ANCHOR_THRESHOLD;
        contradiction[index] = year_mismatch_severity[index]
            .max(affiliation_contradiction_severity[index])
            .max(if title_anchor {
                coauthor_contradiction[index]
            } else {
                0.0
            })
            .max(if title_anchor && first_name_compatibility[index] <= 0.0 {
                1.0
            } else {
                0.0
            });
        exact_anchor_evidence_flag[index] = linker_bool(
            title_overlap[index] >= LINKER_EXACT_TITLE_ANCHOR_THRESHOLD
                && (coauthor_overlap[index] >= LINKER_ANCHOR_SUPPORT_OVERLAP_THRESHOLD
                    || affiliation_overlap[index] >= LINKER_ANCHOR_SUPPORT_OVERLAP_THRESHOLD
                    || year_compatibility[index] >= LINKER_ANCHOR_YEAR_COMPATIBILITY_THRESHOLD),
        );
        let retrieval_gap = group_features.retrieval_score_gap_vs_best_competitor[index];
        anchor_evidence_count[index] = linker_bool(min_distance[index] <= 0.15)
            + linker_bool(specter_exemplar_similarity[index] >= 0.70)
            + linker_bool(title_overlap[index] >= 0.20)
            + linker_bool(coauthor_overlap[index] >= 0.25)
            + linker_bool(affiliation_overlap[index] >= 0.25)
            + linker_bool(venue_overlap[index] >= 0.20)
            + linker_bool(year_compatibility[index] >= 0.90)
            + linker_bool(retrieval_gap >= 0.02);
        let support_strength = 0.20 * (1.0 - linker_clip01(min_distance[index]))
            + 0.20 * linker_clip01(specter_exemplar_similarity[index])
            + 0.18 * linker_clip01(title_overlap[index])
            + 0.18 * linker_clip01(coauthor_overlap[index])
            + 0.12 * linker_clip01(affiliation_overlap[index])
            + 0.06 * linker_clip01(venue_overlap[index])
            + 0.06 * linker_clip01(year_compatibility[index]);
        let same_top1 = group_features.same_family_as_top1[index];
        strong_positive_anchor_score[index] = linker_round(
            linker_clip01(support_strength)
                * (0.5 + 0.5 * linker_clip01(same_top1))
                * (0.35 + 0.65 * linker_clip01(1.0 - contradiction[index])),
            1_000_000.0,
        );
        let retrieval_gap_scaled = linker_clip01((retrieval_gap.clamp(-0.2, 0.3) + 0.2) / 0.5);
        let residual_support = 0.28 * (1.0 - linker_clip01(min_distance[index]))
            + 0.20 * linker_clip01(specter_exemplar_similarity[index])
            + 0.20 * linker_clip01(coauthor_overlap[index])
            + 0.14 * linker_clip01(title_overlap[index])
            + 0.10 * linker_clip01(year_compatibility[index])
            + 0.08 * retrieval_gap_scaled;
        let tiny_candidate =
            linker_bool(cluster_size[index] <= 2.0 || named_signature_count[index] <= 2.0);
        weak_residual_anchor_score[index] = linker_round(
            tiny_candidate * same_top1 * linker_clip01(residual_support),
            1_000_000.0,
        );
        sparse_relative_winner_score[index] = linker_round(
            linker_bool(retrieval_rank[index] <= 1.0)
                * same_top1
                * linker_clip01(retrieval_gap.clamp(0.0, 0.3) / 0.3)
                * (1.0
                    - linker_clip01(
                        group_features.candidate_pair_share_within_coarse_family[index],
                    ))
                * linker_clip01(residual_support),
            1_000_000.0,
        );
        query_first_prefix_match[index] = linker_bool(
            !query_first.is_empty()
                && py_len(query_first) > 1
                && !dominant_first.is_empty()
                && (query_first.starts_with(dominant_first)
                    || dominant_first.starts_with(query_first)),
        );
        if cluster_size[index] > 0.0 {
            cluster_size_log_capped[index] =
                ((1.0 + cluster_size[index]).ln() / log_reference).min(1.0);
        }
        distance_spread_top5_minus_min[index] =
            linker_round(top5_mean_distance[index] - min_distance[index], 1_000_000.0);
        query_view_initial_only[index] = linker_bool(query_view[index] == "initial_only");
        top5_distance_best_gap[index] = linker_round(
            top5_mean_distance[index] - top5_mean_distance[best_top5_indices[index]],
            1_000_000.0,
        );
    }

    let payload = PyDict::new(py);
    linker_set_f32_array(py, &payload, "min_distance", min_distance.clone())?;
    linker_set_f32_array(
        py,
        &payload,
        "retrieval_score_gap_vs_best_competitor",
        group_features.retrieval_score_gap_vs_best_competitor,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "top5_distance_best_gap",
        top5_distance_best_gap,
    )?;
    linker_set_f32_array(py, &payload, "retrieval_score", retrieval_score.clone())?;
    linker_set_f32_array(
        py,
        &payload,
        "affiliation_contradiction_severity",
        affiliation_contradiction_severity,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "coarse_family_top5_best_gap",
        group_features.coarse_family_top5_best_gap,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "same_family_as_best_top5",
        group_features.same_family_as_best_top5,
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
        "same_family_as_top1",
        group_features.same_family_as_top1,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "query_first_prefix_match",
        query_first_prefix_match,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "retrieval_score_best_gap",
        group_features.retrieval_score_best_gap,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "cluster_size_log_capped",
        cluster_size_log_capped,
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
        "query_view__initial_only",
        query_view_initial_only,
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
        "candidate_last_first_name_count_min_rarity",
        candidate_last_first_name_count_min_rarity,
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
        "first_prefix_x_last_first_name_count_min_rarity",
        first_prefix_x_last_first_name_count_min_rarity,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "exact_anchor_evidence_flag",
        exact_anchor_evidence_flag,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "year_mismatch_severity",
        year_mismatch_severity,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "top5_mean_distance",
        top5_mean_distance.clone(),
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "distance_spread_top5_minus_min",
        distance_spread_top5_minus_min,
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

#[pyfunction]
fn get_build_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let build_info = PyDict::new(py);
    build_info.set_item("crate_version", env!("CARGO_PKG_VERSION"))?;
    build_info.set_item("profile", option_env!("PROFILE").unwrap_or("unknown"))?;
    build_info.set_item("debug_assertions", cfg!(debug_assertions))?;
    build_info.set_item("debug", option_env!("DEBUG").unwrap_or("unknown"))?;
    build_info.set_item("opt_level", option_env!("OPT_LEVEL").unwrap_or("unknown"))?;
    build_info.set_item("target", option_env!("TARGET").unwrap_or("unknown"))?;
    build_info.set_item("host", option_env!("HOST").unwrap_or("unknown"))?;
    build_info.set_item("rustc", option_env!("RUSTC").unwrap_or("unknown"))?;
    Ok(build_info.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyString;

    #[test]
    fn reference_details_extraction_errors_are_not_silenced() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let non_tuple = PyString::new(py, "not-a-tuple");
            let result = extract_reference_details_counters(py, non_tuple.as_any());
            assert!(result.is_err(), "non-tuple reference_details should raise");
            let err = result
                .err()
                .unwrap_or_else(|| unreachable!("assert above guarantees error"));
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
        });
    }
}

#[pymodule]
fn _s2and_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("RETRIEVAL_FEATURE_ORDER", RETRIEVAL_FEATURE_ORDER.to_vec())?;
    m.add(
        "DEFAULT_HYBRID_CENTROID_POLICY_NAME",
        DEFAULT_HYBRID_CENTROID_POLICY_NAME,
    )?;
    m.add(
        "DEFAULT_HYBRID_CENTROID_WEIGHTS",
        DEFAULT_HYBRID_CENTROID_WEIGHTS.to_vec(),
    )?;
    m.add(
        "DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS",
        DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS.to_vec(),
    )?;
    m.add(
        "DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS",
        DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS.to_vec(),
    )?;
    m.add(
        "RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE",
        RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE,
    )?;
    m.add(
        "RETRIEVAL_YEAR_SCORE_DECAY_YEARS",
        RETRIEVAL_YEAR_SCORE_DECAY_YEARS,
    )?;
    m.add(
        "RETRIEVAL_YEAR_SCORE_RANGE_GAP",
        RETRIEVAL_YEAR_SCORE_RANGE_GAP,
    )?;
    m.add(
        "RETRIEVAL_YEAR_SCORE_RANGE_PENALTY",
        RETRIEVAL_YEAR_SCORE_RANGE_PENALTY,
    )?;
    m.add(
        "RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP",
        RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP,
    )?;
    m.add_function(wrap_pyfunction!(get_build_info, m)?)?;
    m.add_function(wrap_pyfunction!(promoted_linker_non_pairwise_features, m)?)?;
    m.add_function(wrap_pyfunction!(signature_ngrams_batch, m)?)?;
    m.add_function(wrap_pyfunction!(get_last_json_ingest_telemetry, m)?)?;
    m.add_function(wrap_pyfunction!(reset_last_json_ingest_telemetry, m)?)?;
    m.add_class::<RustFeaturizer>()?;
    m.add_class::<RustHybridCentroidRetriever>()?;
    m.add_class::<RustNameCompatibleSubblockSelector>()?;
    Ok(())
}
