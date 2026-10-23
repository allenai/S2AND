use super::*;
use numpy::IntoPyArray;
use pyo3::types::{PyList, PyString};
/// Shared evidence columns emitted in the same order by both planner routes.
#[derive(Default)]
struct RawArrowEvidenceSignals {
    row_orcid_match: Vec<u8>,
    row_middle_initial_compatibility: Vec<f32>,
    row_affiliation_overlap: Vec<f32>,
    row_coauthor_overlap: Vec<f32>,
    row_venue_overlap: Vec<f32>,
    row_year_compatibility: Vec<f32>,
    row_title_overlap: Vec<f32>,
    row_specter_centroid_similarity: Vec<f32>,
    row_specter_exemplar_similarity: Vec<f32>,
    row_last_name_count_min_rarity: Vec<f32>,
    row_candidate_last_name_count_min_rarity: Vec<f32>,
    row_candidate_last_first_name_count_min_rarity: Vec<f32>,
    row_last_first_name_count_min_rarity: Vec<f32>,
    row_first_prefix_x_last_first_name_count_min_rarity: Vec<f32>,
    row_candidate_cluster_max_paper_author_count: Vec<f32>,
    row_paper_author_list_max_jaccard: Vec<f32>,
    row_paper_author_list_max_containment: Vec<f32>,
    row_paper_author_list_max_overlap_count: Vec<f32>,
    row_local_author_window10_jaccard_max: Vec<f32>,
    row_local_author_window10_overlap_count_max: Vec<f32>,
    row_best_author_count_log_absdiff: Vec<f32>,
}

impl RawArrowEvidenceSignals {
    /// Move the owned columns into NumPy without changing their native dtypes.
    fn append_to(self, py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<()> {
        payload.set_item("row_orcid_match", self.row_orcid_match.into_pyarray(py))?;
        payload.set_item(
            "middle_initial_compatibility",
            self.row_middle_initial_compatibility.into_pyarray(py),
        )?;
        payload.set_item(
            "affiliation_overlap",
            self.row_affiliation_overlap.into_pyarray(py),
        )?;
        payload.set_item(
            "coauthor_overlap",
            self.row_coauthor_overlap.into_pyarray(py),
        )?;
        payload.set_item("venue_overlap", self.row_venue_overlap.into_pyarray(py))?;
        payload.set_item(
            "year_compatibility",
            self.row_year_compatibility.into_pyarray(py),
        )?;
        payload.set_item("title_overlap", self.row_title_overlap.into_pyarray(py))?;
        payload.set_item(
            "specter_centroid_similarity",
            self.row_specter_centroid_similarity.into_pyarray(py),
        )?;
        payload.set_item(
            "specter_exemplar_similarity",
            self.row_specter_exemplar_similarity.into_pyarray(py),
        )?;
        payload.set_item(
            "row_last_name_count_min_rarity",
            self.row_last_name_count_min_rarity.into_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_last_name_count_min_rarity",
            self.row_candidate_last_name_count_min_rarity
                .into_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_last_first_name_count_min_rarity",
            self.row_candidate_last_first_name_count_min_rarity
                .into_pyarray(py),
        )?;
        payload.set_item(
            "row_last_first_name_count_min_rarity",
            self.row_last_first_name_count_min_rarity.into_pyarray(py),
        )?;
        payload.set_item(
            "row_first_prefix_x_last_first_name_count_min_rarity",
            self.row_first_prefix_x_last_first_name_count_min_rarity
                .into_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_cluster_max_paper_author_count",
            self.row_candidate_cluster_max_paper_author_count
                .into_pyarray(py),
        )?;
        payload.set_item(
            "row_paper_author_list_max_jaccard",
            self.row_paper_author_list_max_jaccard.into_pyarray(py),
        )?;
        payload.set_item(
            "row_paper_author_list_max_containment",
            self.row_paper_author_list_max_containment.into_pyarray(py),
        )?;
        payload.set_item(
            "row_paper_author_list_max_overlap_count",
            self.row_paper_author_list_max_overlap_count
                .into_pyarray(py),
        )?;
        payload.set_item(
            "row_local_author_window10_jaccard_max",
            self.row_local_author_window10_jaccard_max.into_pyarray(py),
        )?;
        payload.set_item(
            "row_local_author_window10_overlap_count_max",
            self.row_local_author_window10_overlap_count_max
                .into_pyarray(py),
        )?;
        payload.set_item(
            "row_best_author_count_log_absdiff",
            self.row_best_author_count_log_absdiff.into_pyarray(py),
        )?;
        Ok(())
    }
}

struct RawArrowPlannerBuildTelemetry {
    read_cluster_seeds_secs: f64,
    read_signatures_secs: f64,
    read_papers_secs: f64,
    read_paper_authors_secs: f64,
    read_specter_secs: f64,
    read_name_counts_secs: f64,
    metadata_reads_parallel_secs: f64,
    text_context_secs: f64,
    feature_secs: f64,
    summary_secs: f64,
    component_members_secs: f64,
    signature_index_stats: IndexedArrowReadStats,
    paper_index_stats: IndexedArrowReadStats,
    paper_author_index_stats: IndexedArrowReadStats,
    specter_index_stats: IndexedArrowReadStats,
    indexed_arrow_candidate_plan: bool,
    reused_name_counts_index: bool,
    /// Seed features and component summaries a reload carried over.
    reused_seed_feature_count: usize,
    reused_summary_count: usize,
}

struct ReusableRawArrowCandidatePlanState {
    features_by_signature_id: HashMap<String, RawArrowFeature>,
    signatures: HashMap<String, RawArrowSignature>,
    paper_authors: HashMap<String, Vec<(i64, String)>>,
    /// Interned author names of seed papers needed by candidate summaries.
    seed_author_index: PaperAuthorIndex,
    raw_name_counts: RawNameCountMaps,
    members_by_component: HashMap<String, Vec<String>>,
    component_keys_by_member: HashMap<String, Vec<String>>,
    summary_signals_by_component: HashMap<String, RawArrowSummarySignalData>,
    retriever: RustHybridCentroidRetriever,
    component_order: Vec<String>,
    seed_signature_ids: Vec<String>,
    seed_signature_id_set: HashSet<String>,
    cluster_seed_disallows: HashSet<(String, String)>,
    unidecode_char_map: HashMap<char, String>,
    name_prefixes: HashSet<String>,
    affiliation_stopwords: HashSet<String>,
    seed_paper_count: usize,
    seed_specter_count: usize,
    /// Rows kept for `reload_seed_sidecars`; None unless built with `retain_seed_rows`.
    retained_seed_rows: Option<RetainedRawArrowSeedRows>,
    /// Rows of block signatures that later passes plan; see `PrefetchedRawArrowRows`.
    prefetched_rows: Option<PrefetchedRawArrowRows>,
    /// Query rows of the latest `plan()`, for `build_featurizer`.
    last_query_inputs: Option<RawArrowQueryInputReadResult>,
    build_telemetry: RawArrowPlannerBuildTelemetry,
}

struct RawArrowQueryInputReadResult {
    signatures: HashMap<String, RawArrowSignature>,
    papers: HashMap<String, RawArrowPaper>,
    paper_authors: HashMap<String, Vec<(i64, String)>>,
    specter_by_paper_id: Option<HashMap<String, Arc<Vec<f32>>>>,
    signature_index_stats: IndexedArrowReadStats,
    paper_index_stats: IndexedArrowReadStats,
    paper_author_index_stats: IndexedArrowReadStats,
    specter_index_stats: IndexedArrowReadStats,
    read_signatures_secs: f64,
    read_papers_secs: f64,
    read_paper_authors_secs: f64,
    read_specter_secs: f64,
    metadata_reads_parallel_secs: f64,
}

fn validate_raw_arrow_query_signature_ids_allow_empty(
    query_signature_ids: &[String],
) -> PyResult<()> {
    let mut seen =
        HashSet::<&str>::with_capacity_and_hasher(query_signature_ids.len(), Default::default());
    for signature_id in query_signature_ids.iter() {
        if !seen.insert(signature_id.as_str()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "query_signature_ids must be unique; duplicate signature_id={signature_id:?}"
            )));
        }
    }
    Ok(())
}

fn validate_raw_arrow_query_signature_ids(query_signature_ids: &[String]) -> PyResult<()> {
    if query_signature_ids.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "query_signature_ids must be non-empty",
        ));
    }
    validate_raw_arrow_query_signature_ids_allow_empty(query_signature_ids)
}

fn validate_additional_cluster_seed_disallows(
    pairs: Vec<(String, String)>,
) -> PyResult<HashSet<(String, String)>> {
    let mut out = HashSet::with_capacity_and_hasher(pairs.len(), Default::default());
    for (left, right) in pairs {
        crate::raw_arrow::readers::insert_canonical_cluster_seed_disallow(
            &mut out,
            left,
            right,
            "additional_cluster_seed_disallows",
        )?;
    }
    Ok(out)
}

#[cfg(test)]
mod additional_cluster_seed_disallow_tests {
    use super::{
        raw_arrow_excluded_candidate_indices_by_query, validate_additional_cluster_seed_disallows,
    };
    use crate::{HashMap, HashSet};

    #[test]
    fn additional_pairs_are_canonicalized() {
        let pairs = validate_additional_cluster_seed_disallows(vec![(
            "seed-z".to_string(),
            "query-a".to_string(),
        )])
        .expect("valid pair");

        assert!(pairs.contains(&("query-a".to_string(), "seed-z".to_string())));
    }

    #[test]
    fn additional_pairs_reject_invalid_and_reversed_duplicates() {
        assert!(validate_additional_cluster_seed_disallows(vec![(
            "same".to_string(),
            "same".to_string(),
        )])
        .is_err());
        assert!(validate_additional_cluster_seed_disallows(vec![(
            String::new(),
            "seed".to_string(),
        )])
        .is_err());
        assert!(validate_additional_cluster_seed_disallows(vec![
            ("query".to_string(), "seed".to_string()),
            ("seed".to_string(), "query".to_string()),
        ])
        .is_err());
    }

    #[test]
    fn stored_and_additional_pairs_are_streamed_as_an_exact_union() {
        let query_ids = vec!["query".to_string()];
        let component_order = vec!["component".to_string()];
        let component_keys_by_member = HashMap::from_iter([
            ("seed-a".to_string(), vec!["component".to_string()]),
            ("seed-b".to_string(), vec!["component".to_string()]),
        ]);
        let stored = HashSet::from_iter([("query".to_string(), "seed-a".to_string())]);
        let additional = HashSet::from_iter([
            ("query".to_string(), "seed-a".to_string()),
            ("query".to_string(), "seed-b".to_string()),
        ]);

        let (excluded, candidate_count, pair_count) =
            raw_arrow_excluded_candidate_indices_by_query(
                &query_ids,
                &component_order,
                &component_keys_by_member,
                &stored,
                Some(&additional),
            )
            .expect("valid streamed union");

        assert_eq!(pair_count, 2);
        assert_eq!(candidate_count, 1);
        assert_eq!(
            excluded.expect("one query exclusion")[0],
            Some(HashSet::from_iter([0]))
        );
    }
}

fn validate_required_raw_arrow_paper_metadata(
    required_paper_ids: &HashSet<String>,
    papers: &HashMap<String, RawArrowPaper>,
    num_threads: Option<usize>,
) -> PyResult<()> {
    // Report the smallest missing id, as a sorted scan would, without sorting
    // every id when none is missing.
    let missing = install_with_optional_rayon_pool(num_threads, || {
        required_paper_ids
            .par_iter()
            .filter(|paper_id| !papers.contains_key(*paper_id))
            .min()
    });
    if let Some(paper_id) = missing {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Arrow signatures reference missing paper_id '{paper_id}' in papers Arrow input"
        )));
    }
    Ok(())
}

/// Queries scored together against each summary in `plan()`; bounds the
/// precomputed SPECTER rows to this many times the summary count.
const RETRIEVAL_QUERY_BLOCK: usize = 32;

fn build_retriever_from_raw_arrow_components(
    py: Python<'_>,
    component_order: &[String],
    members_by_component: &HashMap<String, Vec<String>>,
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    max_exemplars: usize,
    num_threads: Option<usize>,
    ranks_candidates: bool,
) -> PyResult<RustHybridCentroidRetriever> {
    let summary_results: Vec<Result<RetrievalSummaryData, RetrievalError>> =
        py.allow_threads(|| {
            let compute = || {
                component_order
                    .par_iter()
                    .map(|component_key| {
                        let members = members_by_component.get(component_key).ok_or_else(|| {
                            RetrievalError::MissingKey(format!(
                                "component_key '{}' disappeared while building summaries",
                                component_key
                            ))
                        })?;
                        build_raw_arrow_summary(
                            component_key,
                            members,
                            features_by_signature_id,
                            max_exemplars,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
    let summaries = summary_results
        .into_iter()
        .collect::<Result<Vec<RetrievalSummaryData>, RetrievalError>>()
        .map_err(retrieval_error_to_py)?;
    py.allow_threads(|| {
        install_with_optional_rayon_pool(num_threads, || {
            RustHybridCentroidRetriever::from_summaries(summaries, ranks_candidates)
        })
    })
    .map_err(retrieval_error_to_py)
}

/// Build the retriever like `build_retriever_from_raw_arrow_components`, moving
/// over each summary whose component has the same ordered members as before.
///
/// Returns the retriever and the number of reused summaries.
#[allow(clippy::too_many_arguments)]
fn build_retriever_reusing_summaries(
    py: Python<'_>,
    component_order: &[String],
    members_by_component: &HashMap<String, Vec<String>>,
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    max_exemplars: usize,
    num_threads: Option<usize>,
    previous_members_by_component: &HashMap<String, Vec<String>>,
    previous_component_keys_by_member: &HashMap<String, Vec<String>>,
    previous_summaries: Vec<RetrievalSummaryData>,
    previous_component_index_by_key: &HashMap<String, usize>,
) -> PyResult<(RustHybridCentroidRetriever, usize)> {
    let mut previous_summaries: Vec<Option<RetrievalSummaryData>> =
        previous_summaries.into_iter().map(Some).collect();
    // A component's first member names the previous component it could match.
    let mut slots: Vec<Option<RetrievalSummaryData>> = component_order
        .iter()
        .map(|component_key| {
            let members = members_by_component.get(component_key)?;
            let previous_key = previous_component_keys_by_member
                .get(members.first()?)?
                .first()?;
            if previous_members_by_component.get(previous_key)? != members {
                return None;
            }
            let index = *previous_component_index_by_key.get(previous_key)?;
            let mut summary = previous_summaries.get_mut(index)?.take()?;
            summary.component_key = component_key.clone();
            Some(summary)
        })
        .collect();
    drop(previous_summaries);
    let reused_count = slots.iter().filter(|slot| slot.is_some()).count();
    let pending: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| slot.is_none().then_some(index))
        .collect();
    let computed: Vec<Result<RetrievalSummaryData, RetrievalError>> = py.allow_threads(|| {
        install_with_optional_rayon_pool(num_threads, || {
            pending
                .par_iter()
                .map(|&index| {
                    let component_key = &component_order[index];
                    let members = members_by_component.get(component_key).ok_or_else(|| {
                        RetrievalError::MissingKey(format!(
                            "component_key '{}' disappeared while building summaries",
                            component_key
                        ))
                    })?;
                    build_raw_arrow_summary(
                        component_key,
                        members,
                        features_by_signature_id,
                        max_exemplars,
                    )
                })
                .collect()
        })
    });
    for (index, summary) in pending.into_iter().zip(computed) {
        slots[index] = Some(summary.map_err(retrieval_error_to_py)?);
    }
    let summaries: Vec<RetrievalSummaryData> = slots
        .into_iter()
        .map(|slot| slot.expect("every component has a reused or computed summary"))
        .collect();
    let retriever = py
        .allow_threads(|| {
            install_with_optional_rayon_pool(num_threads, || {
                RustHybridCentroidRetriever::from_summaries(summaries, true)
            })
        })
        .map_err(retrieval_error_to_py)?;
    Ok((retriever, reused_count))
}

fn raw_arrow_component_member_indices_for_batch(
    component_order: &[String],
    members_by_component: &HashMap<String, Vec<String>>,
    query_count: usize,
) -> PyResult<(HashMap<String, Vec<u32>>, Vec<String>, Vec<String>)> {
    let mut component_member_indices = HashMap::<String, Vec<u32>>::default();
    let mut seed_signature_ids = Vec::<String>::new();
    let mut seed_component_keys = Vec::<String>::new();
    let query_count_u32 = u32::try_from(query_count)
        .map_err(|_| pyo3::exceptions::PyOverflowError::new_err("query count exceeds u32"))?;
    for component_key in component_order.iter() {
        let mut member_indices = Vec::<u32>::new();
        if let Some(members) = members_by_component.get(component_key) {
            for signature_id in members.iter() {
                let seed_offset = u32::try_from(seed_signature_ids.len()).map_err(|_| {
                    pyo3::exceptions::PyOverflowError::new_err("seed signature count exceeds u32")
                })?;
                let member_index = query_count_u32.checked_add(seed_offset).ok_or_else(|| {
                    pyo3::exceptions::PyOverflowError::new_err("signature index exceeds u32")
                })?;
                member_indices.push(member_index);
                seed_signature_ids.push(signature_id.clone());
                seed_component_keys.push(component_key.clone());
            }
        }
        component_member_indices.insert(component_key.clone(), member_indices);
    }
    Ok((
        component_member_indices,
        seed_signature_ids,
        seed_component_keys,
    ))
}

fn raw_arrow_excluded_candidate_indices_by_query(
    query_signature_ids: &[String],
    component_order: &[String],
    component_keys_by_member: &HashMap<String, Vec<String>>,
    cluster_seed_disallows: &HashSet<(String, String)>,
    additional_cluster_seed_disallows: Option<&HashSet<(String, String)>>,
) -> PyResult<(Option<Vec<Option<HashSet<usize>>>>, usize, usize)> {
    let query_signature_id_set: HashSet<&str> =
        query_signature_ids.iter().map(String::as_str).collect();
    let mut disallowed_members_by_query = HashMap::<String, HashSet<String>>::default();
    let mut register_pair = |left: &String, right: &String| -> PyResult<()> {
        let left_is_query = query_signature_id_set.contains(left.as_str());
        let right_is_query = query_signature_id_set.contains(right.as_str());
        let left_is_seed = component_keys_by_member.contains_key(left);
        let right_is_seed = component_keys_by_member.contains_key(right);
        if left_is_query && !right_is_query && !right_is_seed {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cluster_seed_disallows pair references unknown seed endpoint for query {left:?}: {right:?}"
            )));
        }
        if right_is_query && !left_is_query && !left_is_seed {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cluster_seed_disallows pair references unknown seed endpoint for query {right:?}: {left:?}"
            )));
        }
        if left_is_query {
            disallowed_members_by_query
                .entry(left.clone())
                .or_default()
                .insert(right.clone());
        }
        if right_is_query {
            disallowed_members_by_query
                .entry(right.clone())
                .or_default()
                .insert(left.clone());
        }
        Ok(())
    };
    for (left, right) in cluster_seed_disallows.iter() {
        register_pair(left, right)?;
    }
    let mut cluster_seed_disallow_pair_count = cluster_seed_disallows.len();
    if let Some(additional) = additional_cluster_seed_disallows {
        for pair in additional.iter() {
            if cluster_seed_disallows.contains(pair) {
                continue;
            }
            register_pair(&pair.0, &pair.1)?;
            cluster_seed_disallow_pair_count += 1;
        }
    }
    let excluded_indices_by_query = if disallowed_members_by_query.is_empty() {
        None
    } else {
        let mut component_index_by_key = HashMap::<&str, usize>::with_capacity_and_hasher(
            component_order.len(),
            Default::default(),
        );
        for (component_index, component_key) in component_order.iter().enumerate() {
            component_index_by_key.insert(component_key.as_str(), component_index);
        }
        Some(
            query_signature_ids
                .iter()
                .map(|query_signature_id| {
                    let Some(disallowed_members) =
                        disallowed_members_by_query.get(query_signature_id)
                    else {
                        return None;
                    };
                    let mut excluded_indices = HashSet::<usize>::default();
                    for disallowed_member in disallowed_members.iter() {
                        if let Some(component_keys) =
                            component_keys_by_member.get(disallowed_member.as_str())
                        {
                            for component_key in component_keys {
                                if let Some(component_index) =
                                    component_index_by_key.get(component_key.as_str())
                                {
                                    excluded_indices.insert(*component_index);
                                }
                            }
                        }
                    }
                    Some(excluded_indices)
                })
                .collect::<Vec<_>>(),
        )
    };
    let cluster_seed_disallowed_candidate_count =
        excluded_indices_by_query
            .as_ref()
            .map_or(0usize, |indices_by_query| {
                indices_by_query
                    .iter()
                    .filter_map(|excluded| excluded.as_ref())
                    .map(HashSet::len)
                    .sum()
            });
    Ok((
        excluded_indices_by_query,
        cluster_seed_disallowed_candidate_count,
        cluster_seed_disallow_pair_count,
    ))
}

fn raw_arrow_summary_signals_cached<'a>(
    cache: &'a mut HashMap<String, RawArrowSummarySignalData>,
    component_key: &str,
    members_by_component: &HashMap<String, Vec<String>>,
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    signatures: &HashMap<String, RawArrowSignature>,
    author_index: &PaperAuthorIndex,
) -> PyResult<&'a RawArrowSummarySignalData> {
    if let Entry::Vacant(entry) = cache.entry(component_key.to_string()) {
        let members = members_by_component.get(component_key).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "component_key '{}' disappeared while building row signals",
                component_key
            ))
        })?;
        let summary_signals = build_raw_arrow_summary_signals(
            members,
            features_by_signature_id,
            signatures,
            author_index,
        )
        .map_err(pyo3::exceptions::PyKeyError::new_err)?;
        entry.insert(summary_signals);
    }
    cache.get(component_key).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "component_key '{}' is missing raw row signal summary",
            component_key
        ))
    })
}

pub(crate) fn raw_arrow_summary_signals_for_members_cached<'a, S: AsRef<str>>(
    cache: &'a mut HashMap<(String, String), RawArrowSummarySignalData>,
    component_key: &str,
    excluded_query_signature_id: &str,
    members: &[S],
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    signatures: &HashMap<String, RawArrowSignature>,
    author_index: &PaperAuthorIndex,
) -> PyResult<&'a RawArrowSummarySignalData> {
    let cache_key = (
        component_key.to_string(),
        excluded_query_signature_id.to_string(),
    );
    if let Entry::Vacant(entry) = cache.entry(cache_key.clone()) {
        let summary_signals = build_raw_arrow_summary_signals(
            members,
            features_by_signature_id,
            signatures,
            author_index,
        )
        .map_err(pyo3::exceptions::PyKeyError::new_err)?;
        entry.insert(summary_signals);
    }
    cache.get(&cache_key).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "component_key {:?} excluding query signature_id {:?} is missing raw row signal summary",
            component_key, excluded_query_signature_id
        ))
    })
}

fn read_reusable_raw_arrow_query_inputs(
    py: Python<'_>,
    dataset: &ArrowDatasetResources,
    query_signature_ids: &[String],
    num_threads: Option<usize>,
) -> PyResult<RawArrowQueryInputReadResult> {
    let query_signature_id_set: HashSet<String> = query_signature_ids.iter().cloned().collect();
    let read_signatures_start = Instant::now();
    let (signatures, signature_index_stats) =
        py.allow_threads(|| dataset.read_signatures(Some(&query_signature_id_set), num_threads))?;
    let read_signatures_secs = read_signatures_start.elapsed().as_secs_f64();
    for signature_id in query_signature_ids.iter() {
        if !signatures.contains_key(signature_id) {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "query signature_id '{}' is missing from signatures Arrow input",
                signature_id
            )));
        }
    }
    let needed_paper_ids: HashSet<String> = query_signature_ids
        .iter()
        .filter_map(|signature_id| signatures.get(signature_id))
        .map(|signature| signature.paper_id.clone())
        .collect();
    let metadata_reads_parallel_start = Instant::now();
    let ((papers_result, paper_authors_result), raw_specter_by_paper_id_result) =
        py.allow_threads(|| {
            let compute = || {
                rayon::join(
                    || {
                        rayon::join(
                            || -> PyResult<(
                                HashMap<String, RawArrowPaper>,
                                IndexedArrowReadStats,
                                f64,
                            )> {
                                let start = Instant::now();
                                let (loaded, stats) = dataset.read_papers(&needed_paper_ids, num_threads)?;
                                Ok((loaded, stats, start.elapsed().as_secs_f64()))
                            },
                            || -> PyResult<(
                                HashMap<String, Vec<(i64, String)>>,
                                IndexedArrowReadStats,
                                f64,
                            )> {
                                let start = Instant::now();
                                let (loaded, stats) =
                                    dataset.read_paper_authors(&needed_paper_ids, num_threads)?;
                                Ok((loaded, stats, start.elapsed().as_secs_f64()))
                            },
                        )
                    },
                    || -> PyResult<(
                        Option<HashMap<String, Vec<f32>>>,
                        IndexedArrowReadStats,
                        f64,
                    )> {
                        let start = Instant::now();
                        let (loaded, stats) = if dataset.has_specter() {
                            dataset.read_specter(&needed_paper_ids, num_threads)?
                        } else {
                            (HashMap::default(), IndexedArrowReadStats::default())
                        };
                        Ok((
                            if dataset.has_specter() {
                                Some(loaded)
                            } else {
                                None
                            },
                            stats,
                            start.elapsed().as_secs_f64(),
                        ))
                    },
                )
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
    let (papers, paper_index_stats, read_papers_secs) = papers_result?;
    let (paper_authors, paper_author_index_stats, read_paper_authors_secs) = paper_authors_result?;
    validate_required_raw_arrow_paper_metadata(&needed_paper_ids, &papers, num_threads)?;
    let (raw_specter_by_paper_id, specter_index_stats, read_specter_secs) =
        raw_specter_by_paper_id_result?;
    let specter_by_paper_id = raw_specter_by_paper_id.map(|values| {
        values
            .into_iter()
            .map(|(paper_id, vector)| (paper_id, Arc::new(vector)))
            .collect::<HashMap<_, _>>()
    });
    Ok(RawArrowQueryInputReadResult {
        signatures,
        papers,
        paper_authors,
        specter_by_paper_id,
        signature_index_stats,
        paper_index_stats,
        paper_author_index_stats,
        specter_index_stats,
        read_signatures_secs,
        read_papers_secs,
        read_paper_authors_secs,
        read_specter_secs,
        metadata_reads_parallel_secs: metadata_reads_parallel_start.elapsed().as_secs_f64(),
    })
}

#[derive(Clone, Copy)]
enum RawPlannerQueryMode {
    Declared,
    Auto,
}

#[pyclass]
pub(crate) struct RawBlockQueryCandidatePlanner {
    dataset: ArrowDatasetResources,
    state: ReusableRawArrowCandidatePlanState,
    planner_query_signature_ids: Vec<String>,
    planner_query_signature_count: usize,
    planner_query_signature_id_set: HashSet<String>,
    planner_query_requests_by_signature_id: HashMap<String, RawArrowQuerySignatureRequest>,
    query_mode: RawPlannerQueryMode,
    top_k: usize,
    orcid_enabled: bool,
    num_threads: Option<usize>,
    max_exemplars: usize,
    /// Set by `release`; the planner then refuses to plan or reload.
    released: bool,
}

/// Seed components and exclusions read from one request's sidecars.
struct RawArrowSeedLayout {
    component_order: Vec<String>,
    members_by_component: HashMap<String, Vec<String>>,
    component_keys_by_member: HashMap<String, Vec<String>>,
    seed_signature_ids: Vec<String>,
    seed_signature_id_set: HashSet<String>,
    cluster_seed_disallows: HashSet<(String, String)>,
    read_cluster_seeds_secs: f64,
}

impl RawArrowSeedLayout {
    fn read(cluster_seeds_path: &str, cluster_seed_disallows_path: Option<&str>) -> PyResult<Self> {
        let read_cluster_seeds_start = Instant::now();
        let (component_order, members_by_component) =
            read_raw_arrow_cluster_seeds(cluster_seeds_path)?;
        let read_cluster_seeds_secs = read_cluster_seeds_start.elapsed().as_secs_f64();
        let cluster_seed_disallows = match cluster_seed_disallows_path {
            Some(path) => read_raw_arrow_cluster_seed_disallows(path)?,
            None => HashSet::default(),
        };

        let mut seed_signature_ids = Vec::<String>::new();
        let mut component_keys_by_member = HashMap::<String, Vec<String>>::default();
        for component_key in component_order.iter() {
            if let Some(members) = members_by_component.get(component_key) {
                for signature_id in members {
                    if let Some(existing) = component_keys_by_member.get_mut(signature_id) {
                        existing.push(component_key.clone());
                    } else {
                        let owned_id = signature_id.clone();
                        component_keys_by_member
                            .insert(owned_id.clone(), vec![component_key.clone()]);
                        seed_signature_ids.push(owned_id);
                    }
                }
            }
        }
        let seed_signature_id_set: HashSet<String> = seed_signature_ids.iter().cloned().collect();
        Ok(Self {
            component_order,
            members_by_component,
            component_keys_by_member,
            seed_signature_ids,
            seed_signature_id_set,
            cluster_seed_disallows,
            read_cluster_seeds_secs,
        })
    }

    /// Fail on the first seed, in seed order, that has no signature row.
    fn require_signature_rows(
        &self,
        signatures: &HashMap<String, RawArrowSignature>,
        num_threads: Option<usize>,
    ) -> PyResult<()> {
        // The first missing seed in seed order, as the serial loop reported it.
        let missing = install_with_optional_rayon_pool(num_threads, || {
            self.seed_signature_ids
                .par_iter()
                .find_first(|signature_id| !signatures.contains_key(*signature_id))
        });
        if let Some(signature_id) = missing {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "cluster seed signature_id '{}' is missing from signatures Arrow input",
                signature_id
            )));
        }
        Ok(())
    }

    fn needed_paper_ids(&self, signatures: &HashMap<String, RawArrowSignature>) -> HashSet<String> {
        self.seed_signature_ids
            .iter()
            .filter_map(|signature_id| signatures.get(signature_id))
            .map(|signature| signature.paper_id.clone())
            .collect()
    }
}

/// Paper, author, and SPECTER rows read in parallel for one set of papers.
struct RawArrowPaperRows {
    papers: HashMap<String, RawArrowPaper>,
    paper_authors: HashMap<String, Vec<(i64, String)>>,
    specter_by_paper_id: Option<HashMap<String, Arc<Vec<f32>>>>,
    raw_name_counts: RawNameCountMaps,
    paper_index_stats: IndexedArrowReadStats,
    paper_author_index_stats: IndexedArrowReadStats,
    specter_index_stats: IndexedArrowReadStats,
    read_papers_secs: f64,
    read_paper_authors_secs: f64,
    read_specter_secs: f64,
    read_name_counts_secs: f64,
}

fn read_raw_arrow_paper_rows(
    py: Python<'_>,
    dataset: &ArrowDatasetResources,
    raw_name_counts: &RawNameCountMaps,
    needed_paper_ids: &HashSet<String>,
    num_threads: Option<usize>,
) -> PyResult<RawArrowPaperRows> {
    let (
        (papers_result, paper_authors_result),
        (raw_specter_by_paper_id_result, raw_name_counts_result),
    ) = py.allow_threads(|| {
        let compute = || {
            rayon::join(
                || {
                    rayon::join(
                        || -> PyResult<(
                            HashMap<String, RawArrowPaper>,
                            IndexedArrowReadStats,
                            f64,
                        )> {
                            let start = Instant::now();
                            let (loaded, stats) = if needed_paper_ids.is_empty() {
                                (HashMap::default(), IndexedArrowReadStats::default())
                            } else {
                                dataset.read_papers(needed_paper_ids, num_threads)?
                            };
                            Ok((loaded, stats, start.elapsed().as_secs_f64()))
                        },
                        || -> PyResult<(
                            HashMap<String, Vec<(i64, String)>>,
                            IndexedArrowReadStats,
                            f64,
                        )> {
                            let start = Instant::now();
                            let (loaded, stats) = if needed_paper_ids.is_empty() {
                                (HashMap::default(), IndexedArrowReadStats::default())
                            } else {
                                dataset.read_paper_authors(needed_paper_ids, num_threads)?
                            };
                            Ok((loaded, stats, start.elapsed().as_secs_f64()))
                        },
                    )
                },
                || {
                    rayon::join(
                        || -> PyResult<(
                            Option<HashMap<String, Vec<f32>>>,
                            IndexedArrowReadStats,
                            f64,
                        )> {
                            let start = Instant::now();
                            let (loaded, stats) =
                                if dataset.has_specter() && !needed_paper_ids.is_empty() {
                                    dataset.read_specter(needed_paper_ids, num_threads)?
                                } else {
                                    (HashMap::default(), IndexedArrowReadStats::default())
                                };
                            Ok((
                                if dataset.has_specter() {
                                    Some(loaded)
                                } else {
                                    None
                                },
                                stats,
                                start.elapsed().as_secs_f64(),
                            ))
                        },
                        || -> PyResult<(RawNameCountMaps, f64)> {
                            let start = Instant::now();
                            Ok((raw_name_counts.clone(), start.elapsed().as_secs_f64()))
                        },
                    )
                },
            )
        };
        install_with_optional_rayon_pool(num_threads, compute)
    });
    let (papers, paper_index_stats, read_papers_secs) = papers_result?;
    let (paper_authors, paper_author_index_stats, read_paper_authors_secs) = paper_authors_result?;
    let (raw_specter_by_paper_id, specter_index_stats, read_specter_secs) =
        raw_specter_by_paper_id_result?;
    let (raw_name_counts, read_name_counts_secs) = raw_name_counts_result?;
    let specter_by_paper_id = raw_specter_by_paper_id.map(|values| {
        values
            .into_iter()
            .map(|(paper_id, vector)| (paper_id, Arc::new(vector)))
            .collect::<HashMap<_, _>>()
    });
    Ok(RawArrowPaperRows {
        papers,
        paper_authors,
        specter_by_paper_id,
        raw_name_counts,
        paper_index_stats,
        paper_author_index_stats,
        specter_index_stats,
        read_papers_secs,
        read_paper_authors_secs,
        read_specter_secs,
        read_name_counts_secs,
    })
}

/// Raw rows a planner keeps between attachment passes when built with
/// `retain_seed_rows`; the signature and author rows stay in the plan state.
struct RetainedRawArrowSeedRows {
    papers: HashMap<String, RawArrowPaper>,
    specter_by_paper_id: Option<HashMap<String, Arc<Vec<f32>>>>,
}

/// Rows moved out of the planner while a reload reads the new seeds.
struct ReloadParts {
    signatures: HashMap<String, RawArrowSignature>,
    paper_authors: HashMap<String, Vec<(i64, String)>>,
    papers: HashMap<String, RawArrowPaper>,
    specter_by_paper_id: Option<HashMap<String, Arc<Vec<f32>>>>,
    prefetched: Option<PrefetchedRawArrowRows>,
}

/// Rows of block signatures that are not seeds yet, read with the seeds at build.
///
/// Subblocked attachment passes plan each initial-only group as queries, then
/// hold it as seeds. Their papers are scattered across the Arrow batches, so
/// reading them per pass rescans almost every batch twice. Served from here,
/// `plan()` copies query rows and a reload moves new seeds' rows; both equal a
/// read, because they come from the same readers over the same dataset.
struct PrefetchedRawArrowRows {
    signatures: HashMap<String, RawArrowSignature>,
    papers: HashMap<String, RawArrowPaper>,
    paper_authors: HashMap<String, Vec<(i64, String)>>,
    specter_by_paper_id: Option<HashMap<String, Arc<Vec<f32>>>>,
}

impl PrefetchedRawArrowRows {
    /// The query inputs a read would return, or None unless every row is here.
    fn query_inputs(&self, query_signature_ids: &[String]) -> Option<RawArrowQueryInputReadResult> {
        let mut signatures =
            HashMap::with_capacity_and_hasher(query_signature_ids.len(), Default::default());
        for signature_id in query_signature_ids {
            let signature = self.signatures.get(signature_id)?;
            signatures.insert(signature_id.clone(), signature.clone());
        }
        let mut papers = HashMap::default();
        let mut paper_authors = HashMap::default();
        let mut specter_by_paper_id = self
            .specter_by_paper_id
            .as_ref()
            .map(|_| HashMap::default());
        for signature in signatures.values() {
            let paper_id = &signature.paper_id;
            if papers.contains_key(paper_id) {
                continue;
            }
            // A cached paper carries all of its rows: absent authors or SPECTER
            // rows are absent from the dataset too.
            papers.insert(paper_id.clone(), self.papers.get(paper_id)?.clone());
            if let Some(authors) = self.paper_authors.get(paper_id) {
                paper_authors.insert(paper_id.clone(), authors.clone());
            }
            if let (Some(cached), Some(out)) = (
                self.specter_by_paper_id.as_ref(),
                specter_by_paper_id.as_mut(),
            ) {
                if let Some(vector) = cached.get(paper_id) {
                    out.insert(paper_id.clone(), Arc::clone(vector));
                }
            }
        }
        Some(RawArrowQueryInputReadResult {
            signatures,
            papers,
            paper_authors,
            specter_by_paper_id,
            signature_index_stats: IndexedArrowReadStats::default(),
            paper_index_stats: IndexedArrowReadStats::default(),
            paper_author_index_stats: IndexedArrowReadStats::default(),
            specter_index_stats: IndexedArrowReadStats::default(),
            read_signatures_secs: 0.0,
            read_papers_secs: 0.0,
            read_paper_authors_secs: 0.0,
            read_specter_secs: 0.0,
            metadata_reads_parallel_secs: 0.0,
        })
    }
}

/// Move the rows of `prefetch` papers out of `rows`; copy those `seed_needed` keeps.
fn split_prefetched_rows<V: Clone>(
    rows: &mut HashMap<String, V>,
    prefetch: &HashSet<String>,
    seed_needed: &HashSet<String>,
) -> HashMap<String, V> {
    prefetch
        .iter()
        .filter_map(|paper_id| {
            let row = if seed_needed.contains(paper_id) {
                rows.get(paper_id).cloned()
            } else {
                rows.remove(paper_id)
            };
            row.map(|row| (paper_id.clone(), row))
        })
        .collect()
}

/// Plan state a reload reuses instead of recomputing.
///
/// A seed's feature depends only on its own rows, so features of seeds still
/// present stay valid. A component summary depends only on its key and its
/// ordered members' features, so a component whose ordered member list is
/// unchanged keeps its summary under its new key.
struct ReusableSeedState {
    features_by_signature_id: HashMap<String, RawArrowFeature>,
    members_by_component: HashMap<String, Vec<String>>,
    component_keys_by_member: HashMap<String, Vec<String>>,
    summaries: Vec<RetrievalSummaryData>,
    component_index_by_key: HashMap<String, usize>,
    unidecode_char_map: HashMap<char, String>,
}

/// Read timings of one build or reload, reported by `build_telemetry`.
struct RawArrowSeedReadTelemetry {
    read_cluster_seeds_secs: f64,
    read_signatures_secs: f64,
    read_papers_secs: f64,
    read_paper_authors_secs: f64,
    read_specter_secs: f64,
    read_name_counts_secs: f64,
    signature_index_stats: IndexedArrowReadStats,
    paper_index_stats: IndexedArrowReadStats,
    paper_author_index_stats: IndexedArrowReadStats,
    specter_index_stats: IndexedArrowReadStats,
}

impl RawBlockQueryCandidatePlanner {
    fn ensure_not_released(&self) -> PyResult<()> {
        if self.released {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "planner was released; build a new one",
            ));
        }
        Ok(())
    }

    /// Rows of exactly `signature_ids` and their papers from the seed state, the
    /// latest plan's queries, and prefetched rows; None if any is not held.
    ///
    /// Each source holds all of a paper's rows: a paper without author or
    /// SPECTER rows there has none in the dataset either.
    fn featurizer_rows(&self, signature_ids: &[String]) -> Option<RawFeaturizerRows> {
        let retained = self.state.retained_seed_rows.as_ref()?;
        let query = self.state.last_query_inputs.as_ref();
        let prefetched = self.state.prefetched_rows.as_ref();
        // Rows are cloned in parallel: windows cover hundreds of thousands of
        // signatures, and the serial copies dominated the featurizer build.
        let signature_rows = signature_ids
            .par_iter()
            .map(|signature_id| {
                let signature = self
                    .state
                    .signatures
                    .get(signature_id)
                    .or_else(|| query.and_then(|rows| rows.signatures.get(signature_id)))
                    .or_else(|| prefetched.and_then(|rows| rows.signatures.get(signature_id)))?;
                Some((signature_id.clone(), signature.clone()))
            })
            .collect::<Option<Vec<_>>>()?;
        let mut paper_ids: Vec<&str> = signature_rows
            .iter()
            .map(|(_, signature)| signature.paper_id.as_str())
            .collect();
        paper_ids.par_sort_unstable();
        paper_ids.dedup();
        let paper_rows = paper_ids
            .par_iter()
            .map(|paper_id| {
                // The seed state, then the latest queries, then prefetched rows.
                let (paper, authors, specter) = if let Some(paper) = retained.papers.get(*paper_id)
                {
                    (
                        paper,
                        self.state.paper_authors.get(*paper_id),
                        retained
                            .specter_by_paper_id
                            .as_ref()
                            .and_then(|specter| specter.get(*paper_id)),
                    )
                } else if let Some(rows) = query.filter(|rows| rows.papers.contains_key(*paper_id))
                {
                    (
                        &rows.papers[*paper_id],
                        rows.paper_authors.get(*paper_id),
                        rows.specter_by_paper_id
                            .as_ref()
                            .and_then(|specter| specter.get(*paper_id)),
                    )
                } else {
                    let rows = prefetched?;
                    (
                        rows.papers.get(*paper_id)?,
                        rows.paper_authors.get(*paper_id),
                        rows.specter_by_paper_id
                            .as_ref()
                            .and_then(|specter| specter.get(*paper_id)),
                    )
                };
                Some((
                    paper_id.to_string(),
                    paper.clone(),
                    authors.cloned(),
                    specter.map(|vector| vector.as_ref().clone()),
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        let signatures: HashMap<String, RawArrowSignature> = signature_rows.into_iter().collect();
        let mut papers = HashMap::with_capacity_and_hasher(paper_rows.len(), Default::default());
        let mut authors_by_paper =
            HashMap::with_capacity_and_hasher(paper_rows.len(), Default::default());
        let mut specter_by_paper =
            HashMap::with_capacity_and_hasher(paper_rows.len(), Default::default());
        for (paper_id, paper, authors, specter) in paper_rows {
            if let Some(authors) = authors {
                authors_by_paper.insert(paper_id.clone(), authors);
            }
            if let Some(vector) = specter {
                specter_by_paper.insert(paper_id.clone(), vector);
            }
            papers.insert(paper_id, paper);
        }
        Some(RawFeaturizerRows {
            signatures,
            papers,
            authors_by_paper,
            specter_by_paper,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_from_query_signature_ids(
        py: Python<'_>,
        dataset: ArrowDatasetResources,
        cluster_seeds_path: &str,
        cluster_seed_disallows_path: Option<&str>,
        query_signature_ids: Vec<String>,
        top_k: usize,
        orcid_enabled: bool,
        num_threads: Option<usize>,
        max_exemplars: usize,
        retain_seed_rows: bool,
        prefetch_signature_ids: Vec<String>,
    ) -> PyResult<Self> {
        validate_retrieval_rank_top_k(top_k)?;
        validate_raw_arrow_query_signature_ids_allow_empty(&query_signature_ids)?;
        let raw_name_counts = dataset.name_counts();

        let layout = RawArrowSeedLayout::read(cluster_seeds_path, cluster_seed_disallows_path)?;
        let prefetch_ids: HashSet<String> = prefetch_signature_ids
            .into_iter()
            .filter(|signature_id| !layout.seed_signature_id_set.contains(signature_id))
            .collect();

        let read_signatures_start = Instant::now();
        let (mut signatures, signature_index_stats) =
            if layout.seed_signature_id_set.is_empty() && prefetch_ids.is_empty() {
                (HashMap::default(), IndexedArrowReadStats::default())
            } else if prefetch_ids.is_empty() {
                py.allow_threads(|| {
                    dataset.read_signatures(Some(&layout.seed_signature_id_set), num_threads)
                })?
            } else {
                let wanted: HashSet<String> = layout
                    .seed_signature_id_set
                    .union(&prefetch_ids)
                    .cloned()
                    .collect();
                py.allow_threads(|| dataset.read_signatures(Some(&wanted), num_threads))?
            };
        let prefetched_signatures: HashMap<String, RawArrowSignature> = prefetch_ids
            .iter()
            .filter_map(|signature_id| {
                signatures
                    .remove(signature_id)
                    .map(|signature| (signature_id.clone(), signature))
            })
            .collect();
        let read_signatures_secs = read_signatures_start.elapsed().as_secs_f64();
        layout.require_signature_rows(&signatures, num_threads)?;

        let needed_paper_ids = layout.needed_paper_ids(&signatures);
        let prefetch_paper_ids: HashSet<String> = prefetched_signatures
            .values()
            .map(|signature| signature.paper_id.clone())
            .collect();
        let metadata_reads_parallel_start = Instant::now();
        let mut rows = if prefetch_paper_ids.is_empty() {
            read_raw_arrow_paper_rows(
                py,
                &dataset,
                &raw_name_counts,
                &needed_paper_ids,
                num_threads,
            )?
        } else {
            let wanted: HashSet<String> = needed_paper_ids
                .union(&prefetch_paper_ids)
                .cloned()
                .collect();
            read_raw_arrow_paper_rows(py, &dataset, &raw_name_counts, &wanted, num_threads)?
        };
        let prefetched_rows = (!prefetched_signatures.is_empty()).then(|| PrefetchedRawArrowRows {
            signatures: prefetched_signatures,
            papers: split_prefetched_rows(&mut rows.papers, &prefetch_paper_ids, &needed_paper_ids),
            paper_authors: split_prefetched_rows(
                &mut rows.paper_authors,
                &prefetch_paper_ids,
                &needed_paper_ids,
            ),
            specter_by_paper_id: rows.specter_by_paper_id.as_mut().map(|specter| {
                split_prefetched_rows(specter, &prefetch_paper_ids, &needed_paper_ids)
            }),
        });
        validate_required_raw_arrow_paper_metadata(&needed_paper_ids, &rows.papers, num_threads)?;
        let read_telemetry = RawArrowSeedReadTelemetry {
            read_cluster_seeds_secs: layout.read_cluster_seeds_secs,
            read_signatures_secs,
            read_papers_secs: rows.read_papers_secs,
            read_paper_authors_secs: rows.read_paper_authors_secs,
            read_specter_secs: rows.read_specter_secs,
            read_name_counts_secs: rows.read_name_counts_secs,
            signature_index_stats,
            paper_index_stats: rows.paper_index_stats,
            paper_author_index_stats: rows.paper_author_index_stats,
            specter_index_stats: rows.specter_index_stats,
        };
        Self::assemble(
            py,
            dataset,
            layout,
            signatures,
            rows.papers,
            rows.paper_authors,
            rows.specter_by_paper_id,
            rows.raw_name_counts,
            needed_paper_ids.len(),
            read_telemetry,
            metadata_reads_parallel_start,
            query_signature_ids,
            top_k,
            orcid_enabled,
            num_threads,
            max_exemplars,
            retain_seed_rows,
            None,
        )
        .map(|mut planner| {
            planner.state.prefetched_rows = prefetched_rows;
            planner
        })
    }

    /// Build the plan state from complete raw rows of exactly the layout's seeds.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        py: Python<'_>,
        dataset: ArrowDatasetResources,
        layout: RawArrowSeedLayout,
        signatures: HashMap<String, RawArrowSignature>,
        papers: HashMap<String, RawArrowPaper>,
        paper_authors: HashMap<String, Vec<(i64, String)>>,
        specter_by_paper_id: Option<HashMap<String, Arc<Vec<f32>>>>,
        raw_name_counts: RawNameCountMaps,
        seed_paper_count: usize,
        read_telemetry: RawArrowSeedReadTelemetry,
        metadata_reads_parallel_start: Instant,
        query_signature_ids: Vec<String>,
        top_k: usize,
        orcid_enabled: bool,
        num_threads: Option<usize>,
        max_exemplars: usize,
        retain_seed_rows: bool,
        previous: Option<ReusableSeedState>,
    ) -> PyResult<Self> {
        let planner_query_signature_id_set = query_signature_ids.iter().cloned().collect();
        let reused_name_counts_index = raw_name_counts.shared_index().is_some();
        let RawArrowSeedLayout {
            component_order,
            members_by_component,
            component_keys_by_member,
            seed_signature_ids,
            seed_signature_id_set,
            cluster_seed_disallows,
            ..
        } = layout;

        let text_context_start = Instant::now();
        let text_module = py.import("s2and.text")?;
        let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
        let affiliation_stopwords = extract_affiliation_stopwords(py)?;
        let (mut unidecode_char_map, mut features_by_signature_id, previous_summaries) =
            match previous {
                // Transliterations are fixed per character, so the previous map stays valid.
                Some(previous) => (
                    previous.unidecode_char_map,
                    previous.features_by_signature_id,
                    Some((
                        previous.members_by_component,
                        previous.component_keys_by_member,
                        previous.summaries,
                        previous.component_index_by_key,
                    )),
                ),
                None => (HashMap::default(), HashMap::default(), None),
            };
        ensure_unidecode_for_raw_arrow_inputs(
            &signatures,
            &papers,
            &paper_authors,
            &mut unidecode_char_map,
            num_threads,
        )?;
        let text_context_secs = text_context_start.elapsed().as_secs_f64();

        let feature_start = Instant::now();
        features_by_signature_id
            .retain(|signature_id, _| seed_signature_id_set.contains(signature_id));
        let reused_feature_count = features_by_signature_id.len();
        let pending_signature_ids: Vec<&String> = seed_signature_ids
            .iter()
            .filter(|signature_id| !features_by_signature_id.contains_key(*signature_id))
            .collect();
        let raw_feature_results: Result<Vec<(String, RawArrowFeature)>, String> =
            py.allow_threads(|| {
                let compute = || {
                    // A paper's coauthor blocks depend only on its own author rows.
                    let coauthor_blocks = paper_coauthor_blocks(
                        pending_signature_ids
                            .iter()
                            .filter_map(|signature_id| signatures.get(*signature_id)),
                        &paper_authors,
                        &unidecode_char_map,
                    )?;
                    pending_signature_ids
                        .par_iter()
                        .map(|&signature_id| {
                            let signature = signatures.get(signature_id).ok_or_else(|| {
                                format!(
                                    "signature_id '{}' is missing from signatures",
                                    signature_id
                                )
                            })?;
                            let paper = papers.get(&signature.paper_id);
                            Ok((
                                signature_id.clone(),
                                build_raw_arrow_feature(
                                    signature,
                                    paper,
                                    coauthor_blocks.get(&signature.paper_id),
                                    specter_by_paper_id.as_ref(),
                                    &raw_name_counts,
                                    &name_prefixes,
                                    &affiliation_stopwords,
                                    &unidecode_char_map,
                                    orcid_enabled,
                                ),
                            ))
                        })
                        .collect()
                };
                install_with_optional_rayon_pool(num_threads, compute)
            });
        features_by_signature_id
            .extend(raw_feature_results.map_err(pyo3::exceptions::PyKeyError::new_err)?);
        let feature_secs = feature_start.elapsed().as_secs_f64();

        let summary_start = Instant::now();
        let (retriever, reused_summary_count) = match previous_summaries {
            Some((
                previous_members_by_component,
                previous_component_keys_by_member,
                previous_summaries,
                previous_component_index_by_key,
            )) => build_retriever_reusing_summaries(
                py,
                &component_order,
                &members_by_component,
                &features_by_signature_id,
                max_exemplars,
                num_threads,
                &previous_members_by_component,
                &previous_component_keys_by_member,
                previous_summaries,
                &previous_component_index_by_key,
            )?,
            None => (
                build_retriever_from_raw_arrow_components(
                    py,
                    &component_order,
                    &members_by_component,
                    &features_by_signature_id,
                    max_exemplars,
                    num_threads,
                    true,
                )?,
                0,
            ),
        };
        let summary_secs = summary_start.elapsed().as_secs_f64();

        let component_members_start = Instant::now();
        let (_component_member_indices, flat_seed_signature_ids, _seed_component_keys) =
            raw_arrow_component_member_indices_for_batch(
                &component_order,
                &members_by_component,
                0,
            )?;
        let component_members_secs = component_members_start.elapsed().as_secs_f64();

        let seed_specter_count = specter_by_paper_id.as_ref().map_or(0usize, HashMap::len);
        let retained_seed_rows = retain_seed_rows.then(|| RetainedRawArrowSeedRows {
            papers,
            specter_by_paper_id,
        });
        Ok(Self {
            dataset,
            state: ReusableRawArrowCandidatePlanState {
                features_by_signature_id,
                signatures,
                paper_authors,
                seed_author_index: PaperAuthorIndex::default(),
                raw_name_counts,
                members_by_component,
                component_keys_by_member,
                summary_signals_by_component: HashMap::default(),
                retriever,
                component_order,
                seed_signature_ids: flat_seed_signature_ids,
                seed_signature_id_set,
                cluster_seed_disallows,
                unidecode_char_map,
                name_prefixes,
                affiliation_stopwords,
                seed_paper_count,
                seed_specter_count,
                retained_seed_rows,
                prefetched_rows: None,
                last_query_inputs: None,
                build_telemetry: RawArrowPlannerBuildTelemetry {
                    read_cluster_seeds_secs: read_telemetry.read_cluster_seeds_secs,
                    read_signatures_secs: read_telemetry.read_signatures_secs,
                    read_papers_secs: read_telemetry.read_papers_secs,
                    read_paper_authors_secs: read_telemetry.read_paper_authors_secs,
                    read_specter_secs: read_telemetry.read_specter_secs,
                    read_name_counts_secs: read_telemetry.read_name_counts_secs,
                    metadata_reads_parallel_secs: metadata_reads_parallel_start
                        .elapsed()
                        .as_secs_f64(),
                    text_context_secs,
                    feature_secs,
                    summary_secs,
                    component_members_secs,
                    signature_index_stats: read_telemetry.signature_index_stats,
                    paper_index_stats: read_telemetry.paper_index_stats,
                    paper_author_index_stats: read_telemetry.paper_author_index_stats,
                    specter_index_stats: read_telemetry.specter_index_stats,
                    indexed_arrow_candidate_plan: true,
                    reused_name_counts_index,
                    reused_seed_feature_count: reused_feature_count,
                    reused_summary_count,
                },
            },
            planner_query_signature_ids: query_signature_ids.clone(),
            planner_query_signature_count: query_signature_ids.len(),
            planner_query_signature_id_set,
            planner_query_requests_by_signature_id: HashMap::default(),
            query_mode: RawPlannerQueryMode::Declared,
            top_k,
            orcid_enabled,
            num_threads,
            max_exemplars,
            released: false,
        })
    }

    /// Rebuild for new seed sidecars, reading only rows the retained ones lack.
    ///
    /// Returns false, without changes, unless this planner retains its rows and the
    /// new seeds include every current seed. Otherwise the retained rows plus the
    /// missing ones, trimmed to the new seeds, equal a fresh read, and the state is
    /// assembled from them by the same code as a fresh build.
    fn reload_seed_rows(
        &mut self,
        py: Python<'_>,
        cluster_seeds_path: &str,
        cluster_seed_disallows_path: Option<&str>,
    ) -> PyResult<bool> {
        // Declared-query planners carry request rows that a reload would not restore.
        if self.state.retained_seed_rows.is_none()
            || !matches!(self.query_mode, RawPlannerQueryMode::Auto)
        {
            return Ok(false);
        }
        let layout = RawArrowSeedLayout::read(cluster_seeds_path, cluster_seed_disallows_path)?;
        if !layout
            .seed_signature_id_set
            .is_superset(&self.state.seed_signature_id_set)
        {
            return Ok(false);
        }
        let retained = self
            .state
            .retained_seed_rows
            .take()
            .expect("checked above that seed rows are retained");
        let mut parts = ReloadParts {
            signatures: std::mem::take(&mut self.state.signatures),
            paper_authors: std::mem::take(&mut self.state.paper_authors),
            papers: retained.papers,
            specter_by_paper_id: retained.specter_by_paper_id,
            prefetched: self.state.prefetched_rows.take(),
        };
        match self.reload_seed_rows_from_parts(py, layout, &mut parts) {
            Ok(planner) => {
                *self = planner;
                Ok(true)
            }
            Err(error) => {
                // The rows go back, so a later reload or plan still has them.
                self.state.signatures = parts.signatures;
                self.state.paper_authors = parts.paper_authors;
                self.state.retained_seed_rows = Some(RetainedRawArrowSeedRows {
                    papers: parts.papers,
                    specter_by_paper_id: parts.specter_by_paper_id,
                });
                self.state.prefetched_rows = parts.prefetched;
                Err(error)
            }
        }
    }

    /// The reload's reads and assembly over rows moved out of the planner.
    ///
    /// Every read fails before planner state is replaced; only `assemble`,
    /// which runs last, consumes the previous state.
    fn reload_seed_rows_from_parts(
        &mut self,
        py: Python<'_>,
        layout: RawArrowSeedLayout,
        parts: &mut ReloadParts,
    ) -> PyResult<Self> {
        let signatures = &mut parts.signatures;
        let paper_authors = &mut parts.paper_authors;
        let papers = &mut parts.papers;
        let specter_by_paper_id = &mut parts.specter_by_paper_id;
        let prefetched = &mut parts.prefetched;

        let read_signatures_start = Instant::now();
        let mut missing_signature_ids: HashSet<String> = layout
            .seed_signature_id_set
            .iter()
            .filter(|signature_id| !signatures.contains_key(*signature_id))
            .cloned()
            .collect();
        if let Some(cache) = prefetched.as_mut() {
            missing_signature_ids.retain(|signature_id| {
                match cache.signatures.remove(signature_id) {
                    Some(signature) => {
                        signatures.insert(signature_id.clone(), signature);
                        false
                    }
                    None => true,
                }
            });
        }
        let signature_index_stats = if missing_signature_ids.is_empty() {
            IndexedArrowReadStats::default()
        } else {
            let dataset = &self.dataset;
            let num_threads = self.num_threads;
            let (loaded, stats) = py.allow_threads(|| {
                dataset.read_signatures(Some(&missing_signature_ids), num_threads)
            })?;
            signatures.extend(loaded);
            stats
        };
        let read_signatures_secs = read_signatures_start.elapsed().as_secs_f64();
        signatures.retain(|signature_id, _| layout.seed_signature_id_set.contains(signature_id));
        layout.require_signature_rows(&signatures, self.num_threads)?;

        let needed_paper_ids = layout.needed_paper_ids(&signatures);
        let mut missing_paper_ids: HashSet<String> = needed_paper_ids
            .iter()
            .filter(|paper_id| !papers.contains_key(*paper_id))
            .cloned()
            .collect();
        if let Some(cache) = prefetched.as_ref() {
            // Copy, not move: another cached signature may share the paper.
            missing_paper_ids.retain(|paper_id| {
                let Some(paper) = cache.papers.get(paper_id) else {
                    return true;
                };
                papers.insert(paper_id.clone(), paper.clone());
                if let Some(authors) = cache.paper_authors.get(paper_id) {
                    paper_authors.insert(paper_id.clone(), authors.clone());
                }
                if let (Some(cached), Some(specter)) = (
                    cache.specter_by_paper_id.as_ref(),
                    specter_by_paper_id.as_mut(),
                ) {
                    if let Some(vector) = cached.get(paper_id) {
                        specter.insert(paper_id.clone(), Arc::clone(vector));
                    }
                }
                false
            });
        }
        let raw_name_counts = self.dataset.name_counts();
        let metadata_reads_parallel_start = Instant::now();
        let rows = read_raw_arrow_paper_rows(
            py,
            &self.dataset,
            &raw_name_counts,
            &missing_paper_ids,
            self.num_threads,
        )?;
        papers.extend(rows.papers);
        paper_authors.extend(rows.paper_authors);
        match (specter_by_paper_id.as_mut(), rows.specter_by_paper_id) {
            (Some(existing), Some(loaded)) => existing.extend(loaded),
            (None, None) => {}
            _ => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "retained SPECTER rows disagree with the dataset's SPECTER table",
                ))
            }
        }
        papers.retain(|paper_id, _| needed_paper_ids.contains(paper_id));
        paper_authors.retain(|paper_id, _| needed_paper_ids.contains(paper_id));
        if let Some(specter) = specter_by_paper_id.as_mut() {
            specter.retain(|paper_id, _| needed_paper_ids.contains(paper_id));
        }
        validate_required_raw_arrow_paper_metadata(&needed_paper_ids, &papers, self.num_threads)?;
        let read_telemetry = RawArrowSeedReadTelemetry {
            read_cluster_seeds_secs: layout.read_cluster_seeds_secs,
            read_signatures_secs,
            read_papers_secs: rows.read_papers_secs,
            read_paper_authors_secs: rows.read_paper_authors_secs,
            read_specter_secs: rows.read_specter_secs,
            read_name_counts_secs: rows.read_name_counts_secs,
            signature_index_stats,
            paper_index_stats: rows.paper_index_stats,
            paper_author_index_stats: rows.paper_author_index_stats,
            specter_index_stats: rows.specter_index_stats,
        };
        let previous = ReusableSeedState {
            features_by_signature_id: std::mem::take(&mut self.state.features_by_signature_id),
            members_by_component: std::mem::take(&mut self.state.members_by_component),
            component_keys_by_member: std::mem::take(&mut self.state.component_keys_by_member),
            summaries: std::mem::take(&mut self.state.retriever.summaries),
            component_index_by_key: std::mem::take(
                &mut self.state.retriever.component_index_by_key,
            ),
            unidecode_char_map: std::mem::take(&mut self.state.unidecode_char_map),
        };
        let query_signature_ids = self.planner_query_signature_ids.clone();
        let query_mode = self.query_mode;
        let mut planner = Self::assemble(
            py,
            self.dataset.clone(),
            layout,
            std::mem::take(signatures),
            std::mem::take(papers),
            std::mem::take(paper_authors),
            specter_by_paper_id.take(),
            raw_name_counts,
            needed_paper_ids.len(),
            read_telemetry,
            metadata_reads_parallel_start,
            query_signature_ids,
            self.top_k,
            self.orcid_enabled,
            self.num_threads,
            self.max_exemplars,
            true,
            Some(previous),
        )?;
        planner.query_mode = query_mode;
        planner.state.prefetched_rows = prefetched.take();
        Ok(planner)
    }
}

#[pymethods]
impl RawBlockQueryCandidatePlanner {
    #[staticmethod]
    #[pyo3(signature = (
        dataset,
        query_signatures_path,
        cluster_seeds_path,
        top_k,
        *,
        cluster_seed_disallows_path = None,
        orcid_enabled = true,
        num_threads = None,
        max_exemplars = 4
    ))]
    fn from_query_signatures(
        py: Python<'_>,
        dataset: PyRef<'_, ArrowDataset>,
        query_signatures_path: String,
        cluster_seeds_path: String,
        top_k: usize,
        cluster_seed_disallows_path: Option<String>,
        orcid_enabled: bool,
        num_threads: Option<usize>,
        max_exemplars: usize,
    ) -> PyResult<Self> {
        let query_requests = read_raw_arrow_query_signatures(&query_signatures_path)?;
        let query_signature_ids = query_requests
            .iter()
            .map(|request| request.signature_id.clone())
            .collect::<Vec<_>>();
        validate_raw_arrow_query_signature_ids_allow_empty(&query_signature_ids)?;
        let mut planner = Self::build_from_query_signature_ids(
            py,
            dataset.shared(),
            &cluster_seeds_path,
            cluster_seed_disallows_path.as_deref(),
            query_signature_ids.clone(),
            top_k,
            orcid_enabled,
            num_threads,
            max_exemplars,
            false,
            Vec::new(),
        )?;
        planner.planner_query_signature_ids = query_signature_ids;
        planner.planner_query_signature_count = query_requests.len();
        planner.planner_query_requests_by_signature_id = query_requests
            .into_iter()
            .map(|request| (request.signature_id.clone(), request))
            .collect();
        Ok(planner)
    }

    #[staticmethod]
    #[pyo3(signature = (
        dataset,
        cluster_seeds_path,
        top_k,
        *,
        cluster_seed_disallows_path = None,
        orcid_enabled = true,
        num_threads = None,
        max_exemplars = 4,
        retain_seed_rows = false,
        prefetch_signature_ids = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_auto_queries(
        py: Python<'_>,
        dataset: PyRef<'_, ArrowDataset>,
        cluster_seeds_path: String,
        top_k: usize,
        cluster_seed_disallows_path: Option<String>,
        orcid_enabled: bool,
        num_threads: Option<usize>,
        max_exemplars: usize,
        retain_seed_rows: bool,
        prefetch_signature_ids: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut planner = Self::build_from_query_signature_ids(
            py,
            dataset.shared(),
            &cluster_seeds_path,
            cluster_seed_disallows_path.as_deref(),
            Vec::new(),
            top_k,
            orcid_enabled,
            num_threads,
            max_exemplars,
            retain_seed_rows,
            prefetch_signature_ids.unwrap_or_default(),
        )?;
        planner.query_mode = RawPlannerQueryMode::Auto;
        Ok(planner)
    }

    /// Rebuild in place for new seed sidecars when the new seeds include every
    /// current seed; returns false and leaves the planner unchanged otherwise.
    ///
    /// Subblocked attachment passes call this between passes. Only rows of the new
    /// seeds are read, and the rebuilt state equals a fresh `from_auto_queries`
    /// build over the same sidecars.
    #[pyo3(signature = (cluster_seeds_path, cluster_seed_disallows_path = None))]
    fn reload_seed_sidecars(
        &mut self,
        py: Python<'_>,
        cluster_seeds_path: String,
        cluster_seed_disallows_path: Option<String>,
    ) -> PyResult<bool> {
        self.ensure_not_released()?;
        self.reload_seed_rows(
            py,
            &cluster_seeds_path,
            cluster_seed_disallows_path.as_deref(),
        )
    }

    /// Drop the planner's rows on a background thread.
    ///
    /// Freeing a block's rows takes seconds on one thread, so a request
    /// returns before that. The planner then refuses to plan or reload.
    fn release(&mut self) {
        self.released = true;
        let state = &mut self.state;
        let dropped = (
            std::mem::take(&mut state.features_by_signature_id),
            std::mem::take(&mut state.signatures),
            std::mem::take(&mut state.paper_authors),
            std::mem::take(&mut state.seed_author_index),
            std::mem::take(&mut state.raw_name_counts),
            std::mem::take(&mut state.members_by_component),
            std::mem::take(&mut state.component_keys_by_member),
            std::mem::take(&mut state.summary_signals_by_component),
            state.retriever.take_contents(),
            std::mem::take(&mut state.component_order),
            std::mem::take(&mut state.seed_signature_ids),
            std::mem::take(&mut state.seed_signature_id_set),
            std::mem::take(&mut state.cluster_seed_disallows),
            std::mem::take(&mut state.unidecode_char_map),
            state.retained_seed_rows.take(),
            state.prefetched_rows.take(),
            state.last_query_inputs.take(),
            std::mem::take(&mut self.planner_query_signature_ids),
            std::mem::take(&mut self.planner_query_signature_id_set),
            std::mem::take(&mut self.planner_query_requests_by_signature_id),
        );
        std::thread::spawn(move || drop(dropped));
    }

    fn build_telemetry(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let timings = PyDict::new(py);
        let telemetry = &self.state.build_telemetry;
        timings.set_item("read_cluster_seeds_secs", telemetry.read_cluster_seeds_secs)?;
        timings.set_item("read_signatures_secs", telemetry.read_signatures_secs)?;
        timings.set_item("read_papers_secs", telemetry.read_papers_secs)?;
        timings.set_item("read_paper_authors_secs", telemetry.read_paper_authors_secs)?;
        timings.set_item("read_specter_secs", telemetry.read_specter_secs)?;
        timings.set_item("read_name_counts_secs", telemetry.read_name_counts_secs)?;
        timings.set_item(
            "metadata_reads_parallel_secs",
            telemetry.metadata_reads_parallel_secs,
        )?;
        timings.set_item("text_context_secs", telemetry.text_context_secs)?;
        timings.set_item("feature_secs", telemetry.feature_secs)?;
        timings.set_item("summary_secs", telemetry.summary_secs)?;
        timings.set_item("component_members_secs", telemetry.component_members_secs)?;

        let payload = PyDict::new(py);
        payload.set_item("signature_count", self.state.signatures.len())?;
        payload.set_item("paper_count", self.state.seed_paper_count)?;
        payload.set_item("paper_author_paper_count", self.state.paper_authors.len())?;
        payload.set_item("cluster_count", self.state.component_order.len())?;
        payload.set_item("seed_signature_count", self.state.seed_signature_ids.len())?;
        payload.set_item("query_signature_count", self.planner_query_signature_count)?;
        payload.set_item(
            "cluster_seed_disallow_pair_count",
            self.state.cluster_seed_disallows.len(),
        )?;
        payload.set_item("specter_count", self.state.seed_specter_count)?;
        payload.set_item(
            "indexed_arrow_candidate_plan",
            telemetry.indexed_arrow_candidate_plan,
        )?;
        payload.set_item(
            "reused_name_counts_index",
            telemetry.reused_name_counts_index,
        )?;
        payload.set_item(
            "signature_batches_read",
            telemetry.signature_index_stats.batches_read,
        )?;
        payload.set_item(
            "signature_rows_scanned",
            telemetry.signature_index_stats.rows_scanned,
        )?;
        payload.set_item(
            "paper_batches_read",
            telemetry.paper_index_stats.batches_read,
        )?;
        payload.set_item(
            "paper_rows_scanned",
            telemetry.paper_index_stats.rows_scanned,
        )?;
        payload.set_item(
            "paper_author_batches_read",
            telemetry.paper_author_index_stats.batches_read,
        )?;
        payload.set_item(
            "paper_author_rows_scanned",
            telemetry.paper_author_index_stats.rows_scanned,
        )?;
        payload.set_item(
            "specter_batches_read",
            telemetry.specter_index_stats.batches_read,
        )?;
        payload.set_item(
            "specter_rows_scanned",
            telemetry.specter_index_stats.rows_scanned,
        )?;
        payload.set_item("unidecode_char_count", self.state.unidecode_char_map.len())?;
        payload.set_item(
            "reused_seed_feature_count",
            telemetry.reused_seed_feature_count,
        )?;
        payload.set_item("reused_summary_count", telemetry.reused_summary_count)?;
        payload.set_item("planner_seed_state", 1)?;
        payload.set_item("timings", timings)?;
        Ok(payload.unbind())
    }

    /// A featurizer over `signature_ids` built from the rows this planner holds,
    /// equal to `RustFeaturizer.from_arrow_dataset` over the same arguments.
    ///
    /// Returns None when some row is not held (for example a planner built
    /// without retained seed rows); callers then read from Arrow.
    #[pyo3(signature = (
        signature_ids,
        name_tuples,
        *,
        cluster_seeds_path = None,
        cluster_seed_disallows_path = None,
        preprocess = true,
        cluster_seed_require_value = 0.0,
        cluster_seed_disallow_value = 10000.0,
        num_threads = None,
        use_orcid_id = true
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build_featurizer(
        &self,
        py: Python<'_>,
        signature_ids: Vec<String>,
        name_tuples: Option<&Bound<'_, PyAny>>,
        cluster_seeds_path: Option<String>,
        cluster_seed_disallows_path: Option<String>,
        preprocess: bool,
        cluster_seed_require_value: f64,
        cluster_seed_disallow_value: f64,
        num_threads: Option<usize>,
        use_orcid_id: bool,
    ) -> PyResult<Option<RustFeaturizer>> {
        self.ensure_not_released()?;
        // Same de-duplication as `from_arrow_dataset`: first occurrence wins.
        let mut seen = HashSet::with_capacity_and_hasher(signature_ids.len(), Default::default());
        let signature_ids: Vec<String> = signature_ids
            .into_iter()
            .filter(|signature_id| seen.insert(signature_id.clone()))
            .collect();
        let Some(rows) = py.allow_threads(|| {
            install_with_optional_rayon_pool(num_threads, || self.featurizer_rows(&signature_ids))
        }) else {
            return Ok(None);
        };
        let needed_paper_ids: HashSet<String> = rows
            .signatures
            .values()
            .map(|signature| signature.paper_id.clone())
            .collect();
        let (cluster_seeds_require, cluster_seeds_disallow) =
            RustFeaturizer::read_seed_constraints(
                cluster_seeds_path.as_deref(),
                cluster_seed_disallows_path.as_deref(),
                &seen,
            )?;
        RustFeaturizer::from_raw_rows(
            py,
            rows,
            signature_ids,
            needed_paper_ids,
            cluster_seeds_require,
            cluster_seeds_disallow,
            self.state.raw_name_counts.clone(),
            name_tuples,
            preprocess,
            cluster_seed_require_value,
            cluster_seed_disallow_value,
            num_threads,
            use_orcid_id,
        )
        .map(Some)
    }

    fn plan_query_signatures(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.ensure_not_released()?;
        if matches!(self.query_mode, RawPlannerQueryMode::Auto) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "RawBlockQueryCandidatePlanner.from_auto_queries requires explicit plan(query_signature_ids)",
            ));
        }
        self.plan(py, self.planner_query_signature_ids.clone(), None)
    }

    fn name_counts_index(&self) -> Option<NameCountsIndex> {
        self.state
            .raw_name_counts
            .shared_index()
            .map(NameCountsIndex::from_shared)
    }

    #[pyo3(signature = (query_signature_ids, additional_cluster_seed_disallows = None))]
    fn plan(
        &mut self,
        py: Python<'_>,
        query_signature_ids: Vec<String>,
        additional_cluster_seed_disallows: Option<Vec<(String, String)>>,
    ) -> PyResult<Py<PyDict>> {
        self.ensure_not_released()?;
        validate_raw_arrow_query_signature_ids(&query_signature_ids)?;
        let additional_cluster_seed_disallows = additional_cluster_seed_disallows
            .map(validate_additional_cluster_seed_disallows)
            .transpose()?
            .unwrap_or_default();
        if matches!(self.query_mode, RawPlannerQueryMode::Declared) {
            let missing: Vec<&String> = query_signature_ids
                .iter()
                .filter(|signature_id| !self.planner_query_signature_id_set.contains(*signature_id))
                .collect();
            if !missing.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "RawBlockQueryCandidatePlanner plan requested query_signature_ids outside the planner query set: {:?}",
                    missing.into_iter().take(10).collect::<Vec<_>>()
                )));
            }
        }

        let total_start = Instant::now();
        let query_inputs = match self
            .state
            .prefetched_rows
            .as_ref()
            .and_then(|cache| cache.query_inputs(&query_signature_ids))
        {
            Some(query_inputs) => query_inputs,
            None => read_reusable_raw_arrow_query_inputs(
                py,
                &self.dataset,
                &query_signature_ids,
                self.num_threads,
            )?,
        };

        let text_context_start = Instant::now();
        ensure_unidecode_for_raw_arrow_inputs(
            &query_inputs.signatures,
            &query_inputs.papers,
            &query_inputs.paper_authors,
            &mut self.state.unidecode_char_map,
            self.num_threads,
        )?;
        let text_context_secs = text_context_start.elapsed().as_secs_f64();

        let feature_start = Instant::now();
        let raw_feature_results: Result<Vec<(String, RawArrowFeature)>, String> =
            py.allow_threads(|| {
                let compute = || {
                    let coauthor_blocks = paper_coauthor_blocks(
                        query_inputs.signatures.values(),
                        &query_inputs.paper_authors,
                        &self.state.unidecode_char_map,
                    )?;
                    query_signature_ids
                        .par_iter()
                        .map(|signature_id| {
                            let signature =
                                query_inputs.signatures.get(signature_id).ok_or_else(|| {
                                    format!(
                                        "signature_id '{}' is missing from signatures",
                                        signature_id
                                    )
                                })?;
                            let paper = query_inputs.papers.get(&signature.paper_id);
                            Ok((
                                signature_id.clone(),
                                build_raw_arrow_feature(
                                    signature,
                                    paper,
                                    coauthor_blocks.get(&signature.paper_id),
                                    query_inputs.specter_by_paper_id.as_ref(),
                                    &self.state.raw_name_counts,
                                    &self.state.name_prefixes,
                                    &self.state.affiliation_stopwords,
                                    &self.state.unidecode_char_map,
                                    self.orcid_enabled,
                                ),
                            ))
                        })
                        .collect()
                };
                install_with_optional_rayon_pool(self.num_threads, compute)
            });
        let query_features_by_signature_id: HashMap<String, RawArrowFeature> = raw_feature_results
            .map_err(pyo3::exceptions::PyKeyError::new_err)?
            .into_iter()
            .collect();
        let feature_secs = feature_start.elapsed().as_secs_f64();

        let query_start = Instant::now();
        let mut queries = Vec::<RetrievalQueryData>::with_capacity(query_signature_ids.len());
        let mut query_views = Vec::<String>::with_capacity(query_signature_ids.len());
        let mut query_first_tokens = Vec::<String>::with_capacity(query_signature_ids.len());
        let mut query_authors = Vec::<String>::with_capacity(query_signature_ids.len());
        for signature_id in query_signature_ids.iter() {
            let base_feature = &query_features_by_signature_id[signature_id];
            let base = &base_feature.query;
            let request = self
                .planner_query_requests_by_signature_id
                .get(signature_id);
            if let Some(request) = request {
                if !request.query_author.is_empty()
                    && request.query_author != base_feature.query_author
                {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "query_signatures Arrow query_author for signature_id {:?} does not match \
                         signatures-derived query_author: {:?} != {:?}",
                        signature_id, request.query_author, base_feature.query_author
                    )));
                }
            } else if matches!(self.query_mode, RawPlannerQueryMode::Declared) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "RawBlockQueryCandidatePlanner is missing query_signatures request row for \
                     signature_id {signature_id:?}"
                )));
            }
            let query_view = request.map_or("auto", |request| request.query_view.as_str());
            let (masked, resolved_view) = mask_raw_arrow_query(base, query_view)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            queries.push(masked);
            query_views.push(resolved_view);
            query_first_tokens.push(base.first.clone());
            query_authors.push(base_feature.query_author.clone());
        }
        let query_secs = query_start.elapsed().as_secs_f64();

        let query_signature_id_set: HashSet<&str> =
            query_signature_ids.iter().map(String::as_str).collect();
        let overlapping_query_seed_ids = query_signature_ids
            .iter()
            .filter(|signature_id| self.state.seed_signature_id_set.contains(*signature_id))
            .take(10)
            .cloned()
            .collect::<Vec<_>>();
        if query_signature_ids.len() > 1 && !overlapping_query_seed_ids.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "RawBlockQueryCandidatePlanner.plan requires singleton query windows when query ids are also \
                 cluster seed members; overlapping query_signature_ids={overlapping_query_seed_ids:?}"
            )));
        }
        let mut excluded_query_seed_count = 0usize;
        let mut filtered_component_order = Vec::<String>::new();
        let mut filtered_members_by_component = HashMap::<String, Vec<String>>::default();
        let needs_filtered_retriever = !overlapping_query_seed_ids.is_empty();
        let summary_start = Instant::now();
        let filtered_retriever = if needs_filtered_retriever {
            for component_key in self.state.component_order.iter() {
                let members = self
                    .state
                    .members_by_component
                    .get(component_key)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "component_key '{}' disappeared while filtering query seeds",
                            component_key
                        ))
                    })?;
                let mut filtered_members = Vec::with_capacity(members.len());
                for signature_id in members {
                    if query_signature_id_set.contains(signature_id.as_str()) {
                        excluded_query_seed_count += 1;
                    } else {
                        filtered_members.push(signature_id.clone());
                    }
                }
                if !filtered_members.is_empty() {
                    filtered_component_order.push(component_key.clone());
                    filtered_members_by_component.insert(component_key.clone(), filtered_members);
                }
            }
            Some(build_retriever_from_raw_arrow_components(
                py,
                &filtered_component_order,
                &filtered_members_by_component,
                &self.state.features_by_signature_id,
                self.max_exemplars,
                self.num_threads,
                true,
            )?)
        } else {
            None
        };
        let summary_secs = summary_start.elapsed().as_secs_f64();
        let (component_order, members_by_component, retriever) =
            if let Some(retriever) = filtered_retriever.as_ref() {
                (
                    &filtered_component_order,
                    &filtered_members_by_component,
                    retriever,
                )
            } else {
                (
                    &self.state.component_order,
                    &self.state.members_by_component,
                    &self.state.retriever,
                )
            };

        let component_members_start = Instant::now();
        let (component_member_indices, seed_signature_ids, _seed_component_keys) =
            raw_arrow_component_member_indices_for_batch(
                component_order,
                members_by_component,
                query_signature_ids.len(),
            )?;
        let component_members_secs = component_members_start.elapsed().as_secs_f64();

        let (
            excluded_candidate_indices_by_query,
            cluster_seed_disallowed_candidate_count,
            cluster_seed_disallow_pair_count,
        ) = raw_arrow_excluded_candidate_indices_by_query(
            &query_signature_ids,
            component_order,
            &self.state.component_keys_by_member,
            &self.state.cluster_seed_disallows,
            if additional_cluster_seed_disallows.is_empty() {
                None
            } else {
                Some(&additional_cluster_seed_disallows)
            },
        )?;

        let retrieval_start = Instant::now();
        let query_results: Vec<Result<RetrievalPairPlanQueryResult, RetrievalError>> = py
            .allow_threads(|| {
                let compute = || {
                    // Each block's SPECTER scores read every summary once; see
                    // `specter_score_rows`.
                    queries
                        .par_chunks(RETRIEVAL_QUERY_BLOCK)
                        .enumerate()
                        .map(|(block_index, block)| {
                            let specter_rows = retriever.specter_score_rows(block);
                            block
                                .iter()
                                .zip(&specter_rows)
                                .enumerate()
                                .map(|(block_offset, (current_query, specter_row))| {
                                    let query_offset =
                                        block_index * RETRIEVAL_QUERY_BLOCK + block_offset;
                                    let query_index =
                                        u32::try_from(query_offset).map_err(|_| {
                                            RetrievalError::Overflow(
                                                "query index exceeds u32".to_string(),
                                            )
                                        })?;
                                    let excluded_candidate_indices =
                                        excluded_candidate_indices_by_query
                                            .as_ref()
                                            .and_then(|values| values[query_offset].as_ref());
                                    retriever.build_pair_plan_query_result(
                                        current_query,
                                        query_first_tokens[query_offset].as_str(),
                                        query_index,
                                        excluded_candidate_indices,
                                        &component_member_indices,
                                        self.top_k,
                                        Some(specter_row.as_slice()),
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                };
                install_with_optional_rayon_pool(self.num_threads, compute)
            });

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
        let mut row_last_name_count_min_rarity = Vec::<f32>::new();
        let mut row_candidate_last_name_count_min_rarity = Vec::<f32>::new();
        let mut row_candidate_last_first_name_count_min_rarity = Vec::<f32>::new();
        let mut row_last_first_name_count_min_rarity = Vec::<f32>::new();
        let mut row_first_prefix_x_last_first_name_count_min_rarity = Vec::<f32>::new();
        let mut row_candidate_cluster_max_paper_author_count = Vec::<f32>::new();
        let mut row_paper_author_list_max_jaccard = Vec::<f32>::new();
        let mut row_paper_author_list_max_containment = Vec::<f32>::new();
        let mut row_paper_author_list_max_overlap_count = Vec::<f32>::new();
        let mut row_local_author_window10_jaccard_max = Vec::<f32>::new();
        let mut row_local_author_window10_overlap_count_max = Vec::<f32>::new();
        let mut row_best_author_count_log_absdiff = Vec::<f32>::new();
        let mut left_signature_indices = Vec::<u32>::new();
        let mut right_signature_indices = Vec::<u32>::new();
        let mut pair_row_indices = Vec::<u32>::new();
        let mut filtered_summary_signals_by_component =
            HashMap::<String, RawArrowSummarySignalData>::default();
        // Each retrieved component expands once, in first-appearance order: a
        // large cluster retrieved by thousands of queries expanded that many
        // times, and the index skips repeated papers anyway.
        let mut seen_component_keys = HashSet::<&str>::default();
        let retrieved_member_paper_ids: Vec<&str> = query_results
            .iter()
            .flatten()
            .flat_map(|result| result.row_component_keys.iter())
            .filter(|component_key| seen_component_keys.insert(component_key.as_str()))
            .filter_map(|component_key| members_by_component.get(component_key))
            .flatten()
            .filter_map(|signature_id| self.state.signatures.get(signature_id))
            .map(|signature| signature.paper_id.as_str())
            .collect();
        // Only rows with candidates need author-list evidence. Keep their
        // query-only papers local to this plan, including their interned names.
        let query_paper_ids_with_rows: Vec<&str> = query_results
            .iter()
            .flatten()
            .flat_map(|result| result.row_query_signature_indices.iter())
            .filter_map(|index| query_signature_ids.get(*index as usize))
            .filter_map(|signature_id| query_inputs.signatures.get(signature_id))
            .map(|signature| signature.paper_id.as_str())
            .collect();
        py.allow_threads(|| {
            install_with_optional_rayon_pool(self.num_threads, || {
                self.state.seed_author_index.extend(
                    retrieved_member_paper_ids,
                    &self.state.paper_authors,
                    &self.state.unidecode_char_map,
                )
            })
        })
        .map_err(pyo3::exceptions::PyKeyError::new_err)?;
        // Finish seed interning before borrowing it for temporary query ids.
        let query_author_tables = py
            .allow_threads(|| {
                install_with_optional_rayon_pool(self.num_threads, || {
                    self.state.seed_author_index.query_tables(
                        query_paper_ids_with_rows,
                        &query_inputs.paper_authors,
                        &self.state.unidecode_char_map,
                    )
                })
            })
            .map_err(pyo3::exceptions::PyKeyError::new_err)?;
        let query_results = query_results
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(retrieval_error_to_py)?;
        for query_result in query_results.iter() {
            for &query_offset in query_result.row_query_signature_indices.iter() {
                if query_offset as usize >= query_signature_ids.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "row query signature offset {} is outside query_signature_ids",
                        query_offset
                    )));
                }
            }
        }
        // Row signals are pure functions of read-only plan state, and big components
        // make them the costliest part of a plan. Compute them in parallel, then fill
        // the columns below in the original row order.
        let num_threads = self.num_threads;
        let summary_signal_cache = if needs_filtered_retriever {
            &mut filtered_summary_signals_by_component
        } else {
            &mut self.state.summary_signals_by_component
        };
        let mut pending_component_keys = HashSet::<&str>::default();
        let missing_component_keys: Vec<&str> = query_results
            .iter()
            .flat_map(|query_result| query_result.row_component_keys.iter())
            .map(String::as_str)
            .filter(|component_key| {
                !summary_signal_cache.contains_key(*component_key)
                    && pending_component_keys.insert(*component_key)
            })
            .collect();
        let features_by_signature_id = &self.state.features_by_signature_id;
        let seed_signatures = &self.state.signatures;
        let seed_author_index = &self.state.seed_author_index;
        let built_summary_signals = py
            .allow_threads(|| {
                install_with_optional_rayon_pool(num_threads, || {
                    missing_component_keys
                        .par_iter()
                        .map(|component_key| {
                            let members =
                                members_by_component.get(*component_key).ok_or_else(|| {
                                    format!(
                                        "component_key '{}' disappeared while building row signals",
                                        component_key
                                    )
                                })?;
                            build_raw_arrow_summary_signals(
                                members,
                                features_by_signature_id,
                                seed_signatures,
                                seed_author_index,
                            )
                            .map(|signals| (component_key.to_string(), signals))
                        })
                        .collect::<Result<Vec<_>, String>>()
                })
            })
            .map_err(pyo3::exceptions::PyKeyError::new_err)?;
        summary_signal_cache.extend(built_summary_signals);
        let summary_signal_cache = &*summary_signal_cache;

        let mut pending_query_offsets = HashSet::<u32>::default();
        let query_offsets_with_rows: Vec<u32> = query_results
            .iter()
            .flat_map(|query_result| query_result.row_query_signature_indices.iter().copied())
            .filter(|query_offset| pending_query_offsets.insert(*query_offset))
            .collect();
        let query_signatures = &query_inputs.signatures;
        let author_signals_by_query_offset: HashMap<u32, RawArrowAuthorSignalData> = py
            .allow_threads(|| {
                install_with_optional_rayon_pool(num_threads, || {
                    query_offsets_with_rows
                        .par_iter()
                        .map(|&query_offset| {
                            let query_signature_id = &query_signature_ids[query_offset as usize];
                            let query_signature =
                                query_signatures.get(query_signature_id).ok_or_else(|| {
                                    format!(
                                        "query signature_id '{}' is missing from signatures",
                                        query_signature_id
                                    )
                                })?;
                            query_author_tables
                                .author_signals(query_signature)
                                .map(|signals| (query_offset, signals))
                        })
                        .collect::<Result<HashMap<_, _>, String>>()
                })
            })
            .map_err(pyo3::exceptions::PyKeyError::new_err)?;

        let row_refs: Vec<(usize, usize)> = query_results
            .iter()
            .enumerate()
            .flat_map(|(result_index, query_result)| {
                (0..query_result.row_component_keys.len())
                    .map(move |local_row_index| (result_index, local_row_index))
            })
            .collect();
        let row_signals = py.allow_threads(|| {
            install_with_optional_rayon_pool(num_threads, || {
                row_refs
                    .par_iter()
                    .map(|&(result_index, local_row_index)| {
                        let query_result = &query_results[result_index];
                        let component_key = &query_result.row_component_keys[local_row_index];
                        let query_offset =
                            query_result.row_query_signature_indices[local_row_index];
                        let query_signature_id = &query_signature_ids[query_offset as usize];
                        let query_feature = query_features_by_signature_id
                            .get(query_signature_id)
                            .ok_or_else(|| {
                                pyo3::exceptions::PyKeyError::new_err(format!(
                                    "query signature_id '{}' is missing from computed raw Arrow features",
                                    query_signature_id
                                ))
                            })?;
                        // queries has one entry per query_signature_ids entry.
                        let query = &queries[query_offset as usize];
                        let component_index = retriever
                            .component_index_by_key
                            .get(component_key)
                            .ok_or_else(|| {
                                pyo3::exceptions::PyKeyError::new_err(format!(
                                    "component_key '{}' disappeared while building row signals",
                                    component_key
                                ))
                            })?;
                        let summary = retriever.summaries.get(*component_index).ok_or_else(|| {
                            pyo3::exceptions::PyIndexError::new_err(format!(
                                "component index {} is outside summary table",
                                component_index
                            ))
                        })?;
                        let summary_signals =
                            summary_signal_cache.get(component_key).ok_or_else(|| {
                                pyo3::exceptions::PyKeyError::new_err(format!(
                                    "component_key '{}' is missing raw row signal summary",
                                    component_key
                                ))
                            })?;
                        let rarity = raw_arrow_name_count_rarity_row(
                            query,
                            &query_feature.name_counts,
                            summary,
                            summary_signals,
                        );
                        let evidence = raw_arrow_paper_evidence_row(
                            query_signature_id,
                            query_feature.paper_author_count,
                            &author_signals_by_query_offset[&query_offset],
                            summary_signals,
                        );
                        Ok((summary.max_paper_author_count as f32, rarity, evidence))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
        })?;
        let mut row_signals = row_signals.into_iter();

        for mut query_result in query_results {
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
            for _ in 0..query_result.row_component_keys.len() {
                let (candidate_cluster_max_paper_author_count, rarity, evidence) =
                    row_signals.next().ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "raw Arrow row signals ended before the retrieved rows",
                        )
                    })?;
                row_last_name_count_min_rarity.push(rarity.last_name_count_min_rarity);
                row_candidate_last_name_count_min_rarity
                    .push(rarity.candidate_last_name_count_min_rarity);
                row_candidate_last_first_name_count_min_rarity
                    .push(rarity.candidate_last_first_name_count_min_rarity);
                row_last_first_name_count_min_rarity.push(rarity.last_first_name_count_min_rarity);
                row_first_prefix_x_last_first_name_count_min_rarity
                    .push(rarity.first_prefix_x_last_first_name_count_min_rarity);
                row_candidate_cluster_max_paper_author_count
                    .push(candidate_cluster_max_paper_author_count);
                row_paper_author_list_max_jaccard.push(evidence.paper_author_list_max_jaccard);
                row_paper_author_list_max_containment
                    .push(evidence.paper_author_list_max_containment);
                row_paper_author_list_max_overlap_count
                    .push(evidence.paper_author_list_max_overlap_count);
                row_local_author_window10_jaccard_max
                    .push(evidence.local_author_window10_jaccard_max);
                row_local_author_window10_overlap_count_max
                    .push(evidence.local_author_window10_overlap_count_max);
                row_best_author_count_log_absdiff.push(evidence.best_author_count_log_absdiff);
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
        let retrieval_secs = retrieval_start.elapsed().as_secs_f64();

        let pair_signature_ids_start = Instant::now();
        let signature_index_count = query_signature_ids.len() + seed_signature_ids.len();
        for index in left_signature_indices.iter() {
            if (*index as usize) >= signature_index_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "left signature index {} is outside signature id table of length {}",
                    index, signature_index_count
                )));
            }
        }
        for index in right_signature_indices.iter() {
            if (*index as usize) >= signature_index_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "right signature index {} is outside signature id table of length {}",
                    index, signature_index_count
                )));
            }
        }
        // Python orders a plan's signatures as its queries, then right signatures
        // by first appearance. Send that order once, with each pair's positions
        // in it: two id strings per pair cost Python several passes over
        // millions of pairs. Ids are deduplicated by value, as Python's order is.
        let query_count = query_signature_ids.len();
        let table_id = |index: u32| -> &str {
            let index = index as usize;
            if index < query_count {
                &query_signature_ids[index]
            } else {
                &seed_signature_ids[index - query_count]
            }
        };
        let mut pair_signature_ids: Vec<&str> =
            query_signature_ids.iter().map(String::as_str).collect();
        let mut position_by_id: HashMap<&str, u32> = pair_signature_ids
            .iter()
            .enumerate()
            .map(|(position, signature_id)| (*signature_id, position as u32))
            .collect();
        let mut position_by_table_index = vec![u32::MAX; signature_index_count];
        let mut position_of = |index: u32| -> u32 {
            let slot = &mut position_by_table_index[index as usize];
            if *slot == u32::MAX {
                let signature_id = table_id(index);
                *slot = *position_by_id.entry(signature_id).or_insert_with(|| {
                    pair_signature_ids.push(signature_id);
                    (pair_signature_ids.len() - 1) as u32
                });
            }
            *slot
        };
        // Right signatures first, in pair order, so new ids follow first appearance;
        // left signatures are queries and add no ids.
        let right_pair_signature_indices: Vec<u32> = right_signature_indices
            .iter()
            .map(|index| position_of(*index))
            .collect();
        let left_pair_signature_indices: Vec<u32> = left_signature_indices
            .iter()
            .map(|index| position_of(*index))
            .collect();
        let pair_signature_ids = PyList::new(py, pair_signature_ids)?;
        let pair_signature_ids_secs = pair_signature_ids_start.elapsed().as_secs_f64();

        let component_members_payload_start = Instant::now();
        let component_members = PyDict::new(py);
        for component_key in component_order.iter() {
            component_members.set_item(
                component_key,
                members_by_component
                    .get(component_key)
                    .cloned()
                    .unwrap_or_default(),
            )?;
        }
        let component_members_payload_secs =
            component_members_payload_start.elapsed().as_secs_f64();

        let timings = PyDict::new(py);
        timings.set_item("read_cluster_seeds_secs", 0.0)?;
        timings.set_item("read_signatures_secs", query_inputs.read_signatures_secs)?;
        timings.set_item("read_papers_secs", query_inputs.read_papers_secs)?;
        timings.set_item(
            "read_paper_authors_secs",
            query_inputs.read_paper_authors_secs,
        )?;
        timings.set_item("read_specter_secs", query_inputs.read_specter_secs)?;
        timings.set_item("read_name_counts_secs", 0.0)?;
        timings.set_item(
            "metadata_reads_parallel_secs",
            query_inputs.metadata_reads_parallel_secs,
        )?;
        timings.set_item("text_context_secs", text_context_secs)?;
        timings.set_item("feature_secs", feature_secs)?;
        timings.set_item("query_secs", query_secs)?;
        timings.set_item("summary_secs", summary_secs)?;
        timings.set_item("component_members_secs", component_members_secs)?;
        timings.set_item("retrieval_secs", retrieval_secs)?;
        timings.set_item("pair_signature_ids_secs", pair_signature_ids_secs)?;
        timings.set_item(
            "component_members_payload_secs",
            component_members_payload_secs,
        )?;

        let telemetry = PyDict::new(py);
        telemetry.set_item(
            "signature_count",
            query_signature_ids.len() + seed_signature_ids.len(),
        )?;
        let mut telemetry_paper_ids = HashSet::<&str>::default();
        for signature_id in query_signature_ids.iter() {
            if let Some(signature) = query_inputs.signatures.get(signature_id) {
                telemetry_paper_ids.insert(signature.paper_id.as_str());
            }
        }
        for signature_id in seed_signature_ids.iter() {
            if let Some(signature) = self.state.signatures.get(signature_id) {
                telemetry_paper_ids.insert(signature.paper_id.as_str());
            }
        }
        telemetry.set_item("paper_count", telemetry_paper_ids.len())?;
        let mut telemetry_paper_author_count = 0usize;
        for &paper_id in telemetry_paper_ids.iter() {
            if self.state.paper_authors.contains_key(paper_id)
                || query_inputs.paper_authors.contains_key(paper_id)
            {
                telemetry_paper_author_count += 1;
            }
        }
        telemetry.set_item("paper_author_paper_count", telemetry_paper_author_count)?;
        telemetry.set_item("cluster_count", component_order.len())?;
        telemetry.set_item("seed_signature_count", seed_signature_ids.len())?;
        telemetry.set_item("query_signature_count", query_signature_ids.len())?;
        telemetry.set_item("excluded_query_seed_count", excluded_query_seed_count)?;
        telemetry.set_item(
            "cluster_seed_disallow_pair_count",
            cluster_seed_disallow_pair_count,
        )?;
        telemetry.set_item(
            "cluster_seed_disallowed_candidate_count",
            cluster_seed_disallowed_candidate_count,
        )?;
        let mut telemetry_specter_ids = HashSet::<&str>::default();
        for signature_id in seed_signature_ids.iter() {
            if self
                .state
                .features_by_signature_id
                .get(signature_id)
                .and_then(|feature| feature.query.specter.as_ref())
                .is_some()
            {
                if let Some(signature) = self.state.signatures.get(signature_id) {
                    telemetry_specter_ids.insert(signature.paper_id.as_str());
                }
            }
        }
        if let Some(query_specter) = query_inputs.specter_by_paper_id.as_ref() {
            for &paper_id in telemetry_paper_ids.iter() {
                if query_specter.contains_key(paper_id) {
                    telemetry_specter_ids.insert(paper_id);
                }
            }
        }
        telemetry.set_item("specter_count", telemetry_specter_ids.len())?;
        telemetry.set_item(
            "indexed_arrow_candidate_plan",
            self.state.build_telemetry.indexed_arrow_candidate_plan,
        )?;
        telemetry.set_item(
            "signature_batches_read",
            query_inputs.signature_index_stats.batches_read,
        )?;
        telemetry.set_item(
            "signature_rows_scanned",
            query_inputs.signature_index_stats.rows_scanned,
        )?;
        telemetry.set_item(
            "paper_batches_read",
            query_inputs.paper_index_stats.batches_read,
        )?;
        telemetry.set_item(
            "paper_rows_scanned",
            query_inputs.paper_index_stats.rows_scanned,
        )?;
        telemetry.set_item(
            "paper_author_batches_read",
            query_inputs.paper_author_index_stats.batches_read,
        )?;
        telemetry.set_item(
            "paper_author_rows_scanned",
            query_inputs.paper_author_index_stats.rows_scanned,
        )?;
        telemetry.set_item(
            "specter_batches_read",
            query_inputs.specter_index_stats.batches_read,
        )?;
        telemetry.set_item(
            "specter_rows_scanned",
            query_inputs.specter_index_stats.rows_scanned,
        )?;
        telemetry.set_item("unidecode_char_count", self.state.unidecode_char_map.len())?;
        telemetry.set_item("payload_seed_signature_count", 0usize)?;
        telemetry.set_item("planner_seed_state_reused", 1)?;
        telemetry.set_item("timings", &timings)?;

        let payload_start = Instant::now();
        let payload = PyDict::new(py);
        payload.set_item("row_count", row_component_keys.len())?;
        payload.set_item("pair_count", left_signature_indices.len())?;
        payload.set_item("query_signature_ids", query_signature_ids)?;
        payload.set_item("query_views", query_views)?;
        payload.set_item("query_authors", query_authors)?;
        payload.set_item("seed_signature_ids", Vec::<String>::new())?;
        payload.set_item("component_members", component_members)?;
        payload.set_item("pair_signature_ids", pair_signature_ids)?;
        payload.set_item(
            "left_pair_signature_indices",
            left_pair_signature_indices.into_pyarray(py),
        )?;
        payload.set_item(
            "right_pair_signature_indices",
            right_pair_signature_indices.into_pyarray(py),
        )?;
        payload.set_item("pair_row_indices", pair_row_indices.into_pyarray(py))?;
        payload.set_item(
            "row_query_signature_indices",
            row_query_signature_indices.into_pyarray(py),
        )?;
        payload.set_item("row_component_keys", row_component_keys)?;
        payload.set_item("retrieval_scores", row_retrieval_scores.into_pyarray(py))?;
        payload.set_item("retrieval_ranks", row_retrieval_ranks.into_pyarray(py))?;
        payload.set_item("row_component_sizes", row_component_sizes.into_pyarray(py))?;
        payload.set_item(
            "row_named_signature_counts",
            row_named_signature_counts.into_pyarray(py),
        )?;
        payload.set_item("row_dominant_first_names", row_dominant_first_names)?;
        payload.set_item(
            "row_candidate_year_min",
            row_candidate_year_min.into_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_year_max",
            row_candidate_year_max.into_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_year_range_missing",
            row_candidate_year_range_missing.into_pyarray(py),
        )?;
        payload.set_item("row_query_first_tokens", row_query_first_tokens)?;
        payload.set_item("row_query_years", row_query_years.into_pyarray(py))?;
        payload.set_item(
            "row_query_year_missing",
            row_query_year_missing.into_pyarray(py),
        )?;
        payload.set_item(
            "row_query_has_affiliations",
            row_query_has_affiliations.into_pyarray(py),
        )?;
        payload.set_item(
            "row_query_has_coauthors",
            row_query_has_coauthors.into_pyarray(py),
        )?;
        RawArrowEvidenceSignals {
            row_orcid_match,
            row_middle_initial_compatibility,
            row_affiliation_overlap,
            row_coauthor_overlap,
            row_venue_overlap,
            row_year_compatibility,
            row_title_overlap,
            row_specter_centroid_similarity,
            row_specter_exemplar_similarity,
            row_last_name_count_min_rarity,
            row_candidate_last_name_count_min_rarity,
            row_candidate_last_first_name_count_min_rarity,
            row_last_first_name_count_min_rarity,
            row_first_prefix_x_last_first_name_count_min_rarity,
            row_candidate_cluster_max_paper_author_count,
            row_paper_author_list_max_jaccard,
            row_paper_author_list_max_containment,
            row_paper_author_list_max_overlap_count,
            row_local_author_window10_jaccard_max,
            row_local_author_window10_overlap_count_max,
            row_best_author_count_log_absdiff,
        }
        .append_to(py, &payload)?;
        payload.set_item("telemetry", telemetry)?;
        timings.set_item("payload_secs", payload_start.elapsed().as_secs_f64())?;
        timings.set_item("total_secs", total_start.elapsed().as_secs_f64())?;
        timings.set_item("drop_secs", 0.0)?;
        timings.set_item("wall_secs", total_start.elapsed().as_secs_f64())?;
        // The caller builds the plan's featurizer next, over these queries and
        // seeds; keep their rows so it need not read them from Arrow again.
        self.state.last_query_inputs = Some(query_inputs);
        Ok(payload.unbind())
    }
}

fn raw_arrow_labeled_empty_plan(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let telemetry = PyDict::new(py);
    telemetry.set_item("signature_count", 0)?;
    telemetry.set_item("paper_count", 0)?;
    telemetry.set_item("paper_author_paper_count", 0)?;
    telemetry.set_item("component_count", 0)?;
    telemetry.set_item("query_signature_count", 0)?;
    telemetry.set_item("row_count", 0)?;
    telemetry.set_item("pair_count", 0)?;
    let timings = PyDict::new(py);
    timings.set_item("total_secs", 0.0)?;
    telemetry.set_item("timings", timings)?;

    let payload = PyDict::new(py);
    payload.set_item("row_count", 0)?;
    payload.set_item("pair_count", 0)?;
    payload.set_item("signature_ids", Vec::<String>::new())?;
    payload.set_item("query_signature_ids", Vec::<String>::new())?;
    payload.set_item("query_views", Vec::<String>::new())?;
    payload.set_item("query_authors", Vec::<String>::new())?;
    payload.set_item("row_query_signature_ids", Vec::<String>::new())?;
    payload.set_item("row_query_views", Vec::<String>::new())?;
    payload.set_item("row_query_authors", Vec::<String>::new())?;
    payload.set_item("row_query_group_ids", Vec::<String>::new())?;
    payload.set_item("row_component_keys", Vec::<String>::new())?;
    payload.set_item("left_signature_ids", Vec::<String>::new())?;
    payload.set_item("right_signature_ids", Vec::<String>::new())?;
    payload.set_item("pair_row_indices", Vec::<u32>::new().into_pyarray(py))?;
    payload.set_item("retrieval_scores", Vec::<f32>::new().into_pyarray(py))?;
    payload.set_item("retrieval_ranks", Vec::<u16>::new().into_pyarray(py))?;
    payload.set_item("row_component_sizes", Vec::<u32>::new().into_pyarray(py))?;
    payload.set_item(
        "row_named_signature_counts",
        Vec::<u32>::new().into_pyarray(py),
    )?;
    payload.set_item("row_dominant_first_names", Vec::<String>::new())?;
    payload.set_item("row_candidate_year_min", Vec::<i32>::new().into_pyarray(py))?;
    payload.set_item("row_candidate_year_max", Vec::<i32>::new().into_pyarray(py))?;
    payload.set_item(
        "row_candidate_year_range_missing",
        Vec::<u8>::new().into_pyarray(py),
    )?;
    payload.set_item("row_query_first_tokens", Vec::<String>::new())?;
    payload.set_item("row_query_years", Vec::<i32>::new().into_pyarray(py))?;
    payload.set_item("row_query_year_missing", Vec::<u8>::new().into_pyarray(py))?;
    payload.set_item(
        "row_query_has_affiliations",
        Vec::<u8>::new().into_pyarray(py),
    )?;
    payload.set_item("row_query_has_coauthors", Vec::<u8>::new().into_pyarray(py))?;
    payload.set_item("row_query_has_specter", Vec::<u8>::new().into_pyarray(py))?;
    payload.set_item(
        "row_query_has_name_counts",
        Vec::<u8>::new().into_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_affiliations",
        Vec::<u8>::new().into_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_coauthors",
        Vec::<u8>::new().into_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_specter_exemplars",
        Vec::<u8>::new().into_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_name_counts",
        Vec::<u8>::new().into_pyarray(py),
    )?;
    RawArrowEvidenceSignals::default().append_to(py, &payload)?;
    payload.set_item("telemetry", telemetry)?;
    Ok(payload.unbind())
}

fn raw_arrow_labeled_component_members(
    component_key: &str,
    raw_members: &[String],
    signatures: &HashMap<String, RawArrowSignature>,
) -> Vec<String> {
    let Some((block_key, _cluster_key)) = component_key.split_once("::") else {
        return raw_members.to_vec();
    };
    raw_members
        .iter()
        .filter(|signature_id| {
            signatures
                .get(signature_id.as_str())
                .and_then(|signature| signature.author_block.as_deref())
                .is_some_and(|author_block| author_block == block_key)
        })
        .cloned()
        .collect()
}

fn raw_arrow_component_members<'a>(
    component_key: &str,
    members_by_component: &'a HashMap<String, Vec<String>>,
) -> PyResult<&'a [String]> {
    members_by_component
        .get(component_key)
        .map(Vec::as_slice)
        .ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "candidate_component_key missing from members table: {component_key}"
            ))
        })
}

fn raw_arrow_component_summary_for_members<S: AsRef<str>>(
    component_key: &str,
    members: &[S],
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    max_exemplars: usize,
) -> PyResult<RetrievalSummaryData> {
    build_raw_arrow_summary(
        component_key,
        members,
        features_by_signature_id,
        max_exemplars,
    )
    .map_err(retrieval_error_to_py)
}

fn raw_arrow_counter_present(counter: &Option<CounterData>) -> bool {
    counter
        .as_ref()
        .is_some_and(|values| !values.entries.is_empty())
}

#[pyfunction]
#[pyo3(signature = (
    dataset,
    row_query_signature_ids,
    row_query_views,
    row_query_group_ids,
    row_component_keys,
    stored_retrieval_ranks,
    component_members,
    orcid_enabled = false,
    num_threads = None,
    max_exemplars = 4
))]
pub(crate) fn raw_arrow_labeled_candidate_plan<'py>(
    py: Python<'py>,
    dataset: PyRef<'py, ArrowDataset>,
    row_query_signature_ids: Vec<String>,
    row_query_views: Vec<String>,
    row_query_group_ids: Vec<String>,
    row_component_keys: Vec<String>,
    stored_retrieval_ranks: Vec<u16>,
    component_members: &Bound<'py, PyAny>,
    orcid_enabled: bool,
    num_threads: Option<usize>,
    max_exemplars: usize,
) -> PyResult<Py<PyDict>> {
    let total_start = Instant::now();
    let row_count = row_component_keys.len();
    if row_query_signature_ids.len() != row_count
        || row_query_views.len() != row_count
        || row_query_group_ids.len() != row_count
        || stored_retrieval_ranks.len() != row_count
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "labeled candidate row arrays must have equal length: query_ids={} query_views={} query_groups={} component_keys={} ranks={}",
            row_query_signature_ids.len(),
            row_query_views.len(),
            row_query_group_ids.len(),
            row_count,
            stored_retrieval_ranks.len()
        )));
    }
    if row_count == 0 {
        return raw_arrow_labeled_empty_plan(py);
    }
    if stored_retrieval_ranks.iter().any(|rank| *rank == 0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "stored_retrieval_ranks values must be one-based uint16 ranks in [1, 65535]",
        ));
    }

    let dataset = dataset.shared();
    let mut query_signature_ids = Vec::<String>::new();
    let mut query_seen = HashSet::<String>::default();
    for signature_id in row_query_signature_ids.iter() {
        if query_seen.insert(signature_id.clone()) {
            query_signature_ids.push(signature_id.clone());
        }
    }
    validate_raw_arrow_query_signature_ids(&query_signature_ids)?;

    let component_entries = extract_string_vec_entries(component_members)?;
    let row_component_key_set: HashSet<String> = row_component_keys.iter().cloned().collect();
    let mut raw_members_by_component = HashMap::<String, Vec<String>>::default();
    let mut component_order = Vec::<String>::new();
    for (component_key, members) in component_entries {
        if raw_members_by_component
            .insert(component_key.clone(), members)
            .is_some()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "component_members contains duplicate candidate_component_key: {component_key:?}"
            )));
        }
        component_order.push(component_key);
    }
    let missing_component_keys = row_component_key_set
        .iter()
        .filter(|component_key| !raw_members_by_component.contains_key(component_key.as_str()))
        .take(10)
        .cloned()
        .collect::<Vec<_>>();
    if !missing_component_keys.is_empty() {
        return Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "candidate rows reference component keys missing from component_members: {missing_component_keys:?}"
        )));
    }

    let mut needed_signature_ids = Vec::<String>::new();
    let mut needed_seen = HashSet::<String>::default();
    for signature_id in query_signature_ids.iter() {
        if needed_seen.insert(signature_id.clone()) {
            needed_signature_ids.push(signature_id.clone());
        }
    }
    for component_key in component_order.iter() {
        let members = raw_members_by_component.get(component_key).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "component_key '{}' disappeared while collecting needed signatures",
                component_key
            ))
        })?;
        for signature_id in members.iter() {
            if needed_seen.insert(signature_id.clone()) {
                needed_signature_ids.push(signature_id.clone());
            }
        }
    }

    let query_inputs =
        read_reusable_raw_arrow_query_inputs(py, &dataset, &needed_signature_ids, num_threads)?;
    let raw_name_counts = dataset.name_counts();
    let reused_name_counts_index = raw_name_counts.shared_index().is_some();

    let text_context_start = Instant::now();
    let text_module = py.import("s2and.text")?;
    let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
    let affiliation_stopwords = extract_affiliation_stopwords(py)?;
    let mut unidecode_char_map: HashMap<char, String> = HashMap::default();
    ensure_unidecode_for_raw_arrow_inputs(
        &query_inputs.signatures,
        &query_inputs.papers,
        &query_inputs.paper_authors,
        &mut unidecode_char_map,
        num_threads,
    )?;
    let text_context_secs = text_context_start.elapsed().as_secs_f64();

    let specter_by_paper_id = query_inputs.specter_by_paper_id.as_ref();
    let feature_start = Instant::now();
    let mut author_index = PaperAuthorIndex::default();
    let raw_feature_results: Result<Vec<(String, RawArrowFeature)>, String> =
        py.allow_threads(|| {
            let compute = || {
                let coauthor_blocks = paper_coauthor_blocks(
                    query_inputs.signatures.values(),
                    &query_inputs.paper_authors,
                    &unidecode_char_map,
                )?;
                author_index.extend(
                    query_inputs
                        .signatures
                        .values()
                        .map(|signature| signature.paper_id.as_str()),
                    &query_inputs.paper_authors,
                    &unidecode_char_map,
                )?;
                needed_signature_ids
                    .par_iter()
                    .map(|signature_id| {
                        let signature =
                            query_inputs.signatures.get(signature_id).ok_or_else(|| {
                                format!(
                                    "signature_id '{}' is missing from signatures",
                                    signature_id
                                )
                            })?;
                        let paper = query_inputs.papers.get(&signature.paper_id);
                        Ok((
                            signature_id.clone(),
                            build_raw_arrow_feature(
                                signature,
                                paper,
                                coauthor_blocks.get(&signature.paper_id),
                                specter_by_paper_id,
                                &raw_name_counts,
                                &name_prefixes,
                                &affiliation_stopwords,
                                &unidecode_char_map,
                                orcid_enabled,
                            ),
                        ))
                    })
                    .collect()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
    let features_by_signature_id: HashMap<String, RawArrowFeature> = raw_feature_results
        .map_err(pyo3::exceptions::PyKeyError::new_err)?
        .into_iter()
        .collect();
    let feature_secs = feature_start.elapsed().as_secs_f64();

    let mut members_by_component = HashMap::<String, Vec<String>>::default();
    for component_key in component_order.iter() {
        let raw_members = raw_members_by_component.get(component_key).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "component_key '{}' disappeared while filtering members",
                component_key
            ))
        })?;
        let members = raw_arrow_labeled_component_members(
            component_key,
            raw_members,
            &query_inputs.signatures,
        );
        for signature_id in members.iter() {
            if !features_by_signature_id.contains_key(signature_id) {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "component member signature_id '{}' is missing from computed raw Arrow features",
                    signature_id
                )));
            }
        }
        members_by_component.insert(component_key.clone(), members);
    }

    let summary_start = Instant::now();
    let retriever = build_retriever_from_raw_arrow_components(
        py,
        &component_order,
        &members_by_component,
        &features_by_signature_id,
        max_exemplars,
        num_threads,
        false,
    )?;
    let summary_secs = summary_start.elapsed().as_secs_f64();

    let mut group_order = Vec::<String>::new();
    let mut group_seen = HashSet::<String>::default();
    let mut rows_by_group = HashMap::<String, Vec<usize>>::default();
    for (row_index, group_id) in row_query_group_ids.iter().enumerate() {
        if group_seen.insert(group_id.clone()) {
            group_order.push(group_id.clone());
        }
        rows_by_group
            .entry(group_id.clone())
            .or_default()
            .push(row_index);
    }

    let mut row_retrieval_scores = vec![0.0f32; row_count];
    let mut row_retrieval_ranks = vec![0u16; row_count];
    let mut resolved_row_query_views = vec![String::new(); row_count];
    for group_id in group_order.iter() {
        let row_indices = rows_by_group.get(group_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "query group '{}' disappeared while scoring labeled candidates",
                group_id
            ))
        })?;
        let first_row = row_indices[0];
        let query_signature_id = row_query_signature_ids[first_row].as_str();
        let requested_view = row_query_views[first_row].as_str();
        for row_index in row_indices.iter().copied() {
            if row_query_signature_ids[row_index] != query_signature_id
                || row_query_views[row_index] != requested_view
            {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query group {group_id:?} is not a single query/view"
                )));
            }
        }
        let query_feature = features_by_signature_id
            .get(query_signature_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "query signature_id '{}' is missing from computed raw Arrow features",
                    query_signature_id
                ))
            })?;
        let (query, resolved_view) = mask_raw_arrow_query(&query_feature.query, requested_view)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let mut scored_rows = Vec::<(usize, f32, u16, String)>::with_capacity(row_indices.len());
        for row_index in row_indices.iter().copied() {
            resolved_row_query_views[row_index] = resolved_view.clone();
            let component_key = row_component_keys[row_index].as_str();
            let component_members =
                raw_arrow_component_members(component_key, &members_by_component)?;
            let query_is_component_member = component_members
                .iter()
                .any(|signature_id| signature_id == query_signature_id);
            let summary = if !query_is_component_member {
                let component_index = retriever
                    .component_index_by_key
                    .get(component_key)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "Unknown component_key for raw Arrow labeled scoring: {component_key}"
                        ))
                    })?;
                retriever.summaries[*component_index].clone()
            } else {
                let active_members: Vec<&str> = component_members
                    .iter()
                    .map(String::as_str)
                    .filter(|signature_id| *signature_id != query_signature_id)
                    .collect();
                raw_arrow_component_summary_for_members(
                    component_key,
                    &active_members,
                    &features_by_signature_id,
                    max_exemplars,
                )?
            };
            let score = round_six(score_hybrid_centroid_query(
                &query,
                &summary,
                &retriever.coauthor_cluster_df,
                &retriever.non_mega_coauthor_cluster_df,
                &retriever.affiliation_cluster_df,
                retriever.summaries.len(),
            ) as f64);
            scored_rows.push((
                row_index,
                score,
                stored_retrieval_ranks[row_index],
                component_key.to_string(),
            ));
        }
        scored_rows.sort_unstable_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.3.cmp(&right.3))
        });
        for (rank_offset, (row_index, score, _stored_rank, _component_key)) in
            scored_rows.into_iter().enumerate()
        {
            row_retrieval_scores[row_index] = score;
            row_retrieval_ranks[row_index] = retrieval_rank_from_zero_based_offset(
                rank_offset,
                "raw_arrow_labeled_candidate_plan",
            )
            .map_err(retrieval_error_to_py)?;
        }
    }

    let mut signature_ids = Vec::<String>::new();
    let mut signature_index_by_id = HashMap::<String, u32>::default();
    for signature_id in row_query_signature_ids.iter() {
        if !signature_index_by_id.contains_key(signature_id) {
            let signature_index = u32::try_from(signature_ids.len()).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err("signature id table exceeds u32")
            })?;
            signature_index_by_id.insert(signature_id.clone(), signature_index);
            signature_ids.push(signature_id.clone());
        }
    }
    let mut left_signature_indices = Vec::<u32>::new();
    let mut right_signature_indices = Vec::<u32>::new();
    let mut pair_row_indices = Vec::<u32>::new();
    for (row_index, (query_signature_id, component_key)) in row_query_signature_ids
        .iter()
        .zip(row_component_keys.iter())
        .enumerate()
    {
        let query_signature_index =
            *signature_index_by_id
                .get(query_signature_id)
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "query signature_id '{}' is missing from signature id table",
                        query_signature_id
                    ))
                })?;
        let component_members = raw_arrow_component_members(component_key, &members_by_component)?;
        for member_signature_id in component_members
            .iter()
            .filter(|signature_id| signature_id.as_str() != query_signature_id)
        {
            let member_signature_index =
                if let Some(index) = signature_index_by_id.get(member_signature_id) {
                    *index
                } else {
                    let index = u32::try_from(signature_ids.len()).map_err(|_| {
                        pyo3::exceptions::PyOverflowError::new_err("signature id table exceeds u32")
                    })?;
                    signature_index_by_id.insert(member_signature_id.clone(), index);
                    signature_ids.push(member_signature_id.clone());
                    index
                };
            left_signature_indices.push(query_signature_index);
            right_signature_indices.push(member_signature_index);
            pair_row_indices.push(u32::try_from(row_index).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(
                    "labeled candidate row index exceeds u32",
                )
            })?);
        }
    }

    let mut row_query_authors = Vec::<String>::with_capacity(row_count);
    let mut row_component_sizes = Vec::<u32>::with_capacity(row_count);
    let mut row_named_signature_counts = Vec::<u32>::with_capacity(row_count);
    let mut row_dominant_first_names = Vec::<String>::with_capacity(row_count);
    let mut row_candidate_year_min = Vec::<i32>::with_capacity(row_count);
    let mut row_candidate_year_max = Vec::<i32>::with_capacity(row_count);
    let mut row_candidate_year_range_missing = Vec::<u8>::with_capacity(row_count);
    let mut row_query_first_tokens = Vec::<String>::with_capacity(row_count);
    let mut row_query_years = Vec::<i32>::with_capacity(row_count);
    let mut row_query_year_missing = Vec::<u8>::with_capacity(row_count);
    let mut row_query_has_affiliations = Vec::<u8>::with_capacity(row_count);
    let mut row_query_has_coauthors = Vec::<u8>::with_capacity(row_count);
    let mut row_query_has_specter = Vec::<u8>::with_capacity(row_count);
    let mut row_query_has_name_counts = Vec::<u8>::with_capacity(row_count);
    let mut row_candidate_has_affiliations = Vec::<u8>::with_capacity(row_count);
    let mut row_candidate_has_coauthors = Vec::<u8>::with_capacity(row_count);
    let mut row_candidate_has_specter_exemplars = Vec::<u8>::with_capacity(row_count);
    let mut row_candidate_has_name_counts = Vec::<u8>::with_capacity(row_count);
    let mut row_orcid_match = Vec::<u8>::with_capacity(row_count);
    let mut row_middle_initial_compatibility = Vec::<f32>::with_capacity(row_count);
    let mut row_affiliation_overlap = Vec::<f32>::with_capacity(row_count);
    let mut row_coauthor_overlap = Vec::<f32>::with_capacity(row_count);
    let mut row_venue_overlap = Vec::<f32>::with_capacity(row_count);
    let mut row_year_compatibility = Vec::<f32>::with_capacity(row_count);
    let mut row_title_overlap = Vec::<f32>::with_capacity(row_count);
    let mut row_specter_centroid_similarity = Vec::<f32>::with_capacity(row_count);
    let mut row_specter_exemplar_similarity = Vec::<f32>::with_capacity(row_count);
    let mut row_last_name_count_min_rarity = Vec::<f32>::with_capacity(row_count);
    let mut row_candidate_last_name_count_min_rarity = Vec::<f32>::with_capacity(row_count);
    let mut row_candidate_last_first_name_count_min_rarity = Vec::<f32>::with_capacity(row_count);
    let mut row_last_first_name_count_min_rarity = Vec::<f32>::with_capacity(row_count);
    let mut row_first_prefix_x_last_first_name_count_min_rarity =
        Vec::<f32>::with_capacity(row_count);
    let mut row_candidate_cluster_max_paper_author_count = Vec::<f32>::with_capacity(row_count);
    let mut row_paper_author_list_max_jaccard = Vec::<f32>::with_capacity(row_count);
    let mut row_paper_author_list_max_containment = Vec::<f32>::with_capacity(row_count);
    let mut row_paper_author_list_max_overlap_count = Vec::<f32>::with_capacity(row_count);
    let mut row_local_author_window10_jaccard_max = Vec::<f32>::with_capacity(row_count);
    let mut row_local_author_window10_overlap_count_max = Vec::<f32>::with_capacity(row_count);
    let mut row_best_author_count_log_absdiff = Vec::<f32>::with_capacity(row_count);
    let mut full_summary_signal_cache = HashMap::<String, RawArrowSummarySignalData>::default();
    let mut residual_summary_signal_cache =
        HashMap::<(String, String), RawArrowSummarySignalData>::default();
    let mut author_signals_by_query_signature_id =
        HashMap::<String, RawArrowAuthorSignalData>::default();
    for row_index in 0..row_count {
        let query_signature_id = row_query_signature_ids[row_index].as_str();
        let component_key = row_component_keys[row_index].as_str();
        let query_feature = features_by_signature_id
            .get(query_signature_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "query signature_id '{}' is missing from computed raw Arrow features",
                    query_signature_id
                ))
            })?;
        let (query, resolved_view) =
            mask_raw_arrow_query(&query_feature.query, row_query_views[row_index].as_str())
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
        resolved_row_query_views[row_index] = resolved_view;
        let component_members = raw_arrow_component_members(component_key, &members_by_component)?;
        let query_is_component_member = component_members
            .iter()
            .any(|signature_id| signature_id == query_signature_id);
        let (summary, summary_signals) = if !query_is_component_member {
            let component_index = retriever
                .component_index_by_key
                .get(component_key)
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "Unknown component_key for raw Arrow labeled row signals: {component_key}"
                    ))
                })?;
            let summary = retriever.summaries[*component_index].clone();
            let signals = raw_arrow_summary_signals_cached(
                &mut full_summary_signal_cache,
                component_key,
                &members_by_component,
                &features_by_signature_id,
                &query_inputs.signatures,
                &author_index,
            )?;
            (summary, signals)
        } else {
            let active_members: Vec<&str> = component_members
                .iter()
                .map(String::as_str)
                .filter(|signature_id| *signature_id != query_signature_id)
                .collect();
            let summary = raw_arrow_component_summary_for_members(
                component_key,
                &active_members,
                &features_by_signature_id,
                max_exemplars,
            )?;
            let signals = raw_arrow_summary_signals_for_members_cached(
                &mut residual_summary_signal_cache,
                component_key,
                query_signature_id,
                &active_members,
                &features_by_signature_id,
                &query_inputs.signatures,
                &author_index,
            )?;
            (summary, signals)
        };

        if let Entry::Vacant(entry) =
            author_signals_by_query_signature_id.entry(query_signature_id.to_string())
        {
            let query_signature =
                query_inputs
                    .signatures
                    .get(query_signature_id)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "query signature_id '{}' is missing from signatures",
                            query_signature_id
                        ))
                    })?;
            entry.insert(
                author_index
                    .author_signals(query_signature)
                    .map_err(pyo3::exceptions::PyKeyError::new_err)?,
            );
        }
        let query_author_signals = author_signals_by_query_signature_id
            .get(query_signature_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "query signature_id '{}' is missing raw author signals",
                    query_signature_id
                ))
            })?;
        let rarity = raw_arrow_name_count_rarity_row(
            &query,
            &query_feature.name_counts,
            &summary,
            summary_signals,
        );
        let evidence = raw_arrow_paper_evidence_row(
            query_signature_id,
            query_feature.paper_author_count,
            query_author_signals,
            summary_signals,
        );
        let chooser_features = chooser_summary_features(&query, &summary);
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
        let (candidate_year_min, candidate_year_min_missing) =
            year_signal_value(summary.year_min, "candidate year_min")
                .map_err(retrieval_error_to_py)?;
        let (candidate_year_max, candidate_year_max_missing) =
            year_signal_value(summary.year_max, "candidate year_max")
                .map_err(retrieval_error_to_py)?;
        let (query_year, query_year_missing) =
            year_signal_value(query.year, "query year").map_err(retrieval_error_to_py)?;
        row_query_authors.push(query_feature.query_author.clone());
        row_component_sizes.push(summary.size.min(u32::MAX as usize) as u32);
        row_named_signature_counts.push(
            row_named_signature_count(&summary.first_name_counts).map_err(retrieval_error_to_py)?,
        );
        row_dominant_first_names.push(dominant_first_name.to_string());
        row_candidate_year_min.push(candidate_year_min);
        row_candidate_year_max.push(candidate_year_max);
        row_candidate_year_range_missing.push(u8::from(
            candidate_year_min_missing != 0 || candidate_year_max_missing != 0,
        ));
        row_query_first_tokens.push(query_feature.query.first.clone());
        row_query_years.push(query_year);
        row_query_year_missing.push(query_year_missing);
        row_query_has_affiliations.push(u8::from(!query.affiliation_hashes.is_empty()));
        row_query_has_coauthors.push(u8::from(!query.coauthor_hashes.is_empty()));
        row_query_has_specter.push(u8::from(query.specter.is_some()));
        row_query_has_name_counts.push(u8::from(query_feature.name_counts.is_some()));
        row_candidate_has_affiliations.push(u8::from(
            raw_arrow_counter_present(&summary.affiliation_counts) && summary.size > 0,
        ));
        row_candidate_has_coauthors.push(u8::from(
            raw_arrow_counter_present(&summary.coauthor_counts) && summary.size > 0,
        ));
        row_candidate_has_specter_exemplars.push(u8::from(!summary.exemplar_vectors.is_empty()));
        row_candidate_has_name_counts
            .push(u8::from(!summary_signals.name_counts_values.is_empty()));
        row_orcid_match.push(u8::from(query.orcid_hash.is_some_and(|orcid_hash| {
            contains_hashed_value(&summary.orcid_hashes, orcid_hash)
        })));
        row_middle_initial_compatibility.push(chooser_features[0]);
        row_affiliation_overlap.push(chooser_features[1]);
        row_coauthor_overlap.push(chooser_features[2]);
        row_venue_overlap.push(chooser_features[3]);
        row_year_compatibility.push(chooser_features[4]);
        row_title_overlap.push(chooser_features[5]);
        row_specter_centroid_similarity.push(chooser_features[6]);
        row_specter_exemplar_similarity.push(chooser_features[7]);
        row_last_name_count_min_rarity.push(rarity.last_name_count_min_rarity);
        row_candidate_last_name_count_min_rarity.push(rarity.candidate_last_name_count_min_rarity);
        row_candidate_last_first_name_count_min_rarity
            .push(rarity.candidate_last_first_name_count_min_rarity);
        row_last_first_name_count_min_rarity.push(rarity.last_first_name_count_min_rarity);
        row_first_prefix_x_last_first_name_count_min_rarity
            .push(rarity.first_prefix_x_last_first_name_count_min_rarity);
        row_candidate_cluster_max_paper_author_count.push(summary.max_paper_author_count as f32);
        row_paper_author_list_max_jaccard.push(evidence.paper_author_list_max_jaccard);
        row_paper_author_list_max_containment.push(evidence.paper_author_list_max_containment);
        row_paper_author_list_max_overlap_count.push(evidence.paper_author_list_max_overlap_count);
        row_local_author_window10_jaccard_max.push(evidence.local_author_window10_jaccard_max);
        row_local_author_window10_overlap_count_max
            .push(evidence.local_author_window10_overlap_count_max);
        row_best_author_count_log_absdiff.push(evidence.best_author_count_log_absdiff);
    }

    let mut query_view_by_signature_id = HashMap::<String, String>::default();
    let mut query_author_by_signature_id = HashMap::<String, String>::default();
    for ((signature_id, resolved_view), query_author) in row_query_signature_ids
        .iter()
        .zip(resolved_row_query_views.iter())
        .zip(row_query_authors.iter())
    {
        match query_view_by_signature_id.entry(signature_id.clone()) {
            Entry::Occupied(entry) => {
                if entry.get() != resolved_view {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "query signature_id {signature_id:?} has multiple resolved query views"
                    )));
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(resolved_view.clone());
            }
        }
        match query_author_by_signature_id.entry(signature_id.clone()) {
            Entry::Occupied(entry) => {
                if entry.get() != query_author {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "query signature_id {signature_id:?} has multiple query authors"
                    )));
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(query_author.clone());
            }
        }
    }
    let query_views = query_signature_ids
        .iter()
        .map(|signature_id| {
            query_view_by_signature_id
                .get(signature_id)
                .cloned()
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "query signature_id {signature_id:?} is missing a resolved query view"
                    ))
                })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let query_authors = query_signature_ids
        .iter()
        .map(|signature_id| {
            query_author_by_signature_id
                .get(signature_id)
                .cloned()
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "query signature_id {signature_id:?} is missing a query author"
                    ))
                })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let timings = PyDict::new(py);
    timings.set_item("read_signatures_secs", query_inputs.read_signatures_secs)?;
    timings.set_item("read_papers_secs", query_inputs.read_papers_secs)?;
    timings.set_item(
        "read_paper_authors_secs",
        query_inputs.read_paper_authors_secs,
    )?;
    timings.set_item("read_specter_secs", query_inputs.read_specter_secs)?;
    timings.set_item(
        "metadata_reads_parallel_secs",
        query_inputs.metadata_reads_parallel_secs,
    )?;
    timings.set_item("text_context_secs", text_context_secs)?;
    timings.set_item("feature_secs", feature_secs)?;
    timings.set_item("summary_secs", summary_secs)?;
    timings.set_item("total_secs", total_start.elapsed().as_secs_f64())?;

    let signature_id_objects: Vec<_> = signature_ids
        .iter()
        .map(|signature_id| PyString::new(py, signature_id).unbind())
        .collect();
    let signature_ids_payload = PyList::new(
        py,
        signature_id_objects
            .iter()
            .map(|signature_id| signature_id.clone_ref(py)),
    )?;
    let left_signature_ids = PyList::new(
        py,
        left_signature_indices
            .iter()
            .map(|index| signature_id_objects[*index as usize].clone_ref(py)),
    )?;
    let right_signature_ids = PyList::new(
        py,
        right_signature_indices
            .iter()
            .map(|index| signature_id_objects[*index as usize].clone_ref(py)),
    )?;
    let pair_count = left_signature_indices.len();
    let telemetry = PyDict::new(py);
    telemetry.set_item("signature_count", signature_ids.len())?;
    telemetry.set_item("paper_count", query_inputs.papers.len())?;
    telemetry.set_item("paper_author_paper_count", query_inputs.paper_authors.len())?;
    telemetry.set_item("component_count", component_order.len())?;
    telemetry.set_item("query_signature_count", query_signature_ids.len())?;
    telemetry.set_item("row_count", row_count)?;
    telemetry.set_item("pair_count", pair_count)?;
    telemetry.set_item("component_scope", "block-local")?;
    telemetry.set_item("orcid_enabled", orcid_enabled)?;
    telemetry.set_item("reused_name_counts_index", reused_name_counts_index)?;
    telemetry.set_item(
        "signature_batches_read",
        query_inputs.signature_index_stats.batches_read,
    )?;
    telemetry.set_item(
        "signature_rows_scanned",
        query_inputs.signature_index_stats.rows_scanned,
    )?;
    telemetry.set_item(
        "paper_batches_read",
        query_inputs.paper_index_stats.batches_read,
    )?;
    telemetry.set_item(
        "paper_rows_scanned",
        query_inputs.paper_index_stats.rows_scanned,
    )?;
    telemetry.set_item(
        "paper_author_batches_read",
        query_inputs.paper_author_index_stats.batches_read,
    )?;
    telemetry.set_item(
        "paper_author_rows_scanned",
        query_inputs.paper_author_index_stats.rows_scanned,
    )?;
    telemetry.set_item(
        "specter_batches_read",
        query_inputs.specter_index_stats.batches_read,
    )?;
    telemetry.set_item(
        "specter_rows_scanned",
        query_inputs.specter_index_stats.rows_scanned,
    )?;
    telemetry.set_item("unidecode_char_count", unidecode_char_map.len())?;
    telemetry.set_item("timings", &timings)?;

    let payload = PyDict::new(py);
    payload.set_item("row_count", row_count)?;
    payload.set_item("pair_count", pair_count)?;
    payload.set_item("signature_ids", signature_ids_payload)?;
    payload.set_item("query_signature_ids", query_signature_ids)?;
    payload.set_item("row_query_signature_ids", row_query_signature_ids)?;
    payload.set_item("row_query_views", resolved_row_query_views.clone())?;
    payload.set_item("row_query_authors", row_query_authors)?;
    payload.set_item("row_query_group_ids", row_query_group_ids)?;
    payload.set_item("row_component_keys", row_component_keys)?;
    payload.set_item("query_views", query_views)?;
    payload.set_item("query_authors", query_authors)?;
    payload.set_item("left_signature_ids", left_signature_ids)?;
    payload.set_item("right_signature_ids", right_signature_ids)?;
    payload.set_item("pair_row_indices", pair_row_indices.into_pyarray(py))?;
    payload.set_item("retrieval_scores", row_retrieval_scores.into_pyarray(py))?;
    payload.set_item("retrieval_ranks", row_retrieval_ranks.into_pyarray(py))?;
    payload.set_item("row_component_sizes", row_component_sizes.into_pyarray(py))?;
    payload.set_item(
        "row_named_signature_counts",
        row_named_signature_counts.into_pyarray(py),
    )?;
    payload.set_item("row_dominant_first_names", row_dominant_first_names)?;
    payload.set_item(
        "row_candidate_year_min",
        row_candidate_year_min.into_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_year_max",
        row_candidate_year_max.into_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_year_range_missing",
        row_candidate_year_range_missing.into_pyarray(py),
    )?;
    payload.set_item("row_query_first_tokens", row_query_first_tokens)?;
    payload.set_item("row_query_years", row_query_years.into_pyarray(py))?;
    payload.set_item(
        "row_query_year_missing",
        row_query_year_missing.into_pyarray(py),
    )?;
    payload.set_item(
        "row_query_has_affiliations",
        row_query_has_affiliations.into_pyarray(py),
    )?;
    payload.set_item(
        "row_query_has_coauthors",
        row_query_has_coauthors.into_pyarray(py),
    )?;
    payload.set_item(
        "row_query_has_specter",
        row_query_has_specter.into_pyarray(py),
    )?;
    payload.set_item(
        "row_query_has_name_counts",
        row_query_has_name_counts.into_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_affiliations",
        row_candidate_has_affiliations.into_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_coauthors",
        row_candidate_has_coauthors.into_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_specter_exemplars",
        row_candidate_has_specter_exemplars.into_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_name_counts",
        row_candidate_has_name_counts.into_pyarray(py),
    )?;
    RawArrowEvidenceSignals {
        row_orcid_match,
        row_middle_initial_compatibility,
        row_affiliation_overlap,
        row_coauthor_overlap,
        row_venue_overlap,
        row_year_compatibility,
        row_title_overlap,
        row_specter_centroid_similarity,
        row_specter_exemplar_similarity,
        row_last_name_count_min_rarity,
        row_candidate_last_name_count_min_rarity,
        row_candidate_last_first_name_count_min_rarity,
        row_last_first_name_count_min_rarity,
        row_first_prefix_x_last_first_name_count_min_rarity,
        row_candidate_cluster_max_paper_author_count,
        row_paper_author_list_max_jaccard,
        row_paper_author_list_max_containment,
        row_paper_author_list_max_overlap_count,
        row_local_author_window10_jaccard_max,
        row_local_author_window10_overlap_count_max,
        row_best_author_count_log_absdiff,
    }
    .append_to(py, &payload)?;
    payload.set_item("telemetry", telemetry)?;
    Ok(payload.unbind())
}
