use super::*;
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
}

struct ReusableRawArrowCandidatePlanState {
    features_by_signature_id: HashMap<String, RawArrowFeature>,
    signatures: HashMap<String, RawArrowSignature>,
    paper_authors: HashMap<String, Vec<(i64, String)>>,
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
    let mut seen = HashSet::<&str>::with_capacity(query_signature_ids.len());
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

fn validate_signatures_batch_index_before_missing_signature_error(
    paths: &RawArrowPlannerPaths,
) -> PyResult<()> {
    if let Some(index_path) = paths.signatures_batch_index_path.as_deref() {
        crate::arrow_batch_lookup::validate_arrow_batch_lookup_index(
            index_path,
            &paths.signatures_path,
            "signature_id",
        )?;
    }
    Ok(())
}

fn build_retriever_from_raw_arrow_components(
    py: Python<'_>,
    component_order: &[String],
    members_by_component: &HashMap<String, Vec<String>>,
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    max_exemplars: usize,
    num_threads: Option<usize>,
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
    let mut summaries = Vec::<RetrievalSummaryData>::with_capacity(summary_results.len());
    for result in summary_results {
        summaries.push(result.map_err(retrieval_error_to_py)?);
    }
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
    Ok(RustHybridCentroidRetriever {
        summaries,
        component_index_by_key,
        coauthor_cluster_df,
        non_mega_coauthor_cluster_df,
        affiliation_cluster_df,
    })
}

fn raw_arrow_component_member_indices_for_batch(
    component_order: &[String],
    members_by_component: &HashMap<String, Vec<String>>,
    query_count: usize,
) -> PyResult<(HashMap<String, Vec<u32>>, Vec<String>, Vec<String>)> {
    let mut component_member_indices = HashMap::<String, Vec<u32>>::new();
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
) -> PyResult<(Option<Vec<Option<HashSet<usize>>>>, usize)> {
    let query_signature_id_set: HashSet<&str> =
        query_signature_ids.iter().map(String::as_str).collect();
    let mut disallowed_members_by_query = HashMap::<String, HashSet<String>>::new();
    for (left, right) in cluster_seed_disallows.iter() {
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
    }
    let excluded_indices_by_query = if disallowed_members_by_query.is_empty() {
        None
    } else {
        let mut component_index_by_key =
            HashMap::<&str, usize>::with_capacity(component_order.len());
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
                    let mut excluded_indices = HashSet::<usize>::new();
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
    ))
}

fn raw_arrow_summary_signals_cached<'a>(
    cache: &'a mut HashMap<String, RawArrowSummarySignalData>,
    component_key: &str,
    members_by_component: &HashMap<String, Vec<String>>,
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    signatures: &HashMap<String, RawArrowSignature>,
    paper_authors: &HashMap<String, Vec<(i64, String)>>,
    unidecode_char_map: &HashMap<char, String>,
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
            paper_authors,
            unidecode_char_map,
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

pub(crate) fn raw_arrow_summary_signals_for_members_cached<'a>(
    cache: &'a mut HashMap<(String, String), RawArrowSummarySignalData>,
    component_key: &str,
    excluded_query_signature_id: &str,
    members: &[String],
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    signatures: &HashMap<String, RawArrowSignature>,
    paper_authors: &HashMap<String, Vec<(i64, String)>>,
    unidecode_char_map: &HashMap<char, String>,
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
            paper_authors,
            unidecode_char_map,
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
    paths: &RawArrowPlannerPaths,
    query_signature_ids: &[String],
    num_threads: Option<usize>,
) -> PyResult<RawArrowQueryInputReadResult> {
    let query_signature_id_set: HashSet<String> = query_signature_ids.iter().cloned().collect();
    let read_signatures_start = Instant::now();
    let (signatures, signature_index_stats) = read_raw_arrow_signatures_with_optional_index(
        &paths.signatures_path,
        paths.signatures_batch_index_path.as_deref(),
        Some(&query_signature_id_set),
    )?;
    let read_signatures_secs = read_signatures_start.elapsed().as_secs_f64();
    for signature_id in query_signature_ids.iter() {
        if !signatures.contains_key(signature_id) {
            validate_signatures_batch_index_before_missing_signature_error(paths)?;
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
                                let (loaded, stats) = read_raw_arrow_papers_with_optional_index(
                                    &paths.papers_path,
                                    paths.papers_batch_index_path.as_deref(),
                                    &needed_paper_ids,
                                )?;
                                Ok((loaded, stats, start.elapsed().as_secs_f64()))
                            },
                            || -> PyResult<(
                                HashMap<String, Vec<(i64, String)>>,
                                IndexedArrowReadStats,
                                f64,
                            )> {
                                let start = Instant::now();
                                let (loaded, stats) =
                                    read_raw_arrow_paper_authors_with_optional_index(
                                        &paths.paper_authors_path,
                                        paths.paper_authors_batch_index_path.as_deref(),
                                        &needed_paper_ids,
                                    )?;
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
                        let (loaded, stats) = match paths.specter_path.as_ref() {
                            Some(path) => {
                                let (loaded, stats) = read_raw_arrow_specter_with_optional_index(
                                    path,
                                    paths.specter_batch_index_path.as_deref(),
                                    &needed_paper_ids,
                                )?;
                                (loaded, stats)
                            }
                            None => (HashMap::new(), IndexedArrowReadStats::default()),
                        };
                        Ok((
                            if paths.specter_path.is_some() {
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

enum RawPlannerQueryMode {
    Declared,
    Auto,
}

#[pyclass]
pub(crate) struct RawBlockQueryCandidatePlanner {
    paths: RawArrowPlannerPaths,
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
}

impl RawBlockQueryCandidatePlanner {
    fn build_from_query_signature_ids(
        py: Python<'_>,
        paths: &Bound<'_, PyDict>,
        query_signature_ids: Vec<String>,
        top_k: usize,
        orcid_enabled: bool,
        num_threads: Option<usize>,
        max_exemplars: usize,
    ) -> PyResult<Self> {
        validate_retrieval_rank_top_k(top_k)?;
        validate_raw_arrow_query_signature_ids_allow_empty(&query_signature_ids)?;
        let planner_query_signature_id_set = query_signature_ids.iter().cloned().collect();
        let paths = RawArrowPlannerPaths::from_py_dict(paths)?;

        let read_cluster_seeds_start = Instant::now();
        let (component_order, members_by_component) =
            read_raw_arrow_cluster_seeds(&paths.cluster_seeds_path)?;
        let read_cluster_seeds_secs = read_cluster_seeds_start.elapsed().as_secs_f64();
        let cluster_seed_disallows = match paths.cluster_seed_disallows_path.as_ref() {
            Some(path) => read_raw_arrow_cluster_seed_disallows(path)?,
            None => HashSet::new(),
        };

        let mut seed_signature_ids = Vec::<String>::new();
        let mut component_keys_by_member = HashMap::<String, Vec<String>>::new();
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

        let read_signatures_start = Instant::now();
        let (signatures, signature_index_stats) = if seed_signature_id_set.is_empty() {
            (HashMap::new(), IndexedArrowReadStats::default())
        } else {
            read_raw_arrow_signatures_with_optional_index(
                &paths.signatures_path,
                paths.signatures_batch_index_path.as_deref(),
                Some(&seed_signature_id_set),
            )?
        };
        let read_signatures_secs = read_signatures_start.elapsed().as_secs_f64();
        for signature_id in seed_signature_ids.iter() {
            if !signatures.contains_key(signature_id) {
                validate_signatures_batch_index_before_missing_signature_error(&paths)?;
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "cluster seed signature_id '{}' is missing from signatures Arrow input",
                    signature_id
                )));
            }
        }

        let needed_paper_ids: HashSet<String> = seed_signature_ids
            .iter()
            .filter_map(|signature_id| signatures.get(signature_id))
            .map(|signature| signature.paper_id.clone())
            .collect();
        let metadata_reads_parallel_start = Instant::now();
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
                                    (HashMap::new(), IndexedArrowReadStats::default())
                                } else {
                                    read_raw_arrow_papers_with_optional_index(
                                        &paths.papers_path,
                                        paths.papers_batch_index_path.as_deref(),
                                        &needed_paper_ids,
                                    )?
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
                                    (HashMap::new(), IndexedArrowReadStats::default())
                                } else {
                                    read_raw_arrow_paper_authors_with_optional_index(
                                        &paths.paper_authors_path,
                                        paths.paper_authors_batch_index_path.as_deref(),
                                        &needed_paper_ids,
                                    )?
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
                                let (loaded, stats) = match paths.specter_path.as_ref() {
                                    Some(path) if !needed_paper_ids.is_empty() => {
                                        let (loaded, stats) =
                                            read_raw_arrow_specter_with_optional_index(
                                                path,
                                                paths.specter_batch_index_path.as_deref(),
                                                &needed_paper_ids,
                                            )?;
                                        (loaded, stats)
                                    }
                                    Some(_) | None => {
                                        (HashMap::new(), IndexedArrowReadStats::default())
                                    }
                                };
                                Ok((
                                    if paths.specter_path.is_some() {
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
                                let loaded = match paths.name_counts_index_path.as_ref() {
                                    Some(path) => read_raw_name_counts_index(path)?,
                                    None => RawNameCountMaps::default(),
                                };
                                Ok((loaded, start.elapsed().as_secs_f64()))
                            },
                        )
                    },
                )
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        let (papers, paper_index_stats, read_papers_secs) = papers_result?;
        let (paper_authors, paper_author_index_stats, read_paper_authors_secs) =
            paper_authors_result?;
        let (raw_specter_by_paper_id, specter_index_stats, read_specter_secs) =
            raw_specter_by_paper_id_result?;
        let (raw_name_counts, read_name_counts_secs) = raw_name_counts_result?;
        let specter_by_paper_id = raw_specter_by_paper_id.map(|values| {
            values
                .into_iter()
                .map(|(paper_id, vector)| (paper_id, Arc::new(vector)))
                .collect::<HashMap<_, _>>()
        });

        let text_context_start = Instant::now();
        let text_module = py.import("s2and.text")?;
        let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
        let affiliation_stopwords = extract_affiliation_stopwords(py)?;
        let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
        ensure_unidecode_for_raw_arrow_inputs(
            &signatures,
            &papers,
            &paper_authors,
            &mut unidecode_char_map,
        )?;
        let text_context_secs = text_context_start.elapsed().as_secs_f64();

        let feature_start = Instant::now();
        let raw_feature_results: Vec<Result<(String, RawArrowFeature), String>> =
            py.allow_threads(|| {
                let compute = || {
                    seed_signature_ids
                        .par_iter()
                        .map(|signature_id| {
                            let signature = signatures.get(signature_id).ok_or_else(|| {
                                format!(
                                    "signature_id '{}' is missing from signatures",
                                    signature_id
                                )
                            })?;
                            let paper = papers.get(&signature.paper_id);
                            let authors = paper_authors.get(&signature.paper_id);
                            Ok((
                                signature_id.clone(),
                                build_raw_arrow_feature(
                                    signature,
                                    paper,
                                    authors,
                                    specter_by_paper_id.as_ref(),
                                    &raw_name_counts,
                                    &name_prefixes,
                                    &affiliation_stopwords,
                                    &unidecode_char_map,
                                    orcid_enabled,
                                ),
                            ))
                        })
                        .collect::<Vec<_>>()
                };
                install_with_optional_rayon_pool(num_threads, compute)
            });
        let mut features_by_signature_id = HashMap::with_capacity(raw_feature_results.len());
        for result in raw_feature_results {
            let (signature_id, feature) = result.map_err(pyo3::exceptions::PyKeyError::new_err)?;
            features_by_signature_id.insert(signature_id, feature);
        }
        let feature_secs = feature_start.elapsed().as_secs_f64();

        let summary_start = Instant::now();
        let retriever = build_retriever_from_raw_arrow_components(
            py,
            &component_order,
            &members_by_component,
            &features_by_signature_id,
            max_exemplars,
            num_threads,
        )?;
        let summary_secs = summary_start.elapsed().as_secs_f64();

        let component_members_start = Instant::now();
        let (_component_member_indices, flat_seed_signature_ids, _seed_component_keys) =
            raw_arrow_component_member_indices_for_batch(
                &component_order,
                &members_by_component,
                0,
            )?;
        let component_members_secs = component_members_start.elapsed().as_secs_f64();

        let indexed_arrow_candidate_plan = paths.signatures_batch_index_path.is_some()
            || paths.papers_batch_index_path.is_some()
            || paths.paper_authors_batch_index_path.is_some()
            || paths.specter_batch_index_path.is_some();

        Ok(Self {
            paths,
            state: ReusableRawArrowCandidatePlanState {
                features_by_signature_id,
                signatures,
                paper_authors,
                raw_name_counts,
                members_by_component,
                component_keys_by_member,
                summary_signals_by_component: HashMap::new(),
                retriever,
                component_order,
                seed_signature_ids: flat_seed_signature_ids,
                seed_signature_id_set,
                cluster_seed_disallows,
                unidecode_char_map,
                name_prefixes,
                affiliation_stopwords,
                seed_paper_count: needed_paper_ids.len(),
                seed_specter_count: specter_by_paper_id.as_ref().map_or(0usize, HashMap::len),
                build_telemetry: RawArrowPlannerBuildTelemetry {
                    read_cluster_seeds_secs,
                    read_signatures_secs,
                    read_papers_secs,
                    read_paper_authors_secs,
                    read_specter_secs,
                    read_name_counts_secs,
                    metadata_reads_parallel_secs: metadata_reads_parallel_start
                        .elapsed()
                        .as_secs_f64(),
                    text_context_secs,
                    feature_secs,
                    summary_secs,
                    component_members_secs,
                    signature_index_stats,
                    paper_index_stats,
                    paper_author_index_stats,
                    specter_index_stats,
                    indexed_arrow_candidate_plan,
                },
            },
            planner_query_signature_ids: query_signature_ids.clone(),
            planner_query_signature_count: query_signature_ids.len(),
            planner_query_signature_id_set,
            planner_query_requests_by_signature_id: HashMap::new(),
            query_mode: RawPlannerQueryMode::Declared,
            top_k,
            orcid_enabled,
            num_threads,
            max_exemplars,
        })
    }
}

#[pymethods]
impl RawBlockQueryCandidatePlanner {
    #[staticmethod]
    #[pyo3(signature = (
        paths,
        top_k,
        orcid_enabled = true,
        num_threads = None,
        max_exemplars = 4
    ))]
    fn from_query_signatures(
        py: Python<'_>,
        paths: &Bound<'_, PyDict>,
        top_k: usize,
        orcid_enabled: bool,
        num_threads: Option<usize>,
        max_exemplars: usize,
    ) -> PyResult<Self> {
        let query_signatures_path = required_path_from_py_dict(paths, "query_signatures")?;
        let query_requests = read_raw_arrow_query_signatures(&query_signatures_path)?;
        let query_signature_ids = query_requests
            .iter()
            .map(|request| request.signature_id.clone())
            .collect::<Vec<_>>();
        validate_raw_arrow_query_signature_ids_allow_empty(&query_signature_ids)?;
        let mut planner = Self::build_from_query_signature_ids(
            py,
            paths,
            query_signature_ids.clone(),
            top_k,
            orcid_enabled,
            num_threads,
            max_exemplars,
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
        paths,
        top_k,
        orcid_enabled = true,
        num_threads = None,
        max_exemplars = 4
    ))]
    fn from_auto_queries(
        py: Python<'_>,
        paths: &Bound<'_, PyDict>,
        top_k: usize,
        orcid_enabled: bool,
        num_threads: Option<usize>,
        max_exemplars: usize,
    ) -> PyResult<Self> {
        let mut planner = Self::build_from_query_signature_ids(
            py,
            paths,
            Vec::new(),
            top_k,
            orcid_enabled,
            num_threads,
            max_exemplars,
        )?;
        planner.query_mode = RawPlannerQueryMode::Auto;
        Ok(planner)
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
        payload.set_item("planner_seed_state", 1)?;
        payload.set_item("timings", timings)?;
        Ok(payload.unbind())
    }

    fn plan_query_signatures(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        if matches!(self.query_mode, RawPlannerQueryMode::Auto) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "RawBlockQueryCandidatePlanner.from_auto_queries requires explicit plan(query_signature_ids)",
            ));
        }
        self.plan(py, self.planner_query_signature_ids.clone())
    }

    #[pyo3(signature = (query_signature_ids))]
    fn plan(&mut self, py: Python<'_>, query_signature_ids: Vec<String>) -> PyResult<Py<PyDict>> {
        validate_raw_arrow_query_signature_ids(&query_signature_ids)?;
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
        let query_inputs = read_reusable_raw_arrow_query_inputs(
            py,
            &self.paths,
            &query_signature_ids,
            self.num_threads,
        )?;

        let text_context_start = Instant::now();
        ensure_unidecode_for_raw_arrow_inputs(
            &query_inputs.signatures,
            &query_inputs.papers,
            &query_inputs.paper_authors,
            &mut self.state.unidecode_char_map,
        )?;
        let text_context_secs = text_context_start.elapsed().as_secs_f64();

        let feature_start = Instant::now();
        let raw_feature_results: Vec<Result<(String, RawArrowFeature), String>> =
            py.allow_threads(|| {
                let compute = || {
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
                            let authors = query_inputs.paper_authors.get(&signature.paper_id);
                            Ok((
                                signature_id.clone(),
                                build_raw_arrow_feature(
                                    signature,
                                    paper,
                                    authors,
                                    query_inputs.specter_by_paper_id.as_ref(),
                                    &self.state.raw_name_counts,
                                    &self.state.name_prefixes,
                                    &self.state.affiliation_stopwords,
                                    &self.state.unidecode_char_map,
                                    self.orcid_enabled,
                                ),
                            ))
                        })
                        .collect::<Vec<_>>()
                };
                install_with_optional_rayon_pool(self.num_threads, compute)
            });
        let mut query_features_by_signature_id = HashMap::with_capacity(raw_feature_results.len());
        for result in raw_feature_results {
            let (signature_id, feature) = result.map_err(pyo3::exceptions::PyKeyError::new_err)?;
            query_features_by_signature_id.insert(signature_id, feature);
        }
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
        let mut filtered_members_by_component = HashMap::<String, Vec<String>>::new();
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

        let (excluded_candidate_indices_by_query, cluster_seed_disallowed_candidate_count) =
            raw_arrow_excluded_candidate_indices_by_query(
                &query_signature_ids,
                component_order,
                &self.state.component_keys_by_member,
                &self.state.cluster_seed_disallows,
            )?;

        let retrieval_start = Instant::now();
        let query_results: Vec<Result<RetrievalPairPlanQueryResult, RetrievalError>> = py
            .allow_threads(|| {
                let compute = || {
                    queries
                        .par_iter()
                        .enumerate()
                        .map(|(query_offset, current_query)| {
                            let query_index = u32::try_from(query_offset).map_err(|_| {
                                RetrievalError::Overflow("query index exceeds u32".to_string())
                            })?;
                            let excluded_candidate_indices = excluded_candidate_indices_by_query
                                .as_ref()
                                .and_then(|values| values[query_offset].as_ref());
                            retriever.build_pair_plan_query_result(
                                current_query,
                                query_first_tokens[query_offset].as_str(),
                                query_index,
                                None,
                                excluded_candidate_indices,
                                Some(query_signature_ids[query_offset].as_str()),
                                &component_member_indices,
                                self.top_k,
                                None,
                                0,
                                true,
                            )
                        })
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
            HashMap::<String, RawArrowSummarySignalData>::new();
        let mut author_signals_by_query_signature_id =
            HashMap::<String, RawArrowAuthorSignalData>::new();
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
            for (local_row_index, component_key) in
                query_result.row_component_keys.iter().enumerate()
            {
                let query_offset =
                    query_result.row_query_signature_indices[local_row_index] as usize;
                let query_signature_id =
                    query_signature_ids.get(query_offset).ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err(format!(
                            "row query signature offset {} is outside query_signature_ids",
                            query_offset
                        ))
                    })?;
                let query_feature = query_features_by_signature_id
                    .get(query_signature_id)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "query signature_id '{}' is missing from computed raw Arrow features",
                            query_signature_id
                        ))
                    })?;
                let query = queries.get(query_offset).ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err(format!(
                        "row query signature offset {} is outside query feature table",
                        query_offset
                    ))
                })?;
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
                let summary_signals = if needs_filtered_retriever {
                    raw_arrow_summary_signals_cached(
                        &mut filtered_summary_signals_by_component,
                        component_key,
                        members_by_component,
                        &self.state.features_by_signature_id,
                        &self.state.signatures,
                        &self.state.paper_authors,
                        &self.state.unidecode_char_map,
                    )?
                } else {
                    raw_arrow_summary_signals_cached(
                        &mut self.state.summary_signals_by_component,
                        component_key,
                        members_by_component,
                        &self.state.features_by_signature_id,
                        &self.state.signatures,
                        &self.state.paper_authors,
                        &self.state.unidecode_char_map,
                    )?
                };
                let rarity = raw_arrow_name_count_rarity_row(
                    query,
                    &query_feature.name_counts,
                    summary,
                    summary_signals,
                );
                if let Entry::Vacant(entry) =
                    author_signals_by_query_signature_id.entry(query_signature_id.clone())
                {
                    let query_signature = query_inputs
                        .signatures
                        .get(query_signature_id)
                        .ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(format!(
                                "query signature_id '{}' is missing from signatures",
                                query_signature_id
                            ))
                        })?;
                    let author_signals = build_raw_arrow_author_signal_data(
                        query_signature,
                        query_inputs.paper_authors.get(&query_signature.paper_id),
                        &self.state.unidecode_char_map,
                    );
                    entry.insert(author_signals);
                }
                let query_author_signals = author_signals_by_query_signature_id
                    .get(query_signature_id)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "query signature_id '{}' is missing raw author signals",
                            query_signature_id
                        ))
                    })?;
                let evidence = raw_arrow_paper_evidence_row(
                    query_signature_id,
                    query_feature.paper_author_count,
                    query_author_signals,
                    summary_signals,
                );
                row_last_name_count_min_rarity.push(rarity.last_name_count_min_rarity);
                row_candidate_last_name_count_min_rarity
                    .push(rarity.candidate_last_name_count_min_rarity);
                row_candidate_last_first_name_count_min_rarity
                    .push(rarity.candidate_last_first_name_count_min_rarity);
                row_last_first_name_count_min_rarity.push(rarity.last_first_name_count_min_rarity);
                row_first_prefix_x_last_first_name_count_min_rarity
                    .push(rarity.first_prefix_x_last_first_name_count_min_rarity);
                row_candidate_cluster_max_paper_author_count
                    .push(summary.max_paper_author_count as f32);
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
        let mut left_signature_ids = Vec::<String>::with_capacity(left_signature_indices.len());
        let mut right_signature_ids = Vec::<String>::with_capacity(right_signature_indices.len());
        let signature_index_count = query_signature_ids.len() + seed_signature_ids.len();
        let signature_id_for_index = |index: u32| -> Option<&String> {
            let offset = index as usize;
            if offset < query_signature_ids.len() {
                query_signature_ids.get(offset)
            } else {
                seed_signature_ids.get(offset - query_signature_ids.len())
            }
        };
        for index in left_signature_indices.iter() {
            let Some(signature_id) = signature_id_for_index(*index) else {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "left signature index {} is outside signature id table of length {}",
                    index, signature_index_count
                )));
            };
            left_signature_ids.push(signature_id.clone());
        }
        for index in right_signature_indices.iter() {
            let Some(signature_id) = signature_id_for_index(*index) else {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "right signature index {} is outside signature id table of length {}",
                    index, signature_index_count
                )));
            };
            right_signature_ids.push(signature_id.clone());
        }
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
        let mut telemetry_paper_ids = HashSet::<String>::new();
        for signature_id in query_signature_ids.iter() {
            if let Some(signature) = query_inputs.signatures.get(signature_id) {
                telemetry_paper_ids.insert(signature.paper_id.clone());
            }
        }
        for signature_id in seed_signature_ids.iter() {
            if let Some(signature) = self.state.signatures.get(signature_id) {
                telemetry_paper_ids.insert(signature.paper_id.clone());
            }
        }
        telemetry.set_item("paper_count", telemetry_paper_ids.len())?;
        let mut telemetry_paper_author_ids = HashSet::<String>::new();
        for paper_id in telemetry_paper_ids.iter() {
            if self.state.paper_authors.contains_key(paper_id)
                || query_inputs.paper_authors.contains_key(paper_id)
            {
                telemetry_paper_author_ids.insert(paper_id.clone());
            }
        }
        telemetry.set_item("paper_author_paper_count", telemetry_paper_author_ids.len())?;
        telemetry.set_item("cluster_count", component_order.len())?;
        telemetry.set_item("seed_signature_count", seed_signature_ids.len())?;
        telemetry.set_item("query_signature_count", query_signature_ids.len())?;
        telemetry.set_item("excluded_query_seed_count", excluded_query_seed_count)?;
        telemetry.set_item(
            "cluster_seed_disallow_pair_count",
            self.state.cluster_seed_disallows.len(),
        )?;
        telemetry.set_item(
            "cluster_seed_disallowed_candidate_count",
            cluster_seed_disallowed_candidate_count,
        )?;
        let mut telemetry_specter_ids = HashSet::<String>::new();
        for signature_id in seed_signature_ids.iter() {
            if self
                .state
                .features_by_signature_id
                .get(signature_id)
                .and_then(|feature| feature.query.specter.as_ref())
                .is_some()
            {
                if let Some(signature) = self.state.signatures.get(signature_id) {
                    telemetry_specter_ids.insert(signature.paper_id.clone());
                }
            }
        }
        if let Some(query_specter) = query_inputs.specter_by_paper_id.as_ref() {
            for paper_id in telemetry_paper_ids.iter() {
                if query_specter.contains_key(paper_id) {
                    telemetry_specter_ids.insert(paper_id.clone());
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
        payload.set_item("schema_version", "raw_arrow_candidate_plan_v2")?;
        payload.set_item("row_count", row_component_keys.len())?;
        payload.set_item("pair_count", left_signature_indices.len())?;
        payload.set_item("query_signature_ids", query_signature_ids)?;
        payload.set_item("query_views", query_views)?;
        payload.set_item("query_authors", query_authors)?;
        payload.set_item("seed_signature_ids", Vec::<String>::new())?;
        payload.set_item("component_members", component_members)?;
        payload.set_item("left_signature_ids", left_signature_ids)?;
        payload.set_item("right_signature_ids", right_signature_ids)?;
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
        payload.set_item(
            "row_last_name_count_min_rarity",
            row_last_name_count_min_rarity.to_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_last_name_count_min_rarity",
            row_candidate_last_name_count_min_rarity.to_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_last_first_name_count_min_rarity",
            row_candidate_last_first_name_count_min_rarity.to_pyarray(py),
        )?;
        payload.set_item(
            "row_last_first_name_count_min_rarity",
            row_last_first_name_count_min_rarity.to_pyarray(py),
        )?;
        payload.set_item(
            "row_first_prefix_x_last_first_name_count_min_rarity",
            row_first_prefix_x_last_first_name_count_min_rarity.to_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_cluster_max_paper_author_count",
            row_candidate_cluster_max_paper_author_count.to_pyarray(py),
        )?;
        payload.set_item(
            "row_paper_author_list_max_jaccard",
            row_paper_author_list_max_jaccard.to_pyarray(py),
        )?;
        payload.set_item(
            "row_paper_author_list_max_containment",
            row_paper_author_list_max_containment.to_pyarray(py),
        )?;
        payload.set_item(
            "row_paper_author_list_max_overlap_count",
            row_paper_author_list_max_overlap_count.to_pyarray(py),
        )?;
        payload.set_item(
            "row_local_author_window10_jaccard_max",
            row_local_author_window10_jaccard_max.to_pyarray(py),
        )?;
        payload.set_item(
            "row_local_author_window10_overlap_count_max",
            row_local_author_window10_overlap_count_max.to_pyarray(py),
        )?;
        payload.set_item(
            "row_best_author_count_log_absdiff",
            row_best_author_count_log_absdiff.to_pyarray(py),
        )?;
        payload.set_item("telemetry", telemetry)?;
        timings.set_item("payload_secs", payload_start.elapsed().as_secs_f64())?;
        timings.set_item("total_secs", total_start.elapsed().as_secs_f64())?;
        timings.set_item("drop_secs", 0.0)?;
        timings.set_item("wall_secs", total_start.elapsed().as_secs_f64())?;
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
    payload.set_item("schema_version", "raw_arrow_labeled_candidate_plan_v1")?;
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
    payload.set_item("pair_row_indices", Vec::<u32>::new().to_pyarray(py))?;
    payload.set_item("retrieval_scores", Vec::<f32>::new().to_pyarray(py))?;
    payload.set_item("retrieval_ranks", Vec::<u16>::new().to_pyarray(py))?;
    payload.set_item("row_component_sizes", Vec::<u32>::new().to_pyarray(py))?;
    payload.set_item(
        "row_named_signature_counts",
        Vec::<u32>::new().to_pyarray(py),
    )?;
    payload.set_item("row_dominant_first_names", Vec::<String>::new())?;
    payload.set_item("row_candidate_year_min", Vec::<i32>::new().to_pyarray(py))?;
    payload.set_item("row_candidate_year_max", Vec::<i32>::new().to_pyarray(py))?;
    payload.set_item(
        "row_candidate_year_range_missing",
        Vec::<u8>::new().to_pyarray(py),
    )?;
    payload.set_item("row_query_first_tokens", Vec::<String>::new())?;
    payload.set_item("row_query_years", Vec::<i32>::new().to_pyarray(py))?;
    payload.set_item("row_query_year_missing", Vec::<u8>::new().to_pyarray(py))?;
    payload.set_item(
        "row_query_has_affiliations",
        Vec::<u8>::new().to_pyarray(py),
    )?;
    payload.set_item("row_query_has_coauthors", Vec::<u8>::new().to_pyarray(py))?;
    payload.set_item("row_query_has_specter", Vec::<u8>::new().to_pyarray(py))?;
    payload.set_item("row_query_has_name_counts", Vec::<u8>::new().to_pyarray(py))?;
    payload.set_item(
        "row_candidate_has_affiliations",
        Vec::<u8>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_coauthors",
        Vec::<u8>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_specter_exemplars",
        Vec::<u8>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_name_counts",
        Vec::<u8>::new().to_pyarray(py),
    )?;
    payload.set_item("row_orcid_match", Vec::<u8>::new().to_pyarray(py))?;
    payload.set_item(
        "middle_initial_compatibility",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item("affiliation_overlap", Vec::<f32>::new().to_pyarray(py))?;
    payload.set_item("coauthor_overlap", Vec::<f32>::new().to_pyarray(py))?;
    payload.set_item("venue_overlap", Vec::<f32>::new().to_pyarray(py))?;
    payload.set_item("year_compatibility", Vec::<f32>::new().to_pyarray(py))?;
    payload.set_item("title_overlap", Vec::<f32>::new().to_pyarray(py))?;
    payload.set_item(
        "specter_centroid_similarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "specter_exemplar_similarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_last_name_count_min_rarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_last_name_count_min_rarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_last_first_name_count_min_rarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_last_first_name_count_min_rarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_first_prefix_x_last_first_name_count_min_rarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_cluster_max_paper_author_count",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_paper_author_list_max_jaccard",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_paper_author_list_max_containment",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_paper_author_list_max_overlap_count",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_local_author_window10_jaccard_max",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_local_author_window10_overlap_count_max",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_best_author_count_log_absdiff",
        Vec::<f32>::new().to_pyarray(py),
    )?;
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
    let filtered = raw_members
        .iter()
        .filter(|signature_id| {
            signatures
                .get(signature_id.as_str())
                .and_then(|signature| signature.author_block.as_deref())
                .is_some_and(|author_block| author_block == block_key)
        })
        .cloned()
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        raw_members.to_vec()
    } else {
        filtered
    }
}

fn raw_arrow_active_members_for_row(
    component_key: &str,
    query_signature_id: &str,
    members_by_component: &HashMap<String, Vec<String>>,
) -> PyResult<Vec<String>> {
    let members = members_by_component.get(component_key).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "candidate_component_key missing from members table: {component_key}"
        ))
    })?;
    Ok(members
        .iter()
        .filter(|signature_id| signature_id.as_str() != query_signature_id)
        .cloned()
        .collect())
}

fn raw_arrow_component_summary_for_members(
    component_key: &str,
    members: &[String],
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
    paths,
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
    paths: &Bound<'py, PyDict>,
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

    let paths = raw_arrow_feature_paths_from_py_dict(paths)?;
    let mut query_signature_ids = Vec::<String>::new();
    let mut query_seen = HashSet::<String>::new();
    for signature_id in row_query_signature_ids.iter() {
        if query_seen.insert(signature_id.clone()) {
            query_signature_ids.push(signature_id.clone());
        }
    }
    validate_raw_arrow_query_signature_ids(&query_signature_ids)?;

    let component_entries = extract_string_vec_entries(component_members)?;
    let row_component_key_set: HashSet<String> = row_component_keys.iter().cloned().collect();
    let mut raw_members_by_component = HashMap::<String, Vec<String>>::new();
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
    let mut needed_seen = HashSet::<String>::new();
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
        read_reusable_raw_arrow_query_inputs(py, &paths, &needed_signature_ids, num_threads)?;
    let raw_name_counts = match paths.name_counts_index_path.as_ref() {
        Some(path) => read_raw_name_counts_index(path)?,
        None => RawNameCountMaps::default(),
    };

    let text_context_start = Instant::now();
    let text_module = py.import("s2and.text")?;
    let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
    let affiliation_stopwords = extract_affiliation_stopwords(py)?;
    let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
    ensure_unidecode_for_raw_arrow_inputs(
        &query_inputs.signatures,
        &query_inputs.papers,
        &query_inputs.paper_authors,
        &mut unidecode_char_map,
    )?;
    let text_context_secs = text_context_start.elapsed().as_secs_f64();

    let specter_by_paper_id = query_inputs.specter_by_paper_id.as_ref();
    let feature_start = Instant::now();
    let raw_feature_results: Vec<Result<(String, RawArrowFeature), String>> =
        py.allow_threads(|| {
            let compute = || {
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
                        let authors = query_inputs.paper_authors.get(&signature.paper_id);
                        Ok((
                            signature_id.clone(),
                            build_raw_arrow_feature(
                                signature,
                                paper,
                                authors,
                                specter_by_paper_id,
                                &raw_name_counts,
                                &name_prefixes,
                                &affiliation_stopwords,
                                &unidecode_char_map,
                                orcid_enabled,
                            ),
                        ))
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
    let mut features_by_signature_id = HashMap::with_capacity(raw_feature_results.len());
    for result in raw_feature_results {
        let (signature_id, feature) = result.map_err(pyo3::exceptions::PyKeyError::new_err)?;
        features_by_signature_id.insert(signature_id, feature);
    }
    let feature_secs = feature_start.elapsed().as_secs_f64();

    let mut members_by_component = HashMap::<String, Vec<String>>::new();
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
    )?;
    let summary_secs = summary_start.elapsed().as_secs_f64();

    let mut group_order = Vec::<String>::new();
    let mut group_seen = HashSet::<String>::new();
    let mut rows_by_group = HashMap::<String, Vec<usize>>::new();
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
            let active_members = raw_arrow_active_members_for_row(
                component_key,
                query_signature_id,
                &members_by_component,
            )?;
            let summary = if active_members.len()
                == members_by_component
                    .get(component_key)
                    .map_or(usize::MAX, Vec::len)
            {
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
    let mut signature_seen = HashSet::<String>::new();
    for signature_id in row_query_signature_ids.iter() {
        if signature_seen.insert(signature_id.clone()) {
            signature_ids.push(signature_id.clone());
        }
    }
    let mut left_signature_ids = Vec::<String>::new();
    let mut right_signature_ids = Vec::<String>::new();
    let mut pair_row_indices = Vec::<u32>::new();
    for (row_index, (query_signature_id, component_key)) in row_query_signature_ids
        .iter()
        .zip(row_component_keys.iter())
        .enumerate()
    {
        let active_members = raw_arrow_active_members_for_row(
            component_key,
            query_signature_id,
            &members_by_component,
        )?;
        for member_signature_id in active_members {
            if signature_seen.insert(member_signature_id.clone()) {
                signature_ids.push(member_signature_id.clone());
            }
            left_signature_ids.push(query_signature_id.clone());
            right_signature_ids.push(member_signature_id);
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
    let mut full_summary_signal_cache = HashMap::<String, RawArrowSummarySignalData>::new();
    let mut residual_summary_signal_cache =
        HashMap::<(String, String), RawArrowSummarySignalData>::new();
    let mut author_signals_by_query_signature_id =
        HashMap::<String, RawArrowAuthorSignalData>::new();
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
        let active_members = raw_arrow_active_members_for_row(
            component_key,
            query_signature_id,
            &members_by_component,
        )?;
        let full_member_count = members_by_component
            .get(component_key)
            .map_or(usize::MAX, Vec::len);
        let (summary, summary_signals) = if active_members.len() == full_member_count {
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
                &query_inputs.paper_authors,
                &unidecode_char_map,
            )?;
            (summary, signals)
        } else {
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
                &query_inputs.paper_authors,
                &unidecode_char_map,
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
            entry.insert(build_raw_arrow_author_signal_data(
                query_signature,
                query_inputs.paper_authors.get(&query_signature.paper_id),
                &unidecode_char_map,
            ));
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

    let mut query_view_by_signature_id = HashMap::<String, String>::new();
    let mut query_author_by_signature_id = HashMap::<String, String>::new();
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

    let telemetry = PyDict::new(py);
    telemetry.set_item("signature_count", signature_ids.len())?;
    telemetry.set_item("paper_count", query_inputs.papers.len())?;
    telemetry.set_item("paper_author_paper_count", query_inputs.paper_authors.len())?;
    telemetry.set_item("component_count", component_order.len())?;
    telemetry.set_item("query_signature_count", query_signature_ids.len())?;
    telemetry.set_item("row_count", row_count)?;
    telemetry.set_item("pair_count", left_signature_ids.len())?;
    telemetry.set_item("component_scope", "block-local")?;
    telemetry.set_item("orcid_enabled", orcid_enabled)?;
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
    payload.set_item("schema_version", "raw_arrow_labeled_candidate_plan_v1")?;
    payload.set_item("row_count", row_count)?;
    payload.set_item("pair_count", left_signature_ids.len())?;
    payload.set_item("signature_ids", signature_ids)?;
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
    payload.set_item("pair_row_indices", pair_row_indices.to_pyarray(py))?;
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
        "row_query_has_specter",
        row_query_has_specter.to_pyarray(py),
    )?;
    payload.set_item(
        "row_query_has_name_counts",
        row_query_has_name_counts.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_affiliations",
        row_candidate_has_affiliations.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_coauthors",
        row_candidate_has_coauthors.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_specter_exemplars",
        row_candidate_has_specter_exemplars.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_name_counts",
        row_candidate_has_name_counts.to_pyarray(py),
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
    payload.set_item(
        "row_last_name_count_min_rarity",
        row_last_name_count_min_rarity.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_last_name_count_min_rarity",
        row_candidate_last_name_count_min_rarity.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_last_first_name_count_min_rarity",
        row_candidate_last_first_name_count_min_rarity.to_pyarray(py),
    )?;
    payload.set_item(
        "row_last_first_name_count_min_rarity",
        row_last_first_name_count_min_rarity.to_pyarray(py),
    )?;
    payload.set_item(
        "row_first_prefix_x_last_first_name_count_min_rarity",
        row_first_prefix_x_last_first_name_count_min_rarity.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_cluster_max_paper_author_count",
        row_candidate_cluster_max_paper_author_count.to_pyarray(py),
    )?;
    payload.set_item(
        "row_paper_author_list_max_jaccard",
        row_paper_author_list_max_jaccard.to_pyarray(py),
    )?;
    payload.set_item(
        "row_paper_author_list_max_containment",
        row_paper_author_list_max_containment.to_pyarray(py),
    )?;
    payload.set_item(
        "row_paper_author_list_max_overlap_count",
        row_paper_author_list_max_overlap_count.to_pyarray(py),
    )?;
    payload.set_item(
        "row_local_author_window10_jaccard_max",
        row_local_author_window10_jaccard_max.to_pyarray(py),
    )?;
    payload.set_item(
        "row_local_author_window10_overlap_count_max",
        row_local_author_window10_overlap_count_max.to_pyarray(py),
    )?;
    payload.set_item(
        "row_best_author_count_log_absdiff",
        row_best_author_count_log_absdiff.to_pyarray(py),
    )?;
    payload.set_item("telemetry", telemetry)?;
    Ok(payload.unbind())
}
