use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3::Bound;
use std::collections::{hash_map::Entry, BTreeMap, HashMap, HashSet};
use std::time::Instant;

use crate::arrow_batch_lookup::IndexedArrowReadStats;
use crate::constraints::same_prefix_tokens;
use crate::features::word_ngrams_counter_python_compat;
use crate::orcid::normalize_orcid_owned;
use crate::raw_arrow::arrow_io::{
    arrow_column_index, arrow_optional_string_list, ArrowI64Column, ArrowStringColumn,
};
use crate::raw_arrow::paths::extract_path_mapping_string;
use crate::raw_arrow::readers::{
    read_raw_arrow_paper_authors_with_optional_index, read_raw_arrow_specter_with_optional_index,
    read_raw_arrow_with_optional_index,
};
use crate::text_compat::{
    canonicalize_name_parts_compat, compute_block_compat, ensure_unidecode_for_text,
    normalize_text_compat_from_map,
};
use crate::{
    extract_affiliation_stopwords, extract_required_string_set, fnv64_update, py_len, FNV_OFFSET,
};

pub(crate) fn subblock_tokens_from_key(subblock_key: &str) -> Vec<String> {
    let mut values = HashSet::new();
    for raw_token in subblock_key.split(',') {
        let token = raw_token
            .trim()
            .split_once('|')
            .map_or(raw_token.trim(), |(token, _rest)| token.trim());
        if py_len(&token) > 1 {
            values.insert(token.to_string());
        }
    }
    let mut out: Vec<String> = values.into_iter().collect();
    out.sort_unstable();
    out
}

#[derive(Clone)]
pub(crate) struct SubblockingSignatureRow {
    pub(crate) signature_id: String,
    pub(crate) paper_id: String,
    pub(crate) first: String,
    pub(crate) middle: String,
    pub(crate) affiliations: Vec<String>,
    pub(crate) orcid: Option<String>,
    pub(crate) position: Option<i64>,
}

pub(crate) fn normalize_subblocking_signature_rows(
    rows: &mut [SubblockingSignatureRow],
    name_prefixes: &HashSet<String>,
    unidecode_char_map: &HashMap<char, String>,
) {
    for row in rows.iter_mut() {
        let (first, middle, _last) = canonicalize_name_parts_compat(
            &row.first,
            &row.middle,
            "",
            name_prefixes,
            Some(unidecode_char_map),
        );
        row.first = first;
        row.middle = middle;
    }
}

pub(crate) fn read_subblocking_signature_rows_from_batches(
    path: &str,
    batches: Vec<RecordBatch>,
    keep_signature_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, SubblockingSignatureRow>> {
    let mut out = HashMap::new();
    for batch in batches {
        let signature_id_col = batch.column(arrow_column_index(&batch, "signature_id", path)?);
        let signature_id_values =
            ArrowStringColumn::from_string_array(signature_id_col.as_ref(), "signature_id")?;
        let paper_id_col = batch.column(arrow_column_index(&batch, "paper_id", path)?);
        let paper_id_values =
            ArrowStringColumn::from_string_array(paper_id_col.as_ref(), "paper_id")?;
        let first_col = batch.column(arrow_column_index(&batch, "author_first", path)?);
        let first_values =
            ArrowStringColumn::from_string_array(first_col.as_ref(), "author_first")?;
        let middle_col = batch.column(arrow_column_index(&batch, "author_middle", path)?);
        let middle_values =
            ArrowStringColumn::from_string_array(middle_col.as_ref(), "author_middle")?;
        let affiliations_col =
            batch.column(arrow_column_index(&batch, "author_affiliations", path)?);
        let orcid_col = batch.column(arrow_column_index(&batch, "author_orcid", path)?);
        let orcid_values =
            ArrowStringColumn::from_string_array(orcid_col.as_ref(), "author_orcid")?;
        let position_col = batch.column(arrow_column_index(&batch, "author_position", path)?);
        let position_values =
            ArrowI64Column::from_i64_array(position_col.as_ref(), "author_position")?;
        for row in 0..batch.num_rows() {
            let signature_id_value = signature_id_values.required_value(row, "signature_id")?;
            if keep_signature_ids.map_or(false, |keep| !keep.contains(signature_id_value.as_ref()))
            {
                continue;
            }
            if signature_id_value.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "signatures Arrow cannot contain empty signature_id values",
                ));
            }
            let signature_id = signature_id_value.into_owned();
            match out.entry(signature_id.clone()) {
                Entry::Occupied(entry) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "signatures Arrow contains duplicate signature_id: {:?}",
                        entry.key()
                    )));
                }
                Entry::Vacant(entry) => {
                    entry.insert(SubblockingSignatureRow {
                        signature_id,
                        paper_id: paper_id_values
                            .required_value(row, "paper_id")?
                            .into_owned(),
                        first: first_values.optional_owned(row).unwrap_or_default(),
                        middle: middle_values.optional_owned(row).unwrap_or_default(),
                        affiliations: arrow_optional_string_list(
                            affiliations_col.as_ref(),
                            row,
                            "author_affiliations",
                        )?,
                        orcid: orcid_values
                            .optional_owned(row)
                            .and_then(|value| normalize_orcid_owned(&value)),
                        position: position_values.optional_value(row, "author_position")?,
                    });
                }
            }
        }
    }
    Ok(out)
}

pub(crate) fn read_subblocking_signature_rows_with_optional_index(
    path: &str,
    index_path: Option<&str>,
    keep_signature_ids: Option<&HashSet<String>>,
) -> PyResult<(
    HashMap<String, SubblockingSignatureRow>,
    IndexedArrowReadStats,
)> {
    read_raw_arrow_with_optional_index(
        path,
        index_path,
        "signature_id",
        keep_signature_ids,
        read_subblocking_signature_rows_from_batches,
    )
}

#[derive(Clone)]
pub(crate) struct NativeGraphSubblockingConfig {
    pub(crate) neighbor_mode: String,
    pub(crate) neighbors: usize,
    pub(crate) min_edge_score: f64,
    pub(crate) specter_weight: f64,
    pub(crate) coauthor_weight: f64,
    pub(crate) affiliation_weight: f64,
    pub(crate) max_exact_knn_group_size: usize,
    pub(crate) projection_count: usize,
    pub(crate) projection_window: usize,
    pub(crate) max_candidate_edges: usize,
    pub(crate) pack_components: bool,
    pub(crate) component_pack_strategy: String,
    pub(crate) sparse_evidence_edges: bool,
    pub(crate) sparse_evidence_max_posting_size: usize,
    pub(crate) sparse_evidence_neighbors: usize,
    pub(crate) sparse_evidence_min_weight: f64,
    pub(crate) sparse_evidence_include_coauthors: bool,
    pub(crate) sparse_evidence_include_affiliations: bool,
    pub(crate) component_pack_top_k: usize,
    pub(crate) local_move_passes: usize,
    pub(crate) adaptive_projection: bool,
    pub(crate) adaptive_projection_max_group_size: usize,
    pub(crate) adaptive_projection_count: usize,
    pub(crate) adaptive_projection_window: usize,
}

impl Default for NativeGraphSubblockingConfig {
    fn default() -> Self {
        Self {
            neighbor_mode: "projection".to_string(),
            neighbors: 16,
            min_edge_score: 0.30,
            specter_weight: 1.0,
            coauthor_weight: 0.35,
            affiliation_weight: 0.20,
            max_exact_knn_group_size: 25_000,
            projection_count: 12,
            projection_window: 12,
            max_candidate_edges: 5_000_000,
            pack_components: true,
            component_pack_strategy: "edge-greedy".to_string(),
            sparse_evidence_edges: true,
            sparse_evidence_max_posting_size: 8,
            sparse_evidence_neighbors: 1,
            sparse_evidence_min_weight: 0.40,
            sparse_evidence_include_coauthors: true,
            sparse_evidence_include_affiliations: false,
            component_pack_top_k: 8,
            local_move_passes: 0,
            adaptive_projection: false,
            adaptive_projection_max_group_size: 5_000,
            adaptive_projection_count: 24,
            adaptive_projection_window: 24,
        }
    }
}

pub(crate) fn graph_config_get_value<'py>(
    config_obj: Option<&Bound<'py, PyAny>>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some(config) = config_obj else {
        return Ok(None);
    };
    if config.is_none() {
        return Ok(None);
    }
    if let Ok(dict) = config.downcast::<PyDict>() {
        return dict.get_item(key);
    }
    match config.getattr(key) {
        Ok(value) => Ok(Some(value)),
        Err(_) => Ok(None),
    }
}

pub(crate) fn graph_config_get_string(
    config_obj: Option<&Bound<'_, PyAny>>,
    key: &str,
    default_value: &str,
) -> PyResult<String> {
    Ok(match graph_config_get_value(config_obj, key)? {
        Some(value) => value.extract()?,
        None => default_value.to_string(),
    })
}

pub(crate) fn graph_config_get_usize(
    config_obj: Option<&Bound<'_, PyAny>>,
    key: &str,
    default_value: usize,
) -> PyResult<usize> {
    Ok(match graph_config_get_value(config_obj, key)? {
        Some(value) => value.extract()?,
        None => default_value,
    })
}

pub(crate) fn graph_config_get_f64(
    config_obj: Option<&Bound<'_, PyAny>>,
    key: &str,
    default_value: f64,
) -> PyResult<f64> {
    Ok(match graph_config_get_value(config_obj, key)? {
        Some(value) => value.extract()?,
        None => default_value,
    })
}

pub(crate) fn graph_config_get_bool(
    config_obj: Option<&Bound<'_, PyAny>>,
    key: &str,
    default_value: bool,
) -> PyResult<bool> {
    Ok(match graph_config_get_value(config_obj, key)? {
        Some(value) => value.extract()?,
        None => default_value,
    })
}

impl NativeGraphSubblockingConfig {
    pub(crate) fn from_py(config_obj: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let default = Self::default();
        let config = Self {
            neighbor_mode: graph_config_get_string(
                config_obj,
                "neighbor_mode",
                &default.neighbor_mode,
            )?,
            neighbors: graph_config_get_usize(config_obj, "neighbors", default.neighbors)?,
            min_edge_score: graph_config_get_f64(
                config_obj,
                "min_edge_score",
                default.min_edge_score,
            )?,
            specter_weight: graph_config_get_f64(
                config_obj,
                "specter_weight",
                default.specter_weight,
            )?,
            coauthor_weight: graph_config_get_f64(
                config_obj,
                "coauthor_weight",
                default.coauthor_weight,
            )?,
            affiliation_weight: graph_config_get_f64(
                config_obj,
                "affiliation_weight",
                default.affiliation_weight,
            )?,
            max_exact_knn_group_size: graph_config_get_usize(
                config_obj,
                "max_exact_knn_group_size",
                default.max_exact_knn_group_size,
            )?,
            projection_count: graph_config_get_usize(
                config_obj,
                "projection_count",
                default.projection_count,
            )?,
            projection_window: graph_config_get_usize(
                config_obj,
                "projection_window",
                default.projection_window,
            )?,
            max_candidate_edges: graph_config_get_usize(
                config_obj,
                "max_candidate_edges",
                default.max_candidate_edges,
            )?,
            pack_components: graph_config_get_bool(
                config_obj,
                "pack_components",
                default.pack_components,
            )?,
            component_pack_strategy: graph_config_get_string(
                config_obj,
                "component_pack_strategy",
                &default.component_pack_strategy,
            )?,
            sparse_evidence_edges: graph_config_get_bool(
                config_obj,
                "sparse_evidence_edges",
                default.sparse_evidence_edges,
            )?,
            sparse_evidence_max_posting_size: graph_config_get_usize(
                config_obj,
                "sparse_evidence_max_posting_size",
                default.sparse_evidence_max_posting_size,
            )?,
            sparse_evidence_neighbors: graph_config_get_usize(
                config_obj,
                "sparse_evidence_neighbors",
                default.sparse_evidence_neighbors,
            )?,
            sparse_evidence_min_weight: graph_config_get_f64(
                config_obj,
                "sparse_evidence_min_weight",
                default.sparse_evidence_min_weight,
            )?,
            sparse_evidence_include_coauthors: graph_config_get_bool(
                config_obj,
                "sparse_evidence_include_coauthors",
                default.sparse_evidence_include_coauthors,
            )?,
            sparse_evidence_include_affiliations: graph_config_get_bool(
                config_obj,
                "sparse_evidence_include_affiliations",
                default.sparse_evidence_include_affiliations,
            )?,
            component_pack_top_k: graph_config_get_usize(
                config_obj,
                "component_pack_top_k",
                default.component_pack_top_k,
            )?,
            local_move_passes: graph_config_get_usize(
                config_obj,
                "local_move_passes",
                default.local_move_passes,
            )?,
            adaptive_projection: graph_config_get_bool(
                config_obj,
                "adaptive_projection",
                default.adaptive_projection,
            )?,
            adaptive_projection_max_group_size: graph_config_get_usize(
                config_obj,
                "adaptive_projection_max_group_size",
                default.adaptive_projection_max_group_size,
            )?,
            adaptive_projection_count: graph_config_get_usize(
                config_obj,
                "adaptive_projection_count",
                default.adaptive_projection_count,
            )?,
            adaptive_projection_window: graph_config_get_usize(
                config_obj,
                "adaptive_projection_window",
                default.adaptive_projection_window,
            )?,
        };
        if config.neighbor_mode != "projection" && config.neighbor_mode != "exact" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Graph subblocking neighbor_mode must be 'projection' or 'exact'",
            ));
        }
        if !matches!(
            config.component_pack_strategy.as_str(),
            "edge-greedy" | "aggregate-greedy" | "size"
        ) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Graph subblocking component_pack_strategy must be 'edge-greedy', 'aggregate-greedy', or 'size'",
            ));
        }
        if config.projection_count == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Graph subblocking projection_count must be positive",
            ));
        }
        if config.projection_window == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Graph subblocking projection_window must be positive",
            ));
        }
        if config.sparse_evidence_edges && config.sparse_evidence_max_posting_size <= 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Graph subblocking sparse_evidence_max_posting_size must be greater than 1",
            ));
        }
        Ok(config)
    }

    pub(crate) fn effective_for_group(&self, group_size: usize) -> Self {
        if !self.adaptive_projection
            || self.neighbor_mode != "projection"
            || group_size > self.adaptive_projection_max_group_size
        {
            return self.clone();
        }
        let mut out = self.clone();
        out.projection_count = out.projection_count.max(out.adaptive_projection_count);
        out.projection_window = out.projection_window.max(out.adaptive_projection_window);
        out
    }
}

pub(crate) struct NativeGraphArrowPaths {
    pub(crate) paper_authors_path: String,
    pub(crate) paper_authors_batch_index_path: String,
    pub(crate) specter_path: String,
    pub(crate) specter_batch_index_path: String,
}

impl NativeGraphArrowPaths {
    pub(crate) fn from_py(paths: &Bound<'_, PyAny>) -> PyResult<Self> {
        let paper_authors_path = extract_path_mapping_string(paths, "paper_authors", true)?
            .expect("required paper_authors path exists");
        let paper_authors_batch_index_path =
            extract_path_mapping_string(paths, "paper_authors_batch_index", true)?
                .expect("required paper_authors_batch_index path exists");
        let (specter_path, specter_batch_index_path) =
            if let Some(path) = extract_path_mapping_string(paths, "specter", false)? {
                let index_path = extract_path_mapping_string(paths, "specter_batch_index", true)?
                    .expect("required specter_batch_index path exists");
                (path, index_path)
            } else if let Some(path) = extract_path_mapping_string(paths, "specter2", false)? {
                let index_path =
                    match extract_path_mapping_string(paths, "specter2_batch_index", false)? {
                        Some(value) => value,
                        None => extract_path_mapping_string(paths, "specter_batch_index", true)?
                            .expect("required specter_batch_index path exists"),
                    };
                (path, index_path)
            } else {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    "Arrow path bundle is missing required key: specter or specter2",
                ));
            };
        Ok(Self {
            paper_authors_path,
            paper_authors_batch_index_path,
            specter_path,
            specter_batch_index_path,
        })
    }
}

#[derive(Clone)]
pub(crate) struct NativeGraphSignatureEvidence {
    pub(crate) embedding: Vec<f32>,
    pub(crate) coauthor_blocks: HashSet<String>,
    pub(crate) affiliation_keys: HashSet<String>,
}

pub(crate) struct NativeGraphEvidenceStore {
    pub(crate) signatures: HashMap<String, NativeGraphSignatureEvidence>,
    pub(crate) dimension: usize,
}

#[derive(Default)]
pub(crate) struct NativeGraphLoadMetrics {
    pub(crate) signatures_record_batches_scanned: usize,
    pub(crate) signatures_rows_scanned: usize,
    pub(crate) signatures_rows_loaded: usize,
    pub(crate) paper_authors_record_batches_scanned: usize,
    pub(crate) paper_authors_rows_scanned: usize,
    pub(crate) paper_authors_rows_loaded: usize,
    pub(crate) specter_record_batches_scanned: usize,
    pub(crate) specter_rows_scanned: usize,
    pub(crate) specter_rows_loaded: usize,
}

#[derive(Default)]
pub(crate) struct SparseEvidenceStats {
    pub(crate) sparse_evidence_feature_count: usize,
    pub(crate) sparse_evidence_skipped_feature_count: usize,
    pub(crate) sparse_evidence_added_edge_count: usize,
    pub(crate) sparse_evidence_neighbors: usize,
}

pub(crate) struct NativeGraphFallbackStats {
    pub(crate) input_signature_count: usize,
    pub(crate) neighbor_mode: String,
    pub(crate) projection_count: usize,
    pub(crate) projection_window: usize,
    pub(crate) candidate_edge_count: usize,
    pub(crate) sparse_evidence_stats: SparseEvidenceStats,
    pub(crate) raw_component_count: usize,
    pub(crate) raw_max_component_size: usize,
    pub(crate) raw_median_component_size: f64,
    pub(crate) packed_component_count: usize,
    pub(crate) max_component_size: usize,
    pub(crate) median_component_size: f64,
    pub(crate) pack_components: bool,
    pub(crate) component_pack_strategy: String,
    pub(crate) component_pack_top_k: usize,
    pub(crate) local_move_passes: usize,
    pub(crate) edge_build_seconds: f64,
    pub(crate) total_seconds: f64,
}

#[derive(Default)]
pub(crate) struct NativeGraphTelemetry {
    pub(crate) load_seconds: f64,
    pub(crate) load_metrics: NativeGraphLoadMetrics,
    pub(crate) stats: Vec<NativeGraphFallbackStats>,
}

pub(crate) fn native_graph_load_metrics_to_dict(
    py: Python<'_>,
    metrics: &NativeGraphLoadMetrics,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item(
        "signatures_record_batches_scanned",
        metrics.signatures_record_batches_scanned,
    )?;
    out.set_item("signatures_rows_scanned", metrics.signatures_rows_scanned)?;
    out.set_item("signatures_rows_loaded", metrics.signatures_rows_loaded)?;
    out.set_item(
        "paper_authors_record_batches_scanned",
        metrics.paper_authors_record_batches_scanned,
    )?;
    out.set_item(
        "paper_authors_rows_scanned",
        metrics.paper_authors_rows_scanned,
    )?;
    out.set_item(
        "paper_authors_rows_loaded",
        metrics.paper_authors_rows_loaded,
    )?;
    out.set_item(
        "specter_record_batches_scanned",
        metrics.specter_record_batches_scanned,
    )?;
    out.set_item("specter_rows_scanned", metrics.specter_rows_scanned)?;
    out.set_item("specter_rows_loaded", metrics.specter_rows_loaded)?;
    Ok(out.unbind())
}

pub(crate) fn native_graph_stats_to_pylist(
    py: Python<'_>,
    stats: &[NativeGraphFallbackStats],
) -> PyResult<Py<PyList>> {
    let out = PyList::empty(py);
    for stat in stats {
        let item = PyDict::new(py);
        item.set_item("input_signature_count", stat.input_signature_count)?;
        item.set_item("neighbor_mode", &stat.neighbor_mode)?;
        item.set_item("projection_count", stat.projection_count)?;
        item.set_item("projection_window", stat.projection_window)?;
        item.set_item("candidate_edge_count", stat.candidate_edge_count)?;
        item.set_item(
            "sparse_evidence_feature_count",
            stat.sparse_evidence_stats.sparse_evidence_feature_count,
        )?;
        item.set_item(
            "sparse_evidence_skipped_feature_count",
            stat.sparse_evidence_stats
                .sparse_evidence_skipped_feature_count,
        )?;
        item.set_item(
            "sparse_evidence_added_edge_count",
            stat.sparse_evidence_stats.sparse_evidence_added_edge_count,
        )?;
        item.set_item(
            "sparse_evidence_neighbors",
            stat.sparse_evidence_stats.sparse_evidence_neighbors,
        )?;
        item.set_item("raw_component_count", stat.raw_component_count)?;
        item.set_item("raw_max_component_size", stat.raw_max_component_size)?;
        item.set_item("raw_median_component_size", stat.raw_median_component_size)?;
        item.set_item("packed_component_count", stat.packed_component_count)?;
        item.set_item("max_component_size", stat.max_component_size)?;
        item.set_item("median_component_size", stat.median_component_size)?;
        item.set_item("pack_components", stat.pack_components)?;
        item.set_item("component_pack_strategy", &stat.component_pack_strategy)?;
        item.set_item("component_pack_top_k", stat.component_pack_top_k)?;
        item.set_item("local_move_passes", stat.local_move_passes)?;
        item.set_item("edge_build_seconds", stat.edge_build_seconds)?;
        item.set_item("total_seconds", stat.total_seconds)?;
        out.append(item)?;
    }
    Ok(out.unbind())
}

pub(crate) fn insert_native_graph_telemetry(
    py: Python<'_>,
    telemetry: &Bound<'_, PyDict>,
    graph_telemetry: &NativeGraphTelemetry,
) -> PyResult<()> {
    telemetry.set_item("graph_fallback_native", true)?;
    telemetry.set_item("graph_fallback_load_seconds", graph_telemetry.load_seconds)?;
    telemetry.set_item(
        "graph_fallback_load_metrics",
        native_graph_load_metrics_to_dict(py, &graph_telemetry.load_metrics)?,
    )?;
    telemetry.set_item(
        "graph_fallback_invocation_count",
        graph_telemetry.stats.len(),
    )?;
    telemetry.set_item(
        "graph_fallback_stats",
        native_graph_stats_to_pylist(py, &graph_telemetry.stats)?,
    )?;
    Ok(())
}

pub(crate) fn median_usize(values: &[usize]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid] as f64
    } else {
        (sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0
    }
}

pub(crate) fn normalize_f32_vector(mut vector: Vec<f32>, dimension: usize) -> Vec<f32> {
    if vector.len() != dimension {
        vector.resize(dimension, 0.0);
    }
    let norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for value in vector.iter_mut() {
            *value = (f64::from(*value) / norm) as f32;
        }
    }
    vector
}

pub(crate) fn dot_f32(left: &[f32], right: &[f32]) -> f64 {
    f64::from(
        left.iter()
            .zip(right.iter())
            .map(|(left_value, right_value)| *left_value * *right_value)
            .sum::<f32>(),
    )
}

pub(crate) fn jaccard(left: &HashSet<String>, right: &HashSet<String>) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let (smaller, larger) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    let intersection = smaller
        .iter()
        .filter(|value| larger.contains(*value))
        .count();
    let union = left.len() + right.len() - intersection;
    intersection as f64 / union as f64
}

pub(crate) fn native_graph_weighted_edge_score(
    cosine_similarity: f64,
    left: &NativeGraphSignatureEvidence,
    right: &NativeGraphSignatureEvidence,
    config: &NativeGraphSubblockingConfig,
) -> f64 {
    config.specter_weight * cosine_similarity
        + config.coauthor_weight * jaccard(&left.coauthor_blocks, &right.coauthor_blocks)
        + config.affiliation_weight * jaccard(&left.affiliation_keys, &right.affiliation_keys)
}

pub(crate) fn score_native_graph_candidate_edge(
    edge_scores: &mut HashMap<(usize, usize), f64>,
    left_index: usize,
    right_index: usize,
    evidences: &[NativeGraphSignatureEvidence],
    config: &NativeGraphSubblockingConfig,
) {
    if left_index == right_index {
        return;
    }
    let left = left_index.min(right_index);
    let right = left_index.max(right_index);
    let cosine_similarity =
        dot_f32(&evidences[left].embedding, &evidences[right].embedding).max(0.0);
    let score = native_graph_weighted_edge_score(
        cosine_similarity,
        &evidences[left],
        &evidences[right],
        config,
    );
    if score < config.min_edge_score {
        return;
    }
    match edge_scores.entry((left, right)) {
        Entry::Occupied(mut entry) => {
            if score > *entry.get() {
                entry.insert(score);
            }
        }
        Entry::Vacant(entry) => {
            entry.insert(score);
        }
    }
}

pub(crate) fn score_native_graph_candidate_edge_from_cosine(
    edge_scores: &mut HashMap<(usize, usize), f64>,
    left_index: usize,
    right_index: usize,
    cosine_similarity: f64,
    evidences: &[NativeGraphSignatureEvidence],
    config: &NativeGraphSubblockingConfig,
) {
    let score = native_graph_weighted_edge_score(
        cosine_similarity.max(0.0),
        &evidences[left_index],
        &evidences[right_index],
        config,
    );
    if score < config.min_edge_score {
        return;
    }
    match edge_scores.entry((left_index, right_index)) {
        Entry::Occupied(mut entry) => {
            if score > *entry.get() {
                entry.insert(score);
            }
        }
        Entry::Vacant(entry) => {
            entry.insert(score);
        }
    }
}

pub(crate) fn prune_native_graph_edge_scores(
    edge_scores: &mut HashMap<(usize, usize), f64>,
    max_candidate_edges: usize,
) {
    if max_candidate_edges == 0 || edge_scores.len() <= max_candidate_edges {
        return;
    }
    let mut strongest: Vec<((usize, usize), f64)> = edge_scores
        .iter()
        .map(|(key, value)| (*key, *value))
        .collect();
    strongest.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0 .0.cmp(&right.0 .0).reverse())
            .then_with(|| left.0 .1.cmp(&right.0 .1).reverse())
    });
    strongest.truncate(max_candidate_edges);
    edge_scores.clear();
    edge_scores.extend(strongest);
}

pub(crate) fn exact_native_graph_edge_scores(
    evidences: &[NativeGraphSignatureEvidence],
    config: &NativeGraphSubblockingConfig,
) -> HashMap<(usize, usize), f64> {
    let neighbor_count = evidences.len().min((config.neighbors + 1).max(2));
    let mut edge_scores = HashMap::<(usize, usize), f64>::new();
    for left_index in 0..evidences.len() {
        let mut neighbors: Vec<(usize, f64)> = (0..evidences.len())
            .map(|right_index| {
                (
                    right_index,
                    dot_f32(
                        &evidences[left_index].embedding,
                        &evidences[right_index].embedding,
                    ),
                )
            })
            .collect();
        neighbors.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        for (right_index, cosine_similarity) in neighbors.into_iter().take(neighbor_count) {
            if right_index == left_index {
                continue;
            }
            let left = left_index.min(right_index);
            let right = left_index.max(right_index);
            score_native_graph_candidate_edge_from_cosine(
                &mut edge_scores,
                left,
                right,
                cosine_similarity,
                evidences,
                config,
            );
        }
    }
    edge_scores
}

pub(crate) fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

pub(crate) fn splitmix_unit_f64(state: &mut u64) -> f64 {
    let raw = splitmix64_next(state) >> 11;
    (raw as f64) * (1.0 / ((1u64 << 53) as f64))
}

pub(crate) fn splitmix_normal_pair(state: &mut u64) -> (f64, f64) {
    let u1 = splitmix_unit_f64(state).max(f64::MIN_POSITIVE);
    let u2 = splitmix_unit_f64(state);
    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    (radius * theta.cos(), radius * theta.sin())
}

pub(crate) fn projection_seed(signature_ids: &[String], random_seed: u64) -> u64 {
    let mut state = FNV_OFFSET ^ random_seed;
    let mut sorted = signature_ids.to_vec();
    sorted.sort_unstable();
    for signature_id in sorted {
        state = fnv64_update(state, signature_id.as_bytes());
        state = fnv64_update(state, b"\0");
    }
    state
}

pub(crate) fn projection_native_graph_edge_scores(
    signature_ids: &[String],
    evidences: &[NativeGraphSignatureEvidence],
    config: &NativeGraphSubblockingConfig,
    random_seed: u64,
    dimension: usize,
) -> HashMap<(usize, usize), f64> {
    let mut edge_scores = HashMap::<(usize, usize), f64>::new();
    let mut state = projection_seed(signature_ids, random_seed);
    for _projection_index in 0..config.projection_count {
        let mut projection = vec![0.0_f32; dimension];
        let mut offset = 0usize;
        while offset < dimension {
            let (left, right) = splitmix_normal_pair(&mut state);
            projection[offset] = left as f32;
            if offset + 1 < dimension {
                projection[offset + 1] = right as f32;
            }
            offset += 2;
        }
        let projection_norm = projection
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        if projection_norm > 0.0 {
            for value in projection.iter_mut() {
                *value = (f64::from(*value) / projection_norm) as f32;
            }
        }
        let mut order: Vec<(usize, f64)> = evidences
            .iter()
            .enumerate()
            .map(|(index, evidence)| (index, dot_f32(&evidence.embedding, &projection)))
            .collect();
        order.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        for position in 0..order.len() {
            let left_index = order[position].0;
            let stop = order.len().min(position + config.projection_window + 1);
            for &right_index in order[position + 1..stop]
                .iter()
                .map(|(index, _score)| index)
            {
                let left = left_index.min(right_index);
                let right = left_index.max(right_index);
                if edge_scores.contains_key(&(left, right)) {
                    continue;
                }
                score_native_graph_candidate_edge(&mut edge_scores, left, right, evidences, config);
            }
        }
        prune_native_graph_edge_scores(&mut edge_scores, config.max_candidate_edges);
    }
    edge_scores
}

pub(crate) fn add_sparse_native_graph_edge_scores(
    edge_scores: &mut HashMap<(usize, usize), f64>,
    evidences: &[NativeGraphSignatureEvidence],
    config: &NativeGraphSubblockingConfig,
) -> SparseEvidenceStats {
    let mut stats = SparseEvidenceStats {
        sparse_evidence_neighbors: config.sparse_evidence_neighbors,
        ..SparseEvidenceStats::default()
    };
    let mut postings: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, evidence) in evidences.iter().enumerate() {
        if config.sparse_evidence_include_coauthors {
            for value in evidence.coauthor_blocks.iter() {
                if !value.is_empty() {
                    postings
                        .entry(format!("coauthor:{value}"))
                        .or_default()
                        .push(index);
                }
            }
        }
        if config.sparse_evidence_include_affiliations {
            for value in evidence.affiliation_keys.iter() {
                if !value.is_empty() {
                    postings
                        .entry(format!("affiliation:{value}"))
                        .or_default()
                        .push(index);
                }
            }
        }
    }
    let edge_count_before = edge_scores.len();
    for indices in postings.values_mut() {
        indices.sort_unstable();
        indices.dedup();
        if indices.len() <= 1 {
            continue;
        }
        if indices.len() > config.sparse_evidence_max_posting_size {
            stats.sparse_evidence_skipped_feature_count += 1;
            continue;
        }
        stats.sparse_evidence_feature_count += 1;
        for left_offset in 0..indices.len() {
            let left_index = indices[left_offset];
            let right_stop = if config.sparse_evidence_neighbors == 0 {
                indices.len()
            } else {
                indices
                    .len()
                    .min(left_offset + config.sparse_evidence_neighbors + 1)
            };
            for &right_index in indices[left_offset + 1..right_stop].iter() {
                let cosine_similarity = dot_f32(
                    &evidences[left_index].embedding,
                    &evidences[right_index].embedding,
                )
                .max(0.0);
                let score = native_graph_weighted_edge_score(
                    cosine_similarity,
                    &evidences[left_index],
                    &evidences[right_index],
                    config,
                );
                if score < config.sparse_evidence_min_weight {
                    continue;
                }
                let left = left_index.min(right_index);
                let right = left_index.max(right_index);
                match edge_scores.entry((left, right)) {
                    Entry::Occupied(mut entry) => {
                        if score > *entry.get() {
                            entry.insert(score);
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(score);
                    }
                }
            }
        }
        if config.max_candidate_edges > 0
            && edge_scores.len() > config.max_candidate_edges.saturating_mul(2)
        {
            prune_native_graph_edge_scores(edge_scores, config.max_candidate_edges);
        }
    }
    prune_native_graph_edge_scores(edge_scores, config.max_candidate_edges);
    stats.sparse_evidence_added_edge_count = edge_scores.len().saturating_sub(edge_count_before);
    stats
}

pub(crate) struct NativeGraphUnionFind {
    pub(crate) parent: Vec<usize>,
    pub(crate) component_size: Vec<usize>,
}

impl NativeGraphUnionFind {
    pub(crate) fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            component_size: vec![1; size],
        }
    }

    pub(crate) fn find(&mut self, mut item: usize) -> usize {
        while self.parent[item] != item {
            let parent = self.parent[item];
            let grandparent = self.parent[parent];
            self.parent[item] = grandparent;
            item = grandparent;
        }
        item
    }

    pub(crate) fn union_if_capacity(
        &mut self,
        left: usize,
        right: usize,
        maximum_size: usize,
    ) -> bool {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return false;
        }
        let merged_size = self.component_size[left_root] + self.component_size[right_root];
        if merged_size > maximum_size {
            return false;
        }
        if self.component_size[left_root] < self.component_size[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        self.component_size[left_root] = merged_size;
        true
    }
}

pub(crate) fn ordered_native_graph_components(
    signature_ids: &[String],
    uf: &mut NativeGraphUnionFind,
) -> (Vec<Vec<String>>, Vec<usize>, HashMap<usize, usize>) {
    let mut root_by_index = Vec::with_capacity(signature_ids.len());
    let mut components_by_root: HashMap<usize, Vec<String>> = HashMap::new();
    for (index, signature_id) in signature_ids.iter().enumerate() {
        let root = uf.find(index);
        root_by_index.push(root);
        components_by_root
            .entry(root)
            .or_default()
            .push(signature_id.clone());
    }
    let mut roots: Vec<usize> = components_by_root.keys().copied().collect();
    roots.sort_by(|left, right| {
        let left_values = components_by_root.get(left).expect("left root exists");
        let right_values = components_by_root.get(right).expect("right root exists");
        right_values.len().cmp(&left_values.len()).then_with(|| {
            let left_first = left_values.iter().min().expect("left component nonempty");
            let right_first = right_values.iter().min().expect("right component nonempty");
            left_first.cmp(right_first)
        })
    });
    let component_id_by_root: HashMap<usize, usize> = roots
        .iter()
        .enumerate()
        .map(|(component_id, root)| (*root, component_id))
        .collect();
    let mut components = Vec::<Vec<String>>::with_capacity(roots.len());
    for root in roots {
        let mut values = components_by_root
            .remove(&root)
            .expect("component root exists");
        values.sort_unstable();
        components.push(values);
    }
    (components, root_by_index, component_id_by_root)
}

pub(crate) fn native_component_adjacency(
    edge_scores: &HashMap<(usize, usize), f64>,
    root_by_index: &[usize],
    component_id_by_root: &HashMap<usize, usize>,
    aggregate: bool,
) -> HashMap<usize, HashMap<usize, f64>> {
    let mut adjacency: HashMap<usize, HashMap<usize, f64>> = HashMap::new();
    for ((left_index, right_index), score) in edge_scores {
        let left_component = component_id_by_root[&root_by_index[*left_index]];
        let right_component = component_id_by_root[&root_by_index[*right_index]];
        if left_component == right_component {
            continue;
        }
        if aggregate {
            *adjacency
                .entry(left_component)
                .or_default()
                .entry(right_component)
                .or_insert(0.0) += *score;
            *adjacency
                .entry(right_component)
                .or_default()
                .entry(left_component)
                .or_insert(0.0) += *score;
        } else {
            let left_neighbors = adjacency.entry(left_component).or_default();
            if left_neighbors
                .get(&right_component)
                .map_or(true, |current| *score > *current)
            {
                left_neighbors.insert(right_component, *score);
                adjacency
                    .entry(right_component)
                    .or_default()
                    .insert(left_component, *score);
            }
        }
    }
    adjacency
}

pub(crate) fn component_affinities_to_bins(
    component_id: usize,
    component_to_bin: &HashMap<usize, usize>,
    adjacency: &HashMap<usize, HashMap<usize, f64>>,
    top_k: usize,
) -> HashMap<usize, f64> {
    let mut scores_by_bin: HashMap<usize, Vec<f64>> = HashMap::new();
    if let Some(neighbors) = adjacency.get(&component_id) {
        for (neighbor_component_id, score) in neighbors {
            if let Some(bin_index) = component_to_bin.get(neighbor_component_id) {
                scores_by_bin.entry(*bin_index).or_default().push(*score);
            }
        }
    }
    let mut out = HashMap::new();
    for (bin_index, mut scores) in scores_by_bin {
        if top_k > 0 && scores.len() > top_k {
            scores.sort_by(|left, right| right.total_cmp(left));
            scores.truncate(top_k);
        }
        out.insert(bin_index, scores.iter().sum());
    }
    out
}

pub(crate) fn pack_native_components_by_size(
    components: &[Vec<String>],
    target_subblock_size: usize,
) -> PyResult<Vec<Vec<String>>> {
    let mut bins: Vec<Vec<String>> = Vec::new();
    let mut bin_sizes: Vec<usize> = Vec::new();
    for component in components {
        let component_size = component.len();
        if component_size > target_subblock_size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Graph component size {component_size} exceeds target_subblock_size={target_subblock_size}"
            )));
        }
        let mut best_bin = None;
        let mut best_remaining = None;
        for (bin_index, bin_size) in bin_sizes.iter().enumerate() {
            let remaining = target_subblock_size - *bin_size;
            if component_size <= remaining
                && best_remaining.map_or(true, |current| remaining < current)
            {
                best_bin = Some(bin_index);
                best_remaining = Some(remaining);
            }
        }
        if let Some(bin_index) = best_bin {
            bins[bin_index].extend(component.iter().cloned());
            bin_sizes[bin_index] += component_size;
        } else {
            bins.push(component.clone());
            bin_sizes.push(component_size);
        }
    }
    for values in bins.iter_mut() {
        values.sort_unstable();
    }
    Ok(bins)
}

pub(crate) fn pack_native_component_ids_greedy(
    components: &[Vec<String>],
    edge_scores: &HashMap<(usize, usize), f64>,
    root_by_index: &[usize],
    component_id_by_root: &HashMap<usize, usize>,
    target_subblock_size: usize,
    config: &NativeGraphSubblockingConfig,
) -> PyResult<Vec<Vec<usize>>> {
    let use_aggregate = config.component_pack_strategy == "aggregate-greedy";
    let adjacency = native_component_adjacency(
        edge_scores,
        root_by_index,
        component_id_by_root,
        use_aggregate,
    );
    let mut component_order: Vec<usize> = (0..components.len()).collect();
    component_order.sort_by(|left, right| {
        components[*right]
            .len()
            .cmp(&components[*left].len())
            .then_with(|| components[*left][0].cmp(&components[*right][0]))
    });
    let mut bins: Vec<Vec<usize>> = Vec::new();
    let mut bin_sizes: Vec<usize> = Vec::new();
    let mut component_to_bin: HashMap<usize, usize> = HashMap::new();
    for component_id in component_order {
        let component_size = components[component_id].len();
        if component_size > target_subblock_size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Graph component size {component_size} exceeds target_subblock_size={target_subblock_size}"
            )));
        }
        let mut candidate_bins: HashMap<usize, f64> = HashMap::new();
        if use_aggregate {
            for (bin_index, affinity) in component_affinities_to_bins(
                component_id,
                &component_to_bin,
                &adjacency,
                config.component_pack_top_k,
            ) {
                if bin_index < bins.len()
                    && bin_sizes[bin_index] + component_size <= target_subblock_size
                    && affinity > 0.0
                {
                    candidate_bins.insert(bin_index, affinity);
                }
            }
        } else if let Some(neighbors) = adjacency.get(&component_id) {
            for (neighbor_component, score) in neighbors {
                if let Some(bin_index) = component_to_bin.get(neighbor_component) {
                    if bin_sizes[*bin_index] + component_size > target_subblock_size {
                        continue;
                    }
                    if candidate_bins
                        .get(bin_index)
                        .map_or(true, |current| *score > *current)
                    {
                        candidate_bins.insert(*bin_index, *score);
                    }
                }
            }
        }
        let selected_bin = if candidate_bins.is_empty() {
            let mut best_bin = None;
            let mut best_remaining = None;
            for (bin_index, bin_size) in bin_sizes.iter().enumerate() {
                let remaining = target_subblock_size - *bin_size;
                if component_size <= remaining
                    && best_remaining.map_or(true, |current| remaining < current)
                {
                    best_bin = Some(bin_index);
                    best_remaining = Some(remaining);
                }
            }
            best_bin
        } else {
            candidate_bins.keys().copied().min_by(|left, right| {
                let left_affinity = candidate_bins[left];
                let right_affinity = candidate_bins[right];
                right_affinity
                    .total_cmp(&left_affinity)
                    .then_with(|| {
                        let left_remaining =
                            target_subblock_size - (bin_sizes[*left] + component_size);
                        let right_remaining =
                            target_subblock_size - (bin_sizes[*right] + component_size);
                        left_remaining.cmp(&right_remaining)
                    })
                    .then_with(|| left.cmp(right))
            })
        };
        let bin_index = match selected_bin {
            Some(value) => value,
            None => {
                bins.push(Vec::new());
                bin_sizes.push(0);
                bins.len() - 1
            }
        };
        bins[bin_index].push(component_id);
        bin_sizes[bin_index] += component_size;
        component_to_bin.insert(component_id, bin_index);
    }
    if config.local_move_passes == 0 {
        return Ok(bins);
    }
    Ok(local_move_native_component_bins(
        components,
        &bins,
        &adjacency,
        target_subblock_size,
        config.local_move_passes,
        config.component_pack_top_k,
    ))
}

pub(crate) fn local_move_native_component_bins(
    components: &[Vec<String>],
    bins: &[Vec<usize>],
    adjacency: &HashMap<usize, HashMap<usize, f64>>,
    target_subblock_size: usize,
    passes: usize,
    top_k: usize,
) -> Vec<Vec<usize>> {
    let mut working_bins = bins.to_vec();
    let mut bin_sizes: Vec<usize> = working_bins
        .iter()
        .map(|component_ids| {
            component_ids
                .iter()
                .map(|component_id| components[*component_id].len())
                .sum()
        })
        .collect();
    for _pass_index in 0..passes {
        let mut moved = false;
        let mut component_to_bin: HashMap<usize, usize> = HashMap::new();
        for (bin_index, component_ids) in working_bins.iter().enumerate() {
            for component_id in component_ids {
                component_to_bin.insert(*component_id, bin_index);
            }
        }
        let mut component_order: Vec<usize> = component_to_bin.keys().copied().collect();
        component_order.sort_by(|left, right| {
            components[*left]
                .len()
                .cmp(&components[*right].len())
                .then_with(|| components[*left][0].cmp(&components[*right][0]))
        });
        for component_id in component_order {
            let Some(source_bin) = component_to_bin.get(&component_id).copied() else {
                continue;
            };
            let component_size = components[component_id].len();
            let affinities =
                component_affinities_to_bins(component_id, &component_to_bin, adjacency, top_k);
            let current_affinity = affinities.get(&source_bin).copied().unwrap_or(0.0);
            let mut best_bin = None;
            let mut best_gain = 0.0;
            for (target_bin, candidate_affinity) in affinities {
                if target_bin >= working_bins.len() || target_bin == source_bin {
                    continue;
                }
                if bin_sizes[target_bin] + component_size > target_subblock_size {
                    continue;
                }
                let gain = candidate_affinity - current_affinity;
                if gain > best_gain {
                    best_gain = gain;
                    best_bin = Some(target_bin);
                }
            }
            let Some(target_bin) = best_bin else {
                continue;
            };
            if let Some(position) = working_bins[source_bin]
                .iter()
                .position(|candidate| *candidate == component_id)
            {
                working_bins[source_bin].remove(position);
                working_bins[target_bin].push(component_id);
                bin_sizes[source_bin] -= component_size;
                bin_sizes[target_bin] += component_size;
                moved = true;
            }
        }
        if !moved {
            break;
        }
        let mut nonempty_bins = Vec::new();
        let mut nonempty_sizes = Vec::new();
        for (component_ids, bin_size) in working_bins.into_iter().zip(bin_sizes.into_iter()) {
            if !component_ids.is_empty() {
                nonempty_bins.push(component_ids);
                nonempty_sizes.push(bin_size);
            }
        }
        working_bins = nonempty_bins;
        bin_sizes = nonempty_sizes;
    }
    working_bins
}

pub(crate) fn component_ids_to_native_subblocks(
    components: &[Vec<String>],
    bins: &[Vec<usize>],
) -> Vec<Vec<String>> {
    let mut packed = Vec::new();
    for component_ids in bins {
        let mut ordered_component_ids = component_ids.clone();
        ordered_component_ids
            .sort_by(|left, right| components[*left][0].cmp(&components[*right][0]));
        let mut values = Vec::new();
        for component_id in ordered_component_ids {
            values.extend(components[component_id].iter().cloned());
        }
        values.sort_unstable();
        packed.push(values);
    }
    packed
}

pub(crate) fn pack_native_graph_components(
    components: &[Vec<String>],
    edge_scores: &HashMap<(usize, usize), f64>,
    root_by_index: &[usize],
    component_id_by_root: &HashMap<usize, usize>,
    target_subblock_size: usize,
    config: &NativeGraphSubblockingConfig,
) -> PyResult<Vec<Vec<String>>> {
    if !config.pack_components {
        return Ok(components.to_vec());
    }
    if config.component_pack_strategy == "size" {
        return pack_native_components_by_size(components, target_subblock_size);
    }
    let bins = pack_native_component_ids_greedy(
        components,
        edge_scores,
        root_by_index,
        component_id_by_root,
        target_subblock_size,
        config,
    )?;
    Ok(component_ids_to_native_subblocks(components, &bins))
}

pub(crate) fn native_graph_cluster(
    signature_ids: Vec<String>,
    store: &NativeGraphEvidenceStore,
    target_subblock_size: usize,
    config: &NativeGraphSubblockingConfig,
    random_seed: u64,
    telemetry: &mut NativeGraphTelemetry,
) -> PyResult<HashMap<String, Vec<String>>> {
    let fallback_start = Instant::now();
    if signature_ids.is_empty() {
        return Ok(HashMap::new());
    }
    if signature_ids.len() <= target_subblock_size {
        return Ok(HashMap::from([("0".to_string(), signature_ids)]));
    }
    let config = config.effective_for_group(signature_ids.len());
    if config.neighbor_mode == "exact" && signature_ids.len() > config.max_exact_knn_group_size {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Exact graph subblocking fallback group exceeds max_exact_knn_group_size: group_size={} max_exact_knn_group_size={}",
            signature_ids.len(),
            config.max_exact_knn_group_size
        )));
    }
    let mut evidences = Vec::with_capacity(signature_ids.len());
    for signature_id in signature_ids.iter() {
        let evidence = store.signatures.get(signature_id).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow graph subblocking evidence is missing required signature: {signature_id:?}"
            ))
        })?;
        evidences.push(evidence.clone());
    }
    let edge_start = Instant::now();
    let mut edge_scores = if config.neighbor_mode == "exact" {
        exact_native_graph_edge_scores(&evidences, &config)
    } else {
        projection_native_graph_edge_scores(
            &signature_ids,
            &evidences,
            &config,
            random_seed,
            store.dimension,
        )
    };
    let sparse_evidence_stats = if config.sparse_evidence_edges {
        add_sparse_native_graph_edge_scores(&mut edge_scores, &evidences, &config)
    } else {
        SparseEvidenceStats {
            sparse_evidence_neighbors: config.sparse_evidence_neighbors,
            ..SparseEvidenceStats::default()
        }
    };
    let edge_seconds = edge_start.elapsed().as_secs_f64();
    let mut uf = NativeGraphUnionFind::new(signature_ids.len());
    let mut sorted_edges: Vec<(f64, usize, usize)> = edge_scores
        .iter()
        .map(|((left, right), score)| (*score, *left, *right))
        .collect();
    sorted_edges.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| signature_ids[left.1].cmp(&signature_ids[right.1]))
            .then_with(|| signature_ids[left.2].cmp(&signature_ids[right.2]))
    });
    for (_score, left, right) in sorted_edges {
        uf.union_if_capacity(left, right, target_subblock_size);
    }
    let (raw_components, root_by_index, component_id_by_root) =
        ordered_native_graph_components(&signature_ids, &mut uf);
    let mut ordered_components = pack_native_graph_components(
        &raw_components,
        &edge_scores,
        &root_by_index,
        &component_id_by_root,
        target_subblock_size,
        &config,
    )?;
    ordered_components.sort_by(|left, right| {
        right
            .len()
            .cmp(&left.len())
            .then_with(|| left[0].cmp(&right[0]))
    });
    let raw_sizes: Vec<usize> = raw_components.iter().map(Vec::len).collect();
    let sizes: Vec<usize> = ordered_components.iter().map(Vec::len).collect();
    telemetry.stats.push(NativeGraphFallbackStats {
        input_signature_count: signature_ids.len(),
        neighbor_mode: config.neighbor_mode.clone(),
        projection_count: config.projection_count,
        projection_window: config.projection_window,
        candidate_edge_count: edge_scores.len(),
        sparse_evidence_stats,
        raw_component_count: raw_components.len(),
        raw_max_component_size: raw_sizes.iter().copied().max().unwrap_or(0),
        raw_median_component_size: median_usize(&raw_sizes),
        packed_component_count: ordered_components.len(),
        max_component_size: sizes.iter().copied().max().unwrap_or(0),
        median_component_size: median_usize(&sizes),
        pack_components: config.pack_components,
        component_pack_strategy: config.component_pack_strategy.clone(),
        component_pack_top_k: config.component_pack_top_k,
        local_move_passes: config.local_move_passes,
        edge_build_seconds: edge_seconds,
        total_seconds: fallback_start.elapsed().as_secs_f64(),
    });
    Ok(ordered_components
        .into_iter()
        .enumerate()
        .map(|(index, values)| (index.to_string(), values))
        .collect())
}

pub(crate) fn collect_native_graph_signature_groups(
    signature_groups: &[Vec<String>],
) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for group in signature_groups {
        for signature_id in group {
            if seen.insert(signature_id.clone()) {
                out.push(signature_id.clone());
            }
        }
    }
    out
}

pub(crate) fn build_native_graph_evidence_store(
    py: Python<'_>,
    paths: &Bound<'_, PyAny>,
    row_by_signature_id: &HashMap<String, SubblockingSignatureRow>,
    fallback_signature_groups: &[Vec<String>],
    telemetry: &mut NativeGraphTelemetry,
) -> PyResult<NativeGraphEvidenceStore> {
    let load_start = Instant::now();
    let graph_paths = NativeGraphArrowPaths::from_py(paths)?;
    let fallback_signature_ids = collect_native_graph_signature_groups(fallback_signature_groups);
    telemetry.load_metrics.signatures_record_batches_scanned = 0;
    telemetry.load_metrics.signatures_rows_scanned = fallback_signature_ids.len();
    telemetry.load_metrics.signatures_rows_loaded = fallback_signature_ids.len();
    let mut paper_ids = HashSet::<String>::new();
    for signature_id in fallback_signature_ids.iter() {
        let row = row_by_signature_id.get(signature_id).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "signatures Arrow is missing graph-subblocking signature ids: {signature_id:?}"
            ))
        })?;
        if row.paper_id.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "signatures Arrow cannot contain empty paper_id values",
            ));
        }
        if row.position.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "signatures Arrow author_position is null for graph subblocking signature_id '{signature_id}'"
            )));
        }
        paper_ids.insert(row.paper_id.clone());
    }
    let (paper_authors, paper_author_stats) = read_raw_arrow_paper_authors_with_optional_index(
        &graph_paths.paper_authors_path,
        Some(&graph_paths.paper_authors_batch_index_path),
        &paper_ids,
    )?;
    let (specter_by_paper_id, specter_stats) = read_raw_arrow_specter_with_optional_index(
        &graph_paths.specter_path,
        Some(&graph_paths.specter_batch_index_path),
        &paper_ids,
    )?;
    telemetry.load_metrics.paper_authors_record_batches_scanned = paper_author_stats.batches_read;
    telemetry.load_metrics.paper_authors_rows_scanned = paper_author_stats.rows_scanned;
    telemetry.load_metrics.paper_authors_rows_loaded =
        paper_authors.values().map(Vec::len).sum::<usize>();
    telemetry.load_metrics.specter_record_batches_scanned = specter_stats.batches_read;
    telemetry.load_metrics.specter_rows_scanned = specter_stats.rows_scanned;
    telemetry.load_metrics.specter_rows_loaded = specter_by_paper_id.len();

    let affiliation_stopwords = extract_affiliation_stopwords(py)?;
    let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
    for signature_id in fallback_signature_ids.iter() {
        let row = row_by_signature_id
            .get(signature_id)
            .expect("fallback signature row exists after validation");
        for affiliation in row.affiliations.iter() {
            ensure_unidecode_for_text(affiliation, &mut unidecode_char_map)?;
        }
    }
    for authors in paper_authors.values() {
        for (_position, author_name) in authors {
            ensure_unidecode_for_text(author_name, &mut unidecode_char_map)?;
        }
    }

    let dimension = specter_by_paper_id.values().next().map_or(0usize, Vec::len);
    let mut signatures = HashMap::with_capacity(fallback_signature_ids.len());
    for signature_id in fallback_signature_ids {
        let row = row_by_signature_id
            .get(&signature_id)
            .expect("fallback signature row exists after validation");
        let embedding = specter_by_paper_id
            .get(&row.paper_id)
            .cloned()
            .unwrap_or_else(|| vec![0.0; dimension]);
        let position = row
            .position
            .expect("fallback signature row author_position was validated");
        let mut coauthor_blocks = HashSet::new();
        if let Some(authors) = paper_authors.get(&row.paper_id) {
            for (author_position, author_name) in authors {
                if *author_position == position {
                    continue;
                }
                let trimmed = author_name.trim();
                if trimmed.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "paper_authors Arrow cannot contain empty author_name values",
                    ));
                }
                let normalized =
                    normalize_text_compat_from_map(trimmed, false, &unidecode_char_map);
                let block = compute_block_compat(&normalized);
                if !block.is_empty() {
                    coauthor_blocks.insert(block);
                }
            }
        }
        let normalized_affiliations: Vec<String> = row
            .affiliations
            .iter()
            .filter_map(|affiliation| {
                let normalized =
                    normalize_text_compat_from_map(affiliation, false, &unidecode_char_map);
                if normalized.is_empty() {
                    None
                } else {
                    Some(normalized)
                }
            })
            .collect();
        let affiliation_text = normalized_affiliations.join(" ");
        let affiliation_keys: HashSet<String> =
            word_ngrams_counter_python_compat(&affiliation_text, &affiliation_stopwords, true)
                .into_keys()
                .collect();
        signatures.insert(
            signature_id,
            NativeGraphSignatureEvidence {
                embedding: normalize_f32_vector(embedding, dimension),
                coauthor_blocks,
                affiliation_keys,
            },
        );
    }
    telemetry.load_seconds = load_start.elapsed().as_secs_f64();
    Ok(NativeGraphEvidenceStore {
        signatures,
        dimension,
    })
}

#[derive(Default)]
pub(crate) struct OrderedSubblocks {
    pub(crate) entries: Vec<(String, Vec<String>)>,
}

impl OrderedSubblocks {
    pub(crate) fn insert(&mut self, key: String, signature_ids: Vec<String>) {
        if let Some((_existing_key, existing_values)) = self
            .entries
            .iter_mut()
            .find(|(existing_key, _)| *existing_key == key)
        {
            *existing_values = signature_ids;
        } else {
            self.entries.push((key, signature_ids));
        }
    }

    pub(crate) fn remove(&mut self, key: &str) -> Option<Vec<String>> {
        let position = self
            .entries
            .iter()
            .position(|(existing_key, _)| existing_key == key)?;
        Some(self.entries.remove(position).1)
    }

    pub(crate) fn get(&self, key: &str) -> Option<&Vec<String>> {
        self.entries
            .iter()
            .find(|(existing_key, _)| existing_key == key)
            .map(|(_key, values)| values)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&String, &Vec<String>)> {
        self.entries.iter().map(|(key, values)| (key, values))
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn to_hashmap(&self) -> HashMap<String, Vec<String>> {
        self.entries
            .iter()
            .map(|(key, values)| (key.clone(), values.clone()))
            .collect()
    }
}

#[derive(Default)]
pub(crate) struct SubblockingTelemetry {
    pub(crate) maximum_size: usize,
    pub(crate) input_signature_count: usize,
    pub(crate) single_letter_first_name_signature_count: usize,
    pub(crate) multi_letter_first_name_signature_count: usize,
    pub(crate) first_name_dead_end_block_count: usize,
    pub(crate) first_name_dead_end_signature_count: usize,
    pub(crate) specter_fallback_candidate_block_count: usize,
    pub(crate) specter_fallback_candidate_signature_count: usize,
    pub(crate) specter_non_invoked_candidate_block_count: usize,
    pub(crate) specter_non_invoked_candidate_signature_count: usize,
    pub(crate) specter_invocation_count: usize,
    pub(crate) specter_input_signature_count: usize,
    pub(crate) pre_merge_subblock_count: usize,
    pub(crate) pre_merge_specter_labeled_subblock_count: usize,
    pub(crate) pre_merge_specter_labeled_signature_count: usize,
    pub(crate) orcid_subblocking_enabled: bool,
    pub(crate) orcid_merge_skipped_due_to_capacity_count: usize,
    pub(crate) orcid_merge_skipped_due_to_capacity_signature_count: usize,
    pub(crate) final_subblock_count: usize,
    pub(crate) final_specter_labeled_subblock_count: usize,
    pub(crate) final_specter_labeled_signature_count: usize,
}

impl SubblockingTelemetry {
    pub(crate) fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let telemetry = PyDict::new(py);
        telemetry.set_item("maximum_size", self.maximum_size)?;
        telemetry.set_item("input_signature_count", self.input_signature_count)?;
        telemetry.set_item(
            "single_letter_first_name_signature_count",
            self.single_letter_first_name_signature_count,
        )?;
        telemetry.set_item(
            "multi_letter_first_name_signature_count",
            self.multi_letter_first_name_signature_count,
        )?;
        telemetry.set_item(
            "first_name_dead_end_block_count",
            self.first_name_dead_end_block_count,
        )?;
        telemetry.set_item(
            "first_name_dead_end_signature_count",
            self.first_name_dead_end_signature_count,
        )?;
        telemetry.set_item(
            "specter_fallback_candidate_block_count",
            self.specter_fallback_candidate_block_count,
        )?;
        telemetry.set_item(
            "specter_fallback_candidate_signature_count",
            self.specter_fallback_candidate_signature_count,
        )?;
        telemetry.set_item(
            "specter_non_invoked_candidate_block_count",
            self.specter_non_invoked_candidate_block_count,
        )?;
        telemetry.set_item(
            "specter_non_invoked_candidate_signature_count",
            self.specter_non_invoked_candidate_signature_count,
        )?;
        telemetry.set_item("specter_invocation_count", self.specter_invocation_count)?;
        telemetry.set_item(
            "specter_input_signature_count",
            self.specter_input_signature_count,
        )?;
        telemetry.set_item("pre_merge_subblock_count", self.pre_merge_subblock_count)?;
        telemetry.set_item(
            "pre_merge_specter_labeled_subblock_count",
            self.pre_merge_specter_labeled_subblock_count,
        )?;
        telemetry.set_item(
            "pre_merge_specter_labeled_signature_count",
            self.pre_merge_specter_labeled_signature_count,
        )?;
        telemetry.set_item("orcid_subblocking_enabled", self.orcid_subblocking_enabled)?;
        telemetry.set_item(
            "orcid_merge_skipped_due_to_capacity_count",
            self.orcid_merge_skipped_due_to_capacity_count,
        )?;
        telemetry.set_item(
            "orcid_merge_skipped_due_to_capacity_signature_count",
            self.orcid_merge_skipped_due_to_capacity_signature_count,
        )?;
        telemetry.set_item("final_subblock_count", self.final_subblock_count)?;
        telemetry.set_item(
            "final_specter_labeled_subblock_count",
            self.final_specter_labeled_subblock_count,
        )?;
        telemetry.set_item(
            "final_specter_labeled_signature_count",
            self.final_specter_labeled_signature_count,
        )?;
        Ok(telemetry.unbind())
    }
}

#[derive(Clone)]
pub(crate) struct PrefixCount {
    pub(crate) name: String,
    pub(crate) count: usize,
    pub(crate) first_index: usize,
    pub(crate) signature_ids: Vec<String>,
}

pub(crate) fn py_prefix(value: &str, width: usize) -> String {
    value.chars().take(width).collect()
}

pub(crate) fn prefix_counts(
    names: &[String],
    signature_ids: &[String],
    width: usize,
) -> Vec<PrefixCount> {
    let mut counts: HashMap<String, PrefixCount> = HashMap::new();
    for (index, (name, signature_id)) in names.iter().zip(signature_ids.iter()).enumerate() {
        let prefix = py_prefix(name, width);
        let entry = counts.entry(prefix.clone()).or_insert_with(|| PrefixCount {
            name: prefix,
            count: 0,
            first_index: index,
            signature_ids: Vec::new(),
        });
        entry.count += 1;
        entry.signature_ids.push(signature_id.clone());
    }
    let mut out: Vec<PrefixCount> = counts.into_values().collect();
    out.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.first_index.cmp(&right.first_index))
    });
    out
}

pub(crate) fn subdivide_helper_rust(
    mut names: Vec<String>,
    mut signature_ids: Vec<String>,
    maximum_size: usize,
    starting_k: usize,
) -> (OrderedSubblocks, OrderedSubblocks) {
    let n_signature_ids = signature_ids.len();
    if n_signature_ids == 0 {
        return (OrderedSubblocks::default(), OrderedSubblocks::default());
    }
    let mut output = OrderedSubblocks::default();
    let mut output_cant_subdivide = OrderedSubblocks::default();
    let max_len = names.iter().map(|name| py_len(name)).max().unwrap_or(0);
    let mut clean_break = false;

    for width in starting_k..=max_len {
        let counts = prefix_counts(&names, &signature_ids, width);
        let good_size_counts: Vec<&PrefixCount> = counts
            .iter()
            .filter(|count| count.count <= maximum_size)
            .collect();
        if good_size_counts.is_empty() {
            for count in counts.iter() {
                output_cant_subdivide.insert(count.name.clone(), count.signature_ids.clone());
            }
            clean_break = true;
            break;
        }

        for count in good_size_counts {
            output.insert(count.name.clone(), count.signature_ids.clone());
        }

        let bad_names: HashSet<String> = counts
            .iter()
            .filter(|count| count.count > maximum_size)
            .map(|count| count.name.clone())
            .collect();
        let mut next_names = Vec::new();
        let mut next_signature_ids = Vec::new();
        for (name, signature_id) in names.into_iter().zip(signature_ids.into_iter()) {
            if bad_names.contains(&py_prefix(&name, width)) {
                next_names.push(name);
                next_signature_ids.push(signature_id);
            }
        }
        names = next_names;
        signature_ids = next_signature_ids;
    }

    if !names.is_empty() && !clean_break {
        output_cant_subdivide.insert("final".to_string(), signature_ids);
    }
    (output, output_cant_subdivide)
}

pub(crate) fn specter_labeled_subblock_stats(subblocks: &OrderedSubblocks) -> (usize, usize) {
    let mut subblock_count = 0usize;
    let mut signature_count = 0usize;
    for (key, values) in subblocks.iter() {
        if key.contains("|specter=") {
            subblock_count += 1;
            signature_count += values.len();
        }
    }
    (subblock_count, signature_count)
}

pub(crate) struct SubblockMergeMetadata {
    pub(crate) size: usize,
    pub(crate) first_name: String,
    pub(crate) middle_name: Option<String>,
    pub(crate) name_for_splits: Option<String>,
    pub(crate) lookup: Option<String>,
}

pub(crate) fn subblock_merge_candidate_metadata(key: &str, size: usize) -> SubblockMergeMetadata {
    let key_parts: Vec<&str> = key.split('|').collect();
    let first_name = key_parts.first().copied().unwrap_or_default().to_string();
    // Match s2and/subblocking.py:1543-1554: only emit a middle_name when the second
    // segment explicitly carries a `key=value` token, mirroring Python's
    // `if "=" in key_parts[1]` guard. Returning `Some("")` for an `=`-less segment
    // would silently make the subblock eligible for merge fan-out against other
    // empty-middle subblocks.
    let middle_name = key_parts
        .get(1)
        .and_then(|part| part.split_once('='))
        .map(|(_left, right)| right.to_string());
    let name_for_splits = if py_len(&first_name) > 1 {
        Some(first_name.clone())
    } else if py_len(&first_name) == 1 && middle_name.is_some() {
        middle_name.clone()
    } else {
        None
    };
    // canonical_v2: the ORCID first-k-letter prefix-count artifact is keyed on
    // slices of FULL canonical first names (spaces included), so the lookup key
    // is the whole canonical name string. The legacy first-token reduction was
    // a shim for compact-joined keys and is retired.
    let lookup = name_for_splits.clone();
    SubblockMergeMetadata {
        size,
        first_name,
        middle_name,
        name_for_splits,
        lookup,
    }
}

pub(crate) fn common_prefix_char_count(left: &str, right: &str) -> usize {
    left.chars()
        .zip(right.chars())
        .take_while(|(left_char, right_char)| left_char == right_char)
        .count()
}

fn unordered_prefix_pair_score(
    counts: &HashMap<String, HashMap<String, f64>>,
    first: &str,
    second: &str,
) -> Option<f64> {
    let (left, right) = if first <= second {
        (first, second)
    } else {
        (second, first)
    };
    counts
        .get(left)
        .and_then(|nested| nested.get(right))
        .or_else(|| counts.get(right).and_then(|nested| nested.get(left)))
        .copied()
}

pub(crate) fn sorted_subblock_merge_candidates(
    output: &OrderedSubblocks,
    maximum_size: usize,
    first_k_letter_counts_sorted: &HashMap<String, HashMap<String, f64>>,
) -> PyResult<Vec<((String, String), f64)>> {
    let mut metadata: HashMap<String, SubblockMergeMetadata> = HashMap::new();
    let mut mergeable_keys = Vec::<String>::new();
    for (key, values) in output.iter() {
        if values.len() >= maximum_size {
            continue;
        }
        let row = subblock_merge_candidate_metadata(key, values.len());
        if row.name_for_splits.is_none() {
            continue;
        }
        metadata.insert(key.clone(), row);
        mergeable_keys.push(key.clone());
    }

    let mut candidates = Vec::<((String, String), f64)>::new();
    for left_index in 0..mergeable_keys.len() {
        for right_index in (left_index + 1)..mergeable_keys.len() {
            let left_key = &mergeable_keys[left_index];
            let right_key = &mergeable_keys[right_index];
            let left = metadata.get(left_key).expect("merge metadata exists");
            let right = metadata.get(right_key).expect("merge metadata exists");
            if left.size + right.size > maximum_size {
                continue;
            }
            let both_multi_letter = py_len(&left.first_name) > 1 && py_len(&right.first_name) > 1;
            let both_single_letter_with_middle = py_len(&left.first_name) == 1
                && py_len(&right.first_name) == 1
                && left.middle_name.is_some()
                && right.middle_name.is_some();
            if !both_multi_letter && !both_single_letter_with_middle {
                continue;
            }

            let left_name = left.name_for_splits.as_ref().expect("merge name exists");
            let right_name = right.name_for_splits.as_ref().expect("merge name exists");
            let pair = (left_key.clone(), right_key.clone());
            if left_name == right_name {
                let score = match (&left.middle_name, &right.middle_name) {
                    (Some(left_middle), Some(right_middle)) => {
                        common_prefix_char_count(left_middle, right_middle)
                    }
                    _ => 0,
                };
                candidates.push((pair, 1e10 + score as f64));
            } else if same_prefix_tokens(left_name, right_name) {
                let score = py_len(left_name).min(py_len(right_name));
                candidates.push((pair, 1e5 + score as f64));
            } else if let (Some(left_lookup), Some(right_lookup)) = (&left.lookup, &right.lookup) {
                if let Some(score) = unordered_prefix_pair_score(
                    first_k_letter_counts_sorted,
                    left_lookup,
                    right_lookup,
                ) {
                    candidates.push((pair, score));
                }
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| right.0 .0.cmp(&left.0 .0))
            .then_with(|| right.0 .1.cmp(&left.0 .1))
    });
    Ok(candidates)
}

pub(crate) fn merge_small_subblocks(
    output: &mut OrderedSubblocks,
    maximum_size: usize,
    first_k_letter_counts_sorted: &HashMap<String, HashMap<String, f64>>,
) -> PyResult<()> {
    let candidates =
        sorted_subblock_merge_candidates(output, maximum_size, first_k_letter_counts_sorted)?;
    let mut merging_log: BTreeMap<usize, HashSet<String>> = BTreeMap::new();
    let mut inverse_merging_log: HashMap<String, usize> = HashMap::new();
    let mut cluster_id = 0usize;

    for (pair, _score) in candidates {
        let pair_1_cluster_id = inverse_merging_log.get(&pair.0).copied();
        let pair_2_cluster_id = inverse_merging_log.get(&pair.1).copied();
        if pair_1_cluster_id.is_none() && pair_2_cluster_id.is_none() {
            let mut keys = HashSet::new();
            keys.insert(pair.0.clone());
            keys.insert(pair.1.clone());
            merging_log.insert(cluster_id, keys);
            inverse_merging_log.insert(pair.0.clone(), cluster_id);
            inverse_merging_log.insert(pair.1.clone(), cluster_id);
            cluster_id += 1;
        } else if pair_1_cluster_id.is_some()
            && pair_2_cluster_id.is_some()
            && pair_1_cluster_id == pair_2_cluster_id
        {
            continue;
        } else {
            let proposed_cluster = match (pair_1_cluster_id, pair_2_cluster_id) {
                (Some(left_id), Some(right_id)) if left_id != right_id => merging_log
                    .get(&left_id)
                    .expect("left merge cluster exists")
                    .union(
                        merging_log
                            .get(&right_id)
                            .expect("right merge cluster exists"),
                    )
                    .cloned()
                    .collect::<HashSet<_>>(),
                (Some(left_id), None) => {
                    let mut cluster = merging_log
                        .get(&left_id)
                        .expect("left merge cluster exists")
                        .clone();
                    cluster.insert(pair.0.clone());
                    cluster.insert(pair.1.clone());
                    cluster
                }
                (None, Some(right_id)) => {
                    let mut cluster = merging_log
                        .get(&right_id)
                        .expect("right merge cluster exists")
                        .clone();
                    cluster.insert(pair.0.clone());
                    cluster.insert(pair.1.clone());
                    cluster
                }
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "This should never happen",
                    ))
                }
            };
            let size_of_proposed: usize = proposed_cluster
                .iter()
                .map(|key| output.get(key).map_or(0, Vec::len))
                .sum();
            if size_of_proposed <= maximum_size {
                if let Some(left_id) = pair_1_cluster_id {
                    merging_log.insert(left_id, proposed_cluster.clone());
                    if let Some(right_id) = pair_2_cluster_id {
                        merging_log.remove(&right_id);
                    }
                    for key in proposed_cluster {
                        inverse_merging_log.insert(key, left_id);
                    }
                } else if let Some(right_id) = pair_2_cluster_id {
                    merging_log.insert(right_id, proposed_cluster.clone());
                    for key in proposed_cluster {
                        inverse_merging_log.insert(key, right_id);
                    }
                }
            }
        }
    }

    let mut counter_of_keys: HashMap<String, usize> = HashMap::new();
    for keys_to_merge in merging_log.values() {
        for key in keys_to_merge {
            *counter_of_keys.entry(key.clone()).or_insert(0) += 1;
        }
    }
    if counter_of_keys.values().any(|count| *count != 1) {
        return Err(pyo3::exceptions::PyAssertionError::new_err(
            "A subblock key appears in more than one merge cluster",
        ));
    }

    let merge_cluster_ids: Vec<usize> = merging_log.keys().copied().collect();
    for merge_cluster_id in merge_cluster_ids {
        let mut keys_to_merge: Vec<String> = merging_log
            .get(&merge_cluster_id)
            .expect("merge cluster exists")
            .iter()
            .cloned()
            .collect();
        keys_to_merge.sort_unstable();
        let key_of_keys = keys_to_merge.join(", ");
        let mut signature_ids_stacked = Vec::<String>::new();
        for key in keys_to_merge.iter() {
            if let Some(values) = output.get(key) {
                signature_ids_stacked.extend(values.iter().cloned());
            }
        }
        output.insert(key_of_keys, signature_ids_stacked);
        for key in keys_to_merge {
            output.remove(&key);
        }
    }
    Ok(())
}

pub(crate) fn find_orcid_subblock_root(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        let parent_index = parent[index];
        let grandparent_index = parent[parent_index];
        parent[index] = grandparent_index;
        index = grandparent_index;
    }
    index
}

pub(crate) fn union_orcid_subblocks(parent: &mut [usize], left: usize, right: usize) {
    let left_root = find_orcid_subblock_root(parent, left);
    let right_root = find_orcid_subblock_root(parent, right);
    if left_root != right_root {
        parent[right_root] = left_root;
    }
}

pub(crate) fn apply_orcid_subblocking(
    output: &mut OrderedSubblocks,
    row_by_signature_id: &HashMap<String, SubblockingSignatureRow>,
    maximum_size: usize,
    telemetry: &mut SubblockingTelemetry,
) -> PyResult<()> {
    let subblock_ids: Vec<String> = output.iter().map(|(key, _values)| key.clone()).collect();
    let mut sig_id_to_subblock_index: HashMap<String, usize> = HashMap::new();
    let mut sig_id_order = Vec::<String>::new();
    for (subblock_index, (subblock_id, sig_ids)) in output.entries.iter().enumerate() {
        for sig_id in sig_ids {
            // Upstream subblocking must produce a partition; a duplicated
            // signature_id would otherwise bind to an arbitrary subblock here
            // (last-wins) and silently diverge from the Python implementation.
            if sig_id_to_subblock_index
                .insert(sig_id.clone(), subblock_index)
                .is_some()
            {
                return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
                    "duplicate signature_id {sig_id:?} reached ORCID subblocking in subblock {subblock_id:?}; \
                     upstream subblocking must produce a complete partition"
                )));
            }
            sig_id_order.push(sig_id.clone());
        }
    }

    let mut orcid_to_sig_ids: HashMap<String, Vec<String>> = HashMap::new();
    let mut orcid_order = Vec::<String>::new();
    for sig_id in sig_id_order.iter() {
        let Some(row) = row_by_signature_id.get(sig_id) else {
            continue;
        };
        let Some(orcid_raw) = row.orcid.as_ref() else {
            continue;
        };
        let Some(orcid) = normalize_orcid_owned(orcid_raw) else {
            continue;
        };
        if !orcid_to_sig_ids.contains_key(&orcid) {
            orcid_order.push(orcid.clone());
        }
        orcid_to_sig_ids
            .entry(orcid)
            .or_default()
            .push(sig_id.clone());
    }

    let mut parent: Vec<usize> = (0..subblock_ids.len()).collect();
    let mut orcid_to_subblock_indices: HashMap<String, Vec<usize>> = HashMap::new();
    for orcid in orcid_order.iter() {
        let Some(orcid_sig_ids) = orcid_to_sig_ids.get(orcid) else {
            continue;
        };
        let mut seen_subblock_indices = HashSet::<usize>::new();
        let mut unique_subblock_indices = Vec::<usize>::new();
        for sig_id in orcid_sig_ids {
            let Some(subblock_index) = sig_id_to_subblock_index.get(sig_id).copied() else {
                continue;
            };
            if seen_subblock_indices.insert(subblock_index) {
                unique_subblock_indices.push(subblock_index);
            }
        }
        if unique_subblock_indices.len() <= 1 {
            continue;
        }
        let first_subblock_index = unique_subblock_indices[0];
        for subblock_index in unique_subblock_indices.iter().skip(1).copied() {
            union_orcid_subblocks(&mut parent, first_subblock_index, subblock_index);
        }
        orcid_to_subblock_indices.insert(orcid.clone(), unique_subblock_indices);
    }

    let mut components_by_root: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut component_roots = Vec::<usize>::new();
    let mut seen_roots = HashSet::<usize>::new();
    for subblock_index in 0..subblock_ids.len() {
        let root = find_orcid_subblock_root(&mut parent, subblock_index);
        components_by_root
            .entry(root)
            .or_default()
            .push(subblock_index);
        if seen_roots.insert(root) {
            component_roots.push(root);
        }
    }

    let mut skipped_orcid_counts_by_root: HashMap<usize, Vec<(String, usize)>> = HashMap::new();
    for orcid in orcid_order.iter() {
        let Some(unique_subblock_indices) = orcid_to_subblock_indices.get(orcid) else {
            continue;
        };
        let root = find_orcid_subblock_root(&mut parent, unique_subblock_indices[0]);
        let total_orcid_sig_count = orcid_to_sig_ids.get(orcid).map_or(0usize, Vec::len);
        skipped_orcid_counts_by_root
            .entry(root)
            .or_default()
            .push((orcid.clone(), total_orcid_sig_count));
    }

    let mut merge_actions = Vec::<(String, Vec<String>, Vec<String>)>::new();
    for root in component_roots {
        let Some(component_indices) = components_by_root.get(&root) else {
            continue;
        };
        if component_indices.len() <= 1 {
            continue;
        }
        let mut unique_subblock_ids: Vec<String> = component_indices
            .iter()
            .map(|index| subblock_ids[*index].clone())
            .collect();
        unique_subblock_ids.sort_by(|left, right| {
            let left_score = left.matches("specter").count() * 10 + left.matches('|').count();
            let right_score = right.matches("specter").count() * 10 + right.matches('|').count();
            left_score.cmp(&right_score).then_with(|| left.cmp(right))
        });

        let total_subblock_sig_count: usize = unique_subblock_ids
            .iter()
            .map(|subblock_id| output.get(subblock_id).map_or(0, Vec::len))
            .sum();
        if total_subblock_sig_count > maximum_size {
            if let Some(skipped_orcid_counts) = skipped_orcid_counts_by_root.get(&root) {
                for (_orcid, total_orcid_sig_count) in skipped_orcid_counts {
                    telemetry.orcid_merge_skipped_due_to_capacity_count += 1;
                    telemetry.orcid_merge_skipped_due_to_capacity_signature_count +=
                        *total_orcid_sig_count;
                }
            }
            continue;
        }

        let key_of_keys = unique_subblock_ids.join(", ");
        let mut signature_ids_stacked = Vec::<String>::with_capacity(total_subblock_sig_count);
        for subblock_id in unique_subblock_ids.iter() {
            if let Some(values) = output.get(subblock_id) {
                signature_ids_stacked.extend(values.iter().cloned());
            }
        }
        merge_actions.push((key_of_keys, signature_ids_stacked, unique_subblock_ids));
    }

    for (key_of_keys, signature_ids_stacked, unique_subblock_ids) in merge_actions {
        for subblock_id in unique_subblock_ids.iter() {
            output.remove(subblock_id);
        }
        output.insert(key_of_keys, signature_ids_stacked);
    }
    Ok(())
}

pub(crate) fn extract_string_vec_entries(
    obj: &Bound<'_, PyAny>,
) -> PyResult<Vec<(String, Vec<String>)>> {
    let dict = obj.downcast::<PyDict>()?;
    let mut out = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        out.push((key.extract()?, value.extract()?));
    }
    Ok(out)
}

pub(crate) fn make_subblocks_with_telemetry_from_rows_native_graph(
    py: Python<'_>,
    paths: &Bound<'_, PyAny>,
    rows: Vec<SubblockingSignatureRow>,
    maximum_size: usize,
    first_k_letter_counts_sorted: HashMap<String, HashMap<String, f64>>,
    graph_config: NativeGraphSubblockingConfig,
    random_seed: u64,
    use_orcid_subblocking: bool,
) -> PyResult<(HashMap<String, Vec<String>>, Py<PyDict>)> {
    if maximum_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "maximum_size must be positive",
        ));
    }

    let signature_ids: Vec<String> = rows.iter().map(|row| row.signature_id.clone()).collect();
    let row_by_signature_id: HashMap<String, SubblockingSignatureRow> = rows
        .iter()
        .map(|row| (row.signature_id.clone(), row.clone()))
        .collect();
    if row_by_signature_id.len() != rows.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "subblocking signature rows must contain unique signature_id values",
        ));
    }
    let single_letter_flags: Vec<bool> = rows.iter().map(|row| py_len(&row.first) <= 1).collect();
    let single_letter_count = single_letter_flags
        .iter()
        .filter(|is_single_letter| **is_single_letter)
        .count();
    let mut telemetry = SubblockingTelemetry {
        maximum_size,
        input_signature_count: rows.len(),
        single_letter_first_name_signature_count: single_letter_count,
        multi_letter_first_name_signature_count: rows.len().saturating_sub(single_letter_count),
        orcid_subblocking_enabled: use_orcid_subblocking,
        ..SubblockingTelemetry::default()
    };

    let first_letter = rows
        .iter()
        .find_map(|row| row.first.chars().next().map(|ch| ch.to_string()))
        .unwrap_or_else(|| "?".to_string());

    let mut multi_first_names = Vec::<String>::new();
    let mut multi_signature_ids = Vec::<String>::new();
    let mut single_middle_names = Vec::<String>::new();
    let mut single_signature_ids = Vec::<String>::new();
    for (row, is_single_letter) in rows.iter().zip(single_letter_flags.iter()) {
        if *is_single_letter {
            single_middle_names.push(row.middle.clone());
            single_signature_ids.push(row.signature_id.clone());
        } else {
            multi_first_names.push(row.first.clone());
            multi_signature_ids.push(row.signature_id.clone());
        }
    }

    let (mut output, output_cant_subdivide) =
        subdivide_helper_rust(multi_first_names, multi_signature_ids, maximum_size, 2);
    telemetry.first_name_dead_end_block_count = output_cant_subdivide.len();
    telemetry.first_name_dead_end_signature_count = output_cant_subdivide
        .iter()
        .map(|(_key, values)| values.len())
        .sum();

    let mut output_for_specter = OrderedSubblocks::default();
    for (key, sig_ids_loop) in output_cant_subdivide.entries {
        let middle_names_loop: Vec<String> = sig_ids_loop
            .iter()
            .filter_map(|signature_id| row_by_signature_id.get(signature_id))
            .map(|row| row.middle.clone())
            .collect();
        let (output_loop, output_cant_subdivide_loop) =
            subdivide_helper_rust(middle_names_loop, sig_ids_loop, maximum_size, 1);
        for (key_loop, values) in output_loop.entries {
            output.insert(format!("{key}|middle={key_loop}"), values);
        }
        for (key_loop, values) in output_cant_subdivide_loop.entries {
            output_for_specter.insert(format!("{key}|middle={key_loop}"), values);
        }
    }

    if single_signature_ids.len() <= maximum_size {
        if !single_signature_ids.is_empty() {
            output.insert(first_letter.clone(), single_signature_ids);
        }
    } else {
        let (output_single_letter_first_name, output_cant_subdivide_single_letter_first_name) =
            subdivide_helper_rust(single_middle_names, single_signature_ids, maximum_size, 1);
        for (key, values) in output_single_letter_first_name.entries {
            output.insert(format!("{first_letter}|middle={key}"), values);
        }
        for (key, values) in output_cant_subdivide_single_letter_first_name.entries {
            output_for_specter.insert(format!("{first_letter}|middle={key}"), values);
        }
    }

    telemetry.specter_fallback_candidate_block_count = output_for_specter.len();
    telemetry.specter_fallback_candidate_signature_count = output_for_specter
        .iter()
        .map(|(_key, values)| values.len())
        .sum();

    let fallback_signature_groups: Vec<Vec<String>> = output_for_specter
        .iter()
        .filter_map(|(_key, values)| {
            if values.len() > maximum_size {
                Some(values.clone())
            } else {
                None
            }
        })
        .collect();
    let mut graph_telemetry = NativeGraphTelemetry::default();
    let graph_store = if fallback_signature_groups.is_empty() {
        None
    } else {
        Some(build_native_graph_evidence_store(
            py,
            paths,
            &row_by_signature_id,
            &fallback_signature_groups,
            &mut graph_telemetry,
        )?)
    };

    for (key, sig_ids_loop) in output_for_specter.entries {
        if sig_ids_loop.len() <= maximum_size {
            telemetry.specter_non_invoked_candidate_block_count += 1;
            telemetry.specter_non_invoked_candidate_signature_count += sig_ids_loop.len();
            output.insert(key, sig_ids_loop);
        } else {
            telemetry.specter_invocation_count += 1;
            telemetry.specter_input_signature_count += sig_ids_loop.len();
            let store = graph_store.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "Native graph fallback evidence store was not loaded for an oversized group",
                )
            })?;
            let mut specter_clustering = native_graph_cluster(
                sig_ids_loop,
                store,
                maximum_size,
                &graph_config,
                random_seed,
                &mut graph_telemetry,
            )?;
            let mut keys: Vec<String> = specter_clustering.keys().cloned().collect();
            keys.sort_unstable();
            for key_loop in keys {
                let values = specter_clustering
                    .remove(&key_loop)
                    .expect("native graph subblock key exists");
                output.insert(format!("{key}|specter={key_loop}"), values);
            }
        }
    }

    let (pre_merge_specter_subblock_count, pre_merge_specter_signature_count) =
        specter_labeled_subblock_stats(&output);
    telemetry.pre_merge_subblock_count = output.len();
    telemetry.pre_merge_specter_labeled_subblock_count = pre_merge_specter_subblock_count;
    telemetry.pre_merge_specter_labeled_signature_count = pre_merge_specter_signature_count;

    merge_small_subblocks(&mut output, maximum_size, &first_k_letter_counts_sorted)?;

    if use_orcid_subblocking {
        apply_orcid_subblocking(
            &mut output,
            &row_by_signature_id,
            maximum_size,
            &mut telemetry,
        )?;
    }

    // Multiset equality, not set equality: a signature_id duplicated across
    // subblocks would pass a set check but bind to an arbitrary subblock in the
    // downstream ORCID merge, silently diverging from the Python implementation
    // (which asserts multiset equality).
    let mut input_counts: HashMap<String, usize> = HashMap::new();
    for signature_id in signature_ids.into_iter() {
        *input_counts.entry(signature_id).or_insert(0) += 1;
    }
    let mut output_counts: HashMap<String, usize> = HashMap::new();
    for (_key, values) in output.iter() {
        for signature_id in values.iter() {
            *output_counts.entry(signature_id.clone()).or_insert(0) += 1;
        }
    }
    if input_counts != output_counts {
        return Err(pyo3::exceptions::PyAssertionError::new_err(
            "Subblocking did not produce a complete partition",
        ));
    }

    let (final_specter_subblock_count, final_specter_signature_count) =
        specter_labeled_subblock_stats(&output);
    telemetry.final_subblock_count = output.len();
    telemetry.final_specter_labeled_subblock_count = final_specter_subblock_count;
    telemetry.final_specter_labeled_signature_count = final_specter_signature_count;
    let telemetry_dict = telemetry.to_dict(py)?;
    insert_native_graph_telemetry(py, telemetry_dict.bind(py), &graph_telemetry)?;
    Ok((output.to_hashmap(), telemetry_dict))
}

#[pyfunction]
#[pyo3(signature = (
    paths,
    signature_ids,
    maximum_size,
    first_k_letter_counts_sorted,
    graph_config = None,
    random_seed = 0,
    use_orcid_subblocking = true
))]
pub(crate) fn make_subblocks_with_telemetry_arrow_native_graph(
    py: Python<'_>,
    paths: &Bound<'_, PyAny>,
    signature_ids: Vec<String>,
    maximum_size: usize,
    first_k_letter_counts_sorted: HashMap<String, HashMap<String, f64>>,
    graph_config: Option<&Bound<'_, PyAny>>,
    random_seed: u64,
    use_orcid_subblocking: bool,
) -> PyResult<(HashMap<String, Vec<String>>, Py<PyDict>)> {
    let signatures_path = extract_path_mapping_string(paths, "signatures", true)?
        .expect("required signatures path exists");
    let signatures_index_path =
        extract_path_mapping_string(paths, "signatures_batch_index", false)?;
    let mut seen_signature_ids = HashSet::<String>::new();
    let mut requested_signature_ids = Vec::<String>::new();
    for signature_id in signature_ids {
        if seen_signature_ids.insert(signature_id.clone()) {
            requested_signature_ids.push(signature_id);
        }
    }
    let keep_signature_ids: HashSet<String> = requested_signature_ids.iter().cloned().collect();
    let (subblocking_rows, _read_stats) = read_subblocking_signature_rows_with_optional_index(
        &signatures_path,
        signatures_index_path.as_deref(),
        Some(&keep_signature_ids),
    )?;
    let missing_signature_ids: Vec<String> = requested_signature_ids
        .iter()
        .filter(|signature_id| !subblocking_rows.contains_key(*signature_id))
        .take(10)
        .cloned()
        .collect();
    if !missing_signature_ids.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "signatures Arrow is missing subblocking signature ids: {missing_signature_ids:?}"
        )));
    }

    let mut signature_rows: Vec<SubblockingSignatureRow> = requested_signature_ids
        .iter()
        .map(|signature_id| {
            subblocking_rows
                .get(signature_id)
                .expect("requested signature exists after missing check")
                .clone()
        })
        .collect();
    let text_module = py.import("s2and.text")?;
    let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
    let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
    for row in signature_rows.iter() {
        ensure_unidecode_for_text(&row.first, &mut unidecode_char_map)?;
        ensure_unidecode_for_text(&row.middle, &mut unidecode_char_map)?;
    }
    normalize_subblocking_signature_rows(&mut signature_rows, &name_prefixes, &unidecode_char_map);

    make_subblocks_with_telemetry_from_rows_native_graph(
        py,
        paths,
        signature_rows,
        maximum_size,
        first_k_letter_counts_sorted,
        NativeGraphSubblockingConfig::from_py(graph_config)?,
        random_seed,
        use_orcid_subblocking,
    )
}
