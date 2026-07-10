use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use std::collections::{hash_map::Entry, HashMap, HashSet};

use crate::arrow_batch_lookup::{read_indexed_arrow_batches, IndexedArrowReadStats};
use crate::name_counts::{NameCountsData, RawNameCountIndex, RawNameCountMaps};
use crate::orcid::normalize_orcid_owned;
use crate::raw_arrow::arrow_io::{
    arrow_column_index, arrow_optional_bool, arrow_optional_column_index,
    arrow_optional_f32_vector, arrow_optional_f64, arrow_optional_string_list, read_arrow_batches,
    ArrowI64Column, ArrowStringColumn,
};
use crate::{canonical_signature_pair_owned, RetrievalQueryData};

#[derive(Clone)]
pub(crate) struct RawArrowSignature {
    pub(crate) paper_id: String,
    pub(crate) author_first: String,
    pub(crate) author_middle: String,
    pub(crate) author_last: String,
    pub(crate) author_suffix: String,
    pub(crate) author_block: Option<String>,
    pub(crate) affiliations: Vec<String>,
    pub(crate) email: Option<String>,
    pub(crate) orcid: Option<String>,
    pub(crate) position: Option<i64>,
}

#[derive(Clone)]
pub(crate) struct RawArrowPaper {
    pub(crate) title: String,
    pub(crate) abstract_text: String,
    pub(crate) venue: String,
    pub(crate) journal_name: String,
    pub(crate) year: Option<i64>,
    pub(crate) predicted_language: Option<String>,
    pub(crate) is_reliable: Option<bool>,
    pub(crate) language_reliability: Option<f64>,
}

fn validate_stored_language_detection(
    paper_id: &str,
    predicted_language: Option<&str>,
    is_reliable: Option<bool>,
    language_reliability: Option<f64>,
) -> PyResult<()> {
    if let Some(reliability) = language_reliability {
        if !reliability.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "papers Arrow language_reliability must be finite for paper_id {paper_id:?}"
            )));
        }
        if !(0.0..=1.0).contains(&reliability) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "papers Arrow language_reliability must be in [0.0, 1.0] for paper_id {paper_id:?}"
            )));
        }
        if is_reliable == Some(false) && reliability != 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "papers Arrow language_reliability must be 0.0 when is_reliable is false for paper_id {paper_id:?}"
            )));
        }
    }
    if predicted_language.is_some() {
        if is_reliable.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "papers Arrow predicted_language requires is_reliable for paper_id {paper_id:?}"
            )));
        }
        if language_reliability.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "papers Arrow predicted_language requires language_reliability for paper_id {paper_id:?}"
            )));
        }
    }
    Ok(())
}

#[derive(Clone)]
pub(crate) struct RawArrowFeature {
    pub(crate) query: RetrievalQueryData,
    pub(crate) name_counts: Option<NameCountsData>,
    pub(crate) paper_author_count: usize,
    pub(crate) query_author: String,
}

pub(crate) struct RawArrowAuthorSignalData {
    pub(crate) paper_author_names: HashSet<String>,
    pub(crate) local10_author_names: HashSet<String>,
}

pub(crate) struct RawArrowSummarySignalData {
    pub(crate) name_counts_values: Vec<NameCountsData>,
    pub(crate) member_paper_author_names: Vec<HashSet<String>>,
    pub(crate) member_paper_author_counts: Vec<usize>,
    pub(crate) member_local10_author_names: Vec<HashSet<String>>,
    pub(crate) member_signature_ids: Vec<String>,
}

pub(crate) struct RawArrowNameCountRarityRow {
    pub(crate) last_name_count_min_rarity: f32,
    pub(crate) candidate_last_name_count_min_rarity: f32,
    pub(crate) candidate_last_first_name_count_min_rarity: f32,
    pub(crate) last_first_name_count_min_rarity: f32,
    pub(crate) first_prefix_x_last_first_name_count_min_rarity: f32,
}

pub(crate) struct RawArrowPaperEvidenceRow {
    pub(crate) paper_author_list_max_jaccard: f32,
    pub(crate) paper_author_list_max_containment: f32,
    pub(crate) paper_author_list_max_overlap_count: f32,
    pub(crate) local_author_window10_jaccard_max: f32,
    pub(crate) local_author_window10_overlap_count_max: f32,
    pub(crate) best_author_count_log_absdiff: f32,
}

pub(crate) fn read_raw_arrow_signatures_from_batches(
    path: &str,
    batches: Vec<RecordBatch>,
    keep_signature_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, RawArrowSignature>> {
    let mut out = HashMap::new();
    let mut seen_all_signature_ids: HashSet<String> = HashSet::new();
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
        let last_col = batch.column(arrow_column_index(&batch, "author_last", path)?);
        let last_values = ArrowStringColumn::from_string_array(last_col.as_ref(), "author_last")?;
        let suffix_col = batch.column(arrow_column_index(&batch, "author_suffix", path)?);
        let suffix_values =
            ArrowStringColumn::from_string_array(suffix_col.as_ref(), "author_suffix")?;
        let affiliations_col =
            batch.column(arrow_column_index(&batch, "author_affiliations", path)?);
        let orcid_col =
            arrow_optional_column_index(&batch, "author_orcid").map(|index| batch.column(index));
        let orcid_values = match orcid_col.as_ref() {
            Some(col) => Some(ArrowStringColumn::from_string_array(
                col.as_ref(),
                "author_orcid",
            )?),
            None => None,
        };
        let position_col = batch.column(arrow_column_index(&batch, "author_position", path)?);
        let position_values =
            ArrowI64Column::from_i64_array(position_col.as_ref(), "author_position")?;
        let author_block_col =
            arrow_optional_column_index(&batch, "author_block").map(|index| batch.column(index));
        let author_block_values = match author_block_col.as_ref() {
            Some(col) => Some(ArrowStringColumn::from_string_array(
                col.as_ref(),
                "author_block",
            )?),
            None => None,
        };
        let email_col =
            arrow_optional_column_index(&batch, "author_email").map(|index| batch.column(index));
        let email_values = match email_col.as_ref() {
            Some(col) => Some(ArrowStringColumn::from_string_array(
                col.as_ref(),
                "author_email",
            )?),
            None => None,
        };
        for row in 0..batch.num_rows() {
            let signature_id_value = signature_id_values.required_value(row, "signature_id")?;
            if signature_id_value.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "signatures Arrow cannot contain empty signature_id values",
                ));
            }
            // Detect duplicates regardless of the keep-filter so that an upstream
            // corruption is caught symmetrically (a filtered scan must not be more
            // permissive than a full scan).
            if !seen_all_signature_ids.insert(signature_id_value.as_ref().to_string()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "signatures Arrow contains duplicate signature_id: {:?}",
                    signature_id_value.as_ref()
                )));
            }
            if keep_signature_ids.map_or(false, |keep| !keep.contains(signature_id_value.as_ref()))
            {
                continue;
            }
            let signature_id = signature_id_value.into_owned();
            match out.entry(signature_id) {
                Entry::Occupied(_) => unreachable!(
                    "duplicate signature_id should be caught by the pre-filter check above"
                ),
                Entry::Vacant(entry) => {
                    let paper_id = paper_id_values
                        .required_value(row, "paper_id")?
                        .into_owned();
                    entry.insert(RawArrowSignature {
                        paper_id,
                        // These columns are legitimately empty for most rows
                        // (e.g. author_suffix is ~100% empty; empty surnames are
                        // valid per D6). The producer serializes empty as Arrow
                        // NULL (`_optional_str` maps ""/None -> None), so NULL is
                        // the normal "empty" representation, not a corruption
                        // signal. Coerce NULL -> "" to match Python's
                        // normalize_text(None) -> "" and keep Python<->Rust parity.
                        author_first: first_values.optional_owned(row).unwrap_or_default(),
                        author_middle: middle_values.optional_owned(row).unwrap_or_default(),
                        author_last: last_values.optional_owned(row).unwrap_or_default(),
                        author_suffix: suffix_values.optional_owned(row).unwrap_or_default(),
                        author_block: author_block_values
                            .as_ref()
                            .and_then(|col| col.optional_owned(row)),
                        affiliations: arrow_optional_string_list(
                            affiliations_col.as_ref(),
                            row,
                            "author_affiliations",
                        )?,
                        email: email_values
                            .as_ref()
                            .and_then(|col| col.optional_owned(row)),
                        orcid: orcid_values
                            .as_ref()
                            .and_then(|col| col.optional_owned(row))
                            .and_then(|value| normalize_orcid_owned(&value)),
                        position: position_values.optional_value(row, "author_position")?,
                    });
                }
            }
        }
    }
    Ok(out)
}

pub(crate) fn read_raw_arrow_papers_from_batches(
    path: &str,
    batches: Vec<RecordBatch>,
    keep_paper_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, RawArrowPaper>> {
    let mut out = HashMap::new();
    let mut seen_all_paper_ids = HashSet::<String>::new();
    for batch in batches {
        let paper_id_col = batch.column(arrow_column_index(&batch, "paper_id", path)?);
        let paper_id_values =
            ArrowStringColumn::from_string_array(paper_id_col.as_ref(), "paper_id")?;
        let title_col = batch.column(arrow_column_index(&batch, "title", path)?);
        let title_values = ArrowStringColumn::from_string_array(title_col.as_ref(), "title")?;
        let abstract_col =
            arrow_optional_column_index(&batch, "abstract").map(|index| batch.column(index));
        let abstract_values = match abstract_col.as_ref() {
            Some(col) => Some(ArrowStringColumn::from_string_array(
                col.as_ref(),
                "abstract",
            )?),
            None => None,
        };
        let venue_col = batch.column(arrow_column_index(&batch, "venue", path)?);
        let venue_values = ArrowStringColumn::from_string_array(venue_col.as_ref(), "venue")?;
        let journal_col = batch.column(arrow_column_index(&batch, "journal_name", path)?);
        let journal_values =
            ArrowStringColumn::from_string_array(journal_col.as_ref(), "journal_name")?;
        let year_col = arrow_optional_column_index(&batch, "year").map(|index| batch.column(index));
        let year_values = match year_col.as_ref() {
            Some(col) => Some(ArrowI64Column::from_i64_array(col.as_ref(), "year")?),
            None => None,
        };
        let predicted_language_col = arrow_optional_column_index(&batch, "predicted_language")
            .map(|index| batch.column(index));
        let predicted_language_values = match predicted_language_col.as_ref() {
            Some(col) => Some(ArrowStringColumn::from_string_array(
                col.as_ref(),
                "predicted_language",
            )?),
            None => None,
        };
        let is_reliable_col = arrow_optional_column_index(&batch, "is_reliable")
            .map(|index| batch.column(index).as_ref());
        let language_reliability_col = arrow_optional_column_index(&batch, "language_reliability")
            .map(|index| batch.column(index).as_ref());
        for row in 0..batch.num_rows() {
            let paper_id_value = paper_id_values.required_value(row, "paper_id")?;
            if paper_id_value.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "papers Arrow cannot contain empty paper_id values",
                ));
            }
            // Detect duplicates regardless of the keep-filter so that an upstream
            // corruption is caught symmetrically (a filtered scan must not be more
            // permissive than a full scan).
            if !seen_all_paper_ids.insert(paper_id_value.as_ref().to_string()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "papers Arrow contains duplicate paper_id: {:?}",
                    paper_id_value.as_ref()
                )));
            }
            if keep_paper_ids.map_or(false, |keep| !keep.contains(paper_id_value.as_ref())) {
                continue;
            }
            let paper_id = paper_id_value.into_owned();
            let predicted_language = predicted_language_values
                .as_ref()
                .and_then(|col| col.optional_owned(row));
            let is_reliable = match is_reliable_col {
                Some(col) => arrow_optional_bool(col, row, "is_reliable")?,
                None => None,
            };
            let language_reliability = match language_reliability_col {
                Some(col) => arrow_optional_f64(col, row, "language_reliability")?,
                None => None,
            };
            validate_stored_language_detection(
                &paper_id,
                predicted_language.as_deref(),
                is_reliable,
                language_reliability,
            )?;
            match out.entry(paper_id) {
                Entry::Occupied(_) => unreachable!(
                    "duplicate paper_id should be caught by the pre-filter check above"
                ),
                Entry::Vacant(entry) => {
                    entry.insert(RawArrowPaper {
                        // title/venue/journal_name are serialized with the same
                        // ""/None -> NULL producer convention, so NULL means
                        // empty. Coerce NULL -> "" to match Python parity (see
                        // the signature author-name note above).
                        title: title_values.optional_owned(row).unwrap_or_default(),
                        abstract_text: abstract_values
                            .as_ref()
                            .and_then(|col| col.optional_owned(row))
                            .unwrap_or_default(),
                        venue: venue_values.optional_owned(row).unwrap_or_default(),
                        journal_name: journal_values.optional_owned(row).unwrap_or_default(),
                        year: match year_values.as_ref() {
                            Some(values) => values.optional_value(row, "year")?,
                            None => None,
                        },
                        predicted_language,
                        is_reliable,
                        language_reliability,
                    });
                }
            }
        }
    }
    Ok(out)
}

pub(crate) fn read_raw_arrow_paper_authors_from_batches(
    path: &str,
    batches: Vec<RecordBatch>,
    keep_paper_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, Vec<(i64, String)>>> {
    let mut out: HashMap<String, Vec<(i64, String)>> = HashMap::new();
    // Kept rows are checked after sorting their existing output vectors. Only
    // index false-positive rows need compact position storage, avoiding a
    // second per-row structure on full and exact-index reads.
    let mut excluded_positions_by_paper = HashMap::<String, Vec<i64>>::new();
    for batch in batches {
        let paper_id_col = batch.column(arrow_column_index(&batch, "paper_id", path)?);
        let paper_id_values =
            ArrowStringColumn::from_string_array(paper_id_col.as_ref(), "paper_id")?;
        let position_col = batch.column(arrow_column_index(&batch, "position", path)?);
        let position_values = ArrowI64Column::from_i64_array(position_col.as_ref(), "position")?;
        let author_name_col = batch.column(arrow_column_index(&batch, "author_name", path)?);
        let author_name_values =
            ArrowStringColumn::from_string_array(author_name_col.as_ref(), "author_name")?;
        for row in 0..batch.num_rows() {
            let paper_id_value = paper_id_values.required_value(row, "paper_id")?;
            if paper_id_value.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "paper_authors Arrow cannot contain empty paper_id values",
                ));
            }
            let position = position_values.required_value(row, "position")?;
            let author_name = author_name_values
                .required_value(row, "author_name")?
                .into_owned();
            if author_name.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "paper_authors Arrow cannot contain empty author_name values",
                ));
            }
            if keep_paper_ids.map_or(false, |keep| !keep.contains(paper_id_value.as_ref())) {
                excluded_positions_by_paper
                    .entry(paper_id_value.to_string())
                    .or_default()
                    .push(position);
                continue;
            }
            let paper_id = paper_id_value.into_owned();
            out.entry(paper_id)
                .or_default()
                .push((position, author_name));
        }
    }
    for (paper_id, positions) in excluded_positions_by_paper.iter_mut() {
        positions.sort_unstable();
        if let Some(pair) = positions.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "paper_authors Arrow contains duplicate (paper_id, position): ({:?}, {})",
                paper_id, pair[0]
            )));
        }
    }
    drop(excluded_positions_by_paper);
    for (paper_id, authors) in out.iter_mut() {
        authors.sort_by_key(|(position, _name)| *position);
        if let Some(pair) = authors.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "paper_authors Arrow contains duplicate (paper_id, position): ({:?}, {})",
                paper_id, pair[0].0
            )));
        }
    }
    Ok(out)
}

pub(crate) fn read_raw_arrow_cluster_seeds(
    path: &str,
) -> PyResult<(Vec<String>, HashMap<String, Vec<String>>)> {
    let mut component_order = Vec::new();
    let mut members_by_component: HashMap<String, Vec<String>> = HashMap::new();
    let mut component_by_signature_id = HashMap::<String, String>::new();
    for batch in read_arrow_batches(path)? {
        let signature_id_col = batch.column(arrow_column_index(&batch, "signature_id", path)?);
        let signature_id_values =
            ArrowStringColumn::from_string_array(signature_id_col.as_ref(), "signature_id")?;
        let cluster_id_col = batch.column(arrow_column_index(&batch, "cluster_id", path)?);
        let cluster_id_values =
            ArrowStringColumn::from_string_array(cluster_id_col.as_ref(), "cluster_id")?;
        for row in 0..batch.num_rows() {
            let signature_id = signature_id_values
                .required_value(row, "signature_id")?
                .into_owned();
            let component_key = cluster_id_values
                .required_value(row, "cluster_id")?
                .into_owned();
            if signature_id.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "cluster_seeds Arrow cannot contain empty signature_id values",
                ));
            }
            if component_key.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "cluster_seeds Arrow cannot contain empty cluster_id values: {signature_id:?}"
                )));
            }
            if let Some(existing_component_key) = component_by_signature_id.get(&signature_id) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "cluster_seeds Arrow contains duplicate signature_id {signature_id:?}: \
                     {existing_component_key:?} and {component_key:?}"
                )));
            }
            component_by_signature_id.insert(signature_id.clone(), component_key.clone());
            if !members_by_component.contains_key(&component_key) {
                component_order.push(component_key.clone());
            }
            members_by_component
                .entry(component_key)
                .or_default()
                .push(signature_id);
        }
    }
    Ok((component_order, members_by_component))
}

pub(crate) fn read_raw_arrow_cluster_seed_disallows(
    path: &str,
) -> PyResult<HashSet<(String, String)>> {
    let mut out = HashSet::new();
    for batch in read_arrow_batches(path)? {
        let left_col = batch.column(arrow_column_index(&batch, "signature_id_1", path)?);
        let right_col = batch.column(arrow_column_index(&batch, "signature_id_2", path)?);
        let left_values =
            ArrowStringColumn::from_string_array(left_col.as_ref(), "signature_id_1")?;
        let right_values =
            ArrowStringColumn::from_string_array(right_col.as_ref(), "signature_id_2")?;
        for row in 0..batch.num_rows() {
            let left = left_values
                .required_value(row, "signature_id_1")?
                .into_owned();
            let right = right_values
                .required_value(row, "signature_id_2")?
                .into_owned();
            if left.is_empty() || right.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "cluster_seed_disallows cannot contain empty signature_id values",
                ));
            }
            if left == right {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "cluster_seed_disallows contains a self-pair for signature_id={left:?}"
                )));
            }
            let pair = canonical_signature_pair_owned(left, right);
            if !out.insert(pair.clone()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "cluster_seed_disallows contains duplicate pair: {pair:?}"
                )));
            }
        }
    }
    Ok(out)
}

#[derive(Clone)]
pub(crate) struct RawArrowQuerySignatureRequest {
    pub(crate) signature_id: String,
    pub(crate) query_view: String,
    pub(crate) query_author: String,
}

pub(crate) fn validate_raw_arrow_query_view(value: &str) -> PyResult<()> {
    if value == "auto" || value == "full" || value == "initial_only" {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "query_signatures Arrow contains unknown query_view: {value:?}"
        )))
    }
}

pub(crate) fn read_raw_arrow_query_signatures(
    path: &str,
) -> PyResult<Vec<RawArrowQuerySignatureRequest>> {
    let mut rows = Vec::<RawArrowQuerySignatureRequest>::new();
    let mut seen_signature_ids = HashSet::<String>::new();
    for batch in read_arrow_batches(path)? {
        let signature_id_col = batch.column(arrow_column_index(&batch, "signature_id", path)?);
        let signature_id_values =
            ArrowStringColumn::from_string_array(signature_id_col.as_ref(), "signature_id")?;
        let query_view_col = batch.column(arrow_column_index(&batch, "query_view", path)?);
        let query_view_values =
            ArrowStringColumn::from_string_array(query_view_col.as_ref(), "query_view")?;
        let query_author_col = batch.column(arrow_column_index(&batch, "query_author", path)?);
        let query_author_values =
            ArrowStringColumn::from_string_array(query_author_col.as_ref(), "query_author")?;
        for row in 0..batch.num_rows() {
            let signature_id = signature_id_values
                .required_value(row, "signature_id")?
                .into_owned();
            let query_view = query_view_values
                .required_value(row, "query_view")?
                .into_owned();
            let query_author = query_author_values
                .required_value(row, "query_author")?
                .into_owned();
            if signature_id.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "query_signatures Arrow cannot contain empty signature_id values",
                ));
            }
            if query_view.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query_signatures Arrow cannot contain empty query_view values: {signature_id:?}"
                )));
            }
            validate_raw_arrow_query_view(&query_view)?;
            if !seen_signature_ids.insert(signature_id.clone()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query_signatures Arrow contains duplicate signature_id: {signature_id:?}"
                )));
            }
            rows.push(RawArrowQuerySignatureRequest {
                signature_id,
                query_view,
                query_author,
            });
        }
    }
    Ok(rows)
}

pub(crate) fn read_raw_arrow_specter_from_batches(
    path: &str,
    batches: Vec<RecordBatch>,
    keep_paper_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, Vec<f32>>> {
    let mut out = HashMap::new();
    let mut seen_paper_ids = HashSet::<String>::new();
    for batch in batches {
        let paper_id_col = batch.column(arrow_column_index(&batch, "paper_id", path)?);
        let paper_id_values =
            ArrowStringColumn::from_string_array(paper_id_col.as_ref(), "paper_id")?;
        let embedding_col = batch.column(arrow_column_index(&batch, "embedding", path)?);
        for row in 0..batch.num_rows() {
            let paper_id_value = paper_id_values.required_value(row, "paper_id")?;
            if paper_id_value.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "specter Arrow cannot contain empty paper_id values",
                ));
            }
            // Detect duplicates regardless of the keep-filter so that an upstream
            // corruption is caught symmetrically (a filtered scan must not be more
            // permissive than a full scan).
            if !seen_paper_ids.insert(paper_id_value.as_ref().to_string()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "specter Arrow contains duplicate paper_id: {:?}",
                    paper_id_value.as_ref()
                )));
            }
            if keep_paper_ids.map_or(false, |keep| !keep.contains(paper_id_value.as_ref())) {
                continue;
            }
            let paper_id = paper_id_value.into_owned();
            let vector = arrow_optional_f32_vector(embedding_col.as_ref(), row, "embedding")?
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "specter Arrow cannot contain null embedding values: {paper_id:?}"
                    ))
                })?;
            if vector.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "specter Arrow cannot contain zero-dimension embedding values: {paper_id:?}"
                )));
            }
            out.insert(paper_id, vector);
        }
    }
    Ok(out)
}

pub(crate) fn read_raw_arrow_with_optional_index<T, F>(
    path: &str,
    index_path: Option<&str>,
    key_column: &str,
    keep_ids: Option<&HashSet<String>>,
    read_from_batches: F,
) -> PyResult<(T, IndexedArrowReadStats)>
where
    F: Fn(&str, Vec<RecordBatch>, Option<&HashSet<String>>) -> PyResult<T>,
{
    if let (Some(index_path), Some(keep_ids)) = (index_path, keep_ids) {
        let (batches, stats) = read_indexed_arrow_batches(path, index_path, key_column, keep_ids)?;
        return Ok((read_from_batches(path, batches, Some(keep_ids))?, stats));
    }
    if keep_ids.is_some() && index_path.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Refusing filtered full scan of Arrow IPC file '{path}' without a batch lookup index for key column \
             '{key_column}'. Provide the matching *_batch_index path."
        )));
    }
    let batches = read_arrow_batches(path)?;
    let stats = IndexedArrowReadStats {
        batches_read: batches.len(),
        rows_scanned: batches.iter().map(|batch| batch.num_rows()).sum(),
    };
    let loaded = read_from_batches(path, batches, keep_ids)?;
    Ok((loaded, stats))
}

pub(crate) fn read_raw_arrow_signatures_with_optional_index(
    path: &str,
    index_path: Option<&str>,
    keep_signature_ids: Option<&HashSet<String>>,
) -> PyResult<(HashMap<String, RawArrowSignature>, IndexedArrowReadStats)> {
    read_raw_arrow_with_optional_index(
        path,
        index_path,
        "signature_id",
        keep_signature_ids,
        read_raw_arrow_signatures_from_batches,
    )
}

pub(crate) fn read_raw_arrow_papers_with_optional_index(
    path: &str,
    index_path: Option<&str>,
    keep_paper_ids: &HashSet<String>,
) -> PyResult<(HashMap<String, RawArrowPaper>, IndexedArrowReadStats)> {
    read_raw_arrow_with_optional_index(
        path,
        index_path,
        "paper_id",
        Some(keep_paper_ids),
        read_raw_arrow_papers_from_batches,
    )
}

pub(crate) fn read_raw_arrow_paper_authors_with_optional_index(
    path: &str,
    index_path: Option<&str>,
    keep_paper_ids: &HashSet<String>,
) -> PyResult<(HashMap<String, Vec<(i64, String)>>, IndexedArrowReadStats)> {
    read_raw_arrow_with_optional_index(
        path,
        index_path,
        "paper_id",
        Some(keep_paper_ids),
        read_raw_arrow_paper_authors_from_batches,
    )
}

pub(crate) fn read_raw_arrow_specter_with_optional_index(
    path: &str,
    index_path: Option<&str>,
    keep_paper_ids: &HashSet<String>,
) -> PyResult<(HashMap<String, Vec<f32>>, IndexedArrowReadStats)> {
    read_raw_arrow_with_optional_index(
        path,
        index_path,
        "paper_id",
        Some(keep_paper_ids),
        read_raw_arrow_specter_from_batches,
    )
}

pub(crate) fn read_raw_name_counts_index(path: &str) -> PyResult<RawNameCountMaps> {
    Ok(RawNameCountMaps::from_index(
        RawNameCountIndex::open_fully_validated(path)?,
    ))
}

#[cfg(test)]
mod optional_orcid_tests {
    use super::read_raw_arrow_signatures_from_batches;
    use arrow::array::{new_null_array, ArrayRef, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    #[test]
    fn signatures_without_orcid_column_read_as_none() {
        let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        let schema = Arc::new(Schema::new(vec![
            Field::new("signature_id", DataType::Utf8, false),
            Field::new("paper_id", DataType::Utf8, false),
            Field::new("author_first", DataType::Utf8, true),
            Field::new("author_middle", DataType::Utf8, true),
            Field::new("author_last", DataType::Utf8, true),
            Field::new("author_suffix", DataType::Utf8, true),
            Field::new("author_affiliations", list_type.clone(), true),
            Field::new("author_position", DataType::Int64, true),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec!["s1"])),
            Arc::new(StringArray::from(vec!["p1"])),
            Arc::new(StringArray::from(vec!["Ada"])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec!["Lovelace"])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            new_null_array(&list_type, 1),
            Arc::new(Int64Array::from(vec![0])),
        ];
        let batch = RecordBatch::try_new(schema, columns).expect("valid record batch");

        let signatures = read_raw_arrow_signatures_from_batches("<test>", vec![batch], None)
            .expect("author_orcid is optional");

        assert_eq!(signatures.get("s1").expect("s1 present").orcid, None);
    }
}

#[cfg(test)]
mod null_required_string_tests {
    use super::read_raw_arrow_papers_from_batches;
    use arrow::array::{ArrayRef, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    #[test]
    fn null_paper_strings_read_as_empty_not_error() {
        // Regression: the Arrow producer serializes an empty title/venue/
        // journal_name as Arrow NULL (via `_optional_str`, which maps ""/None ->
        // None). NULL is therefore the normal "empty" representation, not a
        // corruption signal, so the reader must coerce NULL -> "" (matching
        // Python's normalize_text(None) -> "") rather than raising. A prior
        // `required_value` change made this path raise on the ~100%-NULL
        // author_suffix / frequently-NULL venue+journal columns of every real
        // bundle; existing fixtures hand-wrote "" and missed it.
        let schema = Arc::new(Schema::new(vec![
            Field::new("paper_id", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, true),
            Field::new("venue", DataType::Utf8, true),
            Field::new("journal_name", DataType::Utf8, true),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec![Some("p1")])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
        ];
        let batch = RecordBatch::try_new(schema, columns).expect("valid record batch");

        let papers = read_raw_arrow_papers_from_batches("<test>", vec![batch], None)
            .expect("NULL required-string columns must coerce to \"\", not error");
        let paper = papers.get("p1").expect("p1 present");
        assert_eq!(paper.title, "");
        assert_eq!(paper.venue, "");
        assert_eq!(paper.journal_name, "");
    }
}

#[cfg(test)]
mod language_reliability_tests {
    use super::read_raw_arrow_papers_from_batches;
    use arrow::array::{ArrayRef, BooleanArray, Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use pyo3::types::PyAnyMethods;
    use pyo3::{PyErr, Python};
    use std::sync::Arc;

    fn py_err_message(err: PyErr) -> String {
        #[cfg(windows)]
        if let Some(python_home) = option_env!("S2AND_RUST_PYTHONHOME") {
            std::env::set_var("PYTHONHOME", python_home);
        }
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            err.value(py)
                .str()
                .expect("PyErr value should stringify")
                .to_string()
        })
    }

    fn papers_batch(
        predicted_language: Option<&str>,
        is_reliable: Option<bool>,
        language_reliability: Option<f64>,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("paper_id", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, true),
            Field::new("venue", DataType::Utf8, true),
            Field::new("journal_name", DataType::Utf8, true),
            Field::new("predicted_language", DataType::Utf8, true),
            Field::new("is_reliable", DataType::Boolean, true),
            Field::new("language_reliability", DataType::Float64, true),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec![Some("p1")])),
            Arc::new(StringArray::from(vec![Some("A title")])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![predicted_language])),
            Arc::new(BooleanArray::from(vec![is_reliable])),
            Arc::new(Float64Array::from(vec![language_reliability])),
        ];
        RecordBatch::try_new(schema, columns).expect("valid language-reliability batch")
    }

    #[test]
    fn malformed_language_reliability_is_rejected() {
        let cases = [
            (Some(true), Some(f64::NAN), "must be finite"),
            (Some(true), Some(f64::INFINITY), "must be finite"),
            (Some(true), Some(-0.01), "must be in [0.0, 1.0]"),
            (Some(true), Some(1.01), "must be in [0.0, 1.0]"),
            (
                Some(false),
                Some(0.25),
                "must be 0.0 when is_reliable is false",
            ),
        ];
        for (is_reliable, language_reliability, expected) in cases {
            let err = read_raw_arrow_papers_from_batches(
                "<test>",
                vec![papers_batch(Some("en"), is_reliable, language_reliability)],
                None,
            )
            .err()
            .expect("malformed language reliability must fail");
            let message = py_err_message(err);
            assert!(message.contains(expected), "{message}");
        }
    }

    #[test]
    fn stored_language_requires_boolean_and_continuous_reliability() {
        let cases = [
            (None, Some(0.75), "predicted_language requires is_reliable"),
            (
                Some(true),
                None,
                "predicted_language requires language_reliability",
            ),
        ];
        for (is_reliable, language_reliability, expected) in cases {
            let err = read_raw_arrow_papers_from_batches(
                "<test>",
                vec![papers_batch(Some("en"), is_reliable, language_reliability)],
                None,
            )
            .err()
            .expect("partial stored language detection must fail");
            let message = py_err_message(err);
            assert!(message.contains(expected), "{message}");
        }
    }

    #[test]
    fn valid_reliable_and_unreliable_scores_are_preserved() {
        for (paper_id, is_reliable, language_reliability) in
            [("p1", true, 0.75), ("p1", false, 0.0)]
        {
            let papers = read_raw_arrow_papers_from_batches(
                "<test>",
                vec![papers_batch(
                    Some("en"),
                    Some(is_reliable),
                    Some(language_reliability),
                )],
                None,
            )
            .expect("valid language detection must read");
            let paper = papers.get(paper_id).expect("paper present");
            assert_eq!(paper.is_reliable, Some(is_reliable));
            assert_eq!(paper.language_reliability, Some(language_reliability));
        }
    }
}

#[cfg(test)]
mod filtered_duplicate_detection_tests {
    use super::{
        read_raw_arrow_paper_authors_from_batches, read_raw_arrow_papers_from_batches,
        read_raw_arrow_specter_from_batches,
    };
    use arrow::array::{ArrayRef, FixedSizeListArray, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Float32Type, Schema};
    use arrow::record_batch::RecordBatch;
    use pyo3::types::PyAnyMethods;
    use pyo3::{PyErr, Python};
    use std::collections::HashSet;
    use std::sync::Arc;

    fn py_err_message(err: PyErr) -> String {
        #[cfg(windows)]
        if let Some(python_home) = option_env!("S2AND_RUST_PYTHONHOME") {
            std::env::set_var("PYTHONHOME", python_home);
        }
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            err.value(py)
                .str()
                .expect("PyErr value should stringify")
                .to_string()
        })
    }

    fn papers_batch(paper_ids: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("paper_id", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, true),
            Field::new("venue", DataType::Utf8, true),
            Field::new("journal_name", DataType::Utf8, true),
        ]));
        let n = paper_ids.len();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(paper_ids.to_vec())),
            Arc::new(StringArray::from(vec![None::<&str>; n])),
            Arc::new(StringArray::from(vec![None::<&str>; n])),
            Arc::new(StringArray::from(vec![None::<&str>; n])),
        ];
        RecordBatch::try_new(schema, columns).expect("valid record batch")
    }

    fn specter_batch(paper_ids: &[&str]) -> RecordBatch {
        let embedding_field = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("paper_id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(embedding_field, 1),
                true,
            ),
        ]));
        let embeddings = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            paper_ids.iter().map(|_| Some(vec![Some(0.5f32)])),
            1,
        );
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(paper_ids.to_vec())),
            Arc::new(embeddings),
        ];
        RecordBatch::try_new(schema, columns).expect("valid record batch")
    }

    fn paper_authors_batch(
        paper_ids: &[&str],
        positions: &[i64],
        author_names: &[&str],
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("paper_id", DataType::Utf8, false),
            Field::new("position", DataType::Int64, false),
            Field::new("author_name", DataType::Utf8, false),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(paper_ids.to_vec())),
            Arc::new(Int64Array::from(positions.to_vec())),
            Arc::new(StringArray::from(author_names.to_vec())),
        ];
        RecordBatch::try_new(schema, columns).expect("valid record batch")
    }

    fn keep(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|id| id.to_string()).collect()
    }

    // A duplicate id whose copies are all excluded by the keep-filter must still
    // fail: a filtered scan must not be more permissive than a full scan.
    #[test]
    fn papers_duplicate_outside_keep_filter_is_rejected() {
        let batch = papers_batch(&["p1", "p1", "p2"]);
        let keep_ids = keep(&["p2"]);
        let err = read_raw_arrow_papers_from_batches("<test>", vec![batch], Some(&keep_ids))
            .err()
            .expect("duplicate paper_id must be detected even when filtered out");
        let message = py_err_message(err);
        assert!(message.contains("duplicate paper_id"), "{message}");
    }

    #[test]
    fn papers_duplicate_inside_keep_filter_is_rejected() {
        let batch = papers_batch(&["p1", "p1"]);
        let keep_ids = keep(&["p1"]);
        let err = read_raw_arrow_papers_from_batches("<test>", vec![batch], Some(&keep_ids))
            .err()
            .expect("duplicate paper_id must be detected inside the keep-filter");
        let message = py_err_message(err);
        assert!(message.contains("duplicate paper_id"), "{message}");
    }

    #[test]
    fn papers_unique_ids_with_keep_filter_read_ok() {
        let batch = papers_batch(&["p1", "p2"]);
        let keep_ids = keep(&["p2"]);
        let papers = read_raw_arrow_papers_from_batches("<test>", vec![batch], Some(&keep_ids))
            .expect("unique ids must read");
        assert_eq!(papers.len(), 1);
        assert!(papers.contains_key("p2"));
    }

    #[test]
    fn specter_duplicate_outside_keep_filter_is_rejected() {
        let batch = specter_batch(&["p1", "p1", "p2"]);
        let keep_ids = keep(&["p2"]);
        let err = read_raw_arrow_specter_from_batches("<test>", vec![batch], Some(&keep_ids))
            .err()
            .expect("duplicate paper_id must be detected even when filtered out");
        let message = py_err_message(err);
        assert!(message.contains("duplicate paper_id"), "{message}");
    }

    #[test]
    fn specter_unique_ids_with_keep_filter_read_ok() {
        let batch = specter_batch(&["p1", "p2"]);
        let keep_ids = keep(&["p2"]);
        let specter = read_raw_arrow_specter_from_batches("<test>", vec![batch], Some(&keep_ids))
            .expect("unique ids must read");
        assert_eq!(specter.len(), 1);
        assert!(specter.contains_key("p2"));
    }

    #[test]
    fn paper_author_duplicate_position_outside_keep_filter_is_rejected() {
        let batch = paper_authors_batch(&["p1", "p1", "p2"], &[0, 0, 0], &["A", "B", "C"]);
        let keep_ids = keep(&["p2"]);
        let err = read_raw_arrow_paper_authors_from_batches("<test>", vec![batch], Some(&keep_ids))
            .err()
            .expect("duplicate paper-author position must be detected before filtering");
        let message = py_err_message(err);
        assert!(
            message.contains("duplicate (paper_id, position)"),
            "{message}"
        );
    }

    #[test]
    fn paper_author_empty_name_is_rejected() {
        let batch = paper_authors_batch(&["p1"], &[0], &[""]);
        let err = read_raw_arrow_paper_authors_from_batches("<test>", vec![batch], None)
            .err()
            .expect("empty paper-author names must fail");
        let message = py_err_message(err);
        assert!(message.contains("empty author_name"), "{message}");
    }
}
