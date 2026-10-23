use crate::{HashMap, HashSet};
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use crate::name_counts::NameCountsData;
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
    /// Whether the abstract is non-empty. Features use only that, so readers
    /// skip copying every abstract.
    pub(crate) has_abstract: bool,
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
    match (predicted_language, is_reliable, language_reliability) {
        (None, None, None) => return Ok(()),
        (None, _, _) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "papers Arrow is_reliable and language_reliability require predicted_language for paper_id {paper_id:?}"
            )));
        }
        (Some(language), _, _) if language.trim().is_empty() => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "papers Arrow predicted_language must be nonempty for paper_id {paper_id:?}"
            )));
        }
        _ => {}
    }
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
    Ok(())
}

#[derive(Clone)]
pub(crate) struct RawArrowFeature {
    pub(crate) query: RetrievalQueryData,
    pub(crate) name_counts: Option<NameCountsData>,
    pub(crate) paper_author_count: usize,
    pub(crate) query_author: String,
}

/// Author-list evidence of one signature, as sorted distinct interned name ids.
pub(crate) struct RawArrowAuthorSignalData {
    /// Every nonempty normalized author name on the signature's paper.
    pub(crate) paper_author_names: Arc<[u32]>,
    /// Names of the other authors within ten positions of the signature.
    pub(crate) local10_author_names: Vec<u32>,
}

pub(crate) struct RawArrowSummarySignalData {
    pub(crate) name_counts_values: Vec<NameCountsData>,
    pub(crate) member_paper_author_names: Vec<Arc<[u32]>>,
    pub(crate) member_paper_author_counts: Vec<usize>,
    pub(crate) member_local10_author_names: Vec<Vec<u32>>,
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

/// Filtered reads that keep fewer ids stay serial. Their batches are mostly
/// skipped rows, and the parallel path's extra cross-batch checks made the many
/// small reads of full-block requests slower. Tests take the parallel path for
/// every multi-batch read.
const PARALLEL_READ_MIN_KEPT_IDS: usize = if cfg!(test) { 1 } else { 10_000 };

/// Whether a read of `batches` keeping `keep_ids` takes the parallel path.
fn read_in_parallel(batches: &[RecordBatch], keep_ids: Option<&HashSet<String>>) -> bool {
    batches.len() > 1 && keep_ids.map_or(true, |keep| keep.len() >= PARALLEL_READ_MIN_KEPT_IDS)
}

/// Reads batches in parallel with `read_batch`, one task per batch, and merges
/// the results; None when any batch fails or a key repeats across batches.
///
/// Callers then rerun their serial reader, which reports the first problem in
/// row order as before, so only valid input takes this path. A filtered read
/// must also reject keys repeated among the rows it skips.
fn read_keyed_batches_in_parallel<V: Send>(
    path: &str,
    batches: &[RecordBatch],
    keep_ids: Option<&HashSet<String>>,
    key_column: &str,
    read_batch: impl Fn(&RecordBatch) -> PyResult<HashMap<String, V>> + Sync,
) -> Option<HashMap<String, V>> {
    if !read_in_parallel(batches, keep_ids) {
        return None;
    }
    if keep_ids.is_some() && !scanned_keys_are_unique(path, batches, key_column) {
        return None;
    }
    let parts = batches
        .par_iter()
        .map(|batch| read_batch(batch).ok())
        .collect::<Option<Vec<_>>>()?;
    let mut out =
        HashMap::with_capacity_and_hasher(parts.iter().map(HashMap::len).sum(), Default::default());
    for part in parts {
        for (key, value) in part {
            if out.insert(key, value).is_some() {
                return None;
            }
        }
    }
    Some(out)
}

/// Whether every row's `key_column` value is present and distinct across batches.
fn scanned_keys_are_unique(path: &str, batches: &[RecordBatch], key_column: &str) -> bool {
    let keys = batches
        .par_iter()
        .map(|batch| -> PyResult<Vec<&str>> {
            let column = batch.column(arrow_column_index(batch, key_column, path)?);
            let values = ArrowStringColumn::from_string_array(column.as_ref(), key_column)?;
            (0..batch.num_rows())
                .map(|row| values.required_value(row, key_column))
                .collect()
        })
        .collect::<PyResult<Vec<_>>>();
    let Ok(keys) = keys else {
        return false;
    };
    let mut keys: Vec<&str> = keys.into_iter().flatten().collect();
    keys.par_sort_unstable();
    keys.windows(2).all(|pair| pair[0] != pair[1])
}

// Each reader below converts batches in parallel and falls back to its serial
// reader on any error, so results and error messages match a serial read.

pub(crate) fn read_raw_arrow_signatures_from_batches(
    path: &str,
    batches: Vec<RecordBatch>,
    keep_signature_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, RawArrowSignature>> {
    let parallel = read_keyed_batches_in_parallel(
        path,
        &batches,
        keep_signature_ids,
        "signature_id",
        |batch| read_signature_batches(path, std::slice::from_ref(batch), keep_signature_ids),
    );
    match parallel {
        Some(out) => Ok(out),
        None => read_signature_batches(path, &batches, keep_signature_ids),
    }
}

fn read_signature_batches(
    path: &str,
    batches: &[RecordBatch],
    keep_signature_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, RawArrowSignature>> {
    let mut out = HashMap::default();
    // Borrow ids directly from the retained Arrow batches so corruption checks
    // do not allocate one String for every index false-positive row.
    let mut seen_filtered_scan_ids = keep_signature_ids.map(|_| HashSet::<&str>::default());
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
            if let Some(seen_ids) = seen_filtered_scan_ids.as_mut() {
                if !seen_ids.insert(signature_id_value) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "signatures Arrow contains duplicate signature_id: {:?}",
                        signature_id_value
                    )));
                }
            }
            if keep_signature_ids.map_or(false, |keep| !keep.contains(signature_id_value)) {
                continue;
            }
            let signature_id = signature_id_value.to_owned();
            match out.entry(signature_id) {
                Entry::Occupied(entry) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "signatures Arrow contains duplicate signature_id: {:?}",
                        entry.key()
                    )))
                }
                Entry::Vacant(entry) => {
                    let paper_id = paper_id_values.required_value(row, "paper_id")?.to_owned();
                    let position = position_values
                        .optional_value(row, "author_position")?
                        .ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "signatures Arrow author_position is null for signature_id {signature_id_value:?}"
                            ))
                        })?;
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
                        // Coauthor exclusion and local-window features cannot be
                        // reconstructed without the focal author's position.  The
                        // full Rust featurizer already rejects this corruption;
                        // raw candidate planning must fail closed as well.
                        position: Some(position),
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
    let parallel =
        read_keyed_batches_in_parallel(path, &batches, keep_paper_ids, "paper_id", |batch| {
            read_paper_batches(path, std::slice::from_ref(batch), keep_paper_ids)
        });
    match parallel {
        Some(out) => Ok(out),
        None => read_paper_batches(path, &batches, keep_paper_ids),
    }
}

fn read_paper_batches(
    path: &str,
    batches: &[RecordBatch],
    keep_paper_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, RawArrowPaper>> {
    let mut out = HashMap::default();
    let mut seen_filtered_scan_ids = keep_paper_ids.map(|_| HashSet::<&str>::default());
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
            if let Some(seen_ids) = seen_filtered_scan_ids.as_mut() {
                if !seen_ids.insert(paper_id_value) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "papers Arrow contains duplicate paper_id: {:?}",
                        paper_id_value
                    )));
                }
            }
            if keep_paper_ids.map_or(false, |keep| !keep.contains(paper_id_value)) {
                continue;
            }
            let paper_id = paper_id_value.to_owned();
            match out.entry(paper_id) {
                Entry::Occupied(entry) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "papers Arrow contains duplicate paper_id: {:?}",
                        entry.key()
                    )))
                }
                Entry::Vacant(entry) => {
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
                        entry.key(),
                        predicted_language.as_deref(),
                        is_reliable,
                        language_reliability,
                    )?;
                    entry.insert(RawArrowPaper {
                        // title/venue/journal_name are serialized with the same
                        // ""/None -> NULL producer convention, so NULL means
                        // empty. Coerce NULL -> "" to match Python parity (see
                        // the signature author-name note above).
                        title: title_values.optional_owned(row).unwrap_or_default(),
                        has_abstract: abstract_values
                            .as_ref()
                            .and_then(|col| col.optional_value(row))
                            .is_some_and(|text| !text.is_empty()),
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

/// Where a paper's rows go while reading `paper_authors`.
#[derive(Clone, Copy)]
enum PaperSlot {
    Kept(usize),
    /// Outside `keep_paper_ids`: only positions are kept, for the duplicate check.
    Skipped(usize),
}

/// Rows of `paper_authors` batches grouped by paper, before sorting and checks.
///
/// Rows arrive grouped by paper (about 45 per paper), and SipHash on every row
/// dominated request time. Each paper resolves once to a slot and rows hash with
/// ahash. A filtered read still rejects duplicate positions in the papers it
/// skips, as an unfiltered read would; their positions are checked per paper at
/// the end, which costs far less than hashing every row of every batch read.
struct PaperAuthorRows<'a> {
    papers: Vec<(&'a str, Vec<(i64, String)>)>,
    skipped_papers: Vec<(&'a str, Vec<i64>)>,
    paper_slots: ahash::AHashMap<&'a str, PaperSlot>,
}

impl<'a> PaperAuthorRows<'a> {
    fn new() -> Self {
        Self {
            papers: Vec::new(),
            skipped_papers: Vec::new(),
            paper_slots: ahash::AHashMap::default(),
        }
    }

    fn slot(&mut self, paper_id: &'a str, keep_paper_ids: Option<&HashSet<String>>) -> PaperSlot {
        *self.paper_slots.entry(paper_id).or_insert_with(|| {
            if keep_paper_ids.map_or(false, |keep| !keep.contains(paper_id)) {
                self.skipped_papers.push((paper_id, Vec::new()));
                PaperSlot::Skipped(self.skipped_papers.len() - 1)
            } else {
                self.papers.push((paper_id, Vec::new()));
                PaperSlot::Kept(self.papers.len() - 1)
            }
        })
    }

    fn add_batch(
        &mut self,
        path: &str,
        batch: &'a RecordBatch,
        keep_paper_ids: Option<&HashSet<String>>,
    ) -> PyResult<()> {
        let paper_id_col = batch.column(arrow_column_index(batch, "paper_id", path)?);
        let paper_id_values =
            ArrowStringColumn::from_string_array(paper_id_col.as_ref(), "paper_id")?;
        let position_col = batch.column(arrow_column_index(batch, "position", path)?);
        let position_values = ArrowI64Column::from_i64_array(position_col.as_ref(), "position")?;
        let author_name_col = batch.column(arrow_column_index(batch, "author_name", path)?);
        let author_name_values =
            ArrowStringColumn::from_string_array(author_name_col.as_ref(), "author_name")?;
        let mut previous_paper: Option<(&str, PaperSlot)> = None;
        for row in 0..batch.num_rows() {
            let paper_id_value = paper_id_values.required_value(row, "paper_id")?;
            if paper_id_value.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "paper_authors Arrow cannot contain empty paper_id values",
                ));
            }
            let position = position_values.required_value(row, "position")?;
            let author_name = author_name_values.required_value(row, "author_name")?;
            let slot = match previous_paper {
                Some((previous_id, slot)) if previous_id == paper_id_value => slot,
                _ => {
                    let slot = self.slot(paper_id_value, keep_paper_ids);
                    previous_paper = Some((paper_id_value, slot));
                    slot
                }
            };
            match slot {
                PaperSlot::Kept(index) => self.papers[index]
                    .1
                    .push((position, author_name.to_owned())),
                PaperSlot::Skipped(index) => self.skipped_papers[index].1.push(position),
            }
        }
        Ok(())
    }

    /// Appends the rows of a later batch range, as if read after this one. Both
    /// sides classify a paper by the same keep set, so their slots agree.
    fn merge(&mut self, later: PaperAuthorRows<'a>, keep_paper_ids: Option<&HashSet<String>>) {
        for (paper_id, rows) in later.papers {
            if let PaperSlot::Kept(index) = self.slot(paper_id, keep_paper_ids) {
                self.papers[index].1.extend(rows);
            }
        }
        for (paper_id, positions) in later.skipped_papers {
            if let PaperSlot::Skipped(index) = self.slot(paper_id, keep_paper_ids) {
                self.skipped_papers[index].1.extend(positions);
            }
        }
    }

    fn finish(self) -> PyResult<HashMap<String, Vec<(i64, String)>>> {
        for (paper_id, mut positions) in self.skipped_papers {
            positions.sort_unstable();
            if let Some(pair) = positions.windows(2).find(|pair| pair[0] == pair[1]) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "paper_authors Arrow contains duplicate (paper_id, position): ({:?}, {})",
                    paper_id, pair[0]
                )));
            }
        }
        let mut out: HashMap<String, Vec<(i64, String)>> =
            HashMap::with_capacity_and_hasher(self.papers.len(), Default::default());
        for (paper_id, mut authors) in self.papers {
            authors.sort_by_key(|(position, _name)| *position);
            if let Some(pair) = authors.windows(2).find(|pair| pair[0].0 == pair[1].0) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "paper_authors Arrow contains duplicate (paper_id, position): ({:?}, {})",
                    paper_id, pair[0].0
                )));
            }
            out.insert(paper_id.to_owned(), authors);
        }
        Ok(out)
    }
}

pub(crate) fn read_raw_arrow_paper_authors_from_batches(
    path: &str,
    batches: Vec<RecordBatch>,
    keep_paper_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, Vec<(i64, String)>>> {
    if read_in_parallel(&batches, keep_paper_ids) {
        let parts = batches
            .par_iter()
            .map(|batch| {
                let mut rows = PaperAuthorRows::new();
                rows.add_batch(path, batch, keep_paper_ids)
                    .ok()
                    .map(|()| rows)
            })
            .collect::<Option<Vec<_>>>();
        if let Some(parts) = parts {
            let mut merged = PaperAuthorRows::new();
            for part in parts {
                merged.merge(part, keep_paper_ids);
            }
            if let Ok(out) = merged.finish() {
                return Ok(out);
            }
        }
    }
    read_paper_author_batches(path, &batches, keep_paper_ids)
}

fn read_paper_author_batches(
    path: &str,
    batches: &[RecordBatch],
    keep_paper_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, Vec<(i64, String)>>> {
    let mut rows = PaperAuthorRows::new();
    for batch in batches {
        rows.add_batch(path, batch, keep_paper_ids)?;
    }
    rows.finish()
}

pub(crate) fn read_raw_arrow_cluster_seeds(
    path: &str,
) -> PyResult<(Vec<String>, HashMap<String, Vec<String>>)> {
    let mut component_order = Vec::new();
    let mut members_by_component: HashMap<String, Vec<String>> = HashMap::default();
    let mut component_by_signature_id = HashMap::<String, String>::default();
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
                .to_owned();
            let component_key = cluster_id_values
                .required_value(row, "cluster_id")?
                .to_owned();
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

pub(crate) fn insert_canonical_cluster_seed_disallow(
    out: &mut HashSet<(String, String)>,
    left: String,
    right: String,
    label: &str,
) -> PyResult<()> {
    if left.is_empty() || right.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label} cannot contain empty signature_id values"
        )));
    }
    if left == right {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label} contains a self-pair for signature_id={left:?}"
        )));
    }
    let pair = canonical_signature_pair_owned(left, right);
    if !out.insert(pair.clone()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label} contains duplicate pair: {pair:?}"
        )));
    }
    Ok(())
}

pub(crate) fn read_raw_arrow_cluster_seed_disallows(
    path: &str,
) -> PyResult<HashSet<(String, String)>> {
    let mut out = HashSet::default();
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
                .to_owned();
            let right = right_values
                .required_value(row, "signature_id_2")?
                .to_owned();
            insert_canonical_cluster_seed_disallow(
                &mut out,
                left,
                right,
                "cluster_seed_disallows",
            )?;
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
    let mut seen_signature_ids = HashSet::<String>::default();
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
                .to_owned();
            let query_view = query_view_values
                .required_value(row, "query_view")?
                .to_owned();
            let query_author = query_author_values
                .required_value(row, "query_author")?
                .to_owned();
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
) -> PyResult<HashMap<String, arrow::buffer::ScalarBuffer<f32>>> {
    let parallel =
        read_keyed_batches_in_parallel(path, &batches, keep_paper_ids, "paper_id", |batch| {
            read_specter_batches(path, std::slice::from_ref(batch), keep_paper_ids)
        });
    match parallel {
        Some(out) => Ok(out),
        None => read_specter_batches(path, &batches, keep_paper_ids),
    }
}

fn read_specter_batches(
    path: &str,
    batches: &[RecordBatch],
    keep_paper_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, arrow::buffer::ScalarBuffer<f32>>> {
    let mut out = HashMap::default();
    let mut seen_filtered_scan_ids = keep_paper_ids.map(|_| HashSet::<&str>::default());
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
            if let Some(seen_ids) = seen_filtered_scan_ids.as_mut() {
                if !seen_ids.insert(paper_id_value) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "specter Arrow contains duplicate paper_id: {:?}",
                        paper_id_value
                    )));
                }
            }
            if keep_paper_ids.map_or(false, |keep| !keep.contains(paper_id_value)) {
                continue;
            }
            let paper_id = paper_id_value.to_owned();
            match out.entry(paper_id) {
                Entry::Occupied(entry) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "specter Arrow contains duplicate paper_id: {:?}",
                        entry.key()
                    )))
                }
                Entry::Vacant(entry) => {
                    let vector =
                        arrow_optional_f32_vector(embedding_col.as_ref(), row, "embedding")?
                            .ok_or_else(|| {
                                pyo3::exceptions::PyValueError::new_err(format!(
                                    "specter Arrow cannot contain null embedding values: {:?}",
                                    entry.key()
                                ))
                            })?;
                    if vector.is_empty() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "specter Arrow cannot contain zero-dimension embedding values: {:?}",
                            entry.key()
                        )));
                    }
                    entry.insert(vector);
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod production_schema_contract_tests {
    use super::read_raw_arrow_papers_from_batches;
    use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
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

    fn papers_batch(title: Option<ArrayRef>, year: ArrayRef) -> RecordBatch {
        let mut fields = vec![Field::new("paper_id", DataType::Utf8, false)];
        let mut columns: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec!["p1"]))];
        if let Some(title) = title {
            fields.push(Field::new("title", title.data_type().clone(), true));
            columns.push(title);
        }
        fields.extend([
            Field::new("venue", DataType::Utf8, true),
            Field::new("journal_name", DataType::Utf8, true),
            Field::new("year", year.data_type().clone(), true),
        ]);
        columns.extend([
            Arc::new(StringArray::from(vec![Some("A venue")])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("A journal")])) as ArrayRef,
            year,
        ]);
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
            .expect("schema test batch should be structurally valid")
    }

    #[test]
    fn canonical_papers_schema_reads_through_production_reader() {
        let papers = read_raw_arrow_papers_from_batches(
            "<test>",
            vec![papers_batch(
                Some(Arc::new(StringArray::from(vec![Some("A title")]))),
                Arc::new(Int64Array::from(vec![Some(2024)])),
            )],
            None,
        )
        .expect("canonical papers batch should read");

        let paper = papers.get("p1").expect("paper should be present");
        assert_eq!(paper.title, "A title");
        assert_eq!(paper.year, Some(2024));
    }

    #[test]
    fn production_reader_rejects_missing_required_papers_column() {
        let err = read_raw_arrow_papers_from_batches(
            "<test>",
            vec![papers_batch(
                None,
                Arc::new(Int64Array::from(vec![Some(2024)])),
            )],
            None,
        )
        .err()
        .expect("missing required title must fail");

        let message = py_err_message(err);
        assert!(
            message.contains("missing Arrow column") && message.contains("title"),
            "{message}"
        );
    }

    #[test]
    fn production_reader_rejects_wrong_papers_column_types() {
        for (title, year, expected) in [
            (
                Arc::new(Int32Array::from(vec![Some(1)])) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(2024)])) as ArrayRef,
                "title must be a string column, got Int32",
            ),
            (
                Arc::new(StringArray::from(vec![Some("A title")])) as ArrayRef,
                Arc::new(Int32Array::from(vec![Some(2024)])) as ArrayRef,
                "year must be an int64 column, got Int32",
            ),
        ] {
            let err = read_raw_arrow_papers_from_batches(
                "<test>",
                vec![papers_batch(Some(title), year)],
                None,
            )
            .err()
            .expect("wrong physical column type must fail");
            let message = py_err_message(err);
            assert!(message.contains(expected), "{message}");
        }
    }
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
    use arrow::array::{ArrayRef, BooleanArray, Float32Array, Float64Array, StringArray};
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
        papers_batch_with_reliability_column(
            predicted_language,
            is_reliable,
            Arc::new(Float64Array::from(vec![language_reliability])),
        )
    }

    fn papers_batch_with_reliability_column(
        predicted_language: Option<&str>,
        is_reliable: Option<bool>,
        language_reliability: ArrayRef,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("paper_id", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, true),
            Field::new("venue", DataType::Utf8, true),
            Field::new("journal_name", DataType::Utf8, true),
            Field::new("predicted_language", DataType::Utf8, true),
            Field::new("is_reliable", DataType::Boolean, true),
            Field::new(
                "language_reliability",
                language_reliability.data_type().clone(),
                true,
            ),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec![Some("p1")])),
            Arc::new(StringArray::from(vec![Some("A title")])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![predicted_language])),
            Arc::new(BooleanArray::from(vec![is_reliable])),
            language_reliability,
        ];
        RecordBatch::try_new(schema, columns).expect("valid language-reliability batch")
    }

    #[test]
    fn float32_language_reliability_is_rejected() {
        let err = read_raw_arrow_papers_from_batches(
            "<test>",
            vec![papers_batch_with_reliability_column(
                Some("en"),
                Some(true),
                Arc::new(Float32Array::from(vec![Some(0.75)])),
            )],
            None,
        )
        .err()
        .expect("Float32 language reliability must fail");
        let message = py_err_message(err);
        assert!(
            message.contains("language_reliability must be a Float64 column, got Float32"),
            "{message}"
        );
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
    fn stored_language_rejects_empty_or_whitespace_only_language() {
        for predicted_language in ["", " \t"] {
            let err = read_raw_arrow_papers_from_batches(
                "<test>",
                vec![papers_batch(
                    Some(predicted_language),
                    Some(true),
                    Some(0.75),
                )],
                None,
            )
            .err()
            .expect("empty stored language must fail");
            let message = py_err_message(err);
            assert!(
                message.contains("predicted_language must be nonempty"),
                "{message}"
            );
        }
    }

    #[test]
    fn stored_language_rejects_reliability_fields_without_language() {
        for (is_reliable, language_reliability) in [
            (Some(true), Some(0.75)),
            (Some(true), None),
            (None, Some(0.75)),
        ] {
            let err = read_raw_arrow_papers_from_batches(
                "<test>",
                vec![papers_batch(None, is_reliable, language_reliability)],
                None,
            )
            .err()
            .expect("reverse partial stored language detection must fail");
            let message = py_err_message(err);
            assert!(message.contains("require predicted_language"), "{message}");
        }
    }

    #[test]
    fn all_null_stored_language_is_accepted() {
        let papers = read_raw_arrow_papers_from_batches(
            "<test>",
            vec![papers_batch(None, None, None)],
            None,
        )
        .expect("all-null stored language detection must read");
        let paper = papers.get("p1").expect("paper present");
        assert_eq!(paper.predicted_language, None);
        assert_eq!(paper.is_reliable, None);
        assert_eq!(paper.language_reliability, None);
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
        read_paper_author_batches, read_paper_batches, read_raw_arrow_paper_authors_from_batches,
        read_raw_arrow_papers_from_batches, read_raw_arrow_specter_from_batches,
        read_specter_batches,
    };
    use crate::HashSet;
    use arrow::array::{ArrayRef, FixedSizeListArray, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Float32Type, Schema};
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
    fn papers_duplicate_without_keep_filter_is_rejected() {
        let batch = papers_batch(&["p1", "p1"]);
        let err = read_raw_arrow_papers_from_batches("<test>", vec![batch], None)
            .err()
            .expect("duplicate paper_id must be detected without an auxiliary keep-filter set");
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
    fn specter_duplicate_without_keep_filter_is_rejected() {
        let batch = specter_batch(&["p1", "p1"]);
        let err = read_raw_arrow_specter_from_batches("<test>", vec![batch], None)
            .err()
            .expect("duplicate paper_id must be detected without an auxiliary keep-filter set");
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
    fn paper_author_duplicate_across_runs_of_a_skipped_paper_is_rejected() {
        // p1's rows are split by p2's; the repeated position spans the two runs.
        let batch = paper_authors_batch(&["p1", "p2", "p1"], &[3, 0, 3], &["A", "B", "C"]);
        let keep_ids = keep(&["p2"]);
        let err = read_raw_arrow_paper_authors_from_batches("<test>", vec![batch], Some(&keep_ids))
            .err()
            .expect("a duplicate split across runs must be detected");
        assert!(py_err_message(err).contains("duplicate (paper_id, position)"));

        let batch = paper_authors_batch(&["p1", "p2", "p1"], &[3, 0, 4], &["A", "B", "C"]);
        let authors =
            read_raw_arrow_paper_authors_from_batches("<test>", vec![batch], Some(&keep_ids))
                .expect("distinct positions read");
        assert_eq!(authors.len(), 1);
        assert_eq!(authors.get("p2"), Some(&vec![(0, "B".to_string())]));
    }

    #[test]
    fn paper_author_blank_names_are_retained() {
        let batch = paper_authors_batch(&["p1", "p1"], &[0, 1], &["", "   "]);
        let authors = read_raw_arrow_paper_authors_from_batches("<test>", vec![batch], None)
            .expect("blank paper-author names must be retained");

        assert_eq!(
            authors.get("p1"),
            Some(&vec![(0, String::new()), (1, "   ".to_string())])
        );
    }

    #[test]
    fn paper_author_null_name_is_rejected() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("paper_id", DataType::Utf8, false),
            Field::new("position", DataType::Int64, false),
            Field::new("author_name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["p1"])) as ArrayRef,
                Arc::new(Int64Array::from(vec![0])) as ArrayRef,
                Arc::new(StringArray::from(vec![None::<&str>])) as ArrayRef,
            ],
        )
        .expect("valid nullable paper-author batch");
        let err = read_raw_arrow_paper_authors_from_batches("<test>", vec![batch], None)
            .err()
            .expect("null paper-author names must fail");
        let message = py_err_message(err);
        assert!(
            message.contains("author_name is null at row 0"),
            "{message}"
        );
    }

    // Multi-batch reads run in parallel and must equal the serial readers,
    // including the error a serial read reports first.

    fn serial_error(result: pyo3::PyResult<impl Sized>) -> String {
        py_err_message(result.err().expect("the serial read fails"))
    }

    #[test]
    fn parallel_multi_batch_reads_equal_serial_reads() {
        let paper_batches = vec![
            papers_batch(&["p1", "p2"]),
            papers_batch(&["p3", "p4"]),
            papers_batch(&["p5"]),
        ];
        let specter_batches = vec![specter_batch(&["p1", "p2"]), specter_batch(&["p3", "p5"])];
        // p1 spans two batches with its positions out of order.
        let author_batches = vec![
            paper_authors_batch(&["p1", "p1", "p2"], &[2, 0, 0], &["C", "A", "X"]),
            paper_authors_batch(&["p1", "p3"], &[1, 0], &["B", "Y"]),
        ];
        for keep_ids in [None, Some(keep(&["p1", "p3", "p5"]))] {
            let keep_ids = keep_ids.as_ref();
            let papers =
                read_raw_arrow_papers_from_batches("<test>", paper_batches.clone(), keep_ids)
                    .expect("papers read");
            let serial = read_paper_batches("<test>", &paper_batches, keep_ids).expect("serial");
            let mut keys: Vec<_> = papers.keys().collect();
            let mut serial_keys: Vec<_> = serial.keys().collect();
            keys.sort();
            serial_keys.sort();
            assert_eq!(keys, serial_keys);

            let specter =
                read_raw_arrow_specter_from_batches("<test>", specter_batches.clone(), keep_ids)
                    .expect("specter read");
            assert_eq!(
                specter,
                read_specter_batches("<test>", &specter_batches, keep_ids).expect("serial")
            );

            let authors = read_raw_arrow_paper_authors_from_batches(
                "<test>",
                author_batches.clone(),
                keep_ids,
            )
            .expect("authors read");
            assert_eq!(
                authors,
                read_paper_author_batches("<test>", &author_batches, keep_ids).expect("serial")
            );
            assert_eq!(
                authors.get("p1"),
                Some(&vec![
                    (0, "A".to_string()),
                    (1, "B".to_string()),
                    (2, "C".to_string())
                ])
            );
        }
    }

    #[test]
    fn parallel_reads_report_cross_batch_duplicates_like_serial_reads() {
        // p1 repeats across batches: kept without a filter, skipped with one.
        let paper_batches = vec![papers_batch(&["p1", "p2"]), papers_batch(&["p1", "p3"])];
        let specter_batches = vec![specter_batch(&["p1", "p2"]), specter_batch(&["p3", "p1"])];
        let author_batches = vec![
            paper_authors_batch(&["p1", "p2"], &[0, 0], &["A", "X"]),
            paper_authors_batch(&["p3", "p1"], &[0, 0], &["Y", "B"]),
        ];
        for keep_ids in [None, Some(keep(&["p2", "p3"]))] {
            let keep_ids = keep_ids.as_ref();
            assert_eq!(
                serial_error(read_raw_arrow_papers_from_batches(
                    "<test>",
                    paper_batches.clone(),
                    keep_ids
                )),
                serial_error(read_paper_batches("<test>", &paper_batches, keep_ids))
            );
            assert_eq!(
                serial_error(read_raw_arrow_specter_from_batches(
                    "<test>",
                    specter_batches.clone(),
                    keep_ids
                )),
                serial_error(read_specter_batches("<test>", &specter_batches, keep_ids))
            );
            let message = serial_error(read_raw_arrow_paper_authors_from_batches(
                "<test>",
                author_batches.clone(),
                keep_ids,
            ));
            assert!(
                message.contains("duplicate (paper_id, position)"),
                "{message}"
            );
            assert_eq!(
                message,
                serial_error(read_paper_author_batches(
                    "<test>",
                    &author_batches,
                    keep_ids
                ))
            );
        }
    }
}
