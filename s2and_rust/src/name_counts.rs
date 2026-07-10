use memmap2::Mmap;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::{Path, PathBuf};

use crate::{
    fnv64, fnv64_update, read_f64_le_unchecked, read_u32_le_unchecked, read_u64_le_unchecked,
    FNV_OFFSET,
};

const NAME_COUNTS_INDEX_SCHEMA_VERSION: &str = "name_counts_index_v1";

// Manifest "normalization_version" values accepted by this crate. An absent
// field means the artifact predates the canonical_v2 migration and carries
// legacy_compat keys. The crate does not hard-fail on legacy_compat here:
// Python asserts model-vs-artifact normalization compatibility upstream; this
// gate only rejects unknown values, like the schema_version gate.
const NAME_COUNTS_NORMALIZATION_LEGACY_COMPAT: &str = "legacy_compat";
const NAME_COUNTS_NORMALIZATION_CANONICAL_V2: &str = "canonical_v2";

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct NameCountsData {
    pub(crate) first: f64,
    pub(crate) first_last: f64,
    pub(crate) last: f64,
    pub(crate) last_first_initial: f64,
}

#[derive(Default)]
pub(crate) struct RawNameCountMaps {
    pub(crate) index: Option<RawNameCountIndex>,
}

#[derive(Clone, Copy)]
pub(crate) enum RawNameCountKind {
    First,
    Last,
    FirstLast,
    LastFirstInitial,
}

impl RawNameCountKind {
    fn key(self) -> &'static str {
        match self {
            RawNameCountKind::First => "first",
            RawNameCountKind::Last => "last",
            RawNameCountKind::FirstLast => "first_last",
            RawNameCountKind::LastFirstInitial => "last_first_initial",
        }
    }
}

const NAME_COUNTS_INDEX_MAGIC: &[u8; 8] = b"S2NCI001";
const NAME_COUNTS_INDEX_HASH_DOMAIN: &[u8] = b"s2and-name-counts-index-v1\0";
const NAME_COUNTS_INDEX_HEADER_LEN: usize = 32;
const NAME_COUNTS_INDEX_RECORD_LEN: usize = 40;

struct NameCountCacheHasher(u64);

impl Default for NameCountCacheHasher {
    fn default() -> Self {
        Self(FNV_OFFSET)
    }
}

impl Hasher for NameCountCacheHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0 = fnv64_update(self.0, bytes);
    }
}

type NameCountCache<'a> = HashMap<&'a str, f64, BuildHasherDefault<NameCountCacheHasher>>;

pub(crate) struct RawNameCountIndex {
    first: RawNameCountIndexFile,
    last: RawNameCountIndexFile,
    first_last: RawNameCountIndexFile,
    last_first_initial: RawNameCountIndexFile,
    normalization_version: String,
    provenance_binding: Option<NameCountsProvenanceBinding>,
}

/// Python-facing, immutable handle over the four memory-mapped name-count
/// indexes. Python validates artifact digests before opening this handle; the
/// native opener validates the manifest schema and O(1) binary boundaries.
#[pyclass(frozen)]
pub(crate) struct NameCountsIndex {
    index: RawNameCountIndex,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NameCountsProvenanceBinding {
    pub(crate) generation_id: String,
    pub(crate) pickle_sha256: String,
    pub(crate) source_snapshot_id: String,
    pub(crate) selected_rows_sha256: String,
}

struct RawNameCountIndexPaths {
    first: PathBuf,
    last: PathBuf,
    first_last: PathBuf,
    last_first_initial: PathBuf,
    normalization_version: String,
    provenance_binding: Option<NameCountsProvenanceBinding>,
}

impl RawNameCountIndex {
    pub(crate) fn open(path: &str) -> PyResult<Self> {
        let paths = resolve_name_counts_index_paths(path)?;
        Self::open_resolved(&paths)
    }

    /// Open and exhaustively validate every record for direct native artifact
    /// boundaries that have no prior Python digest verification.
    pub(crate) fn open_fully_validated(path: &str) -> PyResult<Self> {
        let paths = resolve_name_counts_index_paths(path)?;
        let index = Self::open_resolved(&paths)?;
        index
            .first
            .validate_all_records(&paths.first, RawNameCountKind::First)?;
        index
            .last
            .validate_all_records(&paths.last, RawNameCountKind::Last)?;
        index
            .first_last
            .validate_all_records(&paths.first_last, RawNameCountKind::FirstLast)?;
        index.last_first_initial.validate_all_records(
            &paths.last_first_initial,
            RawNameCountKind::LastFirstInitial,
        )?;
        Ok(index)
    }

    fn open_resolved(paths: &RawNameCountIndexPaths) -> PyResult<Self> {
        Ok(Self {
            first: RawNameCountIndexFile::open(&paths.first, RawNameCountKind::First)?,
            last: RawNameCountIndexFile::open(&paths.last, RawNameCountKind::Last)?,
            first_last: RawNameCountIndexFile::open(
                &paths.first_last,
                RawNameCountKind::FirstLast,
            )?,
            last_first_initial: RawNameCountIndexFile::open(
                &paths.last_first_initial,
                RawNameCountKind::LastFirstInitial,
            )?,
            normalization_version: paths.normalization_version.clone(),
            provenance_binding: paths.provenance_binding.clone(),
        })
    }

    fn get(&self, kind: RawNameCountKind, name: &str) -> Option<f64> {
        match kind {
            RawNameCountKind::First => self.first.get(kind, name),
            RawNameCountKind::Last => self.last.get(kind, name),
            RawNameCountKind::FirstLast => self.first_last.get(kind, name),
            RawNameCountKind::LastFirstInitial => self.last_first_initial.get(kind, name),
        }
    }
}

fn validate_lookup_column_lengths(lengths: [usize; 4]) -> PyResult<usize> {
    let row_count = lengths[0];
    if lengths.iter().any(|length| *length != row_count) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "name-count lookup columns must have equal length: first={} last={} first_last={} last_first_initial={}",
            lengths[0], lengths[1], lengths[2], lengths[3]
        )));
    }
    Ok(row_count)
}

fn lookup_name_count_column<S: AsRef<str>>(
    index: &RawNameCountIndex,
    kind: RawNameCountKind,
    keys: &[Option<S>],
) -> Vec<f64> {
    lookup_name_count_column_with(keys, |key| index.get(kind, key).unwrap_or(1.0))
}

fn lookup_unique_name_count_column<S: AsRef<str>>(
    index: &RawNameCountIndex,
    kind: RawNameCountKind,
    keys: &[Option<S>],
) -> Vec<f64> {
    lookup_unique_name_count_column_with(keys, |key| index.get(kind, key).unwrap_or(1.0))
}

fn lookup_unique_name_count_column_with<S, F>(keys: &[Option<S>], mut resolve: F) -> Vec<f64>
where
    S: AsRef<str>,
    F: FnMut(&str) -> f64,
{
    keys.iter()
        .map(|key| match key {
            None => f64::NAN,
            Some(key) => resolve(key.as_ref()),
        })
        .collect()
}

fn lookup_name_count_column_with<S, F>(keys: &[Option<S>], mut resolve: F) -> Vec<f64>
where
    S: AsRef<str>,
    F: FnMut(&str) -> f64,
{
    // Batch-local borrowed keys avoid both repeated mmap searches and String
    // copies. The map dies with this column task, so retained memory is
    // proportional only to unique keys in the current batch.
    let mut resolved = NameCountCache::default();
    keys.iter()
        .map(|key| match key {
            None => f64::NAN,
            Some(key) => {
                let key = key.as_ref();
                if let Some(value) = resolved.get(key) {
                    *value
                } else {
                    let value = resolve(key);
                    resolved.insert(key, value);
                    value
                }
            }
        })
        .collect()
}

fn lookup_name_count_columns<S: AsRef<str> + Sync>(
    index: &RawNameCountIndex,
    first_keys: &[Option<S>],
    last_keys: &[Option<S>],
    first_last_keys: &[Option<S>],
    last_first_initial_keys: &[Option<S>],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Four column tasks are enough to saturate this random mmap lookup. Avoid
    // per-column parallel iterators: they add scheduling overhead and contend
    // for the same index pages without exposing more independent work.
    let ((first, last), (first_last, last_first_initial)) = rayon::join(
        || {
            rayon::join(
                || lookup_name_count_column(index, RawNameCountKind::First, first_keys),
                || lookup_name_count_column(index, RawNameCountKind::Last, last_keys),
            )
        },
        || {
            rayon::join(
                || lookup_name_count_column(index, RawNameCountKind::FirstLast, first_last_keys),
                || {
                    lookup_name_count_column(
                        index,
                        RawNameCountKind::LastFirstInitial,
                        last_first_initial_keys,
                    )
                },
            )
        },
    );
    (first, last, first_last, last_first_initial)
}

fn lookup_unique_name_count_columns<S: AsRef<str> + Sync>(
    index: &RawNameCountIndex,
    first_keys: &[Option<S>],
    last_keys: &[Option<S>],
    first_last_keys: &[Option<S>],
    last_first_initial_keys: &[Option<S>],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let ((first, last), (first_last, last_first_initial)) = rayon::join(
        || {
            rayon::join(
                || lookup_unique_name_count_column(index, RawNameCountKind::First, first_keys),
                || lookup_unique_name_count_column(index, RawNameCountKind::Last, last_keys),
            )
        },
        || {
            rayon::join(
                || {
                    lookup_unique_name_count_column(
                        index,
                        RawNameCountKind::FirstLast,
                        first_last_keys,
                    )
                },
                || {
                    lookup_unique_name_count_column(
                        index,
                        RawNameCountKind::LastFirstInitial,
                        last_first_initial_keys,
                    )
                },
            )
        },
    );
    (first, last, first_last, last_first_initial)
}

impl NameCountsIndex {
    fn lookup_many_impl<'py>(
        &self,
        py: Python<'py>,
        first_keys: Vec<Option<PyBackedStr>>,
        last_keys: Vec<Option<PyBackedStr>>,
        first_last_keys: Vec<Option<PyBackedStr>>,
        last_first_initial_keys: Vec<Option<PyBackedStr>>,
        keys_are_unique: bool,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
    )> {
        validate_lookup_column_lengths([
            first_keys.len(),
            last_keys.len(),
            first_last_keys.len(),
            last_first_initial_keys.len(),
        ])?;
        let index = &self.index;
        // PyBackedStr holds the Python strings without allocating Rust String
        // copies and is Send + Sync. Borrow the columns into the no-GIL region
        // so their Python owners are dropped only after the GIL is reacquired.
        let (first, last, first_last, last_first_initial) = py.allow_threads(|| {
            if keys_are_unique {
                lookup_unique_name_count_columns(
                    index,
                    &first_keys,
                    &last_keys,
                    &first_last_keys,
                    &last_first_initial_keys,
                )
            } else {
                lookup_name_count_columns(
                    index,
                    &first_keys,
                    &last_keys,
                    &first_last_keys,
                    &last_first_initial_keys,
                )
            }
        });
        use numpy::IntoPyArray;
        Ok((
            first.into_pyarray(py),
            last.into_pyarray(py),
            first_last.into_pyarray(py),
            last_first_initial.into_pyarray(py),
        ))
    }
}

#[pymethods]
impl NameCountsIndex {
    /// Open a manifest-backed name-count index. Callers at the Python artifact
    /// boundary must verify file digests before constructing this mmap handle.
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        Ok(Self {
            index: RawNameCountIndex::open(path)?,
        })
    }

    #[getter]
    fn normalization_version(&self) -> &str {
        &self.index.normalization_version
    }

    /// Return the exact four-field model-binding tuple used by RustFeaturizer.
    #[getter]
    fn name_counts_provenance_binding(&self) -> Option<(String, String, String, String)> {
        self.index.provenance_binding.as_ref().map(|binding| {
            (
                binding.generation_id.clone(),
                binding.pickle_sha256.clone(),
                binding.source_snapshot_id.clone(),
                binding.selected_rows_sha256.clone(),
            )
        })
    }

    /// Resolve four aligned optional-key columns into float64 count arrays.
    ///
    /// A missing key (`None`) produces NaN. An informative string absent from
    /// its index produces the historical default count of 1.0.
    fn lookup_many<'py>(
        &self,
        py: Python<'py>,
        first_keys: Vec<Option<PyBackedStr>>,
        last_keys: Vec<Option<PyBackedStr>>,
        first_last_keys: Vec<Option<PyBackedStr>>,
        last_first_initial_keys: Vec<Option<PyBackedStr>>,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
    )> {
        self.lookup_many_impl(
            py,
            first_keys,
            last_keys,
            first_last_keys,
            last_first_initial_keys,
            false,
        )
    }

    /// Internal fast path for Python callers that already deduplicated each
    /// column. Values and ordering follow `lookup_many`; duplicate inputs are
    /// accepted but deliberately receive no native caching.
    fn _lookup_many_unique<'py>(
        &self,
        py: Python<'py>,
        first_keys: Vec<Option<PyBackedStr>>,
        last_keys: Vec<Option<PyBackedStr>>,
        first_last_keys: Vec<Option<PyBackedStr>>,
        last_first_initial_keys: Vec<Option<PyBackedStr>>,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
    )> {
        self.lookup_many_impl(
            py,
            first_keys,
            last_keys,
            first_last_keys,
            last_first_initial_keys,
            true,
        )
    }
}

struct RawNameCountIndexFile {
    // Memory-mapped index file. Bulk of the file is the variable-length
    // name blob (hundreds of MB per kind); mmap avoids reading or
    // allocating that bulk up front. Lookups page-fault only the records
    // section pages they touch plus the matched name range.
    mmap: Mmap,
    record_count: usize,
    blob_offset: usize,
    blob_len: usize,
}

impl RawNameCountIndexFile {
    fn open(path: &Path, kind: RawNameCountKind) -> PyResult<Self> {
        let file = File::open(path).map_err(|err| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "failed to open name-count index file {}: {}",
                path.display(),
                err
            ))
        })?;
        // SAFETY: the index files are produced by our own offline writer
        // and remain immutable on disk during inference. Concurrent
        // truncation is not part of the supported operating contract.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|err| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "failed to mmap name-count index file {}: {}",
                path.display(),
                err
            ))
        })?;
        let bytes: &[u8] = &mmap;
        if bytes.len() < NAME_COUNTS_INDEX_HEADER_LEN {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index file {} is shorter than the header",
                path.display()
            )));
        }
        if &bytes[0..8] != NAME_COUNTS_INDEX_MAGIC {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index file {} has unsupported magic for kind {}",
                path.display(),
                kind.key(),
            )));
        }
        let record_count = read_u64_usize(bytes, 8, path, "record_count")?;
        let blob_offset = read_u64_usize(bytes, 16, path, "blob_offset")?;
        let blob_len = read_u64_usize(bytes, 24, path, "blob_len")?;
        let records_end = NAME_COUNTS_INDEX_HEADER_LEN
            .checked_add(
                record_count
                    .checked_mul(NAME_COUNTS_INDEX_RECORD_LEN)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyOverflowError::new_err(format!(
                            "name-count index file {} has too many records",
                            path.display()
                        ))
                    })?,
            )
            .ok_or_else(|| {
                pyo3::exceptions::PyOverflowError::new_err(format!(
                    "name-count index file {} record section overflows",
                    path.display()
                ))
            })?;
        let blob_end = blob_offset.checked_add(blob_len).ok_or_else(|| {
            pyo3::exceptions::PyOverflowError::new_err(format!(
                "name-count index file {} blob section overflows",
                path.display()
            ))
        })?;
        if blob_offset < records_end || blob_end > bytes.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index file {} has invalid record/blob offsets",
                path.display()
            )));
        }
        Ok(Self {
            mmap,
            record_count,
            blob_offset,
            blob_len,
        })
    }

    /// Exhaustively validate every record and the global hash-pair ordering.
    /// The Python handle skips this O(N) scan after digest verification;
    /// direct native Arrow ingestion calls it at its artifact boundary.
    fn validate_all_records(&self, path: &Path, kind: RawNameCountKind) -> PyResult<()> {
        let bytes: &[u8] = &self.mmap;
        for index in 0..self.record_count {
            let record_offset = NAME_COUNTS_INDEX_HEADER_LEN + index * NAME_COUNTS_INDEX_RECORD_LEN;
            let name_offset_raw = read_u64_le_unchecked(bytes, record_offset + 16);
            let name_offset = usize::try_from(name_offset_raw).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(format!(
                    "name-count index file {} record {} for kind {} has name offset that overflows usize: {}",
                    path.display(),
                    index,
                    kind.key(),
                    name_offset_raw
                ))
            })?;
            let name_len = read_u32_le_unchecked(bytes, record_offset + 24) as usize;
            let name_end = name_offset.checked_add(name_len).ok_or_else(|| {
                pyo3::exceptions::PyOverflowError::new_err(format!(
                    "name-count index file {} record {} for kind {} name range overflows",
                    path.display(),
                    index,
                    kind.key()
                ))
            })?;
            if name_end > self.blob_len {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "name-count index file {} record {} for kind {} has name range [{}, {}) outside blob length {}",
                    path.display(),
                    index,
                    kind.key(),
                    name_offset,
                    name_end,
                    self.blob_len
                )));
            }
        }
        if self.record_count > 1 {
            let read_pair = |index: usize| {
                let offset = NAME_COUNTS_INDEX_HEADER_LEN + index * NAME_COUNTS_INDEX_RECORD_LEN;
                (
                    read_u64_le_unchecked(bytes, offset),
                    read_u64_le_unchecked(bytes, offset + 8),
                )
            };
            let mut previous_index = 0usize;
            let mut previous_pair = read_pair(0);
            for index in 1..self.record_count {
                let pair = read_pair(index);
                if pair < previous_pair {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "name-count index file {} is not sorted for kind {}: record {} {:?} follows record {} {:?}",
                        path.display(),
                        kind.key(),
                        index,
                        pair,
                        previous_index,
                        previous_pair
                    )));
                }
                previous_index = index;
                previous_pair = pair;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn open_fully_validated(path: &Path, kind: RawNameCountKind) -> PyResult<Self> {
        let index = Self::open(path, kind)?;
        index.validate_all_records(path, kind)?;
        Ok(index)
    }

    fn record_offset(&self, index: usize) -> usize {
        NAME_COUNTS_INDEX_HEADER_LEN + index * NAME_COUNTS_INDEX_RECORD_LEN
    }

    fn record_hash_pair(&self, index: usize) -> (u64, u64) {
        let offset = self.record_offset(index);
        (
            read_u64_le_unchecked(&self.mmap, offset),
            read_u64_le_unchecked(&self.mmap, offset + 8),
        )
    }

    fn get(&self, kind: RawNameCountKind, name: &str) -> Option<f64> {
        let name_bytes = name.as_bytes();
        let (hash_1, hash_2) = name_counts_index_hashes(kind, name_bytes);
        let target_pair = (hash_1, hash_2);
        let mut lower = 0usize;
        let mut upper = self.record_count;
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            if self.record_hash_pair(middle) < target_pair {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        let mut index = lower;
        while index < self.record_count {
            let (record_hash_1, record_hash_2) = self.record_hash_pair(index);
            let record_pair = (record_hash_1, record_hash_2);
            if record_pair > target_pair {
                break;
            }
            if record_pair < target_pair {
                index += 1;
                continue;
            }
            let offset = self.record_offset(index);
            let name_offset = match usize::try_from(read_u64_le_unchecked(&self.mmap, offset + 16))
            {
                Ok(value) => value,
                Err(_) => break,
            };
            let name_len = read_u32_le_unchecked(&self.mmap, offset + 24) as usize;
            if name_offset
                .checked_add(name_len)
                .map_or(false, |end| end <= self.blob_len)
            {
                let start = self.blob_offset + name_offset;
                let end = start + name_len;
                if &self.mmap[start..end] == name_bytes {
                    return Some(read_f64_le_unchecked(&self.mmap, offset + 32));
                }
            }
            index += 1;
        }
        None
    }
}

impl RawNameCountMaps {
    pub(crate) fn from_index(index: RawNameCountIndex) -> Self {
        Self { index: Some(index) }
    }

    pub(crate) fn has_data(&self) -> bool {
        self.index.is_some()
    }

    pub(crate) fn get(&self, kind: RawNameCountKind, name: &str) -> Option<f64> {
        self.index.as_ref().and_then(|index| index.get(kind, name))
    }

    pub(crate) fn provenance_binding(&self) -> Option<&NameCountsProvenanceBinding> {
        self.index
            .as_ref()
            .and_then(|index| index.provenance_binding.as_ref())
    }
}

fn required_provenance_string(
    provenance: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    manifest_path: &Path,
) -> PyResult<String> {
    provenance
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index manifest {} source_provenance requires nonempty string {}",
                manifest_path.display(),
                field
            ))
        })
}

fn required_provenance_sha256(
    provenance: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    manifest_path: &Path,
) -> PyResult<String> {
    let value = required_provenance_string(provenance, field, manifest_path)?;
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|character| character.is_ascii_digit() || (b'a'..=b'f').contains(character))
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "name-count index manifest {} source_provenance requires lowercase SHA-256 {}",
            manifest_path.display(),
            field
        )));
    }
    Ok(value)
}

fn read_name_counts_provenance_binding(
    manifest: &serde_json::Value,
    manifest_path: &Path,
) -> PyResult<Option<NameCountsProvenanceBinding>> {
    let Some(value) = manifest.get("source_provenance") else {
        return Ok(None);
    };
    let provenance = value.as_object().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "name-count index manifest {} has non-object source_provenance",
            manifest_path.display()
        ))
    })?;
    if provenance
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        != Some("name_counts_provenance_v1")
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "name-count index manifest {} source_provenance requires name_counts_provenance_v1",
            manifest_path.display()
        )));
    }
    Ok(Some(NameCountsProvenanceBinding {
        generation_id: required_provenance_string(provenance, "generation_id", manifest_path)?,
        pickle_sha256: required_provenance_sha256(provenance, "pickle_sha256", manifest_path)?,
        source_snapshot_id: required_provenance_string(
            provenance,
            "source_snapshot_id",
            manifest_path,
        )?,
        selected_rows_sha256: required_provenance_sha256(
            provenance,
            "selected_rows_sha256",
            manifest_path,
        )?,
    }))
}

fn name_counts_index_manifest_path(
    index_dir: &Path,
    files: &serde_json::Map<String, serde_json::Value>,
    kind: &str,
) -> PyResult<PathBuf> {
    let path_value = files
        .get(kind)
        .and_then(|entry| entry.get("path"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index manifest {} is missing files.{}.path",
                index_dir.join("manifest.json").display(),
                kind
            ))
        })?;
    let raw_path = PathBuf::from(path_value);
    let resolved = if raw_path.is_absolute() {
        raw_path
    } else {
        index_dir.join(raw_path)
    };
    if !resolved.exists() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "name-count index manifest {} points to missing file {}",
            index_dir.join("manifest.json").display(),
            resolved.display()
        )));
    }
    Ok(resolved)
}

fn read_name_counts_index_manifest(index_dir: &Path) -> PyResult<RawNameCountIndexPaths> {
    let manifest_path = index_dir.join("manifest.json");
    let manifest_text = fs::read_to_string(&manifest_path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to read name-count index manifest {}: {}",
            manifest_path.display(),
            err
        ))
    })?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "failed to parse name-count index manifest {}: {}",
            manifest_path.display(),
            err
        ))
    })?;
    let schema_version = manifest
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index manifest {} is missing schema_version",
                manifest_path.display()
            ))
        })?;
    if schema_version != NAME_COUNTS_INDEX_SCHEMA_VERSION {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "name-count index manifest {} has unsupported schema_version {:?}; expected {:?}",
            manifest_path.display(),
            schema_version,
            NAME_COUNTS_INDEX_SCHEMA_VERSION
        )));
    }
    let normalization_version = match manifest.get("normalization_version") {
        // Artifacts written before the canonical_v2 migration carry no
        // normalization_version field and are legacy_compat by definition.
        None => NAME_COUNTS_NORMALIZATION_LEGACY_COMPAT.to_string(),
        Some(value) => {
            let version = value.as_str().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "name-count index manifest {} has non-string normalization_version",
                    manifest_path.display()
                ))
            })?;
            if version != NAME_COUNTS_NORMALIZATION_LEGACY_COMPAT
                && version != NAME_COUNTS_NORMALIZATION_CANONICAL_V2
            {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "name-count index manifest {} has unsupported normalization_version {:?}; expected {:?} or {:?}",
                    manifest_path.display(),
                    version,
                    NAME_COUNTS_NORMALIZATION_LEGACY_COMPAT,
                    NAME_COUNTS_NORMALIZATION_CANONICAL_V2
                )));
            }
            version.to_string()
        }
    };
    let files = manifest
        .get("files")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index manifest {} is missing files",
                manifest_path.display()
            ))
        })?;
    let provenance_binding = read_name_counts_provenance_binding(&manifest, &manifest_path)?;
    Ok(RawNameCountIndexPaths {
        first: name_counts_index_manifest_path(index_dir, files, "first")?,
        last: name_counts_index_manifest_path(index_dir, files, "last")?,
        first_last: name_counts_index_manifest_path(index_dir, files, "first_last")?,
        last_first_initial: name_counts_index_manifest_path(
            index_dir,
            files,
            "last_first_initial",
        )?,
        normalization_version,
        provenance_binding,
    })
}

/// Return the validated `normalization_version` recorded in a name-count index
/// manifest without opening the index files ("legacy_compat" when the field is
/// absent). Exposed to Python so callers can assert artifact-vs-model
/// normalization compatibility before loading the index.
#[pyfunction]
pub(crate) fn read_name_counts_index_normalization_version(path: &str) -> PyResult<String> {
    Ok(resolve_name_counts_index_paths(path)?.normalization_version)
}

fn resolve_name_counts_index_paths(path: &str) -> PyResult<RawNameCountIndexPaths> {
    let direct = PathBuf::from(path);
    let nested = direct.join("name_counts_index");
    for index_dir in [&direct, &nested] {
        if index_dir.join("manifest.json").exists() {
            return read_name_counts_index_manifest(index_dir);
        }
    }
    Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
        "name-count index path {} does not contain manifest.json",
        path
    )))
}

fn name_counts_index_hashes(kind: RawNameCountKind, name_bytes: &[u8]) -> (u64, u64) {
    let first = fnv64(name_bytes);
    let mut second = FNV_OFFSET;
    second = fnv64_update(second, NAME_COUNTS_INDEX_HASH_DOMAIN);
    second = fnv64_update(second, kind.key().as_bytes());
    second = fnv64_update(second, b"\0");
    second = fnv64_update(second, name_bytes);
    (first, second)
}

#[cfg(test)]
mod name_counts_tests {
    use super::{
        lookup_name_count_column, lookup_name_count_column_with,
        lookup_unique_name_count_column_with, name_counts_index_hashes,
        read_name_counts_index_normalization_version, read_name_counts_provenance_binding,
        validate_lookup_column_lengths, RawNameCountIndex, RawNameCountIndexFile, RawNameCountKind,
    };
    use std::collections::HashMap;
    use std::io::{Seek, SeekFrom, Write};
    use std::sync::atomic::{AtomicU64, Ordering};

    // A valid 32-byte name-count index header describing zero records; the
    // manifest reader only checks the files exist.
    fn empty_index_bytes() -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(b"S2NCI001");
        bytes[16..24].copy_from_slice(&32u64.to_le_bytes());
        bytes
    }

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = format!(
            "{prefix}_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).expect("create temp index dir");
        dir
    }

    fn write_artifact(normalization_version_json: Option<&str>) -> std::path::PathBuf {
        let dir = unique_temp_dir("s2and_nc_version");
        for name in ["first", "last", "first_last", "last_first_initial"] {
            let mut file =
                std::fs::File::create(dir.join(format!("{name}.bin"))).expect("create index file");
            file.write_all(&empty_index_bytes())
                .expect("write index header");
        }
        let version_field = normalization_version_json
            .map(|value| format!(r#""normalization_version":{value},"#))
            .unwrap_or_default();
        let manifest = format!(
            concat!(
                r#"{{"schema_version":"name_counts_index_v1",{}"files":{{"#,
                r#""first":{{"path":"first.bin"}},"last":{{"path":"last.bin"}},"#,
                r#""first_last":{{"path":"first_last.bin"}},"#,
                r#""last_first_initial":{{"path":"last_first_initial.bin"}}}}}}"#,
            ),
            version_field
        );
        std::fs::write(dir.join("manifest.json"), manifest).expect("write manifest");
        dir
    }

    fn write_index_file(path: &std::path::Path, kind: RawNameCountKind, values: &[(&str, f64)]) {
        let mut records = values
            .iter()
            .map(|(name, count)| {
                let name_bytes = name.as_bytes().to_vec();
                let hashes = name_counts_index_hashes(kind, &name_bytes);
                (hashes, name_bytes, *count)
            })
            .collect::<Vec<_>>();
        records.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        let blob_offset = 32 + records.len() * 40;
        let blob_len = records.iter().map(|record| record.1.len()).sum::<usize>();
        let mut output = Vec::with_capacity(blob_offset + blob_len);
        output.extend_from_slice(b"S2NCI001");
        output.extend_from_slice(&(records.len() as u64).to_le_bytes());
        output.extend_from_slice(&(blob_offset as u64).to_le_bytes());
        output.extend_from_slice(&(blob_len as u64).to_le_bytes());
        let mut name_offset = 0usize;
        for ((hash_1, hash_2), name, count) in &records {
            output.extend_from_slice(&hash_1.to_le_bytes());
            output.extend_from_slice(&hash_2.to_le_bytes());
            output.extend_from_slice(&(name_offset as u64).to_le_bytes());
            output.extend_from_slice(&(name.len() as u32).to_le_bytes());
            output.extend_from_slice(&0u32.to_le_bytes());
            output.extend_from_slice(&count.to_le_bytes());
            name_offset += name.len();
        }
        for (_hashes, name, _count) in records {
            output.extend_from_slice(&name);
        }
        std::fs::write(path, output).expect("write lookup index");
    }

    fn write_lookup_artifact() -> std::path::PathBuf {
        let dir = unique_temp_dir("s2and_nc_lookup");
        write_index_file(
            &dir.join("first.bin"),
            RawNameCountKind::First,
            &[("alice", 7.0), ("élodie", 3.0)],
        );
        write_index_file(
            &dir.join("last.bin"),
            RawNameCountKind::Last,
            &[("smith", 11.0)],
        );
        write_index_file(
            &dir.join("first_last.bin"),
            RawNameCountKind::FirstLast,
            &[("alice smith", 5.0)],
        );
        write_index_file(
            &dir.join("last_first_initial.bin"),
            RawNameCountKind::LastFirstInitial,
            &[("smith a", 9.0)],
        );
        let manifest = serde_json::json!({
            "schema_version": "name_counts_index_v1",
            "normalization_version": "canonical_v2",
            "source_provenance": {
                "schema_version": "name_counts_provenance_v1",
                "generation_id": "generation-a",
                "pickle_sha256": "0".repeat(64),
                "source_snapshot_id": "snapshot-a",
                "selected_rows_sha256": "1".repeat(64),
            },
            "files": {
                "first": {"path": "first.bin"},
                "last": {"path": "last.bin"},
                "first_last": {"path": "first_last.bin"},
                "last_first_initial": {"path": "last_first_initial.bin"},
            },
        });
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write lookup manifest");
        dir
    }

    fn read_version(dir: &std::path::Path) -> pyo3::PyResult<String> {
        read_name_counts_index_normalization_version(dir.to_str().expect("utf-8 temp path"))
    }

    fn py_err_message(err: pyo3::PyErr) -> String {
        #[cfg(windows)]
        if let Some(python_home) = option_env!("S2AND_RUST_PYTHONHOME") {
            std::env::set_var("PYTHONHOME", python_home);
        }
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|py| err.value(py).to_string())
    }

    #[test]
    fn source_provenance_binding_is_exact_and_validated() {
        let manifest = serde_json::json!({
            "source_provenance": {
                "schema_version": "name_counts_provenance_v1",
                "generation_id": "generation-a",
                "pickle_sha256": "0".repeat(64),
                "source_snapshot_id": "snapshot-a",
                "selected_rows_sha256": "1".repeat(64),
            }
        });
        let binding =
            read_name_counts_provenance_binding(&manifest, std::path::Path::new("manifest.json"))
                .expect("valid provenance")
                .expect("present provenance");

        assert_eq!(binding.generation_id, "generation-a");
        assert_eq!(binding.pickle_sha256, "0".repeat(64));
        assert_eq!(binding.source_snapshot_id, "snapshot-a");
        assert_eq!(binding.selected_rows_sha256, "1".repeat(64));
    }

    #[test]
    fn source_provenance_binding_rejects_non_sha_digest() {
        let manifest = serde_json::json!({
            "source_provenance": {
                "schema_version": "name_counts_provenance_v1",
                "generation_id": "generation-a",
                "pickle_sha256": "G".repeat(64),
                "source_snapshot_id": "snapshot-a",
                "selected_rows_sha256": "1".repeat(64),
            }
        });
        let error =
            read_name_counts_provenance_binding(&manifest, std::path::Path::new("manifest.json"))
                .expect_err("invalid digest must fail");

        assert!(py_err_message(error).contains("lowercase SHA-256 pickle_sha256"));
    }

    #[test]
    fn column_lookup_preserves_missing_unknown_and_exact_utf8_semantics() {
        let dir = write_lookup_artifact();
        let index = RawNameCountIndex::open(dir.to_str().expect("utf-8 temp path"))
            .expect("open lookup index");

        let values = lookup_name_count_column(
            &index,
            RawNameCountKind::First,
            &[
                Some("alice".to_string()),
                Some("unknown".to_string()),
                None,
                Some("élodie".to_string()),
                Some("elodie".to_string()),
            ],
        );

        drop(index);
        std::fs::remove_dir_all(&dir).expect("remove lookup artifact");
        assert_eq!(values[0], 7.0);
        assert_eq!(values[1], 1.0);
        assert!(values[2].is_nan());
        assert_eq!(values[3], 3.0);
        assert_eq!(values[4], 1.0);
    }

    #[test]
    fn column_lookup_deduplicates_repeated_known_unknown_and_utf8_keys() {
        let utf8_key = "\u{00e9}lodie".to_string();
        let keys = vec![
            Some("known".to_string()),
            Some("known".to_string()),
            Some("unknown".to_string()),
            Some("unknown".to_string()),
            Some(utf8_key.clone()),
            Some(utf8_key),
            None,
        ];
        let mut calls = HashMap::<String, usize>::new();

        let values = lookup_name_count_column_with(&keys, |key| {
            *calls.entry(key.to_string()).or_default() += 1;
            match key {
                "known" => 7.0,
                "\u{00e9}lodie" => 3.0,
                _ => 1.0,
            }
        });

        assert_eq!(values[0..6], [7.0, 7.0, 1.0, 1.0, 3.0, 3.0]);
        assert!(values[6].is_nan());
        assert_eq!(calls.len(), 3);
        assert!(calls.values().all(|count| *count == 1));
    }

    #[test]
    fn unique_column_lookup_preserves_values_without_redundant_cache() {
        let utf8_key = "\u{00e9}lodie".to_string();
        let keys = vec![
            Some("known".to_string()),
            Some("known".to_string()),
            Some("unknown".to_string()),
            Some("unknown".to_string()),
            Some(utf8_key.clone()),
            Some(utf8_key),
            None,
        ];
        let mut calls = Vec::<String>::new();

        let values = lookup_unique_name_count_column_with(&keys, |key| {
            calls.push(key.to_string());
            match key {
                "known" => 7.0,
                "\u{00e9}lodie" => 3.0,
                _ => 1.0,
            }
        });

        assert_eq!(values[0..6], [7.0, 7.0, 1.0, 1.0, 3.0, 3.0]);
        assert!(values[6].is_nan());
        assert_eq!(calls.len(), 6);
        assert_eq!(calls.iter().filter(|key| *key == "known").count(), 2);
        assert_eq!(calls.iter().filter(|key| *key == "unknown").count(), 2);
        assert_eq!(
            calls
                .iter()
                .filter(|key| key.as_str() == "\u{00e9}lodie")
                .count(),
            2
        );
    }

    #[test]
    fn lookup_columns_reject_misaligned_rows_before_access() {
        assert_eq!(validate_lookup_column_lengths([2, 2, 2, 2]).unwrap(), 2);
        let error =
            validate_lookup_column_lengths([2, 1, 2, 2]).expect_err("misaligned columns must fail");
        let message = py_err_message(error);
        assert!(message.contains("must have equal length"), "{message}");
        assert!(message.contains("first=2 last=1"), "{message}");
    }

    #[test]
    fn open_retains_manifest_normalization_and_provenance_binding() {
        let dir = write_lookup_artifact();
        let index = RawNameCountIndex::open(dir.to_str().expect("utf-8 temp path"))
            .expect("open lookup index");

        assert_eq!(index.normalization_version, "canonical_v2");
        let binding = index
            .provenance_binding
            .as_ref()
            .expect("provenance binding");
        assert_eq!(binding.generation_id, "generation-a");
        assert_eq!(binding.pickle_sha256, "0".repeat(64));
        assert_eq!(binding.source_snapshot_id, "snapshot-a");
        assert_eq!(binding.selected_rows_sha256, "1".repeat(64));
        drop(index);
        std::fs::remove_dir_all(&dir).expect("remove lookup artifact");
    }

    #[test]
    fn explicit_full_validation_rejects_corrupt_record_name_range() {
        let dir = write_lookup_artifact();
        let path = dir.join("first.bin");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open index for corruption");
        file.seek(SeekFrom::Start((32 + 16) as u64))
            .expect("seek to first name offset");
        file.write_all(&u64::MAX.to_le_bytes())
            .expect("write corrupt name offset");
        drop(file);

        // The production opener deliberately performs only O(1) structural
        // checks after Python has verified the immutable file digest.
        let structurally_valid = RawNameCountIndexFile::open(&path, RawNameCountKind::First)
            .expect("header remains structurally valid");
        drop(structurally_valid);
        let error =
            match RawNameCountIndexFile::open_fully_validated(&path, RawNameCountKind::First) {
                Ok(_) => panic!("full corruption validation must reject the invalid record"),
                Err(error) => error,
            };
        let message = py_err_message(error);
        assert!(message.contains("name range overflows"), "{message}");
        std::fs::remove_dir_all(&dir).expect("remove corrupt lookup artifact");
    }

    #[test]
    fn absent_normalization_version_defaults_to_legacy_compat() {
        let dir = write_artifact(None);
        let version = read_version(&dir).expect("manifest without version is valid");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(version, "legacy_compat");
    }

    #[test]
    fn canonical_v2_normalization_version_is_accepted_and_exposed() {
        let dir = write_artifact(Some(r#""canonical_v2""#));
        let version = read_version(&dir).expect("canonical_v2 is a supported version");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(version, "canonical_v2");
    }

    #[test]
    fn unknown_normalization_version_fails_like_the_schema_gate() {
        let dir = write_artifact(Some(r#""canonical_v3""#));
        let error = read_version(&dir).expect_err("unknown version must be rejected");
        let _ = std::fs::remove_dir_all(&dir);
        let message = py_err_message(error);
        assert!(
            message.contains("unsupported normalization_version"),
            "{message}"
        );
        assert!(message.contains("canonical_v3"), "{message}");
    }

    #[test]
    fn non_string_normalization_version_is_rejected() {
        let dir = write_artifact(Some("7"));
        let error = read_version(&dir).expect_err("non-string version must be rejected");
        let _ = std::fs::remove_dir_all(&dir);
        let message = py_err_message(error);
        assert!(
            message.contains("non-string normalization_version"),
            "{message}"
        );
    }
}

fn read_u64_le(bytes: &[u8], offset: usize) -> PyResult<u64> {
    let end = offset.checked_add(8).ok_or_else(|| {
        pyo3::exceptions::PyOverflowError::new_err("u64 offset overflows while reading index")
    })?;
    let slice = bytes.get(offset..end).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("u64 offset is outside name-count index")
    })?;
    Ok(read_u64_le_unchecked(slice, 0))
}

fn read_u64_usize(bytes: &[u8], offset: usize, path: &Path, field_name: &str) -> PyResult<usize> {
    let raw = read_u64_le(bytes, offset)?;
    usize::try_from(raw).map_err(|_| {
        pyo3::exceptions::PyOverflowError::new_err(format!(
            "name-count index file {} field {} overflows usize: {}",
            path.display(),
            field_name,
            raw
        ))
    })
}
