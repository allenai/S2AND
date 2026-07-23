use memmap2::Mmap;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(test)]
use std::io::Read;

use crate::{
    fnv64, fnv64_update, read_f64_le_unchecked, read_u32_le_unchecked, read_u64_le_unchecked,
    FNV_OFFSET,
};

const NAME_COUNTS_INDEX_SCHEMA_VERSION: &str = "name_counts_index_v2";
const NAME_COUNTS_PROVENANCE_SCHEMA_VERSION: &str = "name_counts_provenance_v3";
const NAME_COUNTS_NORMALIZATION_VERSION: &str = "canonical_v2";

#[derive(Clone)]
pub(crate) struct NameCountsData {
    pub(crate) first: f64,
    pub(crate) first_last: f64,
    pub(crate) last: f64,
    pub(crate) last_first_initial: f64,
}

#[derive(Default)]
pub(crate) struct RawNameCountMaps {
    pub(crate) index: Option<Arc<RawNameCountIndex>>,
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
const NAME_COUNTS_SHA256_RECORD_CHUNK_BYTES: usize = 1024 * 1024;

pub(crate) struct RawNameCountIndex {
    first: RawNameCountIndexFile,
    last: RawNameCountIndexFile,
    first_last: RawNameCountIndexFile,
    last_first_initial: RawNameCountIndexFile,
    normalization_version: String,
    identity: NameCountsIndexIdentity,
}

/// Python-facing, immutable handle over four manifest- and digest-verified
/// memory-mapped name-count indexes.
#[pyclass(frozen)]
pub(crate) struct NameCountsIndex {
    index: Arc<RawNameCountIndex>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NameCountsIndexIdentity {
    canonical_root: PathBuf,
    manifest_sha256: String,
}

#[derive(Deserialize)]
struct NameCountsManifest {
    schema_version: String,
    normalization_version: String,
    source_provenance: NameCountsProvenance,
    files: NameCountsManifestFiles,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NameCountsProvenance {
    schema_version: String,
    normalization_version: String,
    generation_id: String,
    source_snapshot_id: String,
    source_kind: String,
    source_query_sha256: String,
    selected_rows_sha256: String,
    #[serde(rename = "source_row_count")]
    _source_row_count: u64,
    #[serde(default, rename = "generated_at")]
    _generated_at: Option<serde_json::Value>,
    #[serde(default, rename = "cardinalities")]
    _cardinalities: Option<serde_json::Value>,
    #[serde(default, rename = "rejected_row_count")]
    _rejected_row_count: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct NameCountsManifestFiles {
    first: NameCountsManifestFile,
    last: NameCountsManifestFile,
    first_last: NameCountsManifestFile,
    last_first_initial: NameCountsManifestFile,
}

#[derive(Deserialize)]
struct NameCountsManifestFile {
    path: String,
    byte_count: u64,
    sha256: String,
}

struct RawNameCountIndexPaths {
    first: RawNameCountIndexFileSpec,
    last: RawNameCountIndexFileSpec,
    first_last: RawNameCountIndexFileSpec,
    last_first_initial: RawNameCountIndexFileSpec,
    normalization_version: String,
    identity: NameCountsIndexIdentity,
}

#[derive(Debug)]
struct RawNameCountIndexFileSpec {
    path: PathBuf,
    expected_sha256: String,
}

impl RawNameCountIndex {
    #[cfg(test)]
    pub(crate) fn open(path: &str) -> PyResult<Self> {
        Self::open_fully_validated(path)
    }

    /// Open and exhaustively validate every record at public artifact boundaries.
    pub(crate) fn open_fully_validated(path: &str) -> PyResult<Self> {
        let paths = resolve_name_counts_index_paths(path)?;
        let ((first, last), (first_last, last_first_initial)) = rayon::join(
            || {
                rayon::join(
                    || {
                        RawNameCountIndexFile::open_manifest_validated(
                            &paths.first.path,
                            RawNameCountKind::First,
                            &paths.first.expected_sha256,
                        )
                    },
                    || {
                        RawNameCountIndexFile::open_manifest_validated(
                            &paths.last.path,
                            RawNameCountKind::Last,
                            &paths.last.expected_sha256,
                        )
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        RawNameCountIndexFile::open_manifest_validated(
                            &paths.first_last.path,
                            RawNameCountKind::FirstLast,
                            &paths.first_last.expected_sha256,
                        )
                    },
                    || {
                        RawNameCountIndexFile::open_manifest_validated(
                            &paths.last_first_initial.path,
                            RawNameCountKind::LastFirstInitial,
                            &paths.last_first_initial.expected_sha256,
                        )
                    },
                )
            },
        );
        Ok(Self {
            first: first?,
            last: last?,
            first_last: first_last?,
            last_first_initial: last_first_initial?,
            normalization_version: paths.normalization_version.clone(),
            identity: paths.identity.clone(),
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
    keys.iter()
        .map(|key| match key {
            None => f64::NAN,
            Some(key) => index.get(kind, key.as_ref()).unwrap_or(1.0),
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

impl NameCountsIndex {
    #[cfg(test)]
    fn open(path: &str) -> PyResult<Self> {
        Ok(Self {
            index: Arc::new(RawNameCountIndex::open_fully_validated(path)?),
        })
    }

    pub(crate) fn from_shared(index: Arc<RawNameCountIndex>) -> Self {
        Self { index }
    }

    pub(crate) fn shared_index(&self) -> Arc<RawNameCountIndex> {
        Arc::clone(&self.index)
    }

    pub(crate) fn validate_path_root(&self, path: &str) -> PyResult<()> {
        let requested_root = canonical_name_counts_index_root(path)?;
        if requested_root != self.index.identity.canonical_root {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "name_counts_index handle does not match paths['name_counts_index']: \
                 handle_root={} requested_root={}",
                self.index.identity.canonical_root.display(),
                requested_root.display(),
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl NameCountsIndex {
    /// Open a manifest-backed name-count index after independently verifying
    /// its schema, provenance, paths, byte counts, and file digests.
    #[staticmethod]
    #[pyo3(name = "open")]
    fn open_py(py: Python<'_>, path: &str) -> PyResult<Self> {
        let index = py.allow_threads(|| RawNameCountIndex::open_fully_validated(path))?;
        Ok(Self {
            index: Arc::new(index),
        })
    }

    #[getter]
    fn normalization_version(&self) -> &str {
        &self.index.normalization_version
    }

    /// Return the digest of the exact manifest snapshot this handle opened.
    #[getter]
    fn name_counts_manifest_sha256(&self) -> &str {
        &self.index.identity.manifest_sha256
    }

    /// Resolve four already-deduplicated aligned optional-key columns.
    ///
    /// A missing key (`None`) produces NaN. An informative string absent from
    /// its index produces the historical default count of 1.0.
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
            lookup_name_count_columns(
                index,
                &first_keys,
                &last_keys,
                &first_last_keys,
                &last_first_initial_keys,
            )
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

struct RawNameCountIndexFile {
    // Memory-mapped index file. Bulk of the file is the variable-length
    // name blob (hundreds of MB per kind); mmap avoids a second allocated
    // copy while cold-open digest and semantic validation stream its pages.
    // Subsequent lookups reuse the validated mapping.
    mmap: Mmap,
    record_count: usize,
    blob_offset: usize,
    blob_len: usize,
}

impl RawNameCountIndexFile {
    fn mmap(path: &Path) -> PyResult<Mmap> {
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
        Ok(mmap)
    }

    fn validated_layout(
        bytes: &[u8],
        path: &Path,
        kind: RawNameCountKind,
    ) -> PyResult<(usize, usize, usize)> {
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
        Ok((record_count, blob_offset, blob_len))
    }

    #[cfg(test)]
    fn open(path: &Path, kind: RawNameCountKind) -> PyResult<Self> {
        let mmap = Self::mmap(path)?;
        let (record_count, blob_offset, blob_len) = Self::validated_layout(&mmap, path, kind)?;
        Ok(Self {
            mmap,
            record_count,
            blob_offset,
            blob_len,
        })
    }

    fn open_manifest_validated(
        path: &Path,
        kind: RawNameCountKind,
        expected_sha256: &str,
    ) -> PyResult<Self> {
        let mmap = Self::mmap(path)?;
        let layout = Self::validated_layout(&mmap, path, kind);
        let (record_count, blob_offset, blob_len) = match layout {
            Ok(layout) => layout,
            Err(layout_error) => {
                validate_name_count_manifest_sha256(&mmap, path, kind, expected_sha256)?;
                return Err(layout_error);
            }
        };
        let index = Self {
            mmap,
            record_count,
            blob_offset,
            blob_len,
        };
        index.validate_all_records(path, kind, expected_sha256)?;
        Ok(index)
    }

    fn validate_record(
        &self,
        path: &Path,
        kind: RawNameCountKind,
        index: usize,
        previous_record: Option<(usize, (u64, u64), usize, usize)>,
    ) -> PyResult<(usize, (u64, u64), usize, usize)> {
        let bytes: &[u8] = &self.mmap;
        let record_offset = NAME_COUNTS_INDEX_HEADER_LEN + index * NAME_COUNTS_INDEX_RECORD_LEN;
        let pair = (
            read_u64_le_unchecked(bytes, record_offset),
            read_u64_le_unchecked(bytes, record_offset + 8),
        );
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
        let reserved = read_u32_le_unchecked(bytes, record_offset + 28);
        if reserved != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index file {} record {} for kind {} has nonzero reserved field: {}",
                path.display(),
                index,
                kind.key(),
                reserved
            )));
        }
        let count = read_f64_le_unchecked(bytes, record_offset + 32);
        if !count.is_finite() || count <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index file {} record {} for kind {} has count {}; expected a finite positive value",
                path.display(),
                index,
                kind.key(),
                count
            )));
        }
        let name_start = self.blob_offset + name_offset;
        let name_end = self.blob_offset + name_end;
        let expected_pair = name_counts_index_hashes(kind, &bytes[name_start..name_end]);
        if pair != expected_pair {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index file {} record {} for kind {} has hash pair {:?}, expected {:?} for its stored name",
                path.display(),
                index,
                kind.key(),
                pair,
                expected_pair
            )));
        }
        if let Some((previous_index, previous_pair, previous_name_start, previous_name_end)) =
            previous_record
        {
            let previous_name = &bytes[previous_name_start..previous_name_end];
            let name = &bytes[name_start..name_end];
            if pair < previous_pair || (pair == previous_pair && name < previous_name) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "name-count index file {} is not sorted for kind {}: record {} ({:?}, {:?}) follows record {} ({:?}, {:?})",
                    path.display(),
                    kind.key(),
                    index,
                    pair,
                    String::from_utf8_lossy(name),
                    previous_index,
                    previous_pair,
                    String::from_utf8_lossy(previous_name)
                )));
            }
            if pair == previous_pair && name == previous_name {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "name-count index file {} contains duplicate UTF-8 name {:?} for kind {} at records {} and {}",
                    path.display(),
                    String::from_utf8_lossy(name),
                    kind.key(),
                    previous_index,
                    index
                )));
            }
        }
        Ok((index, pair, name_start, name_end))
    }

    /// Validate the declared digest and every record in one record-table pass.
    ///
    /// SHA-256 requires file order, while record validation follows offsets
    /// into the later name blob. Hash record-table chunks immediately before
    /// validating their records, then hash the post-record tail once. This
    /// avoids the former separate full-file stream while keeping validation
    /// bounded-memory and SHA updates coarse-grained.
    fn validate_all_records(
        &self,
        path: &Path,
        kind: RawNameCountKind,
        expected_sha256: &str,
    ) -> PyResult<()> {
        let bytes: &[u8] = &self.mmap;
        let records_end =
            NAME_COUNTS_INDEX_HEADER_LEN + self.record_count * NAME_COUNTS_INDEX_RECORD_LEN;
        let mut hasher = Sha256::new();
        hasher.update(&bytes[..NAME_COUNTS_INDEX_HEADER_LEN]);
        let mut previous_record: Option<(usize, (u64, u64), usize, usize)> = None;
        let mut semantic_error = None;
        let records_per_hash_chunk =
            (NAME_COUNTS_SHA256_RECORD_CHUNK_BYTES / NAME_COUNTS_INDEX_RECORD_LEN).max(1);
        for chunk_start in (0..self.record_count).step_by(records_per_hash_chunk) {
            let chunk_end = (chunk_start + records_per_hash_chunk).min(self.record_count);
            let byte_start =
                NAME_COUNTS_INDEX_HEADER_LEN + chunk_start * NAME_COUNTS_INDEX_RECORD_LEN;
            let byte_end = NAME_COUNTS_INDEX_HEADER_LEN + chunk_end * NAME_COUNTS_INDEX_RECORD_LEN;
            hasher.update(&bytes[byte_start..byte_end]);
            for index in chunk_start..chunk_end {
                if semantic_error.is_none() {
                    match self.validate_record(path, kind, index, previous_record) {
                        Ok(validated_record) => previous_record = Some(validated_record),
                        Err(error) => semantic_error = Some(error),
                    }
                }
            }
        }
        hasher.update(&bytes[records_end..]);
        let actual_sha256 = format!("{:x}", hasher.finalize());
        require_name_count_manifest_sha256(path, kind, expected_sha256, &actual_sha256)?;
        if let Some(error) = semantic_error {
            return Err(error);
        }
        Ok(())
    }

    #[cfg(test)]
    fn open_fully_validated(path: &Path, kind: RawNameCountKind) -> PyResult<Self> {
        let index = Self::open(path, kind)?;
        let expected_sha256 = sha256_bytes(&index.mmap);
        index.validate_all_records(path, kind, &expected_sha256)?;
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
        Self {
            index: Some(Arc::new(index)),
        }
    }

    pub(crate) fn from_shared_index(index: Arc<RawNameCountIndex>) -> Self {
        Self { index: Some(index) }
    }

    pub(crate) fn shared_index(&self) -> Option<Arc<RawNameCountIndex>> {
        self.index.as_ref().map(Arc::clone)
    }

    pub(crate) fn has_data(&self) -> bool {
        self.index.is_some()
    }

    pub(crate) fn get(&self, kind: RawNameCountKind, name: &str) -> Option<f64> {
        self.index.as_ref().and_then(|index| index.get(kind, name))
    }

    pub(crate) fn manifest_sha256(&self) -> Option<&str> {
        self.index
            .as_ref()
            .map(|index| index.identity.manifest_sha256.as_str())
    }
}

fn manifest_value_error(manifest_path: &Path, detail: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "name-count index manifest {} {}",
        manifest_path.display(),
        detail
    ))
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|character| character.is_ascii_digit() || (b'a'..=b'f').contains(character))
}

impl NameCountsProvenance {
    fn validate(self, manifest_path: &Path) -> PyResult<()> {
        if self.schema_version != NAME_COUNTS_PROVENANCE_SCHEMA_VERSION {
            return Err(manifest_value_error(
                manifest_path,
                format!(
                    "source_provenance requires schema_version {:?}",
                    NAME_COUNTS_PROVENANCE_SCHEMA_VERSION
                ),
            ));
        }
        if self.normalization_version != NAME_COUNTS_NORMALIZATION_VERSION {
            return Err(manifest_value_error(
                manifest_path,
                format!(
                    "source_provenance has invalid normalization_version; expected {:?}",
                    NAME_COUNTS_NORMALIZATION_VERSION
                ),
            ));
        }
        for (field, value) in [
            ("generation_id", self.generation_id.as_str()),
            ("source_snapshot_id", self.source_snapshot_id.as_str()),
            ("source_kind", self.source_kind.as_str()),
        ] {
            if value.is_empty() {
                return Err(manifest_value_error(
                    manifest_path,
                    format!("source_provenance requires nonempty string {field}"),
                ));
            }
        }
        for (field, value) in [
            ("source_query_sha256", self.source_query_sha256.as_str()),
            ("selected_rows_sha256", self.selected_rows_sha256.as_str()),
        ] {
            if !is_lowercase_sha256(value) {
                return Err(manifest_value_error(
                    manifest_path,
                    format!("source_provenance requires lowercase SHA-256 {field}"),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn sha256_file(path: &Path) -> PyResult<String> {
    let mut input = File::open(path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to open name-count index file {} for SHA-256 verification: {}",
            path.display(),
            err
        ))
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let byte_count = input.read(&mut buffer).map_err(|err| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "failed to read name-count index file {} for SHA-256 verification: {}",
                path.display(),
                err
            ))
        })?;
        if byte_count == 0 {
            break;
        }
        hasher.update(&buffer[..byte_count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn require_name_count_manifest_sha256(
    path: &Path,
    kind: RawNameCountKind,
    expected_sha256: &str,
    actual_sha256: &str,
) -> PyResult<()> {
    if actual_sha256 != expected_sha256 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "name-count index manifest files.{} SHA-256 mismatch: {}",
            kind.key(),
            path.display()
        )));
    }
    Ok(())
}

fn validate_name_count_manifest_sha256(
    bytes: &[u8],
    path: &Path,
    kind: RawNameCountKind,
    expected_sha256: &str,
) -> PyResult<()> {
    let actual_sha256 = sha256_bytes(bytes);
    require_name_count_manifest_sha256(path, kind, expected_sha256, &actual_sha256)
}

fn validated_name_counts_index_manifest_path(
    index_dir: &Path,
    canonical_index_dir: &Path,
    entry: NameCountsManifestFile,
    kind: &str,
    generation_name: &mut Option<String>,
) -> PyResult<RawNameCountIndexFileSpec> {
    let manifest_path = index_dir.join("manifest.json");
    let path_parts = entry.path.split('/').collect::<Vec<_>>();
    if path_parts.len() != 3
        || path_parts[0] != "generations"
        || matches!(path_parts[1], "" | "." | "..")
        || entry.path.contains('\\')
        || path_parts[2] != format!("{kind}.bin")
    {
        return Err(manifest_value_error(
            &manifest_path,
            format!("files.{kind}.path must equal generations/<generation-id>/{kind}.bin"),
        ));
    }
    match generation_name {
        Some(expected) if expected != path_parts[1] => {
            return Err(manifest_value_error(
                &manifest_path,
                "files must share one generation directory",
            ));
        }
        Some(_) => {}
        None => *generation_name = Some(path_parts[1].to_string()),
    }
    if !is_lowercase_sha256(&entry.sha256) {
        return Err(manifest_value_error(
            &manifest_path,
            format!("requires lowercase SHA-256 files.{kind}.sha256"),
        ));
    }
    let resolved = index_dir.join(entry.path);
    let canonical_resolved = fs::canonicalize(&resolved).map_err(|err| {
        pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "name-count index manifest {} points to missing files.{} target {}: {}",
            manifest_path.display(),
            kind,
            resolved.display(),
            err
        ))
    })?;
    if !canonical_resolved.starts_with(canonical_index_dir) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "name-count index manifest {} files.{}.path escapes the name_counts_index directory: {}",
            manifest_path.display(),
            kind,
            canonical_resolved.display()
        )));
    }
    let metadata = fs::metadata(&canonical_resolved).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to inspect name-count index file {}: {}",
            canonical_resolved.display(),
            err
        ))
    })?;
    if !metadata.is_file() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "name-count index manifest {} files.{}.path target is not a file: {}",
            manifest_path.display(),
            kind,
            canonical_resolved.display()
        )));
    }
    let published_marker = canonical_resolved
        .parent()
        .expect("canonical file has a parent")
        .join(".published");
    if !published_marker.is_file() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "name-count index manifest {} files.{} requires published marker: {}",
            manifest_path.display(),
            kind,
            published_marker.display()
        )));
    }
    if metadata.len() != entry.byte_count {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "name-count index manifest {} files.{}.byte_count mismatch: {}",
            manifest_path.display(),
            kind,
            canonical_resolved.display()
        )));
    }
    Ok(RawNameCountIndexFileSpec {
        path: canonical_resolved,
        expected_sha256: entry.sha256,
    })
}

impl NameCountsManifest {
    fn into_paths(
        self,
        index_dir: &Path,
        manifest_sha256: String,
    ) -> PyResult<RawNameCountIndexPaths> {
        let manifest_path = index_dir.join("manifest.json");
        if self.schema_version != NAME_COUNTS_INDEX_SCHEMA_VERSION {
            return Err(manifest_value_error(
                &manifest_path,
                format!(
                    "has unsupported schema_version {:?}; expected {:?}",
                    self.schema_version, NAME_COUNTS_INDEX_SCHEMA_VERSION
                ),
            ));
        }
        if self.normalization_version != NAME_COUNTS_NORMALIZATION_VERSION {
            return Err(manifest_value_error(
                &manifest_path,
                format!(
                    "has unsupported normalization_version {:?}; expected {:?}",
                    self.normalization_version, NAME_COUNTS_NORMALIZATION_VERSION
                ),
            ));
        }
        self.source_provenance.validate(&manifest_path)?;
        let canonical_index_dir = fs::canonicalize(index_dir).map_err(|err| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "failed to resolve name-count index directory {}: {}",
                index_dir.display(),
                err
            ))
        })?;
        let files = self.files;
        let entries = [
            ("first", files.first),
            ("last", files.last),
            ("first_last", files.first_last),
            ("last_first_initial", files.last_first_initial),
        ];
        let mut generation_name = None;
        let paths: [RawNameCountIndexFileSpec; 4] = entries
            .into_iter()
            .map(|(kind, entry)| {
                validated_name_counts_index_manifest_path(
                    index_dir,
                    &canonical_index_dir,
                    entry,
                    kind,
                    &mut generation_name,
                )
            })
            .collect::<PyResult<Vec<_>>>()?
            .try_into()
            .expect("four manifest file entries");
        let [first, last, first_last, last_first_initial] = paths;
        Ok(RawNameCountIndexPaths {
            first,
            last,
            first_last,
            last_first_initial,
            normalization_version: self.normalization_version,
            identity: NameCountsIndexIdentity {
                canonical_root: canonical_index_dir,
                manifest_sha256,
            },
        })
    }
}

fn read_name_counts_index_manifest(index_dir: &Path) -> PyResult<RawNameCountIndexPaths> {
    let manifest_path = index_dir.join("manifest.json");
    let manifest_bytes = fs::read(&manifest_path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to read name-count index manifest {}: {}",
            manifest_path.display(),
            err
        ))
    })?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let manifest: NameCountsManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|err| manifest_value_error(&manifest_path, format!("failed to parse: {err}")))?;
    manifest.into_paths(index_dir, manifest_sha256)
}

fn unresolved_name_counts_index_root(path: &str) -> PyResult<PathBuf> {
    let direct = PathBuf::from(path);
    let nested = direct.join("name_counts_index");
    for index_dir in [&direct, &nested] {
        if index_dir.join("manifest.json").exists() {
            return Ok(index_dir.clone());
        }
    }
    Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
        "name-count index path {} does not contain manifest.json",
        path
    )))
}

fn canonical_name_counts_index_root(path: &str) -> PyResult<PathBuf> {
    let index_root = unresolved_name_counts_index_root(path)?;
    fs::canonicalize(&index_root).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to resolve name-count index directory {}: {}",
            index_root.display(),
            err
        ))
    })
}

fn resolve_name_counts_index_paths(path: &str) -> PyResult<RawNameCountIndexPaths> {
    let index_root = unresolved_name_counts_index_root(path)?;
    read_name_counts_index_manifest(&index_root)
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
        lookup_name_count_column, name_counts_index_hashes, resolve_name_counts_index_paths,
        sha256_file, validate_lookup_column_lengths, NameCountsIndex, NameCountsProvenance,
        RawNameCountIndex, RawNameCountIndexFile, RawNameCountKind, RawNameCountMaps,
        NAME_COUNTS_INDEX_HEADER_LEN, NAME_COUNTS_INDEX_RECORD_LEN,
    };
    use std::io::{Seek, SeekFrom, Write};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

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

    fn complete_source_provenance() -> serde_json::Value {
        serde_json::json!({
            "schema_version": "name_counts_provenance_v3",
            "normalization_version": "canonical_v2",
            "generation_id": "generation-a",
            "source_snapshot_id": "snapshot-a",
            "source_kind": "test-fixture",
            "source_query_sha256": "2".repeat(64),
            "selected_rows_sha256": "1".repeat(64),
            "source_row_count": 0,
        })
    }

    const TEST_GENERATION_NAME: &str = "generation-a";

    fn index_file_path(index_dir: &std::path::Path, kind: &str) -> std::path::PathBuf {
        index_dir
            .join("generations")
            .join(TEST_GENERATION_NAME)
            .join(format!("{kind}.bin"))
    }

    fn manifest_file_entry(index_dir: &std::path::Path, kind: &str) -> serde_json::Value {
        let path = index_file_path(index_dir, kind);
        serde_json::json!({
            "path": format!("generations/{TEST_GENERATION_NAME}/{kind}.bin"),
            "byte_count": path.metadata().expect("file metadata").len(),
            "sha256": sha256_file(&path).expect("hash fixture"),
        })
    }

    #[test]
    fn sha256_file_matches_known_digest() {
        let dir = unique_temp_dir("s2and_sha256");
        let path = dir.join("fixture.bin");
        std::fs::write(&path, b"abc").expect("write hash fixture");

        assert_eq!(
            sha256_file(&path).expect("hash fixture"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    fn write_artifact(normalization_version_json: Option<&str>) -> std::path::PathBuf {
        let dir = unique_temp_dir("s2and_nc_version");
        let generation_dir = dir.join("generations").join(TEST_GENERATION_NAME);
        std::fs::create_dir_all(&generation_dir).expect("create generation dir");
        let mut files = serde_json::Map::new();
        for name in ["first", "last", "first_last", "last_first_initial"] {
            let path = index_file_path(&dir, name);
            let mut file = std::fs::File::create(&path).expect("create index file");
            file.write_all(&empty_index_bytes())
                .expect("write index header");
            drop(file);
            files.insert(name.to_string(), manifest_file_entry(&dir, name));
        }
        std::fs::write(generation_dir.join(".published"), []).expect("write published marker");
        let mut manifest = serde_json::json!({
            "schema_version": "name_counts_index_v2",
            "source_provenance": complete_source_provenance(),
            "files": files,
        });
        if let Some(value) = normalization_version_json {
            manifest["normalization_version"] =
                serde_json::from_str(value).expect("valid test JSON value");
        }
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
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
        let generation_dir = dir.join("generations").join(TEST_GENERATION_NAME);
        std::fs::create_dir_all(&generation_dir).expect("create generation dir");
        write_index_file(
            &index_file_path(&dir, "first"),
            RawNameCountKind::First,
            &[("alice", 7.0), ("élodie", 3.0)],
        );
        write_index_file(
            &index_file_path(&dir, "last"),
            RawNameCountKind::Last,
            &[("smith", 11.0)],
        );
        write_index_file(
            &index_file_path(&dir, "first_last"),
            RawNameCountKind::FirstLast,
            &[("alice smith", 5.0)],
        );
        write_index_file(
            &index_file_path(&dir, "last_first_initial"),
            RawNameCountKind::LastFirstInitial,
            &[("smith a", 9.0)],
        );
        std::fs::write(generation_dir.join(".published"), []).expect("write published marker");
        let manifest = serde_json::json!({
            "schema_version": "name_counts_index_v2",
            "normalization_version": "canonical_v2",
            "source_provenance": complete_source_provenance(),
            "files": {
                "first": manifest_file_entry(&dir, "first"),
                "last": manifest_file_entry(&dir, "last"),
                "first_last": manifest_file_entry(&dir, "first_last"),
                "last_first_initial": manifest_file_entry(&dir, "last_first_initial"),
            },
        });
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write lookup manifest");
        dir
    }

    fn refresh_manifest_file_digest(dir: &std::path::Path, kind: &str) {
        let manifest_path = dir.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).expect("read manifest"))
                .expect("parse manifest");
        let path = index_file_path(dir, kind);
        manifest["files"][kind]["sha256"] =
            serde_json::Value::String(sha256_file(&path).expect("hash mutated index"));
        std::fs::write(
            &manifest_path,
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write checksum-consistent manifest");
    }

    fn read_version(dir: &std::path::Path) -> pyo3::PyResult<String> {
        Ok(
            RawNameCountIndex::open_fully_validated(dir.to_str().expect("utf-8 temp path"))?
                .normalization_version,
        )
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
    fn source_provenance_is_validated() {
        let provenance: NameCountsProvenance =
            serde_json::from_value(complete_source_provenance()).expect("valid provenance shape");
        provenance
            .validate(std::path::Path::new("manifest.json"))
            .expect("valid provenance");
    }

    #[test]
    fn source_provenance_rejects_non_sha_digest() {
        let mut provenance = complete_source_provenance();
        provenance["selected_rows_sha256"] = serde_json::Value::String("G".repeat(64));
        let provenance: NameCountsProvenance =
            serde_json::from_value(provenance).expect("valid provenance shape");
        let error = provenance
            .validate(std::path::Path::new("manifest.json"))
            .expect_err("invalid digest must fail");

        assert!(py_err_message(error).contains("lowercase SHA-256 selected_rows_sha256"));
    }

    #[test]
    fn source_provenance_rejects_every_required_lineage_field() {
        for field in ["source_kind", "source_query_sha256", "source_row_count"] {
            let mut provenance = complete_source_provenance();
            provenance
                .as_object_mut()
                .expect("provenance object")
                .remove(field);
            let message = match serde_json::from_value::<NameCountsProvenance>(provenance) {
                Ok(_) => panic!("missing provenance field must fail"),
                Err(error) => error.to_string(),
            };
            assert!(message.contains(field), "{field}: {message}");
        }
    }

    #[test]
    fn source_provenance_rejects_removed_selected_row_count() {
        let mut provenance = complete_source_provenance();
        provenance["selected_row_count"] = serde_json::Value::from(0);

        let error = serde_json::from_value::<NameCountsProvenance>(provenance)
            .err()
            .expect("removed selected_row_count must fail");
        assert!(error.to_string().contains("selected_row_count"));
    }

    #[test]
    fn manifest_rejects_declared_sha256_mismatch() {
        let dir = write_lookup_artifact();
        let path = index_file_path(&dir, "first");
        let mut bytes = std::fs::read(&path).expect("read first index");
        bytes[0] ^= 1;
        std::fs::write(&path, bytes).expect("corrupt first index");

        let error = match RawNameCountIndex::open(dir.to_str().expect("utf-8 temp path")) {
            Ok(_) => panic!("digest mismatch must fail"),
            Err(error) => error,
        };
        let message = py_err_message(error);
        assert!(
            message.contains("files.first SHA-256 mismatch"),
            "{message}"
        );
        std::fs::remove_dir_all(&dir).expect("remove corrupt artifact");
    }

    #[test]
    fn manifest_resolution_defers_payload_digest_to_integrated_validation() {
        let dir = write_lookup_artifact();
        let path = index_file_path(&dir, "first");
        let mut bytes = std::fs::read(&path).expect("read first index");
        let last = bytes.last_mut().expect("fixture has a name blob");
        *last ^= 1;
        std::fs::write(&path, bytes).expect("corrupt first index payload");

        resolve_name_counts_index_paths(dir.to_str().expect("utf-8 temp path"))
            .expect("manifest resolution must remain metadata-only");
        let error = match NameCountsIndex::open(dir.to_str().expect("utf-8 temp path")) {
            Ok(_) => panic!("integrated validation must enforce the declared digest"),
            Err(error) => error,
        };
        let message = py_err_message(error);
        assert!(
            message.contains("files.first SHA-256 mismatch"),
            "{message}"
        );
        std::fs::remove_dir_all(&dir).expect("remove corrupt artifact");
    }

    #[test]
    fn integrated_digest_mismatch_precedes_record_semantic_error() {
        let dir = write_lookup_artifact();
        let path = index_file_path(&dir, "first");
        let mut bytes = std::fs::read(&path).expect("read first index");
        let name_offset = NAME_COUNTS_INDEX_HEADER_LEN + 16;
        bytes[name_offset..name_offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, bytes).expect("corrupt first record name offset");

        let digest_error = match NameCountsIndex::open(dir.to_str().expect("utf-8 temp path")) {
            Ok(_) => panic!("undeclared semantic corruption must fail its digest first"),
            Err(error) => error,
        };
        assert!(py_err_message(digest_error).contains("files.first SHA-256 mismatch"));

        refresh_manifest_file_digest(&dir, "first");
        let semantic_error = match NameCountsIndex::open(dir.to_str().expect("utf-8 temp path")) {
            Ok(_) => panic!("checksum-consistent semantic corruption must still fail"),
            Err(error) => error,
        };
        assert!(py_err_message(semantic_error).contains("name range overflows"));
        std::fs::remove_dir_all(&dir).expect("remove corrupt artifact");
    }

    #[test]
    fn native_handle_rejects_checksum_consistent_unsorted_records() {
        let dir = write_lookup_artifact();
        let path = index_file_path(&dir, "first");
        let mut bytes = std::fs::read(&path).expect("read first index");
        let first_start = NAME_COUNTS_INDEX_HEADER_LEN;
        let second_start = first_start + NAME_COUNTS_INDEX_RECORD_LEN;
        let records_end = second_start + NAME_COUNTS_INDEX_RECORD_LEN;
        let first_record = bytes[first_start..second_start].to_vec();
        let second_record = bytes[second_start..records_end].to_vec();
        bytes[first_start..second_start].copy_from_slice(&second_record);
        bytes[second_start..records_end].copy_from_slice(&first_record);
        std::fs::write(&path, bytes).expect("write unsorted first index");
        refresh_manifest_file_digest(&dir, "first");

        let error = match NameCountsIndex::open(dir.to_str().expect("utf-8 temp path")) {
            Ok(_) => panic!("public native handle must reject unsorted records"),
            Err(error) => error,
        };
        let message = py_err_message(error);
        assert!(
            message.contains("is not sorted for kind first"),
            "{message}"
        );
        std::fs::remove_dir_all(&dir).expect("remove corrupt artifact");
    }

    #[test]
    fn native_handle_rejects_checksum_consistent_hash_name_mismatch() {
        let dir = write_lookup_artifact();
        let path = index_file_path(&dir, "first");
        let mut bytes = std::fs::read(&path).expect("read first index");
        let first_start = NAME_COUNTS_INDEX_HEADER_LEN;
        let second_start = first_start + NAME_COUNTS_INDEX_RECORD_LEN;
        let first_pair = bytes[first_start..first_start + 16].to_vec();
        bytes[second_start..second_start + 16].copy_from_slice(&first_pair);
        std::fs::write(&path, bytes).expect("write hash/name mismatch");
        refresh_manifest_file_digest(&dir, "first");

        let error = match NameCountsIndex::open(dir.to_str().expect("utf-8 temp path")) {
            Ok(_) => panic!("public native handle must reject a hash/name mismatch"),
            Err(error) => error,
        };
        let message = py_err_message(error);
        assert!(message.contains("has hash pair"), "{message}");
        assert!(message.contains("expected"), "{message}");
        std::fs::remove_dir_all(&dir).expect("remove corrupt artifact");
    }

    #[test]
    fn native_handle_rejects_checksum_consistent_nonzero_reserved_field() {
        let dir = write_lookup_artifact();
        let path = index_file_path(&dir, "first");
        let mut bytes = std::fs::read(&path).expect("read first index");
        let reserved_offset = NAME_COUNTS_INDEX_HEADER_LEN + 28;
        bytes[reserved_offset..reserved_offset + 4].copy_from_slice(&1_u32.to_le_bytes());
        std::fs::write(&path, bytes).expect("write nonzero reserved field");
        refresh_manifest_file_digest(&dir, "first");

        let error = match NameCountsIndex::open(dir.to_str().expect("utf-8 temp path")) {
            Ok(_) => panic!("public native handle must reject a nonzero reserved field"),
            Err(error) => error,
        };
        let message = py_err_message(error);
        assert!(message.contains("nonzero reserved field"), "{message}");
        std::fs::remove_dir_all(&dir).expect("remove corrupt artifact");
    }

    #[test]
    fn native_handle_rejects_checksum_consistent_duplicate_name() {
        let dir = write_lookup_artifact();
        let path = index_file_path(&dir, "first");
        let mut bytes = std::fs::read(&path).expect("read first index");
        let first_start = NAME_COUNTS_INDEX_HEADER_LEN;
        let second_start = first_start + NAME_COUNTS_INDEX_RECORD_LEN;
        let first_record = bytes[first_start..second_start].to_vec();
        bytes[second_start..second_start + NAME_COUNTS_INDEX_RECORD_LEN]
            .copy_from_slice(&first_record);
        std::fs::write(&path, bytes).expect("write duplicate first-name record");
        refresh_manifest_file_digest(&dir, "first");

        let error = match NameCountsIndex::open(dir.to_str().expect("utf-8 temp path")) {
            Ok(_) => panic!("public native handle must reject duplicate names"),
            Err(error) => error,
        };
        let message = py_err_message(error);
        assert!(message.contains("duplicate UTF-8 name"), "{message}");
        std::fs::remove_dir_all(&dir).expect("remove corrupt artifact");
    }

    #[test]
    fn native_handle_rejects_checksum_consistent_invalid_counts() {
        for (case, count) in [
            ("nan", f64::NAN),
            ("positive_infinity", f64::INFINITY),
            ("zero", 0.0),
            ("negative", -3.0),
        ] {
            let dir = write_lookup_artifact();
            let path = index_file_path(&dir, "first");
            let mut bytes = std::fs::read(&path).expect("read first index");
            let count_offset = NAME_COUNTS_INDEX_HEADER_LEN + 32;
            bytes[count_offset..count_offset + 8].copy_from_slice(&count.to_le_bytes());
            std::fs::write(&path, bytes).expect("write invalid first-name count");
            refresh_manifest_file_digest(&dir, "first");

            let error = match NameCountsIndex::open(dir.to_str().expect("utf-8 temp path")) {
                Ok(_) => panic!("public native handle must reject {case} count"),
                Err(error) => error,
            };
            let message = py_err_message(error);
            assert!(
                message.contains("finite positive value"),
                "{case}: {message}"
            );
            std::fs::remove_dir_all(&dir).expect("remove corrupt artifact");
        }
    }

    #[test]
    fn manifest_rejects_legacy_direct_file_path() {
        let dir = write_lookup_artifact();
        let manifest_path = dir.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).expect("read manifest"))
                .expect("parse manifest");
        manifest["files"]["first"]["path"] = serde_json::Value::String("first.bin".to_string());
        std::fs::write(
            &manifest_path,
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        let error = match RawNameCountIndex::open(dir.to_str().expect("utf-8 temp path")) {
            Ok(_) => panic!("legacy direct path must fail"),
            Err(error) => error,
        };
        let message = py_err_message(error);
        assert!(
            message.contains("files.first.path must equal generations/<generation-id>/first.bin"),
            "{message}"
        );
        std::fs::remove_dir_all(&dir).expect("remove artifact");
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
    fn native_handle_and_raw_maps_share_one_index() {
        let dir = write_lookup_artifact();
        let shared = Arc::new(
            RawNameCountIndex::open(dir.to_str().expect("utf-8 temp path"))
                .expect("open lookup index"),
        );
        let maps = RawNameCountMaps::from_shared_index(Arc::clone(&shared));
        let handle =
            NameCountsIndex::from_shared(maps.shared_index().expect("maps retain a shared index"));
        let handle_index = handle.shared_index();

        assert!(Arc::ptr_eq(&shared, &handle_index));

        drop(handle_index);
        drop(handle);
        drop(maps);
        drop(shared);
        std::fs::remove_dir_all(&dir).expect("remove lookup artifact");
    }

    #[test]
    fn native_handle_requires_matching_root_but_keeps_opened_manifest_snapshot() {
        let first = write_lookup_artifact();
        let second = write_lookup_artifact();
        let handle = NameCountsIndex::from_shared(Arc::new(
            RawNameCountIndex::open(first.to_str().expect("utf-8 temp path"))
                .expect("open lookup index"),
        ));

        handle
            .validate_path_root(first.to_str().expect("utf-8 temp path"))
            .expect("same root must match");
        let root_mismatch = handle
            .validate_path_root(second.to_str().expect("utf-8 temp path"))
            .expect_err("different root must fail");
        assert!(py_err_message(root_mismatch).contains("does not match paths['name_counts_index']"));

        let manifest_path = first.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).expect("read manifest"))
                .expect("parse manifest");
        manifest["source_provenance"]["generation_id"] =
            serde_json::Value::String("generation-b".to_string());
        std::fs::write(
            &manifest_path,
            serde_json::to_vec(&manifest).expect("serialize replacement manifest"),
        )
        .expect("publish replacement manifest");
        handle
            .validate_path_root(first.to_str().expect("utf-8 temp path"))
            .expect("opened snapshot remains authoritative after same-root publication");

        drop(handle);
        std::fs::remove_dir_all(&first).expect("remove first lookup artifact");
        std::fs::remove_dir_all(&second).expect("remove second lookup artifact");
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
    fn open_retains_manifest_normalization_and_sha256() {
        let dir = write_lookup_artifact();
        let expected_manifest_sha256 =
            sha256_file(&dir.join("manifest.json")).expect("hash manifest");
        let index = RawNameCountIndex::open(dir.to_str().expect("utf-8 temp path"))
            .expect("open lookup index");

        assert_eq!(index.normalization_version, "canonical_v2");
        assert_eq!(index.identity.manifest_sha256, expected_manifest_sha256);
        drop(index);
        std::fs::remove_dir_all(&dir).expect("remove lookup artifact");
    }

    #[test]
    fn explicit_full_validation_rejects_corrupt_record_name_range() {
        let dir = write_lookup_artifact();
        let path = index_file_path(&dir, "first");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open index for corruption");
        file.seek(SeekFrom::Start((32 + 16) as u64))
            .expect("seek to first name offset");
        file.write_all(&u64::MAX.to_le_bytes())
            .expect("write corrupt name offset");
        drop(file);

        // The low-level file opener deliberately performs only O(1) structural
        // checks; the manifest boundary verifies the immutable file digest.
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
    fn absent_normalization_version_is_rejected() {
        let dir = write_artifact(None);
        let error = read_version(&dir).expect_err("manifest without version must fail");
        let _ = std::fs::remove_dir_all(&dir);
        let message = py_err_message(error);
        assert!(message.contains("missing field"), "{message}");
        assert!(message.contains("normalization_version"), "{message}");
    }

    #[test]
    fn canonical_v2_normalization_version_is_accepted_and_exposed() {
        let dir = write_artifact(Some(r#""canonical_v2""#));
        let version = read_version(&dir).expect("canonical_v2 is a supported version");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(version, "canonical_v2");
    }

    #[test]
    fn legacy_compat_normalization_version_is_rejected() {
        let dir = write_artifact(Some(r#""legacy_compat""#));
        let error = read_version(&dir).expect_err("legacy compatibility mode must fail");
        let _ = std::fs::remove_dir_all(&dir);
        let message = py_err_message(error);
        assert!(
            message.contains("unsupported normalization_version"),
            "{message}"
        );
        assert!(message.contains("legacy_compat"), "{message}");
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
        assert!(message.contains("expected a string"), "{message}");
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
