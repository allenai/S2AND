use memmap2::Mmap;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::{
    fnv64, fnv64_update, read_f64_le_unchecked, read_u32_le_unchecked, read_u64_le_unchecked,
    FNV_OFFSET,
};

const NAME_COUNTS_INDEX_SCHEMA_VERSION: &str = "name_counts_index_v1";
const NAME_COUNTS_PROVENANCE_SCHEMA_VERSION: &str = "name_counts_provenance_v1";
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

pub(crate) struct RawNameCountIndex {
    first: RawNameCountIndexFile,
    last: RawNameCountIndexFile,
    first_last: RawNameCountIndexFile,
    last_first_initial: RawNameCountIndexFile,
    normalization_version: String,
    provenance_binding: NameCountsProvenanceBinding,
}

/// Python-facing, immutable handle over four manifest- and digest-verified
/// memory-mapped name-count indexes.
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

#[derive(Deserialize)]
struct NameCountsManifest {
    schema_version: String,
    normalization_version: String,
    source_provenance: NameCountsProvenance,
    files: NameCountsManifestFiles,
}

#[derive(Deserialize)]
struct NameCountsProvenance {
    schema_version: String,
    normalization_version: String,
    generation_id: String,
    pickle_sha256: String,
    source_snapshot_id: String,
    source_kind: String,
    source_query_sha256: String,
    selected_rows_sha256: String,
    selected_row_count: u64,
    source_row_count: u64,
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
    first: PathBuf,
    last: PathBuf,
    first_last: PathBuf,
    last_first_initial: PathBuf,
    normalization_version: String,
    provenance_binding: NameCountsProvenanceBinding,
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

#[pymethods]
impl NameCountsIndex {
    /// Open a manifest-backed name-count index after independently verifying
    /// its schema, provenance, paths, byte counts, and file digests.
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
    fn name_counts_provenance_binding(&self) -> (String, String, String, String) {
        let binding = &self.index.provenance_binding;
        (
            binding.generation_id.clone(),
            binding.pickle_sha256.clone(),
            binding.source_snapshot_id.clone(),
            binding.selected_rows_sha256.clone(),
        )
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
        self.index.as_ref().map(|index| &index.provenance_binding)
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
    fn into_binding(self, manifest_path: &Path) -> PyResult<NameCountsProvenanceBinding> {
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
            ("pickle_sha256", self.pickle_sha256.as_str()),
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
        if self.source_row_count != self.selected_row_count {
            return Err(manifest_value_error(
                manifest_path,
                "source_provenance selected_row_count/source_row_count mismatch",
            ));
        }
        Ok(NameCountsProvenanceBinding {
            generation_id: self.generation_id,
            pickle_sha256: self.pickle_sha256,
            source_snapshot_id: self.source_snapshot_id,
            selected_rows_sha256: self.selected_rows_sha256,
        })
    }
}

pub(crate) fn python_sha256_file(path: &Path) -> PyResult<String> {
    #[cfg(windows)]
    if let Some(python_home) = option_env!("S2AND_RUST_PYTHONHOME") {
        std::env::set_var("PYTHONHOME", python_home);
    }
    pyo3::prepare_freethreaded_python();
    let mut input = File::open(path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to open name-count index file {} for SHA-256 verification: {}",
            path.display(),
            err
        ))
    })?;
    Python::with_gil(|py| {
        use pyo3::types::PyBytes;

        let hasher = py.import("hashlib")?.call_method0("sha256")?;
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
            hasher.call_method1("update", (PyBytes::new(py, &buffer[..byte_count]),))?;
        }
        hasher.call_method0("hexdigest")?.extract()
    })
}

fn validated_name_counts_index_manifest_path(
    index_dir: &Path,
    canonical_index_dir: &Path,
    entry: NameCountsManifestFile,
    kind: &str,
) -> PyResult<PathBuf> {
    let manifest_path = index_dir.join("manifest.json");
    if entry.path.trim().is_empty() {
        return Err(manifest_value_error(
            &manifest_path,
            format!("requires nonempty string files.{kind}.path"),
        ));
    }
    if !is_lowercase_sha256(&entry.sha256) {
        return Err(manifest_value_error(
            &manifest_path,
            format!("requires lowercase SHA-256 files.{kind}.sha256"),
        ));
    }
    let raw_path = PathBuf::from(entry.path);
    let resolved = if raw_path.is_absolute() {
        raw_path
    } else {
        index_dir.join(raw_path)
    };
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
    let actual_sha256 = python_sha256_file(&canonical_resolved)?;
    if actual_sha256 != entry.sha256 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "name-count index manifest {} files.{} SHA-256 mismatch: {}",
            manifest_path.display(),
            kind,
            canonical_resolved.display()
        )));
    }
    Ok(canonical_resolved)
}

impl NameCountsManifest {
    fn into_paths(self, index_dir: &Path) -> PyResult<RawNameCountIndexPaths> {
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
        let provenance_binding = self.source_provenance.into_binding(&manifest_path)?;
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
        let paths: [PathBuf; 4] = entries
            .into_iter()
            .map(|(kind, entry)| {
                validated_name_counts_index_manifest_path(
                    index_dir,
                    &canonical_index_dir,
                    entry,
                    kind,
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
            provenance_binding,
        })
    }
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
    let manifest: NameCountsManifest = serde_json::from_str(&manifest_text)
        .map_err(|err| manifest_value_error(&manifest_path, format!("failed to parse: {err}")))?;
    manifest.into_paths(index_dir)
}

/// Return the `normalization_version` from a fully validated name-count index.
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
        lookup_name_count_column, name_counts_index_hashes, python_sha256_file,
        read_name_counts_index_normalization_version, validate_lookup_column_lengths,
        NameCountsProvenance, RawNameCountIndex, RawNameCountIndexFile, RawNameCountKind,
    };
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

    fn complete_source_provenance() -> serde_json::Value {
        serde_json::json!({
            "schema_version": "name_counts_provenance_v1",
            "normalization_version": "canonical_v2",
            "generation_id": "generation-a",
            "pickle_sha256": "0".repeat(64),
            "source_snapshot_id": "snapshot-a",
            "source_kind": "test-fixture",
            "source_query_sha256": "2".repeat(64),
            "selected_rows_sha256": "1".repeat(64),
            "selected_row_count": 0,
            "source_row_count": 0,
        })
    }

    fn manifest_file_entry(path: &std::path::Path) -> serde_json::Value {
        serde_json::json!({
            "path": path.file_name().expect("file name").to_str().expect("utf-8 file name"),
            "byte_count": path.metadata().expect("file metadata").len(),
            "sha256": python_sha256_file(path).expect("hash fixture"),
        })
    }

    fn write_artifact(normalization_version_json: Option<&str>) -> std::path::PathBuf {
        let dir = unique_temp_dir("s2and_nc_version");
        let mut files = serde_json::Map::new();
        for name in ["first", "last", "first_last", "last_first_initial"] {
            let path = dir.join(format!("{name}.bin"));
            let mut file = std::fs::File::create(&path).expect("create index file");
            file.write_all(&empty_index_bytes())
                .expect("write index header");
            drop(file);
            files.insert(name.to_string(), manifest_file_entry(&path));
        }
        std::fs::write(dir.join(".published"), []).expect("write published marker");
        let mut manifest = serde_json::json!({
            "schema_version": "name_counts_index_v1",
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
        std::fs::write(dir.join(".published"), []).expect("write published marker");
        let manifest = serde_json::json!({
            "schema_version": "name_counts_index_v1",
            "normalization_version": "canonical_v2",
            "source_provenance": complete_source_provenance(),
            "files": {
                "first": manifest_file_entry(&dir.join("first.bin")),
                "last": manifest_file_entry(&dir.join("last.bin")),
                "first_last": manifest_file_entry(&dir.join("first_last.bin")),
                "last_first_initial": manifest_file_entry(&dir.join("last_first_initial.bin")),
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
        let provenance: NameCountsProvenance =
            serde_json::from_value(complete_source_provenance()).expect("valid provenance shape");
        let binding = provenance
            .into_binding(std::path::Path::new("manifest.json"))
            .expect("valid provenance");

        assert_eq!(binding.generation_id, "generation-a");
        assert_eq!(binding.pickle_sha256, "0".repeat(64));
        assert_eq!(binding.source_snapshot_id, "snapshot-a");
        assert_eq!(binding.selected_rows_sha256, "1".repeat(64));
    }

    #[test]
    fn source_provenance_binding_rejects_non_sha_digest() {
        let mut provenance = complete_source_provenance();
        provenance["pickle_sha256"] = serde_json::Value::String("G".repeat(64));
        let provenance: NameCountsProvenance =
            serde_json::from_value(provenance).expect("valid provenance shape");
        let error = provenance
            .into_binding(std::path::Path::new("manifest.json"))
            .expect_err("invalid digest must fail");

        assert!(py_err_message(error).contains("lowercase SHA-256 pickle_sha256"));
    }

    #[test]
    fn source_provenance_rejects_every_required_lineage_field() {
        for field in [
            "source_kind",
            "source_query_sha256",
            "selected_row_count",
            "source_row_count",
        ] {
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
    fn manifest_rejects_declared_sha256_mismatch() {
        let dir = write_lookup_artifact();
        let path = dir.join("first.bin");
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
    fn manifest_rejects_file_path_outside_index_directory() {
        let dir = write_lookup_artifact();
        let other = write_lookup_artifact();
        let manifest_path = dir.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).expect("read manifest"))
                .expect("parse manifest");
        manifest["files"]["first"]["path"] =
            serde_json::Value::String(other.join("first.bin").display().to_string());
        std::fs::write(
            &manifest_path,
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        let error = match RawNameCountIndex::open(dir.to_str().expect("utf-8 temp path")) {
            Ok(_) => panic!("escaping path must fail"),
            Err(error) => error,
        };
        let message = py_err_message(error);
        assert!(
            message.contains("files.first.path escapes the name_counts_index directory"),
            "{message}"
        );
        std::fs::remove_dir_all(&dir).expect("remove artifact");
        std::fs::remove_dir_all(&other).expect("remove outside artifact");
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
        let binding = &index.provenance_binding;
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
