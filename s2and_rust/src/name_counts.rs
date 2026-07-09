use memmap2::Mmap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
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

pub(crate) struct RawNameCountIndex {
    first: RawNameCountIndexFile,
    last: RawNameCountIndexFile,
    first_last: RawNameCountIndexFile,
    last_first_initial: RawNameCountIndexFile,
}

struct RawNameCountIndexPaths {
    first: PathBuf,
    last: PathBuf,
    first_last: PathBuf,
    last_first_initial: PathBuf,
    normalization_version: String,
}

impl RawNameCountIndex {
    pub(crate) fn open(path: &str) -> PyResult<Self> {
        let paths = resolve_name_counts_index_paths(path)?;
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
        for index in 0..record_count {
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
            if name_end > blob_len {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "name-count index file {} record {} for kind {} has name range [{}, {}) outside blob length {}",
                    path.display(),
                    index,
                    kind.key(),
                    name_offset,
                    name_end,
                    blob_len
                )));
            }
        }
        if record_count > 1 {
            let read_pair = |index: usize| {
                let offset = NAME_COUNTS_INDEX_HEADER_LEN + index * NAME_COUNTS_INDEX_RECORD_LEN;
                (
                    read_u64_le_unchecked(bytes, offset),
                    read_u64_le_unchecked(bytes, offset + 8),
                )
            };
            let mut previous_index = 0usize;
            let mut previous_pair = read_pair(0);
            for index in 1..record_count {
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
        Ok(Self {
            mmap,
            record_count,
            blob_offset,
            blob_len,
        })
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
        let mut lower = 0usize;
        let mut upper = self.record_count;
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            let (middle_hash_1, middle_hash_2) = self.record_hash_pair(middle);
            if middle_hash_1 < hash_1 || (middle_hash_1 == hash_1 && middle_hash_2 < hash_2) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }

        let mut index = lower;
        while index < self.record_count {
            let (record_hash_1, record_hash_2) = self.record_hash_pair(index);
            if record_hash_1 != hash_1 || record_hash_2 != hash_2 {
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
        Self { index: Some(index) }
    }

    pub(crate) fn has_data(&self) -> bool {
        self.index.is_some()
    }

    pub(crate) fn get(&self, kind: RawNameCountKind, name: &str) -> Option<f64> {
        self.index.as_ref().and_then(|index| index.get(kind, name))
    }
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
mod normalization_version_tests {
    use super::read_name_counts_index_normalization_version;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    // A valid 32-byte name-count index header describing zero records; the
    // manifest reader only checks the files exist.
    fn empty_index_bytes() -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(b"S2NCI001");
        bytes[16..24].copy_from_slice(&32u64.to_le_bytes());
        bytes
    }

    fn write_artifact(normalization_version_json: Option<&str>) -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = format!(
            "s2and_nc_version_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).expect("create temp index dir");
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
