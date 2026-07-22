use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::record_batch::RecordBatch;
use memmap2::Mmap;
use pyo3::PyResult;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Read;

use crate::raw_arrow::arrow_io::{arrow_error_to_py, io_error_to_py};
use crate::{fnv64, fnv64_update};

const ARROW_BATCH_LOOKUP_INDEX_MAGIC: &[u8; 8] = b"S2ABI002";
const ARROW_BATCH_LOOKUP_INDEX_HEADER_LEN: usize = 40;
const ARROW_BATCH_LOOKUP_INDEX_RECORD_LEN: usize = 16;
const ARROW_BATCH_LOOKUP_INDEX_SOURCE_HASH_DOMAIN: &[u8] =
    b"s2and-arrow-batch-lookup-index-source\0";

fn source_file_fingerprint(path: &str, source_size: u64) -> PyResult<u64> {
    let mut file = File::open(path).map_err(|err| {
        io_error_to_py(
            "failed to open Arrow IPC file for fingerprinting",
            path,
            err,
        )
    })?;
    let mut digest = fnv64(ARROW_BATCH_LOOKUP_INDEX_SOURCE_HASH_DOMAIN);
    digest = fnv64_update(digest, &source_size.to_le_bytes());
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read_len = file.read(&mut buffer).map_err(|err| {
            io_error_to_py("failed to read Arrow IPC file fingerprint bytes", path, err)
        })?;
        if read_len == 0 {
            break;
        }
        digest = fnv64_update(digest, &buffer[..read_len]);
    }
    Ok(digest)
}

pub(crate) struct ArrowBatchLookupIndex {
    bytes: Mmap,
    record_count: usize,
    max_batch_index: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceValidationMode {
    StrictFingerprint,
    RequestTimeSourceSize,
}

impl ArrowBatchLookupIndex {
    #[allow(dead_code)]
    fn open(path: &str, source_arrow_path: &str, key_column: &str) -> PyResult<Self> {
        Self::open_with_source_validation(
            path,
            source_arrow_path,
            key_column,
            SourceValidationMode::StrictFingerprint,
        )
    }

    pub(crate) fn open_for_request(
        path: &str,
        source_arrow_path: &str,
        key_column: &str,
    ) -> PyResult<Self> {
        Self::open_with_source_validation(
            path,
            source_arrow_path,
            key_column,
            SourceValidationMode::RequestTimeSourceSize,
        )
    }

    fn open_with_source_validation(
        path: &str,
        source_arrow_path: &str,
        key_column: &str,
        source_validation: SourceValidationMode,
    ) -> PyResult<Self> {
        let file = File::open(path)
            .map_err(|err| io_error_to_py("failed to open Arrow batch lookup index", path, err))?;
        let index_len = file
            .metadata()
            .map_err(|err| io_error_to_py("failed to stat Arrow batch lookup index", path, err))?
            .len();
        if index_len < ARROW_BATCH_LOOKUP_INDEX_HEADER_LEN as u64 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{path}' is shorter than its header"
            )));
        }
        // The planner's artifact bundle is immutable for the lifetime of its
        // reusable handles, so retaining a read-only mapping is safe.
        let bytes = unsafe { Mmap::map(&file) }.map_err(|err| {
            io_error_to_py("failed to memory-map Arrow batch lookup index", path, err)
        })?;
        let magic = &bytes[0..8];
        if magic != ARROW_BATCH_LOOKUP_INDEX_MAGIC {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{path}' has invalid magic bytes"
            )));
        }
        let source_metadata = fs::metadata(source_arrow_path).map_err(|err| {
            io_error_to_py(
                "failed to stat Arrow IPC file for batch lookup index validation",
                source_arrow_path,
                err,
            )
        })?;
        let source_size = source_metadata.len();
        let record_count = u64::from_le_bytes(
            bytes[8..16]
                .try_into()
                .expect("slice length is checked by fixed header length"),
        ) as usize;
        let indexed_source_size = u64::from_le_bytes(
            bytes[16..24]
                .try_into()
                .expect("indexed source-size slice has fixed length"),
        );
        let indexed_key_column_hash = u64::from_le_bytes(
            bytes[24..32]
                .try_into()
                .expect("indexed key-column hash slice has fixed length"),
        );
        let expected_key_column_hash = fnv64(key_column.as_bytes());
        if indexed_key_column_hash != expected_key_column_hash {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{path}' was built for a different key column: \
                 indexed hash={indexed_key_column_hash} expected hash={expected_key_column_hash} \
                 key_column='{key_column}'"
            )));
        }
        let indexed_source_fingerprint = u64::from_le_bytes(
            bytes[32..40]
                .try_into()
                .expect("indexed source fingerprint slice has fixed length"),
        );
        if indexed_source_size != source_size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{path}' is stale for '{source_arrow_path}': \
                 indexed size={indexed_source_size} current size={source_size}"
            )));
        }
        if source_validation == SourceValidationMode::StrictFingerprint {
            let source_fingerprint = source_file_fingerprint(source_arrow_path, source_size)?;
            if indexed_source_fingerprint != source_fingerprint {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Arrow batch lookup index '{path}' is stale for '{source_arrow_path}': \
                     indexed size/fingerprint=({indexed_source_size}, {indexed_source_fingerprint}) \
                     current size/fingerprint=({source_size}, {source_fingerprint})"
                )));
            }
        }
        let expected_len = ARROW_BATCH_LOOKUP_INDEX_HEADER_LEN
            .checked_add(
                record_count
                    .checked_mul(ARROW_BATCH_LOOKUP_INDEX_RECORD_LEN)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyOverflowError::new_err(
                            "Arrow batch lookup index record count overflows usize",
                        )
                    })?,
            )
            .ok_or_else(|| {
                pyo3::exceptions::PyOverflowError::new_err(
                    "Arrow batch lookup index length overflows usize",
                )
            })?;
        if bytes.len() != expected_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{path}' length {} does not match expected length {expected_len} \
                 (record_count={record_count}, header_len={ARROW_BATCH_LOOKUP_INDEX_HEADER_LEN}, \
                 record_len={ARROW_BATCH_LOOKUP_INDEX_RECORD_LEN})",
                bytes.len()
            )));
        }
        let mut index = Self {
            bytes,
            record_count,
            max_batch_index: None,
        };
        if source_validation == SourceValidationMode::StrictFingerprint {
            let mut previous_hash = None;
            for record_index in 0..record_count {
                let record_hash = index.record_hash(record_index);
                if let Some(previous_hash) = previous_hash {
                    if record_hash < previous_hash {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Arrow batch lookup index '{path}' record hashes are not monotonic at \
                             records {} and {record_index}: {previous_hash} > {record_hash}",
                            record_index - 1
                        )));
                    }
                }
                previous_hash = Some(record_hash);
                let batch_index = index.record_batch_index(record_index);
                index.max_batch_index = Some(
                    index
                        .max_batch_index
                        .map_or(batch_index, |current| current.max(batch_index)),
                );
            }
        }
        Ok(index)
    }

    fn record_offset(&self, index: usize) -> usize {
        ARROW_BATCH_LOOKUP_INDEX_HEADER_LEN + index * ARROW_BATCH_LOOKUP_INDEX_RECORD_LEN
    }

    fn record_hash(&self, index: usize) -> u64 {
        let offset = self.record_offset(index);
        u64::from_le_bytes(
            self.bytes[offset..offset + 8]
                .try_into()
                .expect("record hash slice has fixed length"),
        )
    }

    fn record_batch_index(&self, index: usize) -> u32 {
        let offset = self.record_offset(index) + 8;
        u32::from_le_bytes(
            self.bytes[offset..offset + 4]
                .try_into()
                .expect("record batch-index slice has fixed length"),
        )
    }

    fn lower_bound(&self, hash: u64) -> usize {
        let mut lo = 0usize;
        let mut hi = self.record_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.record_hash(mid) < hash {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    // COLLISION ASSUMPTION: records match on the 64-bit FNV hash only; the raw
    // key is not stored, so two distinct keys that hash equal both return their
    // batches. This is safe ONLY because every consumer re-filters rows by exact
    // id (`keep_ids.contains(...)`) after reading the batches — the result is a
    // superset of the batches actually needed, never a wrong final row. Side
    // effect: `rows_scanned` telemetry counts the extra batches' rows. Any new
    // consumer that treats this as an exact key->batch mapping (no downstream
    // exact-id re-filter) is wrong and must add key verification material to the
    // index format (with a format-version bump) first.
    fn batch_indices_for_keys(&self, keys: &HashSet<String>) -> HashSet<usize> {
        let mut out = HashSet::new();
        for key in keys {
            let hash = fnv64(key.as_bytes());
            let mut index = self.lower_bound(hash);
            while index < self.record_count && self.record_hash(index) == hash {
                out.insert(self.record_batch_index(index) as usize);
                index += 1;
            }
        }
        out
    }

    fn validate_batch_indices(
        &self,
        index_path: &str,
        source_arrow_path: &str,
        source_batch_count: usize,
    ) -> PyResult<()> {
        if let Some(batch_index) = self.max_batch_index {
            if batch_index as usize >= source_batch_count {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Arrow batch lookup index '{index_path}' references record batch {batch_index}, \
                     but Arrow IPC file '{source_arrow_path}' contains {source_batch_count} record batches"
                )));
            }
        }
        Ok(())
    }
}

pub(crate) fn validate_arrow_batch_lookup_index(
    path: &str,
    source_arrow_path: &str,
    key_column: &str,
) -> PyResult<()> {
    let index = ArrowBatchLookupIndex::open(path, source_arrow_path, key_column)?;
    let file = File::open(source_arrow_path)
        .map_err(|err| io_error_to_py("failed to open Arrow IPC file", source_arrow_path, err))?;
    let reader = ArrowFileReader::try_new(file, None).map_err(|err| {
        arrow_error_to_py(
            "failed to read Arrow IPC schema from",
            source_arrow_path,
            err,
        )
    })?;
    index.validate_batch_indices(path, source_arrow_path, reader.num_batches())
}

#[derive(Clone, Copy, Default)]
pub(crate) struct IndexedArrowReadStats {
    pub(crate) batches_read: usize,
    pub(crate) rows_scanned: usize,
}

pub(crate) fn read_indexed_arrow_batches(
    path: &str,
    index_path: &str,
    key_column: &str,
    keep_ids: &HashSet<String>,
) -> PyResult<(Vec<RecordBatch>, IndexedArrowReadStats)> {
    if keep_ids.is_empty() {
        return Ok((Vec::new(), IndexedArrowReadStats::default()));
    }
    let index = ArrowBatchLookupIndex::open_for_request(index_path, path, key_column)?;
    read_indexed_arrow_batches_from_index(path, index_path, &index, keep_ids)
}

pub(crate) fn read_indexed_arrow_batches_from_index(
    path: &str,
    index_path: &str,
    index: &ArrowBatchLookupIndex,
    keep_ids: &HashSet<String>,
) -> PyResult<(Vec<RecordBatch>, IndexedArrowReadStats)> {
    if keep_ids.is_empty() {
        return Ok((Vec::new(), IndexedArrowReadStats::default()));
    }
    let mut batch_indices: Vec<usize> =
        index.batch_indices_for_keys(keep_ids).into_iter().collect();
    batch_indices.sort_unstable();
    let file = File::open(path)
        .map_err(|err| io_error_to_py("failed to open Arrow IPC file", path, err))?;
    let mut reader = ArrowFileReader::try_new(file, None)
        .map_err(|err| arrow_error_to_py("failed to read Arrow IPC schema from", path, err))?;
    index.validate_batch_indices(index_path, path, reader.num_batches())?;
    let mut batches = Vec::with_capacity(batch_indices.len());
    let mut rows_scanned = 0usize;
    for batch_index in batch_indices {
        reader.set_index(batch_index).map_err(|err| {
            arrow_error_to_py("failed to seek Arrow IPC record batch in", path, err)
        })?;
        let batch = reader
            .next()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Arrow IPC file '{path}' is missing indexed record batch {batch_index}"
                ))
            })?
            .map_err(|err| {
                arrow_error_to_py("failed to read Arrow IPC record batch from", path, err)
            })?;
        rows_scanned += batch.num_rows();
        batches.push(batch);
    }
    let batches_read = batches.len();
    Ok((
        batches,
        IndexedArrowReadStats {
            batches_read,
            rows_scanned,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyAnyMethods, PyStringMethods};
    use pyo3::{PyErr, Python};

    fn prepare_python_for_test() {
        #[cfg(windows)]
        if let Some(python_home) = option_env!("S2AND_RUST_PYTHONHOME") {
            std::env::set_var("PYTHONHOME", python_home);
        }
        pyo3::prepare_freethreaded_python();
    }

    fn py_err_message(err: PyErr) -> String {
        prepare_python_for_test();
        Python::with_gil(|py| {
            err.value(py)
                .str()
                .expect("PyErr value should stringify")
                .to_str()
                .expect("test error messages are ASCII")
                .to_string()
        })
    }

    #[test]
    fn arrow_batch_lookup_index_rejects_same_size_middle_rewrite() {
        let temp_root = std::env::temp_dir().join(format!(
            "s2and_arrow_index_digest_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after Unix epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_root).expect("create temp test dir");
        let source_path = temp_root.join("source.arrow");
        let index_path = temp_root.join("source.index.bin");
        let mut source_bytes = vec![b'a'; 200_000];
        source_bytes[100_000..100_016].copy_from_slice(b"middle-key-00000");
        fs::write(&source_path, &source_bytes).expect("write source bytes");
        let source_path_str = source_path
            .to_str()
            .expect("temp path should be valid unicode")
            .to_string();
        let source_fingerprint =
            source_file_fingerprint(&source_path_str, source_bytes.len() as u64)
                .expect("hash source");
        fs::write(
            &index_path,
            ARROW_BATCH_LOOKUP_INDEX_MAGIC
                .iter()
                .copied()
                .chain(0_u64.to_le_bytes())
                .chain((source_bytes.len() as u64).to_le_bytes())
                .chain(fnv64(b"signature_id").to_le_bytes())
                .chain(source_fingerprint.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("write index bytes");
        source_bytes[100_000..100_016].copy_from_slice(b"middle-key-99999");
        fs::write(&source_path, &source_bytes).expect("rewrite source bytes");

        let index_path_str = index_path
            .to_str()
            .expect("temp path should be valid unicode");
        let error =
            match ArrowBatchLookupIndex::open(index_path_str, &source_path_str, "signature_id") {
                Ok(_) => panic!("same-size middle rewrite must stale the index"),
                Err(err) => err,
            };
        assert!(py_err_message(error).contains("is stale"));
        ArrowBatchLookupIndex::open_for_request(index_path_str, &source_path_str, "signature_id")
            .expect("request-time source-size validation should avoid full-file fingerprinting");

        fs::remove_dir_all(&temp_root).ok();
    }

    #[test]
    fn arrow_batch_lookup_index_rejects_decreasing_record_hashes() {
        let temp_root = std::env::temp_dir().join(format!(
            "s2and_arrow_index_order_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after Unix epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_root).expect("create temp test dir");
        let source_path = temp_root.join("source.arrow");
        let index_path = temp_root.join("source.index.bin");
        let source_bytes = b"test Arrow source";
        fs::write(&source_path, source_bytes).expect("write source bytes");
        let source_path_str = source_path
            .to_str()
            .expect("temp path should be valid unicode")
            .to_string();
        let source_fingerprint =
            source_file_fingerprint(&source_path_str, source_bytes.len() as u64)
                .expect("hash source");
        fs::write(
            &index_path,
            ARROW_BATCH_LOOKUP_INDEX_MAGIC
                .iter()
                .copied()
                .chain(2_u64.to_le_bytes())
                .chain((source_bytes.len() as u64).to_le_bytes())
                .chain(fnv64(b"signature_id").to_le_bytes())
                .chain(source_fingerprint.to_le_bytes())
                .chain(20_u64.to_le_bytes())
                .chain(0_u32.to_le_bytes())
                .chain(0_u32.to_le_bytes())
                .chain(10_u64.to_le_bytes())
                .chain(1_u32.to_le_bytes())
                .chain(0_u32.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("write index bytes");

        let error = match ArrowBatchLookupIndex::open(
            index_path
                .to_str()
                .expect("temp path should be valid unicode"),
            &source_path_str,
            "signature_id",
        ) {
            Ok(_) => panic!("decreasing record hashes must be rejected"),
            Err(err) => err,
        };
        assert!(py_err_message(error).contains("record hashes are not monotonic"));
        let request_index = ArrowBatchLookupIndex::open_for_request(
            index_path
                .to_str()
                .expect("temp path should be valid unicode"),
            &source_path_str,
            "signature_id",
        )
        .expect("request-time opens rely on validation at immutable publication");
        assert!(request_index.max_batch_index.is_none());

        fs::remove_dir_all(&temp_root).ok();
    }
}
