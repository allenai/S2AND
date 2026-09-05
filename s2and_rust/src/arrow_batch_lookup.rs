use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::record_batch::RecordBatch;
use memmap2::Mmap;
use pyo3::PyResult;
use std::collections::HashSet;
use std::io::Read;

use crate::arrow_dataset::{RetainedFile, RetainedFileReader};
use crate::raw_arrow::arrow_io::{arrow_error_to_py, io_error_to_py};
use crate::{fnv64, fnv64_update};

pub(crate) const MAGIC: &[u8; 8] = b"S2ABI002";
const HEADER_LEN: usize = 40;
const RECORD_LEN: usize = 16;
pub(crate) const SOURCE_HASH_DOMAIN: &[u8] = b"s2and-arrow-batch-lookup-index-source\0";

fn source_fingerprint(
    mut reader: RetainedFileReader,
    label: &str,
    source_size: u64,
) -> PyResult<u64> {
    let mut digest = fnv64(SOURCE_HASH_DOMAIN);
    digest = fnv64_update(digest, &source_size.to_le_bytes());
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read_len = reader.read(&mut buffer).map_err(|err| {
            io_error_to_py(
                "failed to read Arrow IPC file fingerprint bytes",
                label,
                err,
            )
        })?;
        if read_len == 0 {
            return Ok(digest);
        }
        digest = fnv64_update(digest, &buffer[..read_len]);
    }
}

pub(crate) struct ArrowBatchLookupIndex {
    bytes: Mmap,
    record_count: usize,
    max_batch_index: Option<u32>,
    label: String,
}

impl ArrowBatchLookupIndex {
    pub(crate) fn open_retained(
        index_file: &RetainedFile,
        source: &RetainedFile,
        key_column: &str,
        source_batch_count: usize,
    ) -> PyResult<Self> {
        let index_len = index_file
            .file()
            .metadata()
            .map_err(|err| {
                io_error_to_py(
                    "failed to stat Arrow batch lookup index",
                    index_file.label(),
                    err,
                )
            })?
            .len();
        if index_len < HEADER_LEN as u64 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{}' is shorter than its header",
                index_file.label()
            )));
        }
        // The retained OS handle pins this exact file even if its path is
        // atomically replaced after the dataset opens.
        let bytes = unsafe { Mmap::map(index_file.file()) }.map_err(|err| {
            io_error_to_py(
                "failed to memory-map Arrow batch lookup index",
                index_file.label(),
                err,
            )
        })?;
        if &bytes[0..8] != MAGIC {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{}' has invalid magic bytes",
                index_file.label()
            )));
        }

        let source_size = source
            .file()
            .metadata()
            .map_err(|err| {
                io_error_to_py(
                    "failed to stat retained Arrow IPC file",
                    source.label(),
                    err,
                )
            })?
            .len();
        let record_count: usize =
            u64::from_le_bytes(bytes[8..16].try_into().expect("fixed header"))
                .try_into()
                .map_err(|_| {
                    pyo3::exceptions::PyOverflowError::new_err(
                        "Arrow batch lookup index record count overflows usize",
                    )
                })?;
        let indexed_source_size =
            u64::from_le_bytes(bytes[16..24].try_into().expect("fixed header"));
        let indexed_key_column_hash =
            u64::from_le_bytes(bytes[24..32].try_into().expect("fixed header"));
        let indexed_source_fingerprint =
            u64::from_le_bytes(bytes[32..40].try_into().expect("fixed header"));
        let expected_key_column_hash = fnv64(key_column.as_bytes());
        if indexed_key_column_hash != expected_key_column_hash {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{}' was built for a different key column: \
                 indexed hash={indexed_key_column_hash} expected hash={expected_key_column_hash} \
                 key_column='{key_column}'",
                index_file.label()
            )));
        }
        if indexed_source_size != source_size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{}' is stale for '{}': \
                 indexed size={indexed_source_size} current size={source_size}",
                index_file.label(),
                source.label()
            )));
        }
        let actual_fingerprint = source_fingerprint(source.reader(), source.label(), source_size)?;
        if indexed_source_fingerprint != actual_fingerprint {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{}' is stale for '{}': \
                 indexed size/fingerprint=({indexed_source_size}, {indexed_source_fingerprint}) \
                 current size/fingerprint=({source_size}, {actual_fingerprint})",
                index_file.label(),
                source.label()
            )));
        }
        let expected_len = HEADER_LEN
            .checked_add(record_count.checked_mul(RECORD_LEN).ok_or_else(|| {
                pyo3::exceptions::PyOverflowError::new_err(
                    "Arrow batch lookup index record count overflows usize",
                )
            })?)
            .ok_or_else(|| {
                pyo3::exceptions::PyOverflowError::new_err(
                    "Arrow batch lookup index length overflows usize",
                )
            })?;
        if bytes.len() != expected_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{}' length {} does not match expected length {expected_len} \
                 (record_count={record_count}, header_len={HEADER_LEN}, record_len={RECORD_LEN})",
                index_file.label(),
                bytes.len()
            )));
        }

        let mut index = Self {
            bytes,
            record_count,
            max_batch_index: None,
            label: index_file.label().to_string(),
        };
        let mut previous_hash = None;
        for record_index in 0..record_count {
            let record_hash = index.record_hash(record_index);
            if previous_hash.is_some_and(|previous| record_hash < previous) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Arrow batch lookup index '{}' record hashes are not monotonic at \
                     records {} and {record_index}",
                    index_file.label(),
                    record_index - 1
                )));
            }
            previous_hash = Some(record_hash);
            let batch_index = index.record_batch_index(record_index);
            index.max_batch_index = Some(
                index
                    .max_batch_index
                    .map_or(batch_index, |current| current.max(batch_index)),
            );
        }
        index.validate_batch_indices(source.label(), source_batch_count)?;
        Ok(index)
    }

    fn record_offset(&self, index: usize) -> usize {
        HEADER_LEN + index * RECORD_LEN
    }

    fn record_hash(&self, index: usize) -> u64 {
        let offset = self.record_offset(index);
        u64::from_le_bytes(
            self.bytes[offset..offset + 8]
                .try_into()
                .expect("fixed record"),
        )
    }

    fn record_batch_index(&self, index: usize) -> u32 {
        let offset = self.record_offset(index) + 8;
        u32::from_le_bytes(
            self.bytes[offset..offset + 4]
                .try_into()
                .expect("fixed record"),
        )
    }

    fn lower_bound(&self, hash: u64) -> usize {
        let mut low = 0usize;
        let mut high = self.record_count;
        while low < high {
            let middle = low + (high - low) / 2;
            if self.record_hash(middle) < hash {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        low
    }

    /// Return a superset of needed batches. Consumers must still exact-filter
    /// ids because the compact index stores only 64-bit hashes.
    pub(crate) fn sorted_batch_indices(&self, keys: &HashSet<String>) -> Vec<usize> {
        let mut out = HashSet::new();
        for key in keys {
            let hash = fnv64(key.as_bytes());
            let mut index = self.lower_bound(hash);
            while index < self.record_count && self.record_hash(index) == hash {
                out.insert(self.record_batch_index(index) as usize);
                index += 1;
            }
        }
        let mut out = out.into_iter().collect::<Vec<_>>();
        out.sort_unstable();
        out
    }

    fn validate_batch_indices(
        &self,
        source_label: &str,
        source_batch_count: usize,
    ) -> PyResult<()> {
        if let Some(batch_index) = self.max_batch_index {
            if batch_index as usize >= source_batch_count {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Arrow batch lookup index '{}' references record batch {batch_index}, \
                     but Arrow IPC file '{source_label}' contains {source_batch_count} record batches",
                    self.label
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn read_retained_batches(
        &self,
        source: &RetainedFile,
        keep_ids: &HashSet<String>,
    ) -> PyResult<(Vec<RecordBatch>, IndexedArrowReadStats)> {
        if keep_ids.is_empty() {
            return Ok((Vec::new(), IndexedArrowReadStats::default()));
        }
        let batch_indices = self.sorted_batch_indices(keep_ids);
        let mut reader = ArrowFileReader::try_new(source.reader(), None).map_err(|err| {
            arrow_error_to_py("failed to read Arrow IPC schema from", source.label(), err)
        })?;
        let mut batches = Vec::with_capacity(batch_indices.len());
        let mut rows_scanned = 0usize;
        for batch_index in batch_indices {
            reader.set_index(batch_index).map_err(|err| {
                arrow_error_to_py(
                    "failed to seek Arrow IPC record batch in",
                    source.label(),
                    err,
                )
            })?;
            let batch = reader
                .next()
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Arrow IPC file '{}' is missing indexed record batch {batch_index}",
                        source.label()
                    ))
                })?
                .map_err(|err| {
                    arrow_error_to_py(
                        "failed to read Arrow IPC record batch from",
                        source.label(),
                        err,
                    )
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
}

#[derive(Clone, Copy, Default)]
pub(crate) struct IndexedArrowReadStats {
    pub(crate) batches_read: usize,
    pub(crate) rows_scanned: usize,
}
