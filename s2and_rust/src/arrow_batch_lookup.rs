use crate::HashSet;
use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::record_batch::RecordBatch;
use memmap2::Mmap;
use pyo3::PyResult;
use rayon::prelude::*;

use crate::arrow_dataset::{MappedArrowFile, RetainedFile};
use crate::fnv64;
use crate::raw_arrow::arrow_io::{arrow_error_to_py, io_error_to_py};

pub(crate) const MAGIC: &[u8; 8] = b"S2ABI003";
const HEADER_LEN: usize = 32;
const RECORD_LEN: usize = 16;

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

    /// The first record at or after `start` whose hash is not below `hash`.
    ///
    /// Every record before `start` must hash below `hash`. The search gallops
    /// forward from `start`, so a nearby target costs a few probes.
    fn lower_bound_from(&self, start: usize, hash: u64) -> usize {
        let mut low = start;
        let mut step = 1usize;
        let mut high = loop {
            let probe = low.saturating_add(step);
            if probe >= self.record_count {
                break self.record_count;
            }
            if self.record_hash(probe) >= hash {
                break probe;
            }
            low = probe + 1;
            step = step.saturating_mul(2);
        };
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
        let Some(max_batch_index) = self.max_batch_index else {
            return Vec::new();
        };
        // Sorted hashes walk the sorted records forward. A binary search per
        // key missed cache at every step: 0.6 s for a half-million paper ids.
        let mut hashes: Vec<u64> = keys.iter().map(|key| fnv64(key.as_bytes())).collect();
        hashes.sort_unstable();
        hashes.dedup();
        let mut found = vec![false; max_batch_index as usize + 1];
        let mut index = 0usize;
        for hash in hashes {
            index = self.lower_bound_from(index, hash);
            while index < self.record_count && self.record_hash(index) == hash {
                found[self.record_batch_index(index) as usize] = true;
                index += 1;
            }
        }
        found
            .iter()
            .enumerate()
            .filter_map(|(batch_index, &found)| found.then_some(batch_index))
            .collect()
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
        mapped: Option<&MappedArrowFile>,
        keep_ids: &HashSet<String>,
    ) -> PyResult<(Vec<RecordBatch>, IndexedArrowReadStats)> {
        if keep_ids.is_empty() {
            return Ok((Vec::new(), IndexedArrowReadStats::default()));
        }
        let batch_indices = self.sorted_batch_indices(keep_ids);
        let batches = decode_record_batches(source, mapped, &batch_indices)?;
        let rows_scanned = batches.iter().map(RecordBatch::num_rows).sum();
        Ok((
            batches,
            IndexedArrowReadStats {
                batches_read: batch_indices.len(),
                rows_scanned,
            },
        ))
    }
}

/// Decodes record batches `batch_indices` of `source`, in that order.
///
/// Tasks decode in parallel, from `mapped` when given and otherwise from runs
/// of positional reads. An error reports the first failing batch in order,
/// as a serial decode would.
pub(crate) fn decode_record_batches(
    source: &RetainedFile,
    mapped: Option<&MappedArrowFile>,
    batch_indices: &[usize],
) -> PyResult<Vec<RecordBatch>> {
    if let Some(mapped) = mapped {
        let decoded: Vec<PyResult<RecordBatch>> = batch_indices
            .par_iter()
            .map(|&batch_index| {
                mapped
                    .decode(batch_index)
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
                    })
            })
            .collect();
        return decoded.into_iter().collect();
    }
    let run_len = batch_indices
        .len()
        .div_ceil(rayon::current_num_threads() * 4)
        .max(1);
    let runs = batch_indices
        .par_chunks(run_len)
        .map(|run| -> PyResult<Vec<RecordBatch>> {
            let mut reader = ArrowFileReader::try_new(source.reader(), None).map_err(|err| {
                arrow_error_to_py("failed to read Arrow IPC schema from", source.label(), err)
            })?;
            run.iter()
                .map(|&batch_index| {
                    reader.set_index(batch_index).map_err(|err| {
                        arrow_error_to_py(
                            "failed to seek Arrow IPC record batch in",
                            source.label(),
                            err,
                        )
                    })?;
                    reader
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
                        })
                })
                .collect()
        })
        .collect::<Vec<_>>();
    let mut batches = Vec::with_capacity(batch_indices.len());
    for run in runs {
        batches.extend(run?);
    }
    Ok(batches)
}

#[derive(Clone, Copy, Default)]
pub(crate) struct IndexedArrowReadStats {
    pub(crate) batches_read: usize,
    pub(crate) rows_scanned: usize,
}

#[cfg(test)]
mod tests {
    use super::{ArrowBatchLookupIndex, HEADER_LEN, MAGIC, RECORD_LEN};
    use crate::{fnv64, HashSet};

    /// An index over `records` of (hash, batch), sorted by hash as written.
    fn index_over(records: &mut [(u64, u32)]) -> ArrowBatchLookupIndex {
        records.sort_unstable();
        let mut bytes = Vec::with_capacity(HEADER_LEN + records.len() * RECORD_LEN);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&[0u8; 16]);
        for &(hash, batch) in records.iter() {
            bytes.extend_from_slice(&hash.to_le_bytes());
            bytes.extend_from_slice(&batch.to_le_bytes());
            bytes.extend_from_slice(&[0u8; 4]);
        }
        let mut map = memmap2::MmapMut::map_anon(bytes.len()).expect("anonymous map");
        map.copy_from_slice(&bytes);
        ArrowBatchLookupIndex {
            bytes: map.make_read_only().expect("read-only map"),
            record_count: records.len(),
            max_batch_index: records.iter().map(|&(_, batch)| batch).max(),
            label: "<test>".to_string(),
        }
    }

    /// The batches holding each key's hash, one binary search per key.
    fn expected_batches(records: &[(u64, u32)], keys: &HashSet<String>) -> Vec<usize> {
        let mut out: Vec<usize> = keys
            .iter()
            .flat_map(|key| {
                let hash = fnv64(key.as_bytes());
                let start = records.partition_point(|&(record_hash, _)| record_hash < hash);
                records[start..]
                    .iter()
                    .take_while(move |&&(record_hash, _)| record_hash == hash)
                    .map(|&(_, batch)| batch as usize)
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    #[test]
    fn walk_matches_per_key_binary_search() {
        // Keys k0..k3999 sit in batch i % 37; k5 also has rows in batches 3
        // and 40, as a repeated key does; filler hashes pad the gaps.
        let mut records: Vec<(u64, u32)> = (0..4000u32)
            .map(|i| (fnv64(format!("k{i}").as_bytes()), i % 37))
            .collect();
        let repeated = fnv64(b"k5");
        records.extend([(repeated, 3), (repeated, 40)]);
        records.extend((0..20_000u64).map(|i| (i.wrapping_mul(0x9E37_79B9_7F4A_7C15), 41)));
        let index = index_over(&mut records);
        let key_sets: Vec<HashSet<String>> = vec![
            HashSet::default(),
            ["k5".to_string()].into_iter().collect(),
            ["absent".to_string()].into_iter().collect(),
            (0..4000).step_by(397).map(|i| format!("k{i}")).collect(),
            (0..4000).map(|i| format!("k{i}")).collect(),
            (0..6000).map(|i| format!("k{i}")).collect(),
        ];
        for keys in &key_sets {
            assert_eq!(
                index.sorted_batch_indices(keys),
                expected_batches(&records, keys),
                "{} keys",
                keys.len()
            );
        }
    }
}
