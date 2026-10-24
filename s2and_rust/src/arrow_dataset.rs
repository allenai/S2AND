use crate::{HashMap, HashSet};
use arrow::ipc::reader::FileReader as ArrowFileReader;
use pyo3::prelude::*;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::Arc;

use crate::arrow_batch_lookup::{
    decode_record_batches, ArrowBatchLookupIndex, IndexedArrowReadStats,
};
use crate::name_counts::{NameCountsIndex, RawNameCountMaps};
use crate::raw_arrow::arrow_io::{arrow_error_to_py, io_error_to_py};
use crate::raw_arrow::readers::{
    read_raw_arrow_paper_authors_from_batches, read_raw_arrow_papers_from_batches,
    read_raw_arrow_signatures_from_batches, read_raw_arrow_specter_from_batches, RawArrowPaper,
    RawArrowSignature,
};
use crate::rayon_pool::install_with_optional_rayon_pool;

const REQUIRED_TABLES: [(&str, &str); 3] = [
    ("signatures", "signature_id"),
    ("papers", "paper_id"),
    ("paper_authors", "paper_id"),
];

#[cfg(unix)]
fn duplicate_os_handle(raw: isize, label: &str) -> PyResult<File> {
    use std::os::fd::{BorrowedFd, FromRawFd, IntoRawFd};

    let raw_fd = i32::try_from(raw).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "invalid Unix file descriptor for '{label}': {raw}"
        ))
    })?;
    // The Python owner keeps the borrowed descriptor alive for this call. The
    // duplicate is immediately converted into a File owned by this dataset.
    let borrowed = unsafe { BorrowedFd::borrow_raw(raw_fd) };
    let owned = borrowed.try_clone_to_owned().map_err(|err| {
        io_error_to_py("failed to duplicate retained Arrow file handle", label, err)
    })?;
    Ok(unsafe { File::from_raw_fd(owned.into_raw_fd()) })
}

#[cfg(windows)]
fn duplicate_os_handle(raw: isize, label: &str) -> PyResult<File> {
    use std::ffi::c_void;
    use std::os::windows::io::FromRawHandle;

    #[link(name = "kernel32")]
    extern "system" {
        fn ReOpenFile(
            original_file: *mut c_void,
            desired_access: u32,
            share_mode: u32,
            flags_and_attributes: u32,
        ) -> *mut c_void;
    }

    let raw_handle = raw as *mut c_void;
    if raw_handle.is_null() || raw == -1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid Windows file handle for '{label}': {raw}"
        )));
    }
    // DuplicateHandle shares the Windows file position with Python's buffered
    // lease streams, and FileExt::seek_read changes that shared position.
    // ReOpenFile retains the same underlying file without resolving its path
    // and gives native readers an independent position. Keep delete sharing so an
    // atomic path replacement can coexist with this retained generation.
    // Python keeps the borrowed OS HANDLE alive until this call returns.
    let reopened = unsafe {
        ReOpenFile(
            raw_handle,
            0x8000_0000,                             // GENERIC_READ
            0x0000_0001 | 0x0000_0002 | 0x0000_0004, // SHARE_READ | WRITE | DELETE
            0,
        )
    };
    if reopened as isize == -1 {
        return Err(io_error_to_py(
            "failed to reopen retained Arrow file handle",
            label,
            io::Error::last_os_error(),
        ));
    }
    // ReOpenFile returned a new owned handle; File closes it on drop.
    Ok(unsafe { File::from_raw_handle(reopened) })
}

#[derive(Clone)]
pub(crate) struct RetainedFile {
    file: Arc<File>,
    label: Arc<str>,
}

impl RetainedFile {
    pub(crate) fn from_raw_handle(raw: isize, label: String) -> PyResult<Self> {
        Ok(Self {
            file: Arc::new(duplicate_os_handle(raw, &label)?),
            label: Arc::from(label),
        })
    }

    pub(crate) fn file(&self) -> &File {
        self.file.as_ref()
    }

    pub(crate) fn label(&self) -> &str {
        &self.label
    }

    pub(crate) fn reader(&self) -> RetainedFileReader {
        RetainedFileReader {
            file: Arc::clone(&self.file),
            position: 0,
        }
    }
}

/// A local cursor over one retained file.
///
/// Positional reads keep native readers independent. On Windows the retained
/// file is reopened by handle to isolate its position from Python's buffered
/// readers as well. Both platforms retain exactly the validated open file,
/// even if its original path is later replaced.
pub(crate) struct RetainedFileReader {
    file: Arc<File>,
    position: u64,
}

#[cfg(unix)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buffer, offset)
}

#[cfg(windows)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buffer, offset)
}

impl Read for RetainedFileReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read_len = read_at(self.file.as_ref(), buffer, self.position)?;
        self.position = self
            .position
            .checked_add(read_len as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "file position overflow"))?;
        Ok(read_len)
    }
}

impl Seek for RetainedFileReader {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        let next = match position {
            SeekFrom::Start(offset) => i128::from(offset),
            SeekFrom::Current(offset) => i128::from(self.position) + i128::from(offset),
            SeekFrom::End(offset) => i128::from(self.file.metadata()?.len()) + i128::from(offset),
        };
        if !(0..=i128::from(u64::MAX)).contains(&next) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid retained file seek",
            ));
        }
        self.position = next as u64;
        Ok(self.position)
    }
}

#[derive(Clone)]
pub(crate) struct RetainedArrowTable {
    source: RetainedFile,
    index: Arc<ArrowBatchLookupIndex>,
}

impl RetainedArrowTable {
    fn open(source: RetainedFile, index_file: RetainedFile, key_column: &str) -> PyResult<Self> {
        let reader = ArrowFileReader::try_new(source.reader(), None).map_err(|err| {
            arrow_error_to_py("failed to read Arrow IPC schema from", source.label(), err)
        })?;
        let index = ArrowBatchLookupIndex::open_retained(
            &index_file,
            &source,
            key_column,
            reader.num_batches(),
        )?;
        Ok(Self {
            source,
            index: Arc::new(index),
        })
    }

    pub(crate) fn label(&self) -> &str {
        self.source.label()
    }

    pub(crate) fn read_batches(
        &self,
        keep_ids: Option<&HashSet<String>>,
    ) -> PyResult<(Vec<arrow::record_batch::RecordBatch>, IndexedArrowReadStats)> {
        match keep_ids {
            Some(ids) => self.index.read_retained_batches(&self.source, ids),
            None => {
                let reader =
                    ArrowFileReader::try_new(self.source.reader(), None).map_err(|err| {
                        arrow_error_to_py(
                            "failed to read Arrow IPC schema from",
                            self.source.label(),
                            err,
                        )
                    })?;
                let all_batches: Vec<usize> = (0..reader.num_batches()).collect();
                let batches = decode_record_batches(&self.source, &all_batches)?;
                let batches_read = batches.len();
                let rows_scanned = batches.iter().map(|batch| batch.num_rows()).sum();
                Ok((
                    batches,
                    IndexedArrowReadStats {
                        batches_read,
                        rows_scanned,
                    },
                ))
            }
        }
    }

    pub(crate) fn batch_indices(&self, values: &HashSet<String>) -> Vec<usize> {
        self.index.sorted_batch_indices(values)
    }
}

#[derive(Clone)]
pub(crate) struct ArrowDatasetResources {
    signatures: RetainedArrowTable,
    papers: RetainedArrowTable,
    paper_authors: RetainedArrowTable,
    specter: Option<RetainedArrowTable>,
    name_counts: RawNameCountMaps,
}

impl ArrowDatasetResources {
    fn table(&self, key: &str) -> PyResult<&RetainedArrowTable> {
        match key {
            "signatures" => Ok(&self.signatures),
            "papers" => Ok(&self.papers),
            "paper_authors" => Ok(&self.paper_authors),
            "specter" => self.specter.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(
                    "Arrow dataset does not contain optional table 'specter'",
                )
            }),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "unknown Arrow dataset table: {key}"
            ))),
        }
    }

    pub(crate) fn name_counts(&self) -> RawNameCountMaps {
        self.name_counts.clone()
    }

    pub(crate) fn has_specter(&self) -> bool {
        self.specter.is_some()
    }

    // Reads decode and convert batches in parallel on the `num_threads` pool,
    // or on the current pool when None. Callers holding the GIL release it first.

    pub(crate) fn read_signatures(
        &self,
        keep_ids: Option<&HashSet<String>>,
        num_threads: Option<usize>,
    ) -> PyResult<(HashMap<String, RawArrowSignature>, IndexedArrowReadStats)> {
        let table = &self.signatures;
        install_with_optional_rayon_pool(num_threads, || {
            let (batches, stats) = table.read_batches(keep_ids)?;
            Ok((
                read_raw_arrow_signatures_from_batches(table.label(), batches, keep_ids)?,
                stats,
            ))
        })
    }

    /// The raw `author_orcid` of each requested signature, in Arrow row order:
    /// None for a null, the string otherwise, as pyarrow reads the column.
    /// Returns None when the table has no `author_orcid` column.
    ///
    /// Incremental requests load the ORCIDs of every seed through pyarrow,
    /// which took 2 s of Python on a half-million seeds. Rows outside
    /// `keep_ids` are skipped; a requested id on two rows is an error.
    pub(crate) fn read_signature_orcids(
        &self,
        keep_ids: &HashSet<String>,
        num_threads: Option<usize>,
    ) -> PyResult<Option<Vec<(String, Option<String>)>>> {
        use crate::raw_arrow::arrow_io::{
            arrow_column_index, arrow_optional_column_index, ArrowStringColumn,
        };
        use rayon::prelude::*;

        let table = &self.signatures;
        // Batch decoding runs on the current pool, so the read stays inside it.
        let per_batch: Option<Vec<Vec<(String, Option<String>)>>> =
            install_with_optional_rayon_pool(num_threads, || {
                let (batches, _stats) = table.read_batches(Some(keep_ids))?;
                let Some(first) = batches.first() else {
                    return Ok(Some(Vec::new()));
                };
                if arrow_optional_column_index(first, "author_orcid").is_none() {
                    return Ok(None);
                }
                batches
                    .par_iter()
                    .map(|batch| -> PyResult<Vec<(String, Option<String>)>> {
                        let label = table.label();
                        let ids = batch.column(arrow_column_index(batch, "signature_id", label)?);
                        let ids =
                            ArrowStringColumn::from_string_array(ids.as_ref(), "signature_id")?;
                        let orcids =
                            batch.column(arrow_column_index(batch, "author_orcid", label)?);
                        let orcids =
                            ArrowStringColumn::from_string_array(orcids.as_ref(), "author_orcid")?;
                        let mut rows = Vec::new();
                        for row in 0..batch.num_rows() {
                            let signature_id = ids.required_value(row, "signature_id")?;
                            if keep_ids.contains(signature_id) {
                                rows.push((signature_id.to_owned(), orcids.optional_owned(row)));
                            }
                        }
                        Ok(rows)
                    })
                    .collect::<PyResult<_>>()
                    .map(Some)
            })?;
        let Some(per_batch) = per_batch else {
            return Ok(None);
        };
        // A sorted copy finds a repeated id on the pool; the first repeat in
        // row order is then reported, as the row loop did.
        let repeated = install_with_optional_rayon_pool(num_threads, || {
            let mut sorted: Vec<&str> = per_batch
                .iter()
                .flatten()
                .map(|(signature_id, _)| signature_id.as_str())
                .collect();
            sorted.par_sort_unstable();
            sorted.par_windows(2).any(|pair| pair[0] == pair[1])
        });
        if repeated {
            let mut seen = HashSet::<&str>::default();
            for (signature_id, _) in per_batch.iter().flatten() {
                if !seen.insert(signature_id) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "signatures Arrow contains duplicate signature_id: {signature_id:?}"
                    )));
                }
            }
        }
        Ok(Some(per_batch.into_iter().flatten().collect()))
    }

    pub(crate) fn read_papers(
        &self,
        keep_ids: &HashSet<String>,
        num_threads: Option<usize>,
    ) -> PyResult<(HashMap<String, RawArrowPaper>, IndexedArrowReadStats)> {
        let table = &self.papers;
        install_with_optional_rayon_pool(num_threads, || {
            let (batches, stats) = table.read_batches(Some(keep_ids))?;
            Ok((
                read_raw_arrow_papers_from_batches(table.label(), batches, Some(keep_ids))?,
                stats,
            ))
        })
    }

    pub(crate) fn read_paper_authors(
        &self,
        keep_ids: &HashSet<String>,
        num_threads: Option<usize>,
    ) -> PyResult<(HashMap<String, Vec<(i64, String)>>, IndexedArrowReadStats)> {
        let table = &self.paper_authors;
        install_with_optional_rayon_pool(num_threads, || {
            let (batches, stats) = table.read_batches(Some(keep_ids))?;
            Ok((
                read_raw_arrow_paper_authors_from_batches(table.label(), batches, Some(keep_ids))?,
                stats,
            ))
        })
    }

    pub(crate) fn read_specter(
        &self,
        keep_ids: &HashSet<String>,
        num_threads: Option<usize>,
    ) -> PyResult<(
        HashMap<String, arrow::buffer::ScalarBuffer<f32>>,
        IndexedArrowReadStats,
    )> {
        let table = self.table("specter")?;
        install_with_optional_rayon_pool(num_threads, || {
            let (batches, stats) = table.read_batches(Some(keep_ids))?;
            Ok((
                read_raw_arrow_specter_from_batches(table.label(), batches, Some(keep_ids))?,
                stats,
            ))
        })
    }
}

/// Native retained resource for one already-validated immutable Arrow root.
#[pyclass(name = "_ArrowDataset", frozen)]
pub(crate) struct ArrowDataset {
    resources: ArrowDatasetResources,
}

impl ArrowDataset {
    pub(crate) fn shared(&self) -> ArrowDatasetResources {
        self.resources.clone()
    }
}

fn required_handle(
    handles: &mut HashMap<String, (isize, String)>,
    key: &str,
) -> PyResult<RetainedFile> {
    let (raw, label) = handles.remove(key).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "Arrow dataset handles are missing required key: {key}"
        ))
    })?;
    RetainedFile::from_raw_handle(raw, label)
}

fn open_table(
    handles: &mut HashMap<String, (isize, String)>,
    key: &str,
    key_column: &str,
) -> PyResult<RetainedArrowTable> {
    let source = required_handle(handles, key)?;
    let index = required_handle(handles, &format!("{key}_batch_index"))?;
    RetainedArrowTable::open(source, index, key_column)
}

#[pymethods]
impl ArrowDataset {
    #[new]
    #[pyo3(signature = (handles, name_counts_index=None))]
    fn new(
        py: Python<'_>,
        mut handles: HashMap<String, (isize, String)>,
        name_counts_index: Option<Py<NameCountsIndex>>,
    ) -> PyResult<Self> {
        let mut tables =
            HashMap::with_capacity_and_hasher(REQUIRED_TABLES.len(), Default::default());
        for (key, key_column) in REQUIRED_TABLES {
            tables.insert(key, open_table(&mut handles, key, key_column)?);
        }
        let specter = match (
            handles.contains_key("specter"),
            handles.contains_key("specter_batch_index"),
        ) {
            (true, true) => Some(open_table(&mut handles, "specter", "paper_id")?),
            (false, false) => None,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Arrow dataset requires specter and specter_batch_index together",
                ))
            }
        };
        if !handles.is_empty() {
            let mut unknown = handles.into_keys().collect::<Vec<_>>();
            unknown.sort_unstable();
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow dataset received unsupported handle keys: {unknown:?}"
            )));
        }
        let name_counts = match name_counts_index {
            Some(index) => {
                let index = index.borrow(py);
                RawNameCountMaps::from_shared_index(index.shared_index())
            }
            None => RawNameCountMaps::default(),
        };
        Ok(Self {
            resources: ArrowDatasetResources {
                signatures: tables.remove("signatures").expect("required table"),
                papers: tables.remove("papers").expect("required table"),
                paper_authors: tables.remove("paper_authors").expect("required table"),
                specter,
                name_counts,
            },
        })
    }

    /// Return deterministic record-batch indexes for exact-id re-filtering.
    fn batch_indices(&self, table_key: &str, values: Vec<String>) -> PyResult<Vec<usize>> {
        let values = values.into_iter().collect::<HashSet<_>>();
        Ok(self.resources.table(table_key)?.batch_indices(&values))
    }

    fn has_specter(&self) -> bool {
        self.resources.has_specter()
    }

    /// Raw `author_orcid` values of the requested signatures, or None when
    /// the table has no `author_orcid` column: the ids found, in Arrow row
    /// order, and the positions in that list and values of the non-null ORCIDs.
    ///
    /// A request asks for every seed, and few have an ORCID; one Python
    /// tuple per seed took longer than the read.
    #[pyo3(signature = (signature_ids, num_threads = None))]
    fn signature_orcids(
        &self,
        py: Python<'_>,
        signature_ids: Vec<String>,
        num_threads: Option<usize>,
    ) -> PyResult<Option<(Vec<String>, Vec<usize>, Vec<String>)>> {
        let rows = py.allow_threads(|| {
            let mut keep_ids: HashSet<String> =
                HashSet::with_capacity_and_hasher(signature_ids.len(), Default::default());
            keep_ids.extend(signature_ids);
            self.resources.read_signature_orcids(&keep_ids, num_threads)
        })?;
        Ok(rows.map(|rows| {
            let mut positions = Vec::new();
            let mut orcids = Vec::new();
            let signature_ids = rows
                .into_iter()
                .enumerate()
                .map(|(position, (signature_id, orcid))| {
                    if let Some(orcid) = orcid {
                        positions.push(position);
                        orcids.push(orcid);
                    }
                    signature_id
                })
                .collect();
            (signature_ids, positions, orcids)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, ArrayRef, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::FileWriter;
    use arrow::record_batch::RecordBatch;
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir() -> std::path::PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "s2and_retained_arrow_{}_{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).expect("create test directory");
        directory
    }

    fn write_arrow(path: &Path, value: &str) {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let columns: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![value]))];
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).expect("valid batch");
        let file = File::create(path).expect("create Arrow file");
        let mut writer = FileWriter::try_new(file, schema.as_ref()).expect("create Arrow writer");
        writer.write(&batch).expect("write Arrow batch");
        writer.finish().expect("finish Arrow file");
    }

    fn write_index(path: &Path, source: &Path, key_column: &str, value: &str) {
        let source_size = std::fs::metadata(source).expect("stat Arrow file").len();
        let bytes = crate::arrow_batch_lookup::MAGIC
            .iter()
            .copied()
            .chain(1u64.to_le_bytes())
            .chain(source_size.to_le_bytes())
            .chain(crate::fnv64(key_column.as_bytes()).to_le_bytes())
            .chain(crate::fnv64(value.as_bytes()).to_le_bytes())
            .chain(0u32.to_le_bytes())
            .chain(0u32.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(path, bytes).expect("write lookup index");
    }

    #[cfg(unix)]
    fn raw_handle(file: &File) -> isize {
        use std::os::fd::AsRawFd;
        file.as_raw_fd() as isize
    }

    #[cfg(windows)]
    fn raw_handle(file: &File) -> isize {
        use std::os::windows::io::AsRawHandle;
        file.as_raw_handle() as isize
    }

    #[test]
    fn retained_table_reads_open_file_after_path_replacement() {
        let directory = unique_temp_dir();
        let source_path = directory.join("source.arrow");
        let index_path = directory.join("source.index");
        write_arrow(&source_path, "old");
        write_index(&index_path, &source_path, "id", "old");

        let source_file = File::open(&source_path).expect("open source");
        let index_file = File::open(&index_path).expect("open index");
        let retained_source = RetainedFile::from_raw_handle(
            raw_handle(&source_file),
            source_path.display().to_string(),
        )
        .expect("retain source");
        let retained_index = RetainedFile::from_raw_handle(
            raw_handle(&index_file),
            index_path.display().to_string(),
        )
        .expect("retain index");
        let table = RetainedArrowTable::open(retained_source, retained_index, "id")
            .expect("open retained table");
        drop(source_file);
        drop(index_file);

        let replacement = directory.join("replacement.arrow");
        write_arrow(&replacement, "new");
        #[cfg(unix)]
        std::fs::rename(&replacement, &source_path).expect("atomically replace source path");
        #[cfg(windows)]
        {
            std::fs::remove_file(&source_path).expect("unlink old source path");
            std::fs::rename(&replacement, &source_path).expect("replace source path");
        }

        let (batches, stats) = table.read_batches(None).expect("read retained source");
        let values = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string column");
        assert_eq!(values.value(0), "old");
        assert_eq!(stats.batches_read, 1);
        assert_eq!(
            table.batch_indices(&HashSet::from_iter(["old".to_string()])),
            [0]
        );

        drop(table);
        std::fs::remove_dir_all(directory).expect("remove test directory");
    }
}
