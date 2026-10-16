use pyo3::prelude::*;
use pyo3::types::PyModule;
use sha2::{Digest, Sha256};
use std::io::{self, Read};

use crate::arrow_dataset::RetainedFile;

const FNV64_OFFSET: u64 = 14695981039346656037;
const FNV64_PRIME: u64 = 1099511628211;
const READ_BUFFER_BYTES: usize = 1024 * 1024;

#[inline(always)]
fn fnv64_update(mut digest: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        digest ^= byte as u64;
        digest = digest.wrapping_mul(FNV64_PRIME);
    }
    digest
}

fn reader_sha256(mut source: impl Read) -> io::Result<String> {
    let mut sha256 = Sha256::new();
    let mut buffer = vec![0u8; READ_BUFFER_BYTES];
    loop {
        let byte_count = source.read(&mut buffer)?;
        if byte_count == 0 {
            break;
        }
        sha256.update(&buffer[..byte_count]);
    }
    Ok(format!("{:x}", sha256.finalize()))
}

#[pyfunction]
fn _arrow_retained_file_sha256(py: Python<'_>, handle: isize, label: String) -> PyResult<String> {
    let source = RetainedFile::from_raw_handle(handle, label)?;
    py.allow_threads(|| reader_sha256(source.reader()))
        .map_err(|error| {
            crate::raw_arrow::arrow_io::io_error_to_py(
                "failed to hash retained Arrow file",
                source.label(),
                error,
            )
        })
}

#[pyfunction]
fn fnv64_utf8_batch(py: Python<'_>, values: Vec<String>) -> Vec<u64> {
    py.allow_threads(move || {
        values
            .iter()
            .map(|value| fnv64_update(FNV64_OFFSET, value.as_bytes()))
            .collect()
    })
}

pub(crate) fn add_to_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_arrow_retained_file_sha256, module)?)?;
    module.add_function(wrap_pyfunction!(fnv64_utf8_batch, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_hash_matches_fnv1a64_contract() {
        assert_eq!(fnv64_update(FNV64_OFFSET, b"abc"), 16654208175385433931);
    }
}
