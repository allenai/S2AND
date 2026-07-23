use pyo3::prelude::*;
use pyo3::types::PyModule;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

const FNV64_OFFSET: u64 = 14695981039346656037;
const FNV64_PRIME: u64 = 1099511628211;
const ARROW_SOURCE_HASH_DOMAIN: &[u8] = b"s2and-arrow-batch-lookup-index-source\0";
const READ_BUFFER_BYTES: usize = 1024 * 1024;

#[inline(always)]
fn fnv64_update(mut digest: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        digest ^= byte as u64;
        digest = digest.wrapping_mul(FNV64_PRIME);
    }
    digest
}

fn source_file_digests(
    path: &Path,
    source_size: u64,
    include_sha256: bool,
) -> io::Result<(Option<String>, u64)> {
    let mut source = File::open(path)?;
    let mut fingerprint = fnv64_update(FNV64_OFFSET, ARROW_SOURCE_HASH_DOMAIN);
    fingerprint = fnv64_update(fingerprint, &source_size.to_le_bytes());
    let mut sha256 = include_sha256.then(Sha256::new);
    let mut buffer = vec![0u8; READ_BUFFER_BYTES];
    loop {
        let byte_count = source.read(&mut buffer)?;
        if byte_count == 0 {
            break;
        }
        let chunk = &buffer[..byte_count];
        fingerprint = fnv64_update(fingerprint, chunk);
        if let Some(digest) = sha256.as_mut() {
            digest.update(chunk);
        }
    }
    Ok((
        sha256.map(|digest| format!("{:x}", digest.finalize())),
        fingerprint,
    ))
}

#[pyfunction]
fn arrow_source_file_digests(
    py: Python<'_>,
    path: String,
    source_size: u64,
    include_sha256: bool,
) -> PyResult<(Option<String>, u64)> {
    let source_path = PathBuf::from(path);
    let source_label = source_path.display().to_string();
    py.allow_threads(move || source_file_digests(&source_path, source_size, include_sha256))
        .map_err(|error| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "failed to fingerprint Arrow source file {}: {}",
                source_label, error
            ))
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
    module.add_function(wrap_pyfunction!(arrow_source_file_digests, module)?)?;
    module.add_function(wrap_pyfunction!(fnv64_utf8_batch, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_fingerprint_includes_domain_size_and_content() {
        let mut observed = fnv64_update(FNV64_OFFSET, ARROW_SOURCE_HASH_DOMAIN);
        observed = fnv64_update(observed, &3_u64.to_le_bytes());
        observed = fnv64_update(observed, b"abc");

        assert_eq!(observed, 11851141429550314739);
    }

    #[test]
    fn utf8_hash_matches_fnv1a64_contract() {
        assert_eq!(fnv64_update(FNV64_OFFSET, b"abc"), 16654208175385433931);
    }
}
