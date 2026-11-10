//! Kernels for name-conflict repair; `s2and/name_conflict_repair.py` states the rule.
//!
//! Both kernels read a condensed block matrix front to back.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::rayon_pool::install_with_optional_rayon_pool;

/// Code of an impossible pair.
const HARD: u8 = 255;
/// A code is the distance times the scale, rounded down: [0, 1] maps to 0..=254.
const SCALE: f64 = 254.999;
const HARD_CAP: f64 = (HARD as f64 + 0.5) / SCALE;
/// Pairs per parallel task.
const QUANTIZE_CHUNK: usize = 1 << 16;

/// Write one code per distance and report whether any pair is impossible.
fn quantize(condensed: &[f64], codes: &mut [u8]) -> bool {
    codes
        .par_chunks_mut(QUANTIZE_CHUNK)
        .zip(condensed.par_chunks(QUANTIZE_CHUNK))
        .map(|(codes, distances)| {
            let mut any_hard = false;
            for (code, distance) in codes.iter_mut().zip(distances) {
                *code = (distance.min(HARD_CAP) * SCALE) as u8;
                any_hard |= *code == HARD;
            }
            any_hard
        })
        .reduce(|| false, |left, right| left | right)
}

/// A one-byte-per-pair copy of condensed distances, or None without an impossible pair.
///
/// Distances lie in [0, 1]; an impossible pair holds `hard_distance` or more.
#[pyfunction]
#[pyo3(signature = (condensed, hard_distance, num_threads = None))]
pub(crate) fn quantize_block_distances<'py>(
    py: Python<'py>,
    condensed: PyReadonlyArray1<'py, f64>,
    hard_distance: f64,
    num_threads: Option<usize>,
) -> PyResult<Option<Bound<'py, PyArray1<u8>>>> {
    if hard_distance < HARD_CAP {
        return Err(PyValueError::new_err(
            "hard_distance must exceed every model distance",
        ));
    }
    let distances = condensed.as_slice()?;
    let mut codes = vec![0u8; distances.len()];
    // Linkage has not started, so nothing writes the matrix while the pool reads it.
    let any_hard = py.allow_threads(|| {
        install_with_optional_rayon_pool(num_threads, || quantize(distances, &mut codes))
    });
    Ok(any_hard.then(|| codes.into_pyarray(py)))
}

/// Start of row `i` in a condensed matrix of `n` items: the pair (i, i + 1).
fn row_start(n: usize, i: usize) -> usize {
    n * i - i * (i + 1) / 2
}

/// Code of the pair (i, j), i != j.
fn pair_code(codes: &[u8], n: usize, i: usize, j: usize) -> u8 {
    let (low, high) = if i < j { (i, j) } else { (j, i) };
    codes[row_start(n, low) + high - low - 1]
}

/// Cluster pairs that one odd mention keeps apart.
///
/// Returns `(a, b, odd, code_sum)`: every impossible pair between clusters
/// `a` and `b` holds mention `odd` of `a`, and `code_sum` is the sum of codes
/// between `a` without `odd` and `b`.
///
/// One front-to-back pass over the matrix collects, per cluster pair, the sum
/// of codes and the count of impossible pairs, and per mention and cluster,
/// whether the mention has an impossible pair with that cluster. It runs on
/// the calling thread: the clustering step calls it while the pool scores the
/// next block, and a pool job would wait behind that scoring.
fn find_candidates(
    codes: &[u8],
    cluster: &[i64],
    cluster_count: usize,
    min_side: usize,
    max_mean_code: f64,
) -> Vec<(usize, usize, usize, u64)> {
    let n = cluster.len();
    let k = cluster_count;
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (mention, &label) in cluster.iter().enumerate() {
        if label >= 0 {
            members[label as usize].push(mention);
        }
    }
    // Unordered cluster pair (a <= b) -> position in the triangular tables.
    let pair_index = |a: usize, b: usize| {
        let (low, high) = if a < b { (a, b) } else { (b, a) };
        low * k - low * (low + 1) / 2 + high
    };
    let mut code_sums = vec![0u64; k * (k + 1) / 2];
    let mut hard_counts = vec![0u32; k * (k + 1) / 2];
    // Bit m of row c: mention m has an impossible pair with cluster c.
    let words = n.div_ceil(64);
    let mut conflicts = vec![0u64; k * words];
    let mut row_sums = vec![0u32; k];
    let mut row_hard = vec![0u32; k];
    for i in 0..n.saturating_sub(1) {
        if cluster[i] < 0 {
            continue;
        }
        let own = cluster[i] as usize;
        let start = row_start(n, i);
        let row = &codes[start..start + n - i - 1];
        let own_conflicts = &mut conflicts[own * words..(own + 1) * words];
        for ((&code, &label), j) in row.iter().zip(&cluster[i + 1..]).zip(i + 1..) {
            if label < 0 {
                continue;
            }
            row_sums[label as usize] += code as u32;
            if code == HARD {
                row_hard[label as usize] += 1;
                own_conflicts[j / 64] |= 1 << (j % 64);
            }
        }
        for other in 0..k {
            // An impossible pair adds 255, so a zero sum has none.
            if row_sums[other] == 0 {
                continue;
            }
            let index = pair_index(own, other);
            code_sums[index] += row_sums[other] as u64;
            row_sums[other] = 0;
            if row_hard[other] > 0 {
                hard_counts[index] += row_hard[other];
                row_hard[other] = 0;
                conflicts[other * words + i / 64] |= 1 << (i % 64);
            }
        }
    }

    let has_conflict = |mention: usize, other: usize| {
        (conflicts[other * words + mention / 64] >> (mention % 64)) & 1 == 1
    };
    let mut found = Vec::new();
    for first in 0..k {
        for second in first + 1..k {
            let index = pair_index(first, second);
            if hard_counts[index] == 0 {
                continue;
            }
            // The odd mention may sit on either side.
            for (a, b) in [(first, second), (second, first)] {
                if members[a].len() - 1 < min_side || members[b].len() < min_side {
                    continue;
                }
                // One mention covers at most as many impossible pairs as the other cluster has mentions.
                if hard_counts[index] as usize > members[b].len() {
                    continue;
                }
                let mut conflicting = members[a]
                    .iter()
                    .filter(|&&mention| has_conflict(mention, b));
                let (Some(&odd), None) = (conflicting.next(), conflicting.next()) else {
                    continue;
                };
                let odd_line: u64 = members[b]
                    .iter()
                    .map(|&other| pair_code(codes, n, odd, other) as u64)
                    .sum();
                let code_sum = code_sums[index] - odd_line;
                let rest = ((members[a].len() - 1) * members[b].len()) as f64;
                if (code_sum as f64) < max_mean_code * rest {
                    found.push((a, b, odd, code_sum));
                }
            }
        }
    }
    found.sort_unstable();
    found
}

/// Find cluster pairs that one odd mention keeps apart.
///
/// `cluster` holds each mention's cluster index in `0..cluster_count`, or -1
/// for a mention outside every eligible cluster. Each result is
/// `(a, b, odd, code_sum)`: mention `odd` of cluster `a` is in every impossible
/// pair between `a` and `b`, both clusters keep `min_side` mentions without
/// it, and the mean code between them without it is below `max_mean_code`.
#[pyfunction]
pub(crate) fn name_conflict_candidates<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray1<'py, u8>,
    cluster: PyReadonlyArray1<'py, i64>,
    cluster_count: usize,
    min_side: usize,
    max_mean_code: f64,
) -> PyResult<Vec<(usize, usize, usize, u64)>> {
    let codes = codes.as_slice()?;
    let cluster = cluster.as_slice()?;
    let n = cluster.len();
    if codes.len() != n * n.saturating_sub(1) / 2 {
        return Err(PyValueError::new_err(
            "codes must hold one entry per pair of mentions",
        ));
    }
    if cluster.iter().any(|&label| label >= cluster_count as i64) {
        return Err(PyValueError::new_err(
            "cluster holds an index outside cluster_count",
        ));
    }
    Ok(
        py.allow_threads(|| {
            find_candidates(codes, cluster, cluster_count, min_side, max_mean_code)
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Condensed codes for `n` mentions from a full-matrix function.
    fn condensed(n: usize, code: impl Fn(usize, usize) -> u8) -> Vec<u8> {
        let mut codes = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                codes.push(code(i, j));
            }
        }
        codes
    }

    #[test]
    fn quantize_rounds_down_and_marks_impossible_pairs() {
        let mut codes = vec![0u8; 4];
        assert!(quantize(&[0.0, 0.5, 1.0, 1e4], &mut codes));
        assert_eq!(codes, vec![0, 127, 254, 255]);
        let mut codes = vec![0u8; 2];
        assert!(!quantize(&[0.2, 1.0], &mut codes));
    }

    #[test]
    fn finds_the_odd_mention_on_either_side() {
        // Mentions 0..6 are cluster 0, 6..11 are cluster 1. Mention 5 conflicts with all of cluster 1.
        let cluster: Vec<i64> = (0..11).map(|i| if i < 6 { 0 } else { 1 }).collect();
        let codes = condensed(11, |i, j| {
            if (i < 6) == (j < 6) {
                10
            } else if i == 5 {
                HARD
            } else {
                20
            }
        });
        assert_eq!(
            find_candidates(&codes, &cluster, 2, 5, 60.0),
            vec![(0, 1, 5, 20 * 25)]
        );
        // The same block with the cluster indexes swapped.
        let swapped: Vec<i64> = cluster.iter().map(|label| 1 - label).collect();
        assert_eq!(
            find_candidates(&codes, &swapped, 2, 5, 60.0),
            vec![(1, 0, 5, 20 * 25)]
        );
        // A second conflicting mention, a higher mean, or too few mentions: no candidate.
        let two = condensed(11, |i, j| {
            if (i < 6) == (j < 6) {
                10
            } else if i == 5 || (i == 0 && j == 7) {
                HARD
            } else {
                20
            }
        });
        assert!(find_candidates(&two, &cluster, 2, 5, 60.0).is_empty());
        assert!(find_candidates(&codes, &cluster, 2, 5, 20.0).is_empty());
        assert!(find_candidates(&codes, &cluster, 2, 6, 60.0).is_empty());
    }
}
