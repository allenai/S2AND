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
/// Pairs per parallel task.
const QUANTIZE_CHUNK: usize = 1 << 16;
/// The matrix holds an impossible pair.
const SAW_HARD: u8 = 1;
/// The matrix holds a value that is neither a model distance nor `hard_distance`.
const SAW_OTHER: u8 = 2;

/// Write one code per distance and report which kinds of value the matrix holds.
///
/// Only `hard_distance` itself is an impossible pair. Any other value above one, or
/// NaN, is a caller's own distance: linkage treats it as soft, and a code cannot
/// carry it, so the caller must not repair from these codes.
fn quantize(condensed: &[f64], hard_distance: f64, codes: &mut [u8]) -> u8 {
    codes
        .par_chunks_mut(QUANTIZE_CHUNK)
        .zip(condensed.par_chunks(QUANTIZE_CHUNK))
        .map(|(codes, distances)| quantize_chunk(distances, hard_distance, codes))
        .reduce(|| 0, |left, right| left | right)
}

/// `quantize` for one run of pairs, on the calling thread.
fn quantize_chunk(distances: &[f64], hard_distance: f64, codes: &mut [u8]) -> u8 {
    let mut saw = 0u8;
    for (code, &distance) in codes.iter_mut().zip(distances) {
        let hard = distance == hard_distance;
        // NaN is neither above nor at most one, and is not a model distance.
        let other = !hard && (distance > 1.0 || distance.is_nan());
        *code = if hard {
            HARD
        } else {
            (distance.min(1.0) * SCALE) as u8
        };
        saw |= ((hard as u8) * SAW_HARD) | ((other as u8) * SAW_OTHER);
    }
    saw
}

/// A one-byte-per-pair copy of condensed distances, or None when no repair can apply.
///
/// Model distances lie in [0, 1] and an impossible pair holds exactly `hard_distance`.
/// None means the matrix holds some other value, or, with `require_hard`, has no
/// impossible pair. The year-gap split reads codes of blocks without one.
#[pyfunction]
#[pyo3(signature = (condensed, hard_distance, num_threads = None, require_hard = true))]
pub(crate) fn quantize_block_distances<'py>(
    py: Python<'py>,
    condensed: PyReadonlyArray1<'py, f64>,
    hard_distance: f64,
    num_threads: Option<usize>,
    require_hard: bool,
) -> PyResult<Option<Bound<'py, PyArray1<u8>>>> {
    if hard_distance.is_nan() || hard_distance <= 1.0 {
        return Err(PyValueError::new_err(
            "hard_distance must exceed every model distance",
        ));
    }
    let distances = condensed.as_slice()?;
    let mut codes = vec![0u8; distances.len()];
    // Linkage has not started, so nothing writes the matrix while it is read.
    let saw = py.allow_threads(|| {
        if num_threads == Some(1) {
            // No pool: a pool of any size may be busy scoring the next block, and this
            // call would wait for that scoring to finish.
            quantize_chunk(distances, hard_distance, &mut codes)
        } else {
            install_with_optional_rayon_pool(num_threads, || {
                quantize(distances, hard_distance, &mut codes)
            })
        }
    });
    let usable = if require_hard {
        saw == SAW_HARD
    } else {
        saw & SAW_OTHER == 0
    };
    Ok(usable.then(|| codes.into_pyarray(py)))
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

/// A cluster pair, the odd mentions between them, and the sum of codes and the count of
/// close codes without those mentions.
type Candidate = (usize, usize, Vec<usize>, u64, u64);

/// Whether a code certainly holds a close pair; a negative `close_code` marks none.
fn is_close(code: u8, close_code: i64) -> bool {
    code != HARD && i64::from(code) <= close_code
}

/// Smallest sets of at most `max_odd` mentions that take part in every impossible pair.
///
/// `conflicting_a` and `conflicting_b` hold the mentions of each cluster that have an
/// impossible pair with the other cluster. Returns every such set of the smallest size.
fn odd_mention_sets(
    codes: &[u8],
    n: usize,
    conflicting_a: &[usize],
    conflicting_b: &[usize],
    max_odd: usize,
) -> Vec<Vec<usize>> {
    let mut sets = Vec::new();
    // The only conflicting mention of a cluster is in every impossible pair.
    if let [only] = conflicting_a {
        sets.push(vec![*only]);
    }
    if let [only] = conflicting_b {
        sets.push(vec![*only]);
    }
    if !sets.is_empty() || max_odd < 2 {
        return sets;
    }
    if conflicting_a.len() == 2 {
        sets.push(conflicting_a.to_vec());
    }
    if conflicting_b.len() == 2 {
        sets.push(conflicting_b.to_vec());
    }
    // One mention from each cluster: every other conflicting mention of the first
    // cluster must conflict with that one mention of the second cluster only.
    let partners: Vec<(usize, usize)> = conflicting_a
        .iter()
        .map(|&mention| {
            let mut count = 0;
            let mut last = usize::MAX;
            for &other in conflicting_b {
                if pair_code(codes, n, mention, other) == HARD {
                    count += 1;
                    last = other;
                }
            }
            (count, last)
        })
        .collect();
    let several = partners.iter().filter(|(count, _)| *count > 1).count();
    for (position, &mention) in conflicting_a.iter().enumerate() {
        if several > usize::from(partners[position].0 > 1) {
            continue;
        }
        let mut others = partners
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != position)
            .map(|(_, (_, partner))| *partner);
        let Some(partner) = others.next() else {
            continue;
        };
        if others.all(|other| other == partner) {
            sets.push(vec![mention, partner]);
        }
    }
    sets
}

/// Cluster pairs that at most `max_odd` odd mentions keep apart.
///
/// Returns `(a, b, odd, code_sum, close_count)` with `a < b`: every impossible
/// pair between clusters `a` and `b` holds a mention of `odd`, `code_sum` is the
/// sum of codes between the two clusters without those mentions, and
/// `close_count` counts those codes at most `close_code`.
///
/// The test is on the linkage height: the mean code minus `bonus_code` times the
/// share of close codes must be below `max_mean_code`. A zero `bonus_code` makes
/// it the plain mean code.
///
/// One front-to-back pass over the matrix collects, per cluster pair, the sum
/// of codes, the count of close codes and the count of impossible pairs, and per mention and cluster,
/// whether the mention has an impossible pair with that cluster. It runs on
/// the calling thread: the clustering step calls it while the pool scores the
/// next block, and a pool job would wait behind that scoring.
fn find_candidates(
    codes: &[u8],
    cluster: &[i64],
    cluster_count: usize,
    min_side: usize,
    max_mean_code: f64,
    max_odd: usize,
    close_code: i64,
    bonus_code: f64,
) -> Vec<Candidate> {
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
    let mut close_counts = vec![0u64; k * (k + 1) / 2];
    // Bit m of row c: mention m has an impossible pair with cluster c.
    let words = n.div_ceil(64);
    let mut conflicts = vec![0u64; k * words];
    let mut row_sums = vec![0u32; k];
    let mut row_hard = vec![0u32; k];
    let mut row_close = vec![0u32; k];
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
            if is_close(code, close_code) {
                row_close[label as usize] += 1;
            }
            if code == HARD {
                row_hard[label as usize] += 1;
                own_conflicts[j / 64] |= 1 << (j % 64);
            }
        }
        for other in 0..k {
            // An impossible pair adds 255, so a zero sum has none. Codes of zero can
            // still be close, so a zero sum with close codes is recorded too.
            if row_sums[other] == 0 && row_close[other] == 0 {
                continue;
            }
            let index = pair_index(own, other);
            code_sums[index] += row_sums[other] as u64;
            row_sums[other] = 0;
            close_counts[index] += row_close[other] as u64;
            row_close[other] = 0;
            if row_hard[other] > 0 {
                hard_counts[index] += row_hard[other];
                row_hard[other] = 0;
                conflicts[other * words + i / 64] |= 1 << (i % 64);
            }
        }
    }

    let conflicting = |own: usize, other: usize| -> Vec<usize> {
        members[own]
            .iter()
            .copied()
            .filter(|&mention| (conflicts[other * words + mention / 64] >> (mention % 64)) & 1 == 1)
            .collect()
    };
    let mut found = Vec::new();
    for a in 0..k {
        for b in a + 1..k {
            let index = pair_index(a, b);
            let hard = hard_counts[index] as usize;
            let (size_a, size_b) = (members[a].len(), members[b].len());
            // One mention covers at most as many impossible pairs as the other cluster has mentions.
            if hard == 0 || hard > max_odd * size_a.max(size_b) {
                continue;
            }
            if size_a < min_side || size_b < min_side || size_a + size_b <= 2 * min_side {
                continue;
            }
            let (conflicting_a, conflicting_b) = (conflicting(a, b), conflicting(b, a));
            for odd in odd_mention_sets(codes, n, &conflicting_a, &conflicting_b, max_odd) {
                let odd_in_a = odd.iter().filter(|&&m| cluster[m] as usize == a).count();
                let odd_in_b = odd.len() - odd_in_a;
                if size_a - odd_in_a < min_side || size_b - odd_in_b < min_side {
                    continue;
                }
                let (mut removed, mut removed_close) = (0u64, 0u64);
                for &mention in &odd {
                    let across = if cluster[mention] as usize == a {
                        &members[b]
                    } else {
                        &members[a]
                    };
                    for &other in across {
                        let code = pair_code(codes, n, mention, other);
                        removed += code as u64;
                        removed_close += u64::from(is_close(code, close_code));
                    }
                }
                // The pair of two odd mentions from different clusters was counted twice.
                if let [first, second] = odd[..] {
                    if cluster[first] != cluster[second] {
                        let code = pair_code(codes, n, first, second);
                        removed -= code as u64;
                        removed_close -= u64::from(is_close(code, close_code));
                    }
                }
                let code_sum = code_sums[index] - removed;
                let close_count = close_counts[index] - removed_close;
                let rest = ((size_a - odd_in_a) * (size_b - odd_in_b)) as f64;
                let height_sum = code_sum as f64 - bonus_code * close_count as f64;
                if height_sum < max_mean_code * rest {
                    found.push((a, b, odd, code_sum, close_count));
                }
            }
        }
    }
    found.sort_unstable();
    found
}

/// Find cluster pairs that at most `max_odd` (1 or 2) odd mentions keep apart.
///
/// `cluster` holds each mention's cluster index in `0..cluster_count`, or -1
/// for a mention outside every eligible cluster. Each result is
/// `(a, b, odd, code_sum, close_count)` with `a < b`: every impossible pair
/// between `a` and `b` holds a mention of `odd`, both clusters keep `min_side`
/// mentions without them, and the mean code between the clusters without them,
/// less `bonus_code` times the share of codes at most `close_code`, is below
/// `max_mean_code`. No smaller set of mentions is in every impossible pair.
/// The defaults (`close_code` -1, `bonus_code` 0) test the plain mean code.
#[pyfunction]
#[pyo3(signature = (codes, cluster, cluster_count, min_side, max_mean_code, max_odd, close_code = -1, bonus_code = 0.0))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn name_conflict_candidates<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray1<'py, u8>,
    cluster: PyReadonlyArray1<'py, i64>,
    cluster_count: usize,
    min_side: usize,
    max_mean_code: f64,
    max_odd: usize,
    close_code: i64,
    bonus_code: f64,
) -> PyResult<Vec<Candidate>> {
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
    if !(1..=2).contains(&max_odd) {
        return Err(PyValueError::new_err("max_odd must be 1 or 2"));
    }
    if !bonus_code.is_finite() || bonus_code < 0.0 {
        return Err(PyValueError::new_err(
            "bonus_code must be finite and non-negative",
        ));
    }
    Ok(py.allow_threads(|| {
        find_candidates(
            codes,
            cluster,
            cluster_count,
            min_side,
            max_mean_code,
            max_odd,
            close_code,
            bonus_code,
        )
    }))
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
        assert_eq!(quantize(&[0.0, 0.5, 1.0, 1e4], 1e4, &mut codes), SAW_HARD);
        assert_eq!(codes, vec![0, 127, 254, 255]);
        let mut codes = vec![0u8; 2];
        assert_eq!(quantize(&[0.2, 1.0], 1e4, &mut codes), 0);
    }

    #[test]
    fn quantize_chunk_matches_the_pooled_quantize() {
        let distances = [0.0, 0.5, 1.0, 1e4, 0.25, 0.999];
        let (mut pooled, mut serial) = (vec![0u8; 6], vec![0u8; 6]);
        assert_eq!(
            quantize(&distances, 1e4, &mut pooled),
            quantize_chunk(&distances, 1e4, &mut serial)
        );
        assert_eq!(pooled, serial);
    }

    #[test]
    fn quantize_reports_values_that_are_neither_model_nor_hard() {
        for other in [1.0001, 5.0, 1e12, f64::NAN, f64::INFINITY] {
            let mut codes = vec![0u8; 3];
            let saw = quantize(&[0.5, other, 1e4], 1e4, &mut codes);
            assert_eq!(saw, SAW_HARD | SAW_OTHER, "{other}");
            // The value never reads as an impossible pair.
            assert_eq!(codes, vec![127, 254, 255], "{other}");
        }
    }

    /// Mentions 0..7 are cluster 0 and 7..14 are cluster 1; `hard` marks impossible pairs.
    fn two_clusters(hard: impl Fn(usize, usize) -> bool) -> (Vec<u8>, Vec<i64>) {
        let cluster: Vec<i64> = (0..14).map(|i| if i < 7 { 0 } else { 1 }).collect();
        let codes = condensed(14, |i, j| {
            if (i < 7) == (j < 7) {
                10
            } else if hard(i, j) {
                HARD
            } else {
                20
            }
        });
        (codes, cluster)
    }

    #[test]
    fn finds_one_odd_mention_in_either_cluster() {
        let (codes, cluster) = two_clusters(|i, _| i == 6);
        assert_eq!(
            find_candidates(&codes, &cluster, 2, 5, 60.0, 2, -1, 0.0),
            vec![(0, 1, vec![6], 20 * 6 * 7, 0)]
        );
        // The same block with the cluster indexes swapped.
        let swapped: Vec<i64> = cluster.iter().map(|label| 1 - label).collect();
        assert_eq!(
            find_candidates(&codes, &swapped, 2, 5, 60.0, 2, -1, 0.0),
            vec![(0, 1, vec![6], 20 * 6 * 7, 0)]
        );
        // A higher mean or too few mentions: no candidate.
        assert!(find_candidates(&codes, &cluster, 2, 5, 20.0, 2, -1, 0.0).is_empty());
        assert!(find_candidates(&codes, &cluster, 2, 7, 60.0, 2, -1, 0.0).is_empty());
    }

    #[test]
    fn finds_two_odd_mentions_only_when_allowed() {
        // Both in one cluster.
        let (codes, cluster) = two_clusters(|i, _| i == 5 || i == 6);
        assert!(find_candidates(&codes, &cluster, 2, 5, 60.0, 1, -1, 0.0).is_empty());
        assert_eq!(
            find_candidates(&codes, &cluster, 2, 5, 60.0, 2, -1, 0.0),
            vec![(0, 1, vec![5, 6], 20 * 5 * 7, 0)]
        );
        // One in each cluster: mention 6 against all of cluster 1, mention 7 against all of cluster 0.
        let (codes, cluster) = two_clusters(|i, j| i == 6 || j == 7);
        assert_eq!(
            find_candidates(&codes, &cluster, 2, 5, 60.0, 2, -1, 0.0),
            vec![(0, 1, vec![6, 7], 20 * 6 * 6, 0)]
        );
        // Two separate impossible pairs: one mention from each pair works.
        let (codes, cluster) = two_clusters(|i, j| (i, j) == (5, 8) || (i, j) == (6, 9));
        let odd: Vec<Vec<usize>> = find_candidates(&codes, &cluster, 2, 5, 60.0, 2, -1, 0.0)
            .into_iter()
            .map(|(_, _, odd, _, _)| odd)
            .collect();
        assert_eq!(odd, vec![vec![5, 6], vec![5, 9], vec![6, 8], vec![8, 9]]);
        // Three mentions behind the impossible pairs: no candidate.
        let (codes, cluster) = two_clusters(|i, _| i >= 4);
        assert!(find_candidates(&codes, &cluster, 2, 4, 60.0, 2, -1, 0.0).is_empty());
    }

    #[test]
    fn close_codes_lower_the_height() {
        // Mention 6 conflicts with all of cluster 1. Without it, cross codes are 2
        // (close) when i + j is even and 60 otherwise: 21 of each.
        let cluster: Vec<i64> = (0..14).map(|i| if i < 7 { 0 } else { 1 }).collect();
        let codes = condensed(14, |i, j| {
            if (i < 7) == (j < 7) {
                10
            } else if i == 6 {
                HARD
            } else if (i + j) % 2 == 0 {
                2
            } else {
                60
            }
        });
        // The plain mean code is 31, above a limit of 20.
        assert!(find_candidates(&codes, &cluster, 2, 5, 20.0, 2, -1, 0.0).is_empty());
        // With a bonus of 40 codes per close share the height is 31 - 20 = 11.
        let found = find_candidates(&codes, &cluster, 2, 5, 20.0, 2, 11, 40.0);
        assert_eq!(found, vec![(0, 1, vec![6], 21 * 2 + 21 * 60, 21)]);
        // Codes above close_code are not close.
        assert!(find_candidates(&codes, &cluster, 2, 5, 20.0, 2, 1, 40.0).is_empty());
    }

    #[test]
    fn zero_codes_count_as_close() {
        // Without mention 6, cross pairs with mention 13 have code 0: six close codes.
        let cluster: Vec<i64> = (0..14).map(|i| if i < 7 { 0 } else { 1 }).collect();
        let codes = condensed(14, |i, j| {
            if (i < 7) == (j < 7) {
                10
            } else if i == 6 {
                HARD
            } else if j == 13 {
                0
            } else {
                60
            }
        });
        let found = find_candidates(&codes, &cluster, 2, 5, 255.0, 2, 11, 0.0);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].4, 6);
    }
}
