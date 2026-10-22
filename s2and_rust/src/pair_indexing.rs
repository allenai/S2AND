use pyo3::PyResult;
use rayon::prelude::*;

use crate::rayon_pool::install_with_optional_rayon_pool;

/// Pairs per parallel task when listing a range of the upper triangle.
const PAIRS_PER_TASK: usize = 1 << 16;

pub(crate) fn upper_triangle_total_pairs(block_size: usize) -> usize {
    block_size.saturating_mul(block_size.saturating_sub(1)) / 2
}

/// Number of pairs in `[start_offset, start_offset + max_pairs)` of the
/// row-major upper triangle, rejecting a start beyond the last pair.
pub(crate) fn upper_triangle_range_len(
    block_size: usize,
    start_offset: usize,
    max_pairs: Option<usize>,
) -> PyResult<usize> {
    let total_pairs = upper_triangle_total_pairs(block_size);
    if start_offset > total_pairs {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "start_offset out of range: start_offset={} total_pairs={}",
            start_offset, total_pairs
        )));
    }
    let remaining = total_pairs - start_offset;
    Ok(max_pairs.unwrap_or(remaining).min(remaining))
}

/// Pairs before row `row` of the upper triangle of a block of `block_size`.
fn row_start(block_size: usize, row: usize) -> usize {
    row * block_size - row * (row + 1) / 2
}

/// The `(left, right)` pair at `offset` of the row-major upper triangle.
///
/// Requires `offset < upper_triangle_total_pairs(block_size)`. The row comes
/// from the quadratic's root and is then corrected, so rounding cannot move it.
pub(crate) fn upper_triangle_pair_at(block_size: usize, offset: usize) -> (usize, usize) {
    debug_assert!(offset < upper_triangle_total_pairs(block_size));
    let span = (2 * block_size - 1) as f64;
    let estimate = (span - (span * span - 8.0 * offset as f64).max(0.0).sqrt()) / 2.0;
    let mut row = (estimate.max(0.0) as usize).min(block_size - 2);
    while row > 0 && row_start(block_size, row) > offset {
        row -= 1;
    }
    while row + 2 < block_size && row_start(block_size, row + 1) <= offset {
        row += 1;
    }
    (row, row + 1 + (offset - row_start(block_size, row)))
}

/// Pairs of `[start_offset, start_offset + max_pairs)` of the row-major upper
/// triangle, listed on the pool: chunk lists hold millions of pairs.
pub(crate) fn upper_triangle_pairs_for_range(
    block_size: usize,
    start_offset: usize,
    max_pairs: Option<usize>,
    num_threads: Option<usize>,
) -> PyResult<Vec<(usize, usize)>> {
    let pair_count = upper_triangle_range_len(block_size, start_offset, max_pairs)?;
    let mut pairs = vec![(0usize, 0usize); pair_count];
    install_with_optional_rayon_pool(num_threads, || {
        pairs
            .par_chunks_mut(PAIRS_PER_TASK)
            .enumerate()
            .for_each(|(task, slots)| {
                let (mut left, mut right) =
                    upper_triangle_pair_at(block_size, start_offset + task * PAIRS_PER_TASK);
                for slot in slots {
                    *slot = (left, right);
                    right += 1;
                    if right >= block_size {
                        left += 1;
                        right = left + 1;
                    }
                }
            });
    });
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The serial walk this module used before the closed form.
    fn reference_pairs(
        block_size: usize,
        start_offset: usize,
        pair_count: usize,
    ) -> Vec<(usize, usize)> {
        let mut local_i = 0usize;
        let mut offset_in_row = start_offset;
        while local_i + 1 < block_size {
            let row_pairs = block_size - local_i - 1;
            if offset_in_row < row_pairs {
                break;
            }
            offset_in_row -= row_pairs;
            local_i += 1;
        }
        let mut local_j = local_i + 1 + offset_in_row;
        let mut pairs = Vec::with_capacity(pair_count);
        for _ in 0..pair_count {
            pairs.push((local_i, local_j));
            local_j += 1;
            if local_j >= block_size {
                local_i += 1;
                local_j = local_i + 1;
            }
        }
        pairs
    }

    #[test]
    fn pair_at_matches_the_serial_walk_at_every_offset() {
        for block_size in 2..60 {
            let total = upper_triangle_total_pairs(block_size);
            let expected = reference_pairs(block_size, 0, total);
            for (offset, pair) in expected.iter().enumerate() {
                assert_eq!(upper_triangle_pair_at(block_size, offset), *pair);
            }
        }
        // Large blocks, near row boundaries where rounding could slip.
        for block_size in [100_003usize, 1_000_000, 3_000_017] {
            let total = upper_triangle_total_pairs(block_size);
            for row in [
                0usize,
                1,
                2,
                block_size / 3,
                block_size / 2,
                block_size - 3,
                block_size - 2,
            ] {
                let start = row_start(block_size, row);
                for offset in [start.saturating_sub(1), start, start + 1] {
                    if offset < total {
                        let (left, right) = upper_triangle_pair_at(block_size, offset);
                        assert!(row_start(block_size, left) <= offset);
                        assert_eq!(right, left + 1 + offset - row_start(block_size, left));
                        assert!(right < block_size);
                    }
                }
            }
            assert_eq!(
                upper_triangle_pair_at(block_size, total - 1),
                (block_size - 2, block_size - 1)
            );
        }
    }

    #[test]
    fn ranges_match_the_serial_walk_across_task_boundaries() {
        let block_size = 700;
        let total = upper_triangle_total_pairs(block_size);
        for (start, max_pairs) in [
            (0, None),
            (12_345, Some(3 * PAIRS_PER_TASK + 5)),
            (total - 10, Some(100)),
            (total, None),
        ] {
            let count = upper_triangle_range_len(block_size, start, max_pairs).unwrap();
            assert_eq!(
                upper_triangle_pairs_for_range(block_size, start, max_pairs, Some(4)).unwrap(),
                reference_pairs(block_size, start, count)
            );
        }
        assert!(upper_triangle_pairs_for_range(block_size, total + 1, None, None).is_err());
        assert!(upper_triangle_pairs_for_range(1, 0, None, None)
            .unwrap()
            .is_empty());
    }
}
