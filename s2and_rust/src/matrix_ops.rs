//! Column gathers for pair-feature matrices.

use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::rayon_pool::install_with_optional_rayon_pool;

/// Rows per parallel task; single rows are too small to schedule.
const GATHER_ROWS_PER_TASK: usize = 1024;

fn gather_columns(features: ArrayView2<'_, f64>, columns: &[usize], out: &mut [f64]) {
    let width = columns.len();
    out.par_chunks_mut(width * GATHER_ROWS_PER_TASK)
        .enumerate()
        .for_each(|(task, block)| {
            for (offset, row_out) in block.chunks_exact_mut(width).enumerate() {
                let row = features.row(task * GATHER_ROWS_PER_TASK + offset);
                for (value, column) in row_out.iter_mut().zip(columns) {
                    *value = row[*column];
                }
            }
        });
}

/// `np.take(features, columns, axis=1)` as a C-ordered copy, gathered on the pool.
///
/// numpy gathers on one thread, and linker and block matrices hold millions of
/// rows. Values are copied unchanged, so results equal numpy's.
#[pyfunction]
#[pyo3(signature = (features, columns, num_threads = None))]
pub(crate) fn take_feature_columns<'py>(
    py: Python<'py>,
    features: PyReadonlyArray2<'py, f64>,
    columns: Vec<usize>,
    num_threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = features.as_array();
    let (row_count, column_count) = view.dim();
    if let Some(column) = columns.iter().find(|column| **column >= column_count) {
        return Err(PyValueError::new_err(format!(
            "column {column} is outside the {column_count} feature columns"
        )));
    }
    let mut out = vec![0.0f64; row_count * columns.len()];
    if !out.is_empty() {
        // The GIL stays held, so Python cannot change the input meanwhile;
        // the pool threads only read it.
        install_with_optional_rayon_pool(num_threads, || gather_columns(view, &columns, &mut out));
    }
    let array = Array2::from_shape_vec((row_count, columns.len()), out)
        .expect("the output holds row_count rows of the chosen columns");
    Ok(array.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_matches_indexing_across_task_boundaries() {
        let rows = 2 * GATHER_ROWS_PER_TASK + 7;
        let features = Array2::from_shape_fn((rows, 5), |(row, column)| (row * 10 + column) as f64);
        let columns = [4, 0, 2, 2];
        let mut out = vec![0.0; rows * columns.len()];
        gather_columns(features.view(), &columns, &mut out);
        for row in 0..rows {
            for (position, column) in columns.iter().enumerate() {
                assert_eq!(
                    out[row * columns.len() + position],
                    features[[row, *column]]
                );
            }
        }
        // Strided views, such as a column slice, gather the same values.
        let sliced = features.slice(numpy::ndarray::s![.., 1..4]);
        let mut sliced_out = vec![0.0; rows * 2];
        gather_columns(sliced, &[2, 0], &mut sliced_out);
        assert_eq!(sliced_out[0], features[[0, 3]]);
        assert_eq!(sliced_out[rows * 2 - 1], features[[rows - 1, 1]]);
    }
}
