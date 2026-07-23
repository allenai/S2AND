use arrow::array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int64Array,
    LargeListArray, LargeStringArray, ListArray, StringArray,
};
use arrow::datatypes::DataType;
use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use std::borrow::Cow;
use std::fs::File;

pub(crate) fn io_error_to_py(context: &str, path: &str, err: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyIOError::new_err(format!("{context} '{}': {err}", path))
}

pub(crate) fn arrow_error_to_py(context: &str, path: &str, err: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{context} '{}': {err}", path))
}

pub(crate) fn read_arrow_batches(path: &str) -> PyResult<Vec<RecordBatch>> {
    let file = File::open(path)
        .map_err(|err| io_error_to_py("failed to open Arrow IPC file", path, err))?;
    let reader = ArrowFileReader::try_new(file, None)
        .map_err(|err| arrow_error_to_py("failed to read Arrow IPC schema from", path, err))?;
    reader
        .map(|batch| {
            batch.map_err(|err| {
                arrow_error_to_py("failed to read Arrow IPC record batch from", path, err)
            })
        })
        .collect()
}

pub(crate) fn arrow_column_index(batch: &RecordBatch, name: &str, path: &str) -> PyResult<usize> {
    batch.schema().index_of(name).map_err(|err| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "missing Arrow column '{name}' in '{path}': {err}"
        ))
    })
}

pub(crate) fn arrow_optional_column_index(batch: &RecordBatch, name: &str) -> Option<usize> {
    batch.schema().index_of(name).ok()
}

pub(crate) enum ArrowStringColumn<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
}

impl<'a> ArrowStringColumn<'a> {
    pub(crate) fn from_string_array(array: &'a dyn Array, context: &str) -> PyResult<Self> {
        match array.data_type() {
            DataType::Utf8 => Ok(Self::Utf8(
                array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "{context} is not a Utf8 array"
                        ))
                    })?,
            )),
            DataType::LargeUtf8 => Ok(Self::LargeUtf8(
                array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "{context} is not a LargeUtf8 array"
                        ))
                    })?,
            )),
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{context} must be a string column, got {other:?}"
            ))),
        }
    }

    pub(crate) fn optional_value(&self, row: usize) -> Option<Cow<'a, str>> {
        match self {
            Self::Utf8(values) => (!values.is_null(row)).then(|| Cow::Borrowed(values.value(row))),
            Self::LargeUtf8(values) => {
                (!values.is_null(row)).then(|| Cow::Borrowed(values.value(row)))
            }
        }
    }

    pub(crate) fn optional_owned(&self, row: usize) -> Option<String> {
        self.optional_value(row).map(Cow::into_owned)
    }

    pub(crate) fn required_value(&self, row: usize, context: &str) -> PyResult<Cow<'a, str>> {
        self.optional_value(row).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("{context} is null at row {row}"))
        })
    }

    pub(crate) fn required_borrowed_value(&self, row: usize, context: &str) -> PyResult<&'a str> {
        let value = match self {
            Self::Utf8(values) => (!values.is_null(row)).then(|| (*values).value(row)),
            Self::LargeUtf8(values) => (!values.is_null(row)).then(|| (*values).value(row)),
        };
        value.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("{context} is null at row {row}"))
        })
    }
}

pub(crate) struct ArrowI64Column<'a>(&'a Int64Array);

impl<'a> ArrowI64Column<'a> {
    pub(crate) fn from_i64_array(array: &'a dyn Array, context: &str) -> PyResult<Self> {
        match array.data_type() {
            DataType::Int64 => Ok(Self(
                array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not an Int64 array"
                    ))
                })?,
            )),
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{context} must be an int64 column, got {other:?}"
            ))),
        }
    }

    pub(crate) fn optional_value(&self, row: usize, _context: &str) -> PyResult<Option<i64>> {
        Ok((!self.0.is_null(row)).then(|| self.0.value(row)))
    }

    pub(crate) fn required_value(&self, row: usize, context: &str) -> PyResult<i64> {
        self.optional_value(row, context)?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("{context} is null at row {row}"))
        })
    }
}

pub(crate) fn arrow_optional_bool(
    array: &dyn Array,
    row: usize,
    context: &str,
) -> PyResult<Option<bool>> {
    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::Boolean => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not a Boolean array"
                    ))
                })?;
            Ok(Some(values.value(row)))
        }
        DataType::Null => Ok(None),
        other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{context} must be a boolean column, got {other:?}"
        ))),
    }
}

pub(crate) fn arrow_optional_f64(
    array: &dyn Array,
    row: usize,
    context: &str,
) -> PyResult<Option<f64>> {
    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not a Float64 array"
                    ))
                })?;
            Ok(Some(values.value(row)))
        }
        DataType::Null => Ok(None),
        other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{context} must be a Float64 column, got {other:?}"
        ))),
    }
}

pub(crate) fn arrow_string_array_values(array: &dyn Array, context: &str) -> PyResult<Vec<String>> {
    match array.data_type() {
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!("{context} is not a Utf8 array"))
                })?;
            let mut out = Vec::with_capacity(values.len());
            for idx in 0..values.len() {
                if values.is_null(idx) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{context} cannot contain null list elements"
                    )));
                }
                out.push(values.value(idx).to_string());
            }
            Ok(out)
        }
        DataType::LargeUtf8 => {
            let values = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not a LargeUtf8 array"
                    ))
                })?;
            let mut out = Vec::with_capacity(values.len());
            for idx in 0..values.len() {
                if values.is_null(idx) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{context} cannot contain null list elements"
                    )));
                }
                out.push(values.value(idx).to_string());
            }
            Ok(out)
        }
        DataType::Null => {
            if array.len() == 0 {
                Ok(Vec::new())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{context} cannot contain null list elements"
                )))
            }
        }
        other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{context} list values must be strings, got {other:?}"
        ))),
    }
}

pub(crate) fn arrow_optional_string_list(
    array: &dyn Array,
    row: usize,
    context: &str,
) -> PyResult<Vec<String>> {
    if array.is_null(row) {
        return Ok(Vec::new());
    }
    match array.data_type() {
        DataType::List(_) => {
            let values = array.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err(format!("{context} is not a List array"))
            })?;
            let item_values = values.value(row);
            arrow_string_array_values(item_values.as_ref(), context)
        }
        DataType::LargeList(_) => {
            let values = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not a LargeList array"
                    ))
                })?;
            let item_values = values.value(row);
            arrow_string_array_values(item_values.as_ref(), context)
        }
        other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{context} must be a list<string> column, got {other:?}"
        ))),
    }
}

pub(crate) fn arrow_optional_f32_vector(
    array: &dyn Array,
    row: usize,
    context: &str,
) -> PyResult<Option<Vec<f32>>> {
    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::FixedSizeList(_, _) => {
            let values = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not a FixedSizeList array"
                    ))
                })?;
            let item_values = values.value(row);
            let floats = item_values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} FixedSizeList values must be float32"
                    ))
                })?;
            let mut out = Vec::with_capacity(floats.len());
            for idx in 0..floats.len() {
                if floats.is_null(idx) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{context} has null float value at row {row}, offset {idx}"
                    )));
                }
                out.push(floats.value(idx));
            }
            Ok(Some(out))
        }
        other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{context} must be a fixed_size_list<float32> column, got {other:?}"
        ))),
    }
}
