use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3::Bound;

pub(crate) fn extract_path_mapping_string(
    paths: &Bound<'_, PyAny>,
    key: &str,
    required: bool,
) -> PyResult<Option<String>> {
    let dict = paths.downcast::<PyDict>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("Arrow path bundle must be a dict-like mapping")
    })?;
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract::<String>()?)),
        _ if required => Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "Arrow path bundle is missing required key: {key}"
        ))),
        _ => Ok(None),
    }
}

pub(crate) fn extract_name_counts_index_path(paths: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let dict = paths.downcast::<PyDict>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("Arrow path bundle must be a dict-like mapping")
    })?;
    reject_legacy_name_counts_path(dict)?;
    optional_name_counts_index_path_from_py_dict(dict)
}

fn reject_legacy_name_counts_path(paths: &Bound<'_, PyDict>) -> PyResult<()> {
    for key in ["name_counts", "name_counts_index_dir"] {
        if matches!(paths.get_item(key)?, Some(value) if !value.is_none()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "legacy Arrow path key '{key}' is unsupported; use name_counts_index"
            )));
        }
    }
    Ok(())
}

pub(crate) fn required_path_from_py_dict(paths: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    match paths.get_item(key)? {
        Some(value) => value.extract(),
        None => Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "paths must include '{key}'"
        ))),
    }
}

pub(crate) fn optional_path_from_py_dict(
    paths: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<String>> {
    match paths.get_item(key)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract::<String>()?)),
        _ => Ok(None),
    }
}

pub(crate) fn optional_name_counts_index_path_from_py_dict(
    paths: &Bound<'_, PyDict>,
) -> PyResult<Option<String>> {
    optional_path_from_py_dict(paths, "name_counts_index")
}

pub(crate) struct RawArrowPlannerPaths {
    pub(crate) signatures_path: String,
    pub(crate) papers_path: String,
    pub(crate) paper_authors_path: String,
    pub(crate) cluster_seeds_path: String,
    pub(crate) cluster_seed_disallows_path: Option<String>,
    pub(crate) specter_path: Option<String>,
    pub(crate) name_counts_index_path: Option<String>,
    pub(crate) signatures_batch_index_path: Option<String>,
    pub(crate) papers_batch_index_path: Option<String>,
    pub(crate) paper_authors_batch_index_path: Option<String>,
    pub(crate) specter_batch_index_path: Option<String>,
}

impl RawArrowPlannerPaths {
    pub(crate) fn from_py_dict(paths: &Bound<'_, PyDict>) -> PyResult<Self> {
        Self::from_py_dict_with_cluster_seeds_path(
            paths,
            Some(required_path_from_py_dict(paths, "cluster_seeds")?),
        )
    }

    pub(crate) fn from_py_dict_with_cluster_seeds_path(
        paths: &Bound<'_, PyDict>,
        cluster_seeds_path: Option<String>,
    ) -> PyResult<Self> {
        reject_legacy_name_counts_path(paths)?;
        Ok(Self {
            signatures_path: required_path_from_py_dict(paths, "signatures")?,
            papers_path: required_path_from_py_dict(paths, "papers")?,
            paper_authors_path: required_path_from_py_dict(paths, "paper_authors")?,
            cluster_seeds_path: cluster_seeds_path.unwrap_or_default(),
            cluster_seed_disallows_path: optional_path_from_py_dict(
                paths,
                "cluster_seed_disallows",
            )?,
            specter_path: optional_path_from_py_dict(paths, "specter")?,
            name_counts_index_path: optional_name_counts_index_path_from_py_dict(paths)?,
            signatures_batch_index_path: optional_path_from_py_dict(
                paths,
                "signatures_batch_index",
            )?,
            papers_batch_index_path: optional_path_from_py_dict(paths, "papers_batch_index")?,
            paper_authors_batch_index_path: optional_path_from_py_dict(
                paths,
                "paper_authors_batch_index",
            )?,
            specter_batch_index_path: optional_path_from_py_dict(paths, "specter_batch_index")?,
        })
    }
}

pub(crate) fn raw_arrow_feature_paths_from_py_dict(
    paths: &Bound<'_, PyDict>,
) -> PyResult<RawArrowPlannerPaths> {
    RawArrowPlannerPaths::from_py_dict_with_cluster_seeds_path(paths, None)
}
