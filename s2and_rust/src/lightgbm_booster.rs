//! Pure-Rust evaluator for LightGBM text models (`.lgb`).
//!
//! Scope is exactly the subset the S2AND production boosters use: single-model
//! binary-objective GBDTs with numerical splits and None/Zero/NaN missing-value
//! handling. Categorical splits, linear trees, multiclass models, and
//! random-forest averaging are rejected at load time so an unsupported model
//! fails loudly instead of scoring silently wrong.
//!
//! Dense-input and decision semantics mirror LightGBM: finite values in
//! `[-kZeroThreshold, kZeroThreshold]` become zero at the prediction adapter;
//! then `Tree::NumericalDecision` converts NaN to 0.0 unless the split's
//! missing type is NaN, routes Zero/NaN missing values by `default_left`, and
//! otherwise sends `fval <= threshold` left. Raw scores are sums of leaf values
//! in tree order, and binary probabilities apply
//! `1 / (1 + exp(-sigmoid * raw))`. Deterministic parity tests cover the
//! supported split and missing-value cases; this module does not claim an
//! exhaustive proof over every model LightGBM can produce.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::Bound;
use rayon::prelude::*;

use crate::rayon_pool::install_with_optional_rayon_pool;

pub(super) fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustLightGBMBooster>()?;
    Ok(())
}

/// LightGBM's kZeroThreshold is the float literal 1e-35f widened to double
/// (include/LightGBM/meta.h); the same widened value appears verbatim as
/// zero-boundary split thresholds in saved model files.
const K_ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;
const K_ZERO_THRESHOLD_F32: f32 = 1e-35_f32;

const CATEGORICAL_MASK: u8 = 1;
const DEFAULT_LEFT_MASK: u8 = 2;

const MISSING_TYPE_NONE: u8 = 0;
const MISSING_TYPE_ZERO: u8 = 1;
const MISSING_TYPE_NAN: u8 = 2;
const MAX_NUM_FEATURES: usize = u16::MAX as usize + 1;
// On the production feature widths, each row tile and its margins fit in L1.
const ROW_TILE: usize = 64;

#[inline]
fn missing_type(decision_type: u8) -> u8 {
    (decision_type >> 2) & 3
}

#[inline]
fn is_zero(fval: f64) -> bool {
    (-K_ZERO_THRESHOLD..=K_ZERO_THRESHOLD).contains(&fval)
}

#[inline]
fn is_zero_f32(fval: f32) -> bool {
    (-K_ZERO_THRESHOLD_F32..=K_ZERO_THRESHOLD_F32).contains(&fval)
}

#[inline]
fn normalize_dense_zero(fval: f64) -> f64 {
    if is_zero(fval) {
        0.0
    } else {
        fval
    }
}

#[inline]
fn normalize_dense_zero_f32(fval: f32) -> f32 {
    if is_zero_f32(fval) {
        0.0
    } else {
        fval
    }
}

#[derive(Debug, Clone)]
struct LgbTree {
    num_leaves: usize,
    split_feature: Vec<u16>,
    threshold: Vec<f64>,
    decision_type: Vec<u8>,
    left_child: Vec<i32>,
    right_child: Vec<i32>,
    leaf_value: Vec<f64>,
}

impl LgbTree {
    #[inline]
    fn predict(&self, row: &[f64]) -> f64 {
        if self.num_leaves == 1 {
            return self.leaf_value[0];
        }
        let mut node = 0usize;
        for _ in 0..self.left_child.len() {
            let decision_type = self.decision_type[node];
            let missing = missing_type(decision_type);
            // LightGBM's dense prediction adapter omits every finite value in
            // the zero window, so the tree observes zero for all missing types.
            let mut fval = normalize_dense_zero(row[self.split_feature[node] as usize]);
            if fval.is_nan() && missing != MISSING_TYPE_NAN {
                fval = 0.0;
            }
            let takes_default = (missing == MISSING_TYPE_ZERO && is_zero(fval))
                || (missing == MISSING_TYPE_NAN && fval.is_nan());
            let child = if takes_default {
                if decision_type & DEFAULT_LEFT_MASK != 0 {
                    self.left_child[node]
                } else {
                    self.right_child[node]
                }
            } else if fval <= self.threshold[node] {
                self.left_child[node]
            } else {
                self.right_child[node]
            };
            if child < 0 {
                return self.leaf_value[(!child) as usize];
            }
            node = child as usize;
        }
        unreachable!("validated LightGBM trees must terminate at a leaf")
    }

    #[inline]
    fn predict_f32(&self, row: &[f32]) -> f64 {
        if self.num_leaves == 1 {
            return self.leaf_value[0];
        }
        let mut node = 0usize;
        for _ in 0..self.left_child.len() {
            // Parsing proves all split arrays have identical lengths, every
            // feature index is in range, and every internal child points to a
            // later in-range node. Avoid repeating those bounds checks for
            // every tree and row in the float32 production hot path.
            let (decision_type, feature, threshold, left_child, right_child) = unsafe {
                (
                    *self.decision_type.get_unchecked(node),
                    usize::from(*self.split_feature.get_unchecked(node)),
                    *self.threshold.get_unchecked(node),
                    *self.left_child.get_unchecked(node),
                    *self.right_child.get_unchecked(node),
                )
            };
            let missing = missing_type(decision_type);
            // The batch boundary asserts row width against the same model
            // feature count used to validate split_feature above.
            let mut fval = normalize_dense_zero_f32(unsafe { *row.get_unchecked(feature) });
            if fval.is_nan() && missing != MISSING_TYPE_NAN {
                fval = 0.0;
            }
            let takes_default = (missing == MISSING_TYPE_ZERO && is_zero_f32(fval))
                || (missing == MISSING_TYPE_NAN && fval.is_nan());
            let child = if takes_default {
                if decision_type & DEFAULT_LEFT_MASK != 0 {
                    left_child
                } else {
                    right_child
                }
            } else if f64::from(fval) <= threshold {
                left_child
            } else {
                right_child
            };
            if child < 0 {
                // Negative children were validated against num_leaves.
                return unsafe { *self.leaf_value.get_unchecked((!child) as usize) };
            }
            node = child as usize;
        }
        unreachable!("validated LightGBM trees must terminate at a leaf")
    }
}

#[derive(Debug, Clone)]
struct LgbModel {
    num_features: usize,
    sigmoid: f64,
    objective_name: String,
    trees: Vec<LgbTree>,
}

impl LgbModel {
    #[cfg(test)]
    #[inline]
    fn predict_row_raw(&self, row: &[f64]) -> f64 {
        let mut raw = 0.0f64;
        for tree in &self.trees {
            raw += tree.predict(row);
        }
        raw
    }

    #[cfg(test)]
    #[inline]
    fn predict_row_raw_f32(&self, row: &[f32]) -> f64 {
        let mut raw = 0.0f64;
        for tree in &self.trees {
            raw += tree.predict_f32(row);
        }
        raw
    }

    #[inline]
    fn raw_to_probability(&self, raw: f64) -> f64 {
        1.0 / (1.0 + (-self.sigmoid * raw).exp())
    }
}

fn model_error(message: String) -> PyErr {
    PyValueError::new_err(format!("RustLightGBMBooster: {message}"))
}

fn parse_scalar<T: std::str::FromStr>(fields: &[(String, String)], key: &str) -> Result<T, String> {
    let raw = fields
        .iter()
        .find(|(field_key, _)| field_key == key)
        .map(|(_, value)| value.as_str())
        .ok_or_else(|| format!("missing required field {key:?}"))?;
    raw.trim()
        .parse::<T>()
        .map_err(|_| format!("could not parse field {key:?} value {raw:?}"))
}

fn parse_vec<T: std::str::FromStr>(
    fields: &[(String, String)],
    key: &str,
) -> Result<Vec<T>, String> {
    let raw = fields
        .iter()
        .find(|(field_key, _)| field_key == key)
        .map(|(_, value)| value.as_str())
        .unwrap_or("");
    raw.split_whitespace()
        .map(|token| {
            token
                .parse::<T>()
                .map_err(|_| format!("could not parse field {key:?} token {token:?}"))
        })
        .collect()
}

fn parse_tree(
    tree_index: usize,
    fields: &[(String, String)],
    num_features: usize,
) -> Result<LgbTree, String> {
    let describe = |message: String| format!("tree {tree_index}: {message}");

    let num_leaves: usize = parse_scalar(fields, "num_leaves")?;
    if num_leaves == 0 {
        return Err(describe("num_leaves must be >= 1".to_string()));
    }
    let num_cat: i64 = parse_scalar(fields, "num_cat")?;
    if num_cat != 0 {
        return Err(describe(format!(
            "categorical splits are unsupported (num_cat={num_cat})"
        )));
    }
    if let Ok(is_linear) = parse_scalar::<i64>(fields, "is_linear") {
        if is_linear != 0 {
            return Err(describe(
                "linear trees are unsupported (is_linear=1)".to_string(),
            ));
        }
    }

    let leaf_value: Vec<f64> = parse_vec(fields, "leaf_value")?;
    if leaf_value.len() != num_leaves {
        return Err(describe(format!(
            "leaf_value has {} entries, expected num_leaves={num_leaves}",
            leaf_value.len()
        )));
    }

    if num_leaves == 1 {
        return Ok(LgbTree {
            num_leaves,
            split_feature: Vec::new(),
            threshold: Vec::new(),
            decision_type: Vec::new(),
            left_child: Vec::new(),
            right_child: Vec::new(),
            leaf_value,
        });
    }

    let internal_count = num_leaves - 1;
    let split_feature: Vec<u16> = parse_vec(fields, "split_feature")?;
    let threshold: Vec<f64> = parse_vec(fields, "threshold")?;
    let decision_type: Vec<u8> = parse_vec(fields, "decision_type")?;
    let left_child: Vec<i32> = parse_vec(fields, "left_child")?;
    let right_child: Vec<i32> = parse_vec(fields, "right_child")?;

    for (key, observed) in [
        ("split_feature", split_feature.len()),
        ("threshold", threshold.len()),
        ("decision_type", decision_type.len()),
        ("left_child", left_child.len()),
        ("right_child", right_child.len()),
    ] {
        if observed != internal_count {
            return Err(describe(format!(
                "{key} has {observed} entries, expected num_leaves-1={internal_count}"
            )));
        }
    }

    for value in &decision_type {
        if value & CATEGORICAL_MASK != 0 {
            return Err(describe(
                "categorical splits are unsupported (decision_type categorical bit set)"
                    .to_string(),
            ));
        }
        if missing_type(*value) > MISSING_TYPE_NAN {
            return Err(describe(format!(
                "unsupported missing type in decision_type value {value}"
            )));
        }
    }
    for feature in &split_feature {
        if *feature as usize >= num_features {
            return Err(describe(format!(
                "split_feature {feature} out of range for {num_features} features"
            )));
        }
    }
    for (node_index, child) in left_child.iter().chain(right_child.iter()).enumerate() {
        let parent_index = node_index % internal_count;
        if *child < 0 {
            if ((!*child) as usize) >= num_leaves {
                return Err(describe(format!("child index {child} out of range")));
            }
        } else {
            let child_index = *child as usize;
            if child_index >= internal_count {
                return Err(describe(format!("child index {child} out of range")));
            }
            if child_index <= parent_index {
                return Err(describe(format!(
                    "child index {child} must refer to a later internal node than parent {parent_index}"
                )));
            }
        }
    }

    Ok(LgbTree {
        num_leaves,
        split_feature,
        threshold,
        decision_type,
        left_child,
        right_child,
        leaf_value,
    })
}

fn parse_model(model_text: &str) -> Result<LgbModel, String> {
    let mut header_fields: Vec<(String, String)> = Vec::new();
    let mut tree_blocks: Vec<Vec<(String, String)>> = Vec::new();
    let mut current_tree: Option<Vec<(String, String)>> = None;
    let mut saw_tree_magic = false;
    let mut saw_end_of_trees = false;
    let mut saw_first_content_line = false;

    for raw_line in model_text.lines() {
        let line = raw_line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        if !saw_first_content_line {
            saw_first_content_line = true;
            if line != "tree" {
                return Err(
                    "model text does not start with the LightGBM 'tree' magic line".to_string(),
                );
            }
            saw_tree_magic = true;
            continue;
        }
        if line == "tree" {
            return Err(
                "unexpected LightGBM 'tree' magic line after model header started".to_string(),
            );
        }
        if line == "end of trees" {
            if let Some(fields) = current_tree.take() {
                tree_blocks.push(fields);
            }
            saw_end_of_trees = true;
            break;
        }
        let Some((key, value)) = line.split_once('=') else {
            if line == "average_output" {
                return Err("random-forest average_output models are unsupported".to_string());
            }
            continue;
        };
        if key == "Tree" {
            if let Some(fields) = current_tree.take() {
                tree_blocks.push(fields);
            }
            current_tree = Some(Vec::new());
            continue;
        }
        match current_tree.as_mut() {
            Some(fields) => fields.push((key.to_string(), value.to_string())),
            None => header_fields.push((key.to_string(), value.to_string())),
        }
    }
    if let Some(fields) = current_tree.take() {
        tree_blocks.push(fields);
    }

    if !saw_tree_magic {
        return Err("model text does not start with the LightGBM 'tree' magic line".to_string());
    }
    if !saw_end_of_trees {
        return Err("model text is missing the LightGBM 'end of trees' marker".to_string());
    }
    if tree_blocks.is_empty() {
        return Err("model text contains no trees".to_string());
    }

    let version: String = parse_scalar(&header_fields, "version")?;
    if version != "v3" && version != "v4" {
        return Err(format!("unsupported model version {version:?}"));
    }
    let num_class: i64 = parse_scalar(&header_fields, "num_class")?;
    let num_tree_per_iteration: i64 = parse_scalar(&header_fields, "num_tree_per_iteration")?;
    if num_class != 1 || num_tree_per_iteration != 1 {
        return Err(format!(
            "only single-model binary boosters are supported \
             (num_class={num_class}, num_tree_per_iteration={num_tree_per_iteration})"
        ));
    }

    let objective: String = parse_scalar(&header_fields, "objective")?;
    let mut objective_tokens = objective.split_whitespace();
    let objective_name = objective_tokens.next().unwrap_or("").to_string();
    if objective_name != "binary" {
        return Err(format!(
            "only the binary objective is supported, got {objective:?}"
        ));
    }
    let mut sigmoid = 1.0f64;
    for token in objective_tokens {
        if let Some(value) = token.strip_prefix("sigmoid:") {
            sigmoid = value
                .parse::<f64>()
                .map_err(|_| format!("could not parse objective sigmoid in {objective:?}"))?;
        }
    }

    let max_feature_idx: i64 = parse_scalar(&header_fields, "max_feature_idx")?;
    if max_feature_idx < 0 {
        return Err(format!("invalid max_feature_idx={max_feature_idx}"));
    }
    let num_features = max_feature_idx
        .checked_add(1)
        .ok_or_else(|| format!("max_feature_idx={max_feature_idx} is too large"))?
        as usize;
    if num_features > MAX_NUM_FEATURES {
        return Err(format!(
            "models with more than {MAX_NUM_FEATURES} numerical features are unsupported"
        ));
    }

    let trees = tree_blocks
        .iter()
        .enumerate()
        .map(|(tree_index, fields)| parse_tree(tree_index, fields, num_features))
        .collect::<Result<Vec<LgbTree>, String>>()?;
    let tree_sizes: Vec<usize> = parse_vec(&header_fields, "tree_sizes")?;
    if !tree_sizes.is_empty() && tree_sizes.len() != trees.len() {
        return Err(format!(
            "tree_sizes has {} entries, but parsed {} trees",
            tree_sizes.len(),
            trees.len()
        ));
    }

    Ok(LgbModel {
        num_features,
        sigmoid,
        objective_name,
        trees,
    })
}

fn predict_rows(
    model: &LgbModel,
    rows: &[f64],
    num_features: usize,
    num_threads: usize,
    apply_sigmoid: bool,
) -> Vec<f64> {
    let row_count = rows.len() / num_features;
    let mut scores = vec![0.0; row_count];
    let score_tile = |(tile, margins): (&[f64], &mut [f64])| {
        for tree in &model.trees {
            for (row, raw) in tile.chunks_exact(num_features).zip(margins.iter_mut()) {
                *raw += tree.predict(row);
            }
        }
        if apply_sigmoid {
            for raw in margins {
                *raw = model.raw_to_probability(*raw);
            }
        }
    };
    if num_threads > 1 && row_count > 1 {
        install_with_optional_rayon_pool(Some(num_threads), || {
            rows.par_chunks(num_features * ROW_TILE)
                .zip(scores.par_chunks_mut(ROW_TILE))
                .for_each(score_tile);
        });
    } else {
        rows.chunks(num_features * ROW_TILE)
            .zip(scores.chunks_mut(ROW_TILE))
            .for_each(score_tile);
    }
    scores
}

fn predict_rows_f32(
    model: &LgbModel,
    rows: &[f32],
    num_features: usize,
    num_threads: usize,
    apply_sigmoid: bool,
) -> Vec<f64> {
    assert_eq!(
        num_features, model.num_features,
        "float32 scorer width must match the parsed model"
    );
    assert_eq!(
        rows.len() % num_features,
        0,
        "float32 scorer rows must be rectangular"
    );
    // Reuse each tree across a small row tile while keeping the exact tree
    // accumulation order for every row.
    let row_count = rows.len() / num_features;
    let mut scores = vec![0.0; row_count];
    let score_tile = |(tile, margins): (&[f32], &mut [f64])| {
        for tree in &model.trees {
            for (row, raw) in tile.chunks_exact(num_features).zip(margins.iter_mut()) {
                *raw += tree.predict_f32(row);
            }
        }
        if apply_sigmoid {
            for raw in margins {
                *raw = model.raw_to_probability(*raw);
            }
        }
    };
    if num_threads > 1 && row_count > 1 {
        install_with_optional_rayon_pool(Some(num_threads), || {
            rows.par_chunks(num_features * ROW_TILE)
                .zip(scores.par_chunks_mut(ROW_TILE))
                .for_each(score_tile);
        });
    } else {
        rows.chunks(num_features * ROW_TILE)
            .zip(scores.chunks_mut(ROW_TILE))
            .for_each(score_tile);
    }
    scores
}

/// Pure-Rust scorer for S2AND's native LightGBM binary classifiers.
#[pyclass]
pub(crate) struct RustLightGBMBooster {
    model: LgbModel,
}

impl RustLightGBMBooster {
    fn validate_feature_width(&self, column_count: usize) -> PyResult<()> {
        if column_count != self.model.num_features {
            return Err(model_error(format!(
                "features must have {} columns, got {column_count}",
                self.model.num_features
            )));
        }
        Ok(())
    }

    fn predict_array_f64<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        num_threads: Option<usize>,
        apply_sigmoid: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let shape = features.shape();
        let (row_count, column_count) = (shape[0], shape[1]);
        self.validate_feature_width(column_count)?;
        if row_count == 0 {
            return Ok(Vec::<f64>::new().into_pyarray(py));
        }
        let threads = num_threads.unwrap_or(1).max(1);
        let model = &self.model;
        // Owned standard-layout copy so non-C-contiguous inputs work and the
        // buffer can be scored with the GIL released.
        let rows: Vec<f64> = features.as_array().iter().copied().collect();
        let scores = py.allow_threads(move || {
            predict_rows(model, &rows, column_count, threads, apply_sigmoid)
        });
        Ok(scores.into_pyarray(py))
    }

    fn predict_array_f32<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        num_threads: Option<usize>,
        apply_sigmoid: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let shape = features.shape();
        let (row_count, column_count) = (shape[0], shape[1]);
        self.validate_feature_width(column_count)?;
        if row_count == 0 {
            return Ok(Vec::<f64>::new().into_pyarray(py));
        }
        let threads = num_threads.unwrap_or(1).max(1);
        let model = &self.model;
        // Keep the owned GIL-independent copy at the caller's float32 width.
        let rows: Vec<f32> = features.as_array().iter().copied().collect();
        let scores = py.allow_threads(move || {
            predict_rows_f32(model, &rows, column_count, threads, apply_sigmoid)
        });
        Ok(scores.into_pyarray(py))
    }
}

#[pymethods]
impl RustLightGBMBooster {
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        let model_text = std::fs::read_to_string(model_path)
            .map_err(|err| model_error(format!("could not read {model_path:?}: {err}")))?;
        Ok(Self {
            model: parse_model(&model_text).map_err(model_error)?,
        })
    }

    #[staticmethod]
    fn from_string(model_text: &str) -> PyResult<Self> {
        Ok(Self {
            model: parse_model(model_text).map_err(model_error)?,
        })
    }

    fn num_features(&self) -> usize {
        self.model.num_features
    }

    /// In-memory copy support so Python deepcopy does not re-read model files.
    fn __copy__(&self) -> Self {
        Self {
            model: self.model.clone(),
        }
    }

    fn __deepcopy__(&self, _memo: Bound<'_, pyo3::types::PyAny>) -> Self {
        Self {
            model: self.model.clone(),
        }
    }

    fn num_trees(&self) -> usize {
        self.model.trees.len()
    }

    fn objective_name(&self) -> String {
        self.model.objective_name.clone()
    }

    fn sigmoid(&self) -> f64 {
        self.model.sigmoid
    }

    /// Split-level introspection so parity tests can assert real coverage of
    /// decision_type variants instead of passing vacuously.
    fn decision_type_summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut num_splits = 0usize;
        let mut default_left = 0usize;
        let mut missing_none = 0usize;
        let mut missing_zero = 0usize;
        let mut missing_nan = 0usize;
        for tree in &self.model.trees {
            for decision_type in &tree.decision_type {
                num_splits += 1;
                if decision_type & DEFAULT_LEFT_MASK != 0 {
                    default_left += 1;
                }
                match missing_type(*decision_type) {
                    MISSING_TYPE_NONE => missing_none += 1,
                    MISSING_TYPE_ZERO => missing_zero += 1,
                    _ => missing_nan += 1,
                }
            }
        }
        let summary = PyDict::new(py);
        summary.set_item("num_splits", num_splits)?;
        summary.set_item("default_left", default_left)?;
        summary.set_item("missing_none", missing_none)?;
        summary.set_item("missing_zero", missing_zero)?;
        summary.set_item("missing_nan", missing_nan)?;
        Ok(summary)
    }

    /// Raw margin scores (sum of leaf values), matching
    /// `lgb.Booster.predict(..., raw_score=True)` bit-for-bit.
    #[pyo3(signature = (features, num_threads=None))]
    fn predict_raw<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_array_f64(py, features, num_threads, false)
    }

    /// Positive-class probabilities, matching `lgb.Booster.predict(...)`.
    #[pyo3(signature = (features, num_threads=None))]
    fn predict_proba_positive<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_array_f64(py, features, num_threads, true)
    }

    /// Raw margin scores for an existing contiguous float32 feature matrix.
    #[pyo3(signature = (features, num_threads=None))]
    fn predict_raw_f32<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_array_f32(py, features, num_threads, false)
    }

    /// Positive-class probabilities for a float32 feature matrix.
    #[pyo3(signature = (features, num_threads=None))]
    fn predict_proba_positive_f32<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_array_f32(py, features, num_threads, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-tree toy model: tree 0 splits feature 0 with missing_type=NaN and
    /// default_left set (decision_type = 2 | (2 << 2) = 10); tree 1 splits
    /// feature 1 with missing_type=Zero, default right (decision_type = 1 << 2
    /// = 4).
    const TOY_MODEL: &str = "tree\n\
        version=v4\n\
        num_class=1\n\
        num_tree_per_iteration=1\n\
        label_index=0\n\
        max_feature_idx=1\n\
        objective=binary sigmoid:1\n\
        feature_names=Column_0 Column_1\n\
        feature_infos=[0:1] [0:1]\n\
        tree_sizes=1 1\n\
        \n\
        Tree=0\n\
        num_leaves=2\n\
        num_cat=0\n\
        split_feature=0\n\
        split_gain=1\n\
        threshold=0.5\n\
        decision_type=10\n\
        left_child=-1\n\
        right_child=-2\n\
        leaf_value=1 2\n\
        is_linear=0\n\
        shrinkage=1\n\
        \n\
        Tree=1\n\
        num_leaves=2\n\
        num_cat=0\n\
        split_feature=1\n\
        split_gain=1\n\
        threshold=0.25\n\
        decision_type=4\n\
        left_child=-1\n\
        right_child=-2\n\
        leaf_value=10 20\n\
        is_linear=0\n\
        shrinkage=1\n\
        \n\
        end of trees\n";

    fn next_f32(value: f32) -> f32 {
        if value.is_nan() || value == f32::INFINITY {
            return value;
        }
        if value == 0.0 {
            return f32::from_bits(1);
        }
        let bits = value.to_bits();
        if value.is_sign_positive() {
            f32::from_bits(bits + 1)
        } else {
            f32::from_bits(bits - 1)
        }
    }

    fn previous_f32(value: f32) -> f32 {
        if value.is_nan() || value == f32::NEG_INFINITY {
            return value;
        }
        if value == 0.0 {
            return -f32::from_bits(1);
        }
        let bits = value.to_bits();
        if value.is_sign_positive() {
            f32::from_bits(bits - 1)
        } else {
            f32::from_bits(bits + 1)
        }
    }

    #[test]
    fn toy_model_decision_semantics() {
        let model = parse_model(TOY_MODEL).unwrap();
        assert_eq!(model.num_features, 2);
        assert_eq!(model.trees.len(), 2);
        // Plain numerical decisions: feature0 <= 0.5 left, feature1 <= 0.25 left.
        assert_eq!(model.predict_row_raw(&[0.4, 0.2]), 1.0 + 10.0);
        assert_eq!(model.predict_row_raw(&[0.6, 0.3]), 2.0 + 20.0);
        // NaN on a missing_type=NaN split takes default (left here).
        assert_eq!(model.predict_row_raw(&[f64::NAN, 0.3]), 1.0 + 20.0);
        // NaN on a missing_type=Zero split converts to 0.0, which IsZero sends
        // to the default (right here).
        assert_eq!(model.predict_row_raw(&[0.6, f64::NAN]), 2.0 + 20.0);
        // Exact zero on the missing_type=Zero split also takes the default.
        assert_eq!(model.predict_row_raw(&[0.4, 0.0]), 1.0 + 20.0);
        // Just outside the zero threshold compares numerically again.
        let above_zero_threshold = K_ZERO_THRESHOLD * 1.01;
        assert_eq!(
            model.predict_row_raw(&[0.4, above_zero_threshold]),
            1.0 + 10.0
        );
    }

    #[test]
    fn dense_zero_window_is_normalized_before_tree_comparison() {
        assert_eq!(normalize_dense_zero(-K_ZERO_THRESHOLD), 0.0);
        assert_eq!(normalize_dense_zero(-K_ZERO_THRESHOLD * 0.75), 0.0);
        assert_eq!(normalize_dense_zero(K_ZERO_THRESHOLD), 0.0);
        assert_eq!(
            normalize_dense_zero(-K_ZERO_THRESHOLD * 1.01),
            -K_ZERO_THRESHOLD * 1.01
        );
        assert!(normalize_dense_zero(f64::NAN).is_nan());

        assert_eq!(normalize_dense_zero_f32(-K_ZERO_THRESHOLD_F32), 0.0);
        assert_eq!(normalize_dense_zero_f32(-K_ZERO_THRESHOLD_F32 * 0.75), 0.0);
        assert_eq!(normalize_dense_zero_f32(K_ZERO_THRESHOLD_F32), 0.0);
        assert_eq!(
            normalize_dense_zero_f32(-K_ZERO_THRESHOLD_F32 * 1.01),
            -K_ZERO_THRESHOLD_F32 * 1.01
        );
        assert!(normalize_dense_zero_f32(f32::NAN).is_nan());
    }

    #[test]
    fn float32_rows_match_losslessly_widened_float64_rows() {
        let model = parse_model(TOY_MODEL).unwrap();
        let rows_f32 = vec![0.4_f32, 0.2_f32, 0.6_f32, 0.3_f32, f32::NAN, 0.0_f32];
        let rows_f64 = rows_f32
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>();
        assert_eq!(
            predict_rows_f32(&model, &rows_f32, 2, 1, true),
            predict_rows(&model, &rows_f64, 2, 1, true),
        );
    }

    #[test]
    fn float32_missing_semantics_match_losslessly_widened_rows() {
        let values = [
            f32::NAN,
            f32::NEG_INFINITY,
            -next_f32(K_ZERO_THRESHOLD_F32),
            -K_ZERO_THRESHOLD_F32,
            previous_f32(-K_ZERO_THRESHOLD_F32),
            -0.0,
            0.0,
            previous_f32(K_ZERO_THRESHOLD_F32),
            K_ZERO_THRESHOLD_F32,
            next_f32(K_ZERO_THRESHOLD_F32),
            0.5,
            next_f32(0.5),
            f32::INFINITY,
        ];
        for decision_type in [0, 2, 4, 6, 8, 10] {
            let model_text = TOY_MODEL.replacen(
                "decision_type=10",
                &format!("decision_type={decision_type}"),
                1,
            );
            let model = parse_model(&model_text).unwrap();
            let rows_f32 = values
                .iter()
                .flat_map(|value| [*value, 0.2_f32])
                .collect::<Vec<_>>();
            let rows_f64 = rows_f32
                .iter()
                .map(|value| f64::from(*value))
                .collect::<Vec<_>>();
            assert_eq!(
                predict_rows_f32(&model, &rows_f32, 2, 1, false),
                predict_rows(&model, &rows_f64, 2, 1, false),
                "decision_type={decision_type}",
            );
        }
    }

    #[test]
    fn tiled_scoring_preserves_every_score_bit() {
        let values = [
            f32::NAN,
            f32::from_bits(0xffc00001),
            f32::NEG_INFINITY,
            -next_f32(K_ZERO_THRESHOLD_F32),
            -K_ZERO_THRESHOLD_F32,
            -0.0,
            0.0,
            K_ZERO_THRESHOLD_F32,
            next_f32(K_ZERO_THRESHOLD_F32),
            previous_f32(0.5),
            0.5,
            next_f32(0.5),
            f32::INFINITY,
        ];
        for decision_type in [0, 2, 4, 6, 8, 10] {
            let text = TOY_MODEL.replacen(
                "decision_type=10",
                &format!("decision_type={decision_type}"),
                1,
            );
            let mut model = parse_model(&text).unwrap();
            // Cancellation makes any reassociation of tree sums observable.
            let original_trees = model.trees.clone();
            for magnitude in [1e20, 0.1, -1e20, -0.0, 0.3] {
                let mut tree = original_trees[0].clone();
                tree.leaf_value = vec![magnitude, -magnitude];
                model.trees.push(tree);
            }
            for row_count in [0, 1, 63, 64, 65, 127, 128, 129, 1001] {
                let rows: Vec<f32> = (0..row_count)
                    .flat_map(|index| [values[index % values.len()], 0.2])
                    .collect();
                let rows_f64: Vec<f64> = rows.iter().map(|value| f64::from(*value)).collect();
                for apply_sigmoid in [false, true] {
                    let expected: Vec<u64> = rows
                        .chunks_exact(2)
                        .map(|row| {
                            let raw = model.predict_row_raw_f32(row);
                            if apply_sigmoid {
                                model.raw_to_probability(raw).to_bits()
                            } else {
                                raw.to_bits()
                            }
                        })
                        .collect();
                    for threads in [1, 10] {
                        let actual: Vec<u64> =
                            predict_rows_f32(&model, &rows, 2, threads, apply_sigmoid)
                                .iter()
                                .map(|value| value.to_bits())
                                .collect();
                        assert_eq!(actual, expected,
                            "decision={decision_type}, rows={row_count}, threads={threads}, sigmoid={apply_sigmoid}");
                        let actual_f64: Vec<u64> =
                            predict_rows(&model, &rows_f64, 2, threads, apply_sigmoid)
                                .iter()
                                .map(|value| value.to_bits())
                                .collect();
                        assert_eq!(actual_f64, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn tree_magic_after_header_fields_rejected() {
        let corrupted = format!("objective=binary sigmoid:0.5\n{TOY_MODEL}");
        let err = parse_model(&corrupted).unwrap_err();
        assert!(err.contains("does not start with the LightGBM 'tree' magic line"));
    }

    #[test]
    fn self_referential_child_rejected() {
        let corrupted = TOY_MODEL.replacen("left_child=-1", "left_child=0", 1);
        let err = parse_model(&corrupted).unwrap_err();
        assert!(err.contains("must refer to a later internal node"));
    }

    #[test]
    fn categorical_split_rejected() {
        let categorical = TOY_MODEL.replace("decision_type=10", "decision_type=11");
        let err = parse_model(&categorical).unwrap_err();
        assert!(err.to_string().contains("categorical"));
    }

    #[test]
    fn feature_count_above_compact_index_capacity_rejected() {
        let too_wide = TOY_MODEL.replace("max_feature_idx=1", "max_feature_idx=65536");
        let err = parse_model(&too_wide).unwrap_err();
        assert!(err.contains("more than 65536 numerical features"));
    }

    #[test]
    fn overflowing_feature_count_rejected() {
        let too_wide =
            TOY_MODEL.replace("max_feature_idx=1", "max_feature_idx=9223372036854775807");
        let err = parse_model(&too_wide).unwrap_err();
        assert!(err.contains("max_feature_idx=9223372036854775807 is too large"));
    }

    #[test]
    fn missing_end_of_trees_rejected() {
        let truncated = TOY_MODEL.replace("end of trees\n", "");
        let err = parse_model(&truncated).unwrap_err();
        assert!(err.to_string().contains("end of trees"));
    }

    #[test]
    fn tree_sizes_count_mismatch_rejected() {
        let mismatched = TOY_MODEL.replace("tree_sizes=1 1", "tree_sizes=1");
        let err = parse_model(&mismatched).unwrap_err();
        assert!(err.to_string().contains("tree_sizes"));
    }

    #[test]
    fn linear_tree_rejected() {
        let linear = TOY_MODEL.replacen("is_linear=0", "is_linear=1", 1);
        let err = parse_model(&linear).unwrap_err();
        assert!(err.to_string().contains("linear"));
    }
}
