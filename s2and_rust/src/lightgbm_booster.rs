//! Pure-Rust evaluator for LightGBM text models (`.lgb`).
//!
//! Scope is exactly the subset the S2AND production boosters use: single-model
//! binary-objective GBDTs with numerical splits and None/Zero/NaN missing-value
//! handling. Categorical splits, linear trees, multiclass models, and
//! random-forest averaging are rejected at load time so an unsupported model
//! fails loudly instead of scoring silently wrong.
//!
//! Decision semantics mirror LightGBM's `Tree::NumericalDecision`
//! (include/LightGBM/tree.h): NaN is converted to 0.0 unless the split's
//! missing type is NaN; Zero/NaN missing values take the default_left branch;
//! otherwise `fval <= threshold` goes left. Raw scores are sums of leaf values
//! in tree order (bit-identical to LightGBM's sequential accumulation), and
//! binary probabilities apply `1 / (1 + exp(-sigmoid * raw))`.

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

#[inline]
fn f32_comparison_floor(threshold: f64) -> f32 {
    let rounded = threshold as f32;
    if rounded.is_nan() || f64::from(rounded) <= threshold {
        rounded
    } else {
        previous_f32(rounded)
    }
}

#[derive(Debug, Clone)]
struct LgbTree {
    num_leaves: usize,
    // S2AND boosters have tens of features. Keeping indices at u16 width
    // offsets half of the exact f32 threshold cache, so the retained model
    // stays below the 10% memory-growth budget without slowing traversal.
    split_feature: Vec<u16>,
    threshold: Vec<f64>,
    // Largest f32 whose lossless f64 widening is <= the model threshold.
    // This makes the f32 scorer comparison exact without widening every split
    // read in the hot loop.
    threshold_f32_floor: Box<[f32]>,
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
            let mut fval = row[self.split_feature[node] as usize];
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
                    *self.threshold_f32_floor.get_unchecked(node),
                    *self.left_child.get_unchecked(node),
                    *self.right_child.get_unchecked(node),
                )
            };
            let missing = missing_type(decision_type);
            // The batch boundary asserts row width against the same model
            // feature count used to validate split_feature above.
            let mut fval = unsafe { *row.get_unchecked(feature) };
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
            } else if fval <= threshold {
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
    #[inline]
    fn predict_row_raw(&self, row: &[f64]) -> f64 {
        let mut raw = 0.0f64;
        for tree in &self.trees {
            raw += tree.predict(row);
        }
        raw
    }

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
            threshold_f32_floor: Box::default(),
            decision_type: Vec::new(),
            left_child: Vec::new(),
            right_child: Vec::new(),
            leaf_value,
        });
    }

    let internal_count = num_leaves - 1;
    let split_feature: Vec<u16> = parse_vec(fields, "split_feature")?;
    let threshold: Vec<f64> = parse_vec(fields, "threshold")?;
    let threshold_f32_floor = threshold
        .iter()
        .map(|value| f32_comparison_floor(*value))
        .collect::<Vec<_>>()
        .into_boxed_slice();
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
        threshold_f32_floor,
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
    let score_chunk = |chunk: &[f64]| {
        let raw = model.predict_row_raw(chunk);
        if apply_sigmoid {
            model.raw_to_probability(raw)
        } else {
            raw
        }
    };
    if num_threads > 1 && row_count > 1 {
        install_with_optional_rayon_pool(Some(num_threads), || {
            rows.par_chunks(num_features).map(score_chunk).collect()
        })
    } else {
        rows.chunks(num_features).map(score_chunk).collect()
    }
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
    let row_count = rows.len() / num_features;
    let score_chunk = |chunk: &[f32]| {
        let raw = model.predict_row_raw_f32(chunk);
        if apply_sigmoid {
            model.raw_to_probability(raw)
        } else {
            raw
        }
    };
    if num_threads > 1 && row_count > 1 {
        install_with_optional_rayon_pool(Some(num_threads), || {
            rows.par_chunks(num_features).map(score_chunk).collect()
        })
    } else {
        rows.chunks(num_features).map(score_chunk).collect()
    }
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
        // Keep the owned GIL-independent copy at the caller's float32 width;
        // precomputed exact f32 comparison floors avoid widening in traversal.
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
    fn float32_threshold_floor_matches_lossless_widening() {
        let anchors = [
            f32::NEG_INFINITY,
            -f32::MAX,
            -1.0,
            -f32::from_bits(1),
            -0.0,
            0.0,
            f32::from_bits(1),
            0.1,
            1.0,
            f32::MAX,
            f32::INFINITY,
        ];
        let mut thresholds = vec![
            f64::NEG_INFINITY,
            -(f32::MAX as f64) * 2.0,
            -0.1,
            -f64::from(f32::from_bits(1)) / 2.0,
            -0.0,
            0.0,
            f64::from(f32::from_bits(1)) / 2.0,
            0.1,
            (f64::from(1.0_f32) + f64::from(next_f32(1.0))) / 2.0,
            (f32::MAX as f64) * 2.0,
            f64::INFINITY,
            f64::NAN,
        ];
        thresholds.extend(anchors.iter().copied().map(f64::from));

        for threshold in thresholds {
            let floor = f32_comparison_floor(threshold);
            let rounded = threshold as f32;
            let candidates = [
                f32::NEG_INFINITY,
                previous_f32(rounded),
                rounded,
                next_f32(rounded),
                -0.0,
                0.0,
                f32::INFINITY,
                f32::NAN,
            ];
            for candidate in candidates {
                assert_eq!(
                    f64::from(candidate) <= threshold,
                    candidate <= floor,
                    "candidate={candidate:?} threshold={threshold:?} floor={floor:?}",
                );
            }
        }
    }

    #[test]
    fn float32_threshold_floor_random_bit_patterns_match_lossless_widening() {
        let mut state = 0x8fd5_5a2d_d1b5_4a32_u64;
        for _ in 0..100_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let threshold = f64::from_bits(state);
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let arbitrary = f32::from_bits(state as u32);
            let floor = f32_comparison_floor(threshold);
            let rounded = threshold as f32;
            for candidate in [arbitrary, previous_f32(rounded), rounded, next_f32(rounded)] {
                assert_eq!(
                    f64::from(candidate) <= threshold,
                    candidate <= floor,
                    "candidate={candidate:?} threshold={threshold:?} floor={floor:?}",
                );
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
