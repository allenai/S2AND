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

const CATEGORICAL_MASK: u8 = 1;
const DEFAULT_LEFT_MASK: u8 = 2;

const MISSING_TYPE_NONE: u8 = 0;
const MISSING_TYPE_ZERO: u8 = 1;
const MISSING_TYPE_NAN: u8 = 2;

#[inline]
fn missing_type(decision_type: u8) -> u8 {
    (decision_type >> 2) & 3
}

#[inline]
fn is_zero(fval: f64) -> bool {
    (-K_ZERO_THRESHOLD..=K_ZERO_THRESHOLD).contains(&fval)
}

#[derive(Debug, Clone)]
struct LgbTree {
    num_leaves: usize,
    split_feature: Vec<u32>,
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
    let split_feature: Vec<u32> = parse_vec(fields, "split_feature")?;
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
    let num_features = (max_feature_idx + 1) as usize;

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
    let row_count = rows.len() / num_features.max(1);
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

/// Pure-Rust scorer for S2AND's native LightGBM binary classifiers.
#[pyclass]
pub(crate) struct RustLightGBMBooster {
    model: LgbModel,
}

impl RustLightGBMBooster {
    fn predict_array<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        num_threads: Option<usize>,
        apply_sigmoid: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let shape = features.shape();
        let (row_count, column_count) = (shape[0], shape[1]);
        if column_count != self.model.num_features {
            return Err(model_error(format!(
                "features must have {} columns, got {column_count}",
                self.model.num_features
            )));
        }
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
        self.predict_array(py, features, num_threads, false)
    }

    /// Positive-class probabilities, matching `lgb.Booster.predict(...)`.
    #[pyo3(signature = (features, num_threads=None))]
    fn predict_proba_positive<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_array(py, features, num_threads, true)
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
