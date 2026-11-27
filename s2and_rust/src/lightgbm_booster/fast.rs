//! Guarded, bit-exact fast traversal plans for parsed LightGBM models.
//!
//! [`FastPlan::build`] runs once per model, after parsing. It returns `None`,
//! leaving the parent module's baseline traversal to score the model
//! unchanged, unless all of these hold:
//! - every tree is a proper binary tree (see [`is_proper_tree`]);
//! - every split has None or NaN missing handling and a non-NaN threshold;
//! - the model has at most [`MAX_FAST_FEATURES`] features;
//! - each feature has at most [`MAX_KEY_THRESHOLDS`] distinct thresholds;
//! - every tree has at most [`MAX_LOCKSTEP_LEAVES`] leaves.
//!
//! Dispatch is per model, never per tree. A single constant or stump uses the
//! baseline; otherwise QuickScorer handles models whose trees all have at most
//! [`MAX_QUICKSCORER_LEAVES`] leaves, and larger trees use lockstep traversal.
//!
//! Why the scores are bit-identical to the baseline:
//! - **Bin keys.** Each feature's distinct thresholds form a strictly
//!   increasing list `T` (`==` merges `-0.0` and `0.0`, which every
//!   comparison treats alike). For a non-NaN value `v`, the bin
//!   `b = #{j : T[j] < v}` satisfies `v <= T[j]` exactly when `b <= j`, so
//!   one integer compare reproduces every split on that feature.
//! - **Missing values.** After the same dense-zero normalization as the
//!   baseline, a NaN takes one static direction per split: NaN-missing splits
//!   use `default_left`, and None-missing splits see `0.0` and go left when
//!   `0.0 <= threshold`. Zero-missing splits also route every zero-window value
//!   by `default_left`, so those models stay on the baseline.
//! - **Two keys per feature.** Splits whose NaN goes left read the feature's
//!   NaN-left key (NaN becomes `0`, which goes left at every threshold index);
//!   splits whose NaN goes right read its NaN-right key (NaN becomes `255`,
//!   which goes right at every index `j <= 254`). Non-NaN values use `b` in
//!   both keys. With at most 255 thresholds, `b <= 255` fits a `u8`. When `b`
//!   is 255, `v` exceeds every threshold and routes like NaN on NaN-right
//!   splits.
//! - **Sums.** Each row adds its leaf values in tree order to an `f64` that
//!   starts at `0.0`, exactly as the baseline does. Lockstep walks also add
//!   `-0.0` at every internal node they leave; `x + -0.0` is `x` for every
//!   `x`, signed zeros included. The parent applies the same sigmoid
//!   afterwards.
//!
//! `f32` rows widen losslessly to `f64` before normalization; this is
//! equivalent to the baseline's `f32` normalization followed by widening,
//! because widening is exact and monotone.

use rayon::prelude::*;

use super::{
    missing_type, normalize_dense_zero, LgbModel, LgbTree, DEFAULT_LEFT_MASK, MISSING_TYPE_NAN,
    MISSING_TYPE_NONE,
};
use crate::rayon_pool::install_with_optional_rayon_pool;

/// Widest model the fast plans accept: QuickScorer's fixed per-lane key
/// buffers hold `2 * 256` keys. (Lockstep's 16-bit key offsets would allow
/// 4096 features; one limit keeps dispatch simple.)
const MAX_FAST_FEATURES: usize = 256;
/// Most distinct thresholds per feature that `u8` keys separate exactly.
///
/// LightGBM's default `max_bin=255` yields at most 255 bin upper bounds per
/// feature, including `+inf`, so default-trained models always fit.
const MAX_KEY_THRESHOLDS: usize = 255;
const MAX_QUICKSCORER_LEAVES: usize = 32;
const MAX_LOCKSTEP_LEAVES: usize = 256;
/// Rows per parallel work item, for both kernels.
pub(super) const TILE_ROWS: usize = 256;
/// Rows traversed together by the lockstep kernel.
const LOCKSTEP_LANES: usize = 8;
/// Rows scored per pass over the QuickScorer node lists.
const QUICKSCORER_ROWS: usize = 16;
/// Distinct values of a `u8` key; the QuickScorer prefix table stride.
const KEY_VALUES: usize = 256;

const NAN_LEFT_KEY: u8 = 0;
const NAN_RIGHT_KEY: u8 = u8::MAX;
/// Lockstep leaf pseudo-nodes compare against this, so no key moves them.
const LEAF_HI: u8 = u8::MAX;

/// A dense input element type the fast kernels accept.
pub(super) trait FeatureValue: Copy + Send + Sync {
    fn widen(self) -> f64;
}

impl FeatureValue for f64 {
    #[inline(always)]
    fn widen(self) -> f64 {
        self
    }
}

impl FeatureValue for f32 {
    #[inline(always)]
    fn widen(self) -> f64 {
        f64::from(self)
    }
}

/// Exact fast scorer for one model.
#[derive(Debug, Clone)]
pub(super) enum FastPlan {
    QuickScorer(QuickScorer),
    Lockstep(Lockstep),
}

impl FastPlan {
    /// Builds the fast plan for `model`, or `None` when the baseline must score it.
    pub(super) fn build(model: &LgbModel) -> Option<Self> {
        if model.num_features > MAX_FAST_FEATURES {
            return None;
        }
        let mut max_leaves = 1;
        for tree in &model.trees {
            let splits_supported = tree.decision_type.iter().zip(&tree.threshold).all(
                |(&decision_type, &threshold)| {
                    !threshold.is_nan() && nan_goes_left(decision_type, threshold).is_some()
                },
            );
            if !splits_supported || !is_proper_tree(tree) {
                return None;
            }
            max_leaves = max_leaves.max(tree.num_leaves);
        }
        // A single constant or stump needs at most one split per row. Building
        // keys and kernel state costs more than the direct baseline traversal.
        if max_leaves > MAX_LOCKSTEP_LEAVES || (model.trees.len() == 1 && max_leaves <= 2) {
            return None;
        }
        let keys = KeyTable::build(model)?;
        if max_leaves <= MAX_QUICKSCORER_LEAVES {
            QuickScorer::build(model, keys).map(Self::QuickScorer)
        } else {
            Lockstep::build(model, keys).map(Self::Lockstep)
        }
    }

    /// Raw margins from the portable kernels, skipping any CPU-specific
    /// dispatch, so tests cover the code other CPUs run.
    #[cfg(test)]
    pub(super) fn portable_margins<F: FeatureValue>(
        &self,
        model: &LgbModel,
        rows: &[F],
    ) -> Vec<f64> {
        let num_features = model.num_features;
        let mut margins = vec![0.0; rows.len() / num_features];
        for (tile, scores) in rows
            .chunks(num_features * TILE_ROWS)
            .zip(margins.chunks_mut(TILE_ROWS))
        {
            match self {
                Self::QuickScorer(kernel) => kernel.score_tile_generic(tile, scores),
                Self::Lockstep(kernel) => kernel.score_tile(tile, scores),
            }
        }
        margins
    }

    #[cfg(test)]
    pub(super) fn kernel_name(&self) -> &'static str {
        match self {
            Self::QuickScorer(_) => "quickscorer",
            Self::Lockstep(_) => "lockstep",
        }
    }

    /// Scores rectangular rows; mirrors the baseline `predict_rows` contract.
    pub(super) fn predict_rows<F: FeatureValue>(
        &self,
        model: &LgbModel,
        rows: &[F],
        num_threads: usize,
        apply_sigmoid: bool,
    ) -> Vec<f64> {
        // Keep enough work items for mid-size parallel batches without changing
        // the 256-row tiles used for larger batches or single-threaded calls.
        let row_count = rows.len() / model.num_features;
        let tile_rows = if num_threads > 1 {
            row_count
                .div_ceil(num_threads)
                .next_multiple_of(QUICKSCORER_ROWS)
                .clamp(QUICKSCORER_ROWS, TILE_ROWS)
        } else {
            TILE_ROWS
        };
        self.predict_rows_tiled(model, rows, num_threads, apply_sigmoid, tile_rows)
    }

    /// Single-threaded probabilities (or raw margins) of `rows`, with rows
    /// that share a pattern under `spec` scored on pruned trees. Equal to
    /// `predict_rows_tiled` bit for bit; QuickScorer plans ignore `spec`.
    pub(super) fn predict_rows_specialized<F: FeatureValue>(
        &self,
        model: &LgbModel,
        rows: &[F],
        apply_sigmoid: bool,
        spec: &RowSpecialization,
    ) -> Vec<f64> {
        match self {
            Self::QuickScorer(_) => {
                self.predict_rows_tiled(model, rows, 1, apply_sigmoid, TILE_ROWS)
            }
            Self::Lockstep(kernel) => {
                let mut margins = vec![0.0; rows.len() / model.num_features];
                kernel.score_rows_specialized(rows, &mut margins, spec);
                if apply_sigmoid {
                    for raw in &mut margins {
                        *raw = model.raw_to_probability(*raw);
                    }
                }
                margins
            }
        }
    }

    pub(super) fn predict_rows_tiled<F: FeatureValue>(
        &self,
        model: &LgbModel,
        rows: &[F],
        num_threads: usize,
        apply_sigmoid: bool,
        tile_rows: usize,
    ) -> Vec<f64> {
        let num_features = model.num_features;
        debug_assert_eq!(rows.len() % num_features, 0);
        let row_count = rows.len() / num_features;
        let mut scores = vec![0.0; row_count];
        let score_tile = |(tile, margins): (&[F], &mut [f64])| {
            match self {
                Self::QuickScorer(kernel) => kernel.score_tile(tile, margins),
                Self::Lockstep(kernel) => kernel.score_tile(tile, margins),
            }
            if apply_sigmoid {
                for raw in margins {
                    *raw = model.raw_to_probability(*raw);
                }
            }
        };
        if num_threads > 1 && row_count > 1 {
            install_with_optional_rayon_pool(Some(num_threads), || {
                rows.par_chunks(num_features * tile_rows)
                    .zip(scores.par_chunks_mut(tile_rows))
                    .for_each(score_tile);
            });
        } else {
            rows.chunks(num_features * tile_rows)
                .zip(scores.chunks_mut(tile_rows))
                .for_each(score_tile);
        }
        scores
    }
}

/// Static direction of a NaN input at a split; `None` for Zero-missing splits.
///
/// Mirrors the baseline: NaN stays NaN only for NaN-missing splits, where it
/// takes the default direction; otherwise it becomes `0.0` and, for
/// None-missing splits, compares against the threshold.
fn nan_goes_left(decision_type: u8, threshold: f64) -> Option<bool> {
    match missing_type(decision_type) {
        MISSING_TYPE_NAN => Some(decision_type & DEFAULT_LEFT_MASK != 0),
        MISSING_TYPE_NONE => Some(0.0 <= threshold),
        _ => None,
    }
}

/// True when the parsed child arrays form a proper binary tree.
///
/// The parser only proves that children are in range and point to later
/// internal nodes, so it accepts shared children. This requires
/// `internal_count == num_leaves - 1`, no parent for the root, and exactly one
/// parent for every other internal node and every leaf. QuickScorer's in-order
/// leaf masks and the lockstep layout are only valid for such trees.
pub(super) fn is_proper_tree(tree: &LgbTree) -> bool {
    let internal_count = tree.left_child.len();
    if internal_count + 1 != tree.num_leaves {
        return false;
    }
    if internal_count == 0 {
        return true;
    }
    let mut internal_parents = vec![0u32; internal_count];
    let mut leaf_parents = vec![0u32; tree.num_leaves];
    for &child in tree.left_child.iter().chain(&tree.right_child) {
        if child < 0 {
            leaf_parents[(!child) as usize] += 1;
        } else {
            internal_parents[child as usize] += 1;
        }
    }
    internal_parents[0] == 0
        && internal_parents[1..].iter().all(|&parents| parents == 1)
        && leaf_parents.iter().all(|&parents| parents == 1)
}

/// Per-feature sorted thresholds and the row-to-key transform.
#[derive(Debug, Clone)]
struct KeyTable {
    num_features: usize,
    /// Original feature indices that appear in at least one split.
    used_features: Vec<usize>,
    /// Per feature: strictly increasing, non-NaN, at most `MAX_KEY_THRESHOLDS`.
    thresholds: Vec<Vec<f64>>,
}

impl KeyTable {
    /// Collects thresholds; `None` if a feature exceeds the key capacity.
    /// Callers have already rejected NaN thresholds.
    fn build(model: &LgbModel) -> Option<Self> {
        let mut thresholds = vec![Vec::new(); model.num_features];
        for tree in &model.trees {
            for (&feature, &threshold) in tree.split_feature.iter().zip(&tree.threshold) {
                thresholds[usize::from(feature)].push(threshold);
            }
        }
        for list in &mut thresholds {
            // total_cmp places -0.0 next to 0.0, so dedup by == merges them.
            list.sort_by(f64::total_cmp);
            list.dedup_by(|next, kept| next == kept);
            if list.len() > MAX_KEY_THRESHOLDS {
                return None;
            }
        }
        let used_features = thresholds
            .iter()
            .enumerate()
            .filter_map(|(feature, list)| (!list.is_empty()).then_some(feature))
            .collect();
        Some(Self {
            num_features: model.num_features,
            used_features,
            thresholds,
        })
    }

    /// The key slot and compare bound of internal `node`: the row goes left
    /// exactly when its key in that slot is `<= hi`.
    ///
    /// Slots `0..num_features` are NaN-left keys; slots
    /// `num_features..2 * num_features` are NaN-right keys.
    fn encode_split(&self, tree: &LgbTree, node: usize) -> Option<(usize, u8)> {
        let feature = usize::from(tree.split_feature[node]);
        let threshold = tree.threshold[node];
        let list = &self.thresholds[feature];
        let index = list.partition_point(|value| *value < threshold);
        // The threshold is in the list, and index <= 254 keeps 255 free.
        debug_assert!(list.get(index) == Some(&threshold));
        let hi = u8::try_from(index).ok().filter(|hi| *hi < NAN_RIGHT_KEY)?;
        let slot = if nan_goes_left(tree.decision_type[node], threshold)? {
            feature
        } else {
            self.num_features + feature
        };
        Some((slot, hi))
    }

    /// The (NaN-left, NaN-right) keys of one raw feature value.
    #[inline]
    fn keys(&self, feature: usize, value: f64) -> (u8, u8) {
        let value = normalize_dense_zero(value);
        if value.is_nan() {
            return (NAN_LEFT_KEY, NAN_RIGHT_KEY);
        }
        // At most MAX_KEY_THRESHOLDS = 255 entries, so the count fits.
        let bin = self.thresholds[feature].partition_point(|threshold| *threshold < value) as u8;
        (bin, bin)
    }
}

/// QuickScorer over `u8` keys for models whose trees all have at most 32 leaves.
///
/// Leaves are numbered left to right. Each internal node carries the mask
/// `!(bits of the leaves in its left subtree)`. A row's exit leaf is the lowest
/// set bit of the AND of the masks of every node that sends the row right. In
/// a proper tree, a leaf lies in a node's left subtree only if the node is an
/// ancestor where the leaf's path turns left, so no such mask clears the exit
/// leaf. Every leaf left of the exit leaf is cleared by their lowest common
/// ancestor, where the exit path turns right. The exit leaf survives for any
/// key values, so each state word stays nonzero and its trailing-zero count
/// indexes one of the tree's leaf slots. Within one key slot, the nodes that
/// send key `k` right are those with `hi < k`, a prefix of the slot's list
/// sorted by `hi`.
#[derive(Debug, Clone)]
pub(super) struct QuickScorer {
    keys: KeyTable,
    /// `(tree, mask)` per internal node, grouped by key slot, sorted by `hi`.
    entries: Vec<(u32, u32)>,
    /// `entries` range of each key slot; `2 * num_features + 1` entries.
    slot_offsets: Vec<usize>,
    /// Key slots with nodes; unused features and unused NaN directions are skipped.
    active_slots: Vec<usize>,
    /// `slot * KEY_VALUES + key`: how many of the slot's entries send `key` right.
    right_prefix: Vec<u32>,
    /// In-order leaf values per tree.
    leaves: Vec<[f64; MAX_QUICKSCORER_LEAVES]>,
}

impl QuickScorer {
    fn build(model: &LgbModel, keys: KeyTable) -> Option<Self> {
        let slot_count = 2 * keys.num_features;
        let mut per_slot: Vec<Vec<(u8, u32, u32)>> = vec![Vec::new(); slot_count];
        let mut leaves = vec![[0.0; MAX_QUICKSCORER_LEAVES]; model.trees.len()];
        for ((tree_index, tree), tree_leaves) in model.trees.iter().enumerate().zip(&mut leaves) {
            let tree_id = u32::try_from(tree_index).ok()?;
            if tree.left_child.is_empty() {
                tree_leaves[0] = tree.leaf_value[0];
                continue;
            }
            let mut left_masks = vec![0u32; tree.left_child.len()];
            let mut next_position = 0;
            number_leaves_in_order(tree, 0, &mut next_position, tree_leaves, &mut left_masks);
            for (node, left_mask) in left_masks.iter().enumerate() {
                let (slot, hi) = keys.encode_split(tree, node)?;
                per_slot[slot].push((hi, tree_id, !left_mask));
            }
        }
        let mut entries = Vec::new();
        let mut slot_offsets = vec![0usize];
        let mut active_slots = Vec::new();
        let mut right_prefix = Vec::with_capacity(slot_count * KEY_VALUES);
        for (slot, mut list) in per_slot.into_iter().enumerate() {
            if !list.is_empty() {
                active_slots.push(slot);
            }
            list.sort_by_key(|&(hi, _, _)| hi);
            for key in 0..KEY_VALUES {
                let sent_right = list.partition_point(|&(hi, _, _)| usize::from(hi) < key);
                right_prefix.push(u32::try_from(sent_right).ok()?);
            }
            entries.extend(list.iter().map(|&(_, tree, mask)| (tree, mask)));
            slot_offsets.push(entries.len());
        }
        Some(Self {
            keys,
            entries,
            slot_offsets,
            active_slots,
            right_prefix,
            leaves,
        })
    }

    fn score_tile<F: FeatureValue>(&self, tile: &[F], margins: &mut [f64]) {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2.
            unsafe { self.score_tile_avx2(tile, margins) };
            return;
        }
        self.score_tile_generic(tile, margins);
    }

    /// `score_tile_generic` compiled with AVX2: wider mask updates, the same
    /// scalar `f64` additions in the same order.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn score_tile_avx2<F: FeatureValue>(&self, tile: &[F], margins: &mut [f64]) {
        self.score_tile_generic(tile, margins);
    }

    #[inline(always)]
    fn score_tile_generic<F: FeatureValue>(&self, tile: &[F], margins: &mut [f64]) {
        let num_features = self.keys.num_features;
        let mut state = vec![[u32::MAX; QUICKSCORER_ROWS]; self.leaves.len()];
        let mut lane_keys = [[0u8; 2 * MAX_FAST_FEATURES]; QUICKSCORER_ROWS];
        let full_rows = margins.len() / QUICKSCORER_ROWS * QUICKSCORER_ROWS;
        let (full_tile, tail_tile) = tile.split_at(full_rows * num_features);
        let (full_margins, tail_margins) = margins.split_at_mut(full_rows);
        for (rows, scores) in full_tile
            .chunks_exact(num_features * QUICKSCORER_ROWS)
            .zip(full_margins.chunks_exact_mut(QUICKSCORER_ROWS))
        {
            self.score_group::<F, true>(rows, scores, &mut state, &mut lane_keys);
        }
        if !tail_margins.is_empty() {
            self.score_group::<F, false>(tail_tile, tail_margins, &mut state, &mut lane_keys);
        }
    }

    /// Keep full groups statically sized for vectorization; tails do no work for padding.
    #[inline(always)]
    fn score_group<F: FeatureValue, const FULL: bool>(
        &self,
        rows: &[F],
        margins: &mut [f64],
        state: &mut [[u32; QUICKSCORER_ROWS]],
        lane_keys: &mut [[u8; 2 * MAX_FAST_FEATURES]; QUICKSCORER_ROWS],
    ) {
        let lanes = if FULL {
            QUICKSCORER_ROWS
        } else {
            margins.len()
        };
        let num_features = self.keys.num_features;
        for (row, keys) in rows.chunks_exact(num_features).zip(&mut lane_keys[..lanes]) {
            for &feature in &self.keys.used_features {
                let (nan_left, nan_right) = self.keys.keys(feature, row[feature].widen());
                keys[feature] = nan_left;
                keys[num_features + feature] = nan_right;
            }
        }
        if FULL {
            state.fill([u32::MAX; QUICKSCORER_ROWS]);
        } else {
            for words in state.iter_mut() {
                words[..lanes].fill(u32::MAX);
            }
        }
        for &slot in &self.active_slots {
            let entries = &self.entries[self.slot_offsets[slot]..self.slot_offsets[slot + 1]];
            let prefix = &self.right_prefix[slot * KEY_VALUES..][..KEY_VALUES];
            // u32 lane bounds keep the per-lane select below vectorizable.
            let mut ends = [0u32; QUICKSCORER_ROWS];
            for (end, keys) in ends[..lanes].iter_mut().zip(&lane_keys[..lanes]) {
                *end = prefix[usize::from(keys[slot])];
            }
            let shared = ends[..lanes].iter().copied().min().unwrap_or(0) as usize;
            let any = ends[..lanes].iter().copied().max().unwrap_or(0) as usize;
            for &(tree, mask) in &entries[..shared] {
                for word in &mut state[tree as usize][..lanes] {
                    *word &= mask;
                }
            }
            for (position, &(tree, mask)) in (shared as u32..).zip(&entries[shared..any]) {
                let words = &mut state[tree as usize];
                for (word, &end) in words[..lanes].iter_mut().zip(&ends[..lanes]) {
                    *word &= if position < end { mask } else { u32::MAX };
                }
            }
        }
        // One add chain per row, in tree order, starting at 0.0.
        let mut raw = [0.0f64; QUICKSCORER_ROWS];
        for (words, tree_leaves) in state.iter().zip(&self.leaves) {
            for (sum, word) in raw[..lanes].iter_mut().zip(words) {
                *sum += tree_leaves[word.trailing_zeros() as usize];
            }
        }
        margins.copy_from_slice(&raw[..lanes]);
    }
}

/// Assigns in-order positions to the leaves under the LightGBM child
/// reference `child` and records each internal node's left-subtree leaf bits.
/// Returns the subtree's leaf bits. A proper tree with at most 32 leaves
/// assigns each position below 32 exactly once.
fn number_leaves_in_order(
    tree: &LgbTree,
    child: i32,
    next_position: &mut usize,
    leaves: &mut [f64; MAX_QUICKSCORER_LEAVES],
    left_masks: &mut [u32],
) -> u32 {
    if child < 0 {
        let position = *next_position;
        *next_position += 1;
        leaves[position] = tree.leaf_value[(!child) as usize];
        return 1u32 << position;
    }
    let node = child as usize;
    let left = number_leaves_in_order(
        tree,
        tree.left_child[node],
        next_position,
        leaves,
        left_masks,
    );
    let right = number_leaves_in_order(
        tree,
        tree.right_child[node],
        next_position,
        leaves,
        left_masks,
    );
    left_masks[node] = left;
    left | right
}

/// Feature roles that let groups of rows share pruned trees.
///
/// Rows that agree on every value feature's keys, and on which NaN features
/// are NaN, take the same direction at every split on those features. Their
/// trees can drop those splits: each row still reaches the same leaf and adds
/// the same leaf values in tree order, so margins are bit-identical, while
/// paths get shorter.
///
/// It holds the pruned trees it builds, so they are freed with it at the end
/// of a scoring call rather than kept for the life of the model.
#[derive(Debug)]
pub(crate) struct RowSpecialization {
    /// Per model feature: its position among the value features.
    value_position: Vec<Option<u8>>,
    /// Per model feature: its bit among the NaN features.
    nan_bit: Vec<Option<u8>>,
    value_features: Vec<usize>,
    nan_features: Vec<usize>,
    /// Address of the booster this specialization belongs to; its pruned
    /// trees are only valid for that model.
    pub(super) owner: usize,
    /// Trees pruned for the patterns seen so far.
    pruned: std::sync::RwLock<PrunedCache>,
}

/// Pruned trees by pattern, with their total node count.
#[derive(Debug, Default)]
struct PrunedCache {
    by_pattern: std::collections::HashMap<PatternKey, std::sync::Arc<ChainedTrees>>,
    nodes: usize,
}

/// Most value features one pattern fixes: two key bytes each fill 32 bytes.
const MAX_VALUE_FEATURES: usize = 16;
/// Most NaN features one pattern fixes: one bit each.
const MAX_NAN_FEATURES: usize = 64;
/// Smallest row group that gets pruned trees; smaller groups use the full
/// trees together, so their lanes stay filled and few patterns are built.
const MIN_PRUNED_GROUP: usize = 8;
/// Most pruned-tree nodes one specialization keeps (16 bytes each with
/// their values, 8 MiB in all); it clears them when full, which bounds the
/// memory pruning adds.
const MAX_PRUNED_NODES: usize = 1 << 19;

impl RowSpecialization {
    /// `None` when the lists exceed the pattern capacity or name a feature
    /// outside the model.
    pub(crate) fn new(
        owner: usize,
        num_features: usize,
        value_features: Vec<usize>,
        nan_features: Vec<usize>,
    ) -> Option<Self> {
        if value_features.len() > MAX_VALUE_FEATURES
            || nan_features.len() > MAX_NAN_FEATURES
            || value_features
                .iter()
                .chain(&nan_features)
                .any(|feature| *feature >= num_features)
        {
            return None;
        }
        let mut value_position = vec![None; num_features];
        for (position, feature) in value_features.iter().enumerate() {
            value_position[*feature] = Some(position as u8);
        }
        let mut nan_bit = vec![None; num_features];
        for (bit, feature) in nan_features.iter().enumerate() {
            // A value feature already fixes its splits.
            if value_position[*feature].is_none() {
                nan_bit[*feature] = Some(bit as u8);
            }
        }
        Some(Self {
            value_position,
            nan_bit,
            value_features,
            nan_features,
            owner,
            pruned: std::sync::RwLock::new(PrunedCache::default()),
        })
    }
}

/// One row pattern: its value-feature keys and NaN bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct PatternKey {
    values: [u8; 2 * MAX_VALUE_FEATURES],
    nan: u64,
}

/// Trees laid out for chained walks: each lane walks its own row through a
/// chunk of consecutive trees without waiting for the other lanes.
///
/// Nodes keep the breadth-first word layout of [`Lockstep`], with two changes:
/// a leaf word points to the root of the next tree in its chunk (bits 32..64)
/// and reads key offset 0, and each chunk ends with a sentinel that points to
/// itself. A lane adds `values[index]` at every node it leaves: the leaf
/// value at leaves, and `-0.0`, the exact additive identity, elsewhere. So
/// each row still adds its leaf values in tree order.
#[derive(Debug, Clone)]
struct ChainedTrees {
    words: Vec<u64>,
    values: Vec<f64>,
    /// First node of each chunk; the last entry is one past the final sentinel.
    chunk_starts: Vec<usize>,
}

/// Trees walked per chunk; a chunk's nodes stay hot in L1 across a tile.
/// 16 beat 4, 32, 64 and whole models on the production nameless booster.
const CHAIN_CHUNK_TREES: usize = 16;

/// Lockstep traversal over `u8` keys for trees with at most 256 leaves.
///
/// Trees are renumbered breadth first, so each internal node's children are
/// adjacent and later, and leaves become self-looping pseudo-nodes. These
/// `nodes` are the source for pruning; scoring walks the [`ChainedTrees`]
/// built from them, where each lane moves on to its next tree as soon as it
/// reaches a leaf instead of waiting for the deepest lane. Each node is one
/// `u64`:
/// - bits 0..8: `hi`; the row moves to the right child when its key is
///   greater. Internal nodes have `hi <= 254`; leaves have [`LEAF_HI`].
/// - bits 8..16: leaf index within the tree (leaves only).
/// - bits 16..32: key offset, `slot * LOCKSTEP_LANES`, within a group's keys.
/// - bits 32..64: index of the left child; the right child follows it.
///   Leaves store their own index.
///
/// Rows are transposed into groups of [`LOCKSTEP_LANES`], laid out as
/// `[slot][lane]`, and each group walks a chunk of trees until every lane
/// reaches the chunk's end.
#[derive(Debug, Clone)]
pub(super) struct Lockstep {
    keys: KeyTable,
    nodes: Vec<u64>,
    leaves: Vec<f64>,
    /// `(first node, first leaf)` per tree, in model order.
    trees: Vec<(usize, usize)>,
    /// The full trees in chained layout; `nodes` stays the pruning source.
    chained: ChainedTrees,
}

impl Lockstep {
    fn build(model: &LgbModel, keys: KeyTable) -> Option<Self> {
        let mut nodes = Vec::new();
        let mut leaves = Vec::new();
        let mut trees = Vec::with_capacity(model.trees.len());
        for tree in &model.trees {
            let first_node = nodes.len();
            trees.push((first_node, leaves.len()));
            leaves.extend_from_slice(&tree.leaf_value);
            // Breadth-first queue of LightGBM child references; `!0` is leaf 0.
            let mut queue = vec![if tree.left_child.is_empty() { !0 } else { 0 }];
            let mut position = 0;
            while position < queue.len() {
                let child = queue[position];
                let index = u64::from(u32::try_from(first_node + position).ok()?);
                let word = if child < 0 {
                    let leaf = u64::from(u8::try_from(!child).ok()?);
                    u64::from(LEAF_HI) | (leaf << 8) | (index << 32)
                } else {
                    let node = child as usize;
                    let (slot, hi) = keys.encode_split(tree, node)?;
                    let key_offset = u64::from(u16::try_from(slot * LOCKSTEP_LANES).ok()?);
                    let left = u64::from(u32::try_from(first_node + queue.len()).ok()?);
                    queue.push(tree.left_child[node]);
                    queue.push(tree.right_child[node]);
                    u64::from(hi) | (key_offset << 16) | (left << 32)
                };
                nodes.push(word);
                position += 1;
            }
        }
        let mut plan = Self {
            keys,
            nodes,
            leaves,
            trees,
            chained: ChainedTrees {
                words: Vec::new(),
                values: Vec::new(),
                chunk_starts: Vec::new(),
            },
        };
        if !plan.invariants_hold() {
            return None;
        }
        let starts: Vec<usize> = plan.trees.iter().map(|&(first, _)| first).collect();
        plan.chained = plan.chain(&plan.nodes, &starts)?;
        Some(plan)
    }

    /// Relays out breadth-first trees (the plan's own, or pruned copies whose
    /// tree `t` occupies `starts[t]..starts[t + 1]`) for chained walks, or
    /// `None` if the result breaks an invariant the walk relies on.
    fn chain(&self, nodes: &[u64], starts: &[usize]) -> Option<ChainedTrees> {
        let tree_count = starts.len();
        let tree_end = |tree: usize| starts.get(tree + 1).copied().unwrap_or(nodes.len());
        let chunk_count = tree_count.div_ceil(CHAIN_CHUNK_TREES);
        let mut words = Vec::with_capacity(nodes.len() + chunk_count);
        let mut values = Vec::with_capacity(nodes.len() + chunk_count);
        let mut chunk_starts = Vec::with_capacity(chunk_count + 1);
        for chunk_first in (0..tree_count).step_by(CHAIN_CHUNK_TREES) {
            let chunk_end = (chunk_first + CHAIN_CHUNK_TREES).min(tree_count);
            let old_base = starts[chunk_first];
            let new_base = words.len();
            chunk_starts.push(new_base);
            let renumber = |old: usize| -> u64 { (old - old_base + new_base) as u64 };
            for tree in chunk_first..chunk_end {
                // The next tree's root, or this chunk's sentinel after the last tree.
                let next_root = renumber(tree_end(tree));
                let first_leaf = self.trees[tree].1;
                for &word in &nodes[starts[tree]..tree_end(tree)] {
                    if word as u8 == LEAF_HI {
                        let leaf = usize::from((word >> 8) as u8);
                        words.push(u64::from(LEAF_HI) | (next_root << 32));
                        values.push(self.leaves[first_leaf + leaf]);
                    } else {
                        words.push((word & 0xFFFF_FFFF) | (renumber((word >> 32) as usize) << 32));
                        values.push(-0.0);
                    }
                }
            }
            let sentinel = words.len() as u64;
            words.push(u64::from(LEAF_HI) | (sentinel << 32));
            values.push(-0.0);
        }
        chunk_starts.push(words.len());
        let chained = ChainedTrees {
            words,
            values,
            chunk_starts,
        };
        self.chain_invariants_hold(&chained).then_some(chained)
    }

    /// Checks every invariant the chained walk's unchecked indexing relies on:
    /// within each chunk, every node but the sentinel points forward to a
    /// node of the same chunk (internal nodes to a pair of them), the sentinel
    /// points to itself, and every key offset leaves room for all lanes.
    fn chain_invariants_hold(&self, chained: &ChainedTrees) -> bool {
        let key_block = 2 * self.keys.num_features * LOCKSTEP_LANES;
        chained.words.len() == chained.values.len()
            && chained.chunk_starts.first() == Some(&0)
            && chained.chunk_starts.last() == Some(&chained.words.len())
            && chained.chunk_starts.windows(2).all(|bounds| {
                let (first, end) = (bounds[0], bounds[1]);
                let sentinel = end - 1;
                first < end
                    && chained.words[sentinel] == u64::from(LEAF_HI) | ((sentinel as u64) << 32)
                    && (first..sentinel).all(|index| {
                        let word = chained.words[index];
                        let target = (word >> 32) as usize;
                        let key_offset = usize::from((word >> 16) as u16);
                        let in_block = key_offset + LOCKSTEP_LANES <= key_block;
                        if word as u8 == LEAF_HI {
                            in_block && index < target && target <= sentinel
                        } else {
                            in_block && index < target && target + 1 < sentinel
                        }
                    })
            })
    }

    /// Checks every invariant `score_tile`'s unchecked indexing relies on.
    ///
    /// For each tree's node range `first..end`: leaves loop to themselves and
    /// index one of the tree's leaves; internal nodes point to a left child
    /// with `index < left` and `left + 1 < end`; and every key offset leaves
    /// room for all lanes in a group's key block. So a walk from `first` only
    /// moves forward within `first..end`, stops at a leaf, and reads keys
    /// inside the block.
    fn invariants_hold(&self) -> bool {
        let key_block = 2 * self.keys.num_features * LOCKSTEP_LANES;
        self.trees
            .iter()
            .enumerate()
            .all(|(tree_index, &(first, first_leaf))| {
                let (end, end_leaf) = self
                    .trees
                    .get(tree_index + 1)
                    .copied()
                    .unwrap_or((self.nodes.len(), self.leaves.len()));
                first < end
                    && (first..end).all(|index| {
                        let word = self.nodes[index];
                        let target = (word >> 32) as usize;
                        let key_offset = usize::from((word >> 16) as u16);
                        let in_block = key_offset + LOCKSTEP_LANES <= key_block;
                        if word as u8 == LEAF_HI {
                            let leaf = usize::from((word >> 8) as u8);
                            in_block && target == index && first_leaf + leaf < end_leaf
                        } else {
                            in_block && index < target && target + 1 < end
                        }
                    })
            })
    }

    /// The pattern of one row under `spec`.
    fn pattern<F: FeatureValue>(&self, row: &[F], spec: &RowSpecialization) -> PatternKey {
        let mut values = [0u8; 2 * MAX_VALUE_FEATURES];
        for (position, &feature) in spec.value_features.iter().enumerate() {
            let (nan_left, nan_right) = self.keys.keys(feature, row[feature].widen());
            values[2 * position] = nan_left;
            values[2 * position + 1] = nan_right;
        }
        let mut nan = 0u64;
        for (bit, &feature) in spec.nan_features.iter().enumerate() {
            if spec.nan_bit[feature].is_some()
                && normalize_dense_zero(row[feature].widen()).is_nan()
            {
                nan |= 1u64 << bit;
            }
        }
        PatternKey { values, nan }
    }

    /// Trees pruned for `pattern`, built on first use and shared by the
    /// specialization's callers.
    fn pruned_trees(
        &self,
        pattern: &PatternKey,
        spec: &RowSpecialization,
    ) -> Option<std::sync::Arc<ChainedTrees>> {
        if let Ok(cache) = spec.pruned.read() {
            if let Some(found) = cache.by_pattern.get(pattern) {
                return Some(found.clone());
            }
        }
        let built = std::sync::Arc::new(self.prune(pattern, spec)?);
        if let Ok(mut cache) = spec.pruned.write() {
            if let Some(found) = cache.by_pattern.get(pattern) {
                return Some(found.clone());
            }
            if cache.nodes + built.words.len() > MAX_PRUNED_NODES {
                cache.by_pattern.clear();
                cache.nodes = 0;
            }
            cache.nodes += built.words.len();
            cache.by_pattern.insert(*pattern, built.clone());
        }
        Some(built)
    }

    /// Rebuild every tree breadth first, following the fixed splits.
    ///
    /// A split reads the NaN-left key when its slot is below `num_features`,
    /// else the NaN-right key, and goes right when that key exceeds `hi`; the
    /// pattern fixes both keys of a value feature. A NaN key goes left on the
    /// NaN-left slot (0 never exceeds `hi`) and right on the NaN-right slot
    /// (255 always does).
    fn prune(&self, pattern: &PatternKey, spec: &RowSpecialization) -> Option<ChainedTrees> {
        let num_features = self.keys.num_features;
        let resolve = |mut index: usize| -> usize {
            loop {
                let word = self.nodes[index];
                if word as u8 == LEAF_HI {
                    return index;
                }
                // Bits 16..32 hold the key offset, slot * LOCKSTEP_LANES.
                let slot = usize::from((word >> 16) as u16) / LOCKSTEP_LANES;
                let (feature, nan_right_slot) = if slot < num_features {
                    (slot, false)
                } else {
                    (slot - num_features, true)
                };
                let go_right = if let Some(position) = spec.value_position[feature] {
                    let key =
                        pattern.values[2 * usize::from(position) + usize::from(nan_right_slot)];
                    key > word as u8
                } else if spec.nan_bit[feature].is_some_and(|bit| pattern.nan & (1u64 << bit) != 0)
                {
                    nan_right_slot
                } else {
                    return index;
                };
                index = (word >> 32) as usize + usize::from(go_right);
            }
        };
        // Pruned trees never exceed the full ones, so one allocation suffices.
        let mut nodes: Vec<u64> = Vec::with_capacity(self.nodes.len());
        let mut starts = Vec::with_capacity(self.trees.len());
        let mut queue: Vec<usize> = Vec::new();
        for &(first_node, _) in &self.trees {
            let start = nodes.len();
            starts.push(start);
            queue.clear();
            queue.push(resolve(first_node));
            let mut position = 0;
            while position < queue.len() {
                let original = self.nodes[queue[position]];
                let index = (start + position) as u64;
                let word = if original as u8 == LEAF_HI {
                    (original & 0xFFFF_FFFF) | (index << 32)
                } else {
                    let left = (start + queue.len()) as u64;
                    let original_left = (original >> 32) as usize;
                    queue.push(resolve(original_left));
                    queue.push(resolve(original_left + 1));
                    (original & 0xFFFF_FFFF) | (left << 32)
                };
                nodes.push(word);
                position += 1;
            }
        }
        self.chain(&nodes, &starts)
    }

    /// Margins of `rows`, each group of rows sharing a pattern scored on its
    /// pruned trees. Groups below `MIN_PRUNED_GROUP` rows share the full trees.
    pub(super) fn score_rows_specialized<F: FeatureValue>(
        &self,
        rows: &[F],
        margins: &mut [f64],
        spec: &RowSpecialization,
    ) {
        let num_features = self.keys.num_features;
        let row_count = margins.len();
        debug_assert_eq!(rows.len(), row_count * num_features);
        let mut ordered: Vec<(PatternKey, u32)> = rows
            .chunks_exact(num_features)
            .enumerate()
            .map(|(row_index, row)| (self.pattern(row, spec), row_index as u32))
            .collect();
        ordered.sort_unstable();
        let mut full_tree_rows: Vec<usize> = Vec::new();
        let mut start = 0;
        while start < ordered.len() {
            let mut end = start + 1;
            while end < ordered.len() && ordered[end].0 == ordered[start].0 {
                end += 1;
            }
            let group: Vec<usize> = ordered[start..end]
                .iter()
                .map(|(_, row)| *row as usize)
                .collect();
            if group.len() >= MIN_PRUNED_GROUP {
                match self.pruned_trees(&ordered[start].0, spec) {
                    Some(pruned) => self.score_row_group(rows, margins, &group, &pruned),
                    None => full_tree_rows.extend(group),
                }
            } else {
                full_tree_rows.extend(group);
            }
            start = end;
        }
        if !full_tree_rows.is_empty() {
            self.score_row_group(rows, margins, &full_tree_rows, &self.chained);
        }
    }

    /// Chained walks of the chosen rows over `trees`: the plan's own trees or
    /// pruned copies of them.
    fn score_row_group<F: FeatureValue>(
        &self,
        rows: &[F],
        margins: &mut [f64],
        group: &[usize],
        trees: &ChainedTrees,
    ) {
        const LANES: usize = LOCKSTEP_LANES;
        let num_features = self.keys.num_features;
        let key_block = 2 * num_features * LANES;
        let mut keys = vec![0u8; group.len().div_ceil(LANES) * key_block];
        for (position, &row_index) in group.iter().enumerate() {
            let block = &mut keys[(position / LANES) * key_block..][..key_block];
            let lane = position % LANES;
            let row = &rows[row_index * num_features..][..num_features];
            for &feature in &self.keys.used_features {
                let (nan_left, nan_right) = self.keys.keys(feature, row[feature].widen());
                block[feature * LANES + lane] = nan_left;
                block[(num_features + feature) * LANES + lane] = nan_right;
            }
        }
        // Each row's margin starts from its incoming value and adds leaves in tree order.
        let mut group_margins: Vec<f64> = group.iter().map(|&row| margins[row]).collect();
        Self::walk_chunks(trees, &keys, key_block, &mut group_margins);
        for (position, &row_index) in group.iter().enumerate() {
            margins[row_index] = group_margins[position];
        }
    }

    fn score_tile<F: FeatureValue>(&self, tile: &[F], margins: &mut [f64]) {
        let num_features = self.keys.num_features;
        let row_count = margins.len();
        let key_block = 2 * num_features * LOCKSTEP_LANES;
        // Retain the validated key stride, but never traverse padding lanes.
        let mut keys = vec![0u8; row_count.div_ceil(LOCKSTEP_LANES) * key_block];
        for (row_index, row) in tile.chunks_exact(num_features).take(row_count).enumerate() {
            let group = &mut keys[(row_index / LOCKSTEP_LANES) * key_block..][..key_block];
            let lane = row_index % LOCKSTEP_LANES;
            for &feature in &self.keys.used_features {
                let (nan_left, nan_right) = self.keys.keys(feature, row[feature].widen());
                group[feature * LOCKSTEP_LANES + lane] = nan_left;
                group[(num_features + feature) * LOCKSTEP_LANES + lane] = nan_right;
            }
        }
        Self::walk_chunks(&self.chained, &keys, key_block, margins);
    }

    /// Adds each row's leaf values to `margins`, chunk by chunk, so a chunk's
    /// nodes serve every group of rows while they are hot in cache.
    fn walk_chunks(trees: &ChainedTrees, keys: &[u8], key_block: usize, margins: &mut [f64]) {
        let full_rows = margins.len() / LOCKSTEP_LANES * LOCKSTEP_LANES;
        let full_keys = full_rows / LOCKSTEP_LANES * key_block;
        let (full_margins, tail_margins) = margins.split_at_mut(full_rows);
        for bounds in trees.chunk_starts.windows(2) {
            let (root, sentinel) = (bounds[0], bounds[1] - 1);
            for (group, scores) in keys[..full_keys]
                .chunks_exact(key_block)
                .zip(full_margins.chunks_exact_mut(LOCKSTEP_LANES))
            {
                Self::walk_chunk::<true>(trees, group, scores, root, sentinel);
            }
            if !tail_margins.is_empty() {
                Self::walk_chunk::<false>(trees, &keys[full_keys..], tail_margins, root, sentinel);
            }
        }
    }

    /// Walks each lane's row from `root` through one chunk until every lane
    /// reaches the chunk's `sentinel`, adding the value of each node it leaves.
    #[inline(always)]
    fn walk_chunk<const FULL: bool>(
        trees: &ChainedTrees,
        group: &[u8],
        margins: &mut [f64],
        root: usize,
        sentinel: usize,
    ) {
        let lane_count = if FULL { LOCKSTEP_LANES } else { margins.len() };
        let mut lanes = [root; LOCKSTEP_LANES];
        let mut sums = [0.0f64; LOCKSTEP_LANES];
        sums[..lane_count].copy_from_slice(&margins[..lane_count]);
        loop {
            let mut pending = 0usize;
            for (lane, (index, sum)) in lanes[..lane_count]
                .iter_mut()
                .zip(&mut sums[..lane_count])
                .enumerate()
            {
                // SAFETY: `chain_invariants_hold` proved that walks from a
                // chunk's first node only move forward inside the chunk and
                // stop at its sentinel, and that key offsets plus any lane stay
                // inside one `key_block`-sized group. Partial groups use fewer
                // lanes without changing that validated key stride.
                let (word, key, value) = unsafe {
                    let word = *trees.words.get_unchecked(*index);
                    let key_offset = usize::from((word >> 16) as u16);
                    (
                        word,
                        *group.get_unchecked(key_offset + lane),
                        *trees.values.get_unchecked(*index),
                    )
                };
                *sum += value;
                let next = (word >> 32) as usize + usize::from(key > word as u8);
                pending |= next ^ sentinel;
                *index = next;
            }
            if pending == 0 {
                break;
            }
        }
        margins[..lane_count].copy_from_slice(&sums[..lane_count]);
    }
}
