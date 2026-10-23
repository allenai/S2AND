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
//!   starts at `0.0`, exactly as the baseline does. The parent applies the
//!   same sigmoid afterwards.
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
    #[inline]
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

/// Lockstep traversal over `u8` keys for trees with at most 256 leaves.
///
/// Trees are renumbered breadth first, so each internal node's children are
/// adjacent and later, and leaves become self-looping pseudo-nodes. Each node
/// is one `u64`:
/// - bits 0..8: `hi`; the row moves to the right child when its key is
///   greater. Internal nodes have `hi <= 254`; leaves have [`LEAF_HI`].
/// - bits 8..16: leaf index within the tree (leaves only).
/// - bits 16..32: key offset, `slot * LOCKSTEP_LANES`, within a group's keys.
/// - bits 32..64: index of the left child; the right child follows it.
///   Leaves store their own index.
///
/// Rows are transposed into groups of [`LOCKSTEP_LANES`], laid out as
/// `[slot][lane]`, and each group walks a tree until no lane moves.
#[derive(Debug, Clone)]
pub(super) struct Lockstep {
    keys: KeyTable,
    nodes: Vec<u64>,
    leaves: Vec<f64>,
    /// `(first node, first leaf)` per tree, in model order.
    trees: Vec<(usize, usize)>,
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
        let plan = Self {
            keys,
            nodes,
            leaves,
            trees,
        };
        plan.invariants_hold().then_some(plan)
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
        let full_rows = row_count / LOCKSTEP_LANES * LOCKSTEP_LANES;
        let full_keys = full_rows / LOCKSTEP_LANES * key_block;
        for &(first_node, first_leaf) in &self.trees {
            let (full_margins, tail_margins) = margins.split_at_mut(full_rows);
            for (group, scores) in keys[..full_keys]
                .chunks_exact(key_block)
                .zip(full_margins.chunks_exact_mut(LOCKSTEP_LANES))
            {
                self.score_group::<true>(group, scores, first_node, first_leaf);
            }
            if !tail_margins.is_empty() {
                self.score_group::<false>(&keys[full_keys..], tail_margins, first_node, first_leaf);
            }
        }
    }

    #[inline]
    fn score_group<const FULL: bool>(
        &self,
        group: &[u8],
        margins: &mut [f64],
        first_node: usize,
        first_leaf: usize,
    ) {
        let lane_count = if FULL { LOCKSTEP_LANES } else { margins.len() };
        let mut lanes = [first_node; LOCKSTEP_LANES];
        loop {
            let mut moved = 0usize;
            for (lane, index) in lanes[..lane_count].iter_mut().enumerate() {
                // SAFETY: `invariants_hold` proved at construction that
                // walks from `first_node` stay inside this tree's node
                // range and that key offsets plus any lane stay inside
                // one `key_block`-sized group. Partial groups use fewer lanes
                // without changing that validated key stride.
                let (word, key) = unsafe {
                    let word = *self.nodes.get_unchecked(*index);
                    let key_offset = usize::from((word >> 16) as u16);
                    (word, *group.get_unchecked(key_offset + lane))
                };
                let next = (word >> 32) as usize + usize::from(key > word as u8);
                moved |= next ^ *index;
                *index = next;
            }
            if moved == 0 {
                break;
            }
        }
        for (margin, &index) in margins.iter_mut().zip(&lanes[..lane_count]) {
            // SAFETY: no active lane moved, so each is on a leaf, whose
            // leaf index `invariants_hold` bounded by this tree's leaves.
            *margin += unsafe {
                let leaf = usize::from((*self.nodes.get_unchecked(index) >> 8) as u8);
                *self.leaves.get_unchecked(first_leaf + leaf)
            };
        }
    }
}
