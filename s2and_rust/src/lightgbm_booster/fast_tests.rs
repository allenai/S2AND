//! Fast plans versus the baseline traversal, compared bit-for-bit.
//!
//! Every supported case runs through both kernels: appending one 33-leaf tree
//! on a spare feature moves a QuickScorer model onto the lockstep kernel.
//! Fallback cases assert that no plan is built, so the booster scores them
//! through the unchanged baseline.

use super::fast::{is_proper_tree, FastPlan};
use super::*;

const ROW_COUNTS: [usize; 15] = [0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 255, 256, 257, 1001];
const THREAD_COUNTS: [usize; 3] = [1, 4, 8];
const NAN_AND_NONE_DECISION_TYPES: [u8; 4] = [0, 2, 8, 10];
const K: f64 = K_ZERO_THRESHOLD;

/// Small deterministic xorshift generator; tests need no external RNG.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Debug, Clone)]
enum Node {
    Leaf(f64),
    Split {
        feature: usize,
        threshold: f64,
        decision_type: u8,
        left: Box<Node>,
        right: Box<Node>,
    },
}

fn split(feature: usize, threshold: f64, decision_type: u8, left: Node, right: Node) -> Node {
    Node::Split {
        feature,
        threshold,
        decision_type,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// A right-leaning chain: internal node `j` splits at `thresholds[j]` and its
/// left child is a leaf. `thresholds.len() + 1` leaves.
fn chain(feature: usize, thresholds: &[f64], decision_type: u8, leaf_base: f64) -> Node {
    let mut node = Node::Leaf(leaf_base + thresholds.len() as f64);
    for (depth, threshold) in thresholds.iter().enumerate().rev() {
        node = split(
            feature,
            *threshold,
            decision_type,
            Node::Leaf(leaf_base + depth as f64),
            node,
        );
    }
    node
}

/// A complete tree of `depth` levels; level `d` splits `feature_of(d)`.
fn balanced(
    depth: usize,
    feature_of: &dyn Fn(usize) -> usize,
    threshold_of: &dyn Fn(usize, usize) -> f64,
    decision_type: u8,
    next_leaf: &mut f64,
) -> Node {
    fn build(
        level: usize,
        position: usize,
        depth: usize,
        feature_of: &dyn Fn(usize) -> usize,
        threshold_of: &dyn Fn(usize, usize) -> f64,
        decision_type: u8,
        next_leaf: &mut f64,
    ) -> Node {
        if level == depth {
            *next_leaf += 1.0;
            return Node::Leaf(*next_leaf * 0.125);
        }
        let left = build(
            level + 1,
            2 * position,
            depth,
            feature_of,
            threshold_of,
            decision_type,
            next_leaf,
        );
        let right = build(
            level + 1,
            2 * position + 1,
            depth,
            feature_of,
            threshold_of,
            decision_type,
            next_leaf,
        );
        split(
            feature_of(level),
            threshold_of(level, position),
            decision_type,
            left,
            right,
        )
    }
    build(
        0,
        0,
        depth,
        feature_of,
        threshold_of,
        decision_type,
        next_leaf,
    )
}

/// A random proper tree with `leaves` leaves drawn from per-feature pools.
fn random_tree(rng: &mut Rng, leaves: usize, pools: &[Vec<f64>], decision_types: &[u8]) -> Node {
    if leaves == 1 {
        let magnitude = [1e-3, 0.1, 1.0, 7.0][rng.below(4)];
        return Node::Leaf((rng.unit() - 0.5) * magnitude);
    }
    let left_leaves = 1 + rng.below(leaves - 1);
    let feature = rng.below(pools.len());
    let threshold = pools[feature][rng.below(pools[feature].len())];
    let decision_type = decision_types[rng.below(decision_types.len())];
    let left = random_tree(rng, left_leaves, pools, decision_types);
    let right = random_tree(rng, leaves - left_leaves, pools, decision_types);
    split(feature, threshold, decision_type, left, right)
}

#[derive(Default)]
struct TreeArrays {
    split_feature: Vec<usize>,
    threshold: Vec<f64>,
    decision_type: Vec<u8>,
    left_child: Vec<i32>,
    right_child: Vec<i32>,
    leaf_value: Vec<f64>,
}

/// Preorder numbering, as LightGBM writes it: children follow their parent.
fn flatten(node: &Node, arrays: &mut TreeArrays) -> i32 {
    match node {
        Node::Leaf(value) => {
            arrays.leaf_value.push(*value);
            !((arrays.leaf_value.len() - 1) as i32)
        }
        Node::Split {
            feature,
            threshold,
            decision_type,
            left,
            right,
        } => {
            let index = arrays.split_feature.len();
            arrays.split_feature.push(*feature);
            arrays.threshold.push(*threshold);
            arrays.decision_type.push(*decision_type);
            arrays.left_child.push(0);
            arrays.right_child.push(0);
            arrays.left_child[index] = flatten(left, arrays);
            arrays.right_child[index] = flatten(right, arrays);
            index as i32
        }
    }
}

fn joined<T>(values: &[T], format: impl Fn(&T) -> String) -> String {
    values.iter().map(format).collect::<Vec<_>>().join(" ")
}

/// Shortest round-trip text; also spells `-0e0`, `inf` and `NaN`.
fn float_text(value: &f64) -> String {
    format!("{value:e}")
}

fn model_text(num_features: usize, trees: &[Node]) -> String {
    let mut text = format!(
        "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\n\
         max_feature_idx={}\nobjective=binary sigmoid:1.7\n\n",
        num_features - 1
    );
    for (tree_index, tree) in trees.iter().enumerate() {
        let mut arrays = TreeArrays::default();
        flatten(tree, &mut arrays);
        text += &format!(
            "Tree={tree_index}\nnum_leaves={}\nnum_cat=0\n",
            arrays.leaf_value.len()
        );
        if !arrays.split_feature.is_empty() {
            text += &format!(
                "split_feature={}\nsplit_gain={}\nthreshold={}\ndecision_type={}\n\
                 left_child={}\nright_child={}\n",
                joined(&arrays.split_feature, |value| value.to_string()),
                joined(&arrays.split_feature, |_| "1".to_string()),
                joined(&arrays.threshold, float_text),
                joined(&arrays.decision_type, |value| value.to_string()),
                joined(&arrays.left_child, |value| value.to_string()),
                joined(&arrays.right_child, |value| value.to_string()),
            );
        }
        text += &format!(
            "leaf_value={}\nis_linear=0\nshrinkage=1\n\n",
            joined(&arrays.leaf_value, float_text)
        );
    }
    text + "end of trees\n"
}

fn parse(num_features: usize, trees: &[Node]) -> LgbModel {
    parse_model(&model_text(num_features, trees)).unwrap()
}

/// The lockstep variant of a case: one extra 33-leaf tree on `spare_feature`.
fn with_lockstep_tree(trees: &[Node], spare_feature: usize) -> Vec<Node> {
    let thresholds: Vec<f64> = (0..32).map(|index| index as f64 * 0.25 - 4.0).collect();
    let mut trees = trees.to_vec();
    trees.push(chain(spare_feature, &thresholds, 10, -0.5));
    trees
}

/// Edge values every feature draws from: signed zeros, the dense zero window
/// and its f64/f32 neighbours, infinities, NaN payloads and subnormals.
fn special_values() -> Vec<f64> {
    let k32 = K_ZERO_THRESHOLD_F32;
    vec![
        0.0,
        -0.0,
        K,
        -K,
        K.next_up(),
        K.next_down(),
        (-K).next_up(),
        (-K).next_down(),
        K * 0.5,
        -K * 0.5,
        f64::from(k32.next_up()),
        f64::from(k32.next_down()),
        f64::from((-k32).next_down()),
        f64::from((-k32).next_up()),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7ff0_0000_0000_0001),
        f64::from_bits(0xfff8_0000_0000_0001),
        f64::from(f32::from_bits(0x7f80_0001)),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        5e-324,
        -5e-324,
        f64::from(f32::from_bits(1)),
        f64::from(f32::MIN_POSITIVE),
        f64::MAX,
        f64::MIN,
        1.0,
        -1.0,
        0.5,
    ]
}

/// Rows mixing each feature's exact thresholds, their f64 and f32
/// neighbours, edge values and plain random values.
fn adversarial_rows(model: &LgbModel, row_count: usize, seed: u64) -> Vec<f64> {
    let num_features = model.num_features;
    let mut thresholds: Vec<Vec<f64>> = vec![Vec::new(); num_features];
    for tree in &model.trees {
        for (feature, threshold) in tree.split_feature.iter().zip(&tree.threshold) {
            thresholds[usize::from(*feature)].push(*threshold);
        }
    }
    let specials = special_values();
    let mut rng = Rng(seed);
    let mut rows = Vec::with_capacity(row_count * num_features);
    for _ in 0..row_count {
        for feature_thresholds in &thresholds {
            let choice = rng.below(10);
            let value = if choice < 6 && !feature_thresholds.is_empty() {
                let threshold = feature_thresholds[rng.below(feature_thresholds.len())];
                let narrowed = threshold as f32;
                match choice {
                    0 | 1 => threshold,
                    2 => threshold.next_up(),
                    3 => threshold.next_down(),
                    4 => f64::from(narrowed.next_up()),
                    _ => f64::from(narrowed.next_down()),
                }
            } else if choice < 8 {
                specials[rng.below(specials.len())]
            } else {
                (rng.unit() - 0.5) * 8.0
            };
            rows.push(value);
        }
    }
    rows
}

fn assert_same_bits(actual: &[f64], expected: &[f64], rows: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let num_features = rows.len() / actual.len().max(1);
    if let Some(row) =
        (0..actual.len()).find(|&row| actual[row].to_bits() != expected[row].to_bits())
    {
        panic!(
            "{label}: row {row} {:?} scored {:e}, baseline {:e}",
            &rows[row * num_features..][..num_features],
            actual[row],
            expected[row]
        );
    }
}

/// Fast plan versus the single-threaded baseline for f64 and f32 rows, raw
/// margins and probabilities, at 1 and several threads.
fn assert_matches_baseline(model: &LgbModel, plan: &FastPlan, rows: &[f64], label: &str) {
    let num_features = model.num_features;
    let rows_f32: Vec<f32> = rows.iter().map(|value| *value as f32).collect();
    for apply_sigmoid in [false, true] {
        let expected = predict_rows(model, rows, num_features, 1, apply_sigmoid);
        let expected_f32 = predict_rows_f32(model, &rows_f32, num_features, 1, apply_sigmoid);
        for threads in THREAD_COUNTS {
            let case = format!("{label} threads={threads} sigmoid={apply_sigmoid}");
            let actual = plan.predict_rows(model, rows, threads, apply_sigmoid);
            assert_same_bits(&actual, &expected, rows, &format!("{case} f64"));
            let actual_f32 = plan.predict_rows(model, &rows_f32, threads, apply_sigmoid);
            assert_same_bits(&actual_f32, &expected_f32, rows, &format!("{case} f32"));
        }
    }
}

fn assert_row_count_sweep(model: &LgbModel, plan: &FastPlan, rows: &[f64], label: &str) {
    for row_count in ROW_COUNTS {
        let prefix = &rows[..row_count * model.num_features];
        assert_matches_baseline(model, plan, prefix, &format!("{label} rows={row_count}"));
    }
}

/// Builds the plan, checks its kernel, and compares it on adversarial rows.
fn check_kernel(model: &LgbModel, kernel: &str, sweep_row_counts: bool, label: &str) {
    let plan = FastPlan::build(model).unwrap_or_else(|| panic!("{label}: no fast plan"));
    assert_eq!(plan.kernel_name(), kernel, "{label}");
    let rows = adversarial_rows(model, 1001, 0x9E37_79B9_7F4A_7C15);
    if sweep_row_counts {
        assert_row_count_sweep(model, &plan, &rows, label);
    } else {
        assert_matches_baseline(model, &plan, &rows, label);
    }
}

/// Runs a QuickScorer-sized case on both kernels.
fn check_both_kernels(num_features: usize, trees: &[Node], sweep_row_counts: bool, label: &str) {
    check_kernel(
        &parse(num_features, trees),
        "quickscorer",
        sweep_row_counts,
        &format!("{label} quickscorer"),
    );
    check_kernel(
        &parse(num_features + 1, &with_lockstep_tree(trees, num_features)),
        "lockstep",
        sweep_row_counts,
        &format!("{label} lockstep"),
    );
}

#[test]
fn toy_decision_types_and_thresholds_match_baseline() {
    let thresholds = [
        f64::NEG_INFINITY,
        -1.0,
        -K,
        -K * 0.5,
        -0.0,
        0.0,
        K * 0.5,
        K,
        0.5,
        f64::INFINITY,
    ];
    for decision_type in NAN_AND_NONE_DECISION_TYPES {
        for threshold in thresholds {
            let trees = [
                split(
                    0,
                    threshold,
                    decision_type,
                    Node::Leaf(-2.0),
                    Node::Leaf(3.0),
                ),
                split(
                    1,
                    0.25,
                    decision_type,
                    split(
                        0,
                        threshold,
                        decision_type,
                        Node::Leaf(0.5),
                        Node::Leaf(0.75),
                    ),
                    Node::Leaf(-0.125),
                ),
            ];
            check_both_kernels(
                2,
                &trees,
                true,
                &format!("decision_type={decision_type} threshold={threshold:e}"),
            );
        }
    }
}

#[test]
fn random_models_match_baseline() {
    let pool = |rng: &mut Rng| -> Vec<f64> {
        let mut values = vec![
            f64::NEG_INFINITY,
            -K,
            -0.0,
            0.0,
            K,
            K * 0.5,
            0.5,
            f64::from(0.1f32),
            0.1,
            1.0,
            f64::INFINITY,
            5e-324,
            -1e-300,
        ];
        values.extend((0..12).map(|_| (rng.unit() - 0.5) * 6.0));
        values
    };
    for seed in 1..=6u64 {
        let mut rng = Rng(0xA5A5_5A5A_0000_0000 ^ seed);
        let num_features = 1 + rng.below(6);
        let pools: Vec<Vec<f64>> = (0..num_features).map(|_| pool(&mut rng)).collect();
        for (kernel, max_leaves, tree_count) in [("quickscorer", 32, 40), ("lockstep", 256, 12)] {
            // The last tree has the kernel's maximum leaf count.
            let trees: Vec<Node> = (0..=tree_count)
                .map(|index| {
                    let leaves = if index == tree_count {
                        max_leaves
                    } else {
                        1 + rng.below(max_leaves)
                    };
                    random_tree(&mut rng, leaves, &pools, &NAN_AND_NONE_DECISION_TYPES)
                })
                .collect();
            let model = parse(num_features, &trees);
            let label = format!("seed={seed} {kernel}");
            check_kernel(&model, kernel, seed == 1, &label);
        }
    }
}

#[test]
fn sparse_features_and_every_partial_group_match_baseline() {
    let num_features = 256;
    // Leave feature 0 unused: lockstep leaf pseudo-nodes can still read its
    // zero-initialized key because their compare bound never moves them.
    let small = split(255, 0.5, 8, Node::Leaf(-0.25), Node::Leaf(0.75));
    let mut deep_left = Node::Leaf(-1.0);
    for depth in (0..32).rev() {
        deep_left = split(
            255,
            depth as f64 + 1.0,
            10,
            deep_left,
            Node::Leaf(depth as f64 * 0.125),
        );
    }
    for (kernel, trees) in [
        ("quickscorer", vec![small.clone(), Node::Leaf(0.125)]),
        ("lockstep", vec![small, deep_left, Node::Leaf(0.125)]),
        ("quickscorer", vec![Node::Leaf(0.25), Node::Leaf(-0.125)]),
    ] {
        let model = parse(num_features, &trees);
        let plan = FastPlan::build(&model).unwrap();
        assert_eq!(plan.kernel_name(), kernel);
        for tail in 1..16 {
            // Include full groups before every possible QuickScorer tail;
            // this also covers every lockstep tail. Unused inputs include
            // NaNs/infinities/zero-window values and must not affect scoring.
            let row_count = 16 + tail;
            let specials = special_values();
            let rows: Vec<f64> = (0..row_count)
                .flat_map(|row| {
                    let specials = &specials;
                    (0..num_features).map(move |feature| {
                        if feature == 255 {
                            if row < 16 {
                                [-2.0, 0.5, 2.5, f64::NAN][row % 4]
                            } else {
                                // Actual tail rows exit the deep tree at its
                                // root; artificial key-0 lanes would walk all
                                // 32 levels of its left branch.
                                100.0
                            }
                        } else {
                            specials[(row + feature) % specials.len()]
                        }
                    })
                })
                .collect();
            assert_matches_baseline(&model, &plan, &rows, &format!("{kernel} tail={tail}"));
            assert_matches_baseline(
                &model,
                &plan,
                &rows[16 * num_features..],
                &format!("{kernel} standalone tail={tail}"),
            );
        }
    }
}

#[test]
fn pruned_trees_match_baseline_on_shared_patterns() {
    let pool = |rng: &mut Rng| -> Vec<f64> {
        let mut values = vec![f64::NEG_INFINITY, -K, -0.0, 0.0, K, 0.5, 1.0, f64::INFINITY];
        values.extend((0..10).map(|_| (rng.unit() - 0.5) * 6.0));
        values
    };
    for seed in 1..=5u64 {
        let mut rng = Rng(0x5EED_0000_0000_0000 ^ seed);
        let num_features = 5 + rng.below(4);
        let pools: Vec<Vec<f64>> = (0..num_features).map(|_| pool(&mut rng)).collect();
        // The last tree exceeds QuickScorer's leaf limit, so the plan is lockstep.
        let trees: Vec<Node> = (0..=24)
            .map(|index| {
                let leaves = if index == 24 { 90 } else { 1 + rng.below(80) };
                random_tree(&mut rng, leaves, &pools, &NAN_AND_NONE_DECISION_TYPES)
            })
            .collect();
        let model = parse(num_features, &trees);
        let plan = FastPlan::build(&model).expect("fast plan");
        assert_eq!(plan.kernel_name(), "lockstep");
        let templates = adversarial_rows(&model, 6, seed ^ 0xABCD);
        for (value_features, nan_features) in [
            (vec![0, 1], vec![2, 3]),
            (vec![], vec![0, 1, 2, 3, 4]),
            (vec![4], vec![]),
            (vec![1, 3], vec![1, 2]),
        ] {
            let mut rows = adversarial_rows(&model, 777, seed);
            for row in rows.chunks_exact_mut(num_features) {
                // Value features repeat a template, so rows share patterns.
                let template = &templates[rng.below(6) * num_features..][..num_features];
                for &feature in &value_features {
                    row[feature] = template[feature];
                }
                for &feature in &nan_features {
                    if rng.below(10) < 4 {
                        row[feature] = f64::NAN;
                    }
                }
            }
            let spec = RowSpecialization::new(0, num_features, value_features.clone(), nan_features.clone())
                .expect("specialization fits");
            for apply_sigmoid in [false, true] {
                let expected = predict_rows(&model, &rows, num_features, 1, apply_sigmoid);
                let actual = plan.predict_rows_specialized(&model, &rows, apply_sigmoid, &spec);
                assert_same_bits(
                    &actual,
                    &expected,
                    &rows,
                    &format!("seed={seed} values={value_features:?} nans={nan_features:?} sigmoid={apply_sigmoid}"),
                );
            }
        }
    }
}

#[test]
fn leaf_limits_select_kernels() {
    // At most 128 distinct thresholds, so only the leaf count can fall back.
    let chain_thresholds = |leaves: usize| -> Vec<f64> {
        (0..leaves - 1)
            .map(|index| (index % 128) as f64 * 0.5 - 8.0)
            .collect()
    };
    for (leaves, kernel) in [
        (2, Some("quickscorer")),
        (31, Some("quickscorer")),
        (32, Some("quickscorer")),
        (33, Some("lockstep")),
        (256, Some("lockstep")),
        (257, None),
    ] {
        for decision_type in [2, 8] {
            let trees = [
                chain(0, &chain_thresholds(leaves), decision_type, 0.5),
                split(1, 0.0, 10, Node::Leaf(1.0), Node::Leaf(-1.0)),
            ];
            let model = parse(2, &trees);
            let label = format!("chain leaves={leaves} decision_type={decision_type}");
            match kernel {
                Some(kernel) => check_kernel(&model, kernel, leaves <= 33, &label),
                None => assert!(FastPlan::build(&model).is_none(), "{label}"),
            }
        }
    }
    // Complete trees: 32 leaves (depth 5) and 256 leaves (depth 8).
    for (depth, kernel) in [(5, "quickscorer"), (8, "lockstep")] {
        let mut next_leaf = 0.0;
        let tree = balanced(
            depth,
            &|level| level % 3,
            &|level, position| position as f64 * 0.75 - level as f64,
            8,
            &mut next_leaf,
        );
        let model = parse(3, &[tree.clone(), tree]);
        check_kernel(&model, kernel, true, &format!("balanced depth={depth}"));
    }
}

#[test]
fn single_tree_constants_and_stumps_use_baseline() {
    for tree in [
        Node::Leaf(0.25),
        split(5, 0.5, 10, Node::Leaf(1.5), Node::Leaf(-2.25)),
    ] {
        let booster = RustLightGBMBooster::from_model_text(&model_text(6, &[tree])).unwrap();
        assert!(booster.fast_plan.is_none());
    }
}

#[test]
fn single_leaf_trees_match_baseline() {
    let trees = [
        Node::Leaf(0.25),
        split(0, 0.5, 10, Node::Leaf(1.0), Node::Leaf(2.0)),
        Node::Leaf(-0.125),
    ];
    check_both_kernels(1, &trees, true, "single-leaf mix");
    check_kernel(
        &parse(1, &[Node::Leaf(0.25), Node::Leaf(-3.0)]),
        "quickscorer",
        true,
        "single-leaf only",
    );
}

#[test]
fn catastrophic_cancellation_sums_keep_tree_order() {
    for leaf_values in [[1e20, 1.0, -1e20], [1e20, -1e20, 1.0], [1.0, 1e20, -1e20]] {
        let trees: Vec<Node> = leaf_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                split(
                    index % 2,
                    0.5,
                    [2, 8, 10][index],
                    Node::Leaf(*value),
                    Node::Leaf(-*value * 0.5),
                )
            })
            .chain(leaf_values.iter().map(|value| Node::Leaf(*value)))
            .collect();
        check_both_kernels(2, &trees, true, &format!("cancellation {leaf_values:?}"));
    }
}

/// One stump on feature 0 per threshold in `values`, cycling through NaN/None
/// decision types so both key slots are used.
fn threshold_stumps(values: &[f64]) -> Vec<Node> {
    values
        .iter()
        .enumerate()
        .map(|(index, threshold)| {
            split(
                0,
                *threshold,
                NAN_AND_NONE_DECISION_TYPES[index % 4],
                Node::Leaf(index as f64 * 0.001 + 0.5),
                Node::Leaf(-(index as f64) * 0.002),
            )
        })
        .collect()
}

fn finite_thresholds(count: usize) -> Vec<f64> {
    (0..count).map(|index| index as f64 - 127.0).collect()
}

fn distinct_feature_thresholds(model: &LgbModel, feature: u16) -> usize {
    let mut values: Vec<f64> = model
        .trees
        .iter()
        .flat_map(|tree| {
            tree.split_feature
                .iter()
                .zip(&tree.threshold)
                .filter(move |(split_feature, _)| **split_feature == feature)
                .map(|(_, threshold)| *threshold)
        })
        .collect();
    values.sort_by(f64::total_cmp);
    values.dedup_by(|next, kept| next == kept);
    values.len()
}

#[test]
fn threshold_capacity_is_255_per_feature() {
    let mut with_infinity = finite_thresholds(254);
    with_infinity.push(f64::INFINITY);
    // -0.0 duplicates 0.0 (at index 127) and the repeated values are
    // duplicates, so each set keeps its distinct count.
    let mut duplicated = finite_thresholds(255);
    duplicated.extend([-0.0, 0.0, 5.0, -127.0]);
    let cases: [(&str, Vec<f64>, usize); 5] = [
        ("254 finite", finite_thresholds(254), 254),
        ("254 finite + inf", with_infinity, 255),
        ("255 finite", finite_thresholds(255), 255),
        ("255 finite + duplicates and -0.0", duplicated, 255),
        ("256 finite", finite_thresholds(256), 256),
    ];
    for (label, thresholds, distinct) in cases {
        let trees = threshold_stumps(&thresholds);
        let model = parse(2, &trees);
        assert_eq!(distinct_feature_thresholds(&model, 0), distinct, "{label}");
        if distinct <= 255 {
            check_both_kernels(2, &trees, distinct == 255, label);
        } else {
            assert!(FastPlan::build(&model).is_none(), "{label}");
            let lockstep = parse(3, &with_lockstep_tree(&trees, 2));
            assert!(FastPlan::build(&lockstep).is_none(), "{label} lockstep");
        }
    }
    // Rows above every one of 255 finite thresholds reach key 255, the
    // NaN-right sentinel: both must go right at every NaN-right split.
    let trees = threshold_stumps(&finite_thresholds(255));
    let model = parse(2, &trees);
    let plan = FastPlan::build(&model).unwrap();
    let rows = [
        128.0,
        0.0,
        f64::INFINITY,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        127.0,
        1.0,
        127.0f64.next_up(),
        -1.0,
        f64::NEG_INFINITY,
        f64::INFINITY,
    ];
    assert_matches_baseline(&model, &plan, &rows, "key 255 rows quickscorer");
    let lockstep = parse(3, &with_lockstep_tree(&trees, 2));
    let lockstep_rows: Vec<f64> = rows
        .chunks_exact(2)
        .flat_map(|row| [row[0], row[1], 0.3])
        .collect();
    let plan = FastPlan::build(&lockstep).unwrap();
    assert_eq!(plan.kernel_name(), "lockstep");
    assert_matches_baseline(&lockstep, &plan, &lockstep_rows, "key 255 rows lockstep");
}

#[test]
fn feature_limit_is_256() {
    let trees = |num_features: usize| -> Vec<Node> {
        [0, 17, 128, num_features - 2, num_features - 1]
            .iter()
            .enumerate()
            .map(|(index, feature)| {
                split(
                    *feature,
                    index as f64 * 0.1,
                    NAN_AND_NONE_DECISION_TYPES[index % 4],
                    Node::Leaf(index as f64),
                    Node::Leaf(-(index as f64) - 0.5),
                )
            })
            .collect()
    };
    check_kernel(
        &parse(256, &trees(256)),
        "quickscorer",
        true,
        "256 features",
    );
    let mut lockstep = trees(256);
    lockstep.push(chain(3, &finite_thresholds(40), 8, 0.0));
    check_kernel(&parse(256, &lockstep), "lockstep", true, "256 features");
    assert!(FastPlan::build(&parse(257, &trees(257))).is_none());
}

#[test]
fn zero_missing_splits_fall_back() {
    for decision_type in [4, 6] {
        let trees = [
            split(0, 0.5, 10, Node::Leaf(1.0), Node::Leaf(2.0)),
            split(1, 0.25, decision_type, Node::Leaf(10.0), Node::Leaf(20.0)),
        ];
        assert!(FastPlan::build(&parse(2, &trees)).is_none());
        assert!(FastPlan::build(&parse(3, &with_lockstep_tree(&trees, 2))).is_none());
    }
}

#[test]
fn nan_thresholds_fall_back() {
    let trees = [
        split(0, 0.5, 10, Node::Leaf(1.0), Node::Leaf(2.0)),
        split(1, f64::NAN, 8, Node::Leaf(10.0), Node::Leaf(20.0)),
    ];
    let model = parse(2, &trees);
    assert!(model.trees[1].threshold[0].is_nan());
    assert!(FastPlan::build(&model).is_none());
    assert!(FastPlan::build(&parse(3, &with_lockstep_tree(&trees, 2))).is_none());
}

/// Node 1 is both children of the root. The parser
/// accepts it, but QuickScorer's masks would score row [0, 1] as 10.
const SHARED_CHILD_MODEL: &str = "tree
version=v4
num_class=1
num_tree_per_iteration=1
max_feature_idx=1
objective=binary sigmoid:1
Tree=0
num_leaves=3
num_cat=0
is_linear=0
split_feature=0 1
threshold=0.5 0.5
decision_type=0 0
left_child=1 -1
right_child=1 -2
leaf_value=10 20 30
end of trees
";

#[test]
fn shared_child_model_falls_back_to_baseline_scores() {
    let booster = RustLightGBMBooster::from_model_text(SHARED_CHILD_MODEL).unwrap();
    assert!(!is_proper_tree(&booster.model.trees[0]));
    assert!(booster.fast_plan.is_none());
    let rows = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    for threads in THREAD_COUNTS {
        let scores = predict_rows(&booster.model, &rows, 2, threads, false);
        assert_eq!(scores, vec![20.0, 20.0, 10.0, 10.0]);
    }
}

#[test]
fn improper_topologies_fall_back() {
    let replace_children = |left: &str, right: &str| {
        SHARED_CHILD_MODEL
            .replace("left_child=1 -1", &format!("left_child={left}"))
            .replace("right_child=1 -2", &format!("right_child={right}"))
    };
    for (left, right, description) in [
        ("1 -1", "-3 -2", "proper control"),
        ("1 -1", "-1 -2", "leaf 0 shared, leaf 2 unreferenced"),
        ("-1 -3", "-2 -3", "orphan internal node"),
        ("1 -2", "-2 -3", "leaf 1 shared"),
    ] {
        let model = parse_model(&replace_children(left, right)).unwrap();
        let proper = description == "proper control";
        assert_eq!(is_proper_tree(&model.trees[0]), proper, "{description}");
        assert_eq!(FastPlan::build(&model).is_some(), proper, "{description}");
    }
}

#[test]
fn mixed_leaf_counts_dispatch_per_model() {
    let small = chain(0, &[0.5, 1.5], 10, 0.0);
    let large = chain(0, &finite_thresholds(40), 2, 1.0);
    let quick = parse(1, &[small.clone(), small.clone()]);
    assert_eq!(
        FastPlan::build(&quick).unwrap().kernel_name(),
        "quickscorer"
    );
    check_kernel(
        &parse(1, &[small.clone(), large, small]),
        "lockstep",
        true,
        "mixed",
    );
}

#[test]
fn booster_attaches_and_copies_fast_plan() {
    let tree = split(0, 0.5, 10, Node::Leaf(1.0), Node::Leaf(2.0));
    let text = model_text(2, &[tree.clone(), tree]);
    let booster = RustLightGBMBooster::from_model_text(&text).unwrap();
    assert_eq!(
        booster.fast_plan.as_ref().map(FastPlan::kernel_name),
        Some("quickscorer")
    );
    assert!(booster.__copy__().fast_plan.is_some());
    let fallback =
        RustLightGBMBooster::from_model_text(&text.replace("decision_type=10", "decision_type=4"))
            .unwrap();
    assert!(fallback.fast_plan.is_none());
}

/// Release gate for the actual boosters, including newly trained bundles.
#[test]
#[ignore = "requires S2AND_PAIRWISE_MODEL_DIR pointing to the candidate release bundle"]
fn release_bundle_uses_native_fast_plans() {
    let root = std::path::PathBuf::from(
        std::env::var_os("S2AND_PAIRWISE_MODEL_DIR")
            .expect("set S2AND_PAIRWISE_MODEL_DIR to the candidate pairwise model directory"),
    );
    for filename in ["main.lgb", "nameless.lgb"] {
        let path = root.join(filename);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        let booster = RustLightGBMBooster::from_model_text(&text)
            .unwrap_or_else(|error| panic!("cannot parse {}: {error}", path.display()));
        let plan = booster
            .fast_plan
            .as_ref()
            .unwrap_or_else(|| panic!("{} falls back to baseline native scoring", path.display()));
        println!("{}: {}", path.display(), plan.kernel_name());
    }
}
