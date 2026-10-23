use super::*;
use crate::feature_schema as columns;
use crate::lightgbm_booster::RustLightGBMBooster;

#[pyclass]
#[derive(Clone)]
pub(crate) struct RustFeaturizer {
    signatures: Arc<HashMap<String, SignatureData>>,
    signature_ids: Arc<Vec<String>>,
    papers: Arc<HashMap<PaperId, PaperData>>,
    /// Gram counts of each coauthor name, subtracted from a paper's to exclude a signature's own author.
    name_grams: Arc<NameGrams>,
    name_tuples: Arc<HashMap<String, HashSet<String>>>,
    cluster_seeds_disallow: HashSet<(String, String)>,
    cluster_seeds_require: HashMap<String, ClusterId>,
    cluster_seed_require_value: f64,
    cluster_seed_disallow_value: f64,
    name_counts_manifest_sha256: Option<String>,
    cluster_seeds_disallow_index: OnceLock<HashMap<String, HashSet<String>>>,
}

#[cfg(test)]
mod constraint_key_tests {
    use super::*;

    /// A deterministic xorshift generator, so the test needs no new dependency.
    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next() % bound as u64) as usize
        }

        fn tokens(&mut self, vocabulary: &[&str], max_count: usize) -> String {
            let count = self.below(max_count + 1);
            let tokens: Vec<&str> = (0..count)
                .map(|_| vocabulary[self.below(vocabulary.len())])
                .collect();
            // Odd spacing checks that splitting matches split_whitespace.
            let separator = if self.below(8) == 0 { "  " } else { " " };
            tokens.join(separator)
        }
    }

    const FIRST_TOKENS: [&str; 14] = [
        "j", "jo", "john", "johnny", "james", "jim", "xiao", "qing", "xiaoqing", "jean", "daniel",
        "é", "émile", "x",
    ];
    const MIDDLE_TOKENS: [&str; 18] = [
        "a",
        "b",
        "j",
        "anne",
        "ann",
        "marie",
        "maria",
        "van",
        "de",
        "der",
        "mac",
        "al",
        "é",
        "ö",
        "q",
        "daniel",
        "dan",
        "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
    ];
    const LASTS: [&str; 7] = [
        "smith", "ou yang", "ouyang", "li", "lee", "van dyke", "vandyke",
    ];

    fn signature(first: String, middle: String, last: &str, orcid: Option<&str>) -> SignatureData {
        SignatureData {
            first: Some(first.clone()),
            middle: Some(middle),
            last_normalized: Some(last.to_string()),
            orcid: orcid.map(str::to_string),
            email: None,
            affiliations: None,
            coauthors: SignatureCoauthors::default(),
            position: 0,
            paper_id: "p".to_string(),
            name_counts: None,
            middle_initial_counts: Vec::new(),
            adv_name: Some(first),
        }
    }

    fn featurizer(rng: &mut XorShift, count: usize) -> RustFeaturizer {
        let mut signatures = HashMap::default();
        let mut signature_ids = Vec::new();
        for index in 0..count {
            let first = rng.tokens(&FIRST_TOKENS, 3);
            let middle = if rng.below(10) == 0 {
                // Past the stack capacities, so the hash-set fallback runs too.
                vec!["anne"; 20].join(" ")
            } else {
                rng.tokens(&MIDDLE_TOKENS, 4)
            };
            let last = LASTS[rng.below(LASTS.len())];
            let orcid = [None, None, Some("0000-0001"), Some("0000-0002")][rng.below(4)];
            let signature_id = format!("s{index}");
            signatures.insert(signature_id.clone(), signature(first, middle, last, orcid));
            signature_ids.push(signature_id);
        }
        let mut name_tuples = HashMap::default();
        for (a, b) in [
            ("john", "johnny"),
            ("james", "jim"),
            ("xiao", "xiaoqing"),
            ("jean", "j"),
        ] {
            crate::insert_name_tuple_alias(&mut name_tuples, a.to_string(), b.to_string());
        }
        let mut cluster_seeds_require = HashMap::default();
        for _ in 0..count / 5 {
            let signature_id = format!("s{}", rng.below(count));
            let cluster = match rng.below(4) {
                0 => ClusterId::Str("profile".to_string()),
                value => ClusterId::Int(value as i64),
            };
            cluster_seeds_require.insert(signature_id, cluster);
        }
        let mut cluster_seeds_disallow = HashSet::default();
        for _ in 0..count / 5 {
            let left = format!("s{}", rng.below(count));
            let right = format!("s{}", rng.below(count));
            cluster_seeds_disallow.insert(canonical_signature_pair_owned(left, right));
        }
        RustFeaturizer {
            signatures: Arc::new(signatures),
            signature_ids: Arc::new(signature_ids),
            papers: Arc::new(HashMap::default()),
            name_grams: Arc::new(NameGrams::default()),
            name_tuples: Arc::new(name_tuples),
            cluster_seeds_disallow,
            cluster_seeds_require,
            cluster_seed_require_value: -1.0,
            cluster_seed_disallow_value: 1e6,
            name_counts_manifest_sha256: None,
            cluster_seeds_disallow_index: OnceLock::new(),
        }
    }

    #[test]
    fn keyed_constraints_match_the_per_pair_reference() {
        let mut rng = XorShift(0x5eed_1234_abcd_0001);
        for _ in 0..3 {
            let featurizer = featurizer(&mut rng, 160);
            let entries: Vec<(&str, &SignatureData)> = featurizer
                .signature_ids
                .iter()
                .map(|signature_id| (signature_id.as_str(), &featurizer.signatures[signature_id]))
                .collect();
            let keys = featurizer.constraint_keys(&entries);
            let mut outcomes = HashSet::default();
            for flags in 0..8_u8 {
                let (dont_merge, dont_use_seeds, suppress_orcid) =
                    (flags & 1 != 0, flags & 2 != 0, flags & 4 != 0);
                for left in 0..entries.len() {
                    for right in 0..entries.len() {
                        if left == right {
                            continue;
                        }
                        let (left_id, s1) = entries[left];
                        let (right_id, s2) = entries[right];
                        let expected = featurizer.constraint_value_from_records(
                            left_id,
                            right_id,
                            s1,
                            s2,
                            0.0,
                            1e4,
                            dont_merge,
                            dont_use_seeds,
                            suppress_orcid,
                        );
                        let actual = featurizer.constraint_value_from_keys(
                            &keys,
                            left,
                            right,
                            0.0,
                            1e4,
                            dont_merge,
                            dont_use_seeds,
                            suppress_orcid,
                        );
                        assert_eq!(
                            expected.map(f64::to_bits),
                            actual.map(f64::to_bits),
                            "{left_id} {right_id} flags={flags}"
                        );
                        outcomes.insert(expected.map(f64::to_bits));
                    }
                }
            }
            // Every outcome occurs, so no branch went untested.
            for value in [None, Some(0.0), Some(1e4), Some(-1.0), Some(1e6)] {
                assert!(
                    outcomes.contains(&value.map(f64::to_bits)),
                    "{value:?} never occurred"
                );
            }
        }
    }

    #[test]
    fn views_keep_a_subset_of_the_shared_signature_map() {
        let mut rng = XorShift(0x5eed_0000_0000_0007);
        let full = featurizer(&mut rng, 2);
        let view = full
            .view_with_cluster_seed_sidecars(vec!["s1".to_string()], None, None)
            .expect("view over one signature");
        assert_eq!(view.signature_id_order(), ["s1".to_string()]);
        assert_eq!(view.signatures.len(), 2);
    }
}

#[cfg(test)]
mod compact_index_tests {
    use super::{select_signature_index_layout, SignatureIndexLayout};

    #[test]
    fn dense_indices_keep_global_index_arrays() {
        let selection = select_signature_index_layout(4, 3, || [0, 1, 2, 1, 2, 3].into_iter(), 8)
            .expect("valid dense indices should select a layout");
        assert!(matches!(selection, SignatureIndexLayout::Dense { .. }));
    }

    #[test]
    fn sparse_indices_select_compact_storage() {
        let selection = select_signature_index_layout(
            10_000_001,
            2,
            || [0, 10_000_000, 10_000_000, 0].into_iter(),
            8,
        )
        .expect("valid sparse indices should select a layout");
        let owned_selection = select_signature_index_layout(
            10_000_001,
            2,
            || [0, 10_000_000, 10_000_000, 0].into_iter(),
            0,
        )
        .expect("valid owned sparse indices should select a layout");
        let SignatureIndexLayout::Compact { global_indices } = selection else {
            panic!("sparse indices must select compact storage");
        };
        assert!(matches!(
            owned_selection,
            SignatureIndexLayout::Compact { .. }
        ));
        assert_eq!(global_indices, vec![0, 10_000_000]);
        assert!(global_indices.capacity() <= 4);
    }

    #[test]
    fn repeated_pairs_keep_bounded_direct_dense_layout() {
        let repeated_indices = (0..200).map(|offset| if offset % 2 == 0 { 0 } else { 9 });
        let borrowed = select_signature_index_layout(10, 100, || repeated_indices.clone(), 8)
            .expect("valid borrowed indices should select a layout");
        let owned = select_signature_index_layout(10, 100, || repeated_indices.clone(), 0)
            .expect("valid owned indices should select a layout");
        assert!(matches!(borrowed, SignatureIndexLayout::Dense { .. }));
        assert!(matches!(owned, SignatureIndexLayout::Dense { .. }));
    }

    #[test]
    fn layout_selection_rejects_out_of_range_indices() {
        let error = select_signature_index_layout(2, 1, || [0, 2].into_iter(), 8)
            .expect_err("out-of-range indices must fail");
        assert!(error.contains("index=2 signature_count=2"));
    }
}

/// Per-signature constraint inputs, split once for all pairs of one call.
struct ConstraintKeys<'a> {
    signature_ids: Vec<&'a str>,
    names: Vec<ConstraintNameKey<'a>>,
    orcids: Vec<Option<&'a str>>,
    requires: Vec<Option<&'a ClusterId>>,
    /// Whether a signature appears in any disallowed seed pair.
    in_disallow: Vec<bool>,
    first_ids: Vec<u32>,
    /// Sorted ids of the first names each interned first name aliases.
    first_aliases: Vec<Vec<u32>>,
}

/// Pairs featurized and scored per pool task in fused scoring. Tiles of 64
/// to 256 pairs measured alike; 4,096 cost 15% and 16,384 cost 70% more, as
/// the scorer's tables and the tile buffers left the caches.
const FUSED_SCORING_TILE: usize = 256;

/// A feature value as the pairwise model reads it: `nan_value` for a missing
/// value, unless `nan_value` is itself NaN.
#[inline]
fn model_input(value: f64, nan_value: f64) -> f64 {
    if value.is_nan() && !nan_value.is_nan() {
        nan_value
    } else {
        value
    }
}

/// A row-major `(rows, cols)` NumPy matrix from a flat vector.
fn rows_matrix<'py, T: numpy::Element>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    values: Vec<T>,
    name: &str,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    numpy::ndarray::Array2::from_shape_vec((rows, cols), values)
        .map(|array| array.into_pyarray(py))
        .map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to build {name} matrix: {err}"
            ))
        })
}

#[derive(Clone, Copy)]
struct PairAggregateRowRange {
    row_offset: usize,
    start: usize,
    stop: usize,
}

struct PairAggregateBuffers {
    counts: Vec<u32>,
    valid_counts: Vec<u64>,
    sums: Vec<f64>,
    mins: Vec<f64>,
    maxs: Vec<f64>,
}

struct LinkerPairDistanceAccumulator {
    counts: Vec<u32>,
    sums: Vec<f64>,
    mins: Vec<f64>,
    top_distances: Vec<f64>,
    hard_disallow_pair_count: u64,
}

#[derive(Debug)]
enum SignatureIndexLayout {
    Dense { max_index: usize },
    Compact { global_indices: Vec<usize> },
}

enum BorrowedSignaturePaperLookup<'data> {
    Dense(Vec<Option<(&'data SignatureData, &'data PaperData)>>),
    Compact {
        lookup: Vec<(&'data SignatureData, &'data PaperData)>,
        left_indices: Vec<u32>,
        right_indices: Vec<u32>,
    },
}

enum OwnedSignaturePaperLookup<'data> {
    Dense(Vec<Option<(&'data SignatureData, &'data PaperData)>>),
    Compact(Vec<(&'data SignatureData, &'data PaperData)>),
}

fn select_signature_index_layout<IndexIter, BuildIndexIter>(
    signature_count: usize,
    pair_count: usize,
    indices: BuildIndexIter,
    compact_remap_bytes_per_pair: usize,
) -> Result<SignatureIndexLayout, String>
where
    IndexIter: Iterator<Item = u32>,
    BuildIndexIter: Fn() -> IndexIter,
{
    let mut max_index = None::<usize>;
    for raw_index in indices() {
        let index = raw_index as usize;
        if index >= signature_count {
            return Err(format!(
                "pair index out of range: index={} signature_count={}",
                index, signature_count
            ));
        }
        max_index = Some(max_index.map_or(index, |current| current.max(index)));
    }
    let Some(max_index) = max_index else {
        return Ok(SignatureIndexLayout::Dense { max_index: 0 });
    };

    let dense_lookup_bytes = (max_index as u128 + 1)
        * std::mem::size_of::<Option<(&SignatureData, &PaperData)>>() as u128;
    // Direct dense construction uses the lookup itself as the seen-index set.
    // When that lookup is no larger than the two u32 arrays eliminated from
    // the old tuple path (and no larger than a borrowed compact remap), it is
    // both memory-safe against the old path and avoids all hashing.
    let direct_dense_budget_bytes = pair_count as u128 * 2 * std::mem::size_of::<u32>() as u128;
    if dense_lookup_bytes <= direct_dense_budget_bytes {
        return Ok(SignatureIndexLayout::Dense { max_index });
    }

    let used_indices = indices()
        .map(|index| index as usize)
        .collect::<HashSet<_>>();
    let compact_entry_count = used_indices.len() as u128;
    let compact_lookup_bytes =
        compact_entry_count * std::mem::size_of::<(&SignatureData, &PaperData)>() as u128;
    let compact_global_index_bytes = compact_entry_count * std::mem::size_of::<usize>() as u128;
    // hashbrown stores each u32->u32 remap entry plus control/capacity slack.
    // Sixteen bytes/entry is a conservative planning bound for this temporary
    // map on 64-bit targets; the permanent compact lookup is also accounted.
    let compact_hash_map_bytes = compact_entry_count * 16;
    let compact_remap_bytes = pair_count as u128 * compact_remap_bytes_per_pair as u128;
    let compact_peak_bytes = compact_remap_bytes
        + compact_global_index_bytes
        + compact_lookup_bytes.max(compact_hash_map_bytes);

    // Remapping adds a temporary hash map and, for borrowed arrays, two full
    // u32 buffers. Only pay that cost when the compact representation cuts the
    // peak index-layout memory by more than half.
    if compact_peak_bytes.saturating_mul(2) >= dense_lookup_bytes {
        return Ok(SignatureIndexLayout::Dense { max_index });
    }

    let mut global_indices = used_indices.into_iter().collect::<Vec<_>>();
    global_indices.sort_unstable();
    global_indices.shrink_to_fit();
    Ok(SignatureIndexLayout::Compact { global_indices })
}

impl LinkerPairDistanceAccumulator {
    fn new(row_count: usize) -> Self {
        Self {
            counts: vec![0_u32; row_count],
            sums: vec![0.0_f64; row_count],
            mins: vec![f64::INFINITY; row_count],
            top_distances: vec![f64::INFINITY; row_count * 5],
            hard_disallow_pair_count: 0,
        }
    }

    fn merge_from(&mut self, other: Self) {
        for row in 0..self.counts.len() {
            self.counts[row] = self.counts[row].saturating_add(other.counts[row]);
            self.sums[row] += other.sums[row];
            if other.mins[row] < self.mins[row] {
                self.mins[row] = other.mins[row];
            }
            let top_start = row * 5;
            for value in other.top_distances[top_start..top_start + 5].iter() {
                RustFeaturizer::update_top5_distance(
                    &mut self.top_distances[top_start..top_start + 5],
                    *value,
                );
            }
        }
        self.hard_disallow_pair_count = self
            .hard_disallow_pair_count
            .saturating_add(other.hard_disallow_pair_count);
    }
}

impl RustFeaturizer {
    fn constraints_block_upper_triangle_indexed_impl(
        &self,
        py: Python<'_>,
        block_signature_indices: Vec<u32>,
        start_offset: usize,
        max_pairs: Option<usize>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
        suppress_orcid: bool,
        include_coordinates: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<Option<f64>>)> {
        if block_signature_indices.len() <= 1 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            if global_idx >= signature_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "block signature index out of range: index={} signature_count={}",
                    global_idx, signature_count
                )));
            }
        }

        let mut block_lookup: Vec<(&str, &SignatureData)> =
            Vec::with_capacity(block_signature_indices.len());
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            let signature_id = &signature_ids[global_idx];
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            block_lookup.push((signature_id.as_str(), signature));
        }

        let local_pairs = upper_triangle_pairs_for_range(
            block_lookup.len(),
            start_offset,
            max_pairs,
            num_threads,
        )?;
        if local_pairs.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let (left_indices, right_indices) = if include_coordinates {
            (
                local_pairs.iter().map(|(left, _)| *left as u32).collect(),
                local_pairs.iter().map(|(_, right)| *right as u32).collect(),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let values = py.allow_threads(|| {
            let compute = || {
                let keys = self.constraint_keys(&block_lookup);
                local_pairs
                    .par_iter()
                    .map(|(left_idx, right_idx)| {
                        self.constraint_value_from_keys(
                            &keys,
                            *left_idx,
                            *right_idx,
                            low_value,
                            high_value,
                            dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds,
                            suppress_orcid,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        Ok((left_indices, right_indices, values))
    }

    fn cluster_seeds_disallow_index(&self) -> &HashMap<String, HashSet<String>> {
        self.cluster_seeds_disallow_index.get_or_init(|| {
            let mut index: HashMap<String, HashSet<String>> = HashMap::default();
            for (left, right) in self.cluster_seeds_disallow.iter() {
                index
                    .entry(left.clone())
                    .or_insert_with(HashSet::default)
                    .insert(right.clone());
            }
            index
        })
    }

    fn cluster_seeds_disallow_contains(&self, sig_id1: &str, sig_id2: &str) -> bool {
        let (left, right) = canonical_signature_pair_ref(sig_id1, sig_id2);
        self.cluster_seeds_disallow_index()
            .get(left)
            .is_some_and(|rights| rights.contains(right))
    }

    fn featurize_pair_data(
        &self,
        s1: &SignatureData,
        s2: &SignatureData,
        p1: &PaperData,
        p2: &PaperData,
    ) -> [f64; FULL_FEATURE_COUNT] {
        let mut feats = [f64::NAN; FULL_FEATURE_COUNT];

        let first1 = s1.first_normalized();
        let first2 = s2.first_normalized();
        let middle1 = s1.middle.as_deref();
        let middle2 = s2.middle.as_deref();

        feats[columns::FIRST_NAMES_EQUAL] = first_names_equal(first1, first2);
        feats[columns::MIDDLE_INITIALS_OVERLAP] =
            middle_initials_overlap_counts(&s1.middle_initial_counts, &s2.middle_initial_counts);
        feats[columns::MIDDLE_NAMES_EQUAL] = middle_names_equal(middle1, middle2);
        feats[columns::MIDDLE_ONE_MISSING] = middle_one_missing(middle1, middle2);
        feats[columns::SINGLE_CHAR_FIRST] = single_char_first(first1, first2);
        feats[columns::SINGLE_CHAR_MIDDLE] = single_char_middle(middle1, middle2);

        feats[columns::AFFILIATION_OVERLAP] =
            counter_jaccard_data(&s1.affiliations, &s2.affiliations, f64::INFINITY);

        let (email_prefix, email_suffix) =
            match email_pair_parts(s1.email.as_deref(), s2.email.as_deref()) {
                Some(((p1, sfx1), (p2, sfx2))) => (
                    if p1 == p2 { 1.0 } else { 0.0 },
                    if sfx1 == sfx2 { 1.0 } else { 0.0 },
                ),
                None => (f64::NAN, f64::NAN),
            };
        feats[columns::EMAIL_PREFIX_EQUAL] = email_prefix;
        feats[columns::EMAIL_SUFFIX_EQUAL] = email_suffix;

        let coauthors = coauthors::coauthor_pair_features(
            &p1.coauthors,
            &s1.coauthors,
            &p2.coauthors,
            &s2.coauthors,
            &self.name_grams,
        );
        feats[columns::COAUTHOR_OVERLAP] = coauthors.block_overlap;
        feats[columns::COAUTHOR_SIMILARITY] = coauthors.gram_similarity;
        feats[columns::COAUTHOR_MATCH] = coauthors.name_match;

        feats[columns::VENUE_OVERLAP] =
            counter_jaccard_data(&p1.venue_ngrams, &p2.venue_ngrams, f64::INFINITY);
        feats[columns::YEAR_DIFF] = year_diff(p1.year, p2.year);

        feats[columns::TITLE_OVERLAP_WORDS] =
            counter_jaccard_data(&p1.title_words, &p2.title_words, f64::INFINITY);
        feats[columns::TITLE_OVERLAP_CHARS] =
            counter_jaccard_data(&p1.title_chars, &p2.title_chars, f64::INFINITY);

        let english_or_unknown_count = {
            let mut count = 0i64;
            if let Some(l1) = p1.predicted_language.as_deref() {
                if l1 == "en" || l1 == "un" {
                    count += 1;
                }
            }
            if let Some(l2) = p2.predicted_language.as_deref() {
                if l2 == "en" || l2 == "un" {
                    count += 1;
                }
            }
            count
        };

        feats[columns::POSITION_DIFF] = position_diff(s1.position, s2.position);
        feats[columns::ABSTRACT_COUNT] = (p1.has_abstract as i64 + p2.has_abstract as i64) as f64;
        feats[columns::ENGLISH_COUNT] = english_or_unknown_count as f64;
        let same_lang = match (
            p1.predicted_language.as_deref(),
            p2.predicted_language.as_deref(),
        ) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b,
            _ => false,
        };
        feats[columns::SAME_LANGUAGE] = if same_lang { 1.0 } else { 0.0 };
        feats[columns::LANGUAGE_RELIABILITY_MIN] =
            p1.language_reliability.min(p2.language_reliability);

        let counts = compute_name_counts_data(s1.name_counts.as_ref(), s2.name_counts.as_ref());
        feats[columns::FIRST_NAME_COUNT_MIN] = counts[0];
        feats[columns::LAST_FIRST_NAME_COUNT_MIN] = counts[1];
        feats[columns::LAST_NAME_COUNT_MIN] = counts[2];
        feats[columns::LAST_FIRST_INITIAL_COUNT_MIN] = counts[3];
        feats[columns::FIRST_NAME_COUNT_MAX] = counts[4];
        feats[columns::LAST_FIRST_NAME_COUNT_MAX] = counts[5];

        let specter_sim = if english_or_unknown_count == 2 {
            if let (Some(specter_a), Some(specter_b)) = (p1.specter.as_ref(), p2.specter.as_ref()) {
                // Match s2and.featurizer._specter_vector_or_none: all-zero SPECTER
                // vectors are treated as missing, yielding NaN.
                // The Arrow ingest path keeps all-zero rows as real vectors, so the
                // missing-vector check has to live here instead of at the ingest
                // boundary.
                let a_zero = p1
                    .specter_norm
                    .map_or_else(|| specter_a.iter().all(|v| *v == 0.0), |norm| norm == 0.0);
                let b_zero = p2
                    .specter_norm
                    .map_or_else(|| specter_b.iter().all(|v| *v == 0.0), |norm| norm == 0.0);
                if a_zero || b_zero {
                    f64::NAN
                } else {
                    let score = match (p1.specter_norm, p2.specter_norm) {
                        (Some(norm_a), Some(norm_b)) if specter_a.len() == specter_b.len() => {
                            cosine_sim_with_norms(specter_a, norm_a, specter_b, norm_b)
                        }
                        _ => cosine_sim_vec_f32(specter_a, specter_b),
                    };
                    score + 1.0
                }
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        feats[columns::SPECTER_COSINE_SIM] = specter_sim;

        feats[columns::JOURNAL_OVERLAP] =
            counter_jaccard_data(&p1.journal_ngrams, &p2.journal_ngrams, f64::INFINITY);

        let advanced = name_text_features(s1.adv_name_for_features(), s2.adv_name_for_features());
        feats[columns::LEVENSHTEIN] = advanced[0];
        feats[columns::PREFIX] = advanced[1];
        feats[columns::LCS] = advanced[2];
        feats[columns::JARO] = advanced[3];
        feats
    }

    /// The per-pair reference for `constraint_value_from_keys`.
    #[cfg(test)]
    fn constraint_value_from_records(
        &self,
        sig_id1: &str,
        sig_id2: &str,
        s1: &SignatureData,
        s2: &SignatureData,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        suppress_orcid: bool,
    ) -> Option<f64> {
        if self.cluster_seeds_disallow_contains(sig_id1, sig_id2) {
            return Some(self.cluster_seed_disallow_value);
        }

        if !incremental_dont_use_cluster_seeds {
            if let (Some(c1), Some(c2)) = (
                self.cluster_seeds_require.get(sig_id1),
                self.cluster_seeds_require.get(sig_id2),
            ) {
                if c1 == c2 {
                    return Some(self.cluster_seed_require_value);
                }
            }
        }

        if dont_merge_cluster_seeds && !incremental_dont_use_cluster_seeds {
            if let (Some(c1), Some(c2)) = (
                self.cluster_seeds_require.get(sig_id1),
                self.cluster_seeds_require.get(sig_id2),
            ) {
                if c1 != c2 {
                    return Some(self.cluster_seed_disallow_value);
                }
            }
        }

        if !suppress_orcid {
            if let (Some(o1), Some(o2)) = (s1.orcid.as_deref(), s2.orcid.as_deref()) {
                if o1 == o2 {
                    return Some(low_value);
                }
            }
        }

        let last1 = s1.last_normalized.as_deref().unwrap_or("");
        let last2 = s2.last_normalized.as_deref().unwrap_or("");
        if !lasts_equivalent_for_constraint(last1, last2) {
            return Some(high_value);
        }

        let first1 = s1.first_normalized().unwrap_or("");
        let first2 = s2.first_normalized().unwrap_or("");
        if !first1.is_empty() && !first2.is_empty() {
            if let (Some(c1), Some(c2)) = (first1.chars().next(), first2.chars().next()) {
                if c1 != c2 {
                    return Some(high_value);
                }
            }
        }

        if !first_names_name_compatible(first1, first2, &self.name_tuples) {
            return Some(high_value);
        }

        let (middle1_tokens, middle2_tokens) = constraint_middle_tokens(
            first1,
            s1.middle.as_deref().unwrap_or(""),
            first2,
            s2.middle.as_deref().unwrap_or(""),
        );
        if middle_names_conflict(&middle1_tokens, &middle2_tokens) {
            return Some(high_value);
        }
        None
    }

    /// Split the constraint inputs of `entries` once, for pairs among them.
    fn constraint_keys<'a>(
        &'a self,
        entries: &[(&'a str, &'a SignatureData)],
    ) -> ConstraintKeys<'a> {
        let names = entries
            .par_iter()
            .map(|(_, signature)| {
                ConstraintNameKey::new(
                    signature.first_normalized().unwrap_or(""),
                    signature.middle.as_deref().unwrap_or(""),
                    signature.last_normalized.as_deref().unwrap_or(""),
                )
            })
            .collect();
        let requires = if self.cluster_seeds_require.is_empty() {
            vec![None; entries.len()]
        } else {
            entries
                .iter()
                .map(|(signature_id, _)| self.cluster_seeds_require.get(*signature_id))
                .collect()
        };
        let in_disallow = if self.cluster_seeds_disallow.is_empty() {
            vec![false; entries.len()]
        } else {
            let members: HashSet<&str> = self
                .cluster_seeds_disallow
                .iter()
                .flat_map(|(left, right)| [left.as_str(), right.as_str()])
                .collect();
            entries
                .iter()
                .map(|(signature_id, _)| members.contains(signature_id))
                .collect()
        };
        // Intern first names so the name-tuple lookup becomes a search in a
        // short sorted list instead of two string hashes per pair.
        let mut first_index: ahash::AHashMap<&str, u32> = ahash::AHashMap::default();
        let mut distinct_firsts: Vec<&str> = Vec::new();
        let first_ids = entries
            .iter()
            .map(|(_, signature)| {
                let first = signature.first_normalized().unwrap_or("");
                *first_index.entry(first).or_insert_with(|| {
                    distinct_firsts.push(first);
                    (distinct_firsts.len() - 1) as u32
                })
            })
            .collect();
        let first_aliases = distinct_firsts
            .iter()
            .map(|first| {
                let mut aliases: Vec<u32> = self
                    .name_tuples
                    .get(*first)
                    .map(|names| {
                        names
                            .iter()
                            .filter_map(|name| first_index.get(name.as_str()).copied())
                            .collect()
                    })
                    .unwrap_or_default();
                aliases.sort_unstable();
                aliases
            })
            .collect();
        ConstraintKeys {
            signature_ids: entries
                .iter()
                .map(|(signature_id, _)| *signature_id)
                .collect(),
            names,
            orcids: entries
                .iter()
                .map(|(_, signature)| signature.orcid.as_deref())
                .collect(),
            requires,
            in_disallow,
            first_ids,
            first_aliases,
        }
    }

    /// `constraint_value_from_records` for entries `left` and `right` of `keys`.
    fn constraint_value_from_keys(
        &self,
        keys: &ConstraintKeys<'_>,
        left: usize,
        right: usize,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        suppress_orcid: bool,
    ) -> Option<f64> {
        if keys.in_disallow[left]
            && keys.in_disallow[right]
            && self.cluster_seeds_disallow_contains(
                keys.signature_ids[left],
                keys.signature_ids[right],
            )
        {
            return Some(self.cluster_seed_disallow_value);
        }
        if !incremental_dont_use_cluster_seeds {
            if let (Some(c1), Some(c2)) = (keys.requires[left], keys.requires[right]) {
                if c1 == c2 {
                    return Some(self.cluster_seed_require_value);
                }
                if dont_merge_cluster_seeds {
                    return Some(self.cluster_seed_disallow_value);
                }
            }
        }
        if !suppress_orcid {
            if let (Some(o1), Some(o2)) = (keys.orcids[left], keys.orcids[right]) {
                if o1 == o2 {
                    return Some(low_value);
                }
            }
        }
        let first_id_2 = keys.first_ids[right];
        let aliases = &keys.first_aliases[keys.first_ids[left] as usize];
        if constraint_names_conflict(&keys.names[left], &keys.names[right], || {
            aliases.binary_search(&first_id_2).is_ok()
        }) {
            return Some(high_value);
        }
        None
    }

    /// Constraint keys for the signatures that `pairs` index, and each global
    /// index's slot in them (`u32::MAX` when unused).
    fn constraint_keys_for_pairs<'a>(
        &'a self,
        signature_ids: &'a [String],
        pairs: impl Iterator<Item = (u32, u32)>,
    ) -> PyResult<(ConstraintKeys<'a>, Vec<u32>)> {
        let mut slots = vec![u32::MAX; signature_ids.len()];
        let mut entries: Vec<(&'a str, &'a SignatureData)> = Vec::new();
        for (left, right) in pairs {
            for index in [left as usize, right as usize] {
                if slots[index] == u32::MAX {
                    let signature_id = &signature_ids[index];
                    let signature = self.signatures.get(signature_id).ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(signature_id.clone())
                    })?;
                    slots[index] = entries.len() as u32;
                    entries.push((signature_id.as_str(), signature));
                }
            }
        }
        Ok((self.constraint_keys(&entries), slots))
    }

    fn signature_id_order(&self) -> &[String] {
        // A view keeps a subset of the ids over the shared signature map.
        debug_assert!(self.signature_ids.len() <= self.signatures.len());
        self.signature_ids.as_slice()
    }

    fn full_feature_count(&self) -> usize {
        FULL_FEATURE_COUNT
    }

    fn signature_paper_entry(&self, index: usize) -> PyResult<(&SignatureData, &PaperData)> {
        let signature_ids = self.signature_id_order();
        let signature_id = &signature_ids[index];
        let signature = self
            .signatures
            .get(signature_id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
        let paper = self
            .papers
            .get(&signature.paper_id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string()))?;
        Ok((signature, paper))
    }

    fn dense_signature_paper_lookup(
        &self,
        max_index: usize,
        indices: impl Iterator<Item = u32>,
    ) -> PyResult<Vec<Option<(&SignatureData, &PaperData)>>> {
        let mut used = vec![false; max_index + 1];
        let mut any_used = false;
        for raw_index in indices {
            used[raw_index as usize] = true;
            any_used = true;
        }
        if !any_used {
            return Ok(Vec::new());
        }
        // Two string hashes per signature; resolve them on the caller's pool.
        used.par_iter()
            .enumerate()
            .map(|(index, used)| {
                if *used {
                    self.signature_paper_entry(index).map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect()
    }

    fn compact_signature_paper_lookup(
        &self,
        global_indices: &[usize],
    ) -> PyResult<Vec<(&SignatureData, &PaperData)>> {
        global_indices
            .par_iter()
            .map(|index| self.signature_paper_entry(*index))
            .collect()
    }

    fn adaptive_signature_paper_lookup_for_indices(
        &self,
        left_indices: &[u32],
        right_indices: &[u32],
    ) -> PyResult<BorrowedSignaturePaperLookup<'_>> {
        let selection = select_signature_index_layout(
            self.signature_id_order().len(),
            left_indices.len(),
            || left_indices.iter().chain(right_indices.iter()).copied(),
            2 * std::mem::size_of::<u32>(),
        )
        .map_err(pyo3::exceptions::PyIndexError::new_err)?;
        match selection {
            SignatureIndexLayout::Dense { max_index } => Ok(BorrowedSignaturePaperLookup::Dense(
                self.dense_signature_paper_lookup(
                    max_index,
                    left_indices.iter().chain(right_indices.iter()).copied(),
                )?,
            )),
            SignatureIndexLayout::Compact { global_indices } => {
                let global_to_local = global_indices
                    .iter()
                    .enumerate()
                    .map(|(local_index, global_index)| (*global_index as u32, local_index as u32))
                    .collect::<ahash::AHashMap<_, _>>();
                let remap = |index: &u32| {
                    *global_to_local
                        .get(index)
                        .expect("validated used signature index must have a compact index")
                };
                let remapped_left = left_indices.par_iter().map(remap).collect();
                let remapped_right = right_indices.par_iter().map(remap).collect();
                drop(global_to_local);
                let lookup = self.compact_signature_paper_lookup(&global_indices)?;
                Ok(BorrowedSignaturePaperLookup::Compact {
                    lookup,
                    left_indices: remapped_left,
                    right_indices: remapped_right,
                })
            }
        }
    }

    fn adaptive_signature_paper_lookup_for_pair_tuples(
        &self,
        pairs: &mut [(u32, u32)],
    ) -> PyResult<OwnedSignaturePaperLookup<'_>> {
        let selection = select_signature_index_layout(
            self.signature_id_order().len(),
            pairs.len(),
            || pairs.iter().flat_map(|(left, right)| [*left, *right]),
            0,
        )
        .map_err(pyo3::exceptions::PyIndexError::new_err)?;
        match selection {
            SignatureIndexLayout::Dense { max_index } => Ok(OwnedSignaturePaperLookup::Dense(
                self.dense_signature_paper_lookup(
                    max_index,
                    pairs.iter().flat_map(|(left, right)| [*left, *right]),
                )?,
            )),
            SignatureIndexLayout::Compact { global_indices } => {
                let global_to_local = global_indices
                    .iter()
                    .enumerate()
                    .map(|(local_index, global_index)| (*global_index as u32, local_index as u32))
                    .collect::<ahash::AHashMap<_, _>>();
                pairs.par_iter_mut().for_each(|(left, right)| {
                    *left = *global_to_local
                        .get(left)
                        .expect("validated left signature index must have a compact index");
                    *right = *global_to_local
                        .get(right)
                        .expect("validated right signature index must have a compact index");
                });
                drop(global_to_local);
                Ok(OwnedSignaturePaperLookup::Compact(
                    self.compact_signature_paper_lookup(&global_indices)?,
                ))
            }
        }
    }

    fn featurize_pair_index_matrix<'data, Lookup>(
        &'data self,
        pairs: &[(u32, u32)],
        indices: &[usize],
        nan_value: f64,
        lookup: &Lookup,
    ) -> Vec<f64>
    where
        Lookup: Fn(u32) -> (&'data SignatureData, &'data PaperData) + Sync,
    {
        let out_cols = indices.len();
        let mut buffer = vec![0.0_f64; pairs.len() * out_cols];
        buffer
            .par_chunks_mut(out_cols)
            .zip(pairs.par_iter())
            .for_each(|(out_row, (left_idx, right_idx))| {
                let (s1, p1) = lookup(*left_idx);
                let (s2, p2) = lookup(*right_idx);
                let row = self.featurize_pair_data(s1, s2, p1, p2);
                for (dest, idx) in out_row.iter_mut().zip(indices.iter()) {
                    let mut value = row[*idx];
                    if value.is_nan() && !nan_value.is_nan() {
                        value = nan_value;
                    }
                    *dest = value;
                }
            });
        buffer
    }

    fn featurize_pair_index_arrays_matrix<'data, Lookup>(
        &'data self,
        left_indices: &[u32],
        right_indices: &[u32],
        indices: &[usize],
        nan_value: f64,
        lookup: &Lookup,
    ) -> Vec<f64>
    where
        Lookup: Fn(u32) -> (&'data SignatureData, &'data PaperData) + Sync,
    {
        let out_cols = indices.len();
        let mut buffer = vec![0.0_f64; left_indices.len() * out_cols];
        if out_cols == 0 {
            return buffer;
        }
        buffer
            .par_chunks_mut(out_cols)
            .zip(left_indices.par_iter().zip(right_indices.par_iter()))
            .for_each(|(out_row, (left_idx, right_idx))| {
                let (s1, p1) = lookup(*left_idx);
                let (s2, p2) = lookup(*right_idx);
                let row = self.featurize_pair_data(s1, s2, p1, p2);
                for (dest, idx) in out_row.iter_mut().zip(indices.iter()) {
                    let mut value = row[*idx];
                    if value.is_nan() && !nan_value.is_nan() {
                        value = nan_value;
                    }
                    *dest = value;
                }
            });
        buffer
    }

    fn aggregate_pair_index_arrays<'data, Lookup>(
        &'data self,
        left_indices: &[u32],
        right_indices: &[u32],
        owner_row_indices: &[u32],
        row_ranges: Option<&[PairAggregateRowRange]>,
        row_count: usize,
        aggregate_indices: &[usize],
        nan_value: f64,
        lookup: &Lookup,
    ) -> PairAggregateBuffers
    where
        Lookup: Fn(u32) -> (&'data SignatureData, &'data PaperData) + Sync,
    {
        match row_ranges {
            Some(ranges) => self.aggregate_pair_index_arrays_grouped(
                left_indices,
                right_indices,
                ranges,
                row_count,
                aggregate_indices,
                nan_value,
                lookup,
            ),
            None => self.aggregate_pair_index_arrays_sequential(
                left_indices,
                right_indices,
                owner_row_indices,
                row_count,
                aggregate_indices,
                nan_value,
                lookup,
            ),
        }
    }

    /// Featurize index-array pairs tile by tile, score the unlabeled ones, and
    /// keep every pair's aggregate feature values.
    ///
    /// Returns the model distances, `0.0` for labeled pairs, and the raw
    /// aggregate values laid out `aggregate_stride` per pair. Scoring equals
    /// `_predict_pairwise_model_distances`: the main booster's class-0
    /// probability, averaged with the nameless booster's when one is given.
    #[allow(clippy::too_many_arguments)]
    fn score_linker_pairs_tiled<'data, Lookup>(
        &'data self,
        left_indices: &[u32],
        right_indices: &[u32],
        labels: Option<&[f64]>,
        main_indices: &[usize],
        main: &RustLightGBMBooster,
        nameless: Option<(&[usize], &RustLightGBMBooster)>,
        aggregate_indices: &[usize],
        aggregate_stride: usize,
        nan_value: f64,
        lookup: &Lookup,
    ) -> (Vec<f64>, Vec<f64>)
    where
        Lookup: Fn(u32) -> (&'data SignatureData, &'data PaperData) + Sync,
    {
        let pair_count = left_indices.len();
        let tile = FUSED_SCORING_TILE;
        let main_cols = main_indices.len();
        let nameless_cols = nameless.map_or(0, |(indices, _)| indices.len());
        let aggregate_cols = aggregate_indices.len();
        let mut distances = vec![0.0f64; pair_count];
        let mut aggregate_values = vec![0.0f64; pair_count * aggregate_stride];
        distances
            .par_chunks_mut(tile)
            .zip(aggregate_values.par_chunks_mut(tile * aggregate_stride))
            .enumerate()
            .for_each_init(
                || {
                    (
                        vec![0.0f64; tile * main_cols],
                        vec![0.0f64; tile * nameless_cols],
                        Vec::<usize>::with_capacity(tile),
                    )
                },
                |(main_tile, nameless_tile, scored_rows),
                 (tile_index, (distance_tile, aggregate_tile))| {
                    scored_rows.clear();
                    for row in 0..distance_tile.len() {
                        let pair_offset = tile_index * tile + row;
                        let (s1, p1) = lookup(left_indices[pair_offset]);
                        let (s2, p2) = lookup(right_indices[pair_offset]);
                        let features = self.featurize_pair_data(s1, s2, p1, p2);
                        let aggregate_row = &mut aggregate_tile
                            [row * aggregate_stride..row * aggregate_stride + aggregate_cols];
                        for (dest, index) in aggregate_row.iter_mut().zip(aggregate_indices.iter())
                        {
                            *dest = features[*index];
                        }
                        // A labeled pair takes its label as the distance downstream.
                        if labels.is_some_and(|values| !values[pair_offset].is_nan()) {
                            continue;
                        }
                        let scored = scored_rows.len();
                        scored_rows.push(row);
                        let main_row = &mut main_tile[scored * main_cols..(scored + 1) * main_cols];
                        for (dest, index) in main_row.iter_mut().zip(main_indices.iter()) {
                            *dest = model_input(features[*index], nan_value);
                        }
                        if let Some((indices, _)) = nameless {
                            let nameless_row = &mut nameless_tile
                                [scored * nameless_cols..(scored + 1) * nameless_cols];
                            for (dest, index) in nameless_row.iter_mut().zip(indices.iter()) {
                                *dest = model_input(features[*index], nan_value);
                            }
                        }
                    }
                    let scored = scored_rows.len();
                    if scored == 0 {
                        return;
                    }
                    let main_positive =
                        main.probabilities_serial(&main_tile[..scored * main_cols], main_cols);
                    match nameless {
                        Some((_, booster)) => {
                            let nameless_positive = booster.probabilities_serial(
                                &nameless_tile[..scored * nameless_cols],
                                nameless_cols,
                            );
                            for (position, row) in scored_rows.iter().enumerate() {
                                distance_tile[*row] = ((1.0 - main_positive[position])
                                    + (1.0 - nameless_positive[position]))
                                    / 2.0;
                            }
                        }
                        None => {
                            for (position, row) in scored_rows.iter().enumerate() {
                                distance_tile[*row] = 1.0 - main_positive[position];
                            }
                        }
                    }
                },
            );
        (distances, aggregate_values)
    }

    fn pair_aggregate_row_ranges(owner_row_indices: &[u32]) -> Option<Vec<PairAggregateRowRange>> {
        if owner_row_indices.is_empty() {
            return Some(Vec::new());
        }
        let mut ranges = Vec::new();
        let mut start = 0usize;
        let mut previous = owner_row_indices[0];
        for (offset, row_index) in owner_row_indices.iter().enumerate().skip(1) {
            if *row_index < previous {
                return None;
            }
            if *row_index != previous {
                ranges.push(PairAggregateRowRange {
                    row_offset: previous as usize,
                    start,
                    stop: offset,
                });
                start = offset;
                previous = *row_index;
            }
        }
        ranges.push(PairAggregateRowRange {
            row_offset: previous as usize,
            start,
            stop: owner_row_indices.len(),
        });
        Some(ranges)
    }

    fn empty_pair_aggregate_buffers(
        row_count: usize,
        aggregate_cols: usize,
    ) -> PairAggregateBuffers {
        PairAggregateBuffers {
            counts: vec![0_u32; row_count],
            valid_counts: vec![0_u64; row_count * aggregate_cols],
            sums: vec![0.0_f64; row_count * aggregate_cols],
            mins: vec![f64::INFINITY; row_count * aggregate_cols],
            maxs: vec![f64::NEG_INFINITY; row_count * aggregate_cols],
        }
    }

    fn update_pair_aggregate_cell(
        valid_count: &mut u64,
        sum: &mut f64,
        min: &mut f64,
        max: &mut f64,
        mut value: f64,
        nan_value: f64,
    ) {
        if value.is_nan() {
            if nan_value.is_nan() {
                return;
            }
            value = nan_value;
        }
        *valid_count = valid_count.saturating_add(1);
        *sum += value;
        if value < *min {
            *min = value;
        }
        if value > *max {
            *max = value;
        }
    }

    fn aggregate_pair_index_arrays_grouped<'data, Lookup>(
        &'data self,
        left_indices: &[u32],
        right_indices: &[u32],
        row_ranges: &[PairAggregateRowRange],
        row_count: usize,
        aggregate_indices: &[usize],
        nan_value: f64,
        lookup: &Lookup,
    ) -> PairAggregateBuffers
    where
        Lookup: Fn(u32) -> (&'data SignatureData, &'data PaperData) + Sync,
    {
        let aggregate_cols = aggregate_indices.len();
        let mut out = Self::empty_pair_aggregate_buffers(row_count, aggregate_cols);
        if aggregate_cols == 0 {
            for range in row_ranges.iter() {
                out.counts[range.row_offset] =
                    (range.stop - range.start).min(u32::MAX as usize) as u32;
            }
            return out;
        }

        let group_count = row_ranges.len();
        let mut group_counts = vec![0_u32; group_count];
        let mut group_valid_counts = vec![0_u64; group_count * aggregate_cols];
        let mut group_sums = vec![0.0_f64; group_count * aggregate_cols];
        let mut group_mins = vec![f64::INFINITY; group_count * aggregate_cols];
        let mut group_maxs = vec![f64::NEG_INFINITY; group_count * aggregate_cols];
        group_counts
            .par_iter_mut()
            .zip(group_valid_counts.par_chunks_mut(aggregate_cols))
            .zip(group_sums.par_chunks_mut(aggregate_cols))
            .zip(group_mins.par_chunks_mut(aggregate_cols))
            .zip(group_maxs.par_chunks_mut(aggregate_cols))
            .zip(row_ranges.par_iter())
            .for_each(
                |(((((count, valid_counts_row), sums_row), mins_row), maxs_row), range)| {
                    for pair_offset in range.start..range.stop {
                        *count = count.saturating_add(1);
                        let (s1, p1) = lookup(left_indices[pair_offset]);
                        let (s2, p2) = lookup(right_indices[pair_offset]);
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (aggregate_position, feature_index) in
                            aggregate_indices.iter().enumerate()
                        {
                            Self::update_pair_aggregate_cell(
                                &mut valid_counts_row[aggregate_position],
                                &mut sums_row[aggregate_position],
                                &mut mins_row[aggregate_position],
                                &mut maxs_row[aggregate_position],
                                row[*feature_index],
                                nan_value,
                            );
                        }
                    }
                },
            );

        for (group_offset, range) in row_ranges.iter().enumerate() {
            out.counts[range.row_offset] = group_counts[group_offset];
            let source_start = group_offset * aggregate_cols;
            let target_start = range.row_offset * aggregate_cols;
            out.valid_counts[target_start..target_start + aggregate_cols]
                .copy_from_slice(&group_valid_counts[source_start..source_start + aggregate_cols]);
            out.sums[target_start..target_start + aggregate_cols]
                .copy_from_slice(&group_sums[source_start..source_start + aggregate_cols]);
            out.mins[target_start..target_start + aggregate_cols]
                .copy_from_slice(&group_mins[source_start..source_start + aggregate_cols]);
            out.maxs[target_start..target_start + aggregate_cols]
                .copy_from_slice(&group_maxs[source_start..source_start + aggregate_cols]);
        }
        out
    }

    fn aggregate_pair_index_arrays_sequential<'data, Lookup>(
        &'data self,
        left_indices: &[u32],
        right_indices: &[u32],
        owner_row_indices: &[u32],
        row_count: usize,
        aggregate_indices: &[usize],
        nan_value: f64,
        lookup: &Lookup,
    ) -> PairAggregateBuffers
    where
        Lookup: Fn(u32) -> (&'data SignatureData, &'data PaperData) + Sync,
    {
        let aggregate_cols = aggregate_indices.len();
        let mut out = Self::empty_pair_aggregate_buffers(row_count, aggregate_cols);
        if aggregate_cols == 0 {
            for row_index in owner_row_indices.iter() {
                out.counts[*row_index as usize] = out.counts[*row_index as usize].saturating_add(1);
            }
            return out;
        }

        for (pair_offset, row_index) in owner_row_indices.iter().enumerate() {
            let row_offset = *row_index as usize;
            out.counts[row_offset] = out.counts[row_offset].saturating_add(1);
            let aggregate_row_start = row_offset * aggregate_cols;
            let (s1, p1) = lookup(left_indices[pair_offset]);
            let (s2, p2) = lookup(right_indices[pair_offset]);
            let row = self.featurize_pair_data(s1, s2, p1, p2);
            for (aggregate_position, feature_index) in aggregate_indices.iter().enumerate() {
                let stats_index = aggregate_row_start + aggregate_position;
                Self::update_pair_aggregate_cell(
                    &mut out.valid_counts[stats_index],
                    &mut out.sums[stats_index],
                    &mut out.mins[stats_index],
                    &mut out.maxs[stats_index],
                    row[*feature_index],
                    nan_value,
                );
            }
        }
        out
    }

    fn update_top5_distance(row: &mut [f64], value: f64) {
        if value >= row[4] {
            return;
        }
        row[4] = value;
        row.sort_by(|left, right| left.total_cmp(right));
    }
}

/// Raw rows of the signatures and papers one featurizer covers.
pub(crate) struct RawFeaturizerRows {
    pub(crate) signatures: HashMap<String, RawArrowSignature>,
    pub(crate) papers: HashMap<String, RawArrowPaper>,
    pub(crate) authors_by_paper: HashMap<String, Vec<(i64, String)>>,
    pub(crate) specter_by_paper: HashMap<String, arrow::buffer::ScalarBuffer<f32>>,
}

impl RustFeaturizer {
    /// Seed constraints of the selected signatures, read from request sidecars.
    pub(crate) fn read_seed_constraints(
        cluster_seeds_path: Option<&str>,
        cluster_seed_disallows_path: Option<&str>,
        selected_signature_id_set: &HashSet<String>,
    ) -> PyResult<(HashMap<String, ClusterId>, HashSet<(String, String)>)> {
        let mut cluster_seeds_require = HashMap::<String, ClusterId>::default();
        if let Some(path) = cluster_seeds_path {
            let (_component_order, members_by_component) = read_raw_arrow_cluster_seeds(path)?;
            for (component_key, members) in members_by_component {
                for signature_id in members {
                    if selected_signature_id_set.contains(&signature_id) {
                        cluster_seeds_require
                            .insert(signature_id, ClusterId::Str(component_key.clone()));
                    }
                }
            }
        }
        let mut cluster_seeds_disallow = HashSet::<(String, String)>::default();
        if let Some(path) = cluster_seed_disallows_path {
            for (left, right) in read_raw_arrow_cluster_seed_disallows(path)? {
                if selected_signature_id_set.contains(&left)
                    && selected_signature_id_set.contains(&right)
                {
                    cluster_seeds_disallow.insert((left, right));
                }
            }
        }
        Ok((cluster_seeds_require, cluster_seeds_disallow))
    }

    /// Builds a featurizer from raw rows of exactly `signature_ids` and their
    /// papers. `from_arrow_dataset` reads the rows; a planner that already holds
    /// them passes them in, so both build identical featurizers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_raw_rows(
        py: Python<'_>,
        rows: RawFeaturizerRows,
        signature_ids: Vec<String>,
        needed_paper_ids: HashSet<String>,
        cluster_seeds_require: HashMap<String, ClusterId>,
        cluster_seeds_disallow: HashSet<(String, String)>,
        raw_name_counts: RawNameCountMaps,
        name_tuples: Option<&Bound<'_, PyAny>>,
        preprocess: bool,
        cluster_seed_require_value: f64,
        cluster_seed_disallow_value: f64,
        num_threads: Option<usize>,
        use_orcid_id: bool,
    ) -> PyResult<Self> {
        let RawFeaturizerRows {
            signatures: raw_signatures,
            papers: raw_papers,
            authors_by_paper: mut raw_authors_by_paper,
            mut specter_by_paper,
        } = rows;
        let text_module = py.import("s2and.text")?;
        let stop_words = extract_required_string_set(&text_module.getattr("STOPWORDS")?)?;
        let venue_stop_words =
            extract_required_string_set(&text_module.getattr("VENUE_STOP_WORDS")?)?;
        let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
        let affiliation_stopwords = extract_affiliation_stopwords(py)?;
        // Staging, language detection and assembly run on the request's pool:
        // they were serial loops over every signature and paper of the window.
        let (signatures, papers, name_grams) = py.allow_threads(|| {
            install_with_optional_rayon_pool(num_threads, || -> PyResult<_> {
                let needed_paper_ids: Vec<&String> = needed_paper_ids.iter().collect();
                // Patched CLD2 uses thread-local detection state, so titles share the request pool.
                let (detected_languages, (unidecode_char_map, signature_inputs)) = rayon::join(
                    || {
                        needed_paper_ids
                            .par_iter()
                            .map(|paper_id| {
                                raw_papers
                                    .get(*paper_id)
                                    .filter(|paper| paper.predicted_language.is_none())
                                    .map(|paper| detect_language_compat(&paper.title))
                            })
                            .collect::<Vec<_>>()
                    },
                    || {
                        let unidecode_char_map = unidecode_map_for_raw_arrow_inputs(
                            &raw_signatures,
                            &raw_papers,
                            &raw_authors_by_paper,
                        );
                        let signature_inputs = signature_ids
                            .par_iter()
                            .map(|signature_id| -> PyResult<StageSignatureInput> {
                                let raw_signature = raw_signatures.get(signature_id).ok_or_else(|| {
                                    pyo3::exceptions::PyKeyError::new_err(format!(
                                        "Arrow signatures input is missing signature_id '{signature_id}'"
                                    ))
                                })?;
                                let position = raw_signature.position.ok_or_else(|| {
                                    pyo3::exceptions::PyValueError::new_err(format!(
                                        "signatures Arrow author_position is null for signature_id '{signature_id}'"
                                    ))
                                })?;
                                Ok(StageSignatureInput {
                                    sig_id: signature_id.clone(),
                                    paper_id: raw_signature.paper_id.clone(),
                                    raw_first: raw_signature.author_first.clone(),
                                    raw_middle: raw_signature.author_middle.clone(),
                                    raw_last: raw_signature.author_last.clone(),
                                    email: raw_signature.email.clone(),
                                    position,
                                    affiliation_values: raw_signature.affiliations.clone(),
                                    orcid: if use_orcid_id {
                                        raw_signature.orcid.clone()
                                    } else {
                                        None
                                    },
                                })
                            })
                            .collect::<PyResult<Vec<_>>>();
                        (unidecode_char_map, signature_inputs)
                    },
                );
                let signature_inputs = signature_inputs?;

                // Move each paper's authors out first; the map is then read-only.
                let paper_authors: Vec<Vec<(i64, String)>> = needed_paper_ids
                    .iter()
                    .map(|paper_id| raw_authors_by_paper.remove(*paper_id).unwrap_or_default())
                    .collect();
                let paper_inputs = needed_paper_ids
                    .par_iter()
                    .zip(paper_authors.into_par_iter())
                    .zip(detected_languages.into_par_iter())
                    .map(|((paper_id, raw_authors), detected)| -> PyResult<StagePaperInput> {
                        let Some(raw_paper) = raw_papers.get(*paper_id) else {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Arrow signatures reference missing paper_id '{paper_id}'"
                            )));
                        };
                        let (predicted_language, language_reliability) =
                            if raw_paper.predicted_language.is_some() {
                                raw_paper.is_reliable.ok_or_else(|| {
                                    pyo3::exceptions::PyValueError::new_err(format!(
                                        "papers Arrow predicted_language requires is_reliable for paper_id {paper_id:?}"
                                    ))
                                })?;
                                let language_reliability =
                                    raw_paper.language_reliability.ok_or_else(|| {
                                        pyo3::exceptions::PyValueError::new_err(format!(
                                            "papers Arrow predicted_language requires language_reliability for paper_id {paper_id:?}"
                                        ))
                                    })?;
                                (raw_paper.predicted_language.clone(), language_reliability)
                            } else {
                                let audit = detected
                                    .expect("titles without a stored language were detected");
                                (Some(audit.predicted_language), audit.language_reliability)
                            };
                        Ok(StagePaperInput {
                            paper_id: (*paper_id).clone(),
                            raw_title: raw_paper.title.clone(),
                            raw_venue: raw_paper.venue.clone(),
                            raw_journal: raw_paper.journal_name.clone(),
                            raw_authors,
                            year: raw_paper.year.filter(|year| *year > 0),
                            has_abstract: raw_paper.has_abstract,
                            predicted_language,
                            language_reliability,
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;

                let computed_papers = preprocess_stage_papers(
                    &paper_inputs,
                    preprocess,
                    &unidecode_char_map,
                    &stop_words,
                    &venue_stop_words,
                );
                drop(paper_inputs);
                // Paper order does not change the tables: this used to iterate a
                // HashMap, whose order differs between runs.
                let coauthor_tables = CoauthorTables::build(
                    computed_papers
                        .iter()
                        .map(|(paper_id, paper)| (paper_id, paper.authors.as_slice())),
                    preprocess,
                )?;
                let computed_signatures = preprocess_stage_signatures(
                    &signature_inputs,
                    &coauthor_tables,
                    &raw_name_counts,
                    &name_prefixes,
                    &affiliation_stopwords,
                    &unidecode_char_map,
                    preprocess,
                );
                drop(signature_inputs);
                let signatures: HashMap<String, SignatureData> =
                    computed_signatures.into_iter().collect();
                let (mut paper_coauthors, name_grams) = coauthor_tables.into_parts();
                let paper_parts: Vec<_> = computed_papers
                    .into_iter()
                    .map(|(paper_id, paper)| {
                        let coauthors = paper_coauthors
                            .remove(&paper_id)
                            .expect("every preprocessed paper has coauthor aggregates");
                        let specter = specter_by_paper.remove(&paper_id);
                        (paper_id, paper, coauthors, specter)
                    })
                    .collect();
                let papers: HashMap<PaperId, PaperData> = paper_parts
                    .into_par_iter()
                    .map(|(paper_id, paper, coauthors, specter)| {
                        let specter_norm = specter.as_ref().map(|values| vector_norm_f32(values));
                        (
                            paper_id,
                            PaperData {
                                venue_ngrams: paper.venue_ngrams,
                                title_words: paper.title_words,
                                title_chars: paper.title_chars,
                                year: paper.year,
                                has_abstract: paper.has_abstract,
                                predicted_language: paper.predicted_language,
                                language_reliability: paper.language_reliability,
                                journal_ngrams: paper.journal_ngrams,
                                coauthors,
                                specter,
                                specter_norm,
                            },
                        )
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .collect();
                Ok((signatures, papers, name_grams))
            })
        })?;
        let name_tuples = extract_name_tuples_argument(name_tuples)?;

        Ok(RustFeaturizer {
            signatures: Arc::new(signatures),
            signature_ids: Arc::new(signature_ids),
            papers: Arc::new(papers),
            name_grams: Arc::new(name_grams),
            name_tuples: Arc::new(name_tuples),
            cluster_seeds_disallow,
            cluster_seeds_require,
            cluster_seed_require_value,
            cluster_seed_disallow_value,
            name_counts_manifest_sha256: raw_name_counts.manifest_sha256().map(str::to_owned),
            cluster_seeds_disallow_index: OnceLock::new(),
        })
    }
}

#[pymethods]
impl RustFeaturizer {
    #[staticmethod]
    #[pyo3(
        signature = (
            dataset,
            signature_ids,
            name_tuples,
            *,
            cluster_seeds_path = None,
            cluster_seed_disallows_path = None,
            preprocess = true,
            cluster_seed_require_value = 0.0,
            cluster_seed_disallow_value = 10000.0,
            num_threads = None,
            use_orcid_id = true
        )
    )]
    fn from_arrow_dataset(
        py: Python<'_>,
        dataset: PyRef<'_, ArrowDataset>,
        signature_ids: Option<&Bound<'_, PyAny>>,
        name_tuples: Option<&Bound<'_, PyAny>>,
        cluster_seeds_path: Option<String>,
        cluster_seed_disallows_path: Option<String>,
        preprocess: bool,
        cluster_seed_require_value: f64,
        cluster_seed_disallow_value: f64,
        num_threads: Option<usize>,
        use_orcid_id: bool,
    ) -> PyResult<Self> {
        let dataset = dataset.shared();
        let requested_signature_ids = match signature_ids {
            Some(obj) if !obj.is_none() => Some(
                PyIterator::from_object(obj)?
                    .map(|item| item.and_then(|value| value.extract::<String>()))
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            _ => None,
        };
        let keep_signature_ids: Option<HashSet<String>> = requested_signature_ids
            .as_ref()
            .map(|ids| ids.iter().cloned().collect());

        let (raw_signatures, _) =
            py.allow_threads(|| dataset.read_signatures(keep_signature_ids.as_ref(), num_threads))?;
        let mut signature_ids = match requested_signature_ids {
            Some(ids) => ids,
            None => {
                let mut ids = raw_signatures.keys().cloned().collect::<Vec<_>>();
                ids.sort_unstable();
                ids
            }
        };
        let mut seen_signature_ids =
            HashSet::<String>::with_capacity_and_hasher(signature_ids.len(), Default::default());
        signature_ids.retain(|signature_id| seen_signature_ids.insert(signature_id.clone()));
        let missing_signature_ids = signature_ids
            .iter()
            .filter(|signature_id| !raw_signatures.contains_key(*signature_id))
            .take(10)
            .cloned()
            .collect::<Vec<_>>();
        if !missing_signature_ids.is_empty() {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Arrow signatures input is missing requested signature ids: {missing_signature_ids:?}"
            )));
        }
        let selected_signature_id_set = signature_ids.iter().cloned().collect::<HashSet<_>>();
        let needed_paper_ids = signature_ids
            .iter()
            .filter_map(|signature_id| raw_signatures.get(signature_id))
            .map(|signature| signature.paper_id.clone())
            .collect::<HashSet<_>>();
        // The three tables read concurrently; errors keep their serial precedence.
        let (papers_result, (authors_result, specter_result)) = py.allow_threads(|| {
            install_with_optional_rayon_pool(num_threads, || {
                rayon::join(
                    || dataset.read_papers(&needed_paper_ids, num_threads),
                    || {
                        rayon::join(
                            || dataset.read_paper_authors(&needed_paper_ids, num_threads),
                            || {
                                if dataset.has_specter() {
                                    dataset
                                        .read_specter(&needed_paper_ids, num_threads)
                                        .map(|(specter, _)| specter)
                                } else {
                                    Ok(HashMap::default())
                                }
                            },
                        )
                    },
                )
            })
        });
        let (raw_papers, _) = papers_result?;
        let (raw_authors_by_paper, _) = authors_result?;
        let specter_by_paper = specter_result?;
        let (cluster_seeds_require, cluster_seeds_disallow) = Self::read_seed_constraints(
            cluster_seeds_path.as_deref(),
            cluster_seed_disallows_path.as_deref(),
            &selected_signature_id_set,
        )?;
        Self::from_raw_rows(
            py,
            RawFeaturizerRows {
                signatures: raw_signatures,
                papers: raw_papers,
                authors_by_paper: raw_authors_by_paper,
                specter_by_paper,
            },
            signature_ids,
            needed_paper_ids,
            cluster_seeds_require,
            cluster_seeds_disallow,
            dataset.name_counts(),
            name_tuples,
            preprocess,
            cluster_seed_require_value,
            cluster_seed_disallow_value,
            num_threads,
            use_orcid_id,
        )
    }

    #[getter]
    fn name_counts_manifest_sha256(&self) -> Option<&str> {
        self.name_counts_manifest_sha256.as_deref()
    }

    /// View over some of this featurizer's signatures, in the given order, with seed
    /// constraints read from request sidecars exactly as `from_arrow_dataset` reads them.
    ///
    /// Signature and paper preprocessing does not depend on which other signatures are
    /// loaded, so the view featurizes like a fresh build over `signature_ids`. Subblocked
    /// attachment passes use it instead of rebuilding the block's featurizer every pass.
    #[pyo3(signature = (signature_ids, cluster_seeds_path=None, cluster_seed_disallows_path=None))]
    fn view_with_cluster_seed_sidecars(
        &self,
        signature_ids: Vec<String>,
        cluster_seeds_path: Option<String>,
        cluster_seed_disallows_path: Option<String>,
    ) -> PyResult<Self> {
        let mut selected_signature_id_set =
            HashSet::<String>::with_capacity_and_hasher(signature_ids.len(), Default::default());
        let mut ordered_signature_ids = signature_ids;
        ordered_signature_ids
            .retain(|signature_id| selected_signature_id_set.insert(signature_id.clone()));
        let missing_signature_ids = ordered_signature_ids
            .iter()
            .filter(|signature_id| !self.signatures.contains_key(*signature_id))
            .take(10)
            .cloned()
            .collect::<Vec<_>>();
        if !missing_signature_ids.is_empty() {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "featurizer view is missing requested signature ids: {missing_signature_ids:?}"
            )));
        }
        let mut cluster_seeds_require = HashMap::<String, ClusterId>::default();
        if let Some(path) = cluster_seeds_path.as_ref() {
            let (_component_order, members_by_component) = read_raw_arrow_cluster_seeds(path)?;
            for (component_key, members) in members_by_component {
                for signature_id in members {
                    if selected_signature_id_set.contains(&signature_id) {
                        cluster_seeds_require
                            .insert(signature_id, ClusterId::Str(component_key.clone()));
                    }
                }
            }
        }
        let mut cluster_seeds_disallow = HashSet::<(String, String)>::default();
        if let Some(path) = cluster_seed_disallows_path.as_ref() {
            for (left, right) in read_raw_arrow_cluster_seed_disallows(path)? {
                if selected_signature_id_set.contains(&left)
                    && selected_signature_id_set.contains(&right)
                {
                    cluster_seeds_disallow.insert((left, right));
                }
            }
        }
        Ok(Self {
            signatures: Arc::clone(&self.signatures),
            signature_ids: Arc::new(ordered_signature_ids),
            papers: Arc::clone(&self.papers),
            name_grams: Arc::clone(&self.name_grams),
            name_tuples: Arc::clone(&self.name_tuples),
            cluster_seeds_disallow,
            cluster_seeds_require,
            cluster_seed_require_value: self.cluster_seed_require_value,
            cluster_seed_disallow_value: self.cluster_seed_disallow_value,
            name_counts_manifest_sha256: self.name_counts_manifest_sha256.clone(),
            cluster_seeds_disallow_index: OnceLock::new(),
        })
    }

    /// Create a request-owned seed overlay sharing the prepared feature backing.
    fn with_cluster_seeds(
        &self,
        cluster_seeds_require: &Bound<'_, PyAny>,
        cluster_seeds_disallow: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            signatures: Arc::clone(&self.signatures),
            signature_ids: Arc::clone(&self.signature_ids),
            papers: Arc::clone(&self.papers),
            name_grams: Arc::clone(&self.name_grams),
            name_tuples: Arc::clone(&self.name_tuples),
            cluster_seeds_require: extract_cluster_seeds_require(cluster_seeds_require)?,
            cluster_seeds_disallow: extract_pair_set(cluster_seeds_disallow)?,
            cluster_seed_require_value: self.cluster_seed_require_value,
            cluster_seed_disallow_value: self.cluster_seed_disallow_value,
            name_counts_manifest_sha256: self.name_counts_manifest_sha256.clone(),
            cluster_seeds_disallow_index: OnceLock::new(),
        })
    }

    #[pyo3(
        signature = (
            pairs,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None,
            suppress_orcid = false
        )
    )]
    fn get_constraints_matrix_indexed(
        &self,
        py: Python<'_>,
        pairs: Vec<(u32, u32)>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
        suppress_orcid: bool,
    ) -> PyResult<Vec<Option<f64>>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        for (left_idx, right_idx) in pairs.iter() {
            let left = *left_idx as usize;
            let right = *right_idx as usize;
            if left >= signature_count || right >= signature_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "pair index out of range: left={} right={} signature_count={}",
                    left, right, signature_count
                )));
            }
        }

        let values = py.allow_threads(|| {
            let compute = || -> PyResult<Vec<Option<f64>>> {
                let (keys, slots) =
                    self.constraint_keys_for_pairs(signature_ids, pairs.iter().copied())?;
                Ok(pairs
                    .par_iter()
                    .map(|(left_idx, right_idx)| {
                        self.constraint_value_from_keys(
                            &keys,
                            slots[*left_idx as usize] as usize,
                            slots[*right_idx as usize] as usize,
                            low_value,
                            high_value,
                            dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds,
                            suppress_orcid,
                        )
                    })
                    .collect::<Vec<_>>())
            };
            install_with_optional_rayon_pool(num_threads, compute)
        })?;

        Ok(values)
    }

    #[pyo3(
        signature = (
            left_signature_indices,
            right_signature_indices,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None,
            suppress_orcid = false,
            large_integer = 100000.0
        )
    )]
    fn linker_pair_index_arrays_constraint_labels<'py>(
        &self,
        py: Python<'py>,
        left_signature_indices: PyReadonlyArray1<'py, u32>,
        right_signature_indices: PyReadonlyArray1<'py, u32>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
        suppress_orcid: bool,
        large_integer: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let left_indices = left_signature_indices.as_slice()?;
        let right_indices = right_signature_indices.as_slice()?;
        let pair_count = left_indices.len();
        if right_indices.len() != pair_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "left_signature_indices and right_signature_indices must have equal length: left={} right={}",
                left_indices.len(),
                right_indices.len()
            )));
        }

        let signature_ids = self.signature_id_order();
        for (left_idx, right_idx) in left_indices.iter().zip(right_indices.iter()) {
            let left = *left_idx as usize;
            let right = *right_idx as usize;
            if left >= signature_ids.len() || right >= signature_ids.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "pair index out of range: left={} right={} signature_count={}",
                    left,
                    right,
                    signature_ids.len()
                )));
            }
        }

        let labels = py.allow_threads(|| {
            let compute = || -> PyResult<Vec<f64>> {
                let (keys, slots) = self.constraint_keys_for_pairs(
                    signature_ids,
                    left_indices
                        .iter()
                        .copied()
                        .zip(right_indices.iter().copied()),
                )?;
                Ok(left_indices
                    .par_iter()
                    .zip(right_indices.par_iter())
                    .map(|(left_idx, right_idx)| {
                        match self.constraint_value_from_keys(
                            &keys,
                            slots[*left_idx as usize] as usize,
                            slots[*right_idx as usize] as usize,
                            low_value,
                            high_value,
                            dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds,
                            suppress_orcid,
                        ) {
                            Some(value) => value - large_integer,
                            None => f64::NAN,
                        }
                    })
                    .collect::<Vec<f64>>())
            };
            install_with_optional_rayon_pool(num_threads, compute)
        })?;
        Ok(numpy::ndarray::Array1::from_vec(labels).into_pyarray(py))
    }

    #[pyo3(
        signature = (
            row_indices,
            row_count,
            pair_distances,
            pair_labels = None,
            num_threads = None,
            large_integer = 100000.0,
            hard_disallow_distance = 10000.0
        )
    )]
    fn linker_pair_distance_accumulators<'py>(
        &self,
        py: Python<'py>,
        row_indices: PyReadonlyArray1<'py, u32>,
        row_count: usize,
        pair_distances: PyReadonlyArray1<'py, f64>,
        pair_labels: Option<PyReadonlyArray1<'py, f64>>,
        num_threads: Option<usize>,
        large_integer: f64,
        hard_disallow_distance: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        u64,
    )> {
        let owner_row_indices = row_indices.as_slice()?;
        let model_distances = pair_distances.as_slice()?;
        let pair_count = owner_row_indices.len();
        if model_distances.len() != pair_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "row_indices and pair_distances must have equal length: rows={} distances={}",
                owner_row_indices.len(),
                model_distances.len()
            )));
        }
        let labels = match pair_labels.as_ref() {
            Some(values) => {
                let slice = values.as_slice()?;
                if slice.len() != pair_count {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "pair_labels length must match row_indices length: labels={} rows={}",
                        slice.len(),
                        pair_count
                    )));
                }
                Some(slice)
            }
            None => None,
        };
        for row_index in owner_row_indices.iter() {
            let bounded = *row_index as usize;
            if bounded >= row_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "row index out of range: row_index={} row_count={}",
                    bounded, row_count
                )));
            }
        }

        let accumulate_range =
            |start: usize, end: usize| -> Result<LinkerPairDistanceAccumulator, String> {
                let mut accumulator = LinkerPairDistanceAccumulator::new(row_count);
                for pair_offset in start..end {
                    let label = labels.map(|values| values[pair_offset]).unwrap_or(f64::NAN);
                    let value = if label.is_nan() {
                        model_distances[pair_offset]
                    } else {
                        label + large_integer
                    };
                    if value.is_nan() {
                        return Err("pairwise model returned NaN distance".to_string());
                    }
                    let row = owner_row_indices[pair_offset] as usize;
                    accumulator.counts[row] = accumulator.counts[row].saturating_add(1);
                    accumulator.sums[row] += value;
                    if value < accumulator.mins[row] {
                        accumulator.mins[row] = value;
                    }
                    if value >= hard_disallow_distance {
                        accumulator.hard_disallow_pair_count =
                            accumulator.hard_disallow_pair_count.saturating_add(1);
                    }
                    let top_start = row * 5;
                    Self::update_top5_distance(
                        &mut accumulator.top_distances[top_start..top_start + 5],
                        value,
                    );
                }
                Ok(accumulator)
            };

        let accumulator = if num_threads.is_some_and(|threads| threads > 1) && pair_count > 1 {
            py.allow_threads(|| {
                let compute = || {
                    let requested_threads = num_threads.unwrap_or(1).max(1);
                    let shard_count = requested_threads.min(pair_count);
                    let chunk_size = pair_count.div_ceil(shard_count);
                    let partials = (0..pair_count)
                        .step_by(chunk_size)
                        .collect::<Vec<_>>()
                        .into_par_iter()
                        .map(|start| accumulate_range(start, (start + chunk_size).min(pair_count)))
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut merged = LinkerPairDistanceAccumulator::new(row_count);
                    for partial in partials {
                        merged.merge_from(partial);
                    }
                    Ok::<LinkerPairDistanceAccumulator, String>(merged)
                };
                install_with_optional_rayon_pool(num_threads, compute)
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?
        } else {
            accumulate_range(0, pair_count).map_err(pyo3::exceptions::PyValueError::new_err)?
        };

        let top_array =
            numpy::ndarray::Array2::from_shape_vec((row_count, 5), accumulator.top_distances)
                .map_err(|err| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to build top-distance matrix: {}",
                        err
                    ))
                })?;
        Ok((
            numpy::ndarray::Array1::from_vec(accumulator.counts).into_pyarray(py),
            numpy::ndarray::Array1::from_vec(accumulator.sums).into_pyarray(py),
            numpy::ndarray::Array1::from_vec(accumulator.mins).into_pyarray(py),
            top_array.into_pyarray(py),
            accumulator.hard_disallow_pair_count,
        ))
    }

    #[pyo3(
        signature = (
            block_signature_indices,
            start_offset = 0,
            max_pairs = None,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None,
            suppress_orcid = false
        )
    )]
    fn get_constraints_block_upper_triangle_indexed(
        &self,
        py: Python<'_>,
        block_signature_indices: Vec<u32>,
        start_offset: usize,
        max_pairs: Option<usize>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
        suppress_orcid: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<Option<f64>>)> {
        self.constraints_block_upper_triangle_indexed_impl(
            py,
            block_signature_indices,
            start_offset,
            max_pairs,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
            num_threads,
            suppress_orcid,
            true,
        )
    }

    /// Return `value - large_integer` per pair, or NaN when no constraint applies.
    ///
    /// Bulk prediction uses these as model labels. One array avoids a Python float per pair.
    #[pyo3(
        signature = (
            block_signature_indices,
            large_integer,
            start_offset = 0,
            max_pairs = None,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None,
            suppress_orcid = false
        )
    )]
    fn _constraint_labels_block_upper_triangle_indexed<'py>(
        &self,
        py: Python<'py>,
        block_signature_indices: Vec<u32>,
        large_integer: f64,
        start_offset: usize,
        max_pairs: Option<usize>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
        suppress_orcid: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (_, _, values) = self.constraints_block_upper_triangle_indexed_impl(
            py,
            block_signature_indices,
            start_offset,
            max_pairs,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
            num_threads,
            suppress_orcid,
            false,
        )?;
        let labels: Vec<f64> = values
            .into_iter()
            .map(|value| match value {
                Some(value) => value - large_integer,
                None => f64::NAN,
            })
            .collect();
        Ok(numpy::ndarray::Array1::from_vec(labels).into_pyarray(py))
    }

    /// Drop this handle's share of the block data on a background thread.
    ///
    /// Views and other handles keep their shares; the data is freed with the
    /// last one. This handle has no signatures afterwards.
    fn release(&mut self) {
        let dropped = (
            std::mem::replace(&mut self.signatures, Arc::new(HashMap::default())),
            std::mem::replace(&mut self.signature_ids, Arc::new(Vec::new())),
            std::mem::replace(&mut self.papers, Arc::new(HashMap::default())),
            std::mem::replace(&mut self.name_grams, Arc::new(NameGrams::default())),
            std::mem::replace(&mut self.name_tuples, Arc::new(HashMap::default())),
            std::mem::take(&mut self.cluster_seeds_disallow),
            std::mem::take(&mut self.cluster_seeds_require),
            self.cluster_seeds_disallow_index.take(),
        );
        std::thread::spawn(move || drop(dropped));
    }

    fn signature_ids(&self) -> Vec<String> {
        self.signature_id_order().to_vec()
    }

    fn signature_name_counts_present(&self) -> Vec<(String, bool)> {
        self.signature_id_order()
            .iter()
            .filter_map(|signature_id| {
                self.signatures
                    .get(signature_id)
                    .map(|signature| (signature_id.clone(), signature.name_counts.is_some()))
            })
            .collect()
    }

    fn cluster_seeds_require(&self) -> Vec<(String, String)> {
        let mut pairs: Vec<(String, String)> = self
            .cluster_seeds_require
            .iter()
            .map(|(signature_id, cluster_id)| {
                (signature_id.clone(), cluster_id_to_string(cluster_id))
            })
            .collect();
        pairs.sort_by(|left, right| left.0.cmp(&right.0));
        pairs
    }

    #[pyo3(signature = (pairs, selected_indices = None, num_threads = None, nan_value = f64::NAN))]
    fn featurize_pairs_matrix_indexed<'py>(
        &self,
        py: Python<'py>,
        mut pairs: Vec<(u32, u32)>,
        selected_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let row_count = pairs.len();
        let full_cols = self.full_feature_count();
        let indices = resolve_feature_indices("selected_indices", selected_indices, full_cols)?;
        let out_cols = indices.len();
        if row_count == 0 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, out_cols));
            return Ok(empty.into_pyarray(py));
        }

        let lookup = py.allow_threads(|| {
            install_with_optional_rayon_pool(num_threads, || {
                self.adaptive_signature_paper_lookup_for_pair_tuples(&mut pairs)
            })
        })?;

        if out_cols == 0 {
            let empty_cols = numpy::ndarray::Array2::<f64>::zeros((row_count, 0));
            return Ok(empty_cols.into_pyarray(py));
        }

        let out = py.allow_threads(|| {
            let compute = || match &lookup {
                OwnedSignaturePaperLookup::Dense(dense_lookup) => {
                    self.featurize_pair_index_matrix(&pairs, &indices, nan_value, &|index| {
                        dense_lookup[index as usize]
                            .expect("dense signature index was validated before featurization")
                    })
                }
                OwnedSignaturePaperLookup::Compact(compact_lookup) => self
                    .featurize_pair_index_matrix(&pairs, &indices, nan_value, &|index| {
                        compact_lookup[index as usize]
                    }),
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let array =
            numpy::ndarray::Array2::from_shape_vec((row_count, out_cols), out).map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build output matrix: {}",
                    err
                ))
            })?;
        Ok(array.into_pyarray(py))
    }

    /// Features of chosen rows of one chunk of a block's upper triangle.
    ///
    /// Equals `featurize_pairs_matrix_indexed` over the pairs of those rows, in
    /// the order given, without building the pairs as Python tuples. `scored_rows`
    /// are offsets into the chunk `[start_offset, start_offset + max_pairs)`.
    #[pyo3(signature = (
        block_signature_indices,
        start_offset,
        max_pairs,
        scored_rows,
        selected_indices = None,
        num_threads = None,
        nan_value = f64::NAN
    ))]
    #[allow(clippy::too_many_arguments)]
    fn featurize_block_upper_triangle_rows_indexed<'py>(
        &self,
        py: Python<'py>,
        block_signature_indices: Vec<u32>,
        start_offset: usize,
        max_pairs: Option<usize>,
        scored_rows: PyReadonlyArray1<'py, i64>,
        selected_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let block_size = block_signature_indices.len();
        let chunk_pair_count = upper_triangle_range_len(block_size, start_offset, max_pairs)?;
        // Each scored row's pair comes from its offset; listing the whole chunk
        // only to pick a few rows cost a pass over millions of pairs.
        let pairs = scored_rows
            .as_slice()?
            .iter()
            .map(|&row| {
                usize::try_from(row)
                    .ok()
                    .filter(|row| *row < chunk_pair_count)
                    .map(|row| {
                        let (left, right) = upper_triangle_pair_at(block_size, start_offset + row);
                        (
                            block_signature_indices[left],
                            block_signature_indices[right],
                        )
                    })
                    .ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err(format!(
                            "scored row {} is outside the chunk of {} pairs",
                            row, chunk_pair_count
                        ))
                    })
            })
            .collect::<PyResult<Vec<(u32, u32)>>>()?;
        self.featurize_pairs_matrix_indexed(py, pairs, selected_indices, num_threads, nan_value)
    }

    /// Class-0 probabilities of chosen rows of one chunk of a block's upper
    /// triangle, featurized and scored tile by tile on the pool.
    ///
    /// Equals featurizing the rows with `main_indices` and `nameless_indices`,
    /// scoring each matrix with its booster, and averaging the class-0
    /// probabilities as `((1 - p_main) + (1 - p_nameless)) / 2`, or taking
    /// `1 - p_main` without a nameless booster. No feature matrix is built,
    /// so the batch path's copies, column gathers and page faults are gone.
    /// `scored_rows` are offsets into the chunk; None scores every row.
    #[pyo3(signature = (
        block_signature_indices,
        start_offset,
        max_pairs,
        scored_rows,
        main_indices,
        main_booster,
        nameless_indices = None,
        nameless_booster = None,
        num_threads = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn score_block_upper_triangle_rows_indexed<'py>(
        &self,
        py: Python<'py>,
        block_signature_indices: Vec<u32>,
        start_offset: usize,
        max_pairs: Option<usize>,
        scored_rows: Option<PyReadonlyArray1<'py, i64>>,
        main_indices: Vec<usize>,
        main_booster: PyRef<'py, RustLightGBMBooster>,
        nameless_indices: Option<Vec<usize>>,
        nameless_booster: Option<PyRef<'py, RustLightGBMBooster>>,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let tile = FUSED_SCORING_TILE;

        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            if global_idx >= signature_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "block signature index out of range: index={} signature_count={}",
                    global_idx, signature_count
                )));
            }
        }
        let full_cols = self.full_feature_count();
        let main_indices = resolve_feature_indices("main_indices", Some(main_indices), full_cols)?;
        let nameless_indices = match nameless_indices {
            Some(indices) => Some(resolve_feature_indices(
                "nameless_indices",
                Some(indices),
                full_cols,
            )?),
            None => None,
        };
        if nameless_indices.is_some() != nameless_booster.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "nameless_indices and nameless_booster must be given together",
            ));
        }
        let main: &RustLightGBMBooster = &main_booster;
        let nameless: Option<&RustLightGBMBooster> = nameless_booster.as_deref();
        if main.feature_width() != main_indices.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "main booster expects {} features, got {} main_indices",
                main.feature_width(),
                main_indices.len()
            )));
        }
        let nameless_cols = nameless_indices.as_ref().map_or(0, Vec::len);
        if let Some(booster) = nameless {
            if booster.feature_width() != nameless_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "nameless booster expects {} features, got {} nameless_indices",
                    booster.feature_width(),
                    nameless_cols
                )));
            }
        }

        let mut block_lookup: Vec<(&SignatureData, &PaperData)> =
            Vec::with_capacity(block_signature_indices.len());
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            let signature_id = &signature_ids[global_idx];
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            block_lookup.push((signature, paper));
        }
        let block_size = block_lookup.len();
        let chunk_pair_count = upper_triangle_range_len(block_size, start_offset, max_pairs)?;
        let scored_rows: Option<&[i64]> = match scored_rows.as_ref() {
            Some(rows) => Some(rows.as_slice()?),
            None => None,
        };
        if let Some(rows) = scored_rows {
            if let Some(row) = rows
                .iter()
                .find(|row| usize::try_from(**row).map_or(true, |row| row >= chunk_pair_count))
            {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "scored row {} is outside the chunk of {} pairs",
                    row, chunk_pair_count
                )));
            }
        }
        let row_count = scored_rows.map_or(chunk_pair_count, <[i64]>::len);
        let main_cols = main_indices.len();
        let mut out = vec![0.0f64; row_count];
        py.allow_threads(|| {
            install_with_optional_rayon_pool(num_threads, || {
                out.par_chunks_mut(tile).enumerate().for_each_init(
                    || {
                        (
                            vec![0.0f64; tile * main_cols],
                            vec![0.0f64; tile * nameless_cols],
                        )
                    },
                    |(main_tile, nameless_tile), (tile_index, out_tile)| {
                        let rows = out_tile.len();
                        // Contiguous rows step from the tile's first pair; chosen
                        // rows take each pair from its offset.
                        let mut next_pair = scored_rows.is_none().then(|| {
                            upper_triangle_pair_at(block_size, start_offset + tile_index * tile)
                        });
                        for row in 0..rows {
                            let (left, right) = match scored_rows {
                                Some(scored) => upper_triangle_pair_at(
                                    block_size,
                                    start_offset + scored[tile_index * tile + row] as usize,
                                ),
                                None => {
                                    let current = next_pair.expect("stepping holds a pair");
                                    next_pair = Some(if current.1 + 1 < block_size {
                                        (current.0, current.1 + 1)
                                    } else {
                                        (current.0 + 1, current.0 + 2)
                                    });
                                    current
                                }
                            };
                            let (s1, p1) = block_lookup[left];
                            let (s2, p2) = block_lookup[right];
                            let features = self.featurize_pair_data(s1, s2, p1, p2);
                            let main_row = &mut main_tile[row * main_cols..(row + 1) * main_cols];
                            for (dest, index) in main_row.iter_mut().zip(main_indices.iter()) {
                                *dest = features[*index];
                            }
                            if let Some(indices) = nameless_indices.as_ref() {
                                let nameless_row = &mut nameless_tile
                                    [row * nameless_cols..(row + 1) * nameless_cols];
                                for (dest, index) in nameless_row.iter_mut().zip(indices.iter()) {
                                    *dest = features[*index];
                                }
                            }
                        }
                        let main_positive =
                            main.probabilities_serial(&main_tile[..rows * main_cols], main_cols);
                        match nameless {
                            Some(booster) => {
                                let nameless_positive = booster.probabilities_serial(
                                    &nameless_tile[..rows * nameless_cols],
                                    nameless_cols,
                                );
                                for row in 0..rows {
                                    out_tile[row] = ((1.0 - main_positive[row])
                                        + (1.0 - nameless_positive[row]))
                                        / 2.0;
                                }
                            }
                            None => {
                                for row in 0..rows {
                                    out_tile[row] = 1.0 - main_positive[row];
                                }
                            }
                        }
                    },
                );
            })
        });
        Ok(out.into_pyarray(py))
    }

    #[pyo3(
        signature = (
            left_signature_indices,
            right_signature_indices,
            row_indices,
            row_count,
            matrix_indices = None,
            aggregate_indices = None,
            num_threads = None,
            nan_value = f64::NAN,
            aggregate_nan_value = None,
            emit_matrix = true
        )
    )]
    fn linker_pair_index_arrays_and_aggregate_stats<'py>(
        &self,
        py: Python<'py>,
        left_signature_indices: PyReadonlyArray1<'py, u32>,
        right_signature_indices: PyReadonlyArray1<'py, u32>,
        row_indices: PyReadonlyArray1<'py, u32>,
        row_count: usize,
        matrix_indices: Option<Vec<usize>>,
        aggregate_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
        aggregate_nan_value: Option<f64>,
        emit_matrix: bool,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray2<u64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let left_indices = left_signature_indices.as_slice()?;
        let right_indices = right_signature_indices.as_slice()?;
        let owner_row_indices = row_indices.as_slice()?;
        let pair_count = left_indices.len();
        if right_indices.len() != pair_count || owner_row_indices.len() != pair_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "left_signature_indices, right_signature_indices, and row_indices must have equal length: left={} right={} rows={}",
                left_indices.len(),
                right_indices.len(),
                owner_row_indices.len()
            )));
        }

        let lookup = py.allow_threads(|| {
            install_with_optional_rayon_pool(num_threads, || {
                self.adaptive_signature_paper_lookup_for_indices(left_indices, right_indices)
            })
        })?;
        for row_index in owner_row_indices.iter() {
            let bounded = *row_index as usize;
            if bounded >= row_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "row index out of range: row_index={} row_count={}",
                    bounded, row_count
                )));
            }
        }

        let full_cols = self.full_feature_count();
        if !emit_matrix {
            let resolved_aggregate_indices =
                resolve_feature_indices("aggregate_indices", aggregate_indices, full_cols)?;
            let resolved_aggregate_nan_value = aggregate_nan_value.unwrap_or(nan_value);
            let row_ranges = Self::pair_aggregate_row_ranges(owner_row_indices);
            let aggregate_cols = resolved_aggregate_indices.len();
            let aggregate_buffers = py.allow_threads(|| {
                let compute = || match &lookup {
                    BorrowedSignaturePaperLookup::Dense(dense_lookup) => self
                        .aggregate_pair_index_arrays(
                            left_indices,
                            right_indices,
                            owner_row_indices,
                            row_ranges.as_deref(),
                            row_count,
                            &resolved_aggregate_indices,
                            resolved_aggregate_nan_value,
                            &|index| {
                                dense_lookup[index as usize].expect(
                                    "dense signature index was validated before aggregation",
                                )
                            },
                        ),
                    BorrowedSignaturePaperLookup::Compact {
                        lookup: compact_lookup,
                        left_indices: compact_left,
                        right_indices: compact_right,
                    } => self.aggregate_pair_index_arrays(
                        compact_left,
                        compact_right,
                        owner_row_indices,
                        row_ranges.as_deref(),
                        row_count,
                        &resolved_aggregate_indices,
                        resolved_aggregate_nan_value,
                        &|index| compact_lookup[index as usize],
                    ),
                };
                install_with_optional_rayon_pool(num_threads, compute)
            });
            let matrix_array = numpy::ndarray::Array2::<f64>::zeros((pair_count, 0));
            let valid_counts_array = numpy::ndarray::Array2::from_shape_vec(
                (row_count, aggregate_cols),
                aggregate_buffers.valid_counts,
            )
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate valid counts matrix: {}",
                    err
                ))
            })?;
            let sums_array = numpy::ndarray::Array2::from_shape_vec(
                (row_count, aggregate_cols),
                aggregate_buffers.sums,
            )
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate sums matrix: {}",
                    err
                ))
            })?;
            let mins_array = numpy::ndarray::Array2::from_shape_vec(
                (row_count, aggregate_cols),
                aggregate_buffers.mins,
            )
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate mins matrix: {}",
                    err
                ))
            })?;
            let maxs_array = numpy::ndarray::Array2::from_shape_vec(
                (row_count, aggregate_cols),
                aggregate_buffers.maxs,
            )
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate maxs matrix: {}",
                    err
                ))
            })?;
            return Ok((
                matrix_array.into_pyarray(py),
                numpy::ndarray::Array1::from_vec(aggregate_buffers.counts).into_pyarray(py),
                valid_counts_array.into_pyarray(py),
                sums_array.into_pyarray(py),
                mins_array.into_pyarray(py),
                maxs_array.into_pyarray(py),
            ));
        }

        let index_selection =
            resolve_matrix_aggregate_indices(matrix_indices, aggregate_indices, full_cols)?;
        let resolved_matrix_indices = index_selection.matrix_indices;
        let resolved_aggregate_indices = index_selection.aggregate_indices;
        let aggregate_matrix_positions = index_selection.aggregate_matrix_positions;

        let out_cols = resolved_matrix_indices.len();
        let aggregate_cols = resolved_aggregate_indices.len();
        let resolved_aggregate_nan_value = aggregate_nan_value.unwrap_or(nan_value);
        // Preserve raw NaNs until aggregate statistics have applied their own
        // missing-value policy. The returned matrix policy is applied below.
        let mut matrix_buffer = py.allow_threads(|| {
            let compute = || match &lookup {
                BorrowedSignaturePaperLookup::Dense(dense_lookup) => self
                    .featurize_pair_index_arrays_matrix(
                        left_indices,
                        right_indices,
                        &resolved_matrix_indices,
                        f64::NAN,
                        &|index| {
                            dense_lookup[index as usize]
                                .expect("dense signature index was validated before featurization")
                        },
                    ),
                BorrowedSignaturePaperLookup::Compact {
                    lookup: compact_lookup,
                    left_indices: compact_left,
                    right_indices: compact_right,
                } => self.featurize_pair_index_arrays_matrix(
                    compact_left,
                    compact_right,
                    &resolved_matrix_indices,
                    f64::NAN,
                    &|index| compact_lookup[index as usize],
                ),
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let mut counts = vec![0_u32; row_count];
        let mut valid_counts = vec![0_u64; row_count * aggregate_cols];
        let mut sums = vec![0.0_f64; row_count * aggregate_cols];
        let mut mins = vec![f64::INFINITY; row_count * aggregate_cols];
        let mut maxs = vec![f64::NEG_INFINITY; row_count * aggregate_cols];
        if aggregate_cols > 0 {
            for (pair_offset, row_index) in owner_row_indices.iter().enumerate() {
                let row_offset = *row_index as usize;
                counts[row_offset] = counts[row_offset].saturating_add(1);
                let matrix_row_start = pair_offset * out_cols;
                let aggregate_row_start = row_offset * aggregate_cols;
                for (aggregate_position, matrix_position) in
                    aggregate_matrix_positions.iter().enumerate()
                {
                    let stats_index = aggregate_row_start + aggregate_position;
                    Self::update_pair_aggregate_cell(
                        &mut valid_counts[stats_index],
                        &mut sums[stats_index],
                        &mut mins[stats_index],
                        &mut maxs[stats_index],
                        matrix_buffer[matrix_row_start + *matrix_position],
                        resolved_aggregate_nan_value,
                    );
                }
            }
        } else {
            for row_index in owner_row_indices.iter() {
                counts[*row_index as usize] = counts[*row_index as usize].saturating_add(1);
            }
        }

        if !nan_value.is_nan() {
            py.allow_threads(|| {
                matrix_buffer.par_iter_mut().for_each(|value| {
                    if value.is_nan() {
                        *value = nan_value;
                    }
                });
            });
        }

        let matrix_array =
            numpy::ndarray::Array2::from_shape_vec((pair_count, out_cols), matrix_buffer).map_err(
                |err| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to build pair feature matrix: {}",
                        err
                    ))
                },
            )?;
        let valid_counts_array =
            numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), valid_counts)
                .map_err(|err| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to build aggregate valid counts matrix: {}",
                        err
                    ))
                })?;
        let sums_array = numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), sums)
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate sums matrix: {}",
                    err
                ))
            })?;
        let mins_array = numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), mins)
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate mins matrix: {}",
                    err
                ))
            })?;
        let maxs_array = numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), maxs)
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate maxs matrix: {}",
                    err
                ))
            })?;
        Ok((
            matrix_array.into_pyarray(py),
            numpy::ndarray::Array1::from_vec(counts).into_pyarray(py),
            valid_counts_array.into_pyarray(py),
            sums_array.into_pyarray(py),
            mins_array.into_pyarray(py),
            maxs_array.into_pyarray(py),
        ))
    }

    /// Score index-array pairs with native boosters and build row aggregates
    /// in one fused pass.
    ///
    /// Returns `(distances, counts, valid_counts, sums, mins, maxs)`. The
    /// distances equal `_predict_pairwise_model_distances` over the matrix
    /// that `linker_pair_index_arrays_and_aggregate_stats` returns for the
    /// model columns with `nan_value` applied; a pair with a label is not
    /// scored and reads `0.0`. The aggregates equal that call's: the raw
    /// feature values walked in pair order under `aggregate_nan_value`.
    #[pyo3(signature = (
        left_signature_indices,
        right_signature_indices,
        row_indices,
        row_count,
        main_indices,
        main_booster,
        nameless_indices = None,
        nameless_booster = None,
        aggregate_indices = None,
        pair_labels = None,
        num_threads = None,
        nan_value = f64::NAN,
        aggregate_nan_value = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn score_linker_pair_index_arrays_and_aggregate_stats<'py>(
        &self,
        py: Python<'py>,
        left_signature_indices: PyReadonlyArray1<'py, u32>,
        right_signature_indices: PyReadonlyArray1<'py, u32>,
        row_indices: PyReadonlyArray1<'py, u32>,
        row_count: usize,
        main_indices: Vec<usize>,
        main_booster: PyRef<'py, RustLightGBMBooster>,
        nameless_indices: Option<Vec<usize>>,
        nameless_booster: Option<PyRef<'py, RustLightGBMBooster>>,
        aggregate_indices: Option<Vec<usize>>,
        pair_labels: Option<PyReadonlyArray1<'py, f64>>,
        num_threads: Option<usize>,
        nan_value: f64,
        aggregate_nan_value: Option<f64>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray2<u64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let left_indices = left_signature_indices.as_slice()?;
        let right_indices = right_signature_indices.as_slice()?;
        let owner_row_indices = row_indices.as_slice()?;
        let pair_count = left_indices.len();
        if right_indices.len() != pair_count || owner_row_indices.len() != pair_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "left_signature_indices, right_signature_indices, and row_indices must have equal length: left={} right={} rows={}",
                left_indices.len(),
                right_indices.len(),
                owner_row_indices.len()
            )));
        }
        let labels = match pair_labels.as_ref() {
            Some(values) => {
                let slice = values.as_slice()?;
                if slice.len() != pair_count {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "pair_labels length must match the pair count: labels={} pairs={}",
                        slice.len(),
                        pair_count
                    )));
                }
                Some(slice)
            }
            None => None,
        };
        for row_index in owner_row_indices.iter() {
            let bounded = *row_index as usize;
            if bounded >= row_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "row index out of range: row_index={} row_count={}",
                    bounded, row_count
                )));
            }
        }
        let full_cols = self.full_feature_count();
        let main_indices = resolve_feature_indices("main_indices", Some(main_indices), full_cols)?;
        let nameless_indices = match nameless_indices {
            Some(indices) => Some(resolve_feature_indices(
                "nameless_indices",
                Some(indices),
                full_cols,
            )?),
            None => None,
        };
        if nameless_indices.is_some() != nameless_booster.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "nameless_indices and nameless_booster must be given together",
            ));
        }
        let main: &RustLightGBMBooster = &main_booster;
        if main.feature_width() != main_indices.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "main booster expects {} features, got {} main_indices",
                main.feature_width(),
                main_indices.len()
            )));
        }
        let nameless = nameless_indices.as_deref().zip(nameless_booster.as_deref());
        if let Some((indices, booster)) = nameless {
            if booster.feature_width() != indices.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "nameless booster expects {} features, got {} nameless_indices",
                    booster.feature_width(),
                    indices.len()
                )));
            }
        }
        let aggregate_indices =
            resolve_feature_indices("aggregate_indices", aggregate_indices, full_cols)?;
        let aggregate_cols = aggregate_indices.len();
        // The pool splits the aggregate buffer by tile, so it keeps one value
        // per pair even when no aggregates are asked for.
        let aggregate_stride = aggregate_cols.max(1);
        let resolved_aggregate_nan_value = aggregate_nan_value.unwrap_or(nan_value);

        let lookup = py.allow_threads(|| {
            install_with_optional_rayon_pool(num_threads, || {
                self.adaptive_signature_paper_lookup_for_indices(left_indices, right_indices)
            })
        })?;
        let (distances, aggregates) = py.allow_threads(|| {
            let score = || match &lookup {
                BorrowedSignaturePaperLookup::Dense(dense_lookup) => self.score_linker_pairs_tiled(
                    left_indices,
                    right_indices,
                    labels,
                    &main_indices,
                    main,
                    nameless,
                    &aggregate_indices,
                    aggregate_stride,
                    nan_value,
                    &|index| {
                        dense_lookup[index as usize]
                            .expect("dense signature index was validated before scoring")
                    },
                ),
                BorrowedSignaturePaperLookup::Compact {
                    lookup: compact_lookup,
                    left_indices: compact_left,
                    right_indices: compact_right,
                } => self.score_linker_pairs_tiled(
                    compact_left,
                    compact_right,
                    labels,
                    &main_indices,
                    main,
                    nameless,
                    &aggregate_indices,
                    aggregate_stride,
                    nan_value,
                    &|index| compact_lookup[index as usize],
                ),
            };
            let (distances, aggregate_values) =
                install_with_optional_rayon_pool(num_threads, score);
            // The aggregates walk the pairs in order, as the matrix path does.
            let mut out = Self::empty_pair_aggregate_buffers(row_count, aggregate_cols);
            for (pair_offset, row_index) in owner_row_indices.iter().enumerate() {
                let row_offset = *row_index as usize;
                out.counts[row_offset] = out.counts[row_offset].saturating_add(1);
                let value_start = pair_offset * aggregate_stride;
                let stats_start = row_offset * aggregate_cols;
                for position in 0..aggregate_cols {
                    Self::update_pair_aggregate_cell(
                        &mut out.valid_counts[stats_start + position],
                        &mut out.sums[stats_start + position],
                        &mut out.mins[stats_start + position],
                        &mut out.maxs[stats_start + position],
                        aggregate_values[value_start + position],
                        resolved_aggregate_nan_value,
                    );
                }
            }
            (distances, out)
        });
        Ok((
            distances.into_pyarray(py),
            aggregates.counts.into_pyarray(py),
            rows_matrix(
                py,
                row_count,
                aggregate_cols,
                aggregates.valid_counts,
                "aggregate valid counts",
            )?,
            rows_matrix(
                py,
                row_count,
                aggregate_cols,
                aggregates.sums,
                "aggregate sums",
            )?,
            rows_matrix(
                py,
                row_count,
                aggregate_cols,
                aggregates.mins,
                "aggregate mins",
            )?,
            rows_matrix(
                py,
                row_count,
                aggregate_cols,
                aggregates.maxs,
                "aggregate maxs",
            )?,
        ))
    }

    #[pyo3(
        signature = (
            block_signature_indices,
            start_offset = 0,
            max_pairs = None,
            selected_indices = None,
            num_threads = None,
            nan_value = f64::NAN
        )
    )]
    fn featurize_block_upper_triangle_matrix_indexed<'py>(
        &self,
        py: Python<'py>,
        block_signature_indices: Vec<u32>,
        start_offset: usize,
        max_pairs: Option<usize>,
        selected_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            if global_idx >= signature_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "block signature index out of range: index={} signature_count={}",
                    global_idx, signature_count
                )));
            }
        }
        let full_cols = self.full_feature_count();
        let indices = resolve_feature_indices("selected_indices", selected_indices, full_cols)?;
        let out_cols = indices.len();

        let mut block_lookup: Vec<(&SignatureData, &PaperData)> =
            Vec::with_capacity(block_signature_indices.len());
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            let signature_id = &signature_ids[global_idx];
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            block_lookup.push((signature, paper));
        }

        let local_pairs = upper_triangle_pairs_for_range(
            block_lookup.len(),
            start_offset,
            max_pairs,
            num_threads,
        )?;
        let row_count = local_pairs.len();
        if row_count == 0 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, out_cols));
            return Ok(empty.into_pyarray(py));
        }

        if out_cols == 0 {
            let empty_cols = numpy::ndarray::Array2::<f64>::zeros((row_count, 0));
            return Ok(empty_cols.into_pyarray(py));
        }

        let out = py.allow_threads(|| {
            let compute = || {
                let mut buffer = vec![0.0_f64; row_count * out_cols];
                buffer
                    .par_chunks_mut(out_cols)
                    .zip(local_pairs.par_iter())
                    .for_each(|(out_row, (left_idx, right_idx))| {
                        let (s1, p1) = block_lookup[*left_idx];
                        let (s2, p2) = block_lookup[*right_idx];
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (dest, idx) in out_row.iter_mut().zip(indices.iter()) {
                            let mut value = row[*idx];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            *dest = value;
                        }
                    });
                buffer
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let array =
            numpy::ndarray::Array2::from_shape_vec((row_count, out_cols), out).map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build output matrix: {}",
                    err
                ))
            })?;
        Ok(array.into_pyarray(py))
    }
}
