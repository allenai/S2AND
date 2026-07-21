use super::*;

#[pyclass]
#[derive(Clone)]
pub(crate) struct RustFeaturizer {
    signatures: HashMap<String, SignatureData>,
    signature_ids: Vec<String>,
    papers: HashMap<PaperId, PaperData>,
    name_tuples: HashMap<String, HashSet<String>>,
    cluster_seeds_disallow: HashSet<(String, String)>,
    cluster_seeds_require: HashMap<String, ClusterId>,
    cluster_seed_require_value: f64,
    cluster_seed_disallow_value: f64,
    name_counts_provenance_binding: Option<NameCountsProvenanceBinding>,
    cluster_seeds_disallow_index: OnceLock<HashMap<String, HashSet<String>>>,
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
    fn cluster_seeds_disallow_index(&self) -> &HashMap<String, HashSet<String>> {
        self.cluster_seeds_disallow_index.get_or_init(|| {
            let mut index: HashMap<String, HashSet<String>> = HashMap::new();
            for (left, right) in self.cluster_seeds_disallow.iter() {
                index
                    .entry(left.clone())
                    .or_insert_with(HashSet::new)
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
        let mut feat_i: usize = 0;
        macro_rules! push_feat {
            ($value:expr) => {{
                feats[feat_i] = $value;
                feat_i += 1;
            }};
        }

        let first1 = s1.first_without_apostrophe();
        let first2 = s2.first_without_apostrophe();
        let middle1 = s1.middle.as_deref();
        let middle2 = s2.middle.as_deref();

        push_feat!(first_names_equal(first1, first2));
        push_feat!(middle_initials_overlap(middle1, middle2));
        push_feat!(middle_names_equal(middle1, middle2));
        push_feat!(middle_one_missing(middle1, middle2));
        push_feat!(single_char_first(first1, first2));
        push_feat!(single_char_middle(middle1, middle2));

        push_feat!(counter_jaccard_data(
            &s1.affiliations,
            &s2.affiliations,
            f64::INFINITY,
        ));

        let (email_prefix, email_suffix) =
            match email_pair_parts(s1.email.as_deref(), s2.email.as_deref()) {
                Some(((p1, sfx1), (p2, sfx2))) => (
                    if p1 == p2 { 1.0 } else { 0.0 },
                    if sfx1 == sfx2 { 1.0 } else { 0.0 },
                ),
                None => (f64::NAN, f64::NAN),
            };
        push_feat!(email_prefix);
        push_feat!(email_suffix);

        push_feat!(set_jaccard_data(&s1.coauthor_blocks, &s2.coauthor_blocks));
        push_feat!(counter_jaccard_data(
            &s1.coauthor_ngrams,
            &s2.coauthor_ngrams,
            5000.0,
        ));
        push_feat!(set_jaccard_data(&s1.coauthors, &s2.coauthors));

        push_feat!(counter_jaccard_data(
            &p1.venue_ngrams,
            &p2.venue_ngrams,
            f64::INFINITY,
        ));
        push_feat!(year_diff(p1.year, p2.year));

        push_feat!(counter_jaccard_data(
            &p1.title_words,
            &p2.title_words,
            f64::INFINITY,
        ));
        push_feat!(counter_jaccard_data(
            &p1.title_chars,
            &p2.title_chars,
            f64::INFINITY,
        ));

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

        push_feat!(position_diff(s1.position, s2.position));
        push_feat!((p1.has_abstract as i64 + p2.has_abstract as i64) as f64);
        push_feat!(english_or_unknown_count as f64);
        let same_lang = match (
            p1.predicted_language.as_deref(),
            p2.predicted_language.as_deref(),
        ) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b,
            _ => false,
        };
        push_feat!(if same_lang { 1.0 } else { 0.0 });
        push_feat!(p1.language_reliability.min(p2.language_reliability));

        let counts = compute_name_counts_data(s1.name_counts.as_ref(), s2.name_counts.as_ref());
        for value in counts.iter() {
            push_feat!(*value);
        }

        let specter_sim = if english_or_unknown_count == 2 {
            if let (Some(specter_a), Some(specter_b)) = (p1.specter.as_ref(), p2.specter.as_ref()) {
                // Match s2and.featurizer behavior at s2and/featurizer.py:1223,1227:
                // all-zero SPECTER vectors are treated as missing, yielding NaN.
                // The Arrow ingest path keeps all-zero rows as real vectors (per the
                // Current Decisions table in docs/work_plan.md), so the missing-vector
                // check has to live here instead of at the ingest boundary.
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
        push_feat!(specter_sim);

        push_feat!(counter_jaccard_data(
            &p1.journal_ngrams,
            &p2.journal_ngrams,
            f64::INFINITY,
        ));

        let advanced = name_text_features(s1.adv_name_for_features(), s2.adv_name_for_features());
        for value in advanced.iter() {
            push_feat!(*value);
        }

        debug_assert_eq!(feat_i, FULL_FEATURE_COUNT);
        feats
    }

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

        let first1 = s1.first_without_apostrophe().unwrap_or("");
        let first2 = s2.first_without_apostrophe().unwrap_or("");
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

        let middle1_str = s1.middle.as_deref().unwrap_or("");
        let middle1_tokens: Vec<&str> = middle1_str.split_whitespace().collect();
        if !middle1_tokens.is_empty() {
            let middle2_str = s2.middle.as_deref().unwrap_or("");
            let middle2_tokens: Vec<&str> = middle2_str.split_whitespace().collect();
            if !middle2_tokens.is_empty() {
                let middle1_set: HashSet<&str> = middle1_tokens.iter().copied().collect();
                let middle2_set: HashSet<&str> = middle2_tokens.iter().copied().collect();
                let mut overlapping_affixes: HashSet<&str> = HashSet::new();
                for token in middle1_set.intersection(&middle2_set) {
                    if is_dropped_affix(token) {
                        overlapping_affixes.insert(*token);
                    }
                }

                let middle_1_all: Vec<&str> = middle1_tokens
                    .iter()
                    .copied()
                    .filter(|w| !w.is_empty() && !overlapping_affixes.contains(w))
                    .collect();
                let middle_2_all: Vec<&str> = middle2_tokens
                    .iter()
                    .copied()
                    .filter(|w| !w.is_empty() && !overlapping_affixes.contains(w))
                    .collect();

                let middle_1_words: HashSet<&str> = middle_1_all
                    .iter()
                    .copied()
                    .filter(|w| py_len(w) > 1)
                    .collect();
                let middle_2_words: HashSet<&str> = middle_2_all
                    .iter()
                    .copied()
                    .filter(|w| py_len(w) > 1)
                    .collect();

                let mut middle_1_firsts: HashSet<char> = HashSet::new();
                for word in middle_1_all.iter() {
                    if let Some(ch) = word.chars().next() {
                        middle_1_firsts.insert(ch);
                    }
                }
                let mut middle_2_firsts: HashSet<char> = HashSet::new();
                for word in middle_2_all.iter() {
                    if let Some(ch) = word.chars().next() {
                        middle_2_firsts.insert(ch);
                    }
                }

                let conflicting_initials = !middle_1_firsts.is_empty()
                    && !middle_2_firsts.is_empty()
                    && middle_1_firsts.is_disjoint(&middle_2_firsts);

                let mut middle_1_chars: HashSet<char> = HashSet::new();
                for word in middle_1_words.iter() {
                    for ch in word.chars() {
                        middle_1_chars.insert(ch);
                    }
                }
                let mut middle_2_chars: HashSet<char> = HashSet::new();
                for word in middle_2_words.iter() {
                    for ch in word.chars() {
                        middle_2_chars.insert(ch);
                    }
                }

                let conflicting_full_names = !middle_1_words.is_empty()
                    && !middle_2_words.is_empty()
                    && middle_1_words.is_disjoint(&middle_2_words)
                    && middle_1_chars != middle_2_chars;

                if conflicting_initials || conflicting_full_names {
                    return Some(high_value);
                }
            }
        }
        None
    }

    fn signature_id_order(&self) -> &[String] {
        debug_assert_eq!(self.signature_ids.len(), self.signatures.len());
        self.signature_ids.as_slice()
    }

    fn full_feature_count(&self) -> usize {
        FULL_FEATURE_COUNT
    }

    fn signature_lookup(&self) -> PyResult<Vec<&SignatureData>> {
        let signature_ids = self.signature_id_order();
        let mut lookup: Vec<&SignatureData> = Vec::with_capacity(signature_ids.len());
        for signature_id in signature_ids.iter() {
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            lookup.push(signature);
        }
        Ok(lookup)
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
        let mut indices = indices.peekable();
        if indices.peek().is_none() {
            return Ok(Vec::new());
        }
        let mut lookup = vec![None; max_index + 1];
        for raw_index in indices {
            let index = raw_index as usize;
            if lookup[index].is_none() {
                lookup[index] = Some(self.signature_paper_entry(index)?);
            }
        }
        Ok(lookup)
    }

    fn compact_signature_paper_lookup(
        &self,
        global_indices: &[usize],
    ) -> PyResult<Vec<(&SignatureData, &PaperData)>> {
        global_indices
            .iter()
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
                    .collect::<HashMap<_, _>>();
                let remap = |index: &u32| {
                    *global_to_local
                        .get(index)
                        .expect("validated used signature index must have a compact index")
                };
                let remapped_left = left_indices.iter().map(remap).collect();
                let remapped_right = right_indices.iter().map(remap).collect();
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
                    .collect::<HashMap<_, _>>();
                for (left, right) in pairs.iter_mut() {
                    *left = *global_to_local
                        .get(left)
                        .expect("validated left signature index must have a compact index");
                    *right = *global_to_local
                        .get(right)
                        .expect("validated right signature index must have a compact index");
                }
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
                            let mut value = row[*feature_index];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            if value.is_nan() {
                                continue;
                            }
                            valid_counts_row[aggregate_position] =
                                valid_counts_row[aggregate_position].saturating_add(1);
                            sums_row[aggregate_position] += value;
                            if value < mins_row[aggregate_position] {
                                mins_row[aggregate_position] = value;
                            }
                            if value > maxs_row[aggregate_position] {
                                maxs_row[aggregate_position] = value;
                            }
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
                let mut value = row[*feature_index];
                if value.is_nan() && !nan_value.is_nan() {
                    value = nan_value;
                }
                if value.is_nan() {
                    continue;
                }
                let stats_index = aggregate_row_start + aggregate_position;
                out.valid_counts[stats_index] = out.valid_counts[stats_index].saturating_add(1);
                out.sums[stats_index] += value;
                if value < out.mins[stats_index] {
                    out.mins[stats_index] = value;
                }
                if value > out.maxs[stats_index] {
                    out.maxs[stats_index] = value;
                }
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

#[pymethods]
impl RustFeaturizer {
    #[staticmethod]
    #[pyo3(
        signature = (
            paths,
            signature_ids = None,
            name_tuples = None,
            preprocess = true,
            cluster_seed_require_value = 0.0,
            cluster_seed_disallow_value = 10000.0,
            num_threads = None
        )
    )]
    fn from_arrow_paths(
        py: Python<'_>,
        paths: &Bound<'_, PyAny>,
        signature_ids: Option<&Bound<'_, PyAny>>,
        name_tuples: Option<&Bound<'_, PyAny>>,
        preprocess: bool,
        cluster_seed_require_value: f64,
        cluster_seed_disallow_value: f64,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let signatures_path =
            extract_path_mapping_string(paths, "signatures", true)?.ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("missing signatures Arrow path")
            })?;
        let papers_path = extract_path_mapping_string(paths, "papers", true)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing papers Arrow path"))?;
        let paper_authors_path = extract_path_mapping_string(paths, "paper_authors", true)?
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("missing paper_authors Arrow path")
            })?;
        let cluster_seeds_path = extract_path_mapping_string(paths, "cluster_seeds", false)?;
        let cluster_seed_disallows_path =
            extract_path_mapping_string(paths, "cluster_seed_disallows", false)?;
        let specter_path = extract_path_mapping_string(paths, "specter", false)?;
        let name_counts_index_path = extract_name_counts_index_path(paths)?;
        let signatures_batch_index_path =
            extract_path_mapping_string(paths, "signatures_batch_index", false)?;
        let papers_batch_index_path =
            extract_path_mapping_string(paths, "papers_batch_index", false)?;
        let paper_authors_batch_index_path =
            extract_path_mapping_string(paths, "paper_authors_batch_index", false)?;
        let specter_batch_index_path =
            extract_path_mapping_string(paths, "specter_batch_index", false)?;

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

        let (raw_signatures, _) = read_raw_arrow_signatures_with_optional_index(
            &signatures_path,
            signatures_batch_index_path.as_deref(),
            keep_signature_ids.as_ref(),
        )?;
        let mut signature_ids = match requested_signature_ids {
            Some(ids) => ids,
            None => {
                let mut ids = raw_signatures.keys().cloned().collect::<Vec<_>>();
                ids.sort_unstable();
                ids
            }
        };
        let mut seen_signature_ids = HashSet::<String>::with_capacity(signature_ids.len());
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
        let (raw_papers, _) = read_raw_arrow_papers_with_optional_index(
            &papers_path,
            papers_batch_index_path.as_deref(),
            &needed_paper_ids,
        )?;
        let (mut raw_authors_by_paper, _) = read_raw_arrow_paper_authors_with_optional_index(
            &paper_authors_path,
            paper_authors_batch_index_path.as_deref(),
            &needed_paper_ids,
        )?;
        let specter_by_paper = match specter_path.as_ref() {
            Some(path) => {
                read_raw_arrow_specter_with_optional_index(
                    path,
                    specter_batch_index_path.as_deref(),
                    &needed_paper_ids,
                )?
                .0
            }
            None => HashMap::new(),
        };
        let mut cluster_seeds_require = HashMap::<String, ClusterId>::new();
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
        let mut cluster_seeds_disallow = HashSet::<(String, String)>::new();
        if let Some(path) = cluster_seed_disallows_path.as_ref() {
            for (left, right) in read_raw_arrow_cluster_seed_disallows(path)? {
                if selected_signature_id_set.contains(&left)
                    && selected_signature_id_set.contains(&right)
                {
                    cluster_seeds_disallow.insert((left, right));
                }
            }
        }
        let text_module = py.import("s2and.text")?;
        let stop_words = extract_required_string_set(&text_module.getattr("STOPWORDS")?)?;
        let venue_stop_words =
            extract_required_string_set(&text_module.getattr("VENUE_STOP_WORDS")?)?;
        let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
        let affiliation_stopwords = extract_affiliation_stopwords(py)?;
        let raw_name_counts = match name_counts_index_path.as_ref() {
            Some(path) => read_raw_name_counts_index(path)?,
            None => RawNameCountMaps::default(),
        };
        let mut language_detector: Option<LanguageDetectorCompat> = None;

        let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
        ensure_unidecode_for_raw_arrow_inputs(
            &raw_signatures,
            &raw_papers,
            &raw_authors_by_paper,
            &mut unidecode_char_map,
        )?;
        let mut signature_inputs = Vec::<StageSignatureInput>::with_capacity(signature_ids.len());
        for signature_id in signature_ids.iter() {
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
            signature_inputs.push(StageSignatureInput {
                sig_id: signature_id.clone(),
                paper_id: raw_signature.paper_id.clone(),
                raw_first: raw_signature.author_first.clone(),
                raw_middle: raw_signature.author_middle.clone(),
                raw_last: raw_signature.author_last.clone(),
                email: raw_signature.email.clone(),
                position,
                affiliation_values: raw_signature.affiliations.clone(),
                orcid: raw_signature.orcid.clone(),
            });
        }

        let mut paper_inputs = Vec::<StagePaperInput>::with_capacity(needed_paper_ids.len());
        for paper_id in needed_paper_ids.iter() {
            let Some(raw_paper) = raw_papers.get(paper_id) else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Arrow signatures reference missing paper_id '{paper_id}'"
                )));
            };
            let raw_authors = raw_authors_by_paper.remove(paper_id).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Arrow paper_authors are missing rows for paper_id '{paper_id}'"
                ))
            })?;
            let (is_reliable, predicted_language, language_reliability) = if raw_paper
                .predicted_language
                .is_some()
            {
                let is_reliable = raw_paper.is_reliable.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "papers Arrow predicted_language requires is_reliable for paper_id {paper_id:?}"
                    ))
                })?;
                let language_reliability = raw_paper.language_reliability.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "papers Arrow predicted_language requires language_reliability for paper_id {paper_id:?}"
                    ))
                })?;
                (
                    is_reliable,
                    raw_paper.predicted_language.clone(),
                    language_reliability,
                )
            } else {
                if language_detector.is_none() {
                    language_detector = Some(LanguageDetectorCompat::new(py)?);
                }
                let detector = language_detector
                    .as_ref()
                    .expect("language detector was just initialized");
                let (reliable, _is_english, language, reliability) =
                    detector.detect(&raw_paper.title)?;
                (reliable, Some(language), reliability)
            };
            paper_inputs.push(StagePaperInput {
                paper_id: paper_id.clone(),
                raw_title: raw_paper.title.clone(),
                raw_venue: raw_paper.venue.clone(),
                raw_journal: raw_paper.journal_name.clone(),
                raw_authors,
                year: raw_paper.year.filter(|year| *year > 0),
                has_abstract: !raw_paper.abstract_text.is_empty(),
                predicted_language,
                is_reliable,
                language_reliability,
            });
        }

        let computed_papers = py.allow_threads(|| {
            let compute = || {
                preprocess_stage_papers(
                    &paper_inputs,
                    preprocess,
                    &unidecode_char_map,
                    &stop_words,
                    &venue_stop_words,
                )
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        let mut preprocessed_papers: HashMap<PaperId, StagePaperPreprocessed> =
            HashMap::with_capacity(computed_papers.len());
        for (paper_id, preprocessed) in computed_papers {
            preprocessed_papers.insert(paper_id, preprocessed);
        }
        let computed_signatures = py.allow_threads(|| {
            let compute = || {
                preprocess_stage_signatures(
                    &signature_inputs,
                    &preprocessed_papers,
                    &raw_name_counts,
                    &name_prefixes,
                    &affiliation_stopwords,
                    &unidecode_char_map,
                    preprocess,
                )
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        let mut signatures: HashMap<String, SignatureData> =
            HashMap::with_capacity(computed_signatures.len());
        for (sig_id, signature) in computed_signatures {
            signatures.insert(sig_id, signature);
        }
        let mut papers: HashMap<PaperId, PaperData> =
            HashMap::with_capacity(preprocessed_papers.len());
        for (paper_id, paper) in preprocessed_papers.into_iter() {
            let specter = specter_by_paper.get(&paper_id).cloned();
            let specter_norm = specter.as_ref().map(|values| vector_norm_f32(values));
            papers.insert(
                paper_id,
                PaperData {
                    venue_ngrams: paper.venue_ngrams,
                    title_words: paper.title_words,
                    title_chars: paper.title_chars,
                    year: paper.year,
                    has_abstract: paper.has_abstract,
                    predicted_language: paper.predicted_language,
                    is_reliable: paper.is_reliable,
                    language_reliability: paper.language_reliability,
                    journal_ngrams: paper.journal_ngrams,
                    specter,
                    specter_norm,
                },
            );
        }
        let name_tuples = extract_name_tuples_argument(py, name_tuples)?;

        Ok(RustFeaturizer {
            signatures,
            signature_ids,
            papers,
            name_tuples,
            cluster_seeds_disallow,
            cluster_seeds_require,
            cluster_seed_require_value,
            cluster_seed_disallow_value,
            name_counts_provenance_binding: raw_name_counts.provenance_binding().cloned(),
            cluster_seeds_disallow_index: OnceLock::new(),
        })
    }

    #[getter]
    fn name_counts_provenance_binding(&self) -> Option<(String, String, String, String)> {
        self.name_counts_provenance_binding.as_ref().map(|binding| {
            (
                binding.generation_id.clone(),
                binding.pickle_sha256.clone(),
                binding.source_snapshot_id.clone(),
                binding.selected_rows_sha256.clone(),
            )
        })
    }

    fn update_cluster_seeds(
        &mut self,
        cluster_seeds_require: &Bound<'_, PyAny>,
        cluster_seeds_disallow: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.cluster_seeds_require = extract_cluster_seeds_require(cluster_seeds_require)?;
        self.cluster_seeds_disallow = extract_pair_set(cluster_seeds_disallow)?;
        self.cluster_seeds_disallow_index = OnceLock::new();
        Ok(())
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

        let mut lookup: Vec<(&String, &SignatureData)> = Vec::with_capacity(signature_count);
        for signature_id in signature_ids.iter() {
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            lookup.push((signature_id, signature));
        }

        let values = py.allow_threads(|| {
            let compute = || {
                pairs
                    .par_iter()
                    .map(|(left_idx, right_idx)| {
                        let (left_id, s1) = lookup[*left_idx as usize];
                        let (right_id, s2) = lookup[*right_idx as usize];
                        self.constraint_value_from_records(
                            left_id,
                            right_id,
                            s1,
                            s2,
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

        let lookup = self.signature_lookup()?;
        let labels = py.allow_threads(|| {
            let compute = || {
                left_indices
                    .par_iter()
                    .zip(right_indices.par_iter())
                    .map(|(left_idx, right_idx)| {
                        let left = *left_idx as usize;
                        let right = *right_idx as usize;
                        let sig_id1 = signature_ids[left].as_str();
                        let sig_id2 = signature_ids[right].as_str();
                        let s1 = lookup[left];
                        let s2 = lookup[right];
                        match self.constraint_value_from_records(
                            sig_id1,
                            sig_id2,
                            s1,
                            s2,
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
                    .collect::<Vec<f64>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        Ok(numpy::ndarray::Array1::from_vec(labels).to_pyarray(py))
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
            numpy::ndarray::Array1::from_vec(accumulator.counts).to_pyarray(py),
            numpy::ndarray::Array1::from_vec(accumulator.sums).to_pyarray(py),
            numpy::ndarray::Array1::from_vec(accumulator.mins).to_pyarray(py),
            top_array.to_pyarray(py),
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

        let mut block_lookup: Vec<(&String, &SignatureData)> =
            Vec::with_capacity(block_signature_indices.len());
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            let signature_id = &signature_ids[global_idx];
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            block_lookup.push((signature_id, signature));
        }

        let local_pairs =
            upper_triangle_pairs_for_range(block_lookup.len(), start_offset, max_pairs)?;
        if local_pairs.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let left_indices: Vec<u32> = local_pairs.iter().map(|(left, _)| *left as u32).collect();
        let right_indices: Vec<u32> = local_pairs.iter().map(|(_, right)| *right as u32).collect();
        let values = py.allow_threads(|| {
            let compute = || {
                local_pairs
                    .par_iter()
                    .map(|(left_idx, right_idx)| {
                        let (left_id, s1) = block_lookup[*left_idx];
                        let (right_id, s2) = block_lookup[*right_idx];
                        self.constraint_value_from_records(
                            left_id,
                            right_id,
                            s1,
                            s2,
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

    fn signature_ids(&self) -> Vec<String> {
        self.signature_id_order().to_vec()
    }

    fn signature_rule_metadata(&self) -> Vec<(String, Option<String>, Option<String>)> {
        self.signature_id_order()
            .iter()
            .filter_map(|signature_id| {
                self.signatures.get(signature_id).map(|signature| {
                    (
                        signature_id.clone(),
                        signature.first_without_apostrophe().map(str::to_owned),
                        signature.orcid.clone(),
                    )
                })
            })
            .collect()
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

    fn update_signature_name_counts(&mut self, signatures: &Bound<'_, PyAny>) -> PyResult<usize> {
        let signatures_dict = signatures.downcast::<PyDict>()?;
        let mut updated = 0usize;
        for (sig_id_obj, sig_obj) in signatures_dict.iter() {
            let sig_id: String = sig_id_obj.extract()?;
            let Some(signature) = self.signatures.get_mut(&sig_id) else {
                continue;
            };
            let counts_obj = sig_obj.getattr("author_info_name_counts")?;
            let counts = extract_name_counts_data(&counts_obj)?;
            if counts.is_some() {
                signature.name_counts = counts;
                updated += 1;
            }
        }
        Ok(updated)
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
        if row_count == 0 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

        let lookup = self.adaptive_signature_paper_lookup_for_pair_tuples(&mut pairs)?;

        let full_cols = self.full_feature_count();
        let indices = resolve_feature_indices("selected_indices", selected_indices, full_cols)?;
        let out_cols = indices.len();
        if out_cols == 0 {
            let empty_cols = numpy::ndarray::Array2::<f64>::zeros((row_count, 0));
            return Ok(empty_cols.to_pyarray(py));
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
        Ok(array.to_pyarray(py))
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

        let lookup =
            self.adaptive_signature_paper_lookup_for_indices(left_indices, right_indices)?;
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
                matrix_array.to_pyarray(py),
                numpy::ndarray::Array1::from_vec(aggregate_buffers.counts).to_pyarray(py),
                valid_counts_array.to_pyarray(py),
                sums_array.to_pyarray(py),
                mins_array.to_pyarray(py),
                maxs_array.to_pyarray(py),
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
        let matrix_buffer = py.allow_threads(|| {
            let compute = || match &lookup {
                BorrowedSignaturePaperLookup::Dense(dense_lookup) => self
                    .featurize_pair_index_arrays_matrix(
                        left_indices,
                        right_indices,
                        &resolved_matrix_indices,
                        nan_value,
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
                    nan_value,
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
                    let mut value = matrix_buffer[matrix_row_start + *matrix_position];
                    if value.is_nan() {
                        if resolved_aggregate_nan_value.is_nan() {
                            continue;
                        }
                        value = resolved_aggregate_nan_value;
                    }
                    let stats_index = aggregate_row_start + aggregate_position;
                    valid_counts[stats_index] = valid_counts[stats_index].saturating_add(1);
                    sums[stats_index] += value;
                    if value < mins[stats_index] {
                        mins[stats_index] = value;
                    }
                    if value > maxs[stats_index] {
                        maxs[stats_index] = value;
                    }
                }
            }
        } else {
            for row_index in owner_row_indices.iter() {
                counts[*row_index as usize] = counts[*row_index as usize].saturating_add(1);
            }
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
            matrix_array.to_pyarray(py),
            numpy::ndarray::Array1::from_vec(counts).to_pyarray(py),
            valid_counts_array.to_pyarray(py),
            sums_array.to_pyarray(py),
            mins_array.to_pyarray(py),
            maxs_array.to_pyarray(py),
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
        if block_signature_indices.len() <= 1 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
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

        let local_pairs =
            upper_triangle_pairs_for_range(block_lookup.len(), start_offset, max_pairs)?;
        let row_count = local_pairs.len();
        if row_count == 0 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

        let full_cols = self.full_feature_count();
        let indices = resolve_feature_indices("selected_indices", selected_indices, full_cols)?;
        let out_cols = indices.len();
        if out_cols == 0 {
            let empty_cols = numpy::ndarray::Array2::<f64>::zeros((row_count, 0));
            return Ok(empty_cols.to_pyarray(py));
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
        Ok(array.to_pyarray(py))
    }
}
