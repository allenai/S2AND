//! Coauthor pair features computed from per-paper author aggregates.
//!
//! A signature's coauthors are its paper's authors at every other position.
//! Three pair features compare two signatures' coauthors: the Jaccard of their
//! name sets (`coauthor_match`), of their name blocks (`coauthor_overlap`), and
//! a Jaccard of the character 2-4-gram counts of their names, with the
//! denominator capped at 5000 (`coauthor_similarity`).
//!
//! Materializing those per signature costs O(authors) allocations per signature
//! and O(authors^2) per paper, and INSPIRE papers have up to ~3,200 authors.
//! Instead each paper stores its whole author list once: sorted interned name
//! ids, sorted interned block ids and summed gram counts. Each signature stores
//! only its own author's name and which of its name and block no other author
//! shares. A pair's features subtract each side's own author from its paper's
//! aggregates. The results are exact: names and blocks are interned by string
//! equality, gram keys are injective, and every count is an integer summed
//! without rounding.

use crate::HashMap;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::text_compat::compute_block_compat;
use crate::PaperId;

/// Denominator cap of the coauthor-similarity Jaccard (Python `counter_jaccard(..., denominator_max=5000)`).
const GRAM_SIMILARITY_DENOMINATOR_MAX: f64 = 5000.0;

/// Largest accepted coauthor n-gram total for one paper.
/// The check keeps the stored `u32` total from truncation. The value keeps the
/// input range of the former f32 counters; counts are now integers and
/// similarities use f64.
const MAX_PAPER_GRAM_TOTAL: u64 = 1 << 24;

/// One paper's authors, aggregated once for every signature on it.
#[derive(Clone, Default)]
pub(crate) struct PaperCoauthors {
    /// Sorted distinct interned author names.
    names: Vec<u32>,
    /// Sorted distinct interned name blocks (`compute_block_compat` of each name).
    blocks: Vec<u32>,
    /// Character 2-4-gram counts summed over all author names, sorted by gram
    /// key. Empty when n-gram features are not preprocessed.
    grams: Vec<(u32, u32)>,
    gram_total: u32,
}

/// A signature's own author on its paper, which its coauthors exclude.
#[derive(Clone, Copy, Default)]
pub(crate) struct SignatureCoauthors {
    /// Own author's name id; `None` when the paper lists no author at the
    /// signature's position.
    own_name: Option<u32>,
    /// Own name id when no other author on the paper has that name.
    removed_name: Option<u32>,
    /// Own block id when no other author on the paper has that block.
    removed_block: Option<u32>,
}

/// Character 2-4-gram counts of each interned author name, sorted by gram key.
#[derive(Default)]
pub(crate) struct NameGrams {
    offsets: Vec<u32>,
    entries: Vec<(u32, u32)>,
}

impl NameGrams {
    fn empty(name_count: usize) -> Self {
        Self {
            offsets: vec![0; name_count + 1],
            entries: Vec::new(),
        }
    }

    fn build(names: &[&str]) -> PyResult<Self> {
        let per_name = names
            .par_iter()
            .map(|name| name_grams(name))
            .collect::<PyResult<Vec<_>>>()?;
        let mut offsets = Vec::with_capacity(per_name.len() + 1);
        offsets.push(0u32);
        let mut entries = Vec::with_capacity(per_name.iter().map(Vec::len).sum());
        for grams in per_name {
            entries.extend(grams);
            offsets.push(u32::try_from(entries.len()).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("coauthor n-gram table exceeds u32 offsets")
            })?);
        }
        Ok(Self { offsets, entries })
    }

    fn of(&self, name: Option<u32>) -> &[(u32, u32)] {
        match name {
            Some(name) => {
                let name = name as usize;
                &self.entries[self.offsets[name] as usize..self.offsets[name + 1] as usize]
            }
            None => &[],
        }
    }
}

/// Key of a gram of 2-4 lowercase ASCII letters: its bytes, big-endian.
///
/// Injective because letters are nonzero bytes, so the key's highest nonzero
/// byte fixes the gram's length.
fn gram_key(gram: &[u8]) -> u32 {
    gram.iter()
        .fold(0u32, |key, byte| (key << 8) | u32::from(*byte))
}

/// Sorted gram counts of one normalized author name.
///
/// Matches `char_ngrams_counter` of the space-joined coauthor text: every
/// window without a space lies inside one name, so that counter is the sum of
/// these per-name counters. Normalized author names hold only `[a-z ]`, so a
/// window of characters is a window of bytes.
fn name_grams(name: &str) -> PyResult<Vec<(u32, u32)>> {
    let bytes = name.as_bytes();
    if !bytes
        .iter()
        .all(|byte| *byte == b' ' || byte.is_ascii_lowercase())
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "normalized author name {name:?} has characters outside [a-z ]"
        )));
    }
    let mut keys = Vec::new();
    for width in 2..=4 {
        keys.extend(
            bytes
                .windows(width)
                .filter(|window| !window.contains(&b' '))
                .map(gram_key),
        );
    }
    keys.sort_unstable();
    let mut grams: Vec<(u32, u32)> = Vec::new();
    for key in keys {
        match grams.last_mut() {
            Some((last, count)) if *last == key => *count += 1,
            _ => grams.push((key, 1)),
        }
    }
    Ok(grams)
}

/// Sorted distinct ids and their multiplicities.
fn counted_ids(mut ids: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
    ids.sort_unstable();
    let mut distinct: Vec<u32> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    for id in ids {
        if distinct.last() == Some(&id) {
            *counts.last_mut().expect("counts track distinct ids") += 1;
        } else {
            distinct.push(id);
            counts.push(1);
        }
    }
    (distinct, counts)
}

struct PaperEntry {
    coauthors: PaperCoauthors,
    /// `(position, name id)` of every author, sorted by position.
    authors: Vec<(i64, u32)>,
    /// Multiplicity of each id in `coauthors.names`.
    name_counts: Vec<u32>,
    /// Multiplicity of each id in `coauthors.blocks`.
    block_counts: Vec<u32>,
}

impl PaperEntry {
    fn build(
        paper_id: &str,
        mut authors: Vec<(i64, u32)>,
        name_blocks: &[u32],
        name_grams: &NameGrams,
    ) -> PyResult<Self> {
        authors.sort_unstable_by_key(|(position, _)| *position);
        if let Some(pair) = authors.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "paper {paper_id:?} lists two authors at position {}",
                pair[0].0
            )));
        }
        let (names, name_counts) = counted_ids(authors.iter().map(|(_, name)| *name).collect());
        let (blocks, block_counts) = counted_ids(
            authors
                .iter()
                .map(|(_, name)| name_blocks[*name as usize])
                .collect(),
        );
        let mut all_grams: Vec<(u32, u32)> = authors
            .iter()
            .flat_map(|(_, name)| name_grams.of(Some(*name)).iter().copied())
            .collect();
        all_grams.sort_unstable_by_key(|(key, _)| *key);
        let mut grams: Vec<(u32, u32)> = Vec::new();
        let mut gram_total = 0u64;
        for (key, count) in all_grams {
            gram_total += u64::from(count);
            match grams.last_mut() {
                Some((last, total)) if *last == key => *total += count,
                _ => grams.push((key, count)),
            }
        }
        if gram_total > MAX_PAPER_GRAM_TOTAL {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "paper {paper_id:?} has {gram_total} coauthor n-grams; the limit is {MAX_PAPER_GRAM_TOTAL}"
            )));
        }
        Ok(Self {
            coauthors: PaperCoauthors {
                names,
                blocks,
                grams,
                gram_total: gram_total as u32,
            },
            authors,
            name_counts,
            block_counts,
        })
    }
}

/// Build-time coauthor tables for the papers of a featurizer's signatures.
pub(crate) struct CoauthorTables {
    papers: HashMap<PaperId, PaperEntry>,
    name_blocks: Vec<u32>,
    name_grams: NameGrams,
}

impl CoauthorTables {
    /// Aggregate each paper's normalized `(position, name)` author list.
    ///
    /// Gram counts are built only when `with_grams`; otherwise the
    /// coauthor-similarity feature is missing, as without preprocessing.
    pub(crate) fn build<'a>(
        papers: impl IntoIterator<Item = (&'a PaperId, &'a [(i64, String)])>,
        with_grams: bool,
    ) -> PyResult<Self> {
        let mut name_ids: HashMap<&str, u32> = HashMap::default();
        let mut names: Vec<&str> = Vec::new();
        let mut paper_authors: Vec<(&PaperId, Vec<(i64, u32)>)> = Vec::new();
        for (paper_id, authors) in papers {
            let mut interned = Vec::with_capacity(authors.len());
            for (position, name) in authors {
                let id = match name_ids.get(name.as_str()) {
                    Some(id) => *id,
                    None => {
                        let id = u32::try_from(names.len()).map_err(|_| {
                            pyo3::exceptions::PyValueError::new_err(
                                "more than u32::MAX distinct coauthor names",
                            )
                        })?;
                        name_ids.insert(name.as_str(), id);
                        names.push(name.as_str());
                        id
                    }
                };
                interned.push((*position, id));
            }
            paper_authors.push((paper_id, interned));
        }
        drop(name_ids);

        let block_names: Vec<String> = names
            .par_iter()
            .map(|name| compute_block_compat(name))
            .collect();
        let mut block_ids: HashMap<&str, u32> = HashMap::default();
        let name_blocks: Vec<u32> = block_names
            .iter()
            .map(|block| {
                let next = block_ids.len() as u32;
                *block_ids.entry(block.as_str()).or_insert(next)
            })
            .collect();
        drop(block_ids);
        drop(block_names);

        let name_grams = if with_grams {
            NameGrams::build(&names)?
        } else {
            NameGrams::empty(names.len())
        };
        let papers = paper_authors
            .into_par_iter()
            .map(|(paper_id, authors)| {
                PaperEntry::build(paper_id, authors, &name_blocks, &name_grams)
                    .map(|entry| (paper_id.clone(), entry))
            })
            .collect::<PyResult<HashMap<_, _>>>()?;
        Ok(Self {
            papers,
            name_blocks,
            name_grams,
        })
    }

    /// Own-author record of a signature at `position` on `paper_id`.
    ///
    /// Panics if `paper_id` was not among the built papers.
    pub(crate) fn signature(&self, paper_id: &str, position: i64) -> SignatureCoauthors {
        let paper = self
            .papers
            .get(paper_id)
            .expect("every signature's paper has coauthor tables");
        let Ok(author) = paper
            .authors
            .binary_search_by_key(&position, |(author_position, _)| *author_position)
        else {
            return SignatureCoauthors::default();
        };
        let name = paper.authors[author].1;
        let block = self.name_blocks[name as usize];
        let unique = |ids: &[u32], counts: &[u32], id: u32| {
            let index = ids.binary_search(&id).expect("own id is on its paper");
            (counts[index] == 1).then_some(id)
        };
        SignatureCoauthors {
            own_name: Some(name),
            removed_name: unique(&paper.coauthors.names, &paper.name_counts, name),
            removed_block: unique(&paper.coauthors.blocks, &paper.block_counts, block),
        }
    }

    /// Per-paper aggregates for `PaperData`, and the name gram table the
    /// featurizer keeps for subtracting own authors.
    pub(crate) fn into_parts(self) -> (HashMap<PaperId, PaperCoauthors>, NameGrams) {
        let papers = self
            .papers
            .into_iter()
            .map(|(paper_id, entry)| (paper_id, entry.coauthors))
            .collect();
        (papers, self.name_grams)
    }
}

/// `coauthor_overlap`, `coauthor_similarity` and `coauthor_match` of one pair.
pub(crate) struct CoauthorPairFeatures {
    pub(crate) block_overlap: f64,
    pub(crate) gram_similarity: f64,
    pub(crate) name_match: f64,
}

pub(crate) fn coauthor_pair_features(
    paper1: &PaperCoauthors,
    signature1: &SignatureCoauthors,
    paper2: &PaperCoauthors,
    signature2: &SignatureCoauthors,
    name_grams: &NameGrams,
) -> CoauthorPairFeatures {
    CoauthorPairFeatures {
        block_overlap: removed_set_jaccard(
            &paper1.blocks,
            signature1.removed_block,
            &paper2.blocks,
            signature2.removed_block,
        ),
        gram_similarity: removed_gram_jaccard(
            paper1,
            name_grams.of(signature1.own_name),
            paper2,
            name_grams.of(signature2.own_name),
        ),
        name_match: removed_set_jaccard(
            &paper1.names,
            signature1.removed_name,
            &paper2.names,
            signature2.removed_name,
        ),
    }
}

/// Jaccard of `set1 \ {removed1}` and `set2 \ {removed2}`, NaN if either is empty.
///
/// Each removed id, when present, is a member of its own set.
fn removed_set_jaccard(
    set1: &[u32],
    removed1: Option<u32>,
    set2: &[u32],
    removed2: Option<u32>,
) -> f64 {
    let len1 = set1.len() - usize::from(removed1.is_some());
    let len2 = set2.len() - usize::from(removed2.is_some());
    if len1 == 0 || len2 == 0 {
        return f64::NAN;
    }
    // (set1 ∩ set2) \ {removed1, removed2}: removed1 is in the intersection iff
    // it is in set2, and likewise removed2.
    let shared1 = removed1.filter(|id| set2.binary_search(id).is_ok());
    let shared2 = removed2.filter(|id| set1.binary_search(id).is_ok());
    let removed_shared = match (shared1, shared2) {
        (Some(id1), Some(id2)) if id1 == id2 => 1,
        _ => usize::from(shared1.is_some()) + usize::from(shared2.is_some()),
    };
    let intersection = sorted_intersection_count(set1, set2) - removed_shared;
    let union = len1 + len2 - intersection;
    (intersection as f64) / (union as f64)
}

fn sorted_intersection_count(set1: &[u32], set2: &[u32]) -> usize {
    let mut count = 0;
    for_each_common(
        set1,
        set2,
        |id| *id,
        |_, _| {
            count += 1;
        },
    );
    count
}

/// Call `on_common(item1, item2)` for every key the sorted, key-distinct
/// slices share, in increasing key order.
fn for_each_common<T>(
    items1: &[T],
    items2: &[T],
    key: impl Fn(&T) -> u32,
    mut on_common: impl FnMut(&T, &T),
) {
    let (small, large, swapped) = if items1.len() <= items2.len() {
        (items1, items2, false)
    } else {
        (items2, items1, true)
    };
    let mut emit = |small_item: &T, large_item: &T| {
        if swapped {
            on_common(large_item, small_item)
        } else {
            on_common(small_item, large_item)
        }
    };
    if large.len() >= small.len().saturating_mul(4) {
        let mut low = 0;
        for small_item in small {
            match large[low..].binary_search_by_key(&key(small_item), &key) {
                Ok(offset) => {
                    emit(small_item, &large[low + offset]);
                    low += offset + 1;
                }
                Err(offset) => low += offset,
            }
        }
        return;
    }
    let (mut small_index, mut large_index) = (0, 0);
    while small_index < small.len() && large_index < large.len() {
        let (small_key, large_key) = (key(&small[small_index]), key(&large[large_index]));
        if small_key == large_key {
            emit(&small[small_index], &large[large_index]);
        }
        // Advance without a key-dependent branch; equal keys move both, as before.
        small_index += usize::from(small_key <= large_key);
        large_index += usize::from(large_key <= small_key);
    }
}

/// Gram entries compared per side per step by `for_each_common_gram_avx2`.
const GRAM_BLOCK: usize = 8;

/// Call `on_common(key, count1, count2)` for every gram key two sorted,
/// key-distinct gram lists share, in increasing key order, as
/// `for_each_common` does.
///
/// Eight keys per side are compared all against all per step. A step with no
/// equal keys only advances; matches are reported lane by lane in key order,
/// so callers that read other sorted lists by key see the same sequence.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn for_each_common_gram_avx2(
    grams1: &[(u32, u32)],
    grams2: &[(u32, u32)],
    on_common: &mut impl FnMut(u32, u32, u32),
) {
    use std::arch::x86_64::*;
    debug_assert!(grams1.windows(2).all(|pair| pair[0].0 < pair[1].0));
    debug_assert!(grams2.windows(2).all(|pair| pair[0].0 < pair[1].0));
    // (key, count) pairs are 8 bytes; gather keys into lanes 0..4 and counts into 4..8.
    let split = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    let load = |grams: &[(u32, u32)], at: usize| -> (__m256i, __m256i) {
        let low = _mm256_permutevar8x32_epi32(
            _mm256_loadu_si256(grams.as_ptr().add(at) as *const __m256i),
            split,
        );
        let high = _mm256_permutevar8x32_epi32(
            _mm256_loadu_si256(grams.as_ptr().add(at + 4) as *const __m256i),
            split,
        );
        (
            _mm256_permute2x128_si256::<0x20>(low, high),
            _mm256_permute2x128_si256::<0x31>(low, high),
        )
    };
    let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let (n1, n2) = (grams1.len(), grams2.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i + GRAM_BLOCK <= n1 && j + GRAM_BLOCK <= n2 {
        let (keys1, _) = load(grams1, i);
        let (keys2, _) = load(grams2, j);
        // masks[r] bit l: key1 lane l equals key2 lane (l + r) % 8.
        let mut masks = [0u32; GRAM_BLOCK];
        let mut any = 0u32;
        for (rotation, mask) in masks.iter_mut().enumerate() {
            let index = _mm256_and_si256(
                _mm256_add_epi32(lanes, _mm256_set1_epi32(rotation as i32)),
                _mm256_set1_epi32(7),
            );
            let rotated = _mm256_permutevar8x32_epi32(keys2, index);
            *mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(keys1, rotated))) as u32;
            any |= *mask;
        }
        while any != 0 {
            let lane = any.trailing_zeros() as usize;
            any &= any - 1;
            let rotation = masks
                .iter()
                .position(|mask| mask & (1 << lane) != 0)
                .expect("a set lane has a rotation");
            let (key, count1) = *grams1.get_unchecked(i + lane);
            let (_, count2) = *grams2.get_unchecked(j + (lane + rotation) % GRAM_BLOCK);
            on_common(key, count1, count2);
        }
        let last1 = grams1.get_unchecked(i + GRAM_BLOCK - 1).0;
        let last2 = grams2.get_unchecked(j + GRAM_BLOCK - 1).0;
        i += GRAM_BLOCK * usize::from(last1 <= last2);
        j += GRAM_BLOCK * usize::from(last2 <= last1);
    }
    while i < n1 && j < n2 {
        let (key1, count1) = *grams1.get_unchecked(i);
        let (key2, count2) = *grams2.get_unchecked(j);
        if key1 == key2 {
            on_common(key1, count1, count2);
        }
        i += usize::from(key1 <= key2);
        j += usize::from(key2 <= key1);
    }
}

/// Counts of a sorted gram list, looked up with nondecreasing keys.
struct OwnGramCursor<'a> {
    grams: &'a [(u32, u32)],
    next: usize,
}

impl OwnGramCursor<'_> {
    fn count(&mut self, key: u32) -> u32 {
        while self.next < self.grams.len() && self.grams[self.next].0 < key {
            self.next += 1;
        }
        match self.grams.get(self.next) {
            Some((own_key, count)) if *own_key == key => *count,
            _ => 0,
        }
    }
}

/// Capped Jaccard of `paper1.grams - own1` and `paper2.grams - own2`, NaN if either is empty.
///
/// Each own gram list is one author's grams on its paper, so the differences
/// are the signatures' coauthor gram counters.
fn removed_gram_jaccard(
    paper1: &PaperCoauthors,
    own1: &[(u32, u32)],
    paper2: &PaperCoauthors,
    own2: &[(u32, u32)],
) -> f64 {
    let own_total = |own: &[(u32, u32)]| own.iter().map(|(_, count)| *count).sum::<u32>();
    let sum1 = paper1.gram_total - own_total(own1);
    let sum2 = paper2.gram_total - own_total(own2);
    if sum1 == 0 || sum2 == 0 {
        return f64::NAN;
    }
    let mut own1 = OwnGramCursor {
        grams: own1,
        next: 0,
    };
    let mut own2 = OwnGramCursor {
        grams: own2,
        next: 0,
    };
    let mut intersection = 0u64;
    let mut add = |key: u32, count1: u32, count2: u32| {
        let count1 = count1 - own1.count(key);
        let count2 = count2 - own2.count(key);
        intersection += u64::from(count1.min(count2));
    };
    let (shorter, longer) = if paper1.grams.len() <= paper2.grams.len() {
        (paper1.grams.len(), paper2.grams.len())
    } else {
        (paper2.grams.len(), paper1.grams.len())
    };
    #[cfg(target_arch = "x86_64")]
    let use_simd = shorter >= GRAM_BLOCK
        && longer < shorter.saturating_mul(4)
        && std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_simd = false;
    if use_simd {
        // SAFETY: AVX2 was detected above.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            for_each_common_gram_avx2(&paper1.grams, &paper2.grams, &mut add)
        };
    } else {
        for_each_common(
            &paper1.grams,
            &paper2.grams,
            |(key, _)| *key,
            |(key, count1), (_, count2)| add(*key, *count1, *count2),
        );
    }
    let intersection = intersection as f64;
    let union = f64::from(sum1) + f64::from(sum2) - intersection;
    let score = intersection / union.min(GRAM_SIMILARITY_DENOMINATOR_MAX);
    if score > 1.0 {
        1.0
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::counter_jaccard_data;
    use crate::HashSet;
    use crate::{fnv64, CounterData};

    /// The per-signature coauthor records this module replaces, built exactly
    /// as `preprocess_stage_signatures` did.
    struct ReferenceCoauthors {
        coauthors: Option<HashSet<String>>,
        blocks: Option<HashSet<String>>,
        grams: Option<CounterData>,
    }

    fn reference_char_ngrams_counter(text: &str) -> HashMap<String, usize> {
        let chars: Vec<char> = text.chars().collect();
        let mut out: HashMap<String, usize> = HashMap::default();
        for width in [2usize, 3usize, 4usize] {
            if chars.len() < width {
                continue;
            }
            for idx in 0..=(chars.len() - width) {
                let window = &chars[idx..idx + width];
                if window.contains(&' ') {
                    continue;
                }
                let gram: String = window.iter().collect();
                *out.entry(gram).or_insert(0) += 1;
            }
        }
        out
    }

    fn reference_counter_data(counter: HashMap<String, usize>) -> Option<CounterData> {
        if counter.is_empty() {
            return None;
        }
        let mut entries: Vec<(u64, f32)> = counter
            .iter()
            .map(|(gram, count)| (fnv64(gram.as_bytes()), *count as f32))
            .collect();
        entries.sort_unstable_by_key(|entry| entry.0);
        let sum: f32 = entries.iter().map(|entry| entry.1).sum();
        Some(CounterData::new(entries, sum))
    }

    fn reference_coauthors(
        authors: &[(i64, String)],
        position: i64,
        preprocess: bool,
    ) -> ReferenceCoauthors {
        let coauthor_list: Vec<String> = authors
            .iter()
            .filter(|(author_position, _)| *author_position != position)
            .map(|(_, name)| name.clone())
            .collect();
        let coauthors = (!coauthor_list.is_empty())
            .then(|| coauthor_list.iter().cloned().collect::<HashSet<String>>());
        let blocks: HashSet<String> = coauthor_list
            .iter()
            .map(|name| compute_block_compat(name))
            .collect();
        let text = coauthor_list.join(" ");
        ReferenceCoauthors {
            coauthors,
            blocks: (!blocks.is_empty()).then_some(blocks),
            grams: if preprocess && !text.is_empty() {
                reference_counter_data(reference_char_ngrams_counter(&text))
            } else {
                None
            },
        }
    }

    fn reference_set_jaccard(
        set1: &Option<HashSet<String>>,
        set2: &Option<HashSet<String>>,
    ) -> f64 {
        let (Some(s1), Some(s2)) = (set1.as_ref(), set2.as_ref()) else {
            return f64::NAN;
        };
        if s1.is_empty() || s2.is_empty() {
            return f64::NAN;
        }
        let intersection = s1.intersection(s2).count();
        let union = s1.len() + s2.len() - intersection;
        if union == 0 {
            return f64::NAN;
        }
        (intersection as f64) / (union as f64)
    }

    fn reference_features(first: &ReferenceCoauthors, second: &ReferenceCoauthors) -> [u64; 3] {
        [
            reference_set_jaccard(&first.blocks, &second.blocks).to_bits(),
            counter_jaccard_data(&first.grams, &second.grams, 5000.0).to_bits(),
            reference_set_jaccard(&first.coauthors, &second.coauthors).to_bits(),
        ]
    }

    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, bound: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) % bound as u64) as usize
        }
    }

    /// Papers with duplicate names, shared blocks, blank and one-letter names,
    /// a giant author list, and a paper whose only author is the signature.
    fn synthetic_papers(giant_authors: usize) -> Vec<(PaperId, Vec<(i64, String)>)> {
        let firsts = ["a", "anna", "bo", "j", "jian", "maria", "x", "", "li wei"];
        let lasts = ["wang", "li", "smith", "o neil", "zz", "", "garcia lopez"];
        let mut rng = Lcg(7);
        let name = |rng: &mut Lcg| {
            let first = firsts[rng.below(firsts.len())];
            let last = lasts[rng.below(lasts.len())];
            [first, last]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        };
        let mut papers = vec![
            (
                "giant".to_string(),
                (0..giant_authors)
                    .map(|position| (position as i64 * 2, name(&mut rng)))
                    .collect(),
            ),
            ("solo".to_string(), vec![(0, "anna li".to_string())]),
            (
                "blanks".to_string(),
                vec![
                    (0, "j wang".to_string()),
                    (1, String::new()),
                    (2, String::new()),
                ],
            ),
            (
                "letters".to_string(),
                vec![
                    (0, "x".to_string()),
                    (1, "a".to_string()),
                    (2, "bo".to_string()),
                ],
            ),
            (
                "twins".to_string(),
                vec![
                    (0, "jian wang".to_string()),
                    (1, "jian wang".to_string()),
                    (5, "j wang".to_string()),
                ],
            ),
        ];
        for paper in 0..40 {
            let authors = 1 + rng.below(12);
            papers.push((
                format!("small{paper}"),
                (0..authors)
                    .map(|position| (position as i64, name(&mut rng)))
                    .collect(),
            ));
        }
        papers
    }

    /// Signatures at every author position (every `giant_stride`-th on the
    /// giant paper), plus a position no author holds on each paper.
    fn synthetic_signatures(
        papers: &[(PaperId, Vec<(i64, String)>)],
        giant_stride: usize,
    ) -> Vec<(usize, i64)> {
        let mut signatures = Vec::new();
        for (paper_index, (paper_id, authors)) in papers.iter().enumerate() {
            let stride = if paper_id == "giant" { giant_stride } else { 1 };
            for (position, _) in authors.iter().step_by(stride) {
                signatures.push((paper_index, *position));
            }
            signatures.push((paper_index, 1_000_001));
        }
        signatures
    }

    fn assert_matches_reference(giant_authors: usize, giant_stride: usize, preprocess: bool) {
        let papers = synthetic_papers(giant_authors);
        let tables = CoauthorTables::build(
            papers
                .iter()
                .map(|(paper_id, authors)| (paper_id, authors.as_slice())),
            preprocess,
        )
        .expect("synthetic papers build");
        let signatures = synthetic_signatures(&papers, giant_stride);
        let records: Vec<SignatureCoauthors> = signatures
            .iter()
            .map(|(paper, position)| tables.signature(&papers[*paper].0, *position))
            .collect();
        let references: Vec<ReferenceCoauthors> = signatures
            .iter()
            .map(|(paper, position)| reference_coauthors(&papers[*paper].1, *position, preprocess))
            .collect();
        let (paper_coauthors, name_grams) = tables.into_parts();
        let (mut missing, mut fractional) = ([0usize; 3], [0usize; 3]);
        let mut rng = Lcg(11);
        for first in 0..signatures.len() {
            for _ in 0..40 {
                let second = rng.below(signatures.len());
                let paper1 = &paper_coauthors[&papers[signatures[first].0].0];
                let paper2 = &paper_coauthors[&papers[signatures[second].0].0];
                let features = coauthor_pair_features(
                    paper1,
                    &records[first],
                    paper2,
                    &records[second],
                    &name_grams,
                );
                let observed = [
                    features.block_overlap.to_bits(),
                    features.gram_similarity.to_bits(),
                    features.name_match.to_bits(),
                ];
                let expected = reference_features(&references[first], &references[second]);
                assert_eq!(
                    observed, expected,
                    "signatures {:?} and {:?}",
                    signatures[first], signatures[second]
                );
                for (feature, value) in [
                    features.block_overlap,
                    features.gram_similarity,
                    features.name_match,
                ]
                .into_iter()
                .enumerate()
                {
                    if value.is_nan() {
                        missing[feature] += 1;
                    } else if value > 0.0 && value < 1.0 {
                        fractional[feature] += 1;
                    }
                }
            }
        }
        // The comparison covers missing and strictly fractional values of each feature.
        let grams_expected = usize::from(preprocess);
        assert!(missing.iter().all(|count| *count > 0), "{missing:?}");
        assert!(fractional[0] > 100 && fractional[2] > 100, "{fractional:?}");
        assert_eq!(fractional[1] > 100, grams_expected == 1, "{fractional:?}");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_gram_matches_equal_the_merge_in_order() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Lcg(23);
        for _ in 0..5_000 {
            let universe = 1 + rng.below(3_000) as u32;
            let mut make = |rng: &mut Lcg| {
                let len = rng.below(700);
                let mut keys: Vec<u32> = (0..len).map(|_| rng.below(universe as usize) as u32).collect();
                keys.sort_unstable();
                keys.dedup();
                keys.into_iter()
                    .map(|key| (key, 1 + rng.below(9) as u32))
                    .collect::<Vec<_>>()
            };
            let first = make(&mut rng);
            let second = make(&mut rng);
            let mut expected = Vec::new();
            for_each_common(&first, &second, |(key, _)| *key, |(key, count1), (_, count2)| {
                expected.push((*key, *count1, *count2));
            });
            let mut observed = Vec::new();
            // SAFETY: AVX2 was detected above.
            unsafe {
                for_each_common_gram_avx2(&first, &second, &mut |key, count1, count2| {
                    observed.push((key, count1, count2));
                });
            }
            assert_eq!(observed, expected);
        }
    }

    #[test]
    fn pair_features_match_per_signature_records_bit_for_bit() {
        assert_matches_reference(300, 1, true);
        assert_matches_reference(300, 1, false);
    }

    #[test]
    fn giant_paper_pair_features_match_per_signature_records_bit_for_bit() {
        assert_matches_reference(3000, 97, true);
    }

    #[test]
    fn same_paper_pairs_exclude_both_own_authors() {
        let papers = [(
            "p".to_string(),
            vec![
                (0, "anna li".to_string()),
                (1, "bo li".to_string()),
                (2, "anna li".to_string()),
                (3, "maria garcia".to_string()),
            ],
        )];
        let tables = CoauthorTables::build(
            papers
                .iter()
                .map(|(paper_id, authors)| (paper_id, authors.as_slice())),
            true,
        )
        .expect("paper builds");
        let records: Vec<_> = (0..4)
            .map(|position| tables.signature("p", position))
            .collect();
        let (paper_coauthors, name_grams) = tables.into_parts();
        let paper = &paper_coauthors["p"];
        for first in 0..4 {
            for second in 0..4 {
                let features = coauthor_pair_features(
                    paper,
                    &records[first],
                    paper,
                    &records[second],
                    &name_grams,
                );
                let expected = reference_features(
                    &reference_coauthors(&papers[0].1, first as i64, true),
                    &reference_coauthors(&papers[0].1, second as i64, true),
                );
                assert_eq!(
                    [
                        features.block_overlap.to_bits(),
                        features.gram_similarity.to_bits(),
                        features.name_match.to_bits()
                    ],
                    expected
                );
            }
        }
        // "anna li" appears twice, so neither of its signatures removes the name.
        assert_eq!(records[0].removed_name, None);
        assert_eq!(records[1].removed_name, records[1].own_name);
    }

    /// Normalized author names hold only `[a-z ]`, so every gram is in
    /// `[a-z]{2,4}`. FNV-1a keyed the per-signature counters; it has no
    /// collision on that domain, so those counters kept one entry per distinct
    /// gram, exactly like the injective `gram_key`.
    #[test]
    fn fnv_gram_keys_have_no_collisions_on_letter_grams() {
        fn visit(gram: &mut Vec<u8>, hashes: &mut HashSet<u64>, keys: &mut HashSet<u32>) {
            if gram.len() >= 2 {
                assert!(hashes.insert(fnv64(gram)), "FNV collision at {gram:?}");
                assert!(
                    keys.insert(gram_key(gram)),
                    "gram_key collision at {gram:?}"
                );
            }
            if gram.len() == 4 {
                return;
            }
            for letter in b'a'..=b'z' {
                gram.push(letter);
                visit(gram, hashes, keys);
                gram.pop();
            }
        }
        let (mut hashes, mut keys) = (HashSet::default(), HashSet::default());
        visit(&mut Vec::with_capacity(4), &mut hashes, &mut keys);
        assert_eq!(
            hashes.len(),
            26usize.pow(2) + 26usize.pow(3) + 26usize.pow(4)
        );
        assert_eq!(keys.len(), hashes.len());
    }

    #[test]
    fn blank_coauthor_names_are_coauthors_without_ngrams() {
        let papers = [
            (
                "p1".to_string(),
                vec![(0, "alice smith".to_string()), (1, String::new())],
            ),
            (
                "p2".to_string(),
                vec![(0, "bob jones".to_string()), (1, String::new())],
            ),
        ];
        let tables = CoauthorTables::build(
            papers
                .iter()
                .map(|(paper_id, authors)| (paper_id, authors.as_slice())),
            true,
        )
        .expect("papers build");
        let (first, second) = (tables.signature("p1", 0), tables.signature("p2", 0));
        let (paper_coauthors, name_grams) = tables.into_parts();
        let features = coauthor_pair_features(
            &paper_coauthors["p1"],
            &first,
            &paper_coauthors["p2"],
            &second,
            &name_grams,
        );
        // Both coauthor sets are {""} and both block sets {""}; "" has no n-grams.
        assert_eq!(features.name_match, 1.0);
        assert_eq!(features.block_overlap, 1.0);
        assert!(features.gram_similarity.is_nan());
    }

    #[test]
    fn name_grams_reject_unnormalized_names() {
        assert!(name_grams("anna li").is_ok());
        assert!(name_grams("Anna").is_err());
        assert!(name_grams("an\u{00e9}").is_err());
    }
}
