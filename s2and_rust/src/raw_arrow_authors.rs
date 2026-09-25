//! Per-paper author tables for the raw Arrow retrieval and linker features.
//!
//! Retrieval queries carry the name blocks of a signature's coauthors, and
//! linker rows compare paper author-name sets and position windows. Building
//! those per signature normalized every author of the paper again for every
//! signature, and again for every component summary that held the signature,
//! which is O(authors^2) per paper; INSPIRE-style papers have thousands of
//! authors. These tables normalize each paper's authors once: block terms per
//! paper for query features, and interned name ids per paper for author-list
//! evidence. A signature's values are read off its paper's table, so they
//! equal the per-signature computation exactly.

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::raw_arrow::readers::{RawArrowAuthorSignalData, RawArrowSignature};
use crate::text_compat::{compute_block_compat, normalize_text_compat_from_map};
use crate::{fnv64, term_token_count, RetrievalQueryTerm};

/// Position window, in author positions, of the local author-list evidence.
const LOCAL_AUTHOR_WINDOW: u64 = 10;

fn require_unique_positions<T>(paper_id: &str, authors: &[(i64, T)]) -> Result<(), String> {
    match authors.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        Some(pair) => Err(format!(
            "paper_authors for paper_id {paper_id:?} list two authors at position {}",
            pair[0].0
        )),
        None => Ok(()),
    }
}

/// One paper's coauthor name blocks, shared by every signature on it.
pub(crate) struct PaperCoauthorBlocks {
    /// Author rows on the paper, including blank names.
    author_count: usize,
    /// `(position, block index)` of each author with a nonempty block, sorted by position.
    blocked_authors: Vec<(i64, u32)>,
    /// `(term, block index, authors with the block)` per distinct block, sorted by term hash.
    terms: Vec<(RetrievalQueryTerm, u32, u32)>,
}

impl PaperCoauthorBlocks {
    fn new(
        paper_id: &str,
        authors: &[(i64, String)],
        unidecode_char_map: &HashMap<char, String>,
    ) -> Result<Self, String> {
        let mut block_indices: HashMap<String, u32> = HashMap::new();
        let mut block_counts: Vec<u32> = Vec::new();
        let mut blocked_authors = Vec::new();
        for (position, name) in authors {
            let normalized = normalize_text_compat_from_map(name, false, unidecode_char_map);
            if normalized.is_empty() {
                continue;
            }
            let block = compute_block_compat(&normalized);
            if block.is_empty() {
                continue;
            }
            let next = block_counts.len() as u32;
            let index = *block_indices.entry(block).or_insert(next);
            if index == next {
                block_counts.push(0);
            }
            block_counts[index as usize] += 1;
            blocked_authors.push((*position, index));
        }
        blocked_authors.sort_unstable_by_key(|(position, _)| *position);
        require_unique_positions(paper_id, &blocked_authors)?;
        let mut terms: Vec<(RetrievalQueryTerm, u32, u32)> = block_indices
            .iter()
            .map(|(block, index)| {
                let term = RetrievalQueryTerm {
                    hash: fnv64(block.as_bytes()),
                    token_count: term_token_count(block),
                };
                (term, *index, block_counts[*index as usize])
            })
            .collect();
        terms.sort_unstable_by_key(|(term, index, _)| (term.hash, *index));
        Ok(Self {
            author_count: authors.len(),
            blocked_authors,
            terms,
        })
    }

    pub(crate) fn author_count(&self) -> usize {
        self.author_count
    }

    /// Terms of the blocks of every author except the one at `position`.
    ///
    /// Matches `query_terms_from_values` over those blocks: sorted by hash,
    /// one term per hash. A signature without a position has no coauthors.
    pub(crate) fn coauthor_terms(&self, position: Option<i64>) -> Vec<RetrievalQueryTerm> {
        let Some(position) = position else {
            return Vec::new();
        };
        let own_block = self
            .blocked_authors
            .binary_search_by_key(&position, |(author_position, _)| *author_position)
            .ok()
            .map(|author| self.blocked_authors[author].1);
        let mut terms: Vec<RetrievalQueryTerm> = self
            .terms
            .iter()
            .filter(|(_, index, count)| !(own_block == Some(*index) && *count == 1))
            .map(|(term, _, _)| *term)
            .collect();
        terms.dedup_by_key(|term| term.hash);
        terms
    }
}

/// Coauthor block tables of the papers of `signatures`, built in parallel.
///
/// Papers without author rows are absent, as are their signatures' coauthors.
pub(crate) fn paper_coauthor_blocks<'a>(
    signatures: impl IntoIterator<Item = &'a RawArrowSignature>,
    paper_authors: &HashMap<String, Vec<(i64, String)>>,
    unidecode_char_map: &HashMap<char, String>,
) -> Result<HashMap<String, PaperCoauthorBlocks>, String> {
    let paper_ids: HashSet<&str> = signatures
        .into_iter()
        .map(|signature| signature.paper_id.as_str())
        .collect();
    paper_ids
        .into_par_iter()
        .filter_map(|paper_id| {
            paper_authors.get(paper_id).map(|authors| {
                PaperCoauthorBlocks::new(paper_id, authors, unidecode_char_map)
                    .map(|blocks| (paper_id.to_string(), blocks))
            })
        })
        .collect()
}

/// One paper's normalized author names as interned ids.
struct PaperAuthorNames {
    /// `(position, name id)` of each author with a nonempty name, sorted by position.
    named_authors: Vec<(i64, u32)>,
    /// Sorted distinct name ids.
    names: Arc<[u32]>,
}

/// Low bits of a name id that select its interning shard.
const NAME_SHARD_BITS: u32 = 6;
const NAME_SHARDS: usize = 1 << NAME_SHARD_BITS;

/// Normalized author names interned by string equality, with the name ids of
/// each indexed paper.
///
/// A name id is `(index within shard << NAME_SHARD_BITS) | shard`, where the
/// shard comes from the name's hash, so shards intern in parallel. A planner
/// keeps one index for its lifetime, so the query papers of every plan share
/// ids with the seed papers they are compared against.
pub(crate) struct PaperAuthorIndex {
    name_shards: Vec<HashMap<String, u32>>,
    papers: HashMap<String, PaperAuthorNames>,
}

impl Default for PaperAuthorIndex {
    fn default() -> Self {
        Self {
            name_shards: (0..NAME_SHARDS).map(|_| HashMap::new()).collect(),
            papers: HashMap::new(),
        }
    }
}

fn name_shard(name: &str) -> usize {
    (fnv64(name.as_bytes()) as usize) & (NAME_SHARDS - 1)
}

impl PaperAuthorIndex {
    /// Index the not-yet-indexed papers of `paper_ids` in parallel.
    ///
    /// A paper without author rows is indexed with no authors.
    pub(crate) fn extend<'a>(
        &mut self,
        paper_ids: impl IntoIterator<Item = &'a str>,
        paper_authors: &HashMap<String, Vec<(i64, String)>>,
        unidecode_char_map: &HashMap<char, String>,
    ) -> Result<(), String> {
        let mut seen = HashSet::new();
        let pending: Vec<&str> = paper_ids
            .into_iter()
            .filter(|paper_id| !self.papers.contains_key(*paper_id) && seen.insert(*paper_id))
            .collect();
        let normalized: Vec<Vec<(i64, String, u8)>> = pending
            .par_iter()
            .map(|paper_id| {
                paper_authors
                    .get(*paper_id)
                    .map_or(&[][..], Vec::as_slice)
                    .iter()
                    .filter_map(|(position, name)| {
                        let normalized =
                            normalize_text_compat_from_map(name, false, unidecode_char_map);
                        (!normalized.is_empty()).then(|| {
                            let shard = name_shard(&normalized) as u8;
                            (*position, normalized, shard)
                        })
                    })
                    .collect()
            })
            .collect();
        // Bucket each name occurrence `(paper, author)` by shard, then intern
        // every shard in parallel.
        let mut buckets: Vec<Vec<(u32, u32)>> = vec![Vec::new(); NAME_SHARDS];
        for (paper, authors) in normalized.iter().enumerate() {
            for (author, (_, _, shard)) in authors.iter().enumerate() {
                buckets[*shard as usize].push((paper as u32, author as u32));
            }
        }
        let shard_ids: Vec<Vec<u32>> = self
            .name_shards
            .par_iter_mut()
            .zip(buckets.par_iter())
            .enumerate()
            .map(|(shard, (names, bucket))| {
                bucket
                    .iter()
                    .map(|(paper, author)| {
                        let name = &normalized[*paper as usize][*author as usize].1;
                        let local = match names.get(name.as_str()) {
                            Some(local) => *local,
                            None => {
                                let local = u32::try_from(names.len())
                                    .ok()
                                    .filter(|local| *local < (1 << (32 - NAME_SHARD_BITS)))
                                    .ok_or("paper author names overflow u32 name ids")?;
                                names.insert(name.clone(), local);
                                local
                            }
                        };
                        Ok((local << NAME_SHARD_BITS) | shard as u32)
                    })
                    .collect::<Result<Vec<u32>, String>>()
            })
            .collect::<Result<_, _>>()?;
        let mut named: Vec<Vec<(i64, u32)>> = normalized
            .iter()
            .map(|authors| {
                authors
                    .iter()
                    .map(|(position, _, _)| (*position, 0))
                    .collect()
            })
            .collect();
        for (bucket, ids) in buckets.iter().zip(&shard_ids) {
            for ((paper, author), id) in bucket.iter().zip(ids) {
                named[*paper as usize][*author as usize].1 = *id;
            }
        }
        normalized.into_par_iter().for_each(drop);
        let tables: Vec<Result<PaperAuthorNames, String>> = pending
            .par_iter()
            .zip(named.into_par_iter())
            .map(|(paper_id, mut named_authors)| {
                named_authors.sort_unstable_by_key(|(position, _)| *position);
                require_unique_positions(paper_id, &named_authors)?;
                let mut names: Vec<u32> = named_authors.iter().map(|(_, name)| *name).collect();
                names.sort_unstable();
                names.dedup();
                Ok(PaperAuthorNames {
                    named_authors,
                    names: names.into(),
                })
            })
            .collect();
        for (paper_id, table) in pending.into_iter().zip(tables) {
            self.papers.insert(paper_id.to_string(), table?);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn name_id(&self, name: &str) -> Option<u32> {
        let shard = name_shard(name);
        self.name_shards[shard]
            .get(name)
            .map(|local| (local << NAME_SHARD_BITS) | shard as u32)
    }

    /// Author-list signals of `signature` from its indexed paper.
    pub(crate) fn author_signals(
        &self,
        signature: &RawArrowSignature,
    ) -> Result<RawArrowAuthorSignalData, String> {
        let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
            format!(
                "paper_id {:?} was not indexed for author signals",
                signature.paper_id
            )
        })?;
        let mut local10_author_names = Vec::new();
        if let Some(position) = signature.position {
            let start = paper.named_authors.partition_point(|(author_position, _)| {
                *author_position < position.saturating_sub(LOCAL_AUTHOR_WINDOW as i64)
            });
            for (author_position, name) in paper.named_authors[start..].iter() {
                if *author_position > position.saturating_add(LOCAL_AUTHOR_WINDOW as i64) {
                    break;
                }
                if *author_position != position
                    && author_position.abs_diff(position) <= LOCAL_AUTHOR_WINDOW
                {
                    local10_author_names.push(*name);
                }
            }
            local10_author_names.sort_unstable();
            local10_author_names.dedup();
        }
        Ok(RawArrowAuthorSignalData {
            paper_author_names: Arc::clone(&paper.names),
            local10_author_names,
        })
    }
}

/// Size of the intersection of two sorted, duplicate-free id lists.
pub(crate) fn sorted_intersection_count(left: &[u32], right: &[u32]) -> usize {
    let (small, large) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    if large.len() >= small.len().saturating_mul(8) {
        let mut low = 0;
        let mut count = 0;
        for id in small {
            match large[low..].binary_search(id) {
                Ok(offset) => {
                    count += 1;
                    low += offset + 1;
                }
                Err(offset) => low += offset,
            }
        }
        return count;
    }
    let (mut small_index, mut large_index, mut count) = (0, 0, 0);
    while small_index < small.len() && large_index < large.len() {
        match small[small_index].cmp(&large[large_index]) {
            std::cmp::Ordering::Equal => {
                count += 1;
                small_index += 1;
                large_index += 1;
            }
            std::cmp::Ordering::Less => small_index += 1,
            std::cmp::Ordering::Greater => large_index += 1,
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_terms_from_values;
    use crate::raw_arrow::readers::RawArrowSummarySignalData;
    use crate::raw_arrow_features::{raw_arrow_paper_evidence_row, round_six};

    /// Coauthor blocks exactly as `build_raw_arrow_feature` built them per signature.
    fn reference_coauthor_terms(
        position: Option<i64>,
        authors: &[(i64, String)],
        unidecode_char_map: &HashMap<char, String>,
    ) -> Vec<(u64, u8)> {
        let mut coauthor_blocks = HashSet::new();
        if let Some(signature_position) = position {
            for (position, author_name) in authors.iter() {
                let normalized =
                    normalize_text_compat_from_map(author_name, false, unidecode_char_map);
                if *position == signature_position {
                    continue;
                }
                if normalized.is_empty() {
                    continue;
                }
                let block = compute_block_compat(&normalized);
                if !block.is_empty() {
                    coauthor_blocks.insert(block);
                }
            }
        }
        query_terms_from_values(&coauthor_blocks)
            .iter()
            .map(|term| (term.hash, term.token_count))
            .collect()
    }

    /// `build_raw_arrow_author_signal_data` as it was: string sets per signature.
    fn reference_author_signals(
        position: Option<i64>,
        authors: &[(i64, String)],
        unidecode_char_map: &HashMap<char, String>,
    ) -> (HashSet<String>, HashSet<String>) {
        let mut paper_author_names = HashSet::new();
        let mut local10_author_names = HashSet::new();
        for (author_position, author_name) in authors.iter() {
            let normalized = normalize_text_compat_from_map(author_name, false, unidecode_char_map);
            if normalized.is_empty() {
                continue;
            }
            paper_author_names.insert(normalized.clone());
            if let Some(position) = position {
                if *author_position != position && (*author_position).abs_diff(position) <= 10 {
                    local10_author_names.insert(normalized);
                }
            }
        }
        (paper_author_names, local10_author_names)
    }

    fn reference_intersection(left: &HashSet<String>, right: &HashSet<String>) -> usize {
        if left.len() <= right.len() {
            left.iter().filter(|value| right.contains(*value)).count()
        } else {
            right.iter().filter(|value| left.contains(*value)).count()
        }
    }

    type ReferenceSignals = (HashSet<String>, HashSet<String>);

    /// `raw_arrow_paper_evidence_row` over string sets, as it was.
    fn reference_evidence_row(
        query_signature_id: &str,
        query_count: usize,
        query: &ReferenceSignals,
        members: &[(String, usize, ReferenceSignals)],
    ) -> [u32; 6] {
        let (mut jaccard, mut containment, mut overlap) = (0.0f64, 0.0f64, 0.0f64);
        let (mut local_jaccard, mut local_overlap) = (0.0f64, 0.0f64);
        let mut log_absdiff: Option<f64> = None;
        for (signature_id, count, (names, local)) in members {
            let intersection = reference_intersection(&query.0, names);
            let union = query.0.len() + names.len() - intersection;
            if union > 0 {
                jaccard = jaccard.max((intersection as f64) / (union as f64));
            }
            let denominator = query.0.len().min(names.len());
            if denominator > 0 {
                containment = containment.max((intersection as f64) / (denominator as f64));
            }
            overlap = overlap.max(intersection as f64);
            let delta = ((query_count as f64).ln_1p() - (*count as f64).ln_1p()).abs();
            log_absdiff = Some(log_absdiff.map_or(delta, |current| current.min(delta)));
            if query_signature_id == signature_id {
                continue;
            }
            let local_intersection = reference_intersection(&query.1, local);
            let local_union = query.1.len() + local.len() - local_intersection;
            if local_union > 0 {
                local_jaccard =
                    local_jaccard.max((local_intersection as f64) / (local_union as f64));
            }
            local_overlap = local_overlap.max(local_intersection as f64);
        }
        [
            round_six(jaccard).to_bits(),
            round_six(containment).to_bits(),
            round_six(overlap).to_bits(),
            round_six(local_jaccard).to_bits(),
            round_six(local_overlap).to_bits(),
            round_six(log_absdiff.unwrap_or(0.0)).to_bits(),
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

    /// Raw author strings with case, punctuation, accents, blanks and repeats;
    /// many normalize to the same name or block.
    fn raw_name(rng: &mut Lcg) -> String {
        const FIRSTS: [&str; 9] = [
            "Anna",
            "ANNA",
            "J.",
            "Jian",
            "Jos\u{e9}",
            "",
            "  ",
            "Li-Wei",
            "O\u{2019}Neil",
        ];
        const LASTS: [&str; 7] = [
            "Wang",
            "WANG",
            "Li",
            "Smith",
            "Garc\u{ed}a L\u{f3}pez",
            "",
            "van der Berg",
        ];
        let first = FIRSTS[rng.below(FIRSTS.len())];
        let last = LASTS[rng.below(LASTS.len())];
        format!("{first} {last}")
    }

    fn signature(paper_id: &str, position: Option<i64>) -> RawArrowSignature {
        RawArrowSignature {
            paper_id: paper_id.to_string(),
            author_first: String::new(),
            author_middle: String::new(),
            author_last: String::new(),
            author_suffix: String::new(),
            author_block: None,
            affiliations: Vec::new(),
            email: None,
            orcid: None,
            position,
        }
    }

    /// A 3,000-author paper with gapped, unsorted positions, and smaller papers
    /// drawn from the same names. Keys are paper ids.
    fn synthetic_papers() -> HashMap<String, Vec<(i64, String)>> {
        let mut rng = Lcg(3);
        let mut papers = HashMap::new();
        let mut giant: Vec<(i64, String)> = (0..3000)
            .map(|index| (index as i64 * 3 - 40, raw_name(&mut rng)))
            .collect();
        giant.reverse();
        papers.insert("giant".to_string(), giant);
        for paper in 0..60 {
            let count = 1 + rng.below(25);
            papers.insert(
                format!("small{paper}"),
                (0..count)
                    .map(|position| (position as i64, raw_name(&mut rng)))
                    .collect(),
            );
        }
        papers.insert(
            "blank".to_string(),
            vec![(0, String::new()), (1, "  ".to_string())],
        );
        papers
    }

    fn unidecode_map(papers: &HashMap<String, Vec<(i64, String)>>) -> HashMap<char, String> {
        let mut map = HashMap::new();
        for authors in papers.values() {
            for (_, name) in authors {
                crate::text_compat::ensure_unidecode_for_text(name, &mut map)
                    .expect("unidecode map");
            }
        }
        map
    }

    /// Signatures at every author position of each paper (every 37th on the
    /// giant paper), at a position no author holds, and without a position.
    fn synthetic_signatures(
        papers: &HashMap<String, Vec<(i64, String)>>,
    ) -> Vec<RawArrowSignature> {
        let mut paper_ids: Vec<&String> = papers.keys().collect();
        paper_ids.sort();
        let mut signatures = Vec::new();
        for paper_id in paper_ids {
            let stride = if paper_id == "giant" { 37 } else { 1 };
            for (position, _) in papers[paper_id].iter().step_by(stride) {
                signatures.push(signature(paper_id, Some(*position)));
            }
            signatures.push(signature(paper_id, Some(1_000_001)));
            signatures.push(signature(paper_id, None));
        }
        signatures
    }

    #[test]
    fn coauthor_terms_match_per_signature_blocks() {
        let papers = synthetic_papers();
        let map = unidecode_map(&papers);
        let signatures = synthetic_signatures(&papers);
        let blocks = paper_coauthor_blocks(&signatures, &papers, &map).expect("blocks");
        let mut nonempty = 0;
        for signature in &signatures {
            let paper = &blocks[&signature.paper_id];
            let observed: Vec<(u64, u8)> = paper
                .coauthor_terms(signature.position)
                .iter()
                .map(|term| (term.hash, term.token_count))
                .collect();
            let expected =
                reference_coauthor_terms(signature.position, &papers[&signature.paper_id], &map);
            assert_eq!(
                observed, expected,
                "{:?} {:?}",
                signature.paper_id, signature.position
            );
            assert_eq!(paper.author_count(), papers[&signature.paper_id].len());
            nonempty += usize::from(!observed.is_empty());
        }
        assert!(nonempty > 100, "{nonempty}");
    }

    #[test]
    fn author_signals_and_evidence_rows_match_string_sets() {
        let papers = synthetic_papers();
        let map = unidecode_map(&papers);
        let signatures = synthetic_signatures(&papers);
        let mut index = PaperAuthorIndex::default();
        index
            .extend(papers.keys().map(String::as_str), &papers, &map)
            .expect("index");
        let mut names_by_id = HashMap::new();
        for (shard, names) in index.name_shards.iter().enumerate() {
            for (name, local) in names {
                assert_eq!(name_shard(name), shard);
                let id = (local << NAME_SHARD_BITS) | shard as u32;
                assert!(names_by_id.insert(id, name.clone()).is_none());
            }
        }
        let as_strings = |ids: &[u32]| -> HashSet<String> {
            ids.iter().map(|id| names_by_id[id].clone()).collect()
        };

        let signals: Vec<RawArrowAuthorSignalData> = signatures
            .iter()
            .map(|signature| index.author_signals(signature).expect("indexed"))
            .collect();
        let references: Vec<ReferenceSignals> = signatures
            .iter()
            .map(|signature| {
                reference_author_signals(signature.position, &papers[&signature.paper_id], &map)
            })
            .collect();
        for ((observed, expected), signature) in signals.iter().zip(&references).zip(&signatures) {
            assert!(observed
                .paper_author_names
                .windows(2)
                .all(|pair| pair[0] < pair[1]));
            assert!(observed
                .local10_author_names
                .windows(2)
                .all(|pair| pair[0] < pair[1]));
            assert_eq!(as_strings(&observed.paper_author_names), expected.0);
            assert_eq!(
                as_strings(&observed.local10_author_names),
                expected.1,
                "{:?} {:?}",
                signature.paper_id,
                signature.position
            );
        }

        let ids: Vec<String> = (0..signatures.len())
            .map(|index| format!("s{index}"))
            .collect();
        let counts: Vec<usize> = signatures
            .iter()
            .map(|signature| papers[&signature.paper_id].len())
            .collect();
        let mut rng = Lcg(9);
        let mut fractional = 0;
        for _ in 0..400 {
            let query = rng.below(signatures.len());
            let members: Vec<usize> = (0..1 + rng.below(12))
                .map(|_| rng.below(signatures.len()))
                .collect();
            let summary = RawArrowSummarySignalData {
                name_counts_values: Vec::new(),
                member_paper_author_names: members
                    .iter()
                    .map(|member| Arc::clone(&signals[*member].paper_author_names))
                    .collect(),
                member_paper_author_counts: members.iter().map(|member| counts[*member]).collect(),
                member_local10_author_names: members
                    .iter()
                    .map(|member| signals[*member].local10_author_names.clone())
                    .collect(),
                member_signature_ids: members.iter().map(|member| ids[*member].clone()).collect(),
            };
            let row =
                raw_arrow_paper_evidence_row(&ids[query], counts[query], &signals[query], &summary);
            let observed = [
                row.paper_author_list_max_jaccard.to_bits(),
                row.paper_author_list_max_containment.to_bits(),
                row.paper_author_list_max_overlap_count.to_bits(),
                row.local_author_window10_jaccard_max.to_bits(),
                row.local_author_window10_overlap_count_max.to_bits(),
                row.best_author_count_log_absdiff.to_bits(),
            ];
            let reference_members: Vec<(String, usize, ReferenceSignals)> = members
                .iter()
                .map(|member| {
                    (
                        ids[*member].clone(),
                        counts[*member],
                        references[*member].clone(),
                    )
                })
                .collect();
            assert_eq!(
                observed,
                reference_evidence_row(
                    &ids[query],
                    counts[query],
                    &references[query],
                    &reference_members
                )
            );
            fractional += usize::from(
                row.paper_author_list_max_jaccard > 0.0 && row.paper_author_list_max_jaccard < 1.0,
            );
        }
        assert!(fractional > 20, "{fractional}");
    }

    #[test]
    fn sorted_intersection_count_matches_both_strategies() {
        let large: Vec<u32> = (0..2000).map(|value| value * 3).collect();
        let cases: [Vec<u32>; 3] = [
            Vec::new(),
            vec![0, 3, 4, 5999],
            (0..400).map(|value| value * 5).collect(),
        ];
        for small in cases {
            let expected = small
                .iter()
                .filter(|value| large.binary_search(value).is_ok())
                .count();
            assert_eq!(sorted_intersection_count(&small, &large), expected);
            assert_eq!(sorted_intersection_count(&large, &small), expected);
        }
    }

    #[test]
    fn unindexed_papers_are_errors_and_duplicate_positions_are_rejected() {
        let index = PaperAuthorIndex::default();
        assert!(index.author_signals(&signature("p", Some(0))).is_err());
        let papers = HashMap::from([(
            "p".to_string(),
            vec![(0, "anna li".to_string()), (0, "bo li".to_string())],
        )]);
        let mut index = PaperAuthorIndex::default();
        assert!(index.extend(["p"], &papers, &HashMap::new()).is_err());
        assert!(
            paper_coauthor_blocks([&signature("p", Some(0))], &papers, &HashMap::new()).is_err()
        );
    }
}
