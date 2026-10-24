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
const GRAM_SIMILARITY_DENOMINATOR_MAX_COUNT: u64 = 5000;
const GRAM_SIMILARITY_DENOMINATOR_MAX: f64 = GRAM_SIMILARITY_DENOMINATOR_MAX_COUNT as f64;

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

/// Character 2-4-gram counts of each interned author name, sorted by key.
/// Any fixed bijection of grams gives the same sums of minimums. With many
/// big papers, keys are frequency ranks, so `removed_gram_jaccard` reaches its
/// cap after few entries; otherwise they stay `gram_key`s.
#[derive(Default)]
pub(crate) struct NameGrams {
    offsets: Vec<u32>,
    entries: Vec<(u32, u32)>,
    /// Distinct grams over the names when keys are frequency ranks, which are
    /// then `0..ranked`. `None` when keys are `gram_key`s.
    ranked: Option<usize>,
}

impl NameGrams {
    fn empty(name_count: usize) -> Self {
        Self {
            offsets: vec![0; name_count + 1],
            entries: Vec::new(),
            ranked: None,
        }
    }

    /// `papers` holds each paper's interned author names.
    fn build(names: &[&str], papers: &[(&PaperId, Vec<(i64, u32)>)]) -> PyResult<Self> {
        let mut per_name = names
            .par_iter()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .map(|name| name_grams(name))
            .collect::<PyResult<Vec<_>>>()?;
        // Frequency ranks shorten only the merges of papers big enough to reach
        // the similarity cap. Ranking costs about as much as a build's other
        // coauthor work, so builds with few such papers skip it.
        let big_papers = papers
            .par_iter()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .filter(|(_, authors)| {
                authors
                    .iter()
                    .map(|(_, name)| per_name[*name as usize].len())
                    .sum::<usize>()
                    >= DENSE_GRAM_SUM_MIN_OCCURRENCES
            })
            .count();
        let ranked = if big_papers >= RANKED_GRAMS_MIN_BIG_PAPERS {
            let (ranks, ranked) = gram_ranks(&per_name);
            per_name
                .par_iter_mut()
                .with_min_len(MIN_ITEMS_PER_TASK)
                .for_each(|grams| {
                    for (key, _) in grams.iter_mut() {
                        *key = ranks[gram_index(*key)];
                    }
                    grams.sort_unstable_by_key(|(key, _)| *key);
                });
            Some(ranked)
        } else {
            None
        };
        let mut offsets = Vec::with_capacity(per_name.len() + 1);
        offsets.push(0u32);
        let mut entries = Vec::with_capacity(per_name.iter().map(Vec::len).sum());
        for grams in per_name {
            entries.extend(grams);
            offsets.push(u32::try_from(entries.len()).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("coauthor n-gram table exceeds u32 offsets")
            })?);
        }
        Ok(Self {
            offsets,
            entries,
            ranked,
        })
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

/// Number of grams of 2-4 lowercase ASCII letters.
const GRAM_SPACE: usize = 26 * 26 + 26 * 26 * 26 + 26 * 26 * 26 * 26;

/// Dense index of a `gram_key`: grams of each width in base-26 order,
/// shorter widths first.
fn gram_index(key: u32) -> usize {
    // Letters are nonzero bytes, so the highest nonzero byte fixes the width.
    let width = 4 - (key.leading_zeros() / 8) as usize;
    let offset = match width {
        2 => 0,
        3 => 26 * 26,
        _ => 26 * 26 + 26 * 26 * 26,
    };
    let digits = (0..width).rev().fold(0usize, |index, byte| {
        index * 26 + ((key >> (8 * byte)) & 0xff) as usize - usize::from(b'a')
    });
    offset + digits
}

/// Rank of every gram index that the names contain: grams by decreasing
/// total count over the names, ties by index. Only those entries are read.
/// Also returns how many grams the names contain.
fn gram_ranks(per_name: &[Vec<(u32, u32)>]) -> (Vec<u32>, usize) {
    let entries: usize = per_name.iter().map(Vec::len).sum();
    if entries < GRAM_SPACE {
        sparse_gram_ranks(per_name)
    } else {
        dense_gram_ranks(per_name, entries)
    }
}

/// `gram_ranks` from dense count tables over every gram.
fn dense_gram_ranks(per_name: &[Vec<(u32, u32)>], entries: usize) -> (Vec<u32>, usize) {
    // One dense table per chunk, and a chunk per million gram entries at
    // most: small requests build several featurizers, and a table per thread
    // cost them more than their whole gram work.
    let tables = (entries / GRAM_RANK_ENTRIES_PER_TABLE).clamp(1, rayon::current_num_threads());
    let chunk = per_name.len().div_ceil(tables).max(1);
    let partial: Vec<Vec<u64>> = per_name
        .par_chunks(chunk)
        .map(|names| {
            let mut totals = vec![0u64; GRAM_SPACE];
            for (key, count) in names.iter().flatten() {
                totals[gram_index(*key)] += u64::from(*count);
            }
            totals
        })
        .collect();
    let mut totals = vec![0u64; GRAM_SPACE];
    for table in partial {
        for (total, other) in totals.iter_mut().zip(table) {
            *total += other;
        }
    }
    // Grams that occur, by decreasing total then index, then the rest by
    // index: the order a sort of every gram by (decreasing total, index) gives.
    let mut occurring: Vec<u32> = (0..GRAM_SPACE as u32)
        .filter(|index| totals[*index as usize] > 0)
        .collect();
    occurring
        .par_sort_unstable_by_key(|index| (std::cmp::Reverse(totals[*index as usize]), *index));
    let ranked = occurring.len();
    let mut ranks = vec![0u32; GRAM_SPACE];
    let mut next = 0u32;
    for index in occurring {
        ranks[index as usize] = next;
        next += 1;
    }
    for (index, total) in totals.iter().enumerate() {
        if *total == 0 {
            ranks[index] = next;
            next += 1;
        }
    }
    (ranks, ranked)
}

/// `gram_ranks` for fewer entries than grams: sorting the entries is cheaper
/// than scanning dense tables over every gram, and gives the same ranks.
fn sparse_gram_ranks(per_name: &[Vec<(u32, u32)>]) -> (Vec<u32>, usize) {
    let mut counts: Vec<(u32, u64)> = per_name
        .iter()
        .flatten()
        .map(|(key, count)| (gram_index(*key) as u32, u64::from(*count)))
        .collect();
    counts.sort_unstable_by_key(|(index, _)| *index);
    let mut totals: Vec<(u32, u64)> = Vec::with_capacity(counts.len());
    for (index, count) in counts {
        match totals.last_mut() {
            Some((last, total)) if *last == index => *total += count,
            _ => totals.push((index, count)),
        }
    }
    totals.sort_unstable_by_key(|(index, total)| (std::cmp::Reverse(*total), *index));
    let ranked = totals.len();
    let mut ranks = vec![0u32; GRAM_SPACE];
    for (rank, (index, _)) in totals.into_iter().enumerate() {
        ranks[index as usize] = rank as u32;
    }
    (ranks, ranked)
}

/// Gram entries per dense count table in `gram_ranks`.
const GRAM_RANK_ENTRIES_PER_TABLE: usize = 1 << 20;

/// Fewest papers or names that one pool task takes: a small build then runs
/// on its calling worker instead of waking the whole pool for microseconds.
const MIN_ITEMS_PER_TASK: usize = 1024;

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
        let occurrences: usize = authors
            .iter()
            .map(|(_, name)| name_grams.of(Some(*name)).len())
            .sum();
        // Dense tables pay for themselves only across the many big papers of a
        // build that ranks grams by frequency.
        let (grams, gram_total) = match name_grams.ranked {
            Some(ranked) if occurrences >= DENSE_GRAM_SUM_MIN_OCCURRENCES => dense_gram_sum(
                authors.iter().map(|(_, name)| name_grams.of(Some(*name))),
                ranked,
            ),
            _ => {
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
                (grams, gram_total)
            }
        };
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

/// Big papers from which a build ranks grams by frequency: 9 in the s_feldman
/// block, 744 in s_lee's physics presplit block, thousands in big blocks.
const RANKED_GRAMS_MIN_BIG_PAPERS: usize = 64;

/// Gram occurrences from which a paper sums its grams in a dense table.
/// Thousand-author papers have tens of thousands of occurrences but far fewer
/// distinct grams, so sorting only the distinct ones is cheaper.
const DENSE_GRAM_SUM_MIN_OCCURRENCES: usize = 4096;

thread_local! {
    /// Per-thread count table over gram ranks; all zero between uses.
    static DENSE_GRAM_COUNTS: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether a worker allocated its `DENSE_GRAM_COUNTS` since the last release.
static DENSE_GRAM_COUNTS_ALLOCATED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The summed `(rank, count)` list of several sorted gram lists, sorted by
/// rank, and the total count: what a sort and merge of all entries gives.
/// `ranked` bounds the lists' ranks; the table needs no slot beyond it.
fn dense_gram_sum<'a>(
    lists: impl Iterator<Item = &'a [(u32, u32)]>,
    ranked: usize,
) -> (Vec<(u32, u32)>, u64) {
    DENSE_GRAM_COUNTS.with(|cell| {
        let mut counts = cell.borrow_mut();
        if counts.len() < ranked {
            counts.resize(ranked, 0);
            DENSE_GRAM_COUNTS_ALLOCATED.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let mut touched: Vec<u32> = Vec::new();
        let mut total = 0u64;
        for list in lists {
            for (key, count) in list {
                let slot = &mut counts[*key as usize];
                if *slot == 0 {
                    touched.push(*key);
                }
                *slot += u64::from(*count);
                total += u64::from(*count);
            }
        }
        touched.sort_unstable();
        let grams = touched
            .into_iter()
            .map(|key| {
                let count = std::mem::take(&mut counts[key as usize]);
                // A key's sum fits u32 whenever the paper passes the
                // MAX_PAPER_GRAM_TOTAL check; larger papers are rejected.
                (key, count.min(u64::from(u32::MAX)) as u32)
            })
            .collect();
        (grams, total)
    })
}

/// Build-time coauthor tables for the papers of a featurizer's signatures.
pub(crate) struct CoauthorTables {
    /// Papers in the order `build` received them.
    papers: Vec<PaperEntry>,
    /// Position of each paper in `papers`.
    index: HashMap<PaperId, usize>,
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
        // Intern names by sorting the distinct ones by hash: ids are positions
        // in that list. Features compare ids only for equality and use them as
        // sort keys of id sets, so any consistent interning gives the same
        // values; hashing and sorting run on the pool, unlike a serial map.
        let papers: Vec<(&'a PaperId, &'a [(i64, String)])> = papers.into_iter().collect();
        // Chunks drop their repeated names before the global sort: a big block
        // repeats each collaboration member's name across many papers, and one
        // entry per author occurrence took hundreds of megabytes.
        let chunk = papers
            .len()
            .div_ceil(4 * rayon::current_num_threads())
            .max(MIN_ITEMS_PER_TASK);
        let mut hashed: Vec<(u64, &str)> = papers
            .par_chunks(chunk)
            .flat_map_iter(|chunk| {
                let mut names: Vec<(u64, &str)> = chunk
                    .iter()
                    .flat_map(|(_, authors)| authors.iter())
                    .map(|(_, name)| (crate::fnv64(name.as_bytes()), name.as_str()))
                    .collect();
                names.sort_unstable();
                names.dedup();
                names
            })
            .collect();
        hashed.par_sort_unstable();
        hashed.dedup();
        if u32::try_from(hashed.len()).is_err() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "more than u32::MAX distinct coauthor names",
            ));
        }
        let name_id = |name: &str| -> u32 {
            let hash = crate::fnv64(name.as_bytes());
            let mut index = hashed.partition_point(|(entry, _)| *entry < hash);
            // Distinct names may share a hash; they sit together, sorted by name.
            while hashed[index].1 != name {
                index += 1;
            }
            index as u32
        };
        let paper_authors: Vec<(&PaperId, Vec<(i64, u32)>)> = papers
            .par_iter()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .map(|(paper_id, authors)| {
                let interned = authors
                    .iter()
                    .map(|(position, name)| (*position, name_id(name)))
                    .collect();
                (*paper_id, interned)
            })
            .collect();
        let names: Vec<&str> = hashed.iter().map(|(_, name)| *name).collect();
        drop(hashed);

        let block_names: Vec<String> = names
            .par_iter()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .map(|name| compute_block_compat(name))
            .collect();
        let mut distinct_blocks: Vec<(u64, &str)> = block_names
            .par_iter()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .map(|block| (crate::fnv64(block.as_bytes()), block.as_str()))
            .collect();
        distinct_blocks.par_sort_unstable();
        distinct_blocks.dedup();
        let name_blocks: Vec<u32> = block_names
            .par_iter()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .map(|block| {
                let hash = crate::fnv64(block.as_bytes());
                let mut index = distinct_blocks.partition_point(|(entry, _)| *entry < hash);
                while distinct_blocks[index].1 != block.as_str() {
                    index += 1;
                }
                index as u32
            })
            .collect();
        drop(distinct_blocks);
        drop(block_names);

        let name_grams = if with_grams {
            NameGrams::build(&names, &paper_authors)?
        } else {
            NameGrams::empty(names.len())
        };
        let paper_ids: Vec<PaperId> = paper_authors
            .par_iter()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .map(|(paper_id, _)| (*paper_id).clone())
            .collect();
        let papers = paper_authors
            .into_par_iter()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .map(|(paper_id, authors)| {
                PaperEntry::build(paper_id, authors, &name_blocks, &name_grams)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut index: HashMap<PaperId, usize> =
            HashMap::with_capacity_and_hasher(paper_ids.len(), Default::default());
        index.extend(
            paper_ids
                .into_iter()
                .enumerate()
                .map(|(position, paper_id)| (paper_id, position)),
        );
        // The dense count tables are only for this build; free them on every
        // worker rather than hold a table per thread for the process lifetime.
        // Builds without big papers allocated none, and skip waking the pool.
        if DENSE_GRAM_COUNTS_ALLOCATED.swap(false, std::sync::atomic::Ordering::Relaxed) {
            rayon::broadcast(|_| {
                DENSE_GRAM_COUNTS.with(|counts| drop(std::mem::take(&mut *counts.borrow_mut())))
            });
        }
        Ok(Self {
            papers,
            index,
            name_blocks,
            name_grams,
        })
    }

    /// Own-author record of a signature at `position` on `paper_id`.
    ///
    /// Panics if `paper_id` was not among the built papers.
    pub(crate) fn signature(&self, paper_id: &str, position: i64) -> SignatureCoauthors {
        let paper = &self.papers[*self
            .index
            .get(paper_id)
            .expect("every signature's paper has coauthor tables")];
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

    /// Per-paper aggregates for `PaperData`, in the order `build` received the
    /// papers, and the name gram table the featurizer keeps for subtracting
    /// own authors.
    pub(crate) fn into_paper_list(self) -> (Vec<PaperCoauthors>, NameGrams) {
        let papers = self
            .papers
            .into_iter()
            .map(|entry| entry.coauthors)
            .collect();
        (papers, self.name_grams)
    }

    #[cfg(test)]
    pub(crate) fn into_parts(self) -> (HashMap<PaperId, PaperCoauthors>, NameGrams) {
        let mut coauthors: Vec<Option<PaperCoauthors>> = self
            .papers
            .into_iter()
            .map(|entry| Some(entry.coauthors))
            .collect();
        let papers = self
            .index
            .into_iter()
            .map(|(paper_id, position)| {
                (
                    paper_id,
                    coauthors[position].take().expect("one index per paper"),
                )
            })
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

/// The three coauthor features of one pair; `gram_similarity` is NaN unless
/// `with_gram_similarity`, since callers that read only the set features skip
/// the costliest one.
pub(crate) fn coauthor_pair_features(
    paper1: &PaperCoauthors,
    signature1: &SignatureCoauthors,
    paper2: &PaperCoauthors,
    signature2: &SignatureCoauthors,
    name_grams: &NameGrams,
    with_gram_similarity: bool,
) -> CoauthorPairFeatures {
    CoauthorPairFeatures {
        block_overlap: removed_set_jaccard(
            &paper1.blocks,
            signature1.removed_block,
            &paper2.blocks,
            signature2.removed_block,
        ),
        gram_similarity: if with_gram_similarity {
            removed_gram_jaccard(
                paper1,
                name_grams.of(signature1.own_name),
                paper2,
                name_grams.of(signature2.own_name),
            )
        } else {
            f64::NAN
        },
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
    let (shorter, longer) = (set1.len().min(set2.len()), set1.len().max(set2.len()));
    #[cfg(target_arch = "x86_64")]
    if shorter >= ID_BLOCK
        && longer < shorter.saturating_mul(SIMD_MERGE_MAX_RATIO)
        && std::arch::is_x86_feature_detected!("avx2")
    {
        // SAFETY: AVX2 was detected above.
        return unsafe { sorted_intersection_count_avx2(set1, set2) };
    }
    let _ = (shorter, longer);
    let mut count = 0;
    for_each_common(
        set1,
        set2,
        |id| *id,
        |_, _| {
            count += 1;
            true
        },
    );
    count
}

/// Ids compared per side per step by `sorted_intersection_count_avx2`.
const ID_BLOCK: usize = 8;

/// Largest size ratio of two sorted lists merged with AVX2. A block merge
/// skips the longer list's blocks below the next shorter-list key at a
/// fraction of a cycle per entry, while a search per entry of the shorter
/// list costs about a hundred cycles, so the merge wins well past 4x.
const SIMD_MERGE_MAX_RATIO: usize = 128;

/// Number of ids two sorted, distinct id lists share.
///
/// Eight ids per side are compared all against all per step. Ids are
/// distinct within each list, so each shared id matches exactly once.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sorted_intersection_count_avx2(set1: &[u32], set2: &[u32]) -> usize {
    use std::arch::x86_64::*;
    debug_assert!(set1.windows(2).all(|pair| pair[0] < pair[1]));
    debug_assert!(set2.windows(2).all(|pair| pair[0] < pair[1]));
    let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let (n1, n2) = (set1.len(), set2.len());
    let (mut i, mut j) = (0usize, 0usize);
    let mut count = 0usize;
    while i + ID_BLOCK <= n1 && j + ID_BLOCK <= n2 {
        // Skip a block entirely below the other side's block, as
        // `gram_min_sum_avx2` does.
        if *set2.get_unchecked(j + ID_BLOCK - 1) < *set1.get_unchecked(i) {
            j += ID_BLOCK;
            continue;
        }
        if *set1.get_unchecked(i + ID_BLOCK - 1) < *set2.get_unchecked(j) {
            i += ID_BLOCK;
            continue;
        }
        let ids1 = _mm256_loadu_si256(set1.as_ptr().add(i) as *const __m256i);
        let ids2 = _mm256_loadu_si256(set2.as_ptr().add(j) as *const __m256i);
        let mut equal = _mm256_cmpeq_epi32(ids1, ids2);
        for rotation in 1..ID_BLOCK as i32 {
            let index = _mm256_and_si256(
                _mm256_add_epi32(lanes, _mm256_set1_epi32(rotation)),
                _mm256_set1_epi32(7),
            );
            let rotated = _mm256_permutevar8x32_epi32(ids2, index);
            equal = _mm256_or_si256(equal, _mm256_cmpeq_epi32(ids1, rotated));
        }
        count += (_mm256_movemask_ps(_mm256_castsi256_ps(equal)) as u32).count_ones() as usize;
        let last1 = *set1.get_unchecked(i + ID_BLOCK - 1);
        let last2 = *set2.get_unchecked(j + ID_BLOCK - 1);
        i += ID_BLOCK * usize::from(last1 <= last2);
        j += ID_BLOCK * usize::from(last2 <= last1);
    }
    while i < n1 && j < n2 {
        let (id1, id2) = (*set1.get_unchecked(i), *set2.get_unchecked(j));
        count += usize::from(id1 == id2);
        i += usize::from(id1 <= id2);
        j += usize::from(id2 <= id1);
    }
    count
}

/// `binary_search_by_key` for `target` in key-sorted `items`, costing the log
/// of the target's position rather than of the length.
fn gallop<T>(items: &[T], target: u32, key: impl Fn(&T) -> u32) -> Result<usize, usize> {
    let mut bound = 1usize;
    while bound <= items.len() && key(&items[bound - 1]) < target {
        bound <<= 1;
    }
    // items[..bound / 2] is below the target; items[bound - 1] is not.
    let start = bound >> 1;
    let end = bound.min(items.len());
    match items[start..end].binary_search_by_key(&target, key) {
        Ok(offset) => Ok(start + offset),
        Err(offset) => Err(start + offset),
    }
}

/// Call `on_common(item1, item2)` for every key the sorted, key-distinct
/// slices share, in increasing key order, until it returns false.
fn for_each_common<T>(
    items1: &[T],
    items2: &[T],
    key: impl Fn(&T) -> u32,
    mut on_common: impl FnMut(&T, &T) -> bool,
) {
    let (small, large, swapped) = if items1.len() <= items2.len() {
        (items1, items2, false)
    } else {
        (items2, items1, true)
    };
    let mut emit = |small_item: &T, large_item: &T| -> bool {
        if swapped {
            on_common(large_item, small_item)
        } else {
            on_common(small_item, large_item)
        }
    };
    if large.len() >= small.len().saturating_mul(4) {
        // Gallop from the previous match: a search costs the log of the gap
        // to the next key rather than of the whole remaining list.
        let mut low = 0;
        for small_item in small {
            let rest = &large[low..];
            match gallop(rest, key(small_item), &key) {
                Ok(offset) => {
                    if !emit(small_item, &rest[offset]) {
                        return;
                    }
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
        if small_key == large_key && !emit(&small[small_index], &large[large_index]) {
            return;
        }
        // Advance without a key-dependent branch; equal keys move both, as before.
        small_index += usize::from(small_key <= large_key);
        large_index += usize::from(large_key <= small_key);
    }
}

/// Gram entries compared per side per step by `gram_min_sum_avx2`.
const GRAM_BLOCK: usize = 8;

/// Sum, over the keys two sorted, key-distinct gram lists share, of the
/// smaller count. Stops once the sum reaches `stop_at`: the result is the
/// full sum when that is below `stop_at`, and otherwise a partial sum at or
/// above it.
///
/// Eight keys per side are compared all against all per step, and the
/// minimums of equal keys are added in lanes without branches. Lanes cannot
/// overflow: each adds at most the counts of one list, whose total fits
/// `MAX_PAPER_GRAM_TOTAL`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gram_min_sum_avx2(grams1: &[(u32, u32)], grams2: &[(u32, u32)], stop_at: u64) -> u64 {
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
    let lane_sum = |lanes: __m256i| -> u64 {
        let half = _mm_add_epi32(
            _mm256_castsi256_si128(lanes),
            _mm256_extracti128_si256::<1>(lanes),
        );
        let quarter = _mm_add_epi32(half, _mm_unpackhi_epi64(half, half));
        let total = _mm_add_epi32(quarter, _mm_shuffle_epi32::<0b01>(quarter));
        u64::from(_mm_cvtsi128_si32(total) as u32)
    };
    let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let (n1, n2) = (grams1.len(), grams2.len());
    let (mut i, mut j) = (0usize, 0usize);
    let mut sums = _mm256_setzero_si256();
    while i + GRAM_BLOCK <= n1 && j + GRAM_BLOCK <= n2 {
        // A block entirely below the other side's block shares no key with
        // it or with anything after it; skip it without the compare. Skewed
        // lists spend most steps here.
        if grams2.get_unchecked(j + GRAM_BLOCK - 1).0 < grams1.get_unchecked(i).0 {
            j += GRAM_BLOCK;
            continue;
        }
        if grams1.get_unchecked(i + GRAM_BLOCK - 1).0 < grams2.get_unchecked(j).0 {
            i += GRAM_BLOCK;
            continue;
        }
        let (keys1, counts1) = load(grams1, i);
        let (keys2, counts2) = load(grams2, j);
        for rotation in 0..GRAM_BLOCK as i32 {
            let index = _mm256_and_si256(
                _mm256_add_epi32(lanes, _mm256_set1_epi32(rotation)),
                _mm256_set1_epi32(7),
            );
            let equal = _mm256_cmpeq_epi32(keys1, _mm256_permutevar8x32_epi32(keys2, index));
            let smaller = _mm256_min_epu32(counts1, _mm256_permutevar8x32_epi32(counts2, index));
            sums = _mm256_add_epi32(sums, _mm256_and_si256(equal, smaller));
        }
        if lane_sum(sums) >= stop_at {
            return lane_sum(sums);
        }
        let last1 = grams1.get_unchecked(i + GRAM_BLOCK - 1).0;
        let last2 = grams2.get_unchecked(j + GRAM_BLOCK - 1).0;
        i += GRAM_BLOCK * usize::from(last1 <= last2);
        j += GRAM_BLOCK * usize::from(last2 <= last1);
    }
    let mut sum = lane_sum(sums);
    while i < n1 && j < n2 && sum < stop_at {
        let (key1, count1) = *grams1.get_unchecked(i);
        let (key2, count2) = *grams2.get_unchecked(j);
        if key1 == key2 {
            sum += u64::from(count1.min(count2));
        }
        i += usize::from(key1 <= key2);
        j += usize::from(key2 <= key1);
    }
    sum
}

/// Count of `key` in a sorted gram list, found by galloping from the
/// previous lookup; keys must be looked up in increasing order.
struct GramCursor<'a> {
    grams: &'a [(u32, u32)],
    next: usize,
}

impl GramCursor<'_> {
    fn count(&mut self, key: u32) -> u32 {
        let rest = &self.grams[self.next..];
        match gallop(rest, key, |(gram, _)| *gram) {
            Ok(offset) => {
                self.next += offset + 1;
                rest[offset].1
            }
            Err(offset) => {
                self.next += offset;
                0
            }
        }
    }
}

/// Over the grams of either own author that both papers have: the smaller
/// paper count minus the smaller count without the own authors. Subtracted
/// from the sum of smaller paper counts, it gives the coauthor intersection.
fn own_gram_excess(
    paper1: &PaperCoauthors,
    own1: &[(u32, u32)],
    paper2: &PaperCoauthors,
    own2: &[(u32, u32)],
) -> u64 {
    let mut counts1 = GramCursor {
        grams: &paper1.grams,
        next: 0,
    };
    let mut counts2 = GramCursor {
        grams: &paper2.grams,
        next: 0,
    };
    let (mut a, mut b) = (0usize, 0usize);
    let mut excess = 0u64;
    while a < own1.len() || b < own2.len() {
        let key = match (own1.get(a), own2.get(b)) {
            (Some((key1, _)), Some((key2, _))) => (*key1).min(*key2),
            (Some((key, _)), None) | (None, Some((key, _))) => *key,
            (None, None) => unreachable!("the loop runs while either list has keys"),
        };
        let own_count = |own: &[(u32, u32)], at: &mut usize| match own.get(*at) {
            Some((own_key, count)) if *own_key == key => {
                *at += 1;
                *count
            }
            _ => 0,
        };
        let (own_count1, own_count2) = (own_count(own1, &mut a), own_count(own2, &mut b));
        let (count1, count2) = (counts1.count(key), counts2.count(key));
        if count1 > 0 && count2 > 0 {
            // Each paper's count includes its own author's.
            excess +=
                u64::from(count1.min(count2) - (count1 - own_count1).min(count2 - own_count2));
        }
    }
    excess
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
    removed_gram_jaccard_with(paper1, own1, paper2, own2, true)
}

/// `removed_gram_jaccard`, with the AVX2 path when `allow_simd` and the lists suit it.
fn removed_gram_jaccard_with(
    paper1: &PaperCoauthors,
    own1: &[(u32, u32)],
    paper2: &PaperCoauthors,
    own2: &[(u32, u32)],
    allow_simd: bool,
) -> f64 {
    let own_total = |own: &[(u32, u32)]| own.iter().map(|(_, count)| *count).sum::<u32>();
    let (own_total1, own_total2) = (own_total(own1), own_total(own2));
    let sum1 = paper1.gram_total - own_total1;
    let sum2 = paper2.gram_total - own_total2;
    if sum1 == 0 || sum2 == 0 {
        return f64::NAN;
    }
    // The intersection is at most each side's total, so once it reaches the
    // cap the union is above the cap too, and the score is clamped to exactly
    // 1.0. Terms are nonnegative, so a partial sum at the cap decides it.
    // Large author lists reach the cap early: gram keys are frequency ranks.
    let (shorter, longer) = if paper1.grams.len() <= paper2.grams.len() {
        (paper1.grams.len(), paper2.grams.len())
    } else {
        (paper2.grams.len(), paper1.grams.len())
    };
    #[cfg(target_arch = "x86_64")]
    if allow_simd
        && shorter >= GRAM_BLOCK
        && longer < shorter.saturating_mul(SIMD_MERGE_MAX_RATIO)
        && std::arch::is_x86_feature_detected!("avx2")
    {
        // Removing the own authors lowers a shared gram's smaller count by
        // at most its own counts, so the intersection is at least the paper
        // sum minus both own totals; past this bound it is at the cap.
        let stop_at =
            GRAM_SIMILARITY_DENOMINATOR_MAX_COUNT + u64::from(own_total1) + u64::from(own_total2);
        // SAFETY: AVX2 was detected above.
        let shared = unsafe { gram_min_sum_avx2(&paper1.grams, &paper2.grams, stop_at) };
        if shared >= stop_at {
            return 1.0;
        }
        let intersection = shared - own_gram_excess(paper1, own1, paper2, own2);
        return capped_gram_jaccard(intersection, sum1, sum2);
    }
    let _ = (allow_simd, shorter, longer);
    let mut own1 = OwnGramCursor {
        grams: own1,
        next: 0,
    };
    let mut own2 = OwnGramCursor {
        grams: own2,
        next: 0,
    };
    let mut intersection = 0u64;
    for_each_common(
        &paper1.grams,
        &paper2.grams,
        |(key, _)| *key,
        |(key, count1), (_, count2)| {
            let count1 = count1 - own1.count(*key);
            let count2 = count2 - own2.count(*key);
            intersection += u64::from(count1.min(count2));
            intersection < GRAM_SIMILARITY_DENOMINATOR_MAX_COUNT
        },
    );
    capped_gram_jaccard(intersection, sum1, sum2)
}

/// `counter_jaccard` with the capped denominator, from integer totals.
fn capped_gram_jaccard(intersection: u64, sum1: u32, sum2: u32) -> f64 {
    if intersection >= GRAM_SIMILARITY_DENOMINATOR_MAX_COUNT {
        return 1.0;
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
                    true,
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

    fn random_grams(rng: &mut Lcg, universe: usize, max_len: usize) -> Vec<(u32, u32)> {
        let len = rng.below(max_len);
        let mut keys: Vec<u32> = (0..len).map(|_| rng.below(universe) as u32).collect();
        keys.sort_unstable();
        keys.dedup();
        keys.into_iter()
            .map(|key| (key, 1 + rng.below(9) as u32))
            .collect()
    }

    #[test]
    fn merges_find_exactly_the_shared_keys() {
        let mut rng = Lcg(23);
        for _ in 0..5_000 {
            let universe = 1 + rng.below(3_000);
            let first = random_grams(&mut rng, universe, 700);
            let second = random_grams(&mut rng, universe, 700);
            let second_counts: HashMap<u32, u32> = second.iter().copied().collect();
            let naive: Vec<(u32, u32, u32)> = first
                .iter()
                .filter_map(|(key, count1)| {
                    second_counts
                        .get(key)
                        .map(|count2| (*key, *count1, *count2))
                })
                .collect();
            // The merge, including its galloping path for skewed sizes.
            let mut merged = Vec::new();
            for_each_common(
                &first,
                &second,
                |(key, _)| *key,
                |(key, count1), (_, count2)| {
                    merged.push((*key, *count1, *count2));
                    true
                },
            );
            assert_eq!(merged, naive);
            let ids1: Vec<u32> = first.iter().map(|(key, _)| *key).collect();
            let ids2: Vec<u32> = second.iter().map(|(key, _)| *key).collect();
            assert_eq!(sorted_intersection_count(&ids1, &ids2), naive.len());
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 was detected above.
                let count = unsafe { sorted_intersection_count_avx2(&ids1, &ids2) };
                assert_eq!(count, naive.len());
                let full: u64 = naive
                    .iter()
                    .map(|(_, count1, count2)| u64::from((*count1).min(*count2)))
                    .sum();
                // SAFETY: AVX2 was detected above.
                assert_eq!(
                    unsafe { gram_min_sum_avx2(&first, &second, u64::MAX) },
                    full
                );
                let stop_at = rng.below(full as usize + 2) as u64;
                // SAFETY: AVX2 was detected above.
                let partial = unsafe { gram_min_sum_avx2(&first, &second, stop_at) };
                if full < stop_at {
                    assert_eq!(partial, full);
                } else {
                    assert!(
                        stop_at <= partial && partial <= full,
                        "{stop_at} {partial} {full}"
                    );
                }
            }
        }
    }

    #[test]
    fn simd_gram_jaccard_matches_the_scalar_merge() {
        let mut rng = Lcg(29);
        // Own grams are a sub-counter of their paper's grams.
        let own_of = |rng: &mut Lcg, grams: &[(u32, u32)]| -> Vec<(u32, u32)> {
            let mut own = Vec::new();
            for (key, count) in grams {
                if rng.below(6) == 0 {
                    own.push((*key, 1 + rng.below(*count as usize) as u32));
                }
            }
            own
        };
        let paper = |grams: Vec<(u32, u32)>| {
            let gram_total = grams.iter().map(|(_, count)| *count).sum();
            PaperCoauthors {
                names: Vec::new(),
                blocks: Vec::new(),
                grams,
                gram_total,
            }
        };
        let mut saturated = 0;
        for round in 0..20_000 {
            let universe = 1 + rng.below(4_000);
            let scale = if round % 2 == 0 { 40 } else { 1 };
            let mut first = random_grams(&mut rng, universe, 1_500);
            let mut second = random_grams(&mut rng, universe, 1_500);
            for (_, count) in first.iter_mut().chain(second.iter_mut()) {
                *count *= scale;
            }
            let (own1, own2) = (own_of(&mut rng, &first), own_of(&mut rng, &second));
            let (paper1, paper2) = (paper(first), paper(second));
            let simd = removed_gram_jaccard_with(&paper1, &own1, &paper2, &own2, true);
            let scalar = removed_gram_jaccard_with(&paper1, &own1, &paper2, &own2, false);
            assert_eq!(simd.to_bits(), scalar.to_bits(), "round {round}");
            saturated += usize::from(simd == 1.0);
        }
        assert!(saturated > 1_000, "{saturated}");
    }

    #[test]
    fn saturated_gram_similarity_is_exactly_one() {
        // Two papers whose shared gram mass passes the cap long before the end.
        let grams = |scale: u32| -> Vec<(u32, u32)> {
            (0..2_000u32)
                .map(|key| (key * 3 + 1, scale + key % 7))
                .collect()
        };
        let paper = |scale: u32| {
            let grams = grams(scale);
            let gram_total = grams.iter().map(|(_, count)| *count).sum();
            PaperCoauthors {
                names: Vec::new(),
                blocks: Vec::new(),
                grams,
                gram_total,
            }
        };
        let (paper1, paper2) = (paper(40), paper(55));
        let own: Vec<(u32, u32)> = vec![(1, 2), (4, 1), (5_998, 3)];
        assert_eq!(removed_gram_jaccard(&paper1, &own, &paper2, &[]), 1.0);
        // Below the cap the merge runs to the end and matches the formula.
        let small = |scale: u32| {
            let grams: Vec<(u32, u32)> = grams(scale).into_iter().take(40).collect();
            let gram_total = grams.iter().map(|(_, count)| *count).sum();
            PaperCoauthors {
                names: Vec::new(),
                blocks: Vec::new(),
                grams,
                gram_total,
            }
        };
        let (small1, small2) = (small(2), small(3));
        let intersection: u32 = small1
            .grams
            .iter()
            .zip(&small2.grams)
            .map(|((_, count1), (_, count2))| (*count1).min(*count2))
            .sum();
        let union = small1.gram_total + small2.gram_total - intersection;
        assert_eq!(
            removed_gram_jaccard(&small1, &[], &small2, &[]),
            f64::from(intersection) / f64::from(union)
        );
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
                    true,
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
        // gram_index maps the keys one to one onto 0..GRAM_SPACE.
        let mut indices: Vec<usize> = keys.iter().map(|key| gram_index(*key)).collect();
        indices.sort_unstable();
        assert_eq!(indices, (0..GRAM_SPACE).collect::<Vec<_>>());
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
            true,
        );
        // Both coauthor sets are {""} and both block sets {""}; "" has no n-grams.
        assert_eq!(features.name_match, 1.0);
        assert_eq!(features.block_overlap, 1.0);
        assert!(features.gram_similarity.is_nan());
    }

    #[test]
    fn gram_rank_order_does_not_change_coauthor_features() {
        // Unrelated big papers change every gram's rank but no small paper's features.
        let mut rng = Lcg(41);
        let mut name = |rng: &mut Lcg| -> String {
            let word = |rng: &mut Lcg| -> String {
                (0..2 + rng.below(8))
                    .map(|_| char::from(b'a' + rng.below(26) as u8))
                    .collect()
            };
            format!("{} {}", word(rng), word(rng))
        };
        let small: Vec<(String, Vec<(i64, String)>)> = (0..40)
            .map(|paper| {
                let authors = (0..1 + rng.below(30))
                    .map(|position| (position as i64, name(&mut rng)))
                    .collect();
                (format!("p{paper}"), authors)
            })
            .collect();
        let big: Vec<(String, Vec<(i64, String)>)> = (0..RANKED_GRAMS_MIN_BIG_PAPERS)
            .map(|paper| {
                let authors = (0..300)
                    .map(|position| (position, name(&mut rng)))
                    .collect();
                (format!("big{paper}"), authors)
            })
            .collect();
        let features = |papers: &[&(String, Vec<(i64, String)>)]| -> (Vec<u64>, Option<usize>) {
            let tables = CoauthorTables::build(
                papers
                    .iter()
                    .map(|(paper_id, authors)| (paper_id, authors.as_slice())),
                true,
            )
            .expect("papers build");
            let signatures: Vec<SignatureCoauthors> = small
                .iter()
                .map(|(paper_id, _)| tables.signature(paper_id, 0))
                .collect();
            let (paper_coauthors, name_grams) = tables.into_parts();
            let ranked = name_grams.ranked;
            let mut values = Vec::new();
            for (first, (first_id, _)) in small.iter().enumerate() {
                for (second, (second_id, _)) in small.iter().enumerate() {
                    let pair = coauthor_pair_features(
                        &paper_coauthors[first_id],
                        &signatures[first],
                        &paper_coauthors[second_id],
                        &signatures[second],
                        &name_grams,
                        true,
                    );
                    values.push(pair.gram_similarity.to_bits());
                }
            }
            (values, ranked)
        };
        let index_order: Vec<_> = small.iter().collect();
        let mut frequency_order = index_order.clone();
        frequency_order.extend(&big);
        let ((by_index, index_ranked), (by_frequency, frequency_ranked)) =
            (features(&index_order), features(&frequency_order));
        // The big papers switch the build from gram indices to frequency ranks.
        assert_eq!(index_ranked, None);
        assert!(frequency_ranked.is_some_and(|ranked| ranked < GRAM_SPACE));
        assert_eq!(by_index, by_frequency);
    }

    #[test]
    fn sparse_gram_ranks_match_the_dense_tables() {
        let names: Vec<String> = (0..300)
            .map(|index| {
                let first = ["anna", "li", "wei", "jose", "maria", "ng"][index % 6];
                let last = ["smith", "zhang", "garcia", "o", "kowalski"][index % 5];
                format!("{first} {last}{}", "abc".repeat(index % 4))
            })
            .collect();
        let per_name: Vec<Vec<(u32, u32)>> = names
            .iter()
            .map(|name| name_grams(name).expect("normalized name"))
            .collect();
        let entries = per_name.iter().map(Vec::len).sum();
        let ((sparse, sparse_ranked), (dense, dense_ranked)) = (
            sparse_gram_ranks(&per_name),
            dense_gram_ranks(&per_name, entries),
        );
        assert_eq!(sparse_ranked, dense_ranked);
        for (key, _) in per_name.iter().flatten() {
            assert!((sparse[gram_index(*key)] as usize) < sparse_ranked);
            assert_eq!(sparse[gram_index(*key)], dense[gram_index(*key)]);
        }
    }

    #[test]
    fn name_grams_reject_unnormalized_names() {
        assert!(name_grams("anna li").is_ok());
        assert!(name_grams("Anna").is_err());
        assert!(name_grams("an\u{00e9}").is_err());
    }
}
