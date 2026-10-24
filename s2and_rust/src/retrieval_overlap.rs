//! Coauthor and affiliation overlap terms of the hybrid retrieval score.
//!
//! `score_hybrid_centroid_query` scores every candidate component. Its
//! coauthor and affiliation terms are weighted overlaps between the query's
//! terms and a component's term counter. Computed per component, they cost a
//! weight lookup and a counter binary search for every query term and every
//! component, so a query from a paper with thousands of authors costs
//! thousands of lookups per component.
//!
//! Term weights and their sum depend only on the query, so `PreparedQuery`
//! computes them once. `OverlapIndex` inverts the counters: for each term
//! hash, the components that hold it and the term's capped share of each.
//! A query walks the postings of its own terms in query-term order and adds
//! each term's weighted share to that component's numerator, which is the
//! order in which the per-component loop adds them, so the floating-point
//! sums are the same. Hit counts for the mega-author rescue rule are integer
//! counts of the same lookups. Scores are bit-identical either way.

use crate::HashMap;
use rayon::prelude::*;

use crate::{
    overlap_query_term_weight, CounterData, RetrievalError, RetrievalOverlapConfig,
    RetrievalQueryData, RetrievalQueryTerm, RetrievalSummaryData, AFFILIATION_OVERLAP_CONFIG,
    COAUTHOR_OVERLAP_CONFIG, RETRIEVAL_MEGA_AUTHOR_THRESHOLD,
};

/// Query term weights for one counter family, and their sum.
pub(crate) struct PreparedOverlap {
    weights: Vec<f64>,
    denominator: f64,
}

impl PreparedOverlap {
    fn new(
        terms: &[RetrievalQueryTerm],
        df_map: &HashMap<u64, usize>,
        total_summary_count: usize,
        config: RetrievalOverlapConfig,
    ) -> Self {
        let weights: Vec<f64> = terms
            .iter()
            .map(|term| overlap_query_term_weight(term, df_map, total_summary_count, config))
            .collect();
        let mut denominator = 0.0f64;
        for weight in weights.iter() {
            if *weight <= 0.0 {
                continue;
            }
            denominator += weight;
        }
        Self {
            weights,
            denominator,
        }
    }

    fn finish(&self, numerator: f64, config: RetrievalOverlapConfig) -> f64 {
        if self.denominator <= 0.0 {
            return 0.0;
        }
        let mut score = numerator / self.denominator;
        if let Some(cap) = config.total_cap {
            score = score.min(cap);
        }
        score
    }
}

/// The weights one query needs to score any component.
pub(crate) struct PreparedQuery {
    coauthor: PreparedOverlap,
    non_mega_coauthor: PreparedOverlap,
    affiliation: PreparedOverlap,
}

impl PreparedQuery {
    pub(crate) fn new(
        query: &RetrievalQueryData,
        coauthor_cluster_df: &HashMap<u64, usize>,
        non_mega_coauthor_cluster_df: &HashMap<u64, usize>,
        affiliation_cluster_df: &HashMap<u64, usize>,
        total_summary_count: usize,
    ) -> Self {
        Self {
            coauthor: PreparedOverlap::new(
                &query.coauthor_terms,
                coauthor_cluster_df,
                total_summary_count,
                COAUTHOR_OVERLAP_CONFIG,
            ),
            non_mega_coauthor: PreparedOverlap::new(
                &query.coauthor_terms,
                non_mega_coauthor_cluster_df,
                total_summary_count,
                COAUTHOR_OVERLAP_CONFIG,
            ),
            affiliation: PreparedOverlap::new(
                &query.affiliation_terms,
                affiliation_cluster_df,
                total_summary_count,
                AFFILIATION_OVERLAP_CONFIG,
            ),
        }
    }
}

/// One query term's share of a component: `count / size`, capped per term.
fn term_share(count: f32, size: usize, config: RetrievalOverlapConfig) -> f64 {
    let mut share = (count as f64) / (size as f64);
    if let Some(cap) = config.per_term_cap {
        share = share.min(cap);
    }
    share
}

/// A query's lookups in one component's three counters.
#[derive(Clone, Copy, Default)]
pub(crate) struct ComponentOverlap {
    /// Query coauthor terms present in the full coauthor counter.
    coauthor_hits: u32,
    /// Query coauthor terms present in the non-mega coauthor counter.
    non_mega_coauthor_hits: u32,
    /// Sums of `weight * share` over present positive-weight terms, in query-term order.
    coauthor: f64,
    non_mega_coauthor: f64,
    affiliation: f64,
}

fn counter_hits_and_numerator(
    terms: &[RetrievalQueryTerm],
    weights: &[f64],
    counter: &Option<CounterData>,
    size: usize,
    config: RetrievalOverlapConfig,
) -> (u32, f64) {
    let Some(counter) = counter.as_ref() else {
        return (0, 0.0);
    };
    let (mut hits, mut numerator) = (0u32, 0.0f64);
    for (term, weight) in terms.iter().zip(weights) {
        if let Ok(index) = counter
            .keys.binary_search(&term.hash)
        {
            hits += 1;
            if *weight > 0.0 {
                numerator += weight * term_share(counter.values[index], size, config);
            }
        }
    }
    (hits, numerator)
}

/// A query's lookups in one component, by binary search in its counters.
pub(crate) fn component_overlap(
    query: &RetrievalQueryData,
    prepared: &PreparedQuery,
    summary: &RetrievalSummaryData,
) -> ComponentOverlap {
    let (coauthor_hits, coauthor) = counter_hits_and_numerator(
        &query.coauthor_terms,
        &prepared.coauthor.weights,
        &summary.coauthor_counts,
        summary.size,
        COAUTHOR_OVERLAP_CONFIG,
    );
    let (non_mega_coauthor_hits, non_mega_coauthor) = counter_hits_and_numerator(
        &query.coauthor_terms,
        &prepared.non_mega_coauthor.weights,
        &summary.non_mega_coauthor_counts,
        summary.size,
        COAUTHOR_OVERLAP_CONFIG,
    );
    let (_, affiliation) = counter_hits_and_numerator(
        &query.affiliation_terms,
        &prepared.affiliation.weights,
        &summary.affiliation_counts,
        summary.size,
        AFFILIATION_OVERLAP_CONFIG,
    );
    ComponentOverlap {
        coauthor_hits,
        non_mega_coauthor_hits,
        coauthor,
        non_mega_coauthor,
        affiliation,
    }
}

/// Whether a candidate with mega-author papers keeps its full coauthor counter:
/// at least three query coauthor terms hit it, more than hit its non-mega
/// counter, and they are at least 99.5% of the query's coauthor terms.
pub(crate) fn rescues_mega_coauthors(
    query_term_count: usize,
    full_overlap: usize,
    filtered_overlap: usize,
) -> bool {
    if full_overlap < 3 {
        return false;
    }
    if full_overlap <= filtered_overlap {
        return false;
    }
    (full_overlap as f64) / (query_term_count as f64) >= 0.995
}

fn scored_overlap(
    terms: &[RetrievalQueryTerm],
    prepared: &PreparedOverlap,
    counter: &Option<CounterData>,
    size: usize,
    numerator: f64,
    config: RetrievalOverlapConfig,
) -> f64 {
    let Some(counter) = counter.as_ref() else {
        return 0.0;
    };
    if size == 0 || terms.is_empty() || counter.is_empty() {
        return 0.0;
    }
    prepared.finish(numerator, config)
}

/// The coauthor and affiliation overlap scores of one component.
pub(crate) fn overlap_scores(
    query: &RetrievalQueryData,
    prepared: &PreparedQuery,
    summary: &RetrievalSummaryData,
    overlap: &ComponentOverlap,
) -> (f64, f64) {
    let mega = summary.max_paper_author_count >= RETRIEVAL_MEGA_AUTHOR_THRESHOLD;
    let rescued = mega
        && !query.coauthor_terms.is_empty()
        && rescues_mega_coauthors(
            query.coauthor_terms.len(),
            overlap.coauthor_hits as usize,
            overlap.non_mega_coauthor_hits as usize,
        );
    let coauthor = if mega && !rescued {
        scored_overlap(
            &query.coauthor_terms,
            &prepared.non_mega_coauthor,
            &summary.non_mega_coauthor_counts,
            summary.size,
            overlap.non_mega_coauthor,
            COAUTHOR_OVERLAP_CONFIG,
        )
    } else {
        scored_overlap(
            &query.coauthor_terms,
            &prepared.coauthor,
            &summary.coauthor_counts,
            summary.size,
            overlap.coauthor,
            COAUTHOR_OVERLAP_CONFIG,
        )
    };
    let affiliation = scored_overlap(
        &query.affiliation_terms,
        &prepared.affiliation,
        &summary.affiliation_counts,
        summary.size,
        overlap.affiliation,
        AFFILIATION_OVERLAP_CONFIG,
    );
    (coauthor, affiliation)
}

/// Components and shares of each term of one counter family.
#[derive(Default)]
struct CounterPostings {
    /// Sorted distinct term hashes.
    hashes: Vec<u64>,
    /// Postings of `hashes[i]` are `offsets[i]..offsets[i + 1]`.
    offsets: Vec<u32>,
    components: Vec<u32>,
    shares: Vec<f64>,
}

const COUNTER_FAMILIES: [fn(&RetrievalSummaryData) -> &Option<CounterData>; 3] = [
    |summary| &summary.coauthor_counts,
    |summary| &summary.non_mega_coauthor_counts,
    |summary| &summary.affiliation_counts,
];

/// Check what `CounterPostings::build` relies on: component and posting
/// counts fit in u32, and every counter's hashes are strictly increasing, so a
/// binary search and a posting agree on each term's entry.
fn validate_overlap_counters(summaries: &[RetrievalSummaryData]) -> Result<(), RetrievalError> {
    if u32::try_from(summaries.len()).is_err() {
        return Err(RetrievalError::Overflow(
            "retriever component count exceeds u32".to_string(),
        ));
    }
    for counter_of in COUNTER_FAMILIES {
        let mut postings = 0usize;
        for summary in summaries {
            let Some(counter) = counter_of(summary).as_ref() else {
                continue;
            };
            if let Some(pair) = counter.keys.windows(2).find(|pair| pair[0] >= pair[1]) {
                return Err(RetrievalError::InvalidValue(format!(
                    "component_key {:?} counter hashes must be strictly increasing, got {} then {}; duplicate FNV hashes may indicate a collision",
                    summary.component_key, pair[0], pair[1]
                )));
            }
            postings += counter.len();
        }
        if u32::try_from(postings).is_err() {
            return Err(RetrievalError::Overflow(
                "retriever overlap postings exceed u32".to_string(),
            ));
        }
    }
    Ok(())
}

impl CounterPostings {
    /// Postings of summaries that passed `validate_overlap_counters`.
    fn build(
        summaries: &[RetrievalSummaryData],
        counter_of: fn(&RetrievalSummaryData) -> &Option<CounterData>,
        config: RetrievalOverlapConfig,
    ) -> Self {
        let mut postings: Vec<(u64, u32, f64)> = Vec::new();
        for (component, summary) in summaries.iter().enumerate() {
            let Some(counter) = counter_of(summary).as_ref() else {
                continue;
            };
            postings.extend(counter.iter().map(|(hash, count)| {
                (
                    *hash,
                    component as u32,
                    term_share(*count, summary.size, config),
                )
            }));
        }
        postings.par_sort_unstable_by_key(|(hash, component, _)| (*hash, *component));
        let posting_count = postings.len() as u32;
        let mut index = Self {
            components: Vec::with_capacity(postings.len()),
            shares: Vec::with_capacity(postings.len()),
            ..Self::default()
        };
        for (position, (hash, component, share)) in postings.into_iter().enumerate() {
            if index.hashes.last() != Some(&hash) {
                index.hashes.push(hash);
                index.offsets.push(position as u32);
            }
            index.components.push(component);
            index.shares.push(share);
        }
        index.offsets.push(posting_count);
        index
    }

    /// Add each term's postings to `overlaps`, in `terms` order.
    fn accumulate(
        &self,
        terms: &[RetrievalQueryTerm],
        weights: &[f64],
        overlaps: &mut [ComponentOverlap],
        add: fn(&mut ComponentOverlap, f64, f64),
    ) {
        for (term, weight) in terms.iter().zip(weights) {
            let Ok(position) = self.hashes.binary_search(&term.hash) else {
                continue;
            };
            let range = self.offsets[position] as usize..self.offsets[position + 1] as usize;
            for (component, share) in self.components[range.clone()]
                .iter()
                .zip(&self.shares[range])
            {
                add(&mut overlaps[*component as usize], *weight, *share);
            }
        }
    }
}

/// Inverted coauthor, non-mega coauthor and affiliation counters of a retriever.
#[derive(Default)]
pub(crate) struct OverlapIndex {
    coauthor: CounterPostings,
    non_mega_coauthor: CounterPostings,
    affiliation: CounterPostings,
}

impl OverlapIndex {
    pub(crate) fn build(summaries: &[RetrievalSummaryData]) -> Result<Self, RetrievalError> {
        validate_overlap_counters(summaries)?;
        let [coauthor_of, non_mega_coauthor_of, affiliation_of] = COUNTER_FAMILIES;
        let ((coauthor, non_mega_coauthor), affiliation) = rayon::join(
            || {
                rayon::join(
                    || CounterPostings::build(summaries, coauthor_of, COAUTHOR_OVERLAP_CONFIG),
                    || {
                        CounterPostings::build(
                            summaries,
                            non_mega_coauthor_of,
                            COAUTHOR_OVERLAP_CONFIG,
                        )
                    },
                )
            },
            || CounterPostings::build(summaries, affiliation_of, AFFILIATION_OVERLAP_CONFIG),
        );
        Ok(Self {
            coauthor,
            non_mega_coauthor,
            affiliation,
        })
    }

    /// A query's lookups in every indexed component, one entry per summary.
    pub(crate) fn component_overlaps(
        &self,
        query: &RetrievalQueryData,
        prepared: &PreparedQuery,
        component_count: usize,
    ) -> Vec<ComponentOverlap> {
        let mut overlaps = vec![ComponentOverlap::default(); component_count];
        self.coauthor.accumulate(
            &query.coauthor_terms,
            &prepared.coauthor.weights,
            &mut overlaps,
            |overlap, weight, share| {
                overlap.coauthor_hits += 1;
                if weight > 0.0 {
                    overlap.coauthor += weight * share;
                }
            },
        );
        self.non_mega_coauthor.accumulate(
            &query.coauthor_terms,
            &prepared.non_mega_coauthor.weights,
            &mut overlaps,
            |overlap, weight, share| {
                overlap.non_mega_coauthor_hits += 1;
                if weight > 0.0 {
                    overlap.non_mega_coauthor += weight * share;
                }
            },
        );
        self.affiliation.accumulate(
            &query.affiliation_terms,
            &prepared.affiliation.weights,
            &mut overlaps,
            |overlap, weight, share| {
                if weight > 0.0 {
                    overlap.affiliation += weight * share;
                }
            },
        );
        overlaps
    }
}
