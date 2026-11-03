//! Two-pass SPECTER bounds for planning retrieval.
//!
//! A query's top k needs exact scores only for components whose score bound
//! reaches the k-th exact score. The centroid pass bounds every SPECTER score
//! with one dot product: a component's vectors lie within some angle of its
//! centroid, so no cosine exceeds the centroid cosine by more than that angle.
//! Exact scores of the best estimated components set each query's cutoff, and
//! the exemplar pass computes exemplar dot products only for components whose
//! loose bound still reaches it. Bounding every exemplar of every component was
//! most of incremental planning's CPU time; on s_lee the loose bound skips about
//! two thirds of them. Any sound bound selects the same top k, so the plan is
//! unchanged.

use super::*;

/// Slack on every ball bound, far above the rounding of its few f64 steps.
const BALL_SLACK: f64 = 1e-6;

/// Summaries per pool task in the centroid and exemplar passes.
const SUMMARIES_PER_TASK: usize = 64;

/// The cosine and sine of an angle at or above the largest angle between a
/// summary's centroid and any of its exemplars.
#[derive(Clone, Copy)]
pub(crate) struct SpecterBall {
    cos_theta: f64,
    sin_theta: f64,
}

/// The summary's ball, or None when a ball bound would not be sound: no
/// centroid, more vectors than the fused kernel holds, vectors of different
/// lengths, stored norms that differ from their vectors, or non-finite values.
/// Such summaries take the full bound in the exemplar pass.
pub(crate) fn specter_ball(summary: &RetrievalSummaryData) -> Option<SpecterBall> {
    let centroid = summary.specter_centroid.as_ref()?;
    let centroid_norm = summary.specter_centroid_norm?;
    if summary.exemplar_vectors.len() + 1 > 5
        || centroid_norm.is_nan()
        || centroid_norm <= 0.0
        || !norm_agrees(centroid, centroid_norm)
    {
        return None;
    }
    let mut cos_theta = 1.0f64;
    for (vector, norm) in summary
        .exemplar_vectors
        .iter()
        .zip(summary.exemplar_norms.iter())
    {
        if vector.len() != centroid.len() {
            return None;
        }
        if *norm == 0.0 {
            // Its cosine is 0.0, below every bound.
            continue;
        }
        if !norm_agrees(vector, *norm) {
            return None;
        }
        let dot: f64 = vector
            .iter()
            .zip(centroid.iter())
            .map(|(left, right)| f64::from(*left) * f64::from(*right))
            .sum();
        let cosine = dot / (norm * centroid_norm);
        if !cosine.is_finite() {
            return None;
        }
        cos_theta = cos_theta.min(cosine);
    }
    // The margin keeps the angle at or above the exact one despite rounding.
    let theta = (cos_theta - 1e-9).clamp(-1.0, 1.0).acos();
    Some(SpecterBall {
        cos_theta: theta.cos(),
        sin_theta: theta.sin(),
    })
}

/// Whether `norm` is the vector's norm, so cosines built from it are true cosines.
fn norm_agrees(vector: &[f32], norm: f64) -> bool {
    (vector_norm_f32(vector) - norm).abs() <= 1e-9 * norm
}

/// An upper bound of a SPECTER score, the largest of 0.0 and the summary's
/// cosines, from `centroid_bound`, an upper bound of the centroid cosine.
///
/// A query at angle a from the centroid is at least a - theta from every
/// vector within theta of the centroid, and cos(a - theta) expands to the
/// products below.
fn ball_bound(centroid_bound: f32, ball: SpecterBall) -> f64 {
    let x = f64::from(centroid_bound).clamp(-1.0, 1.0);
    let cosine = if x >= ball.cos_theta {
        1.0
    } else {
        x * ball.cos_theta + (1.0 - x * x).sqrt() * ball.sin_theta
    };
    cosine.clamp(0.0, 1.0) + BALL_SLACK
}

/// One query's state between the passes: its exact scores so far, the cutoff
/// they set, and the candidates whose loose bound reaches the cutoff, each
/// with the exact non-SPECTER part of its score bound.
pub(crate) struct PendingPlanQuery {
    top_k: usize,
    scored: Vec<(usize, f32)>,
    cutoff: f64,
    centroid_weight: f64,
    survivors: Vec<(u32, f64)>,
}

/// A query after the first pass.
pub(crate) enum PlanFirstPass {
    /// The query's top k, sorted.
    Done(Vec<(usize, f32)>),
    /// Candidates whose bounds the exemplar pass tightens.
    Pending(PendingPlanQuery),
}

impl RustHybridCentroidRetriever {
    /// Every summary's ball; empty for retrievers that do not rank.
    pub(crate) fn specter_balls_of(summaries: &[RetrievalSummaryData]) -> Vec<Option<SpecterBall>> {
        summaries.par_iter().map(specter_ball).collect()
    }

    /// Upper bounds of each query's centroid cosine against every summary, by
    /// query: one f32 lane dot product plus its error bound. The bound is NaN
    /// where the ball bound does not apply: the query has no vector, its
    /// stored norm differs from its vector, its length differs from the
    /// centroid's, or the summary has no ball.
    pub(crate) fn specter_centroid_bounds(&self, queries: &[RetrievalQueryData]) -> Vec<Vec<f32>> {
        let query_count = queries.len();
        if query_count == 0 {
            return Vec::new();
        }
        // Queries the ball bound applies to, by vector length.
        let ball_queries: Vec<Option<(&[f32], f64)>> = queries
            .iter()
            .map(|query| match (query.specter.as_ref(), query.specter_norm) {
                (Some(values), Some(norm)) if norm_agrees(values, norm) => {
                    Some((&values[..], norm))
                }
                _ => None,
            })
            .collect();
        let mut columns = vec![f32::NAN; self.summaries.len() * query_count];
        columns
            .par_chunks_mut(query_count * SUMMARIES_PER_TASK)
            .enumerate()
            .for_each(|(chunk_index, chunk_columns)| {
                let first_summary = chunk_index * SUMMARIES_PER_TASK;
                // Centroids with a ball, as (summary offset in chunk, vector, norm).
                let centroids: Vec<(usize, &[f32], f64)> = (0..chunk_columns.len() / query_count)
                    .filter_map(|offset| {
                        let index = first_summary + offset;
                        self.specter_balls.get(index).copied().flatten()?;
                        let summary = &self.summaries[index];
                        Some((
                            offset,
                            &summary.specter_centroid.as_ref()?[..],
                            summary.specter_centroid_norm?,
                        ))
                    })
                    .collect();
                for group in centroids.chunks(5) {
                    let length = group[0].1.len();
                    if group.iter().any(|(_, vector, _)| vector.len() != length) {
                        // Mixed lengths: each centroid alone.
                        for one in group {
                            centroid_column_bounds(
                                std::slice::from_ref(one),
                                &ball_queries,
                                query_count,
                                chunk_columns,
                            );
                        }
                    } else {
                        centroid_column_bounds(group, &ball_queries, query_count, chunk_columns);
                    }
                }
            });
        // Transpose one chunk of summaries at a time, so each row takes a
        // contiguous run. Element by element, the stores to every query's row
        // missed the TLB and cost more than the dot products.
        let mut rows = vec![vec![0.0f32; self.summaries.len()]; query_count];
        for (chunk_index, chunk) in columns.chunks(query_count * SUMMARIES_PER_TASK).enumerate() {
            let first_summary = chunk_index * SUMMARIES_PER_TASK;
            let summaries_in_chunk = chunk.len() / query_count;
            for (query_index, row) in rows.iter_mut().enumerate() {
                let run = &mut row[first_summary..first_summary + summaries_in_chunk];
                for (offset, bound) in run.iter_mut().enumerate() {
                    *bound = chunk[offset * query_count + query_index];
                }
            }
        }
        rows
    }

    /// The first pass of bounded top-k scoring for one query.
    ///
    /// Each candidate's loose bound adds its exact overlap and name terms to
    /// the weighted ball bound; its estimate uses the centroid cosine bound
    /// instead. The candidates with the best estimates are scored exactly,
    /// and their k-th f32 score `t` sets the cutoff, as in
    /// `score_top_k_bounded`: every candidate in the final top k has a bound
    /// at or above `t` less two f32 ulps. Candidates whose loose bound reaches
    /// the cutoff and whose SPECTER score has no tighter bound yet stay pending.
    pub(crate) fn plan_first_pass(
        &self,
        query_data: &RetrievalQueryData,
        candidate_indices: &[usize],
        top_k: usize,
        centroid_bounds: &[f32],
        name_terms: &[f64],
    ) -> PlanFirstPass {
        if top_k == 0 || candidate_indices.is_empty() {
            return PlanFirstPass::Done(Vec::new());
        }
        let prepared = self.prepare_query(query_data);
        let overlaps =
            self.overlap_index()
                .component_overlaps(query_data, &prepared, self.summaries.len());
        let exact = |idx: usize| {
            (
                idx,
                self.score_candidate(query_data, &prepared, &overlaps, idx, None, None, None),
            )
        };
        if candidate_indices.len() <= top_k {
            let mut scored: Vec<(usize, f32)> =
                candidate_indices.iter().map(|idx| exact(*idx)).collect();
            self.keep_sorted_top_k_scored_candidates(&mut scored, top_k, None, None);
            return PlanFirstPass::Done(scored);
        }
        let weights = if query_data.has_full_first {
            DEFAULT_HYBRID_CENTROID_WEIGHTS
        } else {
            DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS
        };
        // Without a query vector every SPECTER score is exactly 0.0.
        let has_specter = query_data.specter.is_some() && query_data.specter_norm.is_some();
        let zero_overlaps_score_zero = prepared.zero_numerators_score_zero();
        // (estimate, loose bound, non-SPECTER part, candidate)
        let mut entries: Vec<(f64, f64, f64, usize)> = candidate_indices
            .iter()
            .map(|&idx| {
                let overlap = &overlaps[idx];
                let (coauthor, affiliation) =
                    if zero_overlaps_score_zero && overlap.has_zero_numerators() {
                        (0.0, 0.0)
                    } else {
                        overlap_scores(query_data, &prepared, &self.summaries[idx], overlap)
                    };
                let rest = weights.coauthor * coauthor
                    + weights.affiliation * affiliation
                    + name_terms[idx];
                let x = centroid_bounds[idx];
                let (specter_loose, specter_estimate) = if !has_specter {
                    (0.0, 0.0)
                } else if x.is_finite() {
                    let ball = self.specter_balls[idx].expect("finite centroid bounds have a ball");
                    (ball_bound(x, ball), f64::from(x).max(0.0))
                } else {
                    (f64::INFINITY, 1.0)
                };
                let loose = with_order_slack(weights.centroid * specter_loose + rest);
                (weights.centroid * specter_estimate + rest, loose, rest, idx)
            })
            .collect();
        let first_pass = (4 * top_k).max(top_k + 16).min(entries.len());
        entries.select_nth_unstable_by(first_pass - 1, |left, right| right.0.total_cmp(&left.0));
        let mut scored: Vec<(usize, f32)> = entries[..first_pass]
            .iter()
            .map(|(_, _, _, idx)| exact(*idx))
            .collect();
        let mut kth = scored.clone();
        self.keep_sorted_top_k_scored_candidates(&mut kth, top_k, None, None);
        let Some(&(_, threshold)) = kth.last().filter(|_| kth.len() == top_k) else {
            scored.extend(
                entries[first_pass..]
                    .iter()
                    .map(|(_, _, _, idx)| exact(*idx)),
            );
            self.keep_sorted_top_k_scored_candidates(&mut scored, top_k, None, None);
            return PlanFirstPass::Done(scored);
        };
        let spacing = f64::from(threshold.abs().next_up() - threshold.abs());
        let cutoff = f64::from(threshold) - 2.0 * spacing;
        let mut survivors = Vec::new();
        for &(_, loose, rest, idx) in &entries[first_pass..] {
            if !(loose >= cutoff || loose.is_nan()) {
                continue;
            }
            // A zero SPECTER bound is exact, and summaries with more vectors
            // than the fused kernel holds have no finite bound: score now.
            let summary = &self.summaries[idx];
            let unbounded = summary.exemplar_vectors.len()
                + usize::from(summary.specter_centroid.is_some())
                > 5;
            if !has_specter || unbounded {
                scored.push(exact(idx));
            } else {
                survivors.push((idx as u32, rest));
            }
        }
        if survivors.is_empty() {
            self.keep_sorted_top_k_scored_candidates(&mut scored, top_k, None, None);
            return PlanFirstPass::Done(scored);
        }
        PlanFirstPass::Pending(PendingPlanQuery {
            top_k,
            scored,
            cutoff,
            centroid_weight: weights.centroid,
            survivors,
        })
    }

    /// Tight SPECTER bounds of each pending query's survivors, in survivor
    /// order: the bound `specter_bound_column` gives, with the centroid term
    /// from `centroid_bounds` where it is finite. Summaries go to the pool, so
    /// each one's exemplars are read once for all queries that need them.
    pub(crate) fn specter_survivor_bounds(
        &self,
        queries: &[RetrievalQueryData],
        pending: &[Option<&PendingPlanQuery>],
        centroid_bounds: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let summary_count = self.summaries.len();
        // Survivor entries (query, position) grouped by summary.
        let mut starts = vec![0usize; summary_count + 1];
        for query in pending.iter().flatten() {
            for &(index, _) in &query.survivors {
                starts[index as usize + 1] += 1;
            }
        }
        for index in 0..summary_count {
            starts[index + 1] += starts[index];
        }
        let mut cursor = starts.clone();
        let mut entries = vec![(0u32, 0u32); starts[summary_count]];
        for (query_index, query) in pending.iter().enumerate() {
            let Some(query) = query else { continue };
            for (position, &(index, _)) in query.survivors.iter().enumerate() {
                let slot = &mut cursor[index as usize];
                entries[*slot] = (query_index as u32, position as u32);
                *slot += 1;
            }
        }
        let bounds: Vec<Vec<(u32, u32, f32)>> = (0..summary_count)
            .into_par_iter()
            .with_min_len(SUMMARIES_PER_TASK)
            .fold(Vec::new, |mut out, index| {
                let group = &entries[starts[index]..starts[index + 1]];
                if !group.is_empty() {
                    survivor_bounds(
                        &self.summaries[index],
                        index,
                        group,
                        queries,
                        centroid_bounds,
                        &mut out,
                    );
                }
                out
            })
            .collect();
        let mut rows: Vec<Vec<f32>> = pending
            .iter()
            .map(|query| vec![0.0f32; query.map_or(0, |query| query.survivors.len())])
            .collect();
        for (query_index, position, bound) in bounds.into_iter().flatten() {
            rows[query_index as usize][position as usize] = bound;
        }
        rows
    }

    /// The query's top k: `pending`'s exact scores plus exact scores of the
    /// survivors whose tight bound (`tight_bounds`, in survivor order) reaches
    /// the cutoff.
    pub(crate) fn plan_finish(
        &self,
        query_data: &RetrievalQueryData,
        pending: PendingPlanQuery,
        tight_bounds: &[f32],
    ) -> Vec<(usize, f32)> {
        let PendingPlanQuery {
            top_k,
            mut scored,
            cutoff,
            centroid_weight,
            survivors,
        } = pending;
        let extra: Vec<usize> = survivors
            .iter()
            .zip(tight_bounds)
            .filter_map(|(&(index, rest), &tight)| {
                let bound = with_order_slack(centroid_weight * f64::from(tight) + rest);
                (bound >= cutoff || bound.is_nan()).then_some(index as usize)
            })
            .collect();
        if !extra.is_empty() {
            let prepared = self.prepare_query(query_data);
            let overlaps = self.overlap_index().component_overlaps(
                query_data,
                &prepared,
                self.summaries.len(),
            );
            scored.extend(extra.into_iter().map(|idx| {
                (
                    idx,
                    self.score_candidate(query_data, &prepared, &overlaps, idx, None, None, None),
                )
            }));
        }
        self.keep_sorted_top_k_scored_candidates(&mut scored, top_k, None, None);
        scored
    }
}

/// Slack for summing a bound in a different order than the exact score.
fn with_order_slack(bound: f64) -> f64 {
    bound + 1e-9 * (1.0 + bound.abs())
}

/// Centroid cosine bounds of the ball queries against up to five equal-length
/// centroids, written into `columns` (one column of `query_count` per summary
/// of the chunk). Queries of another length keep NaN.
fn centroid_column_bounds(
    group: &[(usize, &[f32], f64)],
    ball_queries: &[Option<(&[f32], f64)>],
    query_count: usize,
    columns: &mut [f32],
) {
    let length = group[0].1.len();
    let vectors: Vec<&[f32]> = group.iter().map(|(_, vector, _)| *vector).collect();
    let error = f32_dot_relative_error_bound(length);
    let mut write = |query_index: usize, query_norm: f64, dots: &[f32; 5]| {
        for (slot, (offset, _, norm)) in group.iter().enumerate() {
            let cosine = if query_norm == 0.0 {
                0.0
            } else {
                f64::from(dots[slot]) / (query_norm * norm) + error
            };
            columns[offset * query_count + query_index] = f32_at_or_above(cosine);
        }
    };
    let mut pending: Option<(usize, &[f32], f64)> = None;
    for (query_index, query) in ball_queries.iter().enumerate() {
        let Some((values, norm)) = *query else {
            continue;
        };
        if values.len() != length {
            continue;
        }
        match pending.take() {
            Some((first_index, first_values, first_norm)) => {
                let (first_dots, second_dots) = f32_lane_dots_pair(first_values, values, &vectors);
                write(first_index, first_norm, &first_dots);
                write(query_index, norm, &second_dots);
            }
            None => pending = Some((query_index, values, norm)),
        }
    }
    if let Some((query_index, values, norm)) = pending {
        write(query_index, norm, &f32_lane_dots(values, &vectors));
    }
}

/// Tight bounds of one summary for its survivor entries, appended to `out` as
/// (query, position, bound).
fn survivor_bounds(
    summary: &RetrievalSummaryData,
    summary_index: usize,
    group: &[(u32, u32)],
    queries: &[RetrievalQueryData],
    centroid_bounds: &[Vec<f32>],
    out: &mut Vec<(u32, u32, f32)>,
) {
    let exemplars: Vec<&[f32]> = summary.exemplar_vectors.iter().map(|v| &v[..]).collect();
    let exemplar_length = exemplars.first().map(|vector| vector.len());
    // The centroid bound, then exemplar cosine bounds, as `specter_bound_column` takes them.
    let finish = |query: &RetrievalQueryData, centroid: f32, dots: &[f32; 5]| -> f32 {
        let query_norm = query.specter_norm.expect("ball queries have a norm");
        let len = query.specter.as_ref().map_or(0, |values| values.len());
        let error = f32_dot_relative_error_bound(len);
        let mut bound = f64::from(centroid).max(0.0);
        for (slot, norm) in summary.exemplar_norms.iter().enumerate() {
            let cosine = if query_norm == 0.0 || *norm == 0.0 {
                0.0
            } else {
                f64::from(dots[slot]) / (query_norm * norm) + error
            };
            bound = bound.max(cosine);
        }
        f32_at_or_above(bound)
    };
    let mut pending: Option<(u32, u32, &[f32], f32)> = None;
    for &(query_index, position) in group {
        let query = &queries[query_index as usize];
        let centroid = centroid_bounds[query_index as usize][summary_index];
        let values: &[f32] = query.specter.as_ref().map_or(&[], |values| &values[..]);
        if !centroid.is_finite() {
            // No ball bound for this pair: the full single-query bound.
            let mut column = [0.0f32];
            specter_bound_column(summary, std::slice::from_ref(query), &mut column);
            out.push((query_index, position, column[0]));
            continue;
        }
        // A finite centroid bound means the ball applies: every exemplar has
        // the query's length.
        debug_assert!(exemplar_length.is_none_or(|length| length == values.len()));
        if exemplars.is_empty() {
            out.push((query_index, position, finish(query, centroid, &[0.0; 5])));
            continue;
        }
        match pending.take() {
            Some((first_index, first_position, first_values, first_centroid)) => {
                let (first_dots, second_dots) =
                    f32_lane_dots_pair(first_values, values, &exemplars);
                let first_query = &queries[first_index as usize];
                out.push((
                    first_index,
                    first_position,
                    finish(first_query, first_centroid, &first_dots),
                ));
                out.push((query_index, position, finish(query, centroid, &second_dots)));
            }
            None => pending = Some((query_index, position, values, centroid)),
        }
    }
    if let Some((query_index, position, values, centroid)) = pending {
        let dots = f32_lane_dots(values, &exemplars);
        out.push((
            query_index,
            position,
            finish(&queries[query_index as usize], centroid, &dots),
        ));
    }
}

#[cfg(test)]
pub(super) fn ball_bound_for_tests(centroid_bound: f32, ball: SpecterBall) -> f64 {
    ball_bound(centroid_bound, ball)
}

/// A pending query whose survivors are `summaries`, for bound tests.
#[cfg(test)]
pub(super) fn pending_for_tests(summaries: Vec<u32>) -> PendingPlanQuery {
    PendingPlanQuery {
        top_k: 1,
        scored: Vec::new(),
        cutoff: 0.0,
        centroid_weight: 1.0,
        survivors: summaries
            .into_iter()
            .map(|summary| (summary, 0.0))
            .collect(),
    }
}

#[cfg(test)]
pub(super) fn survivor_indices_for_tests(pending: &PendingPlanQuery) -> Vec<usize> {
    pending
        .survivors
        .iter()
        .map(|(summary, _)| *summary as usize)
        .collect()
}
