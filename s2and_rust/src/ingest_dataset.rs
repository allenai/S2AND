use super::*;

pub(crate) fn extract_counter(obj: &Bound<'_, PyAny>) -> PyResult<Option<CounterData>> {
    if obj.is_none() {
        return Ok(None);
    }
    let dict = obj.downcast::<PyDict>()?;
    if dict.len() == 0 {
        return Ok(None);
    }
    let mut entries: Vec<(u64, f32)> = Vec::with_capacity(dict.len());
    let mut sum = 0.0f32;
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let val: f64 = v.extract()?;
        let val32 = val as f32;
        sum += val32;
        entries.push((fnv64(key.as_bytes()), val32));
    }
    entries.sort_unstable_by_key(|e| e.0);
    Ok(Some(CounterData { entries, sum }))
}

pub(crate) fn canonical_signature_pair_ref<'a>(a: &'a str, b: &'a str) -> (&'a str, &'a str) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

pub(crate) fn canonical_signature_pair_owned(a: String, b: String) -> (String, String) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

pub(crate) fn extract_pair_set(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<(String, String)>> {
    if obj.is_none() {
        return Ok(HashSet::new());
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        let tuple = item?;
        let (a, b): (String, String) = tuple.extract()?;
        out.insert(canonical_signature_pair_owned(a, b));
    }
    Ok(out)
}

pub(crate) fn insert_name_tuple_alias(
    map: &mut HashMap<String, HashSet<String>>,
    a: String,
    b: String,
) {
    map.entry(a.clone())
        .or_insert_with(HashSet::new)
        .insert(b.clone());
    map.entry(b).or_insert_with(HashSet::new).insert(a);
}

pub(crate) fn extract_name_tuples_map(
    obj: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, HashSet<String>>> {
    if obj.extract::<String>().is_ok() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "name_tuples must be an explicit collection of pairs, not a path or string sentinel",
        ));
    }
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for item in PyIterator::from_object(obj)? {
        let tuple = item?;
        let (a, b): (String, String) = tuple.extract()?;
        insert_name_tuple_alias(&mut out, a, b);
    }
    Ok(out)
}

pub(crate) fn extract_cluster_seeds_require(
    obj: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, ClusterId>> {
    if obj.is_none() {
        return Ok(HashMap::new());
    }
    let dict = obj.downcast::<PyDict>()?;
    let mut out = HashMap::with_capacity(dict.len());
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let val: ClusterId = if let Ok(i) = v.extract::<i64>() {
            ClusterId::Int(i)
        } else if let Ok(s) = v.extract::<String>() {
            ClusterId::Str(s)
        } else if let Ok(u) = v.extract::<u64>() {
            ClusterId::Int(u as i64)
        } else {
            ClusterId::Str(v.str()?.to_string())
        };
        out.insert(key, val);
    }
    Ok(out)
}

pub(crate) fn cluster_id_to_string(cluster_id: &ClusterId) -> String {
    match cluster_id {
        ClusterId::Int(value) => value.to_string(),
        ClusterId::Str(value) => value.clone(),
    }
}

pub(crate) fn extract_required_string_set(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<String>> {
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        out.insert(item?.extract()?);
    }
    Ok(out)
}

pub(crate) fn extract_affiliation_stopwords(py: Python<'_>) -> PyResult<HashSet<String>> {
    let text_module = py.import("s2and.text")?;
    let stopwords_obj = text_module.getattr("AFFILIATIONS_STOP_WORDS")?;
    extract_required_string_set(&stopwords_obj)
}

pub(crate) fn prefilter_affiliation_text(
    affiliations: &[String],
    stopwords: &HashSet<String>,
) -> String {
    if affiliations.is_empty() {
        return String::new();
    }
    let mut tokens: Vec<&str> = Vec::new();
    for word in affiliations
        .iter()
        .flat_map(|affiliation| affiliation.split_whitespace())
    {
        if !stopwords.contains(word) && py_len(word) > 1 {
            tokens.push(word);
        }
    }
    tokens.join(" ")
}

pub(crate) fn counter_data_from_usize_map(
    counter_map: HashMap<String, usize>,
) -> Option<CounterData> {
    if counter_map.is_empty() {
        return None;
    }
    let mut entries: Vec<(u64, f32)> = counter_map
        .iter()
        .map(|(k, v)| (fnv64(k.as_bytes()), *v as f32))
        .collect();
    entries.sort_unstable_by_key(|e| e.0);
    let sum: f32 = entries.iter().map(|e| e.1).sum();
    Some(CounterData { entries, sum })
}

pub(crate) fn counter_data_from_hash_count_map(
    counter_map: HashMap<u64, usize>,
) -> Option<CounterData> {
    if counter_map.is_empty() {
        return None;
    }
    let mut entries: Vec<(u64, f32)> = counter_map
        .into_iter()
        .map(|(hash, count)| (hash, count as f32))
        .collect();
    entries.sort_unstable_by_key(|entry| entry.0);
    let sum: f32 = entries.iter().map(|entry| entry.1).sum();
    Some(CounterData { entries, sum })
}

pub(crate) fn increment_df_from_counter(
    counter: &Option<CounterData>,
    df_map: &mut HashMap<u64, usize>,
) {
    if let Some(counter_data) = counter.as_ref() {
        for (hash, _count) in counter_data.entries.iter() {
            *df_map.entry(*hash).or_insert(0) += 1;
        }
    }
}

pub(crate) fn hash_string_values(values: &HashSet<String>) -> Vec<u64> {
    let mut hashes: Vec<u64> = values.iter().map(|value| fnv64(value.as_bytes())).collect();
    hashes.sort_unstable();
    hashes.dedup();
    hashes
}

pub(crate) fn query_terms_from_values(values: &HashSet<String>) -> Vec<RetrievalQueryTerm> {
    let mut terms: Vec<RetrievalQueryTerm> = values
        .iter()
        .map(|value| RetrievalQueryTerm {
            hash: fnv64(value.as_bytes()),
            token_count: term_token_count(value),
        })
        .collect();
    terms.sort_unstable_by_key(|term| term.hash);
    terms.dedup_by_key(|term| term.hash);
    terms
}

pub(crate) fn term_set_from_normalized_text(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .filter(|token| !token.is_empty())
        .map(|token| token.to_string())
        .collect()
}

struct SignatureTextFields<'a> {
    author_first: &'a str,
    author_middle: &'a str,
    author_last: &'a str,
    author_suffix: &'a str,
    affiliations: &'a [String],
}

struct PaperTextFields<'a> {
    title: &'a str,
    venue: &'a str,
    journal_name: &'a str,
}

fn ensure_unidecode_for_signature_texts<'a>(
    signatures: impl IntoIterator<Item = SignatureTextFields<'a>>,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    for signature in signatures {
        ensure_unidecode_for_text(signature.author_first, unidecode_char_map)?;
        ensure_unidecode_for_text(signature.author_middle, unidecode_char_map)?;
        ensure_unidecode_for_text(signature.author_last, unidecode_char_map)?;
        ensure_unidecode_for_text(signature.author_suffix, unidecode_char_map)?;
        for affiliation in signature.affiliations.iter() {
            ensure_unidecode_for_text(affiliation, unidecode_char_map)?;
        }
    }
    Ok(())
}

fn ensure_unidecode_for_paper_texts<'a>(
    papers: impl IntoIterator<Item = PaperTextFields<'a>>,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    for paper in papers {
        ensure_unidecode_for_text(paper.title, unidecode_char_map)?;
        ensure_unidecode_for_text(paper.venue, unidecode_char_map)?;
        ensure_unidecode_for_text(paper.journal_name, unidecode_char_map)?;
    }
    Ok(())
}

pub(crate) fn ensure_unidecode_for_paper_author_texts<'a>(
    paper_authors: impl IntoIterator<Item = &'a [(i64, String)]>,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    for authors in paper_authors {
        for (_position, author_name) in authors.iter() {
            ensure_unidecode_for_text(author_name, unidecode_char_map)?;
        }
    }
    Ok(())
}

pub(crate) fn ensure_unidecode_for_raw_arrow_inputs(
    signatures: &HashMap<String, RawArrowSignature>,
    papers: &HashMap<String, RawArrowPaper>,
    paper_authors: &HashMap<String, Vec<(i64, String)>>,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    ensure_unidecode_for_signature_texts(
        signatures.values().map(|signature| SignatureTextFields {
            author_first: &signature.author_first,
            author_middle: &signature.author_middle,
            author_last: &signature.author_last,
            author_suffix: &signature.author_suffix,
            affiliations: &signature.affiliations,
        }),
        unidecode_char_map,
    )?;
    ensure_unidecode_for_paper_texts(
        papers.values().map(|paper| PaperTextFields {
            title: &paper.title,
            venue: &paper.venue,
            journal_name: &paper.journal_name,
        }),
        unidecode_char_map,
    )?;
    ensure_unidecode_for_paper_author_texts(
        paper_authors.values().map(Vec::as_slice),
        unidecode_char_map,
    )?;
    Ok(())
}

pub(crate) fn preprocess_stage_papers(
    paper_inputs: &[StagePaperInput],
    preprocess: bool,
    unidecode_char_map: &HashMap<char, String>,
    stop_words: &HashSet<String>,
    venue_stop_words: &HashSet<String>,
) -> Vec<(PaperId, StagePaperPreprocessed)> {
    paper_inputs
        .par_iter()
        .map(|paper_input| {
            let title = normalize_title_compat_from_map(&paper_input.raw_title, unidecode_char_map);
            let venue = if preprocess {
                normalize_text_compat_from_map(&paper_input.raw_venue, false, unidecode_char_map)
            } else {
                paper_input.raw_venue.clone()
            };
            let journal_name = if preprocess {
                normalize_text_compat_from_map(&paper_input.raw_journal, false, unidecode_char_map)
            } else {
                paper_input.raw_journal.clone()
            };
            let authors = paper_input
                .raw_authors
                .iter()
                .map(|(position, raw_name)| {
                    (
                        *position,
                        normalize_text_compat_from_map(raw_name, false, unidecode_char_map),
                    )
                })
                .collect::<Vec<_>>();
            let title_words = counter_data_from_usize_map(word_ngrams_counter_python_compat(
                &title, stop_words, false,
            ));
            let title_chars = if preprocess {
                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                    &title,
                    false,
                    true,
                    Some(stop_words),
                    true,
                ))
            } else {
                None
            };
            let venue_ngrams = if preprocess {
                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                    &venue,
                    false,
                    true,
                    Some(venue_stop_words),
                    true,
                ))
            } else {
                None
            };
            let journal_ngrams = if preprocess {
                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                    &journal_name,
                    false,
                    true,
                    Some(venue_stop_words),
                    true,
                ))
            } else {
                None
            };
            (
                paper_input.paper_id.clone(),
                StagePaperPreprocessed {
                    authors,
                    year: paper_input.year,
                    has_abstract: paper_input.has_abstract,
                    predicted_language: paper_input.predicted_language.clone(),
                    language_reliability: paper_input.language_reliability,
                    title_words,
                    title_chars,
                    venue_ngrams,
                    journal_ngrams,
                },
            )
        })
        .collect::<Vec<_>>()
}

pub(crate) fn preprocess_stage_signatures(
    signature_inputs: &[StageSignatureInput],
    preprocessed_papers: &HashMap<PaperId, StagePaperPreprocessed>,
    raw_name_counts: &RawNameCountMaps,
    name_prefixes: &HashSet<String>,
    affiliation_stopwords: &HashSet<String>,
    unidecode_char_map: &HashMap<char, String>,
    preprocess: bool,
) -> Vec<(String, SignatureData)> {
    signature_inputs
        .par_iter()
        .map(|entry| {
            // Signature author-name fields are canonical_v2; paper titles,
            // venues, affiliations, and authors-as-text keep the legacy
            // normalize_text_compat pipeline.
            let (canonical_first, canonical_middle, canonical_last) =
                canonicalize_name_parts_compat(
                    &entry.raw_first,
                    &entry.raw_middle,
                    &entry.raw_last,
                    name_prefixes,
                    Some(unidecode_char_map),
                );
            let mut coauthor_list: Vec<String> = Vec::new();
            if let Some(preprocessed_paper) = preprocessed_papers.get(&entry.paper_id) {
                for (author_position, author_name) in preprocessed_paper.authors.iter() {
                    if *author_position != entry.position {
                        coauthor_list.push(author_name.clone());
                    }
                }
            }
            let coauthors = if coauthor_list.is_empty() {
                None
            } else {
                Some(coauthor_list.iter().cloned().collect::<HashSet<String>>())
            };
            let mut coauthor_blocks_set: HashSet<String> = HashSet::new();
            for coauthor in coauthor_list.iter() {
                coauthor_blocks_set.insert(compute_block_compat(coauthor));
            }
            let coauthor_blocks = if coauthor_blocks_set.is_empty() {
                None
            } else {
                Some(coauthor_blocks_set)
            };

            let normalized_affiliations: Vec<String> = if preprocess {
                entry
                    .affiliation_values
                    .iter()
                    .filter_map(|affiliation| {
                        let normalized =
                            normalize_text_compat_from_map(affiliation, false, unidecode_char_map);
                        if normalized.is_empty() {
                            None
                        } else {
                            Some(normalized)
                        }
                    })
                    .collect()
            } else {
                entry.affiliation_values.clone()
            };
            let affiliation_text = if preprocess {
                prefilter_affiliation_text(&normalized_affiliations, affiliation_stopwords)
            } else {
                String::new()
            };
            let coauthor_text = if preprocess {
                coauthor_list.join(" ")
            } else {
                String::new()
            };
            let affiliations = if preprocess && !affiliation_text.is_empty() {
                counter_data_from_usize_map(word_ngrams_counter(&affiliation_text))
            } else {
                None
            };
            let coauthor_ngrams = if preprocess && !coauthor_text.is_empty() {
                counter_data_from_usize_map(char_ngrams_counter(&coauthor_text))
            } else {
                None
            };
            let normalized_orcid = entry
                .orcid
                .as_ref()
                .and_then(|value| normalize_orcid_compact_owned(value));
            let name_counts = build_name_counts_data_from_artifact(
                raw_name_counts,
                &canonical_first,
                &canonical_last,
            );
            (
                entry.sig_id.clone(),
                SignatureData {
                    first: Some(canonical_first.clone()),
                    middle: Some(canonical_middle),
                    last_normalized: Some(canonical_last),
                    orcid: normalized_orcid,
                    email: entry.email.clone(),
                    affiliations,
                    coauthor_blocks,
                    coauthor_ngrams,
                    coauthors,
                    position: entry.position,
                    paper_id: entry.paper_id.clone(),
                    name_counts,
                    adv_name: Some(canonical_first),
                },
            )
        })
        .collect::<Vec<_>>()
}

pub(crate) fn extract_name_counts_data(obj: &Bound<'_, PyAny>) -> PyResult<Option<NameCountsData>> {
    if obj.is_none() {
        return Ok(None);
    }
    let first: Option<f64> = obj.getattr("first")?.extract()?;
    let first_last: Option<f64> = obj.getattr("first_last")?.extract()?;
    let last: Option<f64> = obj.getattr("last")?.extract()?;
    let last_first_initial: Option<f64> = obj.getattr("last_first_initial")?.extract()?;
    Ok(Some(NameCountsData {
        first: first.unwrap_or(f64::NAN),
        first_last: first_last.unwrap_or(f64::NAN),
        last: last.unwrap_or(f64::NAN),
        last_first_initial: last_first_initial.unwrap_or(f64::NAN),
    }))
}

pub(crate) fn extract_specter_vec(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<f32>>> {
    if obj.is_none() {
        return Ok(None);
    }
    // All-zero vectors are kept as present (real vectors), matching the Arrow
    // ingest path and the Current Decisions table in docs/work_plan.md. The
    // missing-vector treatment for all-zero rows lives at feature time in the
    // featurizer, so both ingest modes share the same semantics here.
    if let Ok(arr) = obj.downcast::<PyArray1<f32>>() {
        let readonly = arr.readonly();
        return Ok(Some(readonly.as_slice()?.to_vec()));
    }
    if let Ok(arr) = obj.downcast::<PyArray1<f64>>() {
        let readonly = arr.readonly();
        return Ok(Some(
            readonly.as_slice()?.iter().map(|v| *v as f32).collect(),
        ));
    }
    // Fallback: try to extract as Vec<f64>
    let vec_f64: Vec<f64> = obj.extract()?;
    Ok(Some(vec_f64.into_iter().map(|v| v as f32).collect()))
}

pub(crate) fn extract_name_tuples_argument(
    name_tuples: Option<&Bound<'_, PyAny>>,
) -> PyResult<HashMap<String, HashSet<String>>> {
    let obj = name_tuples.filter(|value| !value.is_none()).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "RustFeaturizer.from_arrow_paths requires explicit name-tuple pairs; load artifacts in Python",
        )
    })?;
    extract_name_tuples_map(obj)
}

pub(crate) fn extract_specter_vec_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut vectors = Vec::new();
    for item in PyIterator::from_object(obj)? {
        if let Some(vector) = extract_specter_vec(&item?)? {
            vectors.push(vector);
        }
    }
    Ok(vectors)
}

pub(crate) fn extract_integer_count(obj: &Bound<'_, PyAny>, field_name: &str) -> PyResult<u64> {
    if let Ok(value) = obj.extract::<u64>() {
        return Ok(value);
    }
    let value: f64 = obj.extract().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{field_name} values must be integer counts"
        ))
    })?;
    const MAX_EXACT_F64_INTEGER: f64 = 9_007_199_254_740_992.0;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > MAX_EXACT_F64_INTEGER {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{field_name} values must be non-negative integer counts representable without f64 precision loss"
        )));
    }
    Ok(value as u64)
}

pub(crate) fn extract_string_count_pairs(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(String, u64)>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let dict = obj.downcast::<PyDict>()?;
    if dict.len() == 0 {
        return Ok(Vec::new());
    }
    let mut entries = Vec::with_capacity(dict.len());
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let val = extract_integer_count(&v, "first_name_counts")?;
        entries.push((key, val));
    }
    Ok(entries)
}

pub(crate) fn term_token_count(value: &str) -> u8 {
    value
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .count()
        .min(u8::MAX as usize) as u8
}

pub(crate) fn has_name_counts_artifact(raw_name_counts: &RawNameCountMaps) -> bool {
    raw_name_counts.has_data()
}

pub(crate) fn build_name_counts_data_from_artifact(
    raw_name_counts: &RawNameCountMaps,
    canonical_first: &str,
    canonical_last: &str,
) -> Option<NameCountsData> {
    if !has_name_counts_artifact(raw_name_counts) {
        return None;
    }

    // canonical_v2 key construction (D5/D6/D8): keys are the full canonical
    // fields, spaced — no first-token reduction and no compact-joins. A key is
    // looked up only when its components pass the gate; a gated-out key must
    // yield NaN (not the sentinel default 1.0), mirroring the Python
    // `canonical_name_count_keys` path (docs/normalization_migration_blocked.md;
    // work_plan section 2). Without the gates a genuinely missing component
    // would be indistinguishable from a corpus count of 1, diverging from
    // Python. A present key that misses the artifact defaults to 1.0.
    let keys = canonical_name_count_keys_compat(canonical_first, canonical_last);
    let lookup = |kind: RawNameCountKind, key: &Option<String>| match key {
        Some(key) => raw_name_counts.get(kind, key).unwrap_or(1.0),
        None => f64::NAN,
    };

    Some(NameCountsData {
        first: lookup(RawNameCountKind::First, &keys.first),
        first_last: lookup(RawNameCountKind::FirstLast, &keys.first_last),
        last: lookup(RawNameCountKind::Last, &keys.last),
        last_first_initial: lookup(RawNameCountKind::LastFirstInitial, &keys.last_first_initial),
    })
}

#[cfg(test)]
mod blank_paper_author_tests {
    use super::*;
    use crate::name_counts::RawNameCountMaps;

    #[test]
    fn classic_preprocessing_retains_legacy_blank_coauthor_sets() {
        let signature_inputs = vec![StageSignatureInput {
            sig_id: "s1".to_string(),
            paper_id: "p1".to_string(),
            raw_first: "Alice".to_string(),
            raw_middle: String::new(),
            raw_last: "Smith".to_string(),
            email: None,
            position: 0,
            affiliation_values: Vec::new(),
            orcid: None,
        }];
        let papers = HashMap::from([(
            "p1".to_string(),
            StagePaperPreprocessed {
                authors: vec![(0, "alice smith".to_string()), (1, String::new())],
                year: None,
                has_abstract: false,
                predicted_language: None,
                language_reliability: 0.0,
                title_words: None,
                title_chars: None,
                venue_ngrams: None,
                journal_ngrams: None,
            },
        )]);

        let preprocessed = preprocess_stage_signatures(
            &signature_inputs,
            &papers,
            &RawNameCountMaps::default(),
            &HashSet::new(),
            &HashSet::new(),
            &HashMap::new(),
            true,
        );
        let signature = &preprocessed[0].1;

        assert_eq!(signature.coauthors, Some(HashSet::from([String::new()])));
        assert_eq!(
            signature.coauthor_blocks,
            Some(HashSet::from([String::new()]))
        );
        assert!(signature.coauthor_ngrams.is_none());
    }
}

#[cfg(test)]
mod name_counts_empty_surname_tests {
    use crate::name_counts::{sha256_file, RawNameCountIndex, RawNameCountMaps};
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    // A valid 32-byte name-count index header describing zero records. It opens
    // cleanly (so `has_data()` is true) and every lookup misses, which is all
    // the empty-surname gate needs — no hashing or name blob required.
    fn empty_index_bytes() -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(b"S2NCI001");
        // record_count = 0 (bytes 8..16 stay zero)
        bytes[16..24].copy_from_slice(&32u64.to_le_bytes()); // blob_offset == header length
                                                             // blob_len = 0 (bytes 24..32 stay zero)
        bytes
    }

    fn write_empty_artifact() -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = format!(
            "s2and_nc_empty_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let dir = std::env::temp_dir().join(unique);
        let generation_dir = dir.join("generations").join("empty-test-generation");
        std::fs::create_dir_all(&generation_dir).expect("create temp generation dir");
        let mut files = serde_json::Map::new();
        for name in ["first", "last", "first_last", "last_first_initial"] {
            let path = generation_dir.join(format!("{name}.bin"));
            let mut file = std::fs::File::create(&path).expect("create index file");
            file.write_all(&empty_index_bytes())
                .expect("write index header");
            drop(file);
            files.insert(
                name.to_string(),
                serde_json::json!({
                    "path": format!("generations/empty-test-generation/{name}.bin"),
                    "byte_count": path.metadata().expect("file metadata").len(),
                    "sha256": sha256_file(&path).expect("hash fixture"),
                }),
            );
        }
        std::fs::write(generation_dir.join(".published"), []).expect("write published marker");
        let manifest = serde_json::json!({
            "schema_version": "name_counts_index_v2",
            "normalization_version": "canonical_v2",
            "source_provenance": {
                "schema_version": "name_counts_provenance_v3",
                "normalization_version": "canonical_v2",
                "generation_id": "empty-test-generation",
                "source_snapshot_id": "empty-test-snapshot",
                "source_kind": "test-fixture",
                "source_query_sha256": "1".repeat(64),
                "selected_rows_sha256": "2".repeat(64),
                "source_row_count": 0,
            },
            "files": files,
        });
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        dir
    }

    #[test]
    fn empty_surname_yields_nan_matching_python() {
        // Regression for the Python<->Rust divergence found in the 2026-07-04
        // pass: build_name_counts_data_from_artifact must return NaN for every
        // last-dependent key when the surname is empty, exactly like the Python
        // ANDData._compute_signature_name_counts path (D6). Before the fix these
        // defaulted to the sentinel 1.0, so an empty surname was scored as a
        // corpus count of 1 in Rust while Python reported NaN.
        let dir = write_empty_artifact();
        let maps = RawNameCountMaps::from_index(
            RawNameCountIndex::open(dir.to_str().expect("utf-8 temp path")).expect("open index"),
        );

        let empty_last = super::build_name_counts_data_from_artifact(&maps, "alice", "")
            .expect("artifact present -> Some");
        let present_last = super::build_name_counts_data_from_artifact(&maps, "alice", "smith")
            .expect("artifact present -> Some");
        // canonical_v2 D6: an empty first suppresses the last_first_initial
        // lookup (legacy still looked up the bare surname key).
        let empty_first = super::build_name_counts_data_from_artifact(&maps, "", "smith")
            .expect("artifact present -> Some");
        // A single-initial first is uninformative for first/first_last but
        // still contributes its initial char to last_first_initial.
        let initial_first = super::build_name_counts_data_from_artifact(&maps, "j", "doe")
            .expect("artifact present -> Some");
        // D5: spaced canonical fields are looked up as-is (no compact-join).
        let spaced = super::build_name_counts_data_from_artifact(&maps, "sang min", "ou yang")
            .expect("artifact present -> Some");

        // Drop the mmap-backed maps before removing the dir (Windows keeps
        // memory-mapped files locked while they are open).
        drop(maps);
        let _ = std::fs::remove_dir_all(&dir);

        // Empty surname: every last-dependent key is NaN. `first` is still an
        // informative token, so it is looked up and misses -> sentinel 1.0.
        assert!(
            empty_last.last.is_nan(),
            "last must be NaN for empty surname"
        );
        assert!(
            empty_last.first_last.is_nan(),
            "first_last must be NaN for empty surname"
        );
        assert!(
            empty_last.last_first_initial.is_nan(),
            "last_first_initial must be NaN for empty surname"
        );
        assert_eq!(
            empty_last.first, 1.0,
            "informative first is still looked up (miss -> 1.0)"
        );

        // Present surname: nothing is gated to NaN; every key misses -> 1.0.
        assert!(!present_last.last.is_nan());
        assert_eq!(present_last.last, 1.0);
        assert_eq!(present_last.first_last, 1.0);
        assert_eq!(present_last.last_first_initial, 1.0);
        assert_eq!(present_last.first, 1.0);

        // Empty first: every first-dependent key is NaN, including
        // last_first_initial (canonical_v2 change from legacy).
        assert!(empty_first.first.is_nan());
        assert!(empty_first.first_last.is_nan());
        assert!(
            empty_first.last_first_initial.is_nan(),
            "last_first_initial must be NaN for empty first (D6)"
        );
        assert_eq!(empty_first.last, 1.0);

        // Single-initial first: uninformative for first/first_last, but the
        // initial-char last_first_initial key is still looked up.
        assert!(initial_first.first.is_nan());
        assert!(initial_first.first_last.is_nan());
        assert_eq!(initial_first.last_first_initial, 1.0);
        assert_eq!(initial_first.last, 1.0);

        // Spaced canonical fields all pass their gates and are looked up
        // verbatim (misses -> 1.0 against the empty artifact).
        assert_eq!(spaced.first, 1.0);
        assert_eq!(spaced.first_last, 1.0);
        assert_eq!(spaced.last, 1.0);
        assert_eq!(spaced.last_first_initial, 1.0);
    }
}
