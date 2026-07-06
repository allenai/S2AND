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

pub(crate) fn extract_reference_details_counters(
    py: Python<'_>,
    ref_details_obj: &Bound<'_, PyAny>,
) -> PyResult<(
    Option<CounterData>,
    Option<CounterData>,
    Option<CounterData>,
    Option<CounterData>,
)> {
    let tuple = ref_details_obj.extract::<(PyObject, PyObject, PyObject, PyObject)>()?;
    Ok((
        extract_counter(&tuple.0.bind(py))?,
        extract_counter(&tuple.1.bind(py))?,
        extract_counter(&tuple.2.bind(py))?,
        extract_counter(&tuple.3.bind(py))?,
    ))
}

pub(crate) fn extract_optional_string_set(
    obj: &Bound<'_, PyAny>,
) -> PyResult<Option<HashSet<String>>> {
    if obj.is_none() {
        return Ok(None);
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        let v: String = item?.extract()?;
        out.insert(v);
    }
    if out.is_empty() {
        Ok(None)
    } else {
        Ok(Some(out))
    }
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
    // Directed insert to match the Python reference, which stores name tuples as a
    // `set[tuple[str, str]]` keyed on the curated (a, b) order (see
    // `_load_name_tuples_from_file` in s2and/data.py). The shipped
    // `s2and_name_tuples_filtered.txt` lists both directions explicitly for every
    // pair, so directed insertion still yields both lookups while staying faithful
    // to any asymmetric tuples a caller might supply.
    map.entry(a).or_insert_with(HashSet::new).insert(b);
}

pub(crate) fn extract_name_tuples_map(
    obj: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, HashSet<String>>> {
    if obj.is_none() {
        return Ok(HashMap::new());
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

pub(crate) fn extract_id_string(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    let type_name = obj.get_type().name()?;
    if type_name == "bool" {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "expected id value to be str, int, or uint-compatible int; got bool",
        ));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(i.to_string());
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(u.to_string());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "expected id value to be str, int, or uint-compatible int; got {}",
        type_name
    )))
}

pub(crate) fn extract_set_id_string(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<PaperId>> {
    if obj.is_none() {
        return Ok(HashSet::new());
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        let v = extract_id_string(&item?)?;
        out.insert(v);
    }
    Ok(out)
}

pub(crate) fn extract_string_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in PyIterator::from_object(obj)? {
        out.push(item?.extract()?);
    }
    Ok(out)
}

pub(crate) fn get_namedtuple_item_or_attr<'py>(
    obj: &Bound<'py, PyAny>,
    allow_tuple_fastpath: bool,
    tuple_index: usize,
    attr_name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if allow_tuple_fastpath {
        return obj.get_item(tuple_index).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "NamedTuple fast-path index out of range or object not indexable: index={} attr={}",
                tuple_index, attr_name
            ))
        });
    }
    obj.getattr(attr_name)
}

pub(crate) fn validate_namedtuple_fastpath_contract(
    sample_obj: &Bound<'_, PyAny>,
    required_fields: &[(usize, &str)],
    tuple_label: &str,
) -> PyResult<bool> {
    let fields_obj = match sample_obj.getattr("_fields") {
        Ok(fields_obj) => fields_obj,
        Err(_) => return Ok(false),
    };

    let field_names: Vec<String> = fields_obj.extract().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{} fast-path expected _fields to be a tuple/list of field names",
            tuple_label
        ))
    })?;

    let max_required_index = required_fields
        .iter()
        .map(|(index, _)| *index)
        .max()
        .unwrap_or(0);
    if sample_obj.get_item(max_required_index).is_err() || field_names.len() <= max_required_index {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} fast-path contract mismatch: object is not indexable to required max index {} or _fields len={}",
            tuple_label,
            max_required_index,
            field_names.len()
        )));
    }

    for (field_index, expected_name) in required_fields {
        let actual_name = field_names
            .get(*field_index)
            .map(|s| s.as_str())
            .unwrap_or("<missing>");
        if actual_name != *expected_name {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} fast-path contract mismatch at index {}: expected '{}' got '{}'",
                tuple_label, field_index, expected_name, actual_name
            )));
        }
    }

    Ok(true)
}

pub(crate) fn validate_dict_namedtuple_fastpath_contract(
    rows: &Bound<'_, PyDict>,
    required_fields: &[(usize, &str)],
    tuple_label: &str,
) -> PyResult<bool> {
    if let Some((_, sample_obj)) = rows.iter().next() {
        return validate_namedtuple_fastpath_contract(&sample_obj, required_fields, tuple_label);
    }
    Ok(false)
}

pub(crate) fn extract_paper_authors_with_positions(
    obj: &Bound<'_, PyAny>,
) -> PyResult<Vec<(i64, String)>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in PyIterator::from_object(obj)? {
        let author_obj = item?;
        let (position, author_name) = if let Ok(author_tuple) = author_obj.downcast::<PyTuple>() {
            if author_tuple.len() >= 2 {
                let author_name: String = author_tuple.get_item(0)?.extract()?;
                let position: i64 = author_tuple.get_item(1)?.extract()?;
                (position, author_name)
            } else {
                let position: i64 = author_obj.getattr("position")?.extract()?;
                let author_name: String = author_obj.getattr("author_name")?.extract()?;
                (position, author_name)
            }
        } else {
            let position: i64 = author_obj.getattr("position")?.extract()?;
            let author_name: String = author_obj.getattr("author_name")?.extract()?;
            (position, author_name)
        };
        out.push((position, author_name));
    }
    Ok(out)
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
            let title =
                normalize_text_compat_from_map(&paper_input.raw_title, false, unidecode_char_map);
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
            let title_words =
                counter_data_from_usize_map(word_ngrams_counter_python_compat(&title, stop_words));
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
                    is_reliable: paper_input.is_reliable,
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
    name_counts_semantics: NameCountsLastFirstInitialSemantics,
) -> Vec<(String, SignatureData)> {
    signature_inputs
        .par_iter()
        .map(|entry| {
            let (first_without_apostrophe, middle_without_apostrophe) =
                split_first_middle_hyphen_aware_compat(
                    &entry.raw_first,
                    &entry.raw_middle,
                    name_prefixes,
                    unidecode_char_map,
                );
            let last_normalized =
                normalize_text_compat_from_map(&entry.raw_last, false, unidecode_char_map);
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
                &entry.raw_first,
                &first_without_apostrophe,
                &entry.raw_last,
                &last_normalized,
                name_counts_semantics,
            );
            (
                entry.sig_id.clone(),
                SignatureData {
                    first: Some(first_without_apostrophe.clone()),
                    middle: Some(middle_without_apostrophe),
                    last_normalized: Some(last_normalized),
                    orcid: normalized_orcid,
                    email: entry.email.clone(),
                    affiliations,
                    coauthor_blocks,
                    coauthor_ngrams,
                    coauthors,
                    position: entry.position,
                    paper_id: entry.paper_id.clone(),
                    name_counts,
                    adv_name: Some(first_without_apostrophe),
                },
            )
        })
        .collect::<Vec<_>>()
}

pub(crate) fn extract_string_opt(obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if obj.is_none() {
        Ok(None)
    } else {
        Ok(Some(obj.extract()?))
    }
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
    py: Python<'_>,
    name_tuples: Option<&Bound<'_, PyAny>>,
) -> PyResult<HashMap<String, HashSet<String>>> {
    let Some(obj) = name_tuples else {
        return load_name_tuples_from_text_path(py, None);
    };
    if obj.is_none() {
        return Ok(HashMap::new());
    }
    if let Ok(value) = obj.extract::<String>() {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized.is_empty() || normalized == "none" {
            return Ok(HashMap::new());
        }
        if normalized == "filtered" {
            return load_name_tuples_from_text_path(py, None);
        }
        return load_name_tuples_from_text_path(py, Some(value.as_str()));
    }
    extract_name_tuples_map(obj)
}

pub(crate) fn extract_u32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(arr) = obj.downcast::<PyArray1<u32>>() {
        let readonly = arr.readonly();
        return Ok(readonly.as_slice()?.to_vec());
    }
    if let Ok(arr) = obj.downcast::<PyArray1<u64>>() {
        let readonly = arr.readonly();
        return readonly
            .as_slice()?
            .iter()
            .map(|value| {
                u32::try_from(*value).map_err(|_| {
                    pyo3::exceptions::PyOverflowError::new_err(format!(
                        "component member signature index exceeds u32: {value}"
                    ))
                })
            })
            .collect();
    }
    let values: Vec<u64> = obj.extract()?;
    values
        .into_iter()
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(format!(
                    "component member signature index exceeds u32: {value}"
                ))
            })
        })
        .collect()
}

pub(crate) fn extract_component_member_indices(
    obj: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, Vec<u32>>> {
    let mut out = HashMap::new();
    let items = obj.call_method0("items")?;
    for item in PyIterator::from_object(&items)? {
        let tuple = item?.downcast_into::<PyTuple>()?;
        if tuple.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "component_member_indices_by_key.items() yielded a non-pair",
            ));
        }
        let component_key: String = tuple.get_item(0)?.extract()?;
        let members = extract_u32_vec(&tuple.get_item(1)?)?;
        out.insert(component_key, members);
    }
    Ok(out)
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

pub(crate) fn default_name_tuples_path(py: Python<'_>) -> PyResult<String> {
    let consts = py.import("s2and.consts")?;
    let package_data_dir: String = consts.getattr("_PACKAGE_DATA_DIR")?.extract()?;
    let pathlib = py.import("pathlib")?;
    let path_obj = pathlib
        .getattr("Path")?
        .call1((package_data_dir,))?
        .call_method1("joinpath", ("s2and_name_tuples_filtered.txt",))?;
    path_obj.call_method0("as_posix")?.extract()
}

pub(crate) fn load_name_tuples_from_text_path(
    py: Python<'_>,
    path: Option<&str>,
) -> PyResult<HashMap<String, HashSet<String>>> {
    let effective_path = match path {
        Some(value) => value.to_string(),
        None => default_name_tuples_path(py)?,
    };
    if !Path::new(&effective_path).exists() {
        return Ok(HashMap::new());
    }
    let text = fs::read_to_string(&effective_path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to read name tuples path {}: {}",
            effective_path, err
        ))
    })?;
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((a, b)) = trimmed.split_once(',') {
            insert_name_tuple_alias(&mut out, a.to_string(), b.to_string());
        }
    }
    Ok(out)
}

pub(crate) fn has_name_counts_artifact(raw_name_counts: &RawNameCountMaps) -> bool {
    raw_name_counts.has_data()
}

pub(crate) fn canonical_last_for_counts(raw_last: &str, normalized_last: &str) -> String {
    if contains_name_dash(raw_last) || normalized_last.contains(' ') {
        normalized_last.replace(' ', "")
    } else {
        normalized_last.to_string()
    }
}

pub(crate) fn build_name_counts_data_from_artifact(
    raw_name_counts: &RawNameCountMaps,
    raw_first: &str,
    first_without_apostrophe: &str,
    raw_last: &str,
    last_normalized: &str,
    semantics: NameCountsLastFirstInitialSemantics,
) -> Option<NameCountsData> {
    if !has_name_counts_artifact(raw_name_counts) {
        return None;
    }

    let mut first_for_counts = first_without_apostrophe
        .split(' ')
        .next()
        .unwrap_or("")
        .to_string();
    if contains_name_dash(raw_first) {
        let joined = first_without_apostrophe.replace(' ', "");
        if !joined.is_empty() {
            first_for_counts = joined;
        }
    }

    let last_for_counts = canonical_last_for_counts(raw_last, last_normalized);
    let last_first_initial_key = match semantics {
        NameCountsLastFirstInitialSemantics::LegacyFullFirstToken => {
            format!("{} {}", last_for_counts, first_for_counts)
                .trim()
                .to_string()
        }
        NameCountsLastFirstInitialSemantics::InitialChar => {
            let first_initial = first_for_counts
                .chars()
                .next()
                .map(|ch| ch.to_string())
                .unwrap_or_default();
            format!("{} {}", last_for_counts, first_initial)
                .trim()
                .to_string()
        }
    };

    // A count key is looked up only when its components are present. An empty
    // surname must yield NaN (not the sentinel default 1.0), mirroring the
    // Python ANDData._compute_signature_name_counts path (D6 in
    // docs/normalization_migration_blocked.md; work_plan section 2). Without
    // the last_present guard a genuinely missing last name would be
    // indistinguishable from a corpus count of 1, diverging from Python.
    let first_informative = py_len(&first_for_counts) > 1;
    let last_present = py_len(&last_for_counts) > 0;

    let first = if first_informative {
        match raw_name_counts.get(RawNameCountKind::First, &first_for_counts) {
            Some(value) => value,
            None => 1.0,
        }
    } else {
        f64::NAN
    };
    let first_last = if first_informative && last_present {
        let first_last_key = format!("{} {}", first_for_counts, last_for_counts);
        match raw_name_counts.get(RawNameCountKind::FirstLast, first_last_key.trim()) {
            Some(value) => value,
            None => 1.0,
        }
    } else {
        f64::NAN
    };
    let last = if last_present {
        match raw_name_counts.get(RawNameCountKind::Last, &last_for_counts) {
            Some(value) => value,
            None => 1.0,
        }
    } else {
        f64::NAN
    };
    let last_first_initial = if last_present {
        match raw_name_counts.get(RawNameCountKind::LastFirstInitial, &last_first_initial_key) {
            Some(value) => value,
            None => 1.0,
        }
    } else {
        f64::NAN
    };

    Some(NameCountsData {
        first,
        first_last,
        last,
        last_first_initial,
    })
}

#[cfg(test)]
mod name_counts_empty_surname_tests {
    use crate::name_counts::{
        NameCountsLastFirstInitialSemantics, RawNameCountIndex, RawNameCountMaps,
    };
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
        std::fs::create_dir_all(&dir).expect("create temp index dir");
        for name in ["first", "last", "first_last", "last_first_initial"] {
            let mut file =
                std::fs::File::create(dir.join(format!("{name}.bin"))).expect("create index file");
            file.write_all(&empty_index_bytes())
                .expect("write index header");
        }
        let manifest = concat!(
            r#"{"schema_version":"name_counts_index_v1","files":{"#,
            r#""first":{"path":"first.bin"},"last":{"path":"last.bin"},"#,
            r#""first_last":{"path":"first_last.bin"},"#,
            r#""last_first_initial":{"path":"last_first_initial.bin"}}}"#,
        );
        std::fs::write(dir.join("manifest.json"), manifest).expect("write manifest");
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
        let semantics = NameCountsLastFirstInitialSemantics::InitialChar;

        let empty_last =
            super::build_name_counts_data_from_artifact(&maps, "Alice", "alice", "", "", semantics)
                .expect("artifact present -> Some");
        let present_last = super::build_name_counts_data_from_artifact(
            &maps, "Alice", "alice", "Smith", "smith", semantics,
        )
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
    }
}
