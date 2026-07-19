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
                    is_reliable: paper_input.is_reliable,
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
    py: Python<'_>,
    name_tuples: Option<&Bound<'_, PyAny>>,
) -> PyResult<HashMap<String, HashSet<String>>> {
    let Some(obj) = name_tuples else {
        return load_name_tuples_from_text_path(py, None);
    };
    if obj.is_none() {
        return load_name_tuples_from_text_path(py, None);
    }
    if let Ok(value) = obj.extract::<String>() {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized == "filtered" {
            return load_name_tuples_from_text_path(py, None);
        }
        if normalized.is_empty() || normalized == "none" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "disable name tuples with an explicit empty set; string 'none' and empty paths are invalid",
            ));
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
        .call_method1("joinpath", ("s2and_name_tuples_canonical.txt",))?;
    path_obj.call_method0("as_posix")?.extract()
}

const NAME_TUPLE_ARTIFACT_SCHEMA_VERSION: &str = "s2and_name_tuples_v2";
const NAME_TUPLE_ARTIFACT_VERSION: u64 = 2;
const NAME_TUPLE_NORMALIZATION_VERSION: &str = "canonical_v2";

#[derive(Debug, Clone, PartialEq, Eq)]
struct NameTupleArtifactIdentity {
    schema_version: String,
    artifact_version: u64,
    normalization_version: String,
    data_filename: String,
    data_sha256: String,
    data_size_bytes: u64,
    pair_count: u64,
    source_filename: String,
    source_sha256: String,
    source_size_bytes: u64,
}

fn name_tuple_value_error(metadata_path: &Path, message: impl AsRef<str>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "invalid name-tuple metadata {}: {}",
        metadata_path.display(),
        message.as_ref()
    ))
}

fn required_name_tuple_object<'a>(
    value: &'a serde_json::Value,
    field: &str,
    metadata_path: &Path,
) -> PyResult<&'a serde_json::Map<String, serde_json::Value>> {
    value
        .get(field)
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            name_tuple_value_error(metadata_path, format!("requires object field {field:?}"))
        })
}

fn required_name_tuple_string(
    value: Option<&serde_json::Value>,
    field: &str,
    metadata_path: &Path,
) -> PyResult<String> {
    let string = value
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            name_tuple_value_error(
                metadata_path,
                format!("requires nonempty string field {field:?}"),
            )
        })?;
    Ok(string.to_string())
}

fn required_name_tuple_u64(
    value: Option<&serde_json::Value>,
    field: &str,
    metadata_path: &Path,
) -> PyResult<u64> {
    value.and_then(serde_json::Value::as_u64).ok_or_else(|| {
        name_tuple_value_error(
            metadata_path,
            format!("requires nonnegative integer field {field:?}"),
        )
    })
}

fn required_name_tuple_sha256(
    value: Option<&serde_json::Value>,
    field: &str,
    metadata_path: &Path,
) -> PyResult<String> {
    let digest = required_name_tuple_string(value, field, metadata_path)?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(name_tuple_value_error(
            metadata_path,
            format!("requires lowercase SHA-256 field {field:?}"),
        ));
    }
    Ok(digest)
}

fn python_sha256_hex(py: Python<'_>, payload: &[u8]) -> PyResult<String> {
    use pyo3::types::PyBytes;

    py.import("hashlib")?
        .call_method1("sha256", (PyBytes::new(py, payload),))?
        .call_method0("hexdigest")?
        .extract()
}

fn validated_name_tuple_artifact(
    py: Python<'_>,
    effective_path: &Path,
) -> PyResult<(HashMap<String, HashSet<String>>, NameTupleArtifactIdentity)> {
    if !effective_path.is_file() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "name tuples path does not exist: {}",
            effective_path.display()
        )));
    }
    let metadata_path = effective_path.with_file_name(format!(
        "{}.meta.json",
        effective_path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "name tuples path has a non-UTF-8 filename: {}",
                    effective_path.display()
                ))
            })?
    ));
    if !metadata_path.is_file() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "name tuple metadata does not exist: {}",
            metadata_path.display()
        )));
    }

    let metadata_bytes = fs::read(&metadata_path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to read name tuple metadata {}: {}",
            metadata_path.display(),
            err
        ))
    })?;
    let data_bytes = fs::read(effective_path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to read name tuples path {}: {}",
            effective_path.display(),
            err
        ))
    })?;
    let metadata_after = fs::read(&metadata_path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to re-read name tuple metadata {}: {}",
            metadata_path.display(),
            err
        ))
    })?;
    if metadata_bytes != metadata_after {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "name tuple metadata changed while loading: {}",
            metadata_path.display()
        )));
    }

    let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes)
        .map_err(|err| name_tuple_value_error(&metadata_path, format!("invalid JSON: {err}")))?;
    let root = metadata
        .as_object()
        .ok_or_else(|| name_tuple_value_error(&metadata_path, "expected a JSON object"))?;
    let schema_version =
        required_name_tuple_string(root.get("schema_version"), "schema_version", &metadata_path)?;
    if schema_version != NAME_TUPLE_ARTIFACT_SCHEMA_VERSION {
        return Err(name_tuple_value_error(
            &metadata_path,
            format!(
                "unsupported schema_version={schema_version:?}; expected {NAME_TUPLE_ARTIFACT_SCHEMA_VERSION:?}"
            ),
        ));
    }
    let artifact_version = required_name_tuple_u64(
        root.get("artifact_version"),
        "artifact_version",
        &metadata_path,
    )?;
    if artifact_version != NAME_TUPLE_ARTIFACT_VERSION {
        return Err(name_tuple_value_error(
            &metadata_path,
            format!(
                "unsupported artifact_version={artifact_version}; expected {NAME_TUPLE_ARTIFACT_VERSION}"
            ),
        ));
    }
    let normalization_version = required_name_tuple_string(
        root.get("normalization_version"),
        "normalization_version",
        &metadata_path,
    )?;
    if normalization_version != NAME_TUPLE_NORMALIZATION_VERSION {
        return Err(name_tuple_value_error(
            &metadata_path,
            format!(
                "normalization_version={normalization_version:?}; expected {NAME_TUPLE_NORMALIZATION_VERSION:?}"
            ),
        ));
    }
    required_name_tuple_string(root.get("generated_at"), "generated_at", &metadata_path)?;

    let source = required_name_tuple_object(&metadata, "source", &metadata_path)?;
    let source_filename =
        required_name_tuple_string(source.get("filename"), "source.filename", &metadata_path)?;
    let source_sha256 =
        required_name_tuple_sha256(source.get("sha256"), "source.sha256", &metadata_path)?;
    let source_size_bytes = required_name_tuple_u64(
        source.get("size_bytes"),
        "source.size_bytes",
        &metadata_path,
    )?;

    let data = required_name_tuple_object(&metadata, "data", &metadata_path)?;
    let data_filename =
        required_name_tuple_string(data.get("filename"), "data.filename", &metadata_path)?;
    let actual_filename = effective_path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "name tuples path has a non-UTF-8 filename: {}",
                effective_path.display()
            ))
        })?;
    if data_filename != actual_filename {
        return Err(name_tuple_value_error(
            &metadata_path,
            format!("binds data.filename={data_filename:?}, expected {actual_filename:?}"),
        ));
    }
    let data_sha256 =
        required_name_tuple_sha256(data.get("sha256"), "data.sha256", &metadata_path)?;
    let data_size_bytes =
        required_name_tuple_u64(data.get("size_bytes"), "data.size_bytes", &metadata_path)?;
    if data_size_bytes != data_bytes.len() as u64 {
        return Err(name_tuple_value_error(
            &metadata_path,
            format!(
                "data size mismatch: metadata={data_size_bytes} actual={}",
                data_bytes.len()
            ),
        ));
    }
    let actual_sha256 = python_sha256_hex(py, &data_bytes)?;
    if data_sha256 != actual_sha256 {
        return Err(name_tuple_value_error(
            &metadata_path,
            format!("data SHA-256 mismatch: metadata={data_sha256} actual={actual_sha256}"),
        ));
    }
    let pair_count =
        required_name_tuple_u64(data.get("pair_count"), "data.pair_count", &metadata_path)?;

    let expected_semantics = serde_json::json!({
        "encoding": "utf-8",
        "line_format": "name_a,name_b",
        "row_order": "lexicographic_by_fields_unique",
        "pair_order": "name_a_lexicographically_less_than_name_b",
        "directionality": "canonical_unordered_rows",
        "runtime_pair_semantics": "unordered",
        "canonicalizer": "canonicalize_name_text",
        "drop_identity": true,
        "drop_prefix_compatible": true,
    });
    if root.get("semantics") != Some(&expected_semantics) {
        return Err(name_tuple_value_error(
            &metadata_path,
            format!("unsupported semantics; expected {expected_semantics}"),
        ));
    }
    let generation_counts =
        required_name_tuple_object(&metadata, "generation_counts", &metadata_path)?;
    for field in [
        "input_pair_count",
        "dropped_identity",
        "dropped_prefix_compatible",
        "dropped_empty",
    ] {
        required_name_tuple_u64(
            generation_counts.get(field),
            &format!("generation_counts.{field}"),
            &metadata_path,
        )?;
    }

    let text = std::str::from_utf8(&data_bytes).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "name tuple artifact is not valid UTF-8: {}",
            effective_path.display()
        ))
    })?;
    let mut aliases: HashMap<String, HashSet<String>> = HashMap::new();
    let mut previous: Option<(&str, &str)> = None;
    let mut actual_pair_count = 0usize;
    for (line_index, line) in text.lines().enumerate() {
        let mut fields = line.split(',');
        let first_a = fields.next().unwrap_or_default();
        let first_b = fields.next().unwrap_or_default();
        if first_a.is_empty() || first_b.is_empty() || fields.next().is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid name tuple at {}:{}: expected two nonempty fields",
                effective_path.display(),
                line_index + 1
            )));
        }
        let pair = (first_a, first_b);
        if previous.is_some_and(|prior| pair <= prior) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid name tuple ordering at {}:{}: rows must be unique and sorted by fields",
                effective_path.display(),
                line_index + 1
            )));
        }
        previous = Some(pair);
        if crate::text_compat::canonicalize_name_text_compat(first_a, None) != first_a
            || crate::text_compat::canonicalize_name_text_compat(first_b, None) != first_b
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid noncanonical name tuple at {}:{}",
                effective_path.display(),
                line_index + 1
            )));
        }
        if first_a == first_b {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid identity name tuple at {}:{}",
                effective_path.display(),
                line_index + 1
            )));
        }
        if first_a > first_b {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid name tuple field order at {}:{}: name_a must be lexicographically less than name_b",
                effective_path.display(),
                line_index + 1
            )));
        }
        if same_prefix_tokens(first_a, first_b) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid prefix-compatible name tuple at {}:{}",
                effective_path.display(),
                line_index + 1
            )));
        }
        insert_name_tuple_alias(&mut aliases, first_a.to_string(), first_b.to_string());
        actual_pair_count += 1;
    }
    if actual_pair_count as u64 != pair_count {
        return Err(name_tuple_value_error(
            &metadata_path,
            format!(
                "pair_count mismatch: metadata={pair_count} actual={}",
                actual_pair_count
            ),
        ));
    }
    let identity = NameTupleArtifactIdentity {
        schema_version,
        artifact_version,
        normalization_version,
        data_filename,
        data_sha256,
        data_size_bytes,
        pair_count,
        source_filename,
        source_sha256,
        source_size_bytes,
    };
    Ok((aliases, identity))
}

pub(crate) fn load_name_tuples_from_text_path(
    py: Python<'_>,
    path: Option<&str>,
) -> PyResult<HashMap<String, HashSet<String>>> {
    let effective_path = match path {
        Some(value) => value.to_string(),
        None => default_name_tuples_path(py)?,
    };
    Ok(validated_name_tuple_artifact(py, Path::new(&effective_path))?.0)
}

#[pyfunction(signature = (path=None))]
pub(crate) fn read_name_tuple_artifact_identity(
    py: Python<'_>,
    path: Option<&str>,
) -> PyResult<Py<PyDict>> {
    let effective_path = match path {
        Some(value) => value.to_string(),
        None => default_name_tuples_path(py)?,
    };
    let identity = validated_name_tuple_artifact(py, Path::new(&effective_path))?.1;
    let output = PyDict::new(py);
    output.set_item("schema_version", identity.schema_version)?;
    output.set_item("artifact_version", identity.artifact_version)?;
    output.set_item("normalization_version", identity.normalization_version)?;
    output.set_item("data_filename", identity.data_filename)?;
    output.set_item("data_sha256", identity.data_sha256)?;
    output.set_item("data_size_bytes", identity.data_size_bytes)?;
    output.set_item("pair_count", identity.pair_count)?;
    output.set_item("source_filename", identity.source_filename)?;
    output.set_item("source_sha256", identity.source_sha256)?;
    output.set_item("source_size_bytes", identity.source_size_bytes)?;
    Ok(output.unbind())
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
mod name_counts_empty_surname_tests {
    use crate::name_counts::{RawNameCountIndex, RawNameCountMaps};
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
            r#"{"schema_version":"name_counts_index_v1","normalization_version":"canonical_v2","files":{"#,
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

#[cfg(test)]
mod name_tuple_artifact_tests {
    use super::{load_name_tuples_from_text_path, python_sha256_hex};
    use pyo3::types::PyAnyMethods;
    use pyo3::Python;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn metadata_path(path: &Path) -> PathBuf {
        path.with_file_name(format!(
            "{}.meta.json",
            path.file_name().unwrap().to_str().unwrap()
        ))
    }

    fn write_artifact(py: Python<'_>, path: &Path, data: &str, pair_count: u64) {
        let digest = python_sha256_hex(py, data.as_bytes()).expect("hash fixture");
        fs::write(path, data).expect("write tuple fixture");
        let metadata = serde_json::json!({
            "schema_version": "s2and_name_tuples_v2",
            "artifact_version": 2,
            "normalization_version": "canonical_v2",
            "generated_at": "2026-07-10T00:00:00+00:00",
            "source": {
                "filename": "source.txt",
                "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                "size_bytes": 0,
            },
            "data": {
                "filename": path.file_name().unwrap().to_str().unwrap(),
                "sha256": digest,
                "size_bytes": data.len(),
                "pair_count": pair_count,
            },
            "semantics": {
                "encoding": "utf-8",
                "line_format": "name_a,name_b",
                "row_order": "lexicographic_by_fields_unique",
                "pair_order": "name_a_lexicographically_less_than_name_b",
                "directionality": "canonical_unordered_rows",
                "runtime_pair_semantics": "unordered",
                "canonicalizer": "canonicalize_name_text",
                "drop_identity": true,
                "drop_prefix_compatible": true,
            },
            "generation_counts": {
                "input_pair_count": 1,
                "dropped_identity": 0,
                "dropped_prefix_compatible": 0,
                "dropped_empty": 0,
            },
        });
        fs::write(
            metadata_path(path),
            serde_json::to_vec(&metadata).expect("serialize fixture metadata"),
        )
        .expect("write fixture metadata");
    }

    fn remove_artifact(path: &Path) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(metadata_path(path));
    }

    #[test]
    fn strict_name_tuple_artifact_accepts_valid_and_rejects_missing_invalid_and_tampered() {
        #[cfg(windows)]
        if let Some(python_home) = option_env!("S2AND_RUST_PYTHONHOME") {
            std::env::set_var("PYTHONHOME", python_home);
        }
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let missing = std::env::temp_dir().join(format!(
                "s2and-missing-name-tuples-{}-{}.txt",
                std::process::id(),
                std::thread::current().name().unwrap_or("test")
            ));
            let missing_error = load_name_tuples_from_text_path(py, missing.to_str())
                .expect_err("missing name tuples must not become an empty alias map");
            assert!(missing_error
                .value(py)
                .str()
                .unwrap()
                .to_string()
                .contains("does not exist"));

            let valid = std::env::temp_dir().join(format!(
                "s2and-valid-name-tuples-{}.txt",
                std::process::id()
            ));
            write_artifact(py, &valid, "alice,ally\n", 1);
            let aliases = load_name_tuples_from_text_path(py, valid.to_str())
                .expect("valid strict artifact loads");
            assert!(aliases.get("alice").unwrap().contains("ally"));
            assert!(aliases.get("ally").unwrap().contains("alice"));
            fs::write(&valid, "alica,ally\n").expect("tamper tuple fixture");
            let tamper_error = load_name_tuples_from_text_path(py, valid.to_str())
                .expect_err("tampered tuple bytes must fail");
            assert!(tamper_error
                .value(py)
                .str()
                .unwrap()
                .to_string()
                .contains("SHA-256 mismatch"));
            remove_artifact(&valid);

            let invalid = std::env::temp_dir().join(format!(
                "s2and-invalid-name-tuples-{}.txt",
                std::process::id()
            ));
            write_artifact(py, &invalid, "alice,bob,carol\n", 1);
            let invalid_error = load_name_tuples_from_text_path(py, invalid.to_str())
                .expect_err("invalid tuple rows must fail");
            remove_artifact(&invalid);
            assert!(invalid_error
                .value(py)
                .str()
                .unwrap()
                .to_string()
                .contains("expected two nonempty fields"));

            let reversed = std::env::temp_dir().join(format!(
                "s2and-reversed-name-tuples-{}.txt",
                std::process::id()
            ));
            write_artifact(py, &reversed, "ally,alice\n", 1);
            let reversed_error = load_name_tuples_from_text_path(py, reversed.to_str())
                .expect_err("reversed tuple fields must fail");
            remove_artifact(&reversed);
            assert!(reversed_error
                .value(py)
                .str()
                .unwrap()
                .to_string()
                .contains("name_a must be lexicographically less than name_b"));
        });
    }
}
