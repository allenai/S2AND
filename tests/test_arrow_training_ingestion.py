"""Parity tests for Arrow-native training ingestion + featurization.

Round-trip: JSON-built train ANDData -> Arrow bundle (same writers production
conversion uses) -> Arrow-built train ANDData. Parity is asserted at three
levels:

1. Ingestion: every Python signature/paper field train mode consumes matches
   the JSON-built dataset after preprocessing.
2. Featurizer: the from_arrow_paths-built featurizer (fast Arrow door) is used
   for Arrow-backed datasets.
3. End-to-end: featurize() with fixed train/val/test pairs returns identical
   main + nameless matrices and labels through both ingestion paths, using the
   production training feature configuration.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")

from s2and import feature_port  # noqa: E402
from s2and.arrow_training import (  # noqa: E402
    build_training_anddata_from_arrow,
    load_papers_from_arrow,
    load_signatures_from_arrow,
)
from s2and.consts import FEATURIZER_VERSION, NORMALIZATION_VERSION  # noqa: E402
from s2and.data import ANDData, Author  # noqa: E402
from s2and.featurizer import (  # noqa: E402
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_NAMELESS_FEATURE_GROUPS,
    FeaturizationInfo,
    featurize,
)
from s2and.incremental_linking.feature_block import write_arrow_ipc_table  # noqa: E402
from s2and.incremental_linking.feature_block_arrow import (  # noqa: E402
    write_name_counts_index,
    write_raw_arrow_batch_lookup_indexes,
)
from scripts.arrow_conversion_helpers import write_feature_block_arrow_from_anddata  # noqa: E402
from tests.helpers import (  # noqa: E402
    import_s2and_rust,
    tiny_name_counts_index,
    tiny_name_counts_provenance,
    tiny_name_counts_tuple,
    write_test_arrow_artifact_manifest,
)

HAS_RUST, _RUST_MODULE = import_s2and_rust()
if not HAS_RUST:
    raise pytest.skip.Exception(
        "s2and_rust extension is required for arrow training featurization parity",
        allow_module_level=True,
    )

DUMMY_DIR = Path(__file__).resolve().parent / "dummy"


def _write_minimal_signatures_table(path: Path, signature_ids: list[str | None]) -> None:
    row_count = len(signature_ids)
    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(signature_ids, type=pa.string()),
                "paper_id": pa.array([f"p{i}" for i in range(row_count)], type=pa.string()),
                "author_first": pa.array(["Ada"] * row_count, type=pa.string()),
                "author_middle": pa.array([""] * row_count, type=pa.string()),
                "author_last": pa.array(["Lovelace"] * row_count, type=pa.string()),
                "author_suffix": pa.array([""] * row_count, type=pa.string()),
                "author_affiliations": pa.array([[] for _ in range(row_count)], type=pa.list_(pa.string())),
                "author_position": pa.array([0] * row_count, type=pa.int64()),
                "author_block": pa.array(["lovelace"] * row_count, type=pa.string()),
            }
        ),
        path,
    )


def _write_minimal_papers_table(path: Path, paper_ids: list[str | None]) -> None:
    row_count = len(paper_ids)
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(paper_ids, type=pa.string()),
                "title": pa.array(["title"] * row_count, type=pa.string()),
                "venue": pa.array(["venue"] * row_count, type=pa.string()),
                "journal_name": pa.array(["journal"] * row_count, type=pa.string()),
            }
        ),
        path,
    )


def _write_minimal_paper_authors_table(path: Path, paper_ids: list[str | None]) -> None:
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(paper_ids, type=pa.string()),
                "position": pa.array([0] * len(paper_ids), type=pa.int64()),
                "author_name": pa.array(["Ada Lovelace"] * len(paper_ids), type=pa.string()),
            }
        ),
        path,
    )


def _replace_arrow_column(path: Path, column_name: str, values: Any) -> None:
    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
        arrays = [
            values if field.name == column_name else pa.array(table[field.name].to_pylist(), type=field.type)
            for field in table.schema
        ]
        column_names = list(table.schema.names)
    del table
    write_arrow_ipc_table(pa.Table.from_arrays(arrays, names=column_names), path)


@pytest.mark.parametrize(
    ("column_name", "values", "expected_type"),
    [
        ("author_affiliations", pa.array(["Institute"], type=pa.string()), r"list<string>"),
    ],
)
def test_arrow_training_rejects_noncanonical_signature_physical_types(
    tmp_path: Path,
    column_name: str,
    values: Any,
    expected_type: str,
) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    _write_minimal_signatures_table(signatures_path, ["s1"])
    _replace_arrow_column(signatures_path, column_name, values)

    with pytest.raises(
        ValueError,
        match=rf"signatures column '{column_name}' expected {expected_type}",
    ):
        load_signatures_from_arrow(signatures_path)


def test_arrow_training_rejects_noncanonical_paper_id_physical_type(tmp_path: Path) -> None:
    papers_path = tmp_path / "papers.arrow"
    authors_path = tmp_path / "paper_authors.arrow"
    _write_minimal_papers_table(papers_path, ["p1"])
    _write_minimal_paper_authors_table(authors_path, ["p1"])
    _replace_arrow_column(papers_path, "paper_id", pa.array([1], type=pa.int64()))

    with pytest.raises(ValueError, match="papers column 'paper_id' expected string"):
        load_papers_from_arrow(papers_path, authors_path)


def test_arrow_training_rejects_noncanonical_paper_author_position_type(tmp_path: Path) -> None:
    papers_path = tmp_path / "papers.arrow"
    authors_path = tmp_path / "paper_authors.arrow"
    _write_minimal_papers_table(papers_path, ["p1"])
    _write_minimal_paper_authors_table(authors_path, ["p1"])
    _replace_arrow_column(authors_path, "position", pa.array([0], type=pa.int32()))

    with pytest.raises(ValueError, match="paper_authors column 'position' expected int64"):
        load_papers_from_arrow(papers_path, authors_path)


def test_arrow_training_rejects_null_and_duplicate_signature_ids(tmp_path: Path) -> None:
    signatures_path = tmp_path / "signatures.arrow"

    _write_minimal_signatures_table(signatures_path, [None, "s1"])
    with pytest.raises(ValueError, match="null signature_id"):
        load_signatures_from_arrow(signatures_path)

    _write_minimal_signatures_table(signatures_path, ["s1", "s1"])
    with pytest.raises(ValueError, match="duplicate signature_id"):
        load_signatures_from_arrow(signatures_path)


@pytest.mark.parametrize("author_block", [None, ""])
def test_arrow_training_rejects_missing_author_block(tmp_path: Path, author_block: str | None) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    _write_minimal_signatures_table(signatures_path, ["s1"])
    _replace_arrow_column(signatures_path, "author_block", pa.array([author_block], type=pa.string()))

    with pytest.raises(ValueError, match="(null|empty) author_block"):
        load_signatures_from_arrow(signatures_path)


def test_arrow_training_rejects_absent_author_block_column(tmp_path: Path) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    _write_minimal_signatures_table(signatures_path, ["s1"])
    with pa.memory_map(str(signatures_path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
        retained_fields = [field for field in table.schema if field.name != "author_block"]
        arrays = [pa.array(table[field.name].to_pylist(), type=field.type) for field in retained_fields]
    del table
    table_without_block = pa.Table.from_arrays(
        arrays,
        names=[field.name for field in retained_fields],
    )
    write_arrow_ipc_table(table_without_block, signatures_path)

    with pytest.raises(ValueError, match="missing loader-required columns.*author_block"):
        load_signatures_from_arrow(signatures_path)


def test_arrow_training_accepts_missing_orcid_column(tmp_path: Path) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    _write_minimal_signatures_table(signatures_path, ["s1"])

    signatures = load_signatures_from_arrow(signatures_path)

    assert signatures["s1"].author_info_orcid is None


def test_arrow_training_rejects_null_and_duplicate_paper_ids(tmp_path: Path) -> None:
    papers_path = tmp_path / "papers.arrow"
    authors_path = tmp_path / "paper_authors.arrow"
    _write_minimal_paper_authors_table(authors_path, ["p1"])

    _write_minimal_papers_table(papers_path, [None, "p1"])
    with pytest.raises(ValueError, match="null paper_id"):
        load_papers_from_arrow(papers_path, authors_path)

    _write_minimal_papers_table(papers_path, ["p1", "p1"])
    with pytest.raises(ValueError, match="duplicate paper_id"):
        load_papers_from_arrow(papers_path, authors_path)


def test_arrow_training_rejects_null_paper_author_ids(tmp_path: Path) -> None:
    papers_path = tmp_path / "papers.arrow"
    authors_path = tmp_path / "paper_authors.arrow"
    _write_minimal_papers_table(papers_path, ["p1"])
    _write_minimal_paper_authors_table(authors_path, [None])

    with pytest.raises(ValueError, match="null paper_id"):
        load_papers_from_arrow(papers_path, authors_path)


def test_arrow_training_rejects_duplicate_paper_author_positions(tmp_path: Path) -> None:
    papers_path = tmp_path / "papers.arrow"
    authors_path = tmp_path / "paper_authors.arrow"
    _write_minimal_papers_table(papers_path, ["p1"])
    _write_minimal_paper_authors_table(authors_path, ["p1", "p1"])

    with pytest.raises(ValueError, match=r"duplicate \(paper_id, position\)"):
        load_papers_from_arrow(papers_path, authors_path)


@pytest.mark.parametrize("author_name", [None, "   "])
def test_arrow_training_rejects_empty_paper_author_names(tmp_path: Path, author_name: str | None) -> None:
    papers_path = tmp_path / "papers.arrow"
    authors_path = tmp_path / "paper_authors.arrow"
    _write_minimal_papers_table(papers_path, ["p1"])
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "position": pa.array([0], type=pa.int64()),
                "author_name": pa.array([author_name], type=pa.string()),
            }
        ),
        authors_path,
    )

    with pytest.raises(ValueError, match="empty author_name"):
        load_papers_from_arrow(papers_path, authors_path)


def test_arrow_training_filtered_papers_skip_irrelevant_validation_state(tmp_path: Path) -> None:
    papers_path = tmp_path / "papers.arrow"
    authors_path = tmp_path / "paper_authors.arrow"
    _write_minimal_papers_table(papers_path, ["p1", "p2", "p2"])
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1", "p2", "p2"], type=pa.string()),
                "position": pa.array([0, 0, 0], type=pa.int64()),
                "author_name": pa.array(["Ada Lovelace", None, None], type=pa.string()),
            }
        ),
        authors_path,
    )

    papers = load_papers_from_arrow(
        papers_path,
        authors_path,
        needed_paper_ids={"p1"},
    )

    assert list(papers) == ["p1"]
    assert papers["p1"].authors == [Author(author_name="Ada Lovelace", position=0)]
    with pytest.raises(ValueError, match="empty author_name"):
        load_papers_from_arrow(papers_path, authors_path)


def _json_training_anddata(name: str, specter: dict[str, np.ndarray], **overrides: Any) -> ANDData:
    kwargs: dict[str, Any] = {
        "signatures": str(DUMMY_DIR / "signatures.json"),
        "papers": str(DUMMY_DIR / "papers.json"),
        "clusters": str(DUMMY_DIR / "clusters.json"),
        "name": name,
        "mode": "train",
        "specter_embeddings": dict(specter),
        "block_type": "s2",
        "name_counts_index": tiny_name_counts_index(),
        "preprocess": True,
        "random_seed": 42,
        "n_jobs": 1,
    }
    kwargs.update(overrides)
    return ANDData(**kwargs)


@pytest.fixture(scope="module")
def training_bundle(tmp_path_factory: pytest.TempPathFactory) -> Any:
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("S2AND_BACKEND", "python")

        rng = np.random.default_rng(0)
        paper_ids = list(json.loads((DUMMY_DIR / "papers.json").read_text(encoding="utf-8")))
        specter = {str(paper_id): rng.normal(size=16).astype(np.float32) for paper_id in paper_ids}

        json_dataset = _json_training_anddata("dummy_json_training", specter)

        bundle_dir = tmp_path_factory.mktemp("arrow_training_bundle")
        arrow_paths = write_feature_block_arrow_from_anddata(json_dataset, bundle_dir, include_specter=True)
        arrow_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(arrow_paths, bundle_dir)
        name_counts_index_path, _name_counts_metrics = write_name_counts_index(
            bundle_dir, tiny_name_counts_tuple(), tiny_name_counts_provenance()
        )
        arrow_paths["name_counts_index"] = name_counts_index_path
        write_test_arrow_artifact_manifest(bundle_dir, arrow_paths)

        arrow_dataset = build_training_anddata_from_arrow(
            arrow_paths,
            "dummy_arrow_training",
            expected_normalization_version=NORMALIZATION_VERSION,
            clusters=str(DUMMY_DIR / "clusters.json"),
            block_type="s2",
            random_seed=42,
            n_jobs=1,
        )
        yield {
            "json_dataset": json_dataset,
            "arrow_dataset": arrow_dataset,
            "arrow_paths": dict(arrow_paths),
            "specter": specter,
            "monkeypatch": monkeypatch,
        }
    finally:
        monkeypatch.undo()


def _optionalish(value: Any) -> Any:
    """Compare optional strings modulo the ''/None encoding difference (the
    Arrow writer stores empty strings as nulls)."""
    return value if value else None


def test_arrow_ingestion_reconstructs_training_fields(training_bundle: dict[str, Any]) -> None:
    json_dataset: ANDData = training_bundle["json_dataset"]
    arrow_dataset: ANDData = training_bundle["arrow_dataset"]

    assert set(json_dataset.signatures) == set(arrow_dataset.signatures)
    for signature_id, json_signature in json_dataset.signatures.items():
        arrow_signature = arrow_dataset.signatures[signature_id]
        for field in ("author_info_first", "author_info_middle", "author_info_last", "author_info_suffix"):
            assert _optionalish(getattr(json_signature, field)) == _optionalish(getattr(arrow_signature, field)), field
        for field in (
            "author_info_position",
            "author_info_block",
        ):
            assert getattr(json_signature, field) == getattr(arrow_signature, field), field
        for field in (
            "author_info_first_normalized_without_apostrophe",
            "author_info_middle_normalized_without_apostrophe",
            "author_info_last_normalized",
            "author_info_suffix_normalized",
            "author_info_coauthors",
            "author_info_coauthor_blocks",
            "author_info_affiliations_n_grams",
            "author_info_coauthor_n_grams",
        ):
            assert getattr(arrow_signature, field) is None, field
        assert list(json_signature.author_info_affiliations) == list(arrow_signature.author_info_affiliations)
        assert _optionalish(json_signature.author_info_email) == _optionalish(arrow_signature.author_info_email)
        assert _optionalish(json_signature.author_info_orcid) == _optionalish(arrow_signature.author_info_orcid)
        assert str(json_signature.paper_id) == str(arrow_signature.paper_id)
        assert list(json_signature.sourced_author_ids or []) == list(arrow_signature.sourced_author_ids or [])

    json_blocks = {block: sorted(members) for block, members in json_dataset.get_blocks().items()}
    arrow_blocks = {block: sorted(members) for block, members in arrow_dataset.get_blocks().items()}
    assert json_blocks == arrow_blocks
    assert json_dataset.signature_to_cluster_id == arrow_dataset.signature_to_cluster_id

    assert {str(key) for key in json_dataset.papers} == set(arrow_dataset.papers)
    for paper_id, json_paper in json_dataset.papers.items():
        arrow_paper = arrow_dataset.papers[str(paper_id)]
        assert (json_paper.title or "") == (arrow_paper.title or "")
        assert json_paper.has_abstract == arrow_paper.has_abstract
        assert _optionalish(json_paper.venue) == _optionalish(arrow_paper.venue)
        assert _optionalish(json_paper.journal_name) == _optionalish(arrow_paper.journal_name)
        assert json_paper.year == arrow_paper.year
        assert [(author.author_name, author.position) for author in json_paper.authors] == [
            (author.author_name, author.position) for author in arrow_paper.authors
        ]

    assert arrow_dataset.specter_embeddings == {}
    assert arrow_dataset.name_counts_index is None
    assert arrow_dataset.name_counts_loaded is False


def test_arrow_training_constructor_is_always_rust_and_never_materializes_python_sidecars(
    training_bundle: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(
        ANDData,
        "preprocess_signatures",
        lambda _dataset: pytest.fail("Arrow training must not run no-op Python signature preprocessing"),
    )
    monkeypatch.setattr(
        feature_port,
        "evict_rust_featurizer",
        lambda _dataset: pytest.fail("a new Arrow-training dataset cannot have a cached featurizer"),
    )

    arrow_dataset = build_training_anddata_from_arrow(
        training_bundle["arrow_paths"],
        "dummy_arrow_training_python_backend_specter",
        expected_normalization_version=NORMALIZATION_VERSION,
        clusters=str(DUMMY_DIR / "clusters.json"),
        block_type="s2",
        random_seed=42,
        n_jobs=1,
    )

    assert arrow_dataset.runtime_context.backend == "rust"
    assert arrow_dataset.specter_embeddings == {}
    assert arrow_dataset.name_counts_index is None
    assert arrow_dataset.arrow_paths is not None
    assert arrow_dataset.arrow_artifact_generation
    assert arrow_dataset.arrow_artifact_generation == arrow_dataset.arrow_paths.generation_id
    assert arrow_dataset.arrow_paths.name_counts_manifest is not None
    assert arrow_dataset.name_counts_provenance == arrow_dataset.arrow_paths.name_counts_manifest.source_provenance
    assert isinstance(arrow_dataset.name_tuples, frozenset)


def _fixed_pair_frames(json_dataset: ANDData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signature_ids = sorted(json_dataset.signatures)
    labels = json_dataset.signature_to_cluster_id
    assert labels is not None
    # Signatures absent from the dummy cluster ground truth count as singletons.
    rows = [
        (
            first,
            second,
            "YES" if labels.get(first, f"solo_{first}") == labels.get(second, f"solo_{second}") else "NO",
        )
        for first, second in itertools.combinations(signature_ids, 2)
    ]
    frame = pd.DataFrame(rows, columns=["pair1", "pair2", "label"])
    return (
        frame.iloc[0::3].reset_index(drop=True),
        frame.iloc[1::3].reset_index(drop=True),
        frame.iloc[2::3].reset_index(drop=True),
    )


def _assert_feature_matrices_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    feature_names: list[str],
    split_name: str,
    pairs: pd.DataFrame,
) -> None:
    try:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    except AssertionError as exc:
        mismatch_mask = ~((actual == expected) | (np.isnan(actual) & np.isnan(expected)))
        details = []
        for row, column in np.argwhere(mismatch_mask)[:10]:
            pair = pairs.iloc[int(row)]
            details.append(
                f"{split_name} row={int(row)} pair=({pair['pair1']}, {pair['pair2']}, {pair['label']}) "
                f"feature={feature_names[int(column)]} actual={actual[row, column]!r} "
                f"expected={expected[row, column]!r}"
            )
        raise AssertionError("\n".join(details)) from exc


def test_featurize_end_to_end_with_fixed_pairs(
    training_bundle: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_reference: ANDData = training_bundle["json_dataset"]
    train_frame, val_frame, test_frame = _fixed_pair_frames(json_reference)
    pair_kwargs: dict[str, Any] = {
        "clusters": None,
        "train_pairs": train_frame,
        "val_pairs": val_frame,
        "test_pairs": test_frame,
    }

    json_dataset = _json_training_anddata("dummy_json_fixed_pairs", training_bundle["specter"], **pair_kwargs)
    arrow_dataset = build_training_anddata_from_arrow(
        training_bundle["arrow_paths"],
        "dummy_arrow_fixed_pairs",
        expected_normalization_version=NORMALIZATION_VERSION,
        block_type="s2",
        random_seed=42,
        n_jobs=1,
        **pair_kwargs,
    )

    featurizer_info = FeaturizationInfo(
        features_to_use=list(DEFAULT_FEATURE_GROUPS),
        featurizer_version=FEATURIZER_VERSION,
    )
    nameless_info = FeaturizationInfo(
        features_to_use=list(DEFAULT_NAMELESS_FEATURE_GROUPS),
        featurizer_version=FEATURIZER_VERSION,
    )

    feature_port.clear_rust_featurizer_cache()
    results = {}
    for label, dataset in (("json", json_dataset), ("arrow", arrow_dataset)):
        monkeypatch.setenv("S2AND_BACKEND", "rust" if label == "arrow" else "python")
        results[label] = featurize(
            dataset,
            featurizer_info,
            n_jobs=1,
            chunk_size=100,
            nameless_featurizer_info=nameless_info,
            nan_value=np.nan,
        )

    split_pairs = (train_frame, val_frame, test_frame)
    feature_names = featurizer_info.get_feature_names()
    nameless_feature_names = nameless_info.get_feature_names()
    for split_index, split_name in enumerate(("train", "val", "test")):
        json_split = results["json"][split_index]
        arrow_split = results["arrow"][split_index]
        assert json_split is not None and arrow_split is not None
        json_features, json_labels, json_nameless = json_split
        arrow_features, arrow_labels, arrow_nameless = arrow_split
        assert np.array_equal(json_labels, arrow_labels), f"{split_name} labels diverge"
        _assert_feature_matrices_close(
            json_features,
            arrow_features,
            feature_names=feature_names,
            split_name=split_name,
            pairs=split_pairs[split_index],
        )
        assert json_nameless is not None and arrow_nameless is not None
        _assert_feature_matrices_close(
            json_nameless,
            arrow_nameless,
            feature_names=nameless_feature_names,
            split_name=split_name,
            pairs=split_pairs[split_index],
        )


def test_constructor_requires_batch_indexes_and_name_counts_index(training_bundle: dict[str, Any]) -> None:
    complete_paths = dict(training_bundle["arrow_paths"])

    missing_indexes = {key: value for key, value in complete_paths.items() if "batch_index" not in key}
    with pytest.raises((ValueError, FileNotFoundError)):
        build_training_anddata_from_arrow(
            missing_indexes,
            "missing_indexes",
            expected_normalization_version=NORMALIZATION_VERSION,
            clusters=str(DUMMY_DIR / "clusters.json"),
        )

    missing_name_counts = dict(complete_paths)
    missing_name_counts.pop("name_counts_index", None)
    with pytest.raises((ValueError, FileNotFoundError)):
        build_training_anddata_from_arrow(
            missing_name_counts,
            "missing_name_counts",
            expected_normalization_version=NORMALIZATION_VERSION,
            clusters=str(DUMMY_DIR / "clusters.json"),
        )
