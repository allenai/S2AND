"""Parity tests for Arrow-native training ingestion + featurization.

Round-trip: JSON-built train ANDData -> Arrow bundle (same writers production
conversion uses) -> Arrow-built train ANDData. Parity is asserted at three
levels:

1. Ingestion: every signature/paper/specter field train mode consumes matches
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
from s2and import text as s2and_text  # noqa: E402
from s2and.arrow_training import (  # noqa: E402
    attach_training_arrow_featurizer_paths,
    build_training_anddata_from_arrow,
    load_papers_dict_from_arrow,
    load_signatures_dict_from_arrow,
    load_specter_tuple_from_arrow,
)
from s2and.consts import FEATURIZER_VERSION  # noqa: E402
from s2and.data import ANDData  # noqa: E402
from s2and.featurizer import FeaturizationInfo, featurize  # noqa: E402
from s2and.incremental_linking.feature_block import write_arrow_ipc_table  # noqa: E402
from s2and.incremental_linking.feature_block_arrow import (  # noqa: E402
    write_name_counts_index,
    write_raw_arrow_batch_lookup_indexes,
)
from scripts.arrow_conversion_helpers import write_feature_block_arrow_from_anddata  # noqa: E402
from tests.helpers import import_s2and_rust, patch_tiny_name_counts_loader, tiny_name_counts  # noqa: E402

HAS_RUST, _RUST_MODULE = import_s2and_rust()
if not HAS_RUST:
    raise pytest.skip.Exception(
        "s2and_rust extension is required for arrow training featurization parity",
        allow_module_level=True,
    )

DUMMY_DIR = Path(__file__).resolve().parent / "dummy"

# Mirrors scripts/production/model/train_pairwise.py FEATURES_TO_USE.
PRODUCTION_FEATURES = (
    "name_similarity",
    "affiliation_similarity",
    "email_similarity",
    "coauthor_similarity",
    "venue_similarity",
    "year_diff",
    "title_similarity",
    "misc_features",
    "name_counts",
    "embedding_similarity",
    "journal_similarity",
    "advanced_name_similarity",
)
NAMELESS_FEATURES = tuple(
    name for name in PRODUCTION_FEATURES if name not in {"name_similarity", "advanced_name_similarity", "name_counts"}
)


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


def test_arrow_training_rejects_null_and_duplicate_signature_ids(tmp_path: Path) -> None:
    signatures_path = tmp_path / "signatures.arrow"

    _write_minimal_signatures_table(signatures_path, [None, "s1"])
    with pytest.raises(ValueError, match="null signature_id"):
        load_signatures_dict_from_arrow(signatures_path)

    _write_minimal_signatures_table(signatures_path, ["s1", "s1"])
    with pytest.raises(ValueError, match="duplicate signature_id"):
        load_signatures_dict_from_arrow(signatures_path)


def test_arrow_training_rejects_null_and_duplicate_paper_ids(tmp_path: Path) -> None:
    papers_path = tmp_path / "papers.arrow"
    authors_path = tmp_path / "paper_authors.arrow"
    _write_minimal_paper_authors_table(authors_path, ["p1"])

    _write_minimal_papers_table(papers_path, [None, "p1"])
    with pytest.raises(ValueError, match="null paper_id"):
        load_papers_dict_from_arrow(papers_path, authors_path)

    _write_minimal_papers_table(papers_path, ["p1", "p1"])
    with pytest.raises(ValueError, match="duplicate paper_id"):
        load_papers_dict_from_arrow(papers_path, authors_path)


def test_arrow_training_rejects_null_paper_author_ids(tmp_path: Path) -> None:
    papers_path = tmp_path / "papers.arrow"
    authors_path = tmp_path / "paper_authors.arrow"
    _write_minimal_papers_table(papers_path, ["p1"])
    _write_minimal_paper_authors_table(authors_path, [None])

    with pytest.raises(ValueError, match="null paper_id"):
        load_papers_dict_from_arrow(papers_path, authors_path)


def test_arrow_training_rejects_duplicate_specter_ids(tmp_path: Path) -> None:
    specter_path = tmp_path / "specter.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1", "p1"], type=pa.string()),
                "embedding": pa.array([[1.0, 0.0], [0.0, 1.0]], type=pa.list_(pa.float32(), 2)),
            }
        ),
        specter_path,
    )

    with pytest.raises(ValueError, match="duplicate paper_id"):
        load_specter_tuple_from_arrow(specter_path)


def _json_training_anddata(name: str, specter: dict[str, np.ndarray], **overrides: Any) -> ANDData:
    kwargs: dict[str, Any] = {
        "signatures": str(DUMMY_DIR / "signatures.json"),
        "papers": str(DUMMY_DIR / "papers.json"),
        "clusters": str(DUMMY_DIR / "clusters.json"),
        "name": name,
        "mode": "train",
        "specter_embeddings": dict(specter),
        "block_type": "s2",
        "load_name_counts": tiny_name_counts(),
        "preprocess": True,
        "random_seed": 42,
        "n_jobs": 1,
    }
    kwargs.update(overrides)
    return ANDData(**kwargs)


@pytest.fixture(scope="module")
def training_bundle(tmp_path_factory: pytest.TempPathFactory) -> Any:
    monkeypatch = pytest.MonkeyPatch()
    previous_fasttext_enabled = s2and_text.fasttext_loading_enabled()
    previous_fasttext_model = s2and_text._FASTTEXT_MODEL  # noqa: SLF001
    previous_fasttext_initialized = s2and_text._FASTTEXT_MODEL_INITIALIZED  # noqa: SLF001
    previous_fasttext_load_failed = s2and_text._FASTTEXT_LOAD_FAILED  # noqa: SLF001
    try:
        patch_tiny_name_counts_loader(monkeypatch)
        monkeypatch.setenv("S2AND_BACKEND", "auto")
        s2and_text.set_fasttext_loading_enabled(False)

        rng = np.random.default_rng(0)
        paper_ids = list(json.loads((DUMMY_DIR / "papers.json").read_text(encoding="utf-8")))
        specter = {str(paper_id): rng.normal(size=16).astype(np.float32) for paper_id in paper_ids}

        json_dataset = _json_training_anddata("dummy_json_training", specter)

        bundle_dir = tmp_path_factory.mktemp("arrow_training_bundle")
        arrow_paths = write_feature_block_arrow_from_anddata(json_dataset, bundle_dir, include_specter=True)
        arrow_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(arrow_paths, bundle_dir)
        name_counts_index_path, _name_counts_metrics = write_name_counts_index(bundle_dir)
        arrow_paths["name_counts_index"] = name_counts_index_path

        arrow_dataset = build_training_anddata_from_arrow(
            arrow_paths,
            "dummy_arrow_training",
            clusters=str(DUMMY_DIR / "clusters.json"),
            mode="train",
            block_type="s2",
            load_name_counts=tiny_name_counts(),
            load_python_specter=True,
            preprocess=True,
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
        s2and_text.set_fasttext_loading_enabled(previous_fasttext_enabled)
        s2and_text._FASTTEXT_MODEL = previous_fasttext_model  # noqa: SLF001
        s2and_text._FASTTEXT_MODEL_INITIALIZED = previous_fasttext_initialized  # noqa: SLF001
        s2and_text._FASTTEXT_LOAD_FAILED = previous_fasttext_load_failed  # noqa: SLF001
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
            "author_info_first_normalized",
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

    for paper_id, vector in training_bundle["specter"].items():
        np.testing.assert_array_equal(np.asarray(arrow_dataset.specter_embeddings[paper_id]), vector)


def test_arrow_ingestion_loads_specter2_alias_embeddings(training_bundle: dict[str, Any]) -> None:
    alias_paths = dict(training_bundle["arrow_paths"])
    alias_paths["specter2"] = alias_paths.pop("specter")
    alias_paths["specter2_batch_index"] = alias_paths.pop("specter_batch_index")

    alias_dataset = build_training_anddata_from_arrow(
        alias_paths,
        "dummy_arrow_specter2_alias",
        clusters=str(DUMMY_DIR / "clusters.json"),
        mode="train",
        block_type="s2",
        load_name_counts=tiny_name_counts(),
        load_python_specter=True,
        preprocess=True,
        random_seed=42,
        n_jobs=1,
    )

    assert alias_dataset.specter_embeddings is not None
    for paper_id, vector in training_bundle["specter"].items():
        np.testing.assert_array_equal(np.asarray(alias_dataset.specter_embeddings[paper_id]), vector)

    attached_paths = getattr(alias_dataset, "rust_featurizer_arrow_paths", None)
    assert attached_paths is not None
    assert "specter" in attached_paths
    assert "specter_batch_index" in attached_paths


def test_arrow_training_skips_python_specter_by_default_when_rust_attached(
    training_bundle: dict[str, Any],
) -> None:
    arrow_dataset = build_training_anddata_from_arrow(
        training_bundle["arrow_paths"],
        "dummy_arrow_training_skip_python_specter",
        clusters=str(DUMMY_DIR / "clusters.json"),
        mode="train",
        block_type="s2",
        load_name_counts=tiny_name_counts(),
        preprocess=True,
        random_seed=42,
        n_jobs=1,
    )

    assert arrow_dataset.specter_embeddings == {}
    attached_paths = getattr(arrow_dataset, "rust_featurizer_arrow_paths", None)
    assert attached_paths is not None
    assert "specter" in attached_paths


def test_arrow_training_loads_python_specter_by_default_when_backend_python(
    training_bundle: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")

    arrow_dataset = build_training_anddata_from_arrow(
        training_bundle["arrow_paths"],
        "dummy_arrow_training_python_backend_specter",
        clusters=str(DUMMY_DIR / "clusters.json"),
        mode="train",
        block_type="s2",
        load_name_counts=tiny_name_counts(),
        preprocess=True,
        random_seed=42,
        n_jobs=1,
    )

    assert arrow_dataset.specter_embeddings != {}
    for paper_id, vector in training_bundle["specter"].items():
        np.testing.assert_array_equal(np.asarray(arrow_dataset.specter_embeddings[paper_id]), vector)


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


def test_featurize_end_to_end_with_fixed_pairs(training_bundle: dict[str, Any]) -> None:
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
        mode="train",
        block_type="s2",
        load_name_counts=tiny_name_counts(),
        preprocess=True,
        random_seed=42,
        n_jobs=1,
        **pair_kwargs,
    )

    featurizer_info = FeaturizationInfo(
        features_to_use=list(PRODUCTION_FEATURES),
        featurizer_version=FEATURIZER_VERSION,
    )
    nameless_info = FeaturizationInfo(
        features_to_use=list(NAMELESS_FEATURES),
        featurizer_version=FEATURIZER_VERSION,
    )

    feature_port.clear_rust_featurizer_cache()
    results = {}
    for label, dataset in (("json", json_dataset), ("arrow", arrow_dataset)):
        results[label] = featurize(
            dataset,
            featurizer_info,
            n_jobs=1,
            use_cache=False,
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


def test_attach_requires_batch_indexes_and_name_counts_index(training_bundle: dict[str, Any]) -> None:
    arrow_dataset: ANDData = training_bundle["arrow_dataset"]
    complete_paths = dict(training_bundle["arrow_paths"])

    missing_indexes = {key: value for key, value in complete_paths.items() if "batch_index" not in key}
    with pytest.raises((ValueError, FileNotFoundError)):
        attach_training_arrow_featurizer_paths(arrow_dataset, missing_indexes)

    missing_name_counts = dict(complete_paths)
    missing_name_counts.pop("name_counts_index", None)
    with pytest.raises((ValueError, FileNotFoundError)):
        attach_training_arrow_featurizer_paths(arrow_dataset, missing_name_counts)
