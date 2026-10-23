"""Native Arrow features preserve empty author lists and reject invalid paper data."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.arrow_training import load_papers_from_arrow
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
from s2and.feature_schema import FEATURE_SCHEMA
from tests.helpers import write_arrow_tables_bundle

COAUTHOR_COLUMNS = [
    index
    for index, spec in enumerate(FEATURE_SCHEMA)
    if spec.name in {"coauthor_overlap", "coauthor_similarity", "coauthor_match"}
]


def _write_bundle(
    root: Path,
    *,
    paper_ids: list[str],
    author_paper_ids: list[str],
    author_names: list[str | None],
    author_positions: list[int],
) -> dict[str, str]:
    return write_arrow_tables_bundle(
        root,
        {
            "signatures": pa.table(
                {
                    "signature_id": pa.array(["s1", "s2"], type=pa.string()),
                    "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                    "author_first": pa.array(["Ming", "Ming"], type=pa.string()),
                    "author_middle": pa.array(["", ""], type=pa.string()),
                    "author_last": pa.array(["Jiang", "Jiang"], type=pa.string()),
                    "author_suffix": pa.array(["", ""], type=pa.string()),
                    "author_affiliations": pa.array([[], []], type=pa.list_(pa.string())),
                    "author_position": pa.array([1, 0], type=pa.int64()),
                    "author_block": pa.array(["m jiang", "m jiang"], type=pa.string()),
                }
            ),
            "papers": pa.table(
                {
                    "paper_id": pa.array(paper_ids, type=pa.string()),
                    "title": pa.array(["Graph models"] * len(paper_ids), type=pa.string()),
                    "venue": pa.array([""] * len(paper_ids), type=pa.string()),
                    "journal_name": pa.array([""] * len(paper_ids), type=pa.string()),
                }
            ),
            "paper_authors": pa.table(
                {
                    "paper_id": pa.array(author_paper_ids, type=pa.string()),
                    "position": pa.array(author_positions, type=pa.int64()),
                    "author_name": pa.array(author_names, type=pa.string()),
                }
            ),
        },
    )


@pytest.mark.parametrize("other_paper_has_author", [False, True])
def test_native_empty_author_paper_preserves_missing_coauthor_features(
    tmp_path: Path, other_paper_has_author: bool
) -> None:
    paths = _write_bundle(
        tmp_path,
        paper_ids=["p1", "p2"],
        author_paper_ids=["p2"] if other_paper_has_author else [],
        author_names=["Ming Jiang"] if other_paper_has_author else [],
        author_positions=[0] if other_paper_has_author else [],
    )
    papers = load_papers_from_arrow(paths["papers"], paths["paper_authors"])
    assert papers["p1"].authors == []

    with ArrowDataset.open(tmp_path) as arrow:
        featurizer = build_rust_featurizer_from_arrow_dataset(arrow, name_tuples=set(), num_threads=1)
        assert featurizer.signature_ids() == ["s1", "s2"]
        features = np.asarray(featurizer.featurize_pairs_matrix_indexed([(0, 1)], COAUTHOR_COLUMNS, 1, np.nan))

    assert features.shape == (1, 3)
    assert np.isnan(features).all()


def test_native_empty_author_paper_still_rejects_missing_paper(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        paper_ids=["p2"],
        author_paper_ids=["p2"],
        author_names=["Ming Jiang"],
        author_positions=[0],
    )

    with ArrowDataset.open(tmp_path) as arrow:
        with pytest.raises(ValueError, match="Arrow signatures reference missing paper_id 'p1'"):
            build_rust_featurizer_from_arrow_dataset(arrow, name_tuples=set(), num_threads=1)


@pytest.mark.parametrize(
    ("author_names", "author_positions", "message"),
    [
        ([None], [0], "author_name is null"),
        (["Ming Jiang", "Another Author"], [0, 0], r"duplicate \(paper_id, position\)"),
    ],
)
def test_native_empty_author_paper_still_rejects_invalid_author_rows(
    tmp_path: Path, author_names: list[str | None], author_positions: list[int], message: str
) -> None:
    _write_bundle(
        tmp_path,
        paper_ids=["p1", "p2"],
        author_paper_ids=["p2"] * len(author_names),
        author_names=author_names,
        author_positions=author_positions,
    )

    with ArrowDataset.open(tmp_path) as arrow:
        with pytest.raises(ValueError, match=message):
            build_rust_featurizer_from_arrow_dataset(arrow, name_tuples=set(), num_threads=1)
