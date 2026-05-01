import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from scripts.extract_big_block_dataset import (
    census_monolith,
    extract_monolith_dataset,
    iter_monolith_records,
)

MONOLITH_FIXTURE = """{
  "signatures" : [ {
    "signature_id" : "s1",
    "author_info" : {
      "position" : 0,
      "block" : "h wang",
      "first" : "Han",
      "middle" : null,
      "last" : "Wang",
      "suffix" : null,
      "email" : null,
      "affiliations" : [ "Org 1" ],
      "source_id_source" : "Extracted",
      "source_ids" : [ "Han Wang" ]
    },
    "paper_id" : 101
  }, {
    "signature_id" : "s2",
    "author_info" : {
      "position" : 1,
      "block" : "h wang",
      "first" : "Hao",
      "middle" : null,
      "last" : "Wang",
      "suffix" : null,
      "email" : null,
      "affiliations" : [ "Org 2" ],
      "source_id_source" : "Extracted",
      "source_ids" : [ "Hao Wang" ]
    },
    "paper_id" : 102
  } ],
  "papers" : [ {
    "paper_id" : 101,
    "title" : "Pap\u00e9 One",
    "abstract" : "Has Abstract",
    "journal_name" : "J1",
    "venue" : "J1",
    "year" : 2020,
    "sources" : [ ],
    "fields_of_study" : [ "CS" ],
    "authors" : [ {
      "position" : 0,
      "author_name" : "Han Wang"
    } ]
  }, {
    "paper_id" : 102,
    "title" : "Paper Two",
    "abstract" : "Has Abstract",
    "journal_name" : "J2",
    "venue" : "J2",
    "year" : 2021,
    "sources" : [ ],
    "fields_of_study" : [ "Physics" ],
    "authors" : [ {
      "position" : 0,
      "author_name" : "Hao Wang"
    } ]
  } ],
  "paper_embeddings" : {
    "101" : [ 0.1, 0.2, 0.3 ],
    "102" : [ -0.1, 0.0, 0.5 ]
  },
  "cluster_seeds" : { },
  "altered_cluster_signatures" : [ ]
}
"""


def _minify_fixture(pretty_fixture: str) -> str:
    return json.dumps(json.loads(pretty_fixture), separators=(",", ":"))


@pytest.fixture(params=[MONOLITH_FIXTURE, _minify_fixture(MONOLITH_FIXTURE)], ids=["pretty", "minified"])
def monolith_fixture_text(request) -> str:
    return request.param


def _write_fixture(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_iter_monolith_records_yields_expected_sequence(tmp_path, monolith_fixture_text):
    input_path = tmp_path / "big_block_fixture.json"
    _write_fixture(input_path, monolith_fixture_text)

    records = list(iter_monolith_records(input_path))

    assert [record_type for record_type, _ in records] == [
        "signatures",
        "signatures",
        "papers",
        "papers",
        "paper_embedding",
        "paper_embedding",
    ]
    assert records[0][1]["signature_id"] == "s1"
    assert records[2][1]["paper_id"] == 101
    assert records[4][1][0] == "101"
    np.testing.assert_allclose(records[4][1][1], np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_iter_monolith_records_handles_minified_chunk_boundaries(tmp_path):
    input_path = tmp_path / "big_block_fixture.json"
    _write_fixture(input_path, _minify_fixture(MONOLITH_FIXTURE))

    records = list(iter_monolith_records(input_path, chunk_size=64))

    assert [record_type for record_type, _ in records] == [
        "signatures",
        "signatures",
        "papers",
        "papers",
        "paper_embedding",
        "paper_embedding",
    ]


def test_census_monolith_respects_limit_signatures(tmp_path, monolith_fixture_text):
    input_path = tmp_path / "big_block_fixture.json"
    _write_fixture(input_path, monolith_fixture_text)

    census = census_monolith(input_path, limit_signatures=1)

    assert census.signature_count == 1
    assert census.paper_count == 1
    assert census.embedding_count == 1
    assert census.embedding_dim == 3
    assert census.needed_paper_ids == {"101"}
    assert census.block_counts == {"h wang": 1}


def test_extract_monolith_dataset_writes_anddata_friendly_outputs(tmp_path, monolith_fixture_text):
    input_path = tmp_path / "big_block_fixture.json"
    output_dir = tmp_path / "h_wang"
    _write_fixture(input_path, monolith_fixture_text)

    meta = extract_monolith_dataset(input_path, output_dir, limit_signatures=1)

    signatures = json.loads((output_dir / "signatures.json").read_text(encoding="utf-8"))
    papers_text = (output_dir / "papers.json").read_text(encoding="utf-8")
    papers = json.loads(papers_text)
    cluster_seeds = json.loads((output_dir / "cluster_seeds.json").read_text(encoding="utf-8"))
    altered = (output_dir / "altered_cluster_signatures.txt").read_text(encoding="utf-8")
    meta_from_disk = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))

    assert list(signatures) == ["s1"]
    assert signatures["s1"]["paper_id"] == 101
    assert list(papers) == ["101"]
    assert papers["101"]["title"] == "Pap\u00e9 One"
    assert "\\u00e9" in papers_text
    assert cluster_seeds == {}
    assert altered == ""
    assert meta_from_disk["signature_count"] == 1
    assert meta["paper_count"] == 1

    with (output_dir / "specter.pickle").open("rb") as infile:
        matrix, keys = pickle.load(infile)

    assert keys == ["101"]
    assert matrix.shape == (1, 3)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix[0], np.array([0.1, 0.2, 0.3], dtype=np.float32))
