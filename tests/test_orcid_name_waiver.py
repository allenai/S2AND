"""Verify that block constraints drop a name conflict an ORCID's own mentions disprove."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from lightgbm import LGBMClassifier

import s2and.model as model_module
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.feature_port import _get_rust_featurizer
from s2and.featurizer import FeaturizationInfo
from s2and.model import DEFAULT_ORCID_NAME_WAIVER, Clusterer, FastCluster
from s2and.production_model import NativeLightGBMBinaryClassifier
from s2and.rust_calls import _get_constraint_labels_block_upper_triangle_indexed_rust
from tests.helpers import build_arrow_training_dataset

MIXED, OTHER = "0000-0002-1825-0097", "0000-0001-5109-3700"
# (first, middle, ORCID). The first ORCID holds 40 full first names: one "A." and two "J.".
MENTIONS: list[tuple[str, str, str | None]] = [
    *[("André", "", MIXED)] * 37,
    ("André", "A.", MIXED),
    *[("André", "J.", MIXED)] * 2,
    *[("Anna", "", OTHER)] * 5,
    ("André", "J.", None),
    ("Anna", "", None),
    ("André", "", None),
    ("André", "M.", None),
    ("André", "A.", None),
]
A_FORM, J_FORM, J_FORM_2, ANNA_ORCID = 37, 38, 39, 40
J_PLAIN, ANNA_PLAIN, PLAIN, M_PLAIN, A_PLAIN = 45, 46, 47, 48, 49


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Build a real Arrow-backed featurizer whose signature order is the list order."""
    signatures, papers = {}, {}
    for index, (first, middle, orcid) in enumerate(MENTIONS):
        signature_id, paper_id = f"{index:02d}", 100 + index
        signatures[signature_id] = {
            "signature_id": signature_id,
            "paper_id": paper_id,
            "author_info": {
                "first": first,
                "middle": middle,
                "last": "Zundert",
                "suffix": "",
                "position": 0,
                "email": None,
                "affiliations": [],
                "block": "a zundert",
                "source_id_source": "ORCID" if orcid else None,
                "source_ids": [orcid] if orcid else [],
            },
        }
        papers[str(paper_id)] = {
            "paper_id": paper_id,
            "title": "Notes on anaesthesia",
            "abstract": "",
            "year": 2000,
            "venue": "",
            "journal_name": "",
            "authors": [{"position": 0, "author_name": f"{first} Zundert"}],
        }
    dataset = ANDData(
        signatures,
        papers,
        name="orcid_name_waiver",
        mode="inference",
        name_counts_index=None,
        name_tuples=set(),
        specter_embeddings={paper_id: np.array([1.0, 0.0], dtype=np.float32) for paper_id in papers},
    )
    featurizer = _get_rust_featurizer(build_arrow_training_dataset(dataset, tmp_path_factory.mktemp("orcid-waiver")))
    assert list(featurizer.signature_ids()) == sorted(signatures)
    return featurizer


def _block_values(native: Any, **kwargs: Any) -> dict[tuple[int, int], float | None]:
    left, right, values = native.get_constraints_block_upper_triangle_indexed(
        list(range(len(MENTIONS))), high_value=LARGE_DISTANCE, num_threads=2, **kwargs
    )
    return dict(zip(zip(left, right, strict=True), values, strict=True))


def test_waiver_drops_only_the_disproved_conflicts(native: Any) -> None:
    """Waive the odd ORCID mention against a fitting name and keep every other conflict."""
    plain = _block_values(native)
    waived = _block_values(native, orcid_name_waiver=DEFAULT_ORCID_NAME_WAIVER)
    assert plain[A_FORM, J_PLAIN] == LARGE_DISTANCE
    assert waived[A_FORM, J_PLAIN] is None
    changed = {pair for pair in plain if plain[pair] != waived[pair]}
    assert changed == {(A_FORM, J_PLAIN)}
    # Kept: equal ORCIDs, a name that fits no mention of the ORCID, another ORCID,
    # and an ORCID whose own names agree.
    assert waived[A_FORM, J_FORM] == 0.0
    assert waived[A_FORM, ANNA_PLAIN] == LARGE_DISTANCE
    assert waived[A_FORM, ANNA_ORCID] == LARGE_DISTANCE
    assert waived[ANNA_ORCID, J_PLAIN] == LARGE_DISTANCE
    assert waived[J_FORM, PLAIN] is None
    # Kept: "André M." conflicts with only three of forty, but the ORCID shows no name it fits
    # that conflicts with "André A." or "André J.".
    assert waived[A_FORM, M_PLAIN] == waived[J_FORM, M_PLAIN] == LARGE_DISTANCE
    # Kept: one "André A." mention of the ORCID is not enough against its "André J." mentions.
    assert waived[J_FORM, A_PLAIN] == waived[J_FORM_2, A_PLAIN] == LARGE_DISTANCE


@pytest.mark.parametrize(
    "kwargs",
    [
        # Forty full first names are fewer than 41.
        {"orcid_name_waiver": (41, 0.1)},
        # "André J." conflicts with one of forty, more than 2%.
        {"orcid_name_waiver": (5, 0.02)},
        # Hidden ORCIDs cannot disprove a conflict.
        {"orcid_name_waiver": DEFAULT_ORCID_NAME_WAIVER, "suppress_orcid": True},
    ],
)
def test_waiver_needs_enough_fitting_orcid_mentions(native: Any, kwargs: dict[str, Any]) -> None:
    """Keep the conflict when the ORCID is too small, the name fits too little, or ORCIDs are hidden."""
    assert _block_values(native, **kwargs)[A_FORM, J_PLAIN] == LARGE_DISTANCE


def test_waiver_counts_a_repeated_signature_once(native: Any) -> None:
    """Count five ORCID mentions as five when the block lists each twice."""
    once = [0, 1, A_FORM, J_FORM, J_FORM_2]
    for waiver, value in (((5, 0.25), None), ((6, 0.25), LARGE_DISTANCE)):
        left, right, values = native.get_constraints_block_upper_triangle_indexed(
            [*once, *once, J_PLAIN], high_value=LARGE_DISTANCE, num_threads=2, orcid_name_waiver=waiver
        )
        by_pair = dict(zip(zip(left, right, strict=True), values, strict=True))
        assert by_pair[2, 10] == by_pair[7, 10] == value
    # One "André J." mention listed twice is still one.
    left, right, values = native.get_constraints_block_upper_triangle_indexed(
        [0, 1, 2, 3, A_FORM, J_FORM, J_FORM, J_PLAIN],
        high_value=LARGE_DISTANCE,
        num_threads=2,
        orcid_name_waiver=(5, 0.25),
    )
    assert dict(zip(zip(left, right, strict=True), values, strict=True))[4, 7] == LARGE_DISTANCE


@pytest.mark.parametrize("waiver", [(0, 0.1), (5, -0.1), (5, 1.5), (5, float("nan"))])
def test_block_constraints_reject_a_setting_that_needs_no_evidence(native: Any, waiver: Any) -> None:
    """Refuse zero mentions and a share outside zero to one."""
    with pytest.raises(ValueError, match="orcid_name_waiver"):
        _block_values(native, orcid_name_waiver=waiver)


@pytest.mark.parametrize(
    "waiver", ["51", (5.9, 0.1), (True, 0.1), (5, 0.1, 7), (-1, 0.1), (0, 0.1), (5, 1.5), (5, float("nan")), (5, "0.1")]
)
def test_clusterer_rejects_a_malformed_setting(waiver: Any) -> None:
    """Refuse a setting that is not a mention count and a share."""
    with pytest.raises(ValueError, match="orcid_name_waiver"):
        model_module._orcid_name_waiver_setting(SimpleNamespace(orcid_name_waiver=waiver))


def test_pair_constraints_keep_the_conflict(native: Any) -> None:
    """Leave pair calls, which see no block, on the plain name rules."""
    assert native.get_constraints_matrix_indexed([(A_FORM, J_PLAIN)], high_value=LARGE_DISTANCE) == [LARGE_DISTANCE]


def test_label_path_matches_value_path(native: Any) -> None:
    """Give the chunked label path the waiver the value path applies."""
    indices = list(range(len(MENTIONS)))
    waived = _block_values(native, orcid_name_waiver=DEFAULT_ORCID_NAME_WAIVER)
    expected = np.asarray([np.nan if value is None else value - LARGE_INTEGER for value in waived.values()])
    for start in (0, 100):
        labels = _get_constraint_labels_block_upper_triangle_indexed_rust(
            indices,
            featurizer=native,
            large_integer=LARGE_INTEGER,
            start_offset=start,
            num_threads=2,
            orcid_name_waiver=DEFAULT_ORCID_NAME_WAIVER,
        )
        np.testing.assert_array_equal(labels, expected[start:])


@pytest.fixture(scope="module")
def scorer(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Train a small pair model that the native scorer can load."""
    rng = np.random.default_rng(0)
    rows = rng.uniform(0, 1, size=(200, 7))
    model = LGBMClassifier(n_estimators=4, num_leaves=4, min_child_samples=2, verbosity=-1, n_jobs=1)
    model.fit(rows, (rows[:, 0] > 0.5).astype(int))
    path = tmp_path_factory.mktemp("scorer") / "model.txt"
    model.booster_.save_model(str(path))
    return NativeLightGBMBinaryClassifier(path, n_jobs=1)


@pytest.mark.parametrize("waiver", [DEFAULT_ORCID_NAME_WAIVER, None])
def test_one_pass_fill_matches_the_chunked_route(
    native: Any, scorer: Any, monkeypatch: pytest.MonkeyPatch, waiver: Any
) -> None:
    """Give the one-pass block fill the distances of the chunked constraint calls."""
    ids = [f"{index:02d}" for index in range(len(MENTIONS))]
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "name_similarity"]),
        classifier=scorer,
        cluster_model=FastCluster(linkage="average"),
        n_jobs=2,
        batch_size=37,
    )
    clusterer.orcid_name_waiver = waiver

    def distances() -> np.ndarray:
        return clusterer._make_distance_matrices_from_verified_rust_featurizer(
            {"block": ids}, native, signature_index_by_id={s: i for i, s in enumerate(ids)}, partial_supervision={}
        )["block"]

    one_pass = distances()
    monkeypatch.setattr(model_module, "_ONE_PASS_BLOCK_FILL", False)
    np.testing.assert_array_equal(one_pass, distances())
    offset = A_FORM * len(ids) - A_FORM * (A_FORM + 1) // 2 + J_PLAIN - A_FORM - 1
    assert (one_pass[offset] == LARGE_DISTANCE) == (waiver is None)


@pytest.mark.parametrize("waiver", [DEFAULT_ORCID_NAME_WAIVER, None, [5, 0.1]])
def test_clusterer_passes_its_setting_to_block_constraints(monkeypatch: pytest.MonkeyPatch, waiver: Any) -> None:
    """Send ``Clusterer.orcid_name_waiver`` with each block constraint call."""
    assert Clusterer.orcid_name_waiver == DEFAULT_ORCID_NAME_WAIVER == (5, 0.1)
    seen: list[Any] = []

    def constraint_labels(indices: list[int], *, max_pairs: int, **kwargs: Any) -> np.ndarray:
        seen.append(kwargs["orcid_name_waiver"])
        return np.zeros(max_pairs)

    monkeypatch.setattr(model_module, "_get_constraint_labels_block_upper_triangle_indexed_rust", constraint_labels)
    monkeypatch.setattr(
        model_module,
        "_build_block_feature_matrices_indexed_rust",
        lambda _i, *, max_pairs, **_k: (np.zeros((max_pairs, 1)), None),
    )
    monkeypatch.setattr(
        model_module, "_predict_and_combine", lambda _c, _n, _f, labels, *_a, **_k: (np.zeros(len(labels)), 0.0)
    )
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=None,
        cluster_model=FastCluster(linkage="average"),
        n_jobs=2,
    )
    clusterer.orcid_name_waiver = waiver
    clusterer._make_distance_matrices_from_verified_rust_featurizer(
        {"block": ["0", "1", "2"]}, SimpleNamespace(), signature_index_by_id={"0": 0, "1": 1, "2": 2}
    )
    assert seen == [None if waiver is None else (5, 0.1)]
