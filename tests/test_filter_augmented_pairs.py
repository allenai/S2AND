"""Reproducible augmented cleanup against real benchmark splits and draft paths."""

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

import scripts.convert_to_arrow as convert_to_arrow
from s2and._sha256 import sha256_file
from s2and.data import ANDData
from scripts.production.model.augmented_leakage import training_members_from_sources
from tests.pairwise_release_helpers import write_pairwise_dataset


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def _fixture(tmp_path: Path, *, name: str = "qian") -> dict[str, Any]:
    model_files = write_pairwise_dataset(tmp_path / "data", name=name)
    reference = ANDData(
        signatures=str(model_files["signatures"]),
        papers=str(model_files["papers"]),
        clusters=str(model_files["clusters"]),
        name=name,
        mode="train",
        preprocess=False,
        random_seed=1111,
        name_counts_index=None,
        name_tuples=frozenset(),
        n_jobs=1,
    )
    train, val, test = reference.split_cluster_signatures()
    train_ids = [sid for members in train.values() for sid in members]
    val_id = next(iter(val.values()))[0]
    test_id = next(iter(test.values()))[0]

    def prefix(sid: str) -> str:
        return f"{name}___{sid}"

    # Reordered columns, extra metadata, and duplicate rows all survive filtering.
    header = ["label", "signature_id_2", "note", "signature_id_1"]
    safe = ["YES", prefix(train_ids[1]), "keep, exactly\r\nincluding quoted CRLF", prefix(train_ids[0])]
    train_rows = [
        safe,
        list(safe),
        ["NO", prefix(train_ids[0]), "validation", prefix(val_id)],
        ["0", prefix(train_ids[0]), "test", prefix(test_id)],
        ["0", prefix(train_ids[0]), "omitted", prefix("omitted")],
        ["1", "medline___new2", "MEDLINE only in augmented", "medline___new1"],
        ["NO", "medline___new2", "external MEDLINE holdout", "medline___heldout"],
    ]
    train_path, val_path, test_path = (tmp_path / name for name in ("train.csv", "val.csv", "test.json"))
    _write_csv(train_path, header, train_rows)
    _write_csv(val_path, header, [["YES", prefix(val_id), "unchanged", prefix(train_ids[0])]])
    safe_test = {
        "signature_id_1": prefix("external1"),
        "signature_id_2": prefix("external2"),
        "label": 0,
        "review_metadata": {"note": "preserve exact record values"},
    }
    test_rows = [
        {"signature_id_1": prefix(sid), "signature_id_2": prefix("external1"), "label": 1}
        for sid in (train_ids[0], val_id, test_id)
    ] + [
        safe_test,
        {**safe_test, "signature_id_2": prefix("external3")},
        {"signature_id_1": "medline___test1", "signature_id_2": "medline___test2", "label": 1},
    ]
    test_path.write_text(json.dumps(test_rows), encoding="utf-8")
    medline_test = tmp_path / "medline_test.json"
    medline_test.write_text(
        json.dumps([{"signature_id_1": "heldout", "signature_id_2": "other", "label": 0}]), encoding="utf-8"
    )
    release_path = tmp_path / "release.json"
    # Draft paths can be relative to the release, exactly as prepare-run accepts.
    release = {
        "model": {
            "datasets": {
                name: {role: str(path.relative_to(tmp_path)) for role, path in model_files.items()},
                "augmented": {"train_pairs": train_path.name, "val_pairs": val_path.name},
            }
        },
        "evaluation": {
            "pairwise": {"augmented": {"pairs": test_path.name}, "medline": {"pairs": medline_test.name}},
            "cluster": {},
        },
    }
    release_path.write_text(json.dumps(release), encoding="utf-8")
    return {
        "release": release_path,
        "model_files": model_files,
        "train_ids": set(train_ids),
        "train": train_path,
        "validation": val_path,
        "test": test_path,
        "train_header": header,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "reference": reference,
    }


def _run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    release: Path,
    output: Path,
    *,
    seed: str = "1111",
) -> dict[str, Any]:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_arrow.py",
            "filter-augmented-pairs",
            "--release",
            str(release),
            "--random-seed",
            seed,
            "--output-root",
            str(output),
        ],
    )
    convert_to_arrow.main()
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("name", ["qian", "inventors_s2and"])
def test_filter_augmented_pairs_cli_matches_real_split_and_preserves_sources(
    tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture(tmp_path, name=name)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    output = tmp_path / "filtered"
    report = _run(monkeypatch, capsys, fixture["release"], output)
    assert report == json.loads((output / "filter_report.json").read_text())
    assert (report["train"]["input_rows"], report["train"]["retained_rows"], report["train"]["removed_rows"]) == (
        7,
        3,
        4,
    )
    assert (report["test"]["input_rows"], report["test"]["retained_rows"], report["test"]["removed_rows"]) == (6, 3, 3)
    assert report["train"]["retained_rows_by_raw_label"] == {"1": 1, "YES": 2}
    assert report["test"]["retained_rows_by_raw_label"] == {"0": 2, "1": 1}
    assert report["train"]["removed_rows_by_reason"] == {
        "medline:external_heldout": 1,
        f"{name}:outside_training_split": 3,
    }
    with (output / "train_pairs.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [fixture["train_header"], *(fixture["train_rows"][i] for i in (0, 1, 5))]
    assert (output / "val_pairs.csv").read_bytes() == original[fixture["validation"]]
    assert json.loads((output / "test_pairs.json").read_text()) == fixture["test_rows"][3:]
    assert all(path.read_bytes() == content for path, content in original.items())
    for role, entry in report["sources"].items():
        assert entry["sha256"] == sha256_file(Path(entry["path"])), role
    for name, entry in report["outputs"].items():
        assert entry["sha256"] == sha256_file(output / name)
    assert report["random_seed"] == 1111
    second_report = _run(monkeypatch, capsys, fixture["release"], tmp_path / "filtered_again")
    assert {name: row["sha256"] for name, row in report["outputs"].items()} == {
        name: row["sha256"] for name, row in second_report["outputs"].items()
    }
    with pytest.raises(FileExistsError, match="already exists"):
        _run(monkeypatch, capsys, fixture["release"], output)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown-prefix", "undeclared source datasets"),
        ("missing-model", "undeclared source datasets"),
        ("empty-train", "Filtering leaves no training or no held-out pairs"),
        ("empty-test", "Filtering leaves no training or no held-out pairs"),
        ("negative-seed", "must be non-negative"),
    ],
)
def test_filter_augmented_pairs_cli_rejects_unsafe_outputs(
    tmp_path: Path, case: str, message: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture(tmp_path)
    if case in {"unknown-prefix", "missing-model"}:
        prefix = "unknown" if case == "unknown-prefix" else "kisti"
        rows = [["YES", f"{prefix}___one", "bad", f"{prefix}___two"]]
        _write_csv(fixture["train"], fixture["train_header"], rows)
    elif case == "empty-train":
        _write_csv(fixture["train"], fixture["train_header"], [fixture["train_rows"][2]])
    elif case == "empty-test":
        fixture["test"].write_text(json.dumps(fixture["test_rows"][:3]), encoding="utf-8")
    output = tmp_path / "filtered"
    with pytest.raises(ValueError, match=re.escape(message)):
        _run(monkeypatch, capsys, fixture["release"], output, seed="-1" if case == "negative-seed" else "1111")
    assert not output.exists()


def test_training_members_uses_real_split_without_packaged_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)

    def forbid_artifact(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("split-only helper must not load the packaged name-tuple artifact")

    monkeypatch.setattr("s2and.data.load_packaged_name_tuple_artifact", forbid_artifact)
    assert training_members_from_sources("qian", fixture["model_files"], random_seed=1111) == fixture["train_ids"]


def test_filter_augmented_pairs_uses_the_requested_seed_and_original_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture(tmp_path)
    reference = fixture["reference"]
    rows = [
        ["YES", f"qian___{members[1]}", block, f"qian___{members[0]}"]
        for block, members in reference.get_blocks().items()
    ]
    _write_csv(fixture["train"], fixture["train_header"], rows)
    observed = []
    for seed in (1111, 2222):
        reference.random_seed = seed
        train, _, _ = reference.split_cluster_signatures()
        allowed = {f"qian___{sid}" for members in train.values() for sid in members}
        expected = [row for row in rows if row[1] in allowed and row[3] in allowed]
        output = tmp_path / f"filtered_{seed}"
        _run(monkeypatch, capsys, fixture["release"], output, seed=str(seed))
        with (output / "train_pairs.csv").open(newline="", encoding="utf-8") as source:
            actual = list(csv.reader(source))[1:]
        assert actual == expected
        observed.append(actual)
    assert observed[0] != observed[1]
