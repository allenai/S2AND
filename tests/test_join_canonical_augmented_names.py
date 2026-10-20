"""Canonical-name join for the augmented training set, including dropped sources."""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

import scripts.convert_to_arrow as convert_to_arrow
from s2and._sha256 import sha256_file
from s2and.arrow_provenance import verify_conversion_provenance
from s2and.featurizer import DEFAULT_FEATURE_GROUPS, DEFAULT_NAMELESS_FEATURE_GROUPS, FeaturizationInfo
from s2and.name_tuple_artifact import load_name_tuple_artifact
from s2and.production_training_contract import load_model_plan
from scripts.production.model import train_pairwise
from scripts.production.model.plan_datasets import load_development_dataset, open_plan_arrow
from tests.pairwise_release_helpers import write_pairwise_artifacts, write_pairwise_dataset


def _signature(signature_id: str, first: str | None, middle: str | None, last: str) -> dict[str, Any]:
    return {
        "signature_id": signature_id,
        "paper_id": 1,
        "author_info": {
            "first": first,
            "middle": middle,
            "last": last,
            "position": 0,
            "block": "j kim",
            "affiliations": ["Lab"],
            "suffix": None,
            "email": None,
        },
    }


SOURCE_SIGNATURES = {
    "alpha": {
        "s1": _signature("s1", "Jae", "Hyun", "Kim"),
        "s2": _signature("s2", "Maria", None, "Silva"),
        "s3": _signature("s3", "J", None, "Kim"),
        "s4": _signature("s4", None, None, "Park"),
        "s5": _signature("s5", "Otto", None, "Berg"),
    },
    "orcid": {"o1": _signature("o1", "Ann", None, "Lee")},
}
CANONICAL_ROWS = {
    "alpha": [
        {"signature_id": "s1", "first": "Jae-Hyun", "middle": "", "last": "Kim"},
        {"signature_id": "s2", "first": "Maria", "middle": None, "last": "da Silva"},
        {"signature_id": "s3", "first": "Jin", "middle": None, "last": "Kim"},
        {"signature_id": "s4", "first": None, "middle": None, "last": "Park"},
        {"signature_id": "s5", "first": "", "middle": None, "last": "Berg"},
    ],
}


def _augmented() -> dict[str, dict[str, Any]]:
    """alpha___s2 and alpha___s5 were shortened to an initial by the augmentation; the others were not."""
    return {
        "alpha___s1": _signature("alpha___s1", "Jae", "Hyun", "Kim"),
        "alpha___s2": _signature("alpha___s2", "M", None, "Silva"),
        "alpha___s3": _signature("alpha___s3", "J", None, "Kim"),
        "alpha___s4": _signature("alpha___s4", None, None, "Park"),
        "alpha___s5": _signature("alpha___s5", "O", None, "Berg"),
        "orcid___o1": _signature("orcid___o1", "A", None, "Lee"),
    }


def _names(signatures: dict[str, dict[str, Any]]) -> dict[str, tuple[Any, ...]]:
    return {
        key: tuple(value["author_info"][field] for field in ("first", "middle", "last"))
        for key, value in signatures.items()
    }


def test_augmented_join_applies_source_names_and_reshortens_first_names() -> None:
    joined, report = convert_to_arrow.join_canonical_augmented_names(
        _augmented(),
        SOURCE_SIGNATURES,
        CANONICAL_ROWS,
        drop_sources=frozenset({"orcid"}),
    )

    assert _names(joined) == {
        "alpha___s1": ("Jae-Hyun", "", "Kim"),
        "alpha___s2": ("M", None, "da Silva"),
        "alpha___s3": ("Jin", None, "Kim"),
        "alpha___s4": (None, None, "Park"),
        "alpha___s5": ("", None, "Berg"),
    }
    assert joined["alpha___s2"]["author_info"]["affiliations"] == ["Lab"]
    assert report == {
        "kept_signatures": {"alpha": 5},
        "dropped_signatures": {"orcid": 1},
        "changed_signatures": 4,
        "field_changes": {"first": 3, "middle": 1, "last": 1},
        "shortened_first_names": 2,
        "single_character_first_names": 1,
        "single_character_first_names_lengthened": 1,
    }


@pytest.mark.parametrize(
    ("augmented_update", "canonical_update", "drop_sources", "message"),
    [
        ({}, {}, frozenset(), "source signatures and canonical names for: \\['orcid'\\]"),
        ({}, {}, frozenset({"orcid", "ORCID"}), "drop sources match no augmented signatures: \\['ORCID'\\]"),
        ({"alpha___s1": _signature("alpha___s1", "Jae", None, "Kim")}, {}, frozenset({"orcid"}), "no longer match"),
        ({"alpha___s2": _signature("alpha___s2", "Mia", None, "Silva")}, {}, frozenset({"orcid"}), "no longer match"),
        ({"alpha___s1": _signature("alpha___s1", "Jae", "Hyun", "Kimm")}, {}, frozenset({"orcid"}), "no longer match"),
        ({"alpha___s9": _signature("alpha___s9", "X", None, "Y")}, {}, frozenset({"orcid"}), "no original signature"),
        ({"alphas1": _signature("alphas1", "Jae", "Hyun", "Kim")}, {}, frozenset({"orcid"}), "<dataset>___"),
        ({}, {"s2": "(Maria)"}, frozenset({"orcid"}), "do not start with a letter or digit"),
    ],
    ids=[
        "undropped-source-without-export",
        "unmatched-drop-source",
        "middle-changed",
        "first-changed",
        "last-changed",
        "unknown-signature",
        "malformed-id",
        "unshortenable-canonical-first",
    ],
)
def test_augmented_join_rejects_inconsistent_inputs(augmented_update, canonical_update, drop_sources, message) -> None:
    augmented = {**_augmented(), **augmented_update}
    canonical_rows = {
        "alpha": [
            {**row, "first": canonical_update.get(row["signature_id"], row["first"])} for row in CANONICAL_ROWS["alpha"]
        ]
    }

    with pytest.raises(ValueError, match=message):
        convert_to_arrow.join_canonical_augmented_names(
            augmented,
            SOURCE_SIGNATURES,
            canonical_rows,
            drop_sources=drop_sources,
        )


def _write_inputs(tmp_path: Path, *, train_pairs: str | None = None) -> tuple[Path, Path, Path]:
    augmented_dir = tmp_path / "augmented"
    augmented_dir.mkdir()
    (augmented_dir / "augmented_signatures.json").write_text(json.dumps(_augmented()), encoding="utf-8")
    (augmented_dir / "train_pairs.csv").write_text(
        train_pairs
        or "pair1,pair2,label\nalpha___s1,alpha___s2,NO\norcid___o1,alpha___s1,NO\nalpha___s1,alpha___s3,YES\n",
        encoding="utf-8",
    )
    (augmented_dir / "val_pairs.csv").write_text("pair1,pair2,label\nalpha___s2,alpha___s3,NO\n", encoding="utf-8")
    (augmented_dir / "test_pairs.csv").write_text(
        "pairs1,pair2,label\norcid___o1,alpha___s3,NO\nalpha___s4,alpha___s5,YES\n", encoding="utf-8"
    )
    source_root = tmp_path / "source"
    for dataset, signatures in SOURCE_SIGNATURES.items():
        (source_root / dataset).mkdir(parents=True)
        (source_root / dataset / f"{dataset}_signatures.json").write_text(json.dumps(signatures), encoding="utf-8")
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    (canonical_root / "alpha.json").write_text(json.dumps(CANONICAL_ROWS["alpha"]), encoding="utf-8")
    return augmented_dir, source_root, canonical_root


def _run_cli(monkeypatch: pytest.MonkeyPatch, inputs: tuple[Path, Path, Path], output_dir: Path) -> None:
    augmented_dir, source_root, canonical_root = inputs
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_arrow.py",
            "join-canonical-augmented-names",
            "--augmented-dir",
            str(augmented_dir),
            "--source-root",
            str(source_root),
            "--canonical-names-root",
            str(canonical_root),
            "--drop-sources",
            "orcid",
            "--output-dir",
            str(output_dir),
        ],
    )
    convert_to_arrow.main()


def test_augmented_join_cli_writes_release_ready_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "joined"

    _run_cli(monkeypatch, _write_inputs(tmp_path), output_dir)

    signatures = json.loads((output_dir / "augmented_signatures.json").read_text(encoding="utf-8"))
    assert sorted(signatures) == ["alpha___s1", "alpha___s2", "alpha___s3", "alpha___s4", "alpha___s5"]
    # prepare-run reads training pairs by signature_id_1/signature_id_2; labels are kept as written.
    assert (output_dir / "train_pairs.csv").read_text(encoding="utf-8") == (
        "signature_id_1,signature_id_2,label\nalpha___s1,alpha___s2,NO\nalpha___s1,alpha___s3,YES\n"
    )
    assert (output_dir / "val_pairs.csv").read_text(encoding="utf-8") == (
        "signature_id_1,signature_id_2,label\nalpha___s2,alpha___s3,NO\n"
    )
    # Held-out pairs use the pairwise-evaluation JSON format with integer labels.
    assert json.loads((output_dir / "test_pairs.json").read_text(encoding="utf-8")) == [
        {"signature_id_1": "alpha___s4", "signature_id_2": "alpha___s5", "label": 1}
    ]
    report = json.loads((output_dir / "join_report.json").read_text(encoding="utf-8"))
    assert report["pairs"] == {
        "train_pairs.csv": {"kept": 2, "dropped": 1},
        "val_pairs.csv": {"kept": 1, "dropped": 0},
        "test_pairs.csv": {"kept": 1, "dropped": 1},
    }
    assert report["drop_sources"] == ["orcid"]
    assert not list(tmp_path.glob(".joined.staging-*"))


def test_joined_augmented_feature_only_staging_converts_and_selects_frozen_training_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Follow the private release runbook without requiring or reading held-out CSV labels."""

    joined = tmp_path / "joined"
    _run_cli(monkeypatch, _write_inputs(tmp_path), joined)
    originals = write_pairwise_dataset(tmp_path / "papers", name="augmented")
    paper = next(iter(json.loads(originals["papers"].read_text(encoding="utf-8")).values()))
    paper["paper_id"] = "1"
    originals["papers"].write_text(json.dumps({"1": paper}), encoding="utf-8")
    with originals["specter_embeddings"].open("rb") as source:
        embedding = next(iter(pickle.load(source).values()))
    with originals["specter_embeddings"].open("wb") as output:
        pickle.dump({"1": embedding}, output)
    files = {
        "signatures": joined / "augmented_signatures.json",
        "papers": originals["papers"],
        "specter_embeddings": originals["specter_embeddings"],
        "train_pairs": joined / "train_pairs.csv",
        "val_pairs": joined / "val_pairs.csv",
    }
    staged = tmp_path / "release_feature_sources" / "augmented"
    staged.mkdir(parents=True)
    # These are the explicit feature-source roles copied by the release runbook.
    for role, filename in (
        ("signatures", "augmented_signatures.json"),
        ("papers", "augmented_papers.json"),
        ("specter_embeddings", "augmented_specter2.pkl"),
    ):
        shutil.copyfile(files[role], staged / filename)
    assert sorted(path.suffix for path in staged.iterdir()) == [".json", ".json", ".pkl"]
    # The converter and selection authority must not need held-out answers after the join.
    (joined / "test_pairs.json").write_text("not read during feature conversion or training", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    tuple_path, counts_path, _ = write_pairwise_artifacts(artifact_root)
    arrow_root = tmp_path / "arrow"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_arrow.py",
            "benchmark",
            "--source-root",
            str(staged.parent),
            "--output-root",
            str(arrow_root),
            "--datasets",
            "augmented",
            "--name-counts-index-root",
            str(counts_path),
            "--n-jobs",
            "1",
        ],
    )
    convert_to_arrow.main()
    verify_conversion_provenance(
        arrow_root / "augmented",
        expected_source_sha256={
            role: sha256_file(files[role]) for role in ("signatures", "papers", "specter_embeddings")
        },
    )
    plan_files = {
        "arrow": arrow_root / "augmented" / "manifest.json",
        "train_pairs": joined / "train_pairs.csv",
        "val_pairs": joined / "val_pairs.csv",
        "heldout_pairs": joined / "test_pairs.json",
    }
    plan_path = tmp_path / "model_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "release_version": "1.3",
                "datasets": {
                    "augmented": {
                        role: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                        for role, path in plan_files.items()
                    }
                },
                "eps": {"grid": [0.5], "minimum_dataset_f1": 0.0, "minimum_signature_weighted_f1": 0.0},
            }
        ),
        encoding="utf-8",
    )
    model_input = load_model_plan(plan_path).datasets["augmented"]
    with open_plan_arrow(model_input) as arrow:
        assert arrow.name_counts_index is not None
        selection = load_development_dataset(
            "augmented",
            model_input,
            arrow,
            name_tuples=load_name_tuple_artifact(tuple_path).pairs,
            random_seed=1111,
            n_jobs=1,
        )
        train, validation = train_pairwise._featurize_selection(
            selection,
            arrow,
            FeaturizationInfo(features_to_use=list(DEFAULT_FEATURE_GROUPS)),
            name_counts_manifest_sha256=arrow.name_counts_index.manifest_sha256,
            nameless_featurizer_info=FeaturizationInfo(features_to_use=list(DEFAULT_NAMELESS_FEATURE_GROUPS)),
            n_jobs=1,
            total_ram_bytes=None,
        )
    assert train[0].shape[0] == 2 and validation[0].shape[0] == 1
    assert train[1].tolist() == [0.0, 1.0]
    assert validation[1].tolist() == [0.0]
    for role in ("train_pairs", "val_pairs"):
        assert sha256_file(model_input.files[role]) == model_input.sha256[role]


@pytest.mark.parametrize(
    ("train_pairs", "message"),
    [
        ("pair1,pair2,label\nalpha___s1,alpha___s7,NO\n", "missing from the joined set"),
        ("pair1,pair2,label\norcid___o1,alpha___s1,NO\n", "every pair in"),
    ],
    ids=["kept-pair-references-missing-signature", "all-pairs-dropped"],
)
def test_augmented_join_cli_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, train_pairs: str, message: str
) -> None:
    output_dir = tmp_path / "joined"

    with pytest.raises(ValueError, match=message):
        _run_cli(monkeypatch, _write_inputs(tmp_path, train_pairs=train_pairs), output_dir)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".joined.staging-*"))


def test_augmented_join_cli_refuses_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "joined"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        _run_cli(monkeypatch, _write_inputs(tmp_path), output_dir)
