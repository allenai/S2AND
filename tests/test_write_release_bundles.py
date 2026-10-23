"""Release bundles rebuild the prepared populations from published Arrow data."""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from s2and._sha256 import sha256_file
from s2and.production_training_contract import load_model_plan, load_packaged_artifact_authority
from scripts.production.data.hash_release_emails import hash_feature_email
from scripts.production.data.write_release_bundles import write_release_bundles
from scripts.production.model.plan_datasets import (
    load_development_dataset,
    open_plan_arrow,
    plan_unknown_signatures,
)
from tests.fixture_templates import copy_template
from tests.pairwise_release_helpers import convert_release_arrow, write_pairwise_artifacts, write_pairwise_dataset


def _write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _subset(clusters: dict[str, Any], keep: set[str]) -> dict[str, Any]:
    selected = {}
    for cluster_id, cluster in clusters.items():
        members = [member for member in cluster["signature_ids"] if member in keep]
        if members:
            selected[cluster_id] = {**cluster, "signature_ids": members}
    return selected


def _write_snapshot(tmp_path: Path) -> Path:
    """Write a prepared snapshot whose pre-cut sources match its benchmark Arrow."""

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _, counts, _ = write_pairwise_artifacts(artifacts)
    snapshot = tmp_path / "snapshot"
    shutil.copytree(counts, snapshot / "name_counts_index")
    sources = tmp_path / "sources"
    files = write_pairwise_dataset(sources)
    signatures = json.loads(files["signatures"].read_text(encoding="utf-8"))
    clusters = json.loads(files["clusters"].read_text(encoding="utf-8"))
    keep = list(signatures)[:4]
    development = [key for key in signatures if key not in keep]
    benchmark = convert_release_arrow(sources, "qian", snapshot / "benchmarks", snapshot / "name_counts_index")
    _write(benchmark / "restored_unknown_signature_ids_test.json", [keep[3], development[0]])

    pairwise = snapshot / "pairwise"
    rows = [{"signature_id_1": keep[0], "signature_id_2": keep[1], "label": 1}]
    populations = {
        "model": (development, {}, [development[0]]),
        "evaluation": (keep, {"blocks": {"a adams": keep}, "pairs": rows}, [keep[3]]),
    }
    inputs: dict[str, Any] = {"release_version": "1.3", "arrow": {"model": "arrow/release"}, "sources": {}}
    for mode, (members, extra, unknown) in populations.items():
        directory = pairwise / "sources" / mode / "qian"
        roles = {
            "signatures": _write(directory / "qian_signatures.json", {key: signatures[key] for key in members}),
            "clusters": _write(directory / "qian_clusters.json", _subset(clusters, set(members))),
            "papers": files["papers"],
            "specter_embeddings": files["specter_embeddings"],
            "unknown_signatures": _write(directory / "restored_unknown_signature_ids_test.json", unknown),
        }
        roles.update({role: _write(directory / f"test_{role}.json", value) for role, value in extra.items()})
        inputs["sources"][mode] = {"qian": roles}

    augmented_source = sources / "augmented"
    copies = {}
    for key in (*keep, *development[:4]):
        signature = copy.deepcopy(signatures[key])
        signature["signature_id"] = f"qian___{key}"
        copies[signature["signature_id"]] = signature
    _write(augmented_source / "augmented_signatures.json", copies)
    for role in ("papers", "specter_embeddings"):
        (augmented_source / f"augmented_{files[role].name.partition('_')[2]}").write_bytes(files[role].read_bytes())
    convert_release_arrow(sources, "augmented", pairwise / "arrow" / "release", snapshot / "name_counts_index")
    augmented = pairwise / "sources" / "model" / "augmented"
    augmented.mkdir(parents=True)
    for path in augmented_source.iterdir():
        shutil.copyfile(path, augmented / path.name)
    (augmented / "train_pairs.csv").write_text(
        f"signature_id_1,signature_id_2,label\nqian___{development[0]},qian___{development[1]},1\n", encoding="utf-8"
    )
    (augmented / "val_pairs.csv").write_text(
        f"signature_id_1,signature_id_2,label\nqian___{development[2]},qian___{development[3]},0\n", encoding="utf-8"
    )
    test_pairs = _write(
        pairwise / "sources" / "evaluation" / "augmented" / "test_pairs.json",
        [{"signature_id_1": f"qian___{keep[0]}", "signature_id_2": f"qian___{keep[1]}", "label": 1}],
    )
    inputs["sources"]["model"]["augmented"] = {
        role: augmented / name
        for role, name in (
            ("signatures", "augmented_signatures.json"),
            ("papers", "augmented_papers.json"),
            ("specter_embeddings", "augmented_specter2.pkl"),
            ("train_pairs", "train_pairs.csv"),
            ("val_pairs", "val_pairs.csv"),
        )
    }
    inputs["sources"]["evaluation"]["augmented"] = {"pairs": test_pairs}
    for mode in inputs["sources"].values():
        for roles in mode.values():
            for role, path in roles.items():
                roles[role] = {"path": os.path.relpath(path, pairwise), "sha256": sha256_file(path)}
    _write(pairwise / "inputs.json", inputs)
    _write(
        pairwise / "model_plan.json",
        {
            "release_version": "1.3",
            "datasets": {name: {} for name in ("qian", "augmented")},
            "eps": {"grid": [0.5, 0.6], "minimum_dataset_f1": 0.0, "minimum_signature_weighted_f1": 0.0},
        },
    )
    for name in ("LICENSE.txt", "manifest.json", "replay_manifest.json", "bundle.json"):
        _write(snapshot / name, {"file": name})
    _write(snapshot / "orcid_error_query_signatures.json", [["a_khan", "error0"]])
    for name in ("datasets", "labels", "components", "splits", "corrected_sources", "corrected_replay"):
        _write(snapshot / name / "placeholder.json", {"folder": name})
    return snapshot


@pytest.fixture(scope="module")
def _snapshot_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("release_bundles")
    _write_snapshot(root)
    return root


@pytest.fixture
def snapshot(_snapshot_template: Path, tmp_path: Path) -> Path:
    copy_template(_snapshot_template, tmp_path)
    return tmp_path / "snapshot"


def _user_root(output: Path, root: Path) -> Path:
    """Merge inference and dataset the way two S3 syncs into one folder do."""

    for bundle in ("inference", "dataset"):
        shutil.copytree(output / bundle, root, dirs_exist_ok=True)
    return root


def test_bundles_rebuild_the_prepared_training_population(snapshot: Path, tmp_path: Path) -> None:
    output = tmp_path / "bundles"
    summary = write_release_bundles(snapshot, output)

    for written in ("pairwise/model_plan.json", "pairwise/augmented/manifest.json"):
        assert b"\r\n" not in (output / "dataset" / written).read_bytes()
    plan = load_model_plan(output / "dataset" / "pairwise" / "model_plan.json")
    assert summary["model_plan_sha256"] == sha256_file(output / "dataset" / "pairwise" / "model_plan.json")
    assert list(plan.datasets) == ["qian", "augmented"]
    assert set(plan.datasets["qian"].files) == {
        "arrow",
        "clusters",
        "heldout_blocks",
        "heldout_pairs",
        "unknown_signatures",
    }
    assert set(plan.datasets["augmented"].files) == {"arrow", "train_pairs", "val_pairs", "heldout_pairs"}
    assert plan.eps.grid == (0.5, 0.6)
    assert (output / "source" / "augmented" / "augmented_signatures.json").is_file()
    assert (output / "source" / "corrected_sources" / "placeholder.json").is_file()
    assert not (output / "dataset" / "corrected_sources").exists()
    assert not (output / "dataset" / "name_counts_index").exists()
    assert (output / "inference" / "name_counts_index" / "manifest.json").is_file()

    root = _user_root(output, tmp_path / "user_root")
    user_plan = load_model_plan(root / "pairwise" / "model_plan.json")
    authority = load_packaged_artifact_authority(name_counts_index_root=root / "name_counts_index")
    model_ids = set(json.loads((snapshot / "pairwise/sources/model/qian/qian_signatures.json").read_text("utf-8")))
    with open_plan_arrow(user_plan.datasets["qian"]) as arrow_dataset:
        development = load_development_dataset(
            "qian",
            user_plan.datasets["qian"],
            arrow_dataset,
            name_tuples=authority.name_tuples.pairs,
            random_seed=1111,
            n_jobs=1,
        )
        assert set(development.signatures) == model_ids
    with open_plan_arrow(user_plan.datasets["augmented"]) as arrow_dataset:
        assert arrow_dataset.name_counts_index is not None
    assert plan_unknown_signatures(user_plan.datasets["qian"]).ids == {"signature-0-3", "signature-1-0"}


def test_bundles_declare_signatures_reserved_outside_both_populations(snapshot: Path, tmp_path: Path) -> None:
    model = snapshot / "pairwise/sources/model/qian"
    signatures = json.loads((model / "qian_signatures.json").read_text(encoding="utf-8"))
    reserved = sorted(signatures)[-1]
    del signatures[reserved]
    _write(model / "qian_signatures.json", signatures)
    clusters = json.loads((model / "qian_clusters.json").read_text(encoding="utf-8"))
    _write(model / "qian_clusters.json", _subset(clusters, set(signatures)))

    output = tmp_path / "bundles"
    write_release_bundles(snapshot, output)

    plan = load_model_plan(output / "dataset" / "pairwise" / "model_plan.json")
    excluded = plan.datasets["qian"].files["excluded_signatures"]
    assert json.loads(excluded.read_text(encoding="utf-8")) == [reserved]


def test_bundles_hash_giant_emails_before_publication(snapshot: Path, tmp_path: Path) -> None:
    giant = snapshot / "datasets/a_khan"
    shutil.copytree(snapshot / "benchmarks/qian", giant)
    _write(snapshot / "replay_manifest.json", {"dataset_manifests": {"a_khan": {"sha256": "old"}}})
    _write(snapshot / "manifest.json", {"replay_bundles": {"incremental": {"sha256": "old"}}})
    source = snapshot / "corrected_replay/raw/a_khan/signatures.json"
    _write(source, {"s1": {"author_info": {"email": "J.Smith@MIT.edu."}}})
    original_sha = sha256_file(giant / "signatures.arrow")
    with pa.memory_map(str(giant / "signatures.arrow"), "r") as stream:
        original_emails = pa.ipc.open_file(stream).read_all()["author_email"].to_pylist()

    output = tmp_path / "bundles"
    write_release_bundles(snapshot, output)

    assert sha256_file(giant / "signatures.arrow") == original_sha
    with pa.memory_map(str(output / "dataset/datasets/a_khan/signatures.arrow"), "r") as stream:
        emails = pa.ipc.open_file(stream).read_all()["author_email"].to_pylist()
    assert emails == [hash_feature_email(value) for value in original_emails]
    public_source = json.loads((output / "source/corrected_replay/raw/a_khan/signatures.json").read_text())
    assert public_source["s1"]["author_info"]["email"] == hash_feature_email("J.Smith@MIT.edu.")
    assert json.loads(source.read_text())["s1"]["author_info"]["email"] == "J.Smith@MIT.edu."


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("clusters", "do not rebuild the model labels"),
        ("blocks", "held-out blocks differ"),
        ("unknown", "unknown list is not the benchmark list"),
        ("duplicate", "duplicate IDs"),
        ("orcid_errors", "orcid_error_query_signatures.json"),
    ],
)
def test_bundles_reject_a_snapshot_that_arrow_cannot_rebuild(
    snapshot: Path, tmp_path: Path, change: str, message: str
) -> None:
    sources = snapshot / "pairwise" / "sources"
    if change == "clusters":
        clusters = json.loads((sources / "model/qian/qian_clusters.json").read_text(encoding="utf-8"))
        first = next(iter(clusters.values()))
        first["signature_ids"] = list(reversed(first["signature_ids"]))
        _write(sources / "model/qian/qian_clusters.json", clusters)
    elif change == "blocks":
        _write(sources / "evaluation/qian/test_blocks.json", {"a adams": ["signature-0-0"]})
    elif change == "unknown":
        _write(sources / "model/qian/restored_unknown_signature_ids_test.json", [])
    elif change == "duplicate":
        _write(sources / "model/qian/restored_unknown_signature_ids_test.json", ["signature-1-0", "signature-1-0"])
    else:
        (snapshot / "orcid_error_query_signatures.json").unlink()

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        write_release_bundles(snapshot, tmp_path / "bundles")


def test_bundles_require_a_fresh_output(snapshot: Path, tmp_path: Path) -> None:
    (tmp_path / "bundles").mkdir()
    with pytest.raises(FileExistsError):
        write_release_bundles(snapshot, tmp_path / "bundles")
