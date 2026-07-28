from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from s2and.arrow_inputs import (
    ARROW_COLLECTION_KIND,
    PUBLIC_DATA_KIND,
    build_arrow_artifact_manifest,
    require_name_counts_index_artifact,
)
from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
from s2and.incremental_linking.feature_block import (
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
    write_name_counts_index,
)
from scripts.verification.validate_local_arrow_release import validate_release_root
from tests.helpers import tiny_name_counts_tuple


def _touch_json(path: Path, payload: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({} if payload is None else payload), encoding="utf-8")


def _touch_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _validate_required_release_files(release_root: Path, dataset_name: str) -> None:
    pa = pytest.importorskip("pyarrow")
    root_manifest = json.loads((release_root / "manifest.json").read_text(encoding="utf-8"))
    dataset_manifest_path = release_root / dataset_name / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    manifest_paths = dataset_manifest.get("paths", {}) if isinstance(dataset_manifest, dict) else {}
    embedding_path = manifest_paths.get("specter", "specter.arrow")
    dataset_entry = root_manifest["dataset_manifests"][dataset_name]
    assert set(dataset_entry) == {"path", "sha256"}
    assert dataset_entry["sha256"] == hashlib.sha256(dataset_manifest_path.read_bytes()).hexdigest()
    required_paths = [
        release_root / "manifest.json",
        release_root / "LICENSE.txt",
        release_root / "name_counts_index" / "manifest.json",
        release_root / dataset_name / "manifest.json",
        release_root / dataset_name / "signatures.arrow",
        release_root / dataset_name / "papers.arrow",
        release_root / dataset_name / "paper_authors.arrow",
        release_root / dataset_name / str(embedding_path),
        release_root / dataset_name / "signatures.signatures_batch_index.bin",
    ]

    missing_paths = [path.relative_to(release_root) for path in required_paths if not path.exists()]
    assert missing_paths == []
    require_name_counts_index_artifact(
        release_root / "name_counts_index",
        context="release layout test",
        producer_hint="test fixture must include complete name_counts_index",
    )
    for key in ("signatures", "papers", "paper_authors", "specter"):
        path = release_root / dataset_name / manifest_paths[key]
        with pa.memory_map(str(path), "r") as source:
            assert pa.ipc.open_file(source).read_all().num_rows >= 1


def _write_root_manifest(
    release_root: Path,
    dataset_name: str,
    *,
    replay_bundles: dict[str, dict[str, str]] | None = None,
) -> None:
    dataset_manifest_path = release_root / dataset_name / "manifest.json"
    dataset_manifest_bytes = dataset_manifest_path.read_bytes()
    root_manifest = {
        "kind": PUBLIC_DATA_KIND,
        "release_version": "1.3",
        "format_version": PUBLIC_DATA_FORMAT_VERSION,
        "dataset_manifests": {
            dataset_name: {
                "path": f"{dataset_name}/manifest.json",
                "sha256": hashlib.sha256(dataset_manifest_bytes).hexdigest(),
            }
        },
    }
    if replay_bundles is not None:
        root_manifest["replay_bundles"] = replay_bundles
    _touch_json(release_root / "manifest.json", root_manifest)


def _build_arrow_release_fixture(tmp_path: Path, dataset_name: str = "s2and_mini") -> tuple[Path, str]:
    pa = pytest.importorskip("pyarrow")
    release_root = tmp_path / "release"

    write_name_counts_index(
        release_root,
        tiny_name_counts_tuple(),
    )

    for file_path in (release_root / "LICENSE.txt",):
        _touch_file(file_path)
    dataset_root = release_root / dataset_name
    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["s1"], type=pa.string()),
                "paper_id": pa.array(["p1"], type=pa.string()),
                "author_first": pa.array(["Ada"], type=pa.string()),
                "author_middle": pa.array([""], type=pa.string()),
                "author_last": pa.array(["Lovelace"], type=pa.string()),
                "author_suffix": pa.array([""], type=pa.string()),
                "author_affiliations": pa.array([[]], type=pa.list_(pa.string())),
                "author_position": pa.array([0], type=pa.int64()),
            }
        ),
        dataset_root / "signatures.arrow",
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "title": pa.array(["Notes"], type=pa.string()),
                "venue": pa.array([""], type=pa.string()),
                "journal_name": pa.array([""], type=pa.string()),
            }
        ),
        dataset_root / "papers.arrow",
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "position": pa.array([0], type=pa.int64()),
                "author_name": pa.array(["Ada Lovelace"], type=pa.string()),
            }
        ),
        dataset_root / "paper_authors.arrow",
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([1.0, 0.0], type=pa.float32()), 2),
            }
        ),
        dataset_root / "specter2.arrow",
    )
    write_arrow_batch_lookup_index(
        dataset_root / "signatures.arrow",
        dataset_root / "signatures.signatures_batch_index.bin",
        key_column="signature_id",
    )
    write_arrow_batch_lookup_index(
        dataset_root / "papers.arrow",
        dataset_root / "papers.papers_batch_index.bin",
        key_column="paper_id",
    )
    write_arrow_batch_lookup_index(
        dataset_root / "paper_authors.arrow",
        dataset_root / "paper_authors.paper_authors_batch_index.bin",
        key_column="paper_id",
    )
    write_arrow_batch_lookup_index(
        dataset_root / "specter2.arrow",
        dataset_root / "specter2.specter_batch_index.bin",
        key_column="paper_id",
    )
    generation_paths = {
        "signatures": str(dataset_root / "signatures.arrow"),
        "papers": str(dataset_root / "papers.arrow"),
        "paper_authors": str(dataset_root / "paper_authors.arrow"),
        "specter": str(dataset_root / "specter2.arrow"),
        "name_counts_index": str(release_root / "name_counts_index"),
        "signatures_batch_index": str(dataset_root / "signatures.signatures_batch_index.bin"),
        "papers_batch_index": str(dataset_root / "papers.papers_batch_index.bin"),
        "paper_authors_batch_index": str(dataset_root / "paper_authors.paper_authors_batch_index.bin"),
        "specter_batch_index": str(dataset_root / "specter2.specter_batch_index.bin"),
    }
    dataset_manifest = build_arrow_artifact_manifest(
        generation_paths,
        dataset_root,
    )
    _touch_json(dataset_root / "manifest.json", dataset_manifest)
    _write_root_manifest(release_root, dataset_name)
    return release_root, dataset_name


def _rewrite_dataset_manifest_paths(release_root: Path, dataset_name: str, paths: dict[str, str]) -> None:
    dataset_manifest_path = release_root / dataset_name / "manifest.json"
    dataset_root = dataset_manifest_path.parent
    resolved_paths = {
        key: str((dataset_root / value).resolve()) if not Path(value).is_absolute() else value
        for key, value in paths.items()
    }
    _touch_json(
        dataset_manifest_path,
        build_arrow_artifact_manifest(resolved_paths, dataset_root),
    )
    _write_root_manifest(release_root, dataset_name)


def test_arrow_release_layout_required_files(tmp_path: Path) -> None:
    release_root, dataset_name = _build_arrow_release_fixture(tmp_path)

    _validate_required_release_files(release_root, dataset_name)
    assert validate_release_root(release_root) == {
        "release_root": str(release_root.resolve()),
        "dataset_manifest_count": 1,
        "replay_dataset_manifest_count": 0,
        "name_counts_index": str(release_root.resolve() / "name_counts_index"),
        "network_access": False,
    }


@pytest.mark.parametrize(
    ("case_id", "message"),
    (
        ("dataset-checksum", r"dataset_manifests\..*\.sha256 mismatch"),
        ("signatures-file", r"signatures\.arrow"),
        ("papers-index", r"missing required Arrow artifacts.*papers_batch_index"),
        ("specter-index", r"must contain both specter and specter_batch_index"),
        ("replay-manifest", r"replay_bundles\.mini-replay manifest does not exist"),
        ("nested-replay", r"fields mismatch.*extra=\['replay_bundles'\]"),
    ),
)
def test_validate_release_root_reports_corrupt_or_missing_artifacts(
    tmp_path: Path,
    case_id: str,
    message: str,
) -> None:
    release_root, dataset_name = _build_arrow_release_fixture(tmp_path)
    if case_id == "dataset-checksum":
        root_manifest_path = release_root / "manifest.json"
        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        root_manifest["dataset_manifests"][dataset_name]["sha256"] = "0" * 64
        _touch_json(root_manifest_path, root_manifest)
    elif case_id == "signatures-file":
        (release_root / dataset_name / "signatures.arrow").unlink()
    elif case_id in {"papers-index", "specter-index"}:
        manifest_path = release_root / dataset_name / "manifest.json"
        paths = json.loads(manifest_path.read_text(encoding="utf-8"))["paths"]
        del paths["papers_batch_index" if case_id == "papers-index" else "specter_batch_index"]
        _rewrite_dataset_manifest_paths(release_root, dataset_name, paths)
    elif case_id == "replay-manifest":
        _write_root_manifest(
            release_root,
            dataset_name,
            replay_bundles={"mini-replay": {"path": "replay/manifest.json", "sha256": "0" * 64}},
        )
    else:
        replay_root = release_root / "replay"
        replay_dataset_manifest = replay_root / "dataset" / "manifest.json"
        nested_manifest = replay_root / "nested" / "manifest.json"
        _touch_json(replay_dataset_manifest)
        _touch_json(nested_manifest)
        replay_manifest = replay_root / "manifest.json"
        _touch_json(
            replay_manifest,
            {
                "kind": ARROW_COLLECTION_KIND,
                "format_version": PUBLIC_DATA_FORMAT_VERSION,
                "dataset_manifests": {
                    "dataset": {
                        "path": "dataset/manifest.json",
                        "sha256": hashlib.sha256(replay_dataset_manifest.read_bytes()).hexdigest(),
                    }
                },
                "replay_bundles": {
                    "nested": {
                        "path": "nested/manifest.json",
                        "sha256": hashlib.sha256(nested_manifest.read_bytes()).hexdigest(),
                    }
                },
            },
        )
        _write_root_manifest(
            release_root,
            dataset_name,
            replay_bundles={
                "replay": {
                    "path": "replay/manifest.json",
                    "sha256": hashlib.sha256(replay_manifest.read_bytes()).hexdigest(),
                }
            },
        )

    with pytest.raises(ValueError, match=message):
        validate_release_root(release_root)


def test_validate_release_root_owns_name_counts_topology(tmp_path: Path) -> None:
    release_root, dataset_name = _build_arrow_release_fixture(tmp_path)
    alternate_root = release_root / "alternate"
    alternate_index, _metrics = write_name_counts_index(alternate_root, tiny_name_counts_tuple())
    dataset_manifest = json.loads((release_root / dataset_name / "manifest.json").read_text(encoding="utf-8"))
    paths = dataset_manifest["paths"]
    paths["name_counts_index"] = str(Path(alternate_index).resolve())
    _rewrite_dataset_manifest_paths(release_root, dataset_name, paths)

    with pytest.raises(ValueError, match="must resolve to the publication root index"):
        validate_release_root(release_root)


def test_validate_release_root_threads_publication_root_into_replay_bundles(tmp_path: Path) -> None:
    release_root, dataset_name = _build_arrow_release_fixture(tmp_path)
    source_dataset_root = release_root / dataset_name
    replay_root = release_root / "replay"
    replay_dataset_root = replay_root / "datasets" / "nested"
    shutil.copytree(source_dataset_root, replay_dataset_root)

    copied_manifest = json.loads((replay_dataset_root / "manifest.json").read_text(encoding="utf-8"))
    replay_paths = {
        key: (
            str(release_root / "name_counts_index")
            if key == "name_counts_index"
            else str(replay_dataset_root / Path(value).name)
        )
        for key, value in copied_manifest["paths"].items()
    }
    _touch_json(
        replay_dataset_root / "manifest.json",
        build_arrow_artifact_manifest(replay_paths, replay_dataset_root),
    )
    replay_dataset_manifest_bytes = (replay_dataset_root / "manifest.json").read_bytes()
    _touch_json(
        replay_root / "manifest.json",
        {
            "kind": ARROW_COLLECTION_KIND,
            "format_version": PUBLIC_DATA_FORMAT_VERSION,
            "dataset_manifests": {
                "nested": {
                    "path": "datasets/nested/manifest.json",
                    "sha256": hashlib.sha256(replay_dataset_manifest_bytes).hexdigest(),
                }
            },
        },
    )
    _write_root_manifest(
        release_root,
        dataset_name,
        replay_bundles={
            "mini-replay": {
                "path": "replay/manifest.json",
                "sha256": hashlib.sha256((replay_root / "manifest.json").read_bytes()).hexdigest(),
            }
        },
    )

    metrics = validate_release_root(release_root)
    assert metrics["replay_dataset_manifest_count"] == 1
