from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import scripts.giant_block_cluster_retrieval_task as task


def _signature(first: str, paper_id: str, block: str = "block-1") -> SimpleNamespace:
    return SimpleNamespace(
        author_info_first=first,
        author_info_first_normalized_without_apostrophe=first,
        author_info_block=block,
        paper_id=paper_id,
    )


def test_classify_subblocks_uses_first_signature_semantics():
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("A", "p1"),
            "s2": _signature("Alice", "p2"),
            "s3": _signature("Bob", "p3"),
        }
    )
    subblocks = {
        "mixed": ["s1", "s2"],
        "multi": ["s2", "s3"],
        "single": ["s1"],
    }

    multi_letter, single_letter = task._classify_subblocks(subblocks, dataset)

    assert list(single_letter) == ["mixed", "single"]
    assert list(multi_letter) == ["multi"]


def test_load_dataset_auto_uses_clusterer_name_count_contract(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "signatures.json").write_text(json.dumps({"s1": {"paper_id": "p1"}}), encoding="utf-8")
    (data_dir / "papers.json").write_text(json.dumps({"p1": {"authors": []}}), encoding="utf-8")
    (data_dir / "specter.pickle").write_bytes(b"")
    (data_dir / "cluster_seeds.json").write_text(json.dumps({}), encoding="utf-8")
    (data_dir / "altered_cluster_signatures.txt").write_text("", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_read_json(path):
        if path.name == "signatures.json":
            return {"s1": {"paper_id": "p1", "author_info": {"block": "block-1"}}}
        if path.name == "papers.json":
            return {"p1": {"authors": []}}
        if path.name == "cluster_seeds.json":
            return {}
        raise AssertionError(f"Unexpected JSON path={path}")

    monkeypatch.setattr(task, "_read_json", fake_read_json)
    monkeypatch.setattr(task, "_read_text_lines", lambda path: [])
    monkeypatch.setattr(task, "_load_specter_subset", lambda path, payloads: {"p1": [0.0]})
    monkeypatch.setattr(task, "_resolve_target_block", lambda signatures, meta, block_key: "block-1")
    monkeypatch.setattr(task, "_select_block_signature_ids", lambda signatures, block_key: ["s1"])
    monkeypatch.setattr(task, "_filter_papers", lambda papers, selected: {"p1": {"authors": []}})
    monkeypatch.setattr(task, "_filter_cluster_seeds", lambda seeds, ids: {})
    monkeypatch.setattr(task, "_filter_altered_signatures", lambda altered, ids: [])

    def fake_anddata(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(signatures={}, papers={})

    monkeypatch.setattr(task, "ANDData", fake_anddata)

    clusterer = SimpleNamespace(
        featurizer_info=SimpleNamespace(features_to_use=["coauthor_similarity"]),
        nameless_featurizer_info=SimpleNamespace(features_to_use=["name_counts"]),
    )
    dataset, load_info = task.load_dataset(data_dir, block_key=None, n_jobs=2, clusterer=clusterer)

    assert captured["load_name_counts"] is True
    assert load_info["target_block"] == "block-1"
    assert dataset.signatures == {}


def test_run_task_persists_artifacts_and_clusters_only_multi_letter_subblocks(tmp_path, monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("A", "p1"),
            "s2": _signature("Alice", "p2"),
            "s3": _signature("Bob", "p3"),
        },
        papers={},
        cluster_seeds_require={},
        cluster_seeds_disallow=set(),
        altered_cluster_signatures=[],
        max_seed_cluster_id=0,
        random_seed=7,
    )

    load_info = {
        "target_block": "block-1",
        "selected_signature_ids": ["s1", "s2", "s3"],
        "selected_paper_ids": ["p1", "p2", "p3"],
        "source_meta": {"block_key": "block-1"},
    }

    subblocks = {
        "multi": ["s2", "s3"],
        "single": ["s1"],
    }
    telemetry = {"maximum_size": 15000, "input_signature_count": 3, "specter_invocation_count": 0}
    predict_calls: list[tuple[str, list[str]]] = []

    def fake_load_dataset(data_dir, *, block_key, n_jobs, clusterer=None, load_name_counts="auto"):
        assert data_dir.name == "data-dir"
        assert block_key is None
        assert n_jobs == 20
        assert clusterer is not None
        assert load_name_counts == "auto"
        return dataset, load_info

    def fake_load_clusterer(model_path, *, n_jobs):
        assert model_path.name == "production_model_v1.2.pickle"
        assert n_jobs == 20
        return SimpleNamespace(
            use_cache=True,
            n_jobs=0,
            predict_helper=lambda block_dict, dataset_arg, **kwargs: _fake_predict_helper(
                block_dict, dataset_arg, kwargs, predict_calls
            ),
        )

    def fake_make_subblocks_with_telemetry(signature_ids, dataset_arg, maximum_size):
        assert signature_ids == ["s1", "s2", "s3"]
        assert dataset_arg is dataset
        assert maximum_size == 15000
        return subblocks, telemetry

    def _fake_predict_helper(block_dict, dataset_arg, kwargs, calls):
        assert dataset_arg is dataset
        assert kwargs["dists"] is None
        assert kwargs["cluster_model_params"] is None
        assert kwargs["partial_supervision"] == {}
        assert kwargs["use_s2_clusters"] is False
        assert kwargs["incremental_dont_use_cluster_seeds"] is False
        assert kwargs["total_ram_bytes"] == 123
        subblock_key, signature_ids = next(iter(block_dict.items()))
        calls.append((subblock_key, list(signature_ids)))
        if subblock_key != "multi":
            raise AssertionError(f"predict_helper should not be called for {subblock_key!r}")
        return {"cluster-a": ["s2"], "cluster-b": ["s3"]}, None

    monkeypatch.setattr(task, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(task, "load_clusterer", fake_load_clusterer)
    monkeypatch.setattr(task, "make_subblocks_with_telemetry", fake_make_subblocks_with_telemetry)

    output_dir = tmp_path / "artifacts"
    summary = task.run_task(
        data_dir=tmp_path / "data-dir",
        output_dir=output_dir,
        model_path=tmp_path / "production_model_v1.2.pickle",
        maximum_size=15000,
        n_jobs=20,
        total_ram_bytes=123,
    )

    assert predict_calls == [("multi", ["s2", "s3"])]
    assert summary["signature_count"] == 3
    assert summary["multi_letter_subblock_count"] == 1
    assert summary["single_letter_subblock_count"] == 1
    assert summary["clustered_subblock_count"] == 1
    assert summary["predicted_cluster_total"] == 2
    assert summary["total_ram_bytes"] == 123

    manifest = json.loads((output_dir / "subblock_manifest.json").read_text(encoding="utf-8"))
    subblock_telemetry = json.loads((output_dir / "subblock_telemetry.json").read_text(encoding="utf-8"))
    predicted_clusters = json.loads((output_dir / "predicted_clusters.json").read_text(encoding="utf-8"))
    signature_to_cluster_id = json.loads((output_dir / "signature_to_cluster_id.json").read_text(encoding="utf-8"))
    per_subblock_timings = json.loads((output_dir / "multi_letter_subblock_timings.json").read_text(encoding="utf-8"))
    run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))

    assert manifest["target_block"] == "block-1"
    assert manifest["total_ram_bytes"] == 123
    assert manifest["subblocks"] == {"multi": ["s2", "s3"], "single": ["s1"]}
    assert manifest["multi_letter_subblock_keys"] == ["multi"]
    assert manifest["single_letter_subblock_keys"] == ["single"]
    assert subblock_telemetry["make_subblocks_telemetry"] == telemetry
    assert predicted_clusters == {"multi": {"cluster-a": ["s2"], "cluster-b": ["s3"]}}
    assert signature_to_cluster_id == {"s2": "cluster-a", "s3": "cluster-b"}
    assert per_subblock_timings["rows"][0]["subblock_key"] == "multi"
    assert per_subblock_timings["rows"][0]["cluster_count"] == 2
    assert run_summary["artifact_paths"]["predicted_clusters"].endswith("predicted_clusters.json")
    assert run_summary["artifact_paths"]["partial_multi_letter_subblocks"].endswith("partial_multi_letter_subblocks")


def test_run_task_resumes_completed_partial_subblocks(tmp_path, monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("Alice", "p1"),
            "s2": _signature("Bob", "p2"),
            "s3": _signature("Carol", "p3"),
        },
        papers={},
        cluster_seeds_require={},
        cluster_seeds_disallow=set(),
        altered_cluster_signatures=[],
        max_seed_cluster_id=0,
    )

    load_info = {
        "target_block": "block-1",
        "selected_signature_ids": ["s1", "s2", "s3"],
        "selected_paper_ids": ["p1", "p2", "p3"],
        "source_meta": {"block_key": "block-1"},
    }
    subblocks = {
        "multi-a": ["s1"],
        "multi-b": ["s2", "s3"],
    }
    telemetry = {"maximum_size": 15000}
    calls: list[str] = []

    def fake_load_dataset(data_dir, *, block_key, n_jobs, clusterer=None, load_name_counts="auto"):
        del data_dir, block_key, n_jobs
        assert clusterer is not None
        assert load_name_counts == "auto"
        return dataset, load_info

    def fake_load_clusterer(model_path, *, n_jobs):
        del model_path, n_jobs
        return SimpleNamespace(
            use_cache=False,
            n_jobs=20,
            predict_helper=lambda block_dict, dataset_arg, **kwargs: _resume_predict_helper(
                block_dict, dataset_arg, kwargs, calls
            ),
        )

    def fake_make_subblocks_with_telemetry(signature_ids, dataset_arg, maximum_size):
        assert signature_ids == ["s1", "s2", "s3"]
        assert dataset_arg is dataset
        assert maximum_size == 15000
        return subblocks, telemetry

    def _resume_predict_helper(block_dict, dataset_arg, kwargs, observed_calls):
        assert dataset_arg is dataset
        assert kwargs["total_ram_bytes"] == 123
        subblock_key, signature_ids = next(iter(block_dict.items()))
        observed_calls.append(subblock_key)
        return {f"{subblock_key}-cluster": list(signature_ids)}, None

    monkeypatch.setattr(task, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(task, "load_clusterer", fake_load_clusterer)
    monkeypatch.setattr(task, "make_subblocks_with_telemetry", fake_make_subblocks_with_telemetry)

    output_dir = tmp_path / "artifacts"
    partial_dir = output_dir / "partial_multi_letter_subblocks"
    partial_dir.mkdir(parents=True)
    resume_identity = task._build_partial_resume_identity(  # noqa: SLF001
        data_dir=tmp_path / "data-dir",
        model_path=tmp_path / "production_model_v1.2.pickle",
        target_block="block-1",
        maximum_size=15000,
        total_ram_bytes=123,
    )
    (partial_dir / "subblock_000.json").write_text(
        json.dumps(
            {
                "subblock_key": "multi-a",
                "signature_ids": ["s1"],
                "resume_identity": resume_identity,
                "clusters": {"seeded-cluster": ["s1"]},
                "timing": {
                    "subblock_key": "multi-a",
                    "signature_count": 1,
                    "cluster_count": 1,
                    "cluster_sizes": [1],
                    "predict_seconds": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = task.run_task(
        data_dir=tmp_path / "data-dir",
        output_dir=output_dir,
        model_path=tmp_path / "production_model_v1.2.pickle",
        maximum_size=15000,
        n_jobs=20,
        total_ram_bytes=123,
    )

    predicted_clusters = json.loads((output_dir / "predicted_clusters.json").read_text(encoding="utf-8"))
    signature_to_cluster_id = json.loads((output_dir / "signature_to_cluster_id.json").read_text(encoding="utf-8"))
    per_subblock_timings = json.loads((output_dir / "multi_letter_subblock_timings.json").read_text(encoding="utf-8"))

    assert calls == ["multi-b"]
    assert summary["clustered_subblock_count"] == 2
    assert predicted_clusters["multi-a"] == {"seeded-cluster": ["s1"]}
    assert predicted_clusters["multi-b"] == {"multi-b-cluster": ["s2", "s3"]}
    assert signature_to_cluster_id == {
        "s1": "seeded-cluster",
        "s2": "multi-b-cluster",
        "s3": "multi-b-cluster",
    }
    assert [row["subblock_key"] for row in per_subblock_timings["rows"]] == ["multi-a", "multi-b"]


def test_run_task_rejects_partial_subblocks_without_matching_resume_identity(tmp_path, monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("Alice", "p1"),
            "s2": _signature("Bob", "p2"),
        },
        papers={},
        cluster_seeds_require={},
        cluster_seeds_disallow=set(),
        altered_cluster_signatures=[],
        max_seed_cluster_id=0,
    )
    load_info = {
        "target_block": "block-1",
        "selected_signature_ids": ["s1", "s2"],
        "selected_paper_ids": ["p1", "p2"],
        "source_meta": {"block_key": "block-1"},
    }
    subblocks = {"multi-a": ["s1", "s2"]}

    def fake_load_dataset(data_dir, *, block_key, n_jobs, clusterer=None, load_name_counts="auto"):
        del data_dir, block_key, n_jobs, load_name_counts
        assert clusterer is not None
        return dataset, load_info

    def fake_load_clusterer(model_path, *, n_jobs):
        del model_path, n_jobs

        def fail_predict_helper(*_args, **_kwargs):
            raise AssertionError("should not be called")

        return SimpleNamespace(
            use_cache=False,
            n_jobs=20,
            predict_helper=fail_predict_helper,
        )

    monkeypatch.setattr(task, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(task, "load_clusterer", fake_load_clusterer)
    monkeypatch.setattr(task, "make_subblocks_with_telemetry", lambda *_args, **_kwargs: (subblocks, {}))

    output_dir = tmp_path / "artifacts"
    partial_dir = output_dir / "partial_multi_letter_subblocks"
    partial_dir.mkdir(parents=True)
    (partial_dir / "subblock_000.json").write_text(
        json.dumps(
            {
                "subblock_key": "multi-a",
                "signature_ids": ["s1", "s2"],
                "resume_identity": {
                    "data_dir": "wrong",
                    "model_path": "wrong",
                    "target_block": "block-1",
                    "maximum_size": 15000,
                    "total_ram_bytes": 123,
                },
                "clusters": {"seeded-cluster": ["s1", "s2"]},
                "timing": {
                    "subblock_key": "multi-a",
                    "signature_count": 2,
                    "cluster_count": 1,
                    "cluster_sizes": [2],
                    "predict_seconds": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not match the current run identity"):
        task.run_task(
            data_dir=tmp_path / "data-dir",
            output_dir=output_dir,
            model_path=tmp_path / "production_model_v1.2.pickle",
            maximum_size=15000,
            n_jobs=20,
            total_ram_bytes=123,
        )
