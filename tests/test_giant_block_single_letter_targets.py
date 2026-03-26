from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import scripts.giant_block_single_letter_targets as targets


def _signature(
    first: str,
    *,
    normalized_first: str | None = None,
    middle: str = "",
    normalized_middle: str | None = None,
    block: str = "block-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        author_info_first=first,
        author_info_first_normalized_without_apostrophe=normalized_first if normalized_first is not None else first,
        author_info_middle=middle,
        author_info_middle_normalized_without_apostrophe=(
            normalized_middle if normalized_middle is not None else middle
        ),
        author_info_block=block,
    )


def _raw_signature(first: str, *, block: str = "block-1", orcid: str | None = None) -> dict:
    source = "ORCID" if orcid is not None else "Extracted"
    source_ids = [orcid] if orcid is not None else [first]
    return {
        "signature_id": first,
        "paper_id": 1,
        "author_info": {
            "block": block,
            "first": first,
            "middle": None,
            "last": "Wang",
            "suffix": None,
            "email": None,
            "affiliations": [],
            "position": 0,
            "source_id_source": source,
            "source_ids": source_ids,
        },
    }


def test_build_query_metadata_filters_repeated_orcid_single_letter_and_seed_overlap():
    dataset = SimpleNamespace(
        signatures={
            "q1": _signature("H.", normalized_first="h"),
            "q2": _signature("H.", normalized_first="h", middle="M", normalized_middle="m"),
            "q3": _signature("Han", normalized_first="han"),
            "q4": _signature("H.", normalized_first="h"),
        }
    )
    raw_signatures = {
        "q1": _raw_signature("H.", orcid="0000-0001-0000-0001"),
        "q2": _raw_signature("H.", orcid="0000-0002-0000-0002"),
        "q3": _raw_signature("Han", orcid="0000-0001-0000-0001"),
        "q4": _raw_signature("H.", orcid="0000-0002-0000-0002"),
    }
    subblock_manifest = {
        "subblocks": {
            "h": ["q1", "q4"],
            "h|specter=0": ["q2"],
            "han": ["q3"],
        }
    }

    query_ids, metadata_by_query, payload = targets._build_query_metadata(
        raw_signatures,
        dataset,
        target_block="block-1",
        subblock_manifest=subblock_manifest,
        seed_signature_ids={"q4"},
        limit_queries=None,
        random_seed=0,
    )

    assert query_ids == ["q1", "q2"]
    assert payload["repeated_orcid_group_count"] == 2
    assert payload["eligible_query_count_before_seed_overlap_filter"] == 3
    assert payload["seed_overlap_filtered_query_count"] == 1
    assert metadata_by_query["q1"]["query_subblock_type"] == "initial_only"
    assert metadata_by_query["q2"]["query_subblock_type"] == "specter"
    assert metadata_by_query["q2"]["query_in_specter_subblock"] is True
    assert metadata_by_query["q2"]["has_middle_name"] is True


def test_run_task_persists_dual_targets_and_informative_report(tmp_path, monkeypatch):
    data_dir = tmp_path / "data-dir"
    step2_dir = tmp_path / "step2"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    step2_dir.mkdir()

    raw_signatures = {
        "q1": _raw_signature("H.", orcid="0000-0001-0000-0001"),
        "q2": _raw_signature("H.", orcid="0000-0002-0000-0002"),
        "s10": _raw_signature("Han", orcid="0000-0001-0000-0001"),
        "s11": _raw_signature("Han"),
        "s20": _raw_signature("Hao", orcid="0000-0002-0000-0002"),
    }
    (data_dir / "signatures.json").write_text(json.dumps(raw_signatures), encoding="utf-8")

    subblock_manifest = {
        "data_dir": str(data_dir.resolve()),
        "target_block": "block-1",
        "subblocks": {
            "h": ["q1"],
            "h|specter=0": ["q2"],
            "han": ["s10", "s11"],
            "hao": ["s20"],
        },
    }
    signature_to_cluster_id = {"s10": "seed-a", "s11": "seed-a", "s20": "seed-b"}
    (step2_dir / "subblock_manifest.json").write_text(json.dumps(subblock_manifest), encoding="utf-8")
    (step2_dir / "signature_to_cluster_id.json").write_text(json.dumps(signature_to_cluster_id), encoding="utf-8")

    dataset = SimpleNamespace(
        signatures={
            "q1": _signature("H.", normalized_first="h"),
            "q2": _signature("H.", normalized_first="h", middle="M", normalized_middle="m"),
            "s10": _signature("Han", normalized_first="han"),
            "s11": _signature("Han", normalized_first="han"),
            "s20": _signature("Hao", normalized_first="hao"),
        },
        cluster_seeds_require={},
        cluster_seeds_disallow=set(),
        altered_cluster_signatures=[],
        max_seed_cluster_id=0,
    )
    load_info = {"target_block": "block-1", "selected_signature_ids": list(dataset.signatures), "source_meta": {}}

    def fake_load_dataset(data_dir_arg, *, block_key, n_jobs, clusterer=None, load_name_counts="auto"):
        assert data_dir_arg == data_dir
        assert block_key is None
        assert n_jobs == 20
        assert clusterer is not None
        assert load_name_counts == "auto"
        return dataset, load_info

    class FakeClusterer:
        def predict_incremental(self, block_signatures, dataset_arg, batching_threshold=None, total_ram_bytes=None):
            assert dataset_arg is dataset
            assert total_ram_bytes == 456
            if list(block_signatures) == ["q1", "q2"]:
                assert batching_threshold == 2
                return {
                    "clusters": {
                        "seed-a": ["s10", "s11", "q1"],
                        "seed-b": ["s20"],
                        "3": ["q2"],
                    },
                    "phase_b_mode": "exact",
                    "phase_b_budget_bytes": 100,
                    "phase_b_required_bytes": 8,
                    "phase_a_accumulator_overflow_early_stop": False,
                    "phase_a_adaptive_halvings_max": 0,
                }
            if list(block_signatures) == ["q1"]:
                assert batching_threshold is None
                return {
                    "clusters": {
                        "seed-a": ["s10", "s11", "q1"],
                        "seed-b": ["s20"],
                    },
                    "phase_b_mode": "exact",
                    "phase_b_budget_bytes": 100,
                    "phase_b_required_bytes": 0,
                    "phase_a_accumulator_overflow_early_stop": False,
                    "phase_a_adaptive_halvings_max": 0,
                }
            if list(block_signatures) == ["q2"]:
                assert batching_threshold is None
                return {
                    "clusters": {
                        "seed-a": ["s10", "s11"],
                        "seed-b": ["s20", "q2"],
                    },
                    "phase_b_mode": "exact",
                    "phase_b_budget_bytes": 100,
                    "phase_b_required_bytes": 0,
                    "phase_a_accumulator_overflow_early_stop": False,
                    "phase_a_adaptive_halvings_max": 0,
                }
            raise AssertionError(f"Unexpected block_signatures={block_signatures!r}")

    monkeypatch.setattr(targets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(targets, "load_clusterer", lambda model_path, *, n_jobs: FakeClusterer())
    monkeypatch.setattr(targets, "_sync_rust_cluster_seeds", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(targets, "_bump_cluster_seeds_version", lambda *_args, **_kwargs: 1)

    summary = targets.run_task(
        data_dir=data_dir,
        step2_dir=step2_dir,
        output_dir=output_dir,
        model_path=tmp_path / "production_model_v1.2.pickle",
        n_jobs=20,
        total_ram_bytes=456,
        joint_batching_threshold=2,
    )

    query_set = json.loads((output_dir / "query_set.json").read_text(encoding="utf-8"))
    joint_targets = json.loads((output_dir / "joint_targets.json").read_text(encoding="utf-8"))
    per_query_targets = json.loads((output_dir / "per_query_targets.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "target_disagreement_report.json").read_text(encoding="utf-8"))
    progress_rows = (output_dir / "per_query_progress.jsonl").read_text(encoding="utf-8").splitlines()

    assert summary["query_count"] == 2
    assert summary["joint_primary_reference_valid"] is True
    assert query_set["query_ids"] == ["q1", "q2"]
    assert joint_targets["targets"]["q1"]["target_cluster_id"] == "seed-a"
    assert joint_targets["targets"]["q2"]["target_cluster_id"] == "3"
    assert per_query_targets["targets"]["q2"]["target_cluster_id"] == "seed-b"
    assert report["disagreement_count"] == 1
    assert report["transition_counts"]["joint_new_cluster_per_query_existing_seed"] == 1
    assert report["slices"]["query_in_specter_subblock"]["True"]["disagreement_count"] == 1
    assert report["disagreement_examples"][0]["query_id"] == "q2"
    assert len(progress_rows) == 2


def test_run_per_query_targets_resumes_existing_progress(tmp_path):
    progress_path = tmp_path / "per_query_progress.jsonl"
    progress_identity_path = tmp_path / "per_query_progress.meta.json"
    progress_identity = {
        "data_dir": "data-dir",
        "step2_dir": "step2-dir",
        "model_path": "model.pkl",
        "target_block": "block-1",
        "total_ram_bytes": 456,
        "query_ids": ["q1", "q2"],
    }
    progress_identity_path.write_text(json.dumps(progress_identity), encoding="utf-8")
    existing_row = {
        "query_id": "q1",
        "normalized_orcid": "O1",
        "orcid_group_size": 2,
        "orcid_group_size_bucket": "2",
        "first_name": "H.",
        "first_name_normalized": "h",
        "middle_name": "",
        "middle_name_normalized": "",
        "has_middle_name": False,
        "query_subblock_key": "h",
        "query_subblock_type": "initial_only",
        "query_in_specter_subblock": False,
        "target_cluster_id": "seed-a",
        "target_cluster_size": 2,
        "target_is_existing_seed_cluster": True,
        "target_seed_member_count": 1,
        "target_query_member_count": 1,
        "target_query_member_ids": ["q1"],
        "target_other_query_ids": [],
        "per_query_index": 1,
        "per_query_elapsed_seconds": 1.5,
        "phase_b_mode": "exact",
        "phase_b_budget_bytes": 100,
        "phase_b_required_bytes": 0,
        "phase_a_accumulator_overflow_early_stop": False,
        "phase_a_adaptive_halvings_max": 0,
    }
    progress_path.write_text(json.dumps(existing_row) + "\n", encoding="utf-8")

    metadata_by_query = {
        "q1": {
            "query_id": "q1",
            "normalized_orcid": "O1",
            "orcid_group_size": 2,
            "orcid_group_size_bucket": "2",
            "first_name": "H.",
            "first_name_normalized": "h",
            "middle_name": "",
            "middle_name_normalized": "",
            "has_middle_name": False,
            "query_subblock_key": "h",
            "query_subblock_type": "initial_only",
            "query_in_specter_subblock": False,
        },
        "q2": {
            "query_id": "q2",
            "normalized_orcid": "O2",
            "orcid_group_size": 2,
            "orcid_group_size_bucket": "2",
            "first_name": "H.",
            "first_name_normalized": "h",
            "middle_name": "",
            "middle_name_normalized": "",
            "has_middle_name": False,
            "query_subblock_key": "h",
            "query_subblock_type": "initial_only",
            "query_in_specter_subblock": False,
        },
    }

    class FakeClusterer:
        def __init__(self):
            self.calls = []

        def predict_incremental(self, block_signatures, dataset_arg, batching_threshold=None, total_ram_bytes=None):
            self.calls.append(list(block_signatures))
            assert batching_threshold is None
            assert total_ram_bytes == 456
            return {
                "clusters": {
                    "seed-a": ["s10"],
                    "seed-b": ["s20", "q2"],
                },
                "phase_b_mode": "exact",
                "phase_b_budget_bytes": 100,
                "phase_b_required_bytes": 0,
                "phase_a_accumulator_overflow_early_stop": False,
                "phase_a_adaptive_halvings_max": 0,
            }

    clusterer = FakeClusterer()
    targets_by_query, summary = targets._run_per_query_targets(
        clusterer=clusterer,
        dataset=SimpleNamespace(),
        query_ids=["q1", "q2"],
        metadata_by_query=metadata_by_query,
        total_ram_bytes=456,
        seed_signature_ids={"s10", "s20"},
        existing_seed_cluster_ids={"seed-a", "seed-b"},
        progress_path=progress_path,
        progress_identity_path=progress_identity_path,
        progress_identity=progress_identity,
    )

    progress_rows = progress_path.read_text(encoding="utf-8").splitlines()
    assert clusterer.calls == [["q2"]]
    assert sorted(targets_by_query) == ["q1", "q2"]
    assert summary["query_count"] == 2
    assert summary["resumed_query_count"] == 1
    assert len(progress_rows) == 2


def test_run_per_query_targets_rejects_progress_identity_mismatch(tmp_path):
    progress_path = tmp_path / "per_query_progress.jsonl"
    progress_path.write_text(json.dumps({"query_id": "q1"}) + "\n", encoding="utf-8")
    progress_identity_path = tmp_path / "per_query_progress.meta.json"
    progress_identity_path.write_text(
        json.dumps(
            {
                "data_dir": "other-data-dir",
                "step2_dir": "step2-dir",
                "model_path": "model.pkl",
                "target_block": "block-1",
                "total_ram_bytes": 456,
                "query_ids": ["q1"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not match the current run identity"):
        targets._run_per_query_targets(
            clusterer=SimpleNamespace(),
            dataset=SimpleNamespace(),
            query_ids=["q1"],
            metadata_by_query={"q1": {"query_id": "q1"}},
            total_ram_bytes=456,
            seed_signature_ids=set(),
            existing_seed_cluster_ids=set(),
            progress_path=progress_path,
            progress_identity_path=progress_identity_path,
            progress_identity={
                "data_dir": "data-dir",
                "step2_dir": "step2-dir",
                "model_path": "model.pkl",
                "target_block": "block-1",
                "total_ram_bytes": 456,
                "query_ids": ["q1"],
            },
        )


def test_run_task_rejects_step2_manifest_for_different_block(tmp_path, monkeypatch):
    data_dir = tmp_path / "data-dir"
    step2_dir = tmp_path / "step2"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    step2_dir.mkdir()

    (data_dir / "signatures.json").write_text(json.dumps({"q1": _raw_signature("H.", orcid="0000-0001-0000-0001")}))
    (step2_dir / "subblock_manifest.json").write_text(
        json.dumps(
            {
                "data_dir": str(data_dir.resolve()),
                "target_block": "other-block",
                "subblocks": {"h": ["q1"]},
            }
        ),
        encoding="utf-8",
    )
    (step2_dir / "signature_to_cluster_id.json").write_text(json.dumps({"q1": "seed-a"}), encoding="utf-8")

    dataset = SimpleNamespace(
        signatures={"q1": _signature("H.", normalized_first="h")},
        cluster_seeds_require={},
        cluster_seeds_disallow=set(),
        altered_cluster_signatures=[],
        max_seed_cluster_id=0,
    )
    load_info = {"target_block": "block-1", "selected_signature_ids": ["q1"], "source_meta": {}}

    monkeypatch.setattr(targets, "load_dataset", lambda *args, **kwargs: (dataset, load_info))
    monkeypatch.setattr(targets, "load_clusterer", lambda model_path, *, n_jobs: SimpleNamespace())

    with pytest.raises(RuntimeError, match="does not match current block"):
        targets.run_task(
            data_dir=data_dir,
            step2_dir=step2_dir,
            output_dir=output_dir,
            model_path=tmp_path / "production_model_v1.2.pickle",
            n_jobs=20,
            total_ram_bytes=456,
            joint_batching_threshold=2,
        )
