"""Verify derived linker labels with real Parquet source files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from s2and._sha256 import sha256_file
from scripts.production.data.prepare_linker_support import prepare_linker_support


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _row(
    query: str,
    component: str = "same",
    label: int = 1,
    *,
    dataset: str = "pubmed",
    view: str = "full",
    supervision: str = "labeled",
    base: str | None = None,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "query_group_id": f"{dataset}:{query}:{view}",
        "base_group_id": base or f"{dataset}:{query}",
        "query_signature_id": query,
        "query_view": view,
        "candidate_component_key": component,
        "candidate_cluster_id": component,
        "label": label,
        "retrieval_rank": 1,
        "supervision_type": supervision,
        "source_key": "s2and_eval" if dataset == "pubmed" else "hwang_eval",
        "source_kind": "public_test",
        "split": "source",
        "query_first_token": "a",
    }


def _fixture(root: Path) -> dict[str, Any]:
    source, gold, handoff = root / "source", root / "gold", root / "handoff"
    train_rows = [
        _row("positive", label=0),
        _row("negative", label=1),
        _row("self", component="self", label=1),
        _row("partial", component="mixed", label=0),
        _row("unknown_query", label=0),
        _row("bad_candidate"),
        _row("bad_candidate", component="unknown", label=0),
        _row("holdout", view="initial_only", base="old:holdout"),
        _row("error0", dataset="h_wang", view="initial_only", supervision="labeled"),
    ]
    train_rows[6]["retrieval_rank"] = 2
    cal_rows = [_row("cal_fit")]
    eval_rows = [_row("eval_test"), _row("holdout"), _row("unknown_query", label=0)]
    hwang_rows = [
        _row("cal_check", dataset="h_wang", supervision="positive_repeat_orcid"),
        _row("restored", dataset="h_wang", supervision="positive_repeat_orcid", label=0),
        _row("restored", component="other", dataset="h_wang", supervision="positive_repeat_orcid", label=0),
        *[_row(f"error{index}", dataset="h_wang", supervision="positive_repeat_orcid") for index in range(19)],
    ]
    hwang_rows[2]["retrieval_rank"] = 2
    assignments = [
        {**{key: row[key] for key in ("query_group_id", "base_group_id", "dataset", "source_key")}, "split": split}
        for row, split in (
            (cal_rows[0], "calibration_fit"),
            (eval_rows[0], "test"),
            (eval_rows[1], "calibration_fit"),
            (eval_rows[2], "calibration_fit"),
            (hwang_rows[0], "calibration_check"),
            (hwang_rows[1], "test"),
            *[(row, "test") for row in hwang_rows[3:]],
        )
    ]
    for index in range(18):
        query = f"leak{index}"
        for view, split, base in (
            ("full", "test", f"labeled:pubmed:{query}"),
            ("initial_only", "calibration_fit", f"pubmed:{query}"),
        ):
            row = _row(query, view=view, base=base)
            eval_rows.append(row)
            assignments.append(
                {
                    **{key: row[key] for key in ("query_group_id", "base_group_id", "dataset", "source_key")},
                    "split": split,
                }
            )
    files = {
        "train_path": "labels/train.parquet",
        "classic_gate_source_path": "labels/calibration.parquet",
        "s2and_eval_path": "labels/eval.parquet",
        "hwang_eval_path": "labels/hwang.parquet",
    }
    for (_key, relative), rows in zip(files.items(), (train_rows, cal_rows, eval_rows, hwang_rows), strict=True):
        frame = pd.DataFrame(rows)
        frame.insert(0, "row_index", pd.Series(range(len(frame)), dtype="int64"))
        frame["label"] = frame["label"].astype("int32")
        frame["retrieval_rank"] = frame["retrieval_rank"].astype("int64")
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    member_rows = [
        ("same", "member"),
        ("mixed", "member"),
        ("mixed", "unknown_member"),
        ("unknown", "unknown_member"),
        ("self", "self"),
    ]
    member_path = source / "components/pubmed.parquet"
    member_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"candidate_component_key": [row[0] for row in member_rows], "signature_id": [row[1] for row in member_rows]}
    ).to_parquet(member_path, index=False)
    split_path = source / "splits/seed13.csv"
    split_path.parent.mkdir(parents=True)
    pd.DataFrame(assignments).to_csv(split_path, index=False)
    pd.DataFrame({"base_group_id": ["pubmed:cal_fit"]}).to_csv(source / "splits/internal.csv", index=False)
    _json(source / "splits/summary.json", {"assignment_rows": len(assignments), "split_counts": {}})
    _json(
        source / "bundle.json",
        {
            "bundle_name": "test",
            "notes": "Test source.",
            "expected_metrics": {},
            "assets": {
                "featureless_rows": {"files": files},
                "candidate_members": {"datasets": {"pubmed": "components/pubmed.parquet"}},
                "splits": {"assignments_path": "splits/seed13.csv", "summary_path": "splits/summary.json"},
            },
            "models": {
                "classic": {
                    "classic_gate_internal_eval_base_groups_path": "splits/internal.csv",
                    "promoted_stratified_gate": {
                        "calibration_splits": ["calibration_fit", "calibration_check"],
                        "test_split": "test",
                    },
                    "stratified_eval_test_split": {
                        "assignments_path": "splits/seed13.csv",
                        "summary_path": "splits/summary.json",
                    },
                }
            },
        },
    )
    same = [
        "member",
        "positive",
        "partial",
        "cal_fit",
        "eval_test",
        "holdout",
        "bad_candidate",
        *[f"leak{index}" for index in range(18)],
    ]
    _json(
        gold / "pubmed/pubmed_clusters.json",
        {
            "same": {"signature_ids": same},
            "negative": {"signature_ids": ["negative"]},
            "self": {"signature_ids": ["self"]},
            "unknown": {"signature_ids": ["unknown_query", "unknown_member"]},
        },
    )
    _json(gold / "pubmed/unknown.json", ["unknown_query", "unknown_member"])
    _json(handoff / "orcid_error_query_signatures.json", [["h_wang", f"error{index}"] for index in range(19)])
    (handoff / "final_linker_review.jsonl").write_text("{}\n", encoding="utf-8")
    (handoff / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(handoff / name)}  {name}\n"
            for name in ("final_linker_review.jsonl", "orcid_error_query_signatures.json")
        ),
        encoding="utf-8",
    )
    units = root / "units.json"
    _json(units, [{"kind": "big_block_orcid_group", "dataset": "h_wang", "pairs": [["restored", "same"]]}])
    return {
        "source_root": source,
        "benchmark_labels_root": gold,
        "review_handoff": handoff,
        "review_units": units,
        "unknown_filename": "unknown.json",
        "benchmark_datasets": ["pubmed"],
        "output_root": root / "output",
        "report_path": root / "report.json",
    }


def test_current_targets_masks_orcid_and_split_identity_contract(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    original = {
        path.relative_to(args["source_root"]): sha256_file(path)
        for path in args["source_root"].rglob("*")
        if path.is_file()
    }
    report = prepare_linker_support(**args)
    output = args["output_root"]
    train = pd.read_parquet(output / "labels/train.parquet")
    assert train["query_signature_id"].tolist() == ["positive", "negative", "self", "partial", "error0"]
    assert train["label"].tolist() == [1, 0, 0, 1, 1]
    assert str(train["label"].dtype) == "int32"
    assert report["train_holdout_filter"]["rows_removed"] == 1
    assert report["orcid_excluded_query_signatures"] == 19
    assert len(report["legacy_cross_split_signature_identities"]) == 18
    hwang = pd.read_parquet(output / "labels/hwang.parquet")
    restored = hwang.loc[hwang["query_signature_id"].eq("restored")]
    assert restored["label"].tolist() == [1, 0]
    assert not hwang["query_signature_id"].str.startswith("error").any()
    active = pd.read_csv(Path(report["active_assignments"]["path"]))
    assert len(active) == 5
    assert not active["query_group_id"].str.contains("leak|unknown_query|error").any()
    assert report["active_split_loader_counts"]["calibration_fit"]["query_views"] == 2
    assert report["active_split_loader_counts"]["calibration_check"]["query_views"] == 1
    assert report["active_split_loader_counts"]["test"]["query_views"] == 2
    frozen = pd.read_csv(args["source_root"] / "splits/seed13.csv")
    pd.testing.assert_frame_equal(
        active, frozen.loc[frozen["query_group_id"].isin(active["query_group_id"])].reset_index(drop=True)
    )
    for relative, digest in original.items():
        assert sha256_file(args["source_root"] / relative) == digest
        if relative.parts[0] in {"components", "splits"}:
            assert sha256_file(output / relative) == digest


def test_reject_unverified_review_before_output(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    (args["review_handoff"] / "orcid_error_query_signatures.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Review digest mismatch"):
        prepare_linker_support(**args)
    assert not args["output_root"].exists()


def test_reject_all_unknown_train_and_preserve_source(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    path = args["source_root"] / "labels/train.parquet"
    rows = pd.read_parquet(path)
    rows.loc[rows["dataset"].eq("pubmed"), "query_signature_id"] = "unknown_query"
    rows.loc[rows["dataset"].eq("pubmed"), "query_group_id"] = "pubmed:unknown_query:full"
    rows = rows.loc[rows["dataset"].eq("pubmed")]
    rows.to_parquet(path, index=False)
    before = sha256_file(path)
    with pytest.raises(ValueError, match="no usable query views: train_path"):
        prepare_linker_support(**args)
    assert sha256_file(path) == before


def test_reject_nested_output_and_source_asset_escape(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    with pytest.raises(ValueError, match="must not overlap"):
        prepare_linker_support(**{**args, "output_root": args["source_root"] / "nested"})
    with pytest.raises(ValueError, match="report must remain outside"):
        prepare_linker_support(**{**args, "report_path": args["output_root"] / "bundle.json"})
    bundle_path = args["source_root"] / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["assets"]["candidate_members"]["datasets"]["pubmed"] = "../../outside.parquet"
    _json(bundle_path, bundle)
    with pytest.raises(ValueError, match="remain inside"):
        prepare_linker_support(**args)


def test_reject_train_holdout_identity_that_legacy_base_ids_hide(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    train_path = args["source_root"] / "labels/train.parquet"
    eval_path = args["source_root"] / "labels/hwang.parquet"
    for path, row in (
        (train_path, _row("unassigned_holdout", dataset="h_wang", view="initial_only", base="train:alias")),
        (eval_path, _row("unassigned_holdout", dataset="h_wang", base="eval:alias")),
    ):
        rows = pd.read_parquet(path)
        row["row_index"] = len(rows)
        extra = pd.DataFrame([row]).astype({"label": "int32", "row_index": "int64", "retrieval_rank": "int64"})
        pd.concat([rows, extra], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="retains a held signature identity"):
        prepare_linker_support(**args)


def _append_source_row(source: Path, relative: str, row: dict[str, Any]) -> None:
    path = source / relative
    rows = pd.read_parquet(path)
    row["row_index"] = len(rows)
    extra = pd.DataFrame([row]).astype({"label": "int32", "row_index": "int64", "retrieval_rank": "int64"})
    pd.concat([rows, extra], ignore_index=True).to_parquet(path, index=False)


def _add_assignment(source: Path, query: str, split: str) -> None:
    path = source / "splits/seed13.csv"
    assignments = pd.read_csv(path)
    row = _row(query)
    payload = {key: row[key] for key in ("query_group_id", "base_group_id", "dataset", "source_key")}
    pd.concat([assignments, pd.DataFrame([{**payload, "split": split}])], ignore_index=True).to_csv(path, index=False)


def test_external_holdout_excludes_train_and_calibration_views_and_preserves_test_context(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    source = args["source_root"]
    _append_source_row(source, "labels/train.parquet", _row("positive", view="initial_only"))
    _append_source_row(source, "labels/calibration.parquet", _row("other_cal"))
    _append_source_row(source, "labels/eval.parquet", _row("cal_fit", view="initial_only"))
    gold_path = args["benchmark_labels_root"] / "pubmed/pubmed_clusters.json"
    gold = json.loads(gold_path.read_text())
    gold["same"]["signature_ids"].append("other_cal")
    _json(gold_path, gold)
    _add_assignment(source, "other_cal", "calibration_fit")
    external = tmp_path / "external"
    identities = ["positive", "cal_fit", "eval_test", "member"]
    _json(external / "pubmed/pubmed_signatures.json", {signature: {} for signature in identities})
    _json(external / "pubmed/test_blocks.json", {"reserved": identities})
    component_hash = sha256_file(source / "components/pubmed.parquet")
    report = prepare_linker_support(**args, external_holdout_root=external)
    train = pd.read_parquet(args["output_root"] / "labels/train.parquet")
    assert "positive" not in set(train["query_signature_id"])
    assert report["external_holdout_filter"]["train_path"]["removed_rows"] == 2
    assert report["external_holdout_filter"]["train_path"]["removed_query_views"] == 2
    active = pd.read_csv(Path(report["active_assignments"]["path"]))
    assert not active["query_group_id"].eq("pubmed:cal_fit:full").any()
    for path in ("labels/calibration.parquet", "labels/eval.parquet"):
        rows = pd.read_parquet(args["output_root"] / path)
        assert "cal_fit" not in set(rows["query_signature_id"])
    assert "pubmed:eval_test:full" in set(active.loc[active["split"].eq("test"), "query_group_id"])
    assert "eval_test" in set(pd.read_parquet(args["output_root"] / "labels/eval.parquet")["query_signature_id"])
    assert sha256_file(args["output_root"] / "components/pubmed.parquet") == component_hash
    assert report["external_holdout_query_signatures"] == 4
    assert report["active_split_loader_counts"]["calibration_fit"]["query_views"] == 2


def test_reject_external_holdout_membership_mismatch_before_output(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    external = tmp_path / "external"
    _json(external / "pubmed/pubmed_signatures.json", {"positive": {}})
    _json(external / "pubmed/test_blocks.json", {"reserved": ["other"]})
    with pytest.raises(ValueError, match="signature/block membership mismatch"):
        prepare_linker_support(**args, external_holdout_root=external)
    assert not args["output_root"].exists()


def test_test_annotations_distinguish_endpoint_eps_paper_and_unaudited_datasets(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    source = args["source_root"]
    for query in ("eps_only", "paper_only", "unexposed"):
        _append_source_row(source, "labels/eval.parquet", _row(query))
        _add_assignment(source, query, "test")
    gold_path = args["benchmark_labels_root"] / "pubmed/pubmed_clusters.json"
    gold = json.loads(gold_path.read_text())
    gold["same"]["signature_ids"].extend(["eps_only", "paper_only", "unexposed"])
    _json(gold_path, gold)
    audit_path = tmp_path / "exposure.csv"
    pd.DataFrame(
        [
            {
                "dataset": "pubmed",
                "query_signature_id": "eval_test",
                "pairwise_exposure": "actual_train_pair_endpoint",
                "same_signature": True,
                "same_paper": True,
            },
            {
                "dataset": "pubmed",
                "query_signature_id": "eps_only",
                "pairwise_exposure": "validation_population",
                "same_signature": True,
                "same_paper": True,
            },
            {
                "dataset": "pubmed",
                "query_signature_id": "paper_only",
                "pairwise_exposure": "actual_train_pair_endpoint",
                "same_signature": False,
                "same_paper": True,
            },
        ]
    ).to_csv(audit_path, index=False)
    report = prepare_linker_support(**args, pairwise_exposure_path=audit_path)
    active = pd.read_csv(Path(report["active_assignments"]["path"]))
    annotations = pd.read_csv(report["test_exposure_annotations"]["output"]["path"])
    assert set(annotations["query_group_id"]) == set(active.loc[active["split"].eq("test"), "query_group_id"])
    by_query = annotations.set_index("query_signature_id")
    assert bool(by_query.loc["eval_test", "query_pairwise_endpoint_exposed"])
    assert not bool(by_query.loc["eval_test", "query_eps_selection_exposed"])
    assert bool(by_query.loc["eps_only", "query_eps_selection_exposed"])
    assert not bool(by_query.loc["eps_only", "query_pairwise_endpoint_exposed"])
    assert bool(by_query.loc["paper_only", "paper_pairwise_or_eps_exposed"])
    assert not bool(by_query.loc["paper_only", "query_pairwise_endpoint_exposed"])
    assert bool(by_query.loc["unexposed", "query_and_paper_unexposed_in_dataset_audit"])
    assert pd.isna(by_query.loc["restored", "query_and_paper_unexposed_in_dataset_audit"])
    assert not bool(by_query.loc["restored", "upstream_dataset_audit_available"])
    assert annotations["evaluation_population"].eq("inherited_regression").all()


def test_calibration_source_only_test_assignment_preserves_rows_as_regression(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    source = args["source_root"]
    _append_source_row(source, "labels/calibration.parquet", _row("cal_only_test"))
    _add_assignment(source, "cal_only_test", "test")
    gold_path = args["benchmark_labels_root"] / "pubmed/pubmed_clusters.json"
    gold = json.loads(gold_path.read_text())
    gold["same"]["signature_ids"].append("cal_only_test")
    _json(gold_path, gold)
    original_assignments_hash = sha256_file(source / "splits/seed13.csv")
    component_hash = sha256_file(source / "components/pubmed.parquet")
    report = prepare_linker_support(**args)
    active = pd.read_csv(report["active_assignments"]["path"])
    assert "pubmed:cal_only_test:full" not in set(active["query_group_id"])
    assert report["active_split_loader_counts"]["test"]["query_views"] == 2
    assert report["unscorable_test_assignments"]["count"] == 1
    regression = pd.read_csv(report["unscorable_test_assignments"]["output"]["path"])
    assert regression["query_group_id"].tolist() == ["pubmed:cal_only_test:full"]
    assert regression["split"].tolist() == ["test"]
    assert regression["reason"].tolist() == ["unavailable_in_public_evaluation_sources"]
    calibration = pd.read_parquet(args["output_root"] / "labels/calibration.parquet")
    preserved = calibration.loc[calibration["query_signature_id"].eq("cal_only_test")]
    assert preserved["label"].tolist() == [1]
    assert sha256_file(args["output_root"] / "splits/seed13.csv") == original_assignments_hash
    assert sha256_file(args["output_root"] / "components/pubmed.parquet") == component_hash
