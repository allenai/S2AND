"""Complete holdout identity authority across release training stages."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from s2and.incremental_linking_training import classic
from scripts.production.model import train_linker_and_finalize as release
from tests.test_real_tiny_trainers import _write_classic_tiny_bundle


def test_source_authority_projects_complete_identity_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    keys = (
        "classic_gate_source_path",
        "s2and_eval_path",
        "hwang_eval_path",
        "s_park_eval_path",
        "s_lee_eval_path",
        "extra_eval_paths.example",
    )
    files = {}
    for index, key in enumerate(keys):
        path = tmp_path / f"{index}.parquet"
        pd.DataFrame(
            {
                "query_group_id": [f"test-{index}"],
                "base_group_id": [f"base-{index}"],
                "label": [1],
                "secret_feature": [42],
            }
        ).to_parquet(path, index=False)
        files[key] = path.name
    bundle = classic.OfficialBundle(
        tmp_path,
        "source",
        {"featureless_rows": {"files": files}},
        {"classic": {"s2and_eval_path": "nonexistent-calibration-subset.parquet"}},
        {},
    )
    reads = []
    read_parquet = pd.read_parquet

    def projected_read(path: Any, **kwargs: Any) -> pd.DataFrame:
        assert kwargs["columns"] == ["query_group_id", "base_group_id"]
        reads.append(Path(path).name)
        return read_parquet(path, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", projected_read)
    identities = classic.read_classic_holdout_identities(bundle)
    assert identities.query_group_ids == frozenset(f"test-{index}" for index in range(len(keys)))
    assert identities.base_group_ids == frozenset(f"base-{index}" for index in range(len(keys)))
    assert set(reads) == set(files.values())


def test_filter_drops_whole_query_and_base_variants_and_resets_indices() -> None:
    rows = pd.DataFrame(
        {
            "query_group_id": ["keep", "test:initial", "mixed", "mixed", "last"],
            "base_group_id": ["safe", "test-base", "test-base", "another", "safe-last"],
            "label": [1, 1, 0, 1, 0],
        },
        index=pd.Index([3, 8, 12, 15, 20]),
    )
    filtered, summary = classic._apply_classic_train_holdout_filter(
        rows, holdout_query_group_ids={"test:full"}, holdout_base_group_ids={"test-base"}
    )
    assert filtered.query_group_id.tolist() == ["keep", "last"]
    assert filtered.index.tolist() == [0, 1]
    assert summary["rows_removed"] == 3
    assert summary["queries_removed"] == 2


def test_removing_query_preserves_retained_native_retrieval_and_row_features(tmp_path: Path) -> None:
    from tests.test_raw_block_candidate_plan_arrow import _base_arrow_paths, _native_labeled_plan

    paths = _base_arrow_paths(tmp_path)
    paths.pop("cluster_seeds")
    components = {"c_match": ["s1"], "c_other": ["s2"]}
    full = _native_labeled_plan(
        paths,
        ["q1", "q1", "s2"],
        ["full"] * 3,
        ["keep", "keep", "drop"],
        ["c_match", "c_other", "c_match"],
        [1, 2, 1],
        components,
        orcid_enabled=False,
        num_threads=1,
    )
    retained = _native_labeled_plan(
        paths,
        ["q1", "q1"],
        ["full"] * 2,
        ["keep", "keep"],
        ["c_match", "c_other"],
        [1, 2],
        components,
        orcid_enabled=False,
        num_threads=1,
    )
    assert retained["telemetry"]["component_count"] == full["telemetry"]["component_count"] == 2
    for key in full:
        if (key.startswith("row_") and key != "row_count") or key in {"retrieval_scores", "retrieval_ranks"}:
            np.testing.assert_equal(retained[key], full[key][:2], err_msg=key)


def test_fit_rejects_reintroduced_overlap_before_classifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _write_classic_tiny_bundle(tmp_path / "bundle")
    rows = pd.read_csv(bundle.root / "train.csv.gz")
    identities = classic.ClassicHoldoutIdentities(frozenset(rows.query_group_id.astype(str)), frozenset(), ())

    def no_classifier(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("classifier construction must not precede overlap rejection")

    monkeypatch.setattr(classic, "_build_classic_classifier", no_classifier)
    with pytest.raises(ValueError, match="Materialized training rows overlap"):
        classic.fit_classic(
            bundle, n_jobs=1, holdout_identities=identities, pre_materialization_holdout_summary={"rows_removed": 2}
        )


def test_real_fit_retains_early_exclusion_counts(tmp_path: Path) -> None:
    bundle = _write_classic_tiny_bundle(tmp_path / "bundle")
    identities = classic.read_classic_holdout_identities(bundle)
    rows = pd.read_csv(bundle.root / "train.csv.gz")
    rows, early = classic._apply_classic_train_holdout_filter(
        rows,
        holdout_query_group_ids=set(identities.query_group_ids),
        holdout_base_group_ids=set(identities.base_group_ids),
    )
    early["stage"] = "before_feature_materialization"
    assert early["rows_removed"] > 0
    rows.to_csv(bundle.root / "train.csv.gz", index=False, compression="gzip")
    # This test-only source has no calibration queries and is not materialized
    # until evaluation. The supplied full authority must avoid reopening it.
    bundle.models["classic"]["s2and_eval_path"] = "not-materialized.csv.gz"
    fitted = classic.fit_classic(
        bundle, n_jobs=1, holdout_identities=identities, pre_materialization_holdout_summary=early
    )
    assert fitted.training_summary["train_holdout_filter_summary"] == early


@pytest.mark.parametrize("remove_all", [False, True])
def test_materializer_filters_before_structural_or_feature_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remove_all: bool,
) -> None:
    rows = pd.DataFrame(
        {
            "dataset": ["toy"] * 3,
            "query_group_id": ["drop", "keep", "drop"],
            "base_group_id": ["test", "safe", "test"],
            "label": [1, 1, 0],
        }
    )
    source = tmp_path / "source"
    source.mkdir()
    rows.to_parquet(source / "train.parquet", index=False)
    bundle = classic.OfficialBundle(
        source, "source", {"featureless_rows": {"files": {"train_path": "train.parquet"}}}, {"classic": {}}, {}
    )
    monkeypatch.setattr(release, "_copy_bundle_support_files", lambda *_args: {})
    identities = classic.ClassicHoldoutIdentities(
        frozenset({"drop", "keep"} if remove_all else {"drop"}), frozenset(), ()
    )

    class ReachedStructuralCheck(Exception):
        pass

    def check_rows(**kwargs: Any) -> Any:
        assert not remove_all
        assert kwargs["rows"].query_group_id.tolist() == ["keep"]
        assert kwargs["rows"].index.tolist() == [0]
        raise ReachedStructuralCheck

    monkeypatch.setattr(release, "_clean_arrow_rust_structural_rows", check_rows)
    expected = ValueError if remove_all else ReachedStructuralCheck
    with pytest.raises(expected, match="No training rows remain" if remove_all else None):
        release._materialize_arrow_rust_feature_bundle(
            source_bundle=bundle,
            output_bundle_root=tmp_path / "output",
            target={"features": []},
            name_tuples=frozenset(),
            clusterer=None,
            n_jobs=1,
            total_ram_bytes=1,
            table_keys=("train_path",),
            holdout_identities=identities,
            max_exemplars=4,
            pairwise_model_nan_value=float("nan"),
            pairwise_aggregate_nan_value=0.0,
            arrow_datasets=cast(Any, {"toy": None}),
        )
