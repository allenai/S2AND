from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest

from scripts._pair_ablation.linker_pairs import (
    BIG_BLOCK_ORCID_LABEL_RULE,
    LINKER_COMPONENT_PROXY_LABEL_RULE,
    PUBLIC_GOLD_LABEL_RULE,
    canonicalize_pair_rows,
    extract_linker_pair_catalog,
)


def _write_bundle(
    root: Path,
    *,
    labels: pd.DataFrame,
    members: dict[str, pd.DataFrame],
    signature_orcids: dict[str, dict[str, str | None]] | None = None,
    signature_blocks: dict[str, dict[str, str | None]] | None = None,
) -> Path:
    labels_dir = root / "labels"
    components_dir = root / "components"
    labels_dir.mkdir(parents=True)
    components_dir.mkdir(parents=True)
    labels.to_parquet(labels_dir / "declared.parquet", index=False)
    component_assets: dict[str, str] = {}
    for dataset, table in members.items():
        path = components_dir / f"{dataset}.parquet"
        table.to_parquet(path, index=False)
        component_assets[dataset] = f"components/{dataset}.parquet"

    orcids_by_dataset = signature_orcids or {}
    blocks_by_dataset = signature_blocks or {}
    for dataset in sorted(set(orcids_by_dataset) | set(blocks_by_dataset)):
        orcids = orcids_by_dataset.get(dataset, {})
        blocks = blocks_by_dataset.get(dataset, {})
        signature_ids = list(dict.fromkeys((*orcids, *blocks)))
        dataset_root = root / "datasets" / dataset
        dataset_root.mkdir(parents=True)
        columns = {"signature_id": pa.array(signature_ids, type=pa.string())}
        if dataset in orcids_by_dataset:
            columns["author_orcid"] = pa.array(
                [orcids.get(signature_id) for signature_id in signature_ids], type=pa.string()
            )
        if dataset in blocks_by_dataset:
            columns["author_block"] = pa.array(
                [blocks.get(signature_id) for signature_id in signature_ids], type=pa.string()
            )
        table = pa.table(columns)
        with pa_ipc.new_file(dataset_root / "signatures.arrow", table.schema) as writer:
            writer.write_table(table)
        (dataset_root / "manifest.json").write_text(
            json.dumps({"paths": {"signatures": "signatures.arrow"}}),
            encoding="utf-8",
        )

    bundle = {
        "assets": {
            "candidate_members": {"datasets": component_assets},
            "featureless_rows": {"files": {"train_path": "labels/declared.parquet"}},
        },
        "runtime_contract": {"arrow_dataset_root": "datasets"},
    }
    bundle_path = root / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    return bundle_path


def _label_row(dataset: str, query: str, component: str, label: int, *, view: str = "full") -> dict[str, object]:
    return {
        "dataset": dataset,
        "query_group_id": f"{dataset}:{query}:{view}",
        "query_signature_id": query,
        "candidate_component_key": component,
        "label": label,
    }


def _member_rows(dataset: str, components: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "candidate_component_key": component,
                "member_index": index,
                "signature_id": signature_id,
            }
            for component, signature_ids in components.items()
            for index, signature_id in enumerate(signature_ids)
        ]
    )


def _write_gold(path: Path, clusters: dict[str, list[str]]) -> Path:
    path.write_text(
        json.dumps(
            {
                cluster_id: {"cluster_id": cluster_id, "signature_ids": signature_ids}
                for cluster_id, signature_ids in clusters.items()
            }
        ),
        encoding="utf-8",
    )
    return path


def test_public_components_are_expanded_then_gold_relabelled_without_broadcast(tmp_path: Path) -> None:
    labels = pd.DataFrame(
        [
            _label_row("public", "q", "component_labeled_positive", 1),
            _label_row("public", "q", "component_labeled_negative", 0),
        ]
    )
    members = {
        "public": _member_rows(
            "public",
            {
                "component_labeled_positive": ["same", "different"],
                "component_labeled_negative": ["same", "other"],
            },
        )
    }
    bundle_path = _write_bundle(tmp_path / "bundle", labels=labels, members=members)
    gold_path = _write_gold(
        tmp_path / "gold.json",
        {"author_q": ["q", "same"], "author_different": ["different"], "author_other": ["other"]},
    )

    catalog = extract_linker_pair_catalog(
        bundle_path,
        public_gold_cluster_paths={"public": gold_path},
        big_block_datasets=(),
    )

    strict = catalog.strict.set_index(["pair1", "pair2"])
    assert strict.loc[("q", "same"), "label"] == 1
    assert strict.loc[("different", "q"), "label"] == 0
    assert strict.loc[("other", "q"), "label"] == 0
    assert set(strict["label_rule"]) == {PUBLIC_GOLD_LABEL_RULE}
    assert len(strict) == 3  # q/same appeared under both component labels and was deduplicated.
    assert catalog.linker_component_proxy.empty


def test_prefixed_public_component_uses_only_block_local_members(tmp_path: Path) -> None:
    labels = pd.DataFrame([_label_row("public", "q", "target::component", 1)])
    members = {
        "public": _member_rows(
            "public",
            {"target::component": ["wrong-first", "right-a", "right-b", "wrong-last"]},
        )
    }
    bundle_path = _write_bundle(
        tmp_path / "bundle",
        labels=labels,
        members=members,
        signature_blocks={
            "public": {
                "q": "target",
                "wrong-first": "other",
                "right-a": "target",
                "right-b": "target",
                "wrong-last": "other",
            }
        },
    )
    gold_path = _write_gold(
        tmp_path / "gold.json",
        {
            "query": ["q", "right-a"],
            "right-b": ["right-b"],
            "wrong-first": ["wrong-first"],
            "wrong-last": ["wrong-last"],
        },
    )

    catalog = extract_linker_pair_catalog(
        bundle_path,
        public_gold_cluster_paths={"public": gold_path},
        big_block_datasets=(),
    )

    assert catalog.strict[["pair1", "pair2", "label"]].to_dict("records") == [
        {"pair1": "q", "pair2": "right-a", "label": 1},
        {"pair1": "q", "pair2": "right-b", "label": 0},
    ]


def test_prefixed_component_falls_back_to_raw_members_only_when_block_filter_is_empty(tmp_path: Path) -> None:
    labels = pd.DataFrame([_label_row("public", "q", "missing-block::component", 1)])
    members = {"public": _member_rows("public", {"missing-block::component": ["first", "second"]})}
    bundle_path = _write_bundle(
        tmp_path / "bundle",
        labels=labels,
        members=members,
        signature_blocks={"public": {"q": "query", "first": "other", "second": None}},
    )
    gold_path = _write_gold(
        tmp_path / "gold.json",
        {"query": ["q", "first"], "second": ["second"]},
    )

    catalog = extract_linker_pair_catalog(
        bundle_path,
        public_gold_cluster_paths={"public": gold_path},
        big_block_datasets=(),
    )

    assert catalog.strict[["pair1", "pair2", "label"]].to_dict("records") == [
        {"pair1": "first", "pair2": "q", "label": 1},
        {"pair1": "q", "pair2": "second", "label": 0},
    ]


def test_prefixed_component_requires_declared_block_schema_and_dataset_local_path(tmp_path: Path) -> None:
    labels = pd.DataFrame([_label_row("public", "q", "target::component", 1)])
    members = {"public": _member_rows("public", {"target::component": ["member"]})}
    gold_path = _write_gold(tmp_path / "gold.json", {"query": ["q"], "member": ["member"]})

    missing_block_bundle = _write_bundle(
        tmp_path / "missing-block-bundle",
        labels=labels,
        members=members,
        signature_orcids={"public": {"q": None, "member": None}},
    )
    with pytest.raises(ValueError, match="missing columns.*author_block"):
        extract_linker_pair_catalog(
            missing_block_bundle,
            public_gold_cluster_paths={"public": gold_path},
            big_block_datasets=(),
        )

    escaping_bundle = _write_bundle(
        tmp_path / "escaping-bundle",
        labels=labels,
        members=members,
        signature_blocks={"public": {"q": "target", "member": "target"}},
    )
    escaping_root = escaping_bundle.parent
    (escaping_root / "outside.arrow").write_bytes(
        (escaping_root / "datasets" / "public" / "signatures.arrow").read_bytes()
    )
    (escaping_root / "datasets" / "public" / "manifest.json").write_text(
        json.dumps({"paths": {"signatures": "../../outside.arrow"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes the Arrow dataset root"):
        extract_linker_pair_catalog(
            escaping_bundle,
            public_gold_cluster_paths={"public": gold_path},
            big_block_datasets=(),
        )


def test_big_block_strict_catalog_uses_only_same_normalized_orcid_positives(tmp_path: Path) -> None:
    labels = pd.DataFrame([_label_row("big", "q", "c", 0)])
    members = {
        "big": _member_rows(
            "big",
            {"c": ["same", "different", "missing"], "unused_component": ["not_in_signature_table"]},
        )
    }
    bundle_path = _write_bundle(
        tmp_path / "bundle",
        labels=labels,
        members=members,
        signature_orcids={
            "big": {
                "q": "ORCID: 0000-0002-1825-0097",
                "same": "0000000218250097",
                "different": "0000-0002-1825-0098",
                "missing": None,
            }
        },
    )

    catalog = extract_linker_pair_catalog(
        bundle_path,
        public_gold_cluster_paths={},
        big_block_datasets={"big"},
        proxy_negatives_per_query=3,
        proxy_negatives_per_domain=None,
    )

    assert catalog.strict[["pair1", "pair2", "label"]].to_dict("records") == [
        {"pair1": "q", "pair2": "same", "label": 1}
    ]
    assert set(catalog.strict["label_rule"]) == {BIG_BLOCK_ORCID_LABEL_RULE}
    assert ("q", "same") not in set(
        catalog.linker_component_proxy[["pair1", "pair2"]].itertuples(index=False, name=None)
    )
    assert set(catalog.linker_component_proxy["label"]) == {0}
    assert set(catalog.linker_component_proxy["label_rule"]) == {LINKER_COMPONENT_PROXY_LABEL_RULE}


def test_big_block_orcid_and_proxy_expansion_use_block_local_members(tmp_path: Path) -> None:
    labels = pd.DataFrame([_label_row("big", "q", "target::component", 0)])
    members = {"big": _member_rows("big", {"target::component": ["local", "cross-block"]})}
    bundle_path = _write_bundle(
        tmp_path / "bundle",
        labels=labels,
        members=members,
        signature_orcids={
            "big": {
                "q": "0000-0002-1825-0097",
                "local": "0000-0002-1825-0097",
                "cross-block": "0000-0002-1825-0097",
            }
        },
        signature_blocks={"big": {"q": "target", "local": "target", "cross-block": "other"}},
    )

    catalog = extract_linker_pair_catalog(
        bundle_path,
        public_gold_cluster_paths={},
        big_block_datasets={"big"},
        proxy_negatives_per_query=2,
        proxy_negatives_per_domain=None,
    )

    assert catalog.strict[["pair1", "pair2", "label"]].to_dict("records") == [
        {"pair1": "local", "pair2": "q", "label": 1}
    ]
    assert catalog.linker_component_proxy.empty


def test_prefixed_component_rejects_duplicate_arrow_signature_ids(tmp_path: Path) -> None:
    labels = pd.DataFrame([_label_row("public", "q", "target::component", 1)])
    members = {"public": _member_rows("public", {"target::component": ["member"]})}
    bundle_path = _write_bundle(
        tmp_path / "bundle",
        labels=labels,
        members=members,
        signature_blocks={"public": {"q": "target", "member": "target"}},
    )
    signatures_path = bundle_path.parent / "datasets" / "public" / "signatures.arrow"
    table = pa.table(
        {
            "signature_id": ["q", "member", "member"],
            "author_block": ["target", "target", "target"],
        }
    )
    with pa_ipc.new_file(signatures_path, table.schema) as writer:
        writer.write_table(table)
    gold_path = _write_gold(tmp_path / "gold.json", {"query": ["q", "member"]})

    with pytest.raises(ValueError, match="duplicate signature_id 'member'"):
        extract_linker_pair_catalog(
            bundle_path,
            public_gold_cluster_paths={"public": gold_path},
            big_block_datasets=(),
        )


@pytest.mark.parametrize("missing_signature_id", ["q", "member"])
def test_prefixed_component_rejects_referenced_ids_missing_from_arrow(
    tmp_path: Path,
    missing_signature_id: str,
) -> None:
    labels = pd.DataFrame([_label_row("public", "q", "target::component", 1)])
    members = {"public": _member_rows("public", {"target::component": ["member"]})}
    bundle_path = _write_bundle(
        tmp_path / "bundle",
        labels=labels,
        members=members,
        signature_blocks={"public": {"q": "target", "member": "target"}},
    )
    retained_signature_id = "member" if missing_signature_id == "q" else "q"
    table = pa.table(
        {
            "signature_id": pa.array([retained_signature_id], type=pa.string()),
            "author_block": pa.array(["target"], type=pa.string()),
        }
    )
    signatures_path = bundle_path.parent / "datasets" / "public" / "signatures.arrow"
    with pa_ipc.new_file(signatures_path, table.schema) as writer:
        writer.write_table(table)
    gold_path = _write_gold(tmp_path / "gold.json", {"query": ["q", "member"]})

    with pytest.raises(ValueError, match=rf"missing referenced IDs.*{missing_signature_id}"):
        extract_linker_pair_catalog(
            bundle_path,
            public_gold_cluster_paths={"public": gold_path},
            big_block_datasets=(),
        )


@pytest.mark.parametrize("invalid_column", ["signature_id", "author_block", "author_orcid"])
def test_linker_signature_metadata_requires_production_string_types(
    tmp_path: Path,
    invalid_column: str,
) -> None:
    labels = pd.DataFrame([_label_row("big", "q", "target::component", 0)])
    members = {"big": _member_rows("big", {"target::component": ["candidate"]})}
    bundle_path = _write_bundle(
        tmp_path / "bundle",
        labels=labels,
        members=members,
        signature_orcids={"big": {"q": None, "candidate": None}},
        signature_blocks={"big": {"q": "target", "candidate": "target"}},
    )
    values: dict[str, pa.Array] = {
        "signature_id": pa.array(["q", "candidate"], type=pa.string()),
        "author_block": pa.array(["target", "target"], type=pa.string()),
        "author_orcid": pa.array([None, None], type=pa.string()),
    }
    values[invalid_column] = pa.array([1, 2], type=pa.int64())
    table = pa.table(values)
    signatures_path = bundle_path.parent / "datasets" / "big" / "signatures.arrow"
    with pa_ipc.new_file(signatures_path, table.schema) as writer:
        writer.write_table(table)

    with pytest.raises(ValueError, match=rf"column '{invalid_column}'.*string or large_string"):
        extract_linker_pair_catalog(
            bundle_path,
            public_gold_cluster_paths={},
            big_block_datasets={"big"},
        )


def test_linker_signature_metadata_accepts_production_large_string_types(tmp_path: Path) -> None:
    labels = pd.DataFrame([_label_row("big", "q", "target::component", 0)])
    members = {"big": _member_rows("big", {"target::component": ["candidate"]})}
    bundle_path = _write_bundle(
        tmp_path / "bundle",
        labels=labels,
        members=members,
        signature_orcids={"big": {"q": None, "candidate": None}},
        signature_blocks={"big": {"q": "target", "candidate": "target"}},
    )
    table = pa.table(
        {
            "signature_id": pa.array(["q", "candidate"], type=pa.large_string()),
            "author_block": pa.array(["target", "target"], type=pa.large_string()),
            "author_orcid": pa.array(
                ["0000-0002-1825-0097", "0000-0002-1825-0097"],
                type=pa.large_string(),
            ),
        }
    )
    signatures_path = bundle_path.parent / "datasets" / "big" / "signatures.arrow"
    with pa_ipc.new_file(signatures_path, table.schema) as writer:
        writer.write_table(table)

    catalog = extract_linker_pair_catalog(
        bundle_path,
        public_gold_cluster_paths={},
        big_block_datasets={"big"},
    )

    assert catalog.strict[["pair1", "pair2", "label"]].to_dict("records") == [
        {"pair1": "candidate", "pair2": "q", "label": 1}
    ]


def test_canonical_pair_dedup_rejects_conflicting_final_labels() -> None:
    base = {
        "source_domain": "d",
        "source_family": "linker",
        "label_rule": "rule",
        "origin": "asset",
        "group_id": "q",
    }
    duplicates = pd.DataFrame(
        [
            {**base, "pair1": "b", "pair2": "a", "label": 1},
            {**base, "pair1": "a", "pair2": "b", "label": 1},
        ]
    )
    canonical = canonicalize_pair_rows(duplicates)
    assert canonical[["pair1", "pair2", "label"]].to_dict("records") == [{"pair1": "a", "pair2": "b", "label": 1}]

    conflicting = pd.concat([duplicates, pd.DataFrame([{**base, "pair1": "a", "pair2": "b", "label": 0}])])
    with pytest.raises(ValueError, match="Conflicting final labels"):
        canonicalize_pair_rows(conflicting)


def test_extractor_ignores_undeclared_label_and_component_files(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    labels = pd.DataFrame([_label_row("public", "q", "declared_component", 1)])
    members = {"public": _member_rows("public", {"declared_component": ["same"]})}
    bundle_path = _write_bundle(bundle_root, labels=labels, members=members)
    (bundle_root / "labels" / "undeclared.parquet").write_bytes(b"not a parquet file")
    (bundle_root / "components" / "undeclared.parquet").write_bytes(b"not a parquet file")
    gold_path = _write_gold(tmp_path / "gold.json", {"author": ["q", "same"]})

    catalog = extract_linker_pair_catalog(
        bundle_path,
        public_gold_cluster_paths={"public": gold_path},
        big_block_datasets=(),
    )

    assert catalog.strict[["pair1", "pair2"]].to_dict("records") == [{"pair1": "q", "pair2": "same"}]
    assert "declared.parquet" in catalog.strict.iloc[0]["origin"]


def test_linker_component_proxy_caps_are_deterministic_and_never_use_label1_components(tmp_path: Path) -> None:
    labels = pd.DataFrame(
        [
            _label_row("big", "q2", "negative_b", 0),
            _label_row("big", "q1", "positive_only", 1),
            _label_row("big", "q1", "negative_a", 0),
            _label_row("big", "q2", "negative_a", 0),
        ]
    )
    component_map = {
        "negative_a": ["a", "b", "c", "d"],
        "negative_b": ["e", "f", "g"],
        "positive_only": ["must_not_be_broadcast"],
    }
    members = {"big": _member_rows("big", component_map)}
    all_ids = {"q1", "q2", *(signature for values in component_map.values() for signature in values)}
    signature_orcids = {signature_id: None for signature_id in all_ids}
    bundle_path = _write_bundle(
        tmp_path / "bundle",
        labels=labels,
        members=members,
        signature_orcids={"big": signature_orcids},
    )

    def extract_proxy() -> pd.DataFrame:
        return extract_linker_pair_catalog(
            bundle_path,
            public_gold_cluster_paths={},
            big_block_datasets={"big"},
            proxy_negatives_per_query=2,
            proxy_negatives_per_domain=3,
            seed=71,
        ).linker_component_proxy

    first = extract_proxy()
    labels.sample(frac=1, random_state=9).to_parquet(tmp_path / "bundle" / "labels" / "declared.parquet", index=False)
    members["big"].sample(frac=1, random_state=11).to_parquet(
        tmp_path / "bundle" / "components" / "big.parquet", index=False
    )
    second = extract_proxy()

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 3
    assert first.groupby("query_signature_id").size().max() <= 2
    assert "must_not_be_broadcast" not in set(first["pair1"]) | set(first["pair2"])
    assert set(first["label"]) == {0}


def test_linker_component_proxy_excludes_relation_observed_with_both_labels(tmp_path: Path) -> None:
    labels = pd.DataFrame(
        [
            _label_row("big", "q", "component", 0, view="full"),
            _label_row("big", "q", "component", 1, view="first_initial"),
        ]
    )
    members = {"big": _member_rows("big", {"component": ["candidate"]})}
    bundle_path = _write_bundle(
        tmp_path / "bundle",
        labels=labels,
        members=members,
        signature_orcids={"big": {"q": None, "candidate": None}},
    )

    catalog = extract_linker_pair_catalog(
        bundle_path,
        public_gold_cluster_paths={},
        big_block_datasets={"big"},
        proxy_negatives_per_query=2,
        proxy_negatives_per_domain=None,
    )

    assert catalog.linker_component_proxy.empty
