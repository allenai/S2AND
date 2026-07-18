from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest

import scripts.run_pair_source_ablation as pair_ablation_runner
from scripts._pair_ablation.evaluation import (
    B3DomainEvaluationPlans,
    B3EvaluationPlan,
    GoldBlockData,
)
from scripts._pair_ablation.modeling import ablation_arm_registry, default_ablation_arms
from scripts._pair_ablation.pair_sources import PAIR_COLUMNS
from scripts._pair_ablation.results import load_strict_json, recipe_id_for
from scripts.run_pair_source_ablation import (
    ExperimentConfig,
    _catalog_input_paths,
    _config_from_args,
    _fixed_binary_evaluation_cap,
    _git_identity,
    _implementation_identity,
    _persist_summary_with_ranking_input,
    _public_evaluation_pairs,
    _purpose_seed,
    _recipe_metadata,
    _result_path,
    _select_arms,
    _training_gold_catalogs,
    _validate_config,
    build_parser,
    validated_reused_b3_builder_identity,
)


def _config(**overrides: object) -> ExperimentConfig:
    values = {
        "training_seed": 1111,
        "evaluation_seed": 1111,
        "n_jobs": 20,
        "total_ram_gib": 200,
        "uniform_pairs_per_domain": 100,
        "name_pairs_per_domain": 10,
        "balanced_medium_pairs_per_domain": 50,
        "balanced_pool_pairs_per_domain": 100,
        "linker_pairs_per_domain": 10,
        "catalog_pool_cap_per_domain": None,
        "big_proxy_eval_pairs_per_class": 10,
        "eval_pairs_per_domain": 20,
        "threshold_pairs_per_domain": 20,
        "estimator_scale": 0.01,
        "b3_scope": "test",
        "public_domains": ("pubmed", "qian"),
        "big_block_domains": ("h_wang",),
        "fold_domains": ("pubmed", "qian", "medline", "h_wang"),
        "arm_names": ("uniform_100k", "uniform_budget_balanced_random"),
    }
    values.update(overrides)
    return ExperimentConfig(**values)  # type: ignore[arg-type]


def _gold() -> GoldBlockData:
    signatures = [str(index) for index in range(40)]
    blocks = {"block_a": signatures[:20], "block_b": signatures[20:]}
    clusters = {signature: f"author_{int(signature) // 2}" for signature in signatures}
    names = {signature: f"given family{int(signature) % 5}" for signature in signatures}
    return GoldBlockData("qian", blocks, clusters, names)


def _pair_row(domain: str, pair1: str, pair2: str, label: int) -> dict[str, object]:
    return {
        "source_domain": domain,
        "source_family": "raw",
        "pair1": pair1,
        "pair2": pair2,
        "label": label,
        "label_rule": "fixture",
        "origin": "fixture",
        "group_id": f"{domain}:{pair1}",
    }


def test_named_seed_derivation_is_deterministic_and_purpose_specific() -> None:
    assert _purpose_seed(1111, "evaluation", "qian") == _purpose_seed(1111, "evaluation", "qian")
    assert _purpose_seed(1111, "evaluation", "qian") != _purpose_seed(1111, "training", "qian")
    assert _purpose_seed(1111, "evaluation", "qian") != _purpose_seed(2222, "evaluation", "qian")


def test_implementation_identity_covers_all_runtime_python_sources() -> None:
    identity = _implementation_identity()
    expected = {
        str(Path("scripts") / "run_pair_source_ablation.py"),
        *(
            str(path.relative_to(pair_ablation_runner.REPO_ROOT))
            for path in (pair_ablation_runner.REPO_ROOT / "scripts" / "_pair_ablation").rglob("*.py")
        ),
        *(
            str(path.relative_to(pair_ablation_runner.REPO_ROOT))
            for path in (pair_ablation_runner.REPO_ROOT / "s2and").rglob("*.py")
        ),
    }

    assert set(identity) == expected
    assert all(len(digest) == 64 and set(digest) <= set("0123456789abcdef") for digest in identity.values())


def test_catalog_identity_includes_bundle_signatures_consumed_for_block_filter_and_orcid(tmp_path: Path) -> None:
    linker_root = tmp_path / "linker"
    (linker_root / "labels").mkdir(parents=True)
    (linker_root / "components").mkdir()
    labels = pd.DataFrame(
        [
            {
                "dataset": "pubmed",
                "query_group_id": "pubmed:q:full",
                "query_signature_id": "q",
                "candidate_component_key": "smith::component",
                "label": 1,
            },
            {
                "dataset": "h_wang",
                "query_group_id": "h_wang:q:full",
                "query_signature_id": "q",
                "candidate_component_key": "component",
                "label": 0,
            },
        ]
    )
    labels.to_parquet(linker_root / "labels" / "rows.parquet", index=False)
    component_assets: dict[str, str] = {}
    for dataset, component in (("pubmed", "smith::component"), ("h_wang", "component")):
        relative_path = f"components/{dataset}.parquet"
        pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "candidate_component_key": component,
                    "member_index": 0,
                    "signature_id": "member",
                }
            ]
        ).to_parquet(linker_root / relative_path, index=False)
        component_assets[dataset] = relative_path

        dataset_root = linker_root / "declared-arrow-root" / dataset
        dataset_root.mkdir(parents=True)
        table = pa.table({"signature_id": ["q", "member"]})
        with pa_ipc.new_file(dataset_root / "declared-signatures.arrow", table.schema) as writer:
            writer.write_table(table)
        (dataset_root / "manifest.json").write_text(
            json.dumps({"paths": {"signatures": "declared-signatures.arrow"}}),
            encoding="utf-8",
        )

    (linker_root / "bundle.json").write_text(
        json.dumps(
            {
                "assets": {
                    "candidate_members": {"datasets": component_assets},
                    "featureless_rows": {"files": {"train": "labels/rows.parquet"}},
                },
                "runtime_contract": {"arrow_dataset_root": "declared-arrow-root"},
            }
        ),
        encoding="utf-8",
    )

    paths = _catalog_input_paths(
        data_root=tmp_path / "data",
        backup_data_root=tmp_path / "backup",
        linker_bundle_root=linker_root,
        config=_config(
            public_domains=("pubmed",),
            big_block_domains=("h_wang",),
            fold_domains=("pubmed", "medline", "h_wang"),
        ),
    )

    for dataset in ("pubmed", "h_wang"):
        assert (
            paths[f"linker.dataset_manifest.{dataset}"]
            == (linker_root / "declared-arrow-root" / dataset / "manifest.json").resolve()
        )
        assert (
            paths[f"linker.signatures.{dataset}"]
            == (linker_root / "declared-arrow-root" / dataset / "declared-signatures.arrow").resolve()
        )


def test_git_identity_hashes_raw_binary_diff_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    diff_bytes = b"binary-diff\x00\xff\r\n"

    def fake_check_output(
        arguments: tuple[str, ...],
        *,
        cwd: object,
        text: bool = False,
    ) -> str | bytes:
        assert cwd == pair_ablation_runner.REPO_ROOT
        if arguments == ("git", "diff", "--binary", "HEAD"):
            assert text is False
            return diff_bytes
        assert text is True
        return {
            ("git", "rev-parse", "HEAD"): "abc123\n",
            ("git", "branch", "--show-current"): "fixture\n",
            ("git", "status", "--short"): " M tracked.py\n",
        }[arguments]

    monkeypatch.setattr(pair_ablation_runner.subprocess, "check_output", fake_check_output)

    identity = _git_identity()

    assert identity == {
        "commit": "abc123",
        "branch": "fixture",
        "diff_binary_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "status_short": "M tracked.py",
    }


def test_b3_fold_digest_binds_member_identity_version(monkeypatch: pytest.MonkeyPatch) -> None:
    def domain_plans(dataset: str) -> B3DomainEvaluationPlans:
        return B3DomainEvaluationPlans(
            calibration=B3EvaluationPlan(dataset, "calibration", 1111, 10, (), ()),
            heldout=B3EvaluationPlan(dataset, "heldout_test", 1111, None, (), ()),
        )

    plans = {
        "heldout": domain_plans("heldout"),
        "calibration": domain_plans("calibration"),
    }
    original = pair_ablation_runner._b3_fold_digest(
        plans,
        held_out_domain="heldout",
        b3_scope="test",
    )
    monkeypatch.setattr(pair_ablation_runner, "B3_MEMBER_IDENTITY_VERSION", "changed-for-test")
    changed = pair_ablation_runner._b3_fold_digest(
        plans,
        held_out_domain="heldout",
        b3_scope="test",
    )

    assert changed != original


def test_training_seed_changes_training_but_not_fixed_evaluation() -> None:
    gold = _gold()
    first_catalogs = _training_gold_catalogs(
        gold,
        uniform_pairs=50,
        balanced_pool_pairs=20,
        training_seed=1111,
    )
    second_catalogs = _training_gold_catalogs(
        gold,
        uniform_pairs=50,
        balanced_pool_pairs=20,
        training_seed=2222,
    )
    first_training = first_catalogs[0]
    second_training = second_catalogs[0]
    first_evaluation = _public_evaluation_pairs(gold, eval_pairs=30, evaluation_seed=1111)
    repeated_evaluation = _public_evaluation_pairs(gold, eval_pairs=30, evaluation_seed=1111)
    changed_evaluation = _public_evaluation_pairs(gold, eval_pairs=30, evaluation_seed=2222)

    assert not first_training[["pair1", "pair2"]].equals(second_training[["pair1", "pair2"]])
    assert first_catalogs[1]["label"].value_counts().to_dict() == {0: 10, 1: 10}
    assert not first_catalogs[1][["pair1", "pair2"]].equals(second_catalogs[1][["pair1", "pair2"]])
    pd.testing.assert_frame_equal(first_evaluation, repeated_evaluation)
    assert not first_evaluation[["pair1", "pair2"]].equals(changed_evaluation[["pair1", "pair2"]])


def test_big_proxy_evaluation_cap_has_no_majority_backfill() -> None:
    rows = [
        *[_pair_row("h_wang", f"p{index}", f"q{index}", 1) for index in range(8)],
        *[_pair_row("h_wang", f"n{index}", f"m{index}", 0) for index in range(2)],
    ]
    selected = _fixed_binary_evaluation_cap(
        pd.DataFrame(rows, columns=list(PAIR_COLUMNS)),
        cap_per_class=3,
        evaluation_seed=1111,
    )

    assert selected["label"].value_counts().to_dict() == {1: 3, 0: 2}
    assert set(selected["source_family"]) == {"evaluation_linker_component_proxy"}


def test_recipe_identity_changes_with_linker_dose_only_when_recipe_uses_linker() -> None:
    by_name = {arm.name: arm for arm in default_ablation_arms()}
    config_10k = _config(linker_pairs_per_domain=10_000)
    config_50k = _config(linker_pairs_per_domain=50_000)
    recipe_ids_10k = {recipe_id_for(_recipe_metadata(arm, config_10k)) for arm in by_name.values()}
    registry_recipe_ids = {recipe_id_for(_recipe_metadata(arm, config_10k)) for arm in ablation_arm_registry()}
    baseline_10k = _recipe_metadata(by_name["uniform_100k"], config_10k)
    baseline_50k = _recipe_metadata(by_name["uniform_100k"], config_50k)
    linker_10k = _recipe_metadata(by_name["uniform_budget_balanced_plus_linker_balanced"], config_10k)
    linker_50k = _recipe_metadata(by_name["uniform_budget_balanced_plus_linker_balanced"], config_50k)
    proxy_10k = _recipe_metadata(by_name["uniform_budget_linker_proxy_negative_only"], config_10k)
    proxy_50k = _recipe_metadata(by_name["uniform_budget_linker_proxy_negative_only"], config_50k)
    balanced_low = _recipe_metadata(by_name["uniform_budget_balanced_random"], config_10k)
    balanced_medium = _recipe_metadata(by_name["uniform_budget_balanced_random_50k"], config_10k)
    balanced_max = _recipe_metadata(by_name["uniform_budget_balanced_random_100k"], config_10k)

    assert len(recipe_ids_10k) == len(by_name)
    assert len(registry_recipe_ids) == len(ablation_arm_registry()) == 15
    assert recipe_id_for(baseline_10k) == recipe_id_for(baseline_50k)
    assert recipe_id_for(linker_10k) != recipe_id_for(linker_50k)
    assert recipe_id_for(proxy_10k) != recipe_id_for(proxy_50k)
    assert proxy_10k["auxiliary_sources"] == ["capped_proxy_negative"]
    assert proxy_10k["balancing"] == "proxy_per_domain_deterministic_cap_negative_only_no_backfill"
    assert proxy_10k["source_caps"]["linker_pairs_per_domain"] == 10_000
    assert balanced_low["source_caps"] == {
        "uniform_pairs_per_domain": 100,
        "balanced_pairs_per_domain": 10,
        "balanced_pool_pairs_per_domain": 100,
    }
    assert balanced_medium["source_caps"]["balanced_pairs_per_domain"] == 50
    assert balanced_max["source_caps"]["balanced_pairs_per_domain"] == 100
    registry = {arm.name: arm for arm in ablation_arm_registry()}
    linker_50k = _recipe_metadata(registry["uniform_budget_linker_balanced_50k"], config_10k)
    assert linker_50k["source_caps"]["linker_pairs_per_domain"] == 50_000


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"b3_scope": "unknown"}, "b3_scope"),
        ({"fold_domains": ("pubmed", "pubmed")}, "fold_domains"),
        ({"arm_names": ("uniform_100k", "uniform_100k")}, "arm_names"),
        ({"public_domains": ("pubmed",), "fold_domains": ("pubmed", "medline")}, "at least two"),
        ({"training_seed": -1}, "non-negative"),
        ({"big_proxy_eval_pairs_per_class": 0}, "must be positive"),
        ({"name_pairs_per_domain": 50}, "low < medium"),
        ({"name_pairs_per_domain": 9}, "must be even"),
        ({"uniform_pairs_per_domain": 99}, "must fit"),
    ],
)
def test_config_rejects_invalid_states(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_config(_config(**overrides))


def test_parser_exposes_separate_training_and_evaluation_seeds() -> None:
    args = build_parser().parse_args(["--training-seed", "2222", "--evaluation-seed", "1111", "--smoke"])

    assert args.training_seed == 2222
    assert args.evaluation_seed == 1111
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--seed", "2222", "--smoke"])


def test_default_smoke_config_validates_and_includes_h_wang() -> None:
    config = _config_from_args(build_parser().parse_args(["--smoke"]))
    smaller_cap = _config_from_args(build_parser().parse_args(["--smoke", "--catalog-pool-cap-per-domain", "25"]))

    _validate_config(config)

    assert config.big_block_domains == ("h_wang",)
    assert config.fold_domains == ("pubmed", "qian", "medline", "h_wang")
    assert config.catalog_pool_cap_per_domain == 50
    assert smaller_cap.catalog_pool_cap_per_domain == 25
    assert config.arm_names == tuple(arm.name for arm in default_ablation_arms())
    assert len(config.arm_names) == 11
    assert (
        config.name_pairs_per_domain,
        config.balanced_medium_pairs_per_domain,
        config.balanced_pool_pairs_per_domain,
    ) == (40, 100, 200)
    # After holding out one of two public domains, the largest default arm can
    # draw one balanced-random pool plus three capped auxiliary domain pools.
    assert config.name_pairs_per_domain + 3 * config.catalog_pool_cap_per_domain <= config.uniform_pairs_per_domain


def test_result_path_is_namespaced_by_arm_and_domain(tmp_path) -> None:
    assert _result_path(tmp_path, "recipe", "qian") == tmp_path / "results" / "recipe" / "qian.json"


def test_optional_linker_dose_arm_is_selectable_from_frozen_registry() -> None:
    selected = _select_arms(("uniform_budget_pairwise_linker_balanced_50k",))

    assert len(selected) == 1
    assert selected[0].exact_budget_recipe is not None
    assert selected[0].exact_budget_recipe.linker_pairs_per_domain == 50_000


def test_dynamic_additive_linker_arm_is_selectable_and_recipe_binds_set_and_dose() -> None:
    selected = _select_arms(("uniform_100k_plus_linker_big7_2500",))

    assert len(selected) == 1
    arm = selected[0]
    assert arm.exact_budget_recipe is None
    assert arm.additive_linker_recipe is not None
    assert arm.additive_linker_recipe.source_set == "big7"
    assert arm.additive_linker_recipe.linker_pairs_per_domain == 2_500
    recipe = _recipe_metadata(arm, _config(arm_names=(arm.name,)))
    assert recipe["assembly_version"] == "additive_linker_lodo_v1"
    assert recipe["budget_policy"] == "additive_to_unchanged_uniform_after_lodo"
    assert recipe["fixed_budget"] is False
    assert recipe["auxiliary_sources"] == ["balanced_linker_big7"]
    assert recipe["source_caps"]["linker_pairs_per_domain"] == 2_500


@pytest.mark.parametrize(
    "name",
    (
        "uniform_100k_plus_linker_big7_625",
        "uniform_100k_plus_linker_unknown_2500",
        "uniform_100k_plus_linker_big7_0",
    ),
)
def test_dynamic_additive_linker_arm_rejects_invalid_name_or_odd_dose(name: str) -> None:
    with pytest.raises(ValueError):
        _select_arms((name,))


@pytest.mark.parametrize("separator", ("/", "\\"))
def test_reused_b3_builder_identity_allows_only_runner_and_modeling_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    separator: str,
) -> None:
    runner_path = f"scripts{separator}run_pair_source_ablation.py"
    modeling_path = f"scripts{separator}_pair_ablation{separator}modeling.py"
    b3_cache_path = f"scripts{separator}_pair_ablation{separator}b3_cache.py"
    source_implementation = {
        runner_path: "a" * 64,
        modeling_path: "b" * 64,
        b3_cache_path: "c" * 64,
        "s2and/model.py": "d" * 64,
    }
    current_implementation = {
        **source_implementation,
        runner_path: "e" * 64,
        modeling_path: "f" * 64,
    }
    runtime_versions = {"python": "fixture"}
    source_manifest = {
        "implementation_sha256": source_implementation,
        "runtime_versions": runtime_versions,
        "rust_version": "0.60.0",
        "rust_extension_sha256": "1" * 64,
        "featurizer_version": pair_ablation_runner.FEATURIZER_VERSION,
    }
    written: dict[str, object] = {}
    monkeypatch.setattr(pair_ablation_runner, "load_run_manifest", lambda _path: source_manifest)
    monkeypatch.setattr(pair_ablation_runner, "sha256_file", lambda _path: "2" * 64)
    monkeypatch.setattr(
        pair_ablation_runner,
        "b3_cache_builder_identity",
        lambda **_kwargs: "3" * 64,
    )
    monkeypatch.setattr(
        pair_ablation_runner,
        "_write_json_atomic",
        lambda path, payload: written.update(path=path, payload=payload),
    )

    observed = validated_reused_b3_builder_identity(
        artifact_source_dir=tmp_path / "source",
        output_dir=tmp_path / "output",
        current_implementation_sha256=current_implementation,
        current_runtime_versions=runtime_versions,
        current_rust_version="0.60.0",
        current_rust_extension_sha256="1" * 64,
    )

    assert observed == "3" * 64
    assert written["path"] == tmp_path / "output" / "b3_builder_reuse_verification.json"
    assert written["payload"]["observed_changed_paths"] == [  # type: ignore[index]
        "scripts/_pair_ablation/modeling.py",
        "scripts/run_pair_source_ablation.py",
    ]

    changed_builder = {**current_implementation, b3_cache_path: "4" * 64}
    with pytest.raises(ValueError, match="builder dependencies changed"):
        validated_reused_b3_builder_identity(
            artifact_source_dir=tmp_path / "source",
            output_dir=tmp_path / "output",
            current_implementation_sha256=changed_builder,
            current_runtime_versions=runtime_versions,
            current_rust_version="0.60.0",
            current_rust_extension_sha256="1" * 64,
        )


def test_completed_summary_persists_same_ranking_path_as_returned_payload(tmp_path) -> None:
    summary_path = tmp_path / "summary" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text('{"complete": true, "run_id": "fixture"}', encoding="utf-8")
    ranking_input_path = tmp_path / "ranking_input.json"

    returned = _persist_summary_with_ranking_input(
        output_dir=tmp_path,
        summary={"complete": True, "run_id": "fixture"},
        ranking_input_path=ranking_input_path,
    )

    persisted = load_strict_json(summary_path)
    assert persisted == returned
    assert returned["ranking_input_path"] == str(ranking_input_path)
    assert not summary_path.with_suffix(".json.tmp").exists()
