from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from s2and import production_training_contract
from s2and._sha256 import sha256_file
from scripts.production.model import train_pairwise
from tests.helpers import pairwise_training_args as _args


def _artifact_authority(
    *,
    tuple_hash: str = "a" * 64,
) -> train_pairwise.ProductionArtifactAuthority:
    return train_pairwise.ProductionArtifactAuthority(
        name_counts_index=SimpleNamespace(
            manifest_sha256="d" * 64,
        ),
        name_tuples=SimpleNamespace(data_sha256="a" * 64, pairs=frozenset()),
        orcid_prefix_counts=SimpleNamespace(
            name_tuples_sha256=tuple_hash,
            data_sha256="b" * 64,
        ),
    )


def _write_model_plan(tmp_path: Path, dataset_names: tuple[str, ...] | None = None) -> Path:
    """Write an Arrow model plan whose files only need stable digests for preflight."""

    input_root = tmp_path / "planned_inputs"
    datasets = {}
    names = (*train_pairwise.DEFAULT_SOURCE_DATASET_NAMES, "augmented") if dataset_names is None else dataset_names
    for name in names:
        dataset_dir = input_root / name
        dataset_dir.mkdir(parents=True)
        roles = {
            "arrow": "manifest.json",
            "heldout_pairs": "test_pairs.json",
            **(
                {"train_pairs": "train_pairs.csv", "val_pairs": "val_pairs.csv"}
                if name == "augmented"
                else {"clusters": f"{name}_clusters.json", "heldout_blocks": "test_blocks.json"}
            ),
        }
        files = {}
        for role, filename in sorted(roles.items()):
            path = dataset_dir / filename
            if role in {"train_pairs", "val_pairs"}:
                path.write_text("signature_id_1,signature_id_2,label\nmedline___a,medline___b,1\n", encoding="utf-8")
            else:
                path.write_text(f"{name}:{role}\n", encoding="utf-8")
            files[role] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        datasets[name] = files
    plan = {
        "release_version": "9.9",
        "datasets": datasets,
        "eps": {
            "grid": [0.3, 0.5, 0.7],
            "minimum_dataset_f1": 0.0,
            "minimum_signature_weighted_f1": 0.0,
        },
    }
    path = tmp_path / "model_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_preflight_loads_test_free_arrow_plan(tmp_path: Path) -> None:
    plan_path = _write_model_plan(tmp_path)
    args = _args(tmp_path, model_plan=plan_path)

    plan = train_pairwise._preflight_pairwise(args)
    config = train_pairwise._training_config(args, plan, artifact_hashes={})

    assert plan.dataset_names == (*train_pairwise.DEFAULT_SOURCE_DATASET_NAMES, "augmented")
    assert plan.release_version == "9.9"
    assert plan.model_plan_sha256 == sha256_file(plan_path)
    assert plan.datasets["aminer"].split_mode == "random_blocks"
    assert plan.datasets["augmented"].split_mode == "fixed_pairs"
    assert config["model_plan_sha256"] == plan.model_plan_sha256
    assert "release_version" not in config
    assert all(
        isinstance(path, str) for dataset in config["dataset_inputs"].values() for path in dataset["files"].values()
    )
    assert config["featurization_backend"] == "rust_from_model_plan_arrow"
    assert "chunk_size" not in config
    for name, spec in config["dataset_inputs"].items():
        assert spec["arrow_dir"] == str((tmp_path / "planned_inputs" / name).resolve())
        assert set(spec) == {"arrow_dir", "files", "split_mode"}
    assert list(plan.matrix_work_dir.iterdir()) == []


@pytest.mark.parametrize(
    "dataset_names",
    [
        (*train_pairwise.DEFAULT_SOURCE_DATASET_NAMES, "inventors_s2and", "augmented"),
        (*train_pairwise.DEFAULT_SOURCE_DATASET_NAMES[1:], "patents", "augmented"),
        ("patents", "qian"),
    ],
    ids=["adds-inventors", "excludes-aminer", "without-augmentation"],
)
def test_preflight_uses_only_declared_datasets(tmp_path: Path, dataset_names: tuple[str, ...]) -> None:
    plan_path = _write_model_plan(tmp_path, dataset_names)
    args = _args(tmp_path, model_plan=plan_path)

    plan = train_pairwise._preflight_pairwise(args)
    config = train_pairwise._training_config(args, plan, artifact_hashes={})

    assert plan.dataset_names == dataset_names
    assert tuple(plan.arrow_dirs) == dataset_names
    assert config["source_dataset_names"] == list(dataset_names)


def test_preflight_requires_a_cluster_population_for_eps(tmp_path: Path) -> None:
    args = _args(tmp_path, model_plan=_write_model_plan(tmp_path, ("augmented",)))

    with pytest.raises(SystemExit, match="requires a random-block dataset for EPS calibration"):
        train_pairwise._preflight_pairwise(args)


def test_preflight_requires_fixed_augmented_pairs(tmp_path: Path) -> None:
    plan_path = _write_model_plan(tmp_path, ("patents", "augmented"))
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["datasets"]["augmented"] = payload["datasets"]["patents"]
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    args = _args(tmp_path, model_plan=plan_path)

    with pytest.raises(SystemExit, match="augmented dataset requires fixed train and validation pairs"):
        train_pairwise._preflight_pairwise(args)


def test_preflight_rejects_a_plan_without_published_arrow_data(tmp_path: Path) -> None:
    plan_path = _write_model_plan(tmp_path, ("qian",))
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    source = tmp_path / "qian_signatures.json"
    source.write_text("{}", encoding="utf-8")
    spec = {"path": str(source), "sha256": sha256_file(source)}
    payload["datasets"]["qian"] = {role: spec for role in ("signatures", "papers", "specter_embeddings", "clusters")}
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match=r"must name published Arrow data: \['qian'\]"):
        train_pairwise._preflight_pairwise(_args(tmp_path, model_plan=plan_path))


def test_training_without_augmented_data_reaches_artifact_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, model_plan=_write_model_plan(tmp_path, ("patents",)))

    def stop_at_authority(**_kwargs: Any) -> None:
        raise RuntimeError("reached artifact authority")

    monkeypatch.setattr(train_pairwise, "load_packaged_artifact_authority", stop_at_authority)
    with pytest.raises(RuntimeError, match="reached artifact authority"):
        train_pairwise.train_pairwise_bundle(args)


def test_training_rejects_source_mutation_before_loading_artifacts_or_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = _write_model_plan(tmp_path)
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    changed_file = Path(plan_payload["datasets"]["aminer"]["clusters"]["path"])
    changed_file.write_text("mutated after preflight\n", encoding="utf-8")
    args = _args(tmp_path, model_plan=plan_path)
    monkeypatch.setattr(
        train_pairwise,
        "load_packaged_artifact_authority",
        lambda **_kwargs: pytest.fail("artifact loading must not run"),
    )
    monkeypatch.setattr(
        train_pairwise, "load_development_dataset", lambda *_args, **_kwargs: pytest.fail("data must not load")
    )

    with pytest.raises(ValueError, match="model file digest mismatch"):
        train_pairwise.train_pairwise_bundle(args)


def test_augmented_dataset_contract_is_checked_before_featurization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = _write_model_plan(tmp_path)
    args = _args(tmp_path, model_plan=plan_path)
    full_plan = train_pairwise._preflight_pairwise(args)
    augmented_plan = train_pairwise.PairwisePreflightPlan(
        output_dir=full_plan.output_dir,
        release_version=full_plan.release_version,
        dataset_names=("augmented",),
        datasets={"augmented": full_plan.datasets["augmented"]},
        model_plan_sha256=full_plan.model_plan_sha256,
        model_plan_content_sha256=full_plan.model_plan_content_sha256,
        arrow_dirs={"augmented": full_plan.arrow_dirs["augmented"]},
        matrix_work_dir=full_plan.matrix_work_dir,
        matrix_work_free_bytes=full_plan.matrix_work_free_bytes,
        total_ram_bytes=full_plan.total_ram_bytes,
    )
    authority = _artifact_authority()
    fake_arrow = object()
    fake_dataset = SimpleNamespace(name_tuples=frozenset({("not", "canonical")}))
    loads: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    @contextmanager
    def open_arrow(dataset: Any) -> Iterator[object]:
        assert dataset is augmented_plan.datasets["augmented"]
        yield fake_arrow

    monkeypatch.setattr(train_pairwise, "_preflight_pairwise", lambda _args: augmented_plan)
    monkeypatch.setattr(train_pairwise, "load_packaged_artifact_authority", lambda **_kwargs: authority)
    monkeypatch.setattr(train_pairwise, "open_plan_arrow", open_arrow)
    monkeypatch.setattr(
        train_pairwise,
        "load_development_dataset",
        lambda *call_args, **kwargs: loads.append((call_args, kwargs)) or fake_dataset,
    )
    monkeypatch.setattr(
        train_pairwise,
        "_featurize_selection",
        lambda *_args, **_kwargs: pytest.fail("featurization ran before dataset contract validation"),
    )

    with pytest.raises(ValueError, match="dataset 'augmented' does not use the canonical name tuples"):
        train_pairwise.train_pairwise_bundle(args)
    # The same open Arrow dataset selects the pairs that it later featurizes.
    [(call_args, kwargs)] = loads
    assert call_args == ("augmented", augmented_plan.datasets["augmented"], fake_arrow)
    assert kwargs["random_seed"] == args.random_seed
    assert kwargs["train_pairs_size"] == args.train_pairs_size
    assert kwargs["val_pairs_size"] == args.validation_pairs_size


def test_preflight_rejects_unsafe_launch(tmp_path: Path) -> None:
    cases = (
        ("n-jobs", {"n_jobs": 0}, "--n-jobs must be positive"),
        ("total-ram", {"total_ram_bytes": 0}, "--total-ram-bytes must be positive"),
        ("run-full", {"run_full": False}, "requires --run-full"),
    )
    for case_id, change, message in cases:
        case_root = tmp_path / case_id
        case_root.mkdir()
        plan_path = _write_model_plan(case_root)
        args = _args(
            case_root,
            model_plan=plan_path,
        )
        for field, value in change.items():
            setattr(args, field, value)

        with pytest.raises(SystemExit, match=message):
            train_pairwise._preflight_pairwise(args)


def test_preflight_rejects_existing_output_and_nonempty_scratch(tmp_path: Path) -> None:
    plan_path = _write_model_plan(tmp_path)
    args = _args(
        tmp_path,
        model_plan=plan_path,
    )
    args.output_dir.mkdir()
    with pytest.raises(SystemExit, match="must name a new directory"):
        train_pairwise._preflight_pairwise(args)

    args.output_dir.rmdir()
    (args.matrix_work_dir / "stale.npy").write_bytes(b"stale")
    with pytest.raises(SystemExit, match="--matrix-work-dir must be empty"):
        train_pairwise._preflight_pairwise(args)


def test_artifact_authority_uses_packaged_runtime_data(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: dict[str, Path] = {}

    def open_orcid(root: Path) -> SimpleNamespace:
        opened["orcid"] = root
        return SimpleNamespace(
            name_tuples_sha256="a" * 64,
            data_sha256="b" * 64,
        )

    def open_name_counts(path: Path) -> SimpleNamespace:
        opened["name_counts"] = path
        return SimpleNamespace(
            manifest_sha256="d" * 64,
        )

    monkeypatch.setattr(
        production_training_contract.NameCountsIndex,
        "open",
        staticmethod(open_name_counts),
    )
    monkeypatch.setattr(
        production_training_contract,
        "load_packaged_name_tuple_artifact",
        lambda: SimpleNamespace(data_sha256="a" * 64),
    )
    monkeypatch.setattr(production_training_contract, "load_canonical_orcid_prefix_counts", open_orcid)

    authority = production_training_contract.load_packaged_artifact_authority(
        name_counts_index_root=Path("name-counts"),
    )

    assert train_pairwise._canonical_training_artifact_hashes(_artifact_authority()) == authority.hashes
    with pytest.raises(RuntimeError, match="different canonical name-tuple"):
        train_pairwise._canonical_training_artifact_hashes(_artifact_authority(tuple_hash="x" * 64))
    assert opened == {
        "name_counts": Path("name-counts"),
        "orcid": Path(production_training_contract._PACKAGE_DATA_DIR),
    }


def test_staging_keeps_only_train_validation_and_checks_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = (
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([1], dtype=np.int8),
        np.asarray([[2.0, 3.0]], dtype=np.float32),
    )
    paths = train_pairwise._stage_dataset_features(
        tmp_path,
        "toy",
        train=split,
        val=split,
    )
    union_path = train_pairwise._concatenate_staged_arrays(
        tmp_path / "union.npy",
        [paths["X_train"], paths["X_val"]],
    )
    np.testing.assert_array_equal(
        train_pairwise._load_staged_array(union_path),
        np.vstack([split[0], split[0]]),
    )

    monkeypatch.setattr(train_pairwise.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="Insufficient disk"):
        train_pairwise._stage_array(tmp_path / "too_large.npy", np.ones(10))


def test_selected_validation_roc_auc_must_be_finite() -> None:
    for value in (np.nan, np.inf):
        with pytest.raises(RuntimeError, match="must be finite"):
            train_pairwise._finite_validation_roc_auc(value)


class _StubSearch:
    def __init__(self, **kwargs: Any) -> None:
        self.init = kwargs
        self.classifier = "search-classifier"
        self.best_params = {"num_leaves": 7}

    def fit(self, *_args: Any) -> None:
        pass

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.full((len(X), 2), 0.5)


class _StubLightGBM:
    def __init__(self) -> None:
        self.params: dict[str, Any] = {}

    def set_params(self, **params: Any) -> _StubLightGBM:
        self.params = params
        return self

    def fit(self, *_args: Any) -> _StubLightGBM:
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.full((len(X), 2), 0.5)


@pytest.mark.parametrize(
    ("search_auc", "incumbent_auc", "selected"),
    [(0.8, 0.9, "incumbent"), (0.9, 0.9, "incumbent"), (0.9, 0.8, "search")],
    ids=["incumbent-better", "tie-keeps-incumbent", "search-better"],
)
def test_incumbent_parameters_compete_with_the_search_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    search_auc: float,
    incumbent_auc: float,
    selected: str,
) -> None:
    arrays = {}
    for role, values in (
        ("X_train", np.ones((4, 2))),
        ("y_train", np.array([0, 1, 0, 1])),
        ("X_val", np.ones((2, 2))),
        ("y_val", np.array([0, 1])),
    ):
        arrays[role] = tmp_path / f"{role}.npy"
        np.save(arrays[role], values)
    scores = iter([search_auc, incumbent_auc])
    built: list[_StubLightGBM] = []
    monkeypatch.setattr(train_pairwise, "PairwiseModeler", _StubSearch)
    monkeypatch.setattr(train_pairwise, "roc_auc_score", lambda *_args: next(scores))
    monkeypatch.setattr(train_pairwise, "LGBMClassifier", lambda: built.append(_StubLightGBM()) or built[-1])
    args = _args(tmp_path, n_iter=3, n_jobs=5)

    model, auc, selection = train_pairwise._fit_pairwise_model(
        args,
        arrays,
        feature_prefix="",
        monotone_constraints="1,0",
        incumbent={"num_leaves": 99, "n_jobs": 20, "random_state": 7},
    )

    [incumbent] = built
    assert incumbent.params == {"num_leaves": 99, "n_jobs": 5, "random_state": 42, "monotone_constraints": "1,0"}
    assert selection == {
        "selected_source": selected,
        "search_validation_roc_auc": search_auc,
        "incumbent_validation_roc_auc": incumbent_auc,
    }
    assert auc == max(search_auc, incumbent_auc)
    if selected == "incumbent":
        assert model.classifier is incumbent
        assert model.best_params == {"num_leaves": 99, "n_jobs": 20, "random_state": 7}
    else:
        assert model.classifier == "search-classifier"


def test_incumbent_parameters_require_named_heads(tmp_path: Path) -> None:
    path = tmp_path / "incumbent.json"
    for payload in ({}, {"other": {"num_leaves": 1}}, {"main": {}}, []):
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SystemExit, match="must map 'main' and/or 'nameless'"):
            train_pairwise._load_incumbent_parameters(path)
    shipped = Path(train_pairwise.__file__).with_name("pairwise_incumbent_h13.json")
    assert set(train_pairwise._load_incumbent_parameters(shipped)) == {"main", "nameless"}
    assert train_pairwise._load_incumbent_parameters(None) == {}
