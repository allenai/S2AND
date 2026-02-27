import json

import pytest

import s2and.feature_port as feature_port
import s2and.rust_capabilities as rust_capabilities


class DummyRustFeaturizer:
    created = []
    from_json_created = []
    signature_overlay_payloads = []

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    @classmethod
    def from_dataset(cls, dataset, _require_value, _disallow_value, _num_threads=None):
        cls.created.append(dataset.name)
        return cls(dataset.name)

    @classmethod
    def from_json_paths(cls, *_args, **_kwargs):
        cls.from_json_created.append((_args, _kwargs))
        return cls("json")

    def update_signature_name_counts(self, signatures):
        self.__class__.signature_overlay_payloads.append(signatures)
        return len(signatures)

    def featurize_pairs_matrix_indexed(self, *_args, **_kwargs):
        return []

    @classmethod
    def load(cls, _path):
        raise AssertionError("Disk cache path should not be used in this test")

    def update_cluster_seeds(self, _require_map, _disallow_set):
        return None


class DummyRustModule:
    __version__ = "0.31.0"
    RustFeaturizer = DummyRustFeaturizer


class DummyDataset:
    def __init__(self, name: str, mode: str = "train"):
        self.name = name
        self.mode = mode
        self.signatures = {}
        self.papers = {}
        self.name_tuples = {}
        self.compute_reference_features = False
        self.preprocess = True
        self.cluster_seeds_require = {}
        self.cluster_seeds_disallow = set()


def _cache_size() -> int:
    return len(list(feature_port._RUST_FEATURIZER_CACHE.keys()))


@pytest.fixture(autouse=True)
def _reset_feature_port_state(monkeypatch, tmp_path):
    feature_port.clear_rust_featurizer_cache()
    feature_port._NAME_COUNTS_NORMALIZATION_VERSION_CACHE.clear()
    feature_port._RUST_FEATURIZER_MAX_INMEM_CACHE = None
    DummyRustFeaturizer.created = []
    DummyRustFeaturizer.from_json_created = []
    DummyRustFeaturizer.signature_overlay_payloads = []
    monkeypatch.setattr(feature_port, "s2and_rust", DummyRustModule)
    monkeypatch.setattr(
        feature_port,
        "_rust_cache_path",
        lambda dataset: str(tmp_path / f"{dataset.name}_{id(dataset)}.bin"),
    )
    monkeypatch.delenv("S2AND_RUST_FEATURIZER_MAX_INMEM", raising=False)
    monkeypatch.delenv("S2AND_RUST_NAME_COUNTS_JSON", raising=False)
    yield
    feature_port.clear_rust_featurizer_cache()


def test_use_cache_true_keeps_unbounded_cache_in_train_mode():
    """Default MAX_INMEM=0 keeps all featurizers, matching Python CACHED_FEATURES."""
    d1 = DummyDataset("d1", mode="train")
    d2 = DummyDataset("d2", mode="train")

    feature_port._get_rust_featurizer(d1, use_cache=True)
    assert _cache_size() == 1

    feature_port._get_rust_featurizer(d2, use_cache=True)
    assert _cache_size() == 2
    assert feature_port._RUST_FEATURIZER_CACHE.get(d1) is not None
    assert feature_port._RUST_FEATURIZER_CACHE.get(d2) is not None

    # Re-access d1 — should be a cache hit, no rebuild.
    feature_port._get_rust_featurizer(d1, use_cache=True)
    assert DummyRustFeaturizer.created == ["d1", "d2"]


def test_max_inmem_env_uses_lru_eviction_policy(monkeypatch):
    monkeypatch.setenv("S2AND_RUST_FEATURIZER_MAX_INMEM", "2")
    d1 = DummyDataset("d1", mode="train")
    d2 = DummyDataset("d2", mode="train")
    d3 = DummyDataset("d3", mode="train")

    feature_port._get_rust_featurizer(d1, use_cache=True)
    feature_port._get_rust_featurizer(d2, use_cache=True)
    feature_port._get_rust_featurizer(d1, use_cache=True)  # Make d1 most-recently-used
    feature_port._get_rust_featurizer(d3, use_cache=True)

    assert _cache_size() == 2
    assert feature_port._RUST_FEATURIZER_CACHE.get(d1) is not None
    assert feature_port._RUST_FEATURIZER_CACHE.get(d2) is None
    assert feature_port._RUST_FEATURIZER_CACHE.get(d3) is not None


def test_use_cache_true_is_unbounded_in_inference_mode():
    d1 = DummyDataset("d1", mode="inference")
    d2 = DummyDataset("d2", mode="inference")
    d3 = DummyDataset("d3", mode="inference")

    feature_port._get_rust_featurizer(d1, use_cache=True)
    feature_port._get_rust_featurizer(d2, use_cache=True)
    feature_port._get_rust_featurizer(d3, use_cache=True)

    assert _cache_size() == 3


def test_use_cache_false_reuses_rust_featurizer_in_memory():
    dataset = DummyDataset("no_cache_dataset", mode="train")

    feature_port._get_rust_featurizer(dataset)
    feature_port._get_rust_featurizer(dataset)

    assert DummyRustFeaturizer.created == ["no_cache_dataset"]
    assert _cache_size() == 1


def test_use_cache_false_skips_disk_cache_lookup(monkeypatch):
    dataset = DummyDataset("no_cache_disk_dataset", mode="train")

    monkeypatch.setattr(
        feature_port,
        "_try_load_rust_featurizer_from_disk_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Disk cache lookup should be disabled")),
    )

    feature_port._get_rust_featurizer(dataset, use_cache=False)


def test_warm_rust_featurizer_persists_without_use_cache():
    dataset = DummyDataset("warm_dataset", mode="train")

    feature_port.warm_rust_featurizer(dataset, use_cache=False)
    feature_port._get_rust_featurizer(dataset, use_cache=False)

    assert DummyRustFeaturizer.created == ["warm_dataset"]
    assert _cache_size() == 1


def test_use_cache_true_checks_disk_cache_in_inference_mode(monkeypatch):
    dataset = DummyDataset("inference_cache_dataset", mode="inference")
    lookup_state = {"called": False}

    def _track_disk_lookup(*args, **kwargs):
        lookup_state["called"] = True
        return None, "skipped"

    monkeypatch.setattr(feature_port, "_try_load_rust_featurizer_from_disk_cache", _track_disk_lookup)
    feature_port._get_rust_featurizer(dataset, use_cache=True)

    assert lookup_state["called"] is True


def test_disk_cache_save_runs_outside_global_cache_lock(monkeypatch):
    dataset = DummyDataset("save_outside_lock", mode="train")
    save_calls: list[str] = []

    monkeypatch.setattr(
        feature_port,
        "_try_load_rust_featurizer_from_disk_cache",
        lambda *args, **kwargs: (None, "skipped"),
    )

    def _save_outside_lock(_featurizer, *, cache_path, cache_metadata):
        assert feature_port._RUST_FEATURIZER_CACHE_LOCK.locked() is False
        assert isinstance(cache_metadata, dict)
        save_calls.append(cache_path)

    monkeypatch.setattr(feature_port, "_save_rust_featurizer_cache_best_effort", _save_outside_lock)

    feature_port._get_rust_featurizer(dataset, use_cache=True)

    assert len(save_calls) == 1


def test_disk_cache_metadata_mismatch_skips_load(monkeypatch, tmp_path):
    dataset = DummyDataset("cache_meta_miss", mode="train")
    cache_path = str(tmp_path / "cache_meta_miss.bin")
    metadata_path = feature_port._rust_cache_metadata_path(cache_path)

    monkeypatch.setattr(feature_port, "_rust_cache_path", lambda _dataset: cache_path)
    monkeypatch.setattr(
        feature_port,
        "_save_rust_featurizer_cache_best_effort",
        lambda *_args, **_kwargs: None,
    )

    # Simulate an old/stale cache artifact with non-matching metadata.
    with open(cache_path, "wb") as cache_file:
        cache_file.write(b"stale")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump({"schema_version": 1, "dataset_name": "different_dataset"}, metadata_file)

    def _unexpected_load(_path):
        raise AssertionError("RustFeaturizer.load should not run when metadata mismatches")

    monkeypatch.setattr(DummyRustFeaturizer, "load", staticmethod(_unexpected_load))
    feature_port._get_rust_featurizer(dataset, use_cache=True)

    assert DummyRustFeaturizer.created == ["cache_meta_miss"]


def test_disk_cache_metadata_match_loads_without_rebuild(monkeypatch, tmp_path):
    dataset = DummyDataset("cache_meta_hit", mode="train")
    cache_path = str(tmp_path / "cache_meta_hit.bin")
    metadata_path = feature_port._rust_cache_metadata_path(cache_path)

    monkeypatch.setattr(feature_port, "_rust_cache_path", lambda _dataset: cache_path)
    monkeypatch.setattr(
        feature_port,
        "_save_rust_featurizer_cache_best_effort",
        lambda *_args, **_kwargs: None,
    )

    with open(cache_path, "wb") as cache_file:
        cache_file.write(b"fresh")
    expected_metadata = feature_port._rust_featurizer_cache_metadata(dataset, requested_build_path="from_dataset")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(expected_metadata, metadata_file, sort_keys=True, separators=(",", ":"))

    loaded = DummyRustFeaturizer("loaded")
    load_calls: list[str] = []

    def _load(path):
        load_calls.append(path)
        return loaded

    monkeypatch.setattr(DummyRustFeaturizer, "load", staticmethod(_load))
    featurizer = feature_port._get_rust_featurizer(dataset, use_cache=True)

    assert featurizer is loaded
    assert load_calls == [cache_path]
    assert DummyRustFeaturizer.created == []


def test_json_ingest_is_inference_only(monkeypatch):
    dataset = DummyDataset("train_dataset", mode="train")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"

    feature_port._get_rust_featurizer(dataset)

    assert DummyRustFeaturizer.created == ["train_dataset"]
    assert DummyRustFeaturizer.from_json_created == []


def test_inference_without_json_paths_uses_from_dataset():
    dataset = DummyDataset("inference_no_paths", mode="inference")

    feature_port._get_rust_featurizer(dataset)

    assert DummyRustFeaturizer.created == ["inference_no_paths"]
    assert DummyRustFeaturizer.from_json_created == []


def test_json_ingest_routes_canonical_payload(monkeypatch):
    monkeypatch.delenv("S2AND_RUST_NAME_COUNTS_JSON", raising=False)
    dataset = DummyDataset("inference_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"
    dataset.clusters_path = "clusters.json"
    dataset.cluster_seeds_path = "cluster_seeds.json"
    dataset.specter_embeddings_path = "specter.pkl"
    dataset.compute_reference_features = True
    dataset.preprocess = False
    dataset.n_jobs = 8

    feature_port._get_rust_featurizer(dataset)

    assert DummyRustFeaturizer.created == []
    assert len(DummyRustFeaturizer.from_json_created) == 1
    args, kwargs = DummyRustFeaturizer.from_json_created[0]
    assert kwargs == {}
    assert args == (
        "signatures.json",
        "papers.json",
        "clusters.json",
        "cluster_seeds.json",
        "specter.pkl",
        None,
        None,
        False,
        True,
        feature_port.CLUSTER_SEEDS_LOOKUP["require"],
        feature_port.CLUSTER_SEEDS_LOOKUP["disallow"],
        8,
    )


def test_featurizer_telemetry_logs_runtime_callsite(caplog):
    dataset = DummyDataset("telemetry_dataset", mode="train")
    runtime_context = type("RuntimeContext", (), {"operation": "constraints", "run_id": "run-123"})()

    with caplog.at_level("INFO", logger="s2and"):
        feature_port._get_rust_featurizer(dataset, runtime_context=runtime_context, use_cache=True)
        feature_port._get_rust_featurizer(dataset, runtime_context=runtime_context, use_cache=True)

    logs = "\n".join(caplog.messages)
    assert "rust_featurizer_cache cache=miss" in logs
    assert "rust_featurizer_cache cache=hit" in logs
    assert "rust_core_build seconds=" in logs
    assert "op=constraints" in logs
    assert "run=run-123" in logs
    assert "path=from_dataset" in logs


def test_featurizer_telemetry_logs_json_build_path(caplog, monkeypatch):
    dataset = DummyDataset("telemetry_json_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"

    with caplog.at_level("INFO", logger="s2and"):
        feature_port._get_rust_featurizer(dataset)

    logs = "\n".join(caplog.messages)
    assert "rust_core_build seconds=" in logs
    assert "path=from_json_paths" in logs


def test_json_ingest_overlay_payload_includes_only_signatures_with_name_counts(monkeypatch):
    dataset = DummyDataset("inference_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"
    dataset.signatures = {
        "s1": type("Sig", (), {"author_info_name_counts": (1.0, 2.0, 3.0, 4.0), "extra": "full_signature"})(),
        "s2": type("Sig", (), {"author_info_name_counts": None, "extra": "ignored"})(),
    }

    feature_port._get_rust_featurizer(dataset)

    assert len(DummyRustFeaturizer.signature_overlay_payloads) == 1
    payload = DummyRustFeaturizer.signature_overlay_payloads[0]
    assert list(payload.keys()) == ["s1"]
    payload_entry = payload["s1"]
    assert hasattr(payload_entry, "author_info_name_counts")
    assert payload_entry.author_info_name_counts == (1.0, 2.0, 3.0, 4.0)
    assert payload_entry.extra == "full_signature"


def test_json_ingest_source_telemetry_prefers_dataset_over_artifact(caplog, monkeypatch):
    monkeypatch.setenv("S2AND_RUST_NAME_COUNTS_JSON", "name_counts.json")
    dataset = DummyDataset("telemetry_json_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"
    dataset.signatures = {
        "s1": type("Sig", (), {"author_info_name_counts": (1.0, 2.0, 3.0, 4.0)})(),
        "s2": type("Sig", (), {"author_info_name_counts": None})(),
    }

    with caplog.at_level("INFO", logger="s2and"):
        feature_port._get_rust_featurizer(dataset)

    logs = "\n".join(caplog.messages)
    assert "stage=rust_json_ingest_name_counts_source" in logs
    assert "name_counts_source=dataset" in logs
    assert "signatures_total=2" in logs
    assert "signatures_with_counts=1" in logs
    assert "artifact_configured=True" in logs

    args, _kwargs = DummyRustFeaturizer.from_json_created[0]
    assert args[6] is None


def test_json_ingest_source_telemetry_uses_artifact_when_non_minimal(tmp_path, caplog, monkeypatch):
    artifact_path = tmp_path / "name_counts.json"
    artifact_path.write_text('{"normalization_version":"legacy_compat","counts":{}}', encoding="utf-8")

    monkeypatch.setenv("S2AND_RUST_NAME_COUNTS_JSON", str(artifact_path))
    dataset = DummyDataset("telemetry_json_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"
    dataset.signatures = {"s1": type("Sig", (), {"author_info_name_counts": None})()}

    with caplog.at_level("INFO", logger="s2and"):
        feature_port._get_rust_featurizer(dataset)

    logs = "\n".join(caplog.messages)
    assert "stage=rust_json_ingest_name_counts_source" in logs
    assert "name_counts_source=artifact" in logs
    assert "normalization_check_executed=True" in logs

    args, _kwargs = DummyRustFeaturizer.from_json_created[0]
    assert args[6] == str(artifact_path)


def test_explicit_evict_and_clear_api(monkeypatch):
    d1 = DummyDataset("d1", mode="train")
    d2 = DummyDataset("d2", mode="train")

    # Disable cap in this test so we can keep two entries at once.
    monkeypatch.setenv("S2AND_RUST_FEATURIZER_MAX_INMEM", "0")
    feature_port._get_rust_featurizer(d1, use_cache=True)
    feature_port._get_rust_featurizer(d2, use_cache=True)
    assert _cache_size() == 2

    assert feature_port.evict_rust_featurizer(d1) is True
    assert feature_port.evict_rust_featurizer(d1) is False
    assert _cache_size() == 1

    cleared = feature_port.clear_rust_featurizer_cache()
    assert cleared == 1
    assert _cache_size() == 0


def test_load_s2and_rust_extension_falls_back_from_namespace_package(monkeypatch):
    class NamespaceOnly:
        pass

    class NativeExtension:
        RustFeaturizer = object()

    def fake_import_module(name: str):
        if name == "s2and_rust":
            return NamespaceOnly()
        if name == "s2and_rust._s2and_rust":
            return NativeExtension
        raise ImportError(name)

    monkeypatch.setattr(rust_capabilities.importlib, "import_module", fake_import_module)

    loaded = rust_capabilities.load_s2and_rust_extension()
    assert loaded is NativeExtension


def test_load_s2and_rust_extension_returns_first_valid_module(monkeypatch):
    class ValidRustFeaturizer:
        @staticmethod
        def from_dataset(*args, **kwargs):
            return None

        @staticmethod
        def from_json_paths(*args, **kwargs):
            return None

        def signature_ids(self):
            return []

    class TopLevelModule:
        RustFeaturizer = ValidRustFeaturizer

    def fake_import_module(name: str):
        if name == "s2and_rust":
            return TopLevelModule
        raise ImportError(name)

    monkeypatch.setattr(rust_capabilities.importlib, "import_module", fake_import_module)

    loaded = rust_capabilities.load_s2and_rust_extension()
    assert loaded is TopLevelModule
