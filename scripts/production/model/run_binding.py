"""Content-based identity helpers for one prepared production release run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from s2and._sha256 import is_lowercase_sha256, sha256_file
from s2and.incremental_linking.contracts import canonical_json_digest

_RUN_BINDING_IDENTITY_FIELDS = frozenset(
    {
        "baseline_record_sha256",
        "candidate_model_manifest_sha256",
        "evaluation_plan_content_sha256",
        "model_plan_content_sha256",
        "public_data_root_manifest_sha256",
    }
)
_RUN_BINDING_FIELDS = _RUN_BINDING_IDENTITY_FIELDS | {"run_binding_sha256"}


def _object(value: Any, *, label: str, keys: set[str] | frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")
    return cast(Mapping[str, Any], value)


def _project_file_specs(value: Any, *, label: str) -> Any:
    """Replace absolute file locations with their content digests."""

    if isinstance(value, Mapping):
        if set(value) == {"path", "sha256"}:
            spec = _object(value, label=label, keys={"path", "sha256"})
            path, digest = spec["path"], spec["sha256"]
            if not isinstance(path, str) or not Path(path).is_absolute() or not is_lowercase_sha256(digest):
                raise ValueError(f"{label} must contain an absolute path and lowercase SHA-256")
            return str(digest)
        return {str(key): _project_file_specs(item, label=f"{label}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [_project_file_specs(item, label=f"{label}[]") for item in value]
    return value


def model_plan_content_identity(payload: Any) -> dict[str, Any]:
    """Return the path-independent content and policy identity of a model plan."""

    plan = _object(payload, label="model plan", keys={"release_version", "datasets", "eps"})
    return cast(dict[str, Any], _project_file_specs(plan, label="model plan"))


def evaluation_plan_content_identity(payload: Any) -> dict[str, Any]:
    """Return the path-independent content and policy identity of an evaluation plan."""

    plan = _object(
        payload,
        label="evaluation plan",
        keys={
            "baseline_record_sha256",
            "baselines",
            "cluster",
            "gates",
            "parity",
            "pairwise",
            "performance",
            "subblocking",
        },
    )
    identity = cast(
        dict[str, Any],
        _project_file_specs(plan, label="evaluation plan"),
    )
    baseline_digest = identity["baseline_record_sha256"]
    if not is_lowercase_sha256(baseline_digest):
        raise ValueError("evaluation plan baseline_record_sha256 must be lowercase SHA-256")
    performance = dict(
        _object(
            identity["performance"],
            label="evaluation plan performance",
            keys={"arrow_root", "arrow_root_manifest_sha256", "workload"},
        )
    )
    arrow_root = performance.pop("arrow_root")
    if not isinstance(arrow_root, str) or not Path(arrow_root).is_absolute():
        raise ValueError("evaluation plan performance arrow_root must be absolute")
    if not is_lowercase_sha256(performance["arrow_root_manifest_sha256"]):
        raise ValueError("evaluation plan performance root identity must be lowercase SHA-256")
    parity = dict(
        _object(
            identity["parity"],
            label="evaluation plan parity",
            keys={"block", "dataset", "files", "fixture_dir", "workload"},
        )
    )
    fixture_dir = parity.pop("fixture_dir")
    if not isinstance(fixture_dir, str) or not Path(fixture_dir).is_absolute():
        raise ValueError("evaluation plan parity fixture_dir must be absolute")
    subblocking = dict(
        _object(
            identity["subblocking"],
            label="evaluation plan subblocking",
            keys={"component_members", "dataset", "workload"},
        )
    )
    dataset = subblocking["dataset"]
    if not isinstance(dataset, str) or not dataset or dataset.strip() != dataset:
        raise ValueError("evaluation plan subblocking dataset must be a nonempty trimmed string")
    subblocking["component_members_sha256"] = subblocking.pop("component_members")
    identity["performance"] = performance
    identity["parity"] = parity
    identity["subblocking"] = subblocking
    return identity


def build_run_binding_identity(
    *,
    model_plan: Path,
    evaluation_plan: Path,
    candidate_model_dir: Path,
    public_data_root: Path,
) -> dict[str, str]:
    """Build the five immutable identities shared by final acceptance reports."""

    evaluation_payload = json.loads(Path(evaluation_plan).read_text(encoding="utf-8"))
    evaluation_identity = evaluation_plan_content_identity(evaluation_payload)
    candidate_manifest = Path(candidate_model_dir) / "manifest.json"
    public_manifest = Path(public_data_root) / "manifest.json"
    return {
        "baseline_record_sha256": str(evaluation_identity["baseline_record_sha256"]),
        "candidate_model_manifest_sha256": sha256_file(candidate_manifest),
        "evaluation_plan_content_sha256": canonical_json_digest(evaluation_identity),
        "model_plan_content_sha256": canonical_json_digest(
            model_plan_content_identity(json.loads(Path(model_plan).read_text(encoding="utf-8")))
        ),
        "public_data_root_manifest_sha256": sha256_file(public_manifest),
    }


def build_run_binding_payload(identity: Mapping[str, Any]) -> dict[str, str]:
    """Validate one binding identity and attach its canonical digest."""

    normalized = _object(identity, label="run binding identity", keys=_RUN_BINDING_IDENTITY_FIELDS)
    result = {field: str(normalized[field]) for field in sorted(_RUN_BINDING_IDENTITY_FIELDS)}
    for field, digest in result.items():
        if not is_lowercase_sha256(digest):
            raise ValueError(f"Run binding {field} must be lowercase SHA-256")
    return {**result, "run_binding_sha256": canonical_json_digest(result)}


def load_run_binding(path: Path) -> dict[str, str]:
    """Load one strict run-binding file and verify its self-digest."""

    payload = _object(
        json.loads(Path(path).read_text(encoding="utf-8")),
        label="run binding",
        keys=_RUN_BINDING_FIELDS,
    )
    identity = {field: payload[field] for field in _RUN_BINDING_IDENTITY_FIELDS}
    expected = build_run_binding_payload(identity)
    if payload["run_binding_sha256"] != expected["run_binding_sha256"]:
        raise ValueError("run binding self-digest mismatch")
    return expected


def require_run_binding_matches(
    binding: Mapping[str, str],
    *,
    evaluation_plan: Path | None = None,
    candidate_model_dir: Path | None = None,
    public_data_root: Path | None = None,
) -> None:
    """Reject a run binding that does not identify the supplied final inputs."""

    if evaluation_plan is not None:
        evaluation_payload = json.loads(Path(evaluation_plan).read_text(encoding="utf-8"))
        evaluation_identity = evaluation_plan_content_identity(evaluation_payload)
        observed = canonical_json_digest(evaluation_identity)
        if observed != binding["evaluation_plan_content_sha256"]:
            raise ValueError("run binding does not match the evaluation plan")
        if evaluation_identity["baseline_record_sha256"] != binding["baseline_record_sha256"]:
            raise ValueError("run binding baseline record does not match the evaluation plan")
    if candidate_model_dir is not None:
        observed = sha256_file(Path(candidate_model_dir) / "manifest.json")
        if observed != binding["candidate_model_manifest_sha256"]:
            raise ValueError("run binding does not match the candidate model manifest")
    if public_data_root is not None:
        observed = sha256_file(Path(public_data_root) / "manifest.json")
        if observed != binding["public_data_root_manifest_sha256"]:
            raise ValueError("run binding does not match the public-data root manifest")
