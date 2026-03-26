from types import SimpleNamespace

import numpy as np

import s2and.subblocking as subblocking
from s2and.data import Signature


def _signature(signature_id: str, *, first: str, middle: str | None = None, orcid: str | None = None) -> Signature:
    return Signature(
        author_info_first=first,
        author_info_first_normalized_without_apostrophe=first,
        author_info_middle=middle,
        author_info_middle_normalized_without_apostrophe=middle or "",
        author_info_last_normalized="wang",
        author_info_last="Wang",
        author_info_suffix_normalized=None,
        author_info_suffix=None,
        author_info_first_normalized=first,
        author_info_coauthors=None,
        author_info_coauthor_blocks=None,
        author_info_full_name=None,
        author_info_affiliations=[],
        author_info_affiliations_n_grams=None,
        author_info_coauthor_n_grams=None,
        author_info_email=None,
        author_info_orcid=orcid,
        author_info_name_counts=None,
        author_info_position=0,
        author_info_block="h wang",
        author_info_given_block=None,
        author_info_estimated_gender=None,
        author_info_estimated_ethnicity=None,
        paper_id=int(signature_id[1:]) if signature_id[1:].isdigit() else 0,
        sourced_author_source=None,
        sourced_author_ids=[],
        author_id=None,
        signature_id=signature_id,
    )


def test_make_subblocks_with_telemetry_reports_specter_invocations(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="h", middle=""),
            "s2": _signature("s2", first="h", middle=""),
            "s3": _signature("s3", first="h", middle=""),
            "s4": _signature("s4", first="h", middle=""),
        },
        random_seed=0,
    )

    monkeypatch.setattr(
        subblocking,
        "cluster_with_specter",
        lambda signature_ids, anddata, target_subblock_size: {
            "0": list(signature_ids[:2]),
            "1": list(signature_ids[2:]),
        },
    )

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4"],
        dataset,
        maximum_size=2,
        first_k_letter_counts_sorted={},
    )

    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [["s1", "s2"], ["s3", "s4"]]
    assert telemetry["single_letter_first_name_signature_count"] == 4
    assert telemetry["specter_fallback_candidate_block_count"] == 1
    assert telemetry["specter_invocation_count"] == 1
    assert telemetry["specter_input_signature_count"] == 4
    assert telemetry["final_specter_labeled_subblock_count"] == 2
    assert telemetry["final_specter_labeled_signature_count"] == 4


def test_make_subblocks_with_telemetry_distinguishes_non_invoked_specter_candidates(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="h", middle=""),
            "s2": _signature("s2", first="h", middle=""),
        },
        random_seed=0,
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("cluster_with_specter should not be called when the dead-end block is already in budget")

    monkeypatch.setattr(subblocking, "cluster_with_specter", _fail_if_called)

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2"],
        dataset,
        maximum_size=2,
        first_k_letter_counts_sorted={},
    )

    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [["s1", "s2"]]
    assert telemetry["specter_fallback_candidate_block_count"] == 1
    assert telemetry["specter_non_invoked_candidate_block_count"] == 1
    assert telemetry["specter_non_invoked_candidate_signature_count"] == 2
    assert telemetry["specter_invocation_count"] == 0
    assert telemetry["final_specter_labeled_subblock_count"] == 0
    assert telemetry["final_specter_labeled_signature_count"] == 0


def test_make_subblocks_with_telemetry_counts_orcid_merges_skipped_for_capacity(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="aa", middle="", orcid="O1"),
            "s2": _signature("s2", first="aa", middle="", orcid="O2"),
            "s3": _signature("s3", first="bb", middle="", orcid="O1"),
            "s4": _signature("s4", first="bb", middle=""),
            "s5": _signature("s5", first="cc", middle="", orcid="O2"),
            "s6": _signature("s6", first="cc", middle=""),
        },
        random_seed=0,
    )

    call_count = {"value": 0}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        del names, sig_ids, maximum_size, starting_k
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "a": np.array(["s1", "s2"]),
                "b": np.array(["s3", "s4"]),
                "c": np.array(["s5", "s6"]),
            }, {}
        raise AssertionError("Unexpected extra call to subdivide_helper")

    def fail_if_specter_called(*_args, **_kwargs):
        raise AssertionError("cluster_with_specter should not be called in this regression test")

    monkeypatch.setattr(subblocking, "subdivide_helper", fake_subdivide_helper)
    monkeypatch.setattr(subblocking, "cluster_with_specter", fail_if_specter_called)

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4", "s5", "s6"],
        dataset,
        maximum_size=2,
        first_k_letter_counts_sorted={},
    )

    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [
        ["s1", "s2"],
        ["s3", "s4"],
        ["s5", "s6"],
    ]
    assert telemetry["orcid_merge_skipped_due_to_capacity_count"] == 2
    assert telemetry["orcid_merge_skipped_due_to_capacity_signature_count"] == 4
