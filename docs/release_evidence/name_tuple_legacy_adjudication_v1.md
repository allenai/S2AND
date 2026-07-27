# Legacy name-tuple adjudication v1

This evidence records a manual review of every canonical pair present in
`s2and/data/s2and_name_tuples_filtered.txt` but absent from the pre-review
`s2and/data/s2and_name_tuples_canonical.txt`.

## Result

- Candidate pairs reviewed: 2,266
- Accepted: 1,343
- Rejected: 906
- Uncertain and not promoted: 17
- Reviewer: Codex, record-by-record manual adjudication
- Annotation JSONL SHA-256:
  `e0148fd715569f95304951c883696fccaf1b1ed57b0dbb8fdf770d8e372b65e5`
- Deterministic gzip SHA-256:
  `4af80417dba07720b115545468620c14a8df94889ad1a7db0e486108f331887b`

The complete ordered ledger, including stable record ID, source index, pair,
label, confidence, and reason, is
`name_tuple_legacy_adjudication_v1.jsonl.gz`. Its source indices cover
`0..2265` exactly, its IDs and pair keys are unique, and its source fields
exactly match the sorted legacy-only candidate set.

## Rubric

- `accept`: credibly the same personal name under a typographical,
  orthographic, or established transliteration variation.
- `reject`: distinct names, an initials/non-name fragment, or resemblance too
  weak to support a global alias relationship.
- `uncertain`: the two forms alone cannot distinguish a typo from a meaningful
  name difference. Uncertain pairs are not promoted.

The review did not use heuristic or model-generated labels. Scripts only
assembled fixed packets, joined manually authored decisions, and validated
coverage and invariants.

## Uncertain pairs

1. `abdulnaser,abdulnasr`
2. `adi,ah`
3. `ah,an`
4. `andu,andy`
5. `angi,anqi`
6. `bong,bonk`
7. `chai,chia`
8. `check,cheuk`
9. `chia,chih`
10. `dae,dai`
11. `deog,dong`
12. `heeran,hyiran`
13. `iii,ill`
14. `lei,lie`
15. `liat,lyt`
16. `nira,nyrh`
17. `yuliia,yuliiy`

## Promotion

The accepted canonical unordered rows are appended once to the existing
curated source; reverse-direction duplicates are not added. The existing
generator then produces the exact union of the prior canonical artifact and
the 1,343 accepted pairs.

Expected promoted identities and counts:

- Curated source: 11,268 rows, 179,210 bytes,
  SHA-256 `d9ae444af7f06a482b110d4f5a0ad39a1b48ace6a2362cdbb62554061c57d4d5`
- Canonical artifact: 5,027 pairs, 80,298 bytes,
  SHA-256 `b21638351149389c57eca547b0f79c80084e56ad273f31e778cb1db1866945a8`
- All 3,684 prior canonical pairs retained
- Exactly 1,343 accepted pairs added
- Generator drops unchanged: 1,385 identity, 1,088 prefix-compatible,
  3,768 duplicate-canonical, and 0 empty
