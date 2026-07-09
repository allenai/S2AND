"""
Note: This script won't run outside internal infrastructure because it relies on an
internal Semantic Scholar package called pys2; it documents (and, on internal infra,
performs) how the name count features are built.

canonical_v2 rewrite (docs/normalization_migration_blocked.md, migration step 3):
counts are keyed by the canonical fields from s2and.text.canonicalize_name_parts /
canonical_name_count_keys — spaced surnames with particles preserved, dash-bound
given-name compounds kept together as spaced tokens, apostrophe-like marks deleted,
no compact-join shims. Keys with a missing or uninformative component are skipped
(the lookup side returns NaN for them; sentinel counts are never stored).

The pickle payload stays a 4-tuple of dicts for loader compatibility; provenance
(normalization_version, source snapshot, date, cardinalities) goes in a JSON sidecar
that the name_counts_index/ manifest must carry forward as `normalization_version`.
"""

import datetime
import json
import pickle
from collections import Counter

from pys2 import _evaluate_redshift_query  # type: ignore

from s2and.text import canonical_name_count_keys, canonicalize_name_parts

NORMALIZATION_VERSION = "canonical_v2"

# this queries our internal databases
query = """
    select concat(concat(nvl(first_name, ''), '|||'), nvl(last_name, '')), count(*)
    from content.authors
    group by concat(concat(nvl(first_name, ''), '|||'), nvl(last_name, ''))
"""
first_last_count = _evaluate_redshift_query(query)

first_counter: Counter = Counter()
last_counter: Counter = Counter()
first_last_counter: Counter = Counter()
last_first_initial_counter: Counter = Counter()

for raw_concat, count in zip(first_last_count["concat"], first_last_count["count"], strict=False):
    raw_first, _, raw_last = raw_concat.partition("|||")
    # The corpus rows carry no middle field; middle spill is irrelevant for counts.
    parts = canonicalize_name_parts(raw_first, None, raw_last)
    keys = canonical_name_count_keys(parts)
    if keys["first"] is not None:
        first_counter[keys["first"]] += count
    if keys["last"] is not None:
        last_counter[keys["last"]] += count
    if keys["first_last"] is not None:
        first_last_counter[keys["first_last"]] += count
    if keys["last_first_initial"] is not None:
        last_first_initial_counter[keys["last_first_initial"]] += count

# save space by filtering out anything with count = 1 as we can get that by default
first_dict = {key: value for key, value in first_counter.items() if value > 1}
last_dict = {key: value for key, value in last_counter.items() if value > 1}
first_last_dict = {key: value for key, value in first_last_counter.items() if value > 1}
last_first_initial_dict = {key: value for key, value in last_first_initial_counter.items() if value > 1}

# this ends up in S3
with open("name_counts.pickle", "wb") as f:
    pickle.dump(
        (first_dict, last_dict, first_last_dict, last_first_initial_dict),
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

# Generation provenance + integrity stats (migration exit criteria: counts, key
# cardinalities, and basic spot checks must be logged and shipped with the artifact).
metadata = {
    "normalization_version": NORMALIZATION_VERSION,
    "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "source": "content.authors (Redshift)",
    "cardinalities": {
        "first": len(first_dict),
        "last": len(last_dict),
        "first_last": len(first_last_dict),
        "last_first_initial": len(last_first_initial_dict),
    },
    "total_mass": {
        "first": sum(first_dict.values()),
        "last": sum(last_dict.values()),
        "first_last": sum(first_last_dict.values()),
        "last_first_initial": sum(last_first_initial_dict.values()),
    },
}
with open("name_counts.meta.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(json.dumps(metadata, indent=2))

# Spot checks: canonical policy must be visible in the shipped keys.
for spot_key, spot_dict in [("ou yang", last_dict), ("van der berg", last_dict)]:
    print(f"spot check last[{spot_key!r}] present: {spot_key in spot_dict}")
for compact_key in ("ouyang", "vanderberg"):
    print(f"joined variant last[{compact_key!r}] present (expected, as its own spelling): {compact_key in last_dict}")
