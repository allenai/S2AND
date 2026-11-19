# Known clustering problems after 1.0.0b12

This page lists what the 1.0.0b12 work on wrongly split and wrongly joined profiles did not fix.
The numbers come from 128 production name blocks run through the full production route with
model 1.3.2 (968,353 mentions, 69,885 clusters before the year-gap split), and from blind two-judge
reviews of sampled changes. The judges are LLM reviewers, not curators, so treat small counts as
estimates.

## Wrong joins across decades

The pair model sees the years between two papers capped at 50 (`YEAR_DIFF_CAP`), and few training
pairs predate 1950. Topic overlap can therefore put a 1928 paper into a profile that starts in 2000.
1.0.0b12 splits a cluster at an empty stretch of more than 50 years unless a certain close pair, an
ORCID or a required seed group bridges it (see "Name-conflict repair" in
[production_inference.md](production_inference.md)). That cut clusters spanning 70 or more years from
1,215 to 785. What remains:

- **A close pair bridges the gap: about 360 clusters.** The split keeps them whole. In the judged
  sample these were 13 wrong joins to 7 right ones (reprints, re-dated records).
- **No single gap over 50 years: about 410 clusters.** Papers chain across the decades, for example
  1900, 1940, 1985, 2020. In the judged sample 8 of 10 were not one person. A year rule cannot separate
  them without capping a career's span, which split correct long careers in testing.
- **Wrong splits by the year-gap step: about 1 in 10.** Both judges called 4 of 40 sampled splits
  wrong. Each moved one or two mentions: a reprint or a record with the wrong year that the model was
  not certain of (an 1818 paper reprinted in 1931, a 1930s patent dated 2017).
- **Blocks under 11 mentions** are not repaired. The year-gap split runs on them. In two samples
  of production profiles with such a gap (259 run), judged blind to S2AND's output, the split cut
  183: both judges called 131 of them two people or a mixture, 30 one person, and 22 were unsure
  or split. In the 200-profile
  sample every wrong cut moved one mention, 21 of 23 dated 2000 or later (mostly reprints and
  digitized old works). The first sample had two larger wrong cuts (3 of 8 and 1 of 10 mentions).

A lasting fix belongs in the model: lift the year cap and add old labeled pairs, or add a
cluster-level year signal. A year-gap veto inside linkage was tested and rejected: it changed
125,000 mentions and split 2,208 clusters, many of them correct long careers.

## Profiles still split

Of the 41 production profiles with at least 1,000 mentions in the 128 blocks, 24 still span more than
one large cluster (32 before 1.0.0b12). The old profiles are not truth: many of the remaining splits
separate real namesakes, such as physicists with the same initials. The known wrong splits:

- **Spelling variants of a given name** (Hahn: Kristan and Kristian). The name rules call two different
  full given names a conflict. No rule here links a one-letter variant, and no independent byline
  confirms one. Fixing it needs an alias entry or a variant rule with its own cutoff; a "form-pair"
  rule with a 0.12 distance cutoff was rejected as fragile.
- **Nicknames** (McKenna: Jim and Janis A.). "Jim" spells nothing in the other cluster, so neither the
  joined-initials reading nor the repair applies.
- **Initials-only sides** (Lyons, Evans). One side has no full first name, which the repair requires.
- **Lipton.** Still split; the cause was not traced.

## Wrong joins from the name-rule changes

- **Joined-initials reading.** The rule reads a short given name as initials when another name spells
  them ("Ka" for Kristan Allan). Small clusters can then merge with namesakes whose initials match:
  in judging, 29 wrongly joined mentions against 1,064 rightly joined ones.
- **The repair's reading step.** Both judges called 16 of 19 sampled merges right. The 3 wrong ones
  moved 2 to 4 mentions each. The final code no longer makes one of them (F. M. Jones); the other
  two (K. M. E. Turner, P. E. M. Phillips) remain.
- **Knock-on splits.** One reading mention, such as "Ec", vetoes every bare "Edward" in its cluster.
  Before the reading step, judges found these splits wrong for 147 mentions and right for 55. The
  reading step rejoins the pieces when either cluster spells the reading; pieces under the
  repair's 5-mention minimum stay apart when other conflicts remain. The remaining count was
  not re-measured.
