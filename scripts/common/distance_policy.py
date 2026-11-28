"""Version the EPS distance policy that EPS curves and linker task tables record."""

# Bump when a distance rule changes within the same package version. Distance caches and
# sweep plans record it, and release EPS review rejects curves from another policy.
# 1: version field added; 2: CJK given-name veto (same-script, complete names only);
# 3: joined-initials reading of short given names; 4: joined-capitals middles read as
# initials, and the repair's reading step; 5: year-gap split after the repair; 6: the
# split also reads blocks without a name conflict, and a joined-capitals last middle word
# after other middle words reads as initials; 7: the reading step counts a speller in the
# reading mention's own cluster; 8: the year-gap split runs on blocks under 11 mentions;
# 9: the split treats 1601 and 1800 as undated and, in blocks under 11 mentions, keeps parts
# that share a first name; 10: the bridge counts the surname block, not the clustered unit,
# and a 1601 or 1800 stamp follows the part whose years span it; 11: the repair's
# three-cluster extension is removed (later rounds join the third cluster);
# 12–13: guarded ORCID joins within blocks and across Arrow subblocks.
# 14: treat corpus placeholder years as missing when computing features and ranking candidates.
# 15: apply the pairwise bundle year contract when building EPS distance matrices.
DISTANCE_POLICY_VERSION = 15
