# Name count and ORCID prefix experiment results

Keep M, the plain mention count policy. Keep EVENT threshold 10 with all ORCIDs as the existing prefix default.
The experiments do not identify a unique best prefix threshold.

Compute corpus counts once per data refresh. Apply the ORCID event threshold before the alias merge, then save the prefix table.
Prediction loads the saved table; it does not repeat the corpus pass or the threshold sweep.
The runtime caches the packaged table after its first load in each process.
The alias threshold remains two. EVENT counts can contain multiple contributions from one ORCID; ten does not mean ten distinct ORCIDs.

The selected study table contains 5,974 prefix pairs across 1,606 outer keys.
Its SHA256 is `332067cf29651fd077a50dc4d1754bf3f407b5b463dceb03f774810bec6aede5`.
The saved file is `prefix/giant_followup/target_counts/tables/event_10_all.json` under the local experiment output root.
The packaged prefix data and manifest are absent from this checkout. Release preparation must create the packaged artifact.
This commit records the decision and fixes the Arrow boundary; it does not replace production count files or models.

The source pass covers 699,054,083 mentions with surnames. Five indexes cover 75,085 benchmark keys; seven Arrow bundles contain 766,133 signatures.
LODO uses 70 fresh LightGBM fits with fixed 1.21 parameters, eight CPU threads, and deterministic mode.
No alternative passes the fixed gain and uncertainty gates. Both development seeds retain M.
Two datasets fail the bootstrap group minimum. No alternative qualifies for the sealed-test comparison.

The benchmark prefix sweep completes 6,400 cells. Its real-cluster coverage fails the fixed minimums, so its result remains proxy-only.
The cap-250 follow-up does not support a production recommendation.
The production-cap sweep completes 80 cells on h wang, s lee, s park, and j smith: 1,115,133 signatures.
It uses native ORCID repair, fixed graph defaults, and seed 1111.

| Prefix table | Retained pairs, cap 5,000 | Comparisons, cap 5,000 | Retained pairs, cap 10,000 | Comparisons, cap 10,000 |
| --- | ---: | ---: | ---: | ---: |
| Empty | 906,953 | 2,390,756,555 | 927,459 | 4,800,854,087 |
| Alias only | 906,998 | 2,402,260,696 | 927,459 | 4,839,420,714 |
| EVENT 5 | 906,998 | 2,416,157,681 | 927,459 | 4,874,280,652 |
| EVENT 10 | 906,998 | 2,407,638,220 | 927,459 | 4,843,664,940 |
| EVENT 20 | 906,998 | 2,404,848,289 | 927,459 | 4,836,066,504 |
| EVENT 50 | 906,998 | 2,402,260,696 | 927,459 | 4,839,420,714 |

Each cap has 1,009,764 reachable ORCID pairs. All four EVENT thresholds retain the same exact pair identities at each cap.
Target exclusion changes no partition membership. EVENT 5 adds 0.354% and 0.632% comparisons without a measured retention gain.
EVENT 20 and 50 save less than 0.23% comparisons. The fixed rule keeps 10 on a retention tie.
Native ORCID repair and proxy labels use the same identity source. These results measure identity preservation, not independent author accuracy.
Final cluster quality remains untested. A null result does not establish equal candidate effects or inadequate statistical power.

The Arrow fix accepts papers with empty source author lists, which have no rows in the Arrow author table.
Regression tests verify missing coauthor features and errors for missing papers, null author names, and duplicate author positions.
The experiment logs, configurations, and bulk artifacts remain outside this commit in the local experiment output root.
