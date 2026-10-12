# Middle-name prefix policy, October 12, 2026

Full middle names accept aligned token prefixes through `same_prefix_tokens`.
The rule uses the same symmetric comparison as first names.
Each aligned token must prefix its counterpart, up to the shorter token list.
For example, `Meg` matches `Megan`, and `Ch` matches `Charles` after canonicalization.

Middle initial conflicts still cause a veto.
Existing full-token overlap and character-set exceptions remain valid.
Seed, ORCID, surname, and first-name rules retain their precedence.
The middle-name rule does not use the alias tuple file.

The implemented rule removes 702 gold-pair vetoes from the latest seven-dataset export.
It removes 1,625 gold-pair vetoes from the current Arrow metadata under the same revised gold labels.
It also removes 19 labeled negative-pair vetoes from the augmented data, including `Yi`/`Yik` and `Yun`/`Yung`.
The Medline conflict inventory has no change.
All removed vetoes concern pairs in the same block.
These counts measure constraint results, not model accuracy.

The comparison fixes first-name aliases at their pre-addition state to isolate the middle-name policy.
The real Python constraint boundary confirms the predicted result for all 84,761 retained effective conflict states.
The real Arrow/native boundary agrees on all 66 distinct changed name-state pairs in both directions.
Other prior name conflicts remain subject to their existing rules.

The user approves the measured tradeoff and requests this policy change.
The 19 negative cases remain potential merge risks after the hard veto disappears.
Model scores and cluster decisions require a separate model evaluation.
