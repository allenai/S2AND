from s2and.text import normalize_text

"""
This script is used to create the file of known name pairs
"""

pairs = set()
# source here: https://github.com/Christopher-Thornton/hmni/blob/master/dev/name_pairs.txt
with open("name_pairs.txt") as f:
    for line in f:
        line_split = line.strip().lower().split(",")
        if len(line_split) > 1:
            a, b = normalize_text(line_split[0]), normalize_text(line_split[1])
            # same first letter
            if a[0] != b[0]:
                continue
            pairs.add((a, b))
            pairs.add((b, a))

# write to disk
with open("../data/s2and_name_tuples.txt", "w") as f:
    for name1, name2 in pairs:
        f.write(f"{name1},{name2}\n")
