"""
This script was used to create the file of known name pairs back in the day
but the name_pairs have since been modified. Don't rerun this -> it's historical.
Important note: the github file has no names with apostraphes or dashes.
"""

from s2and.text import normalize_text
from s2and.consts import CONFIG
import os
import urllib.request


# source here: https://github.com/Christopher-Thornton/hmni/blob/master/dev/name_pairs.txt
url = "https://raw.githubusercontent.com/Christopher-Thornton/hmni/master/dev/name_pairs.txt"
with urllib.request.urlopen(url) as response:
    content = response.read().decode('utf-8')
    lines = content.splitlines()

pairs = set()
for line in lines:
    line_split = line.strip().lower().split(",")
    if len(line_split) > 1:
        a, b = normalize_text(line_split[0]), normalize_text(line_split[1])
        # same first letter
        if a[0] != b[0]:
            continue
        pairs.add((a, b))
        pairs.add((b, a))

# write to disk
with open(os.path.join(CONFIG["main_data_dir"], "s2and_name_tuples.txt") , "w") as f:
    for name1, name2 in pairs:
        f.write(f"{name1},{name2}\n")
