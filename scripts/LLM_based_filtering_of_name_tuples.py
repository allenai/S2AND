# To run this code you need to install the following dependencies:
# uv pip install google-genai

import os
from google import genai
from google.genai import types
from itertools import islice
import unicodedata

API_KEY = "<insert your gemini key here>"


def generate(input_tuples):
    tuples_str = "\n".join(input_tuples)
    client = genai.Client(
        api_key=API_KEY,
    )

    model = "gemini-2.5-pro"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"""Here are name tuples that might be synonyms or variants or spelling variants of one another:

{tuples_str}

output the ones that are correct. 
exclude wrong ones like (jon,jack) but include spelling variants, nicknames, pet names, misspellings etc etc. 
output every pair that is plausibly referring to the same human.
also output are not plausibly referring to the same human
we also want to separate out chinese names, so output those separately.

output the VERBATIM correct ones 
then a newline 
then the bad ones 
then a newline
then the chinese ones only
and nothing else at all"""
                ),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="text/plain",
    )

    output = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    try:
        out = output.candidates[0].content.parts[0].text
    except:
        out = "Error: No output from model"
    return input_tuples, out


# import all the rows of s2and_name_tuples_filtered.txt
def read_name_tuples(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file if line.strip()]


input_tuples = read_name_tuples(os.path.join(CONFIG["main_data_dir"], "s2and_name_tuples_filtered.txt"))

# testing
import concurrent.futures


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


# Process in batches of 100 in parallel
batch_size = 100
batches = list(batched(input_tuples, batch_size))

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    future_to_batch = {executor.submit(generate, batch): batch for batch in batches}

    results = []
    for future in concurrent.futures.as_completed(future_to_batch):
        batch = future_to_batch[future]
        try:
            input_batch, output = future.result()
            results.append((input_batch, output))
            print(f"Completed batch of {len(input_batch)} items")
        except Exception as exc:
            print(f"Batch generated an exception: {exc}")


# parse the results
keep_tuples = []
bad_tuples = []
chinese_tuples = []
for input_batch, output in results:
    if output.startswith("Error:"):
        print(f"Error in batch")
        continue

    parts = output.split("\n\n")

    if len(parts) == 3:
        keep_part, bad_part, chinese_part = parts
    elif len(parts) == 2:
        keep_part, bad_part = parts
        chinese_part = ""
    elif len(parts) == 1:
        keep_part = parts[0]
        bad_part = ""
        chinese_part = ""
    else:
        print(f"Unexpected output format: {output}")
        continue
    keep_tuples.extend(keep_part.strip().split("\n"))
    bad_tuples.extend(bad_part.strip().split("\n"))
    chinese_tuples.extend(chinese_part.strip().split("\n"))

# remove empty strings
keep_tuples = [t for t in keep_tuples if t]
bad_tuples = [t for t in bad_tuples if t]
chinese_tuples = [t for t in chinese_tuples if t]

# now let's try to find genuine chinese variants only


def generate_chinese(input_tuples):
    tuples_str = "\n".join(input_tuples)
    client = genai.Client(
        api_key=API_KEY,
    )

    model = "gemini-2.5-pro"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"""Here are chinese name tuples that might be spelling variants of one another:

{tuples_str}

output the ones that are correct. 
output every pair that is plausibly referring to the same human with spelling variants
also output tuples that are not plausibly referring to the same human right after
output the VERBATIM correct ones 
then a newline 
then the bad ones 
and nothing else at all"""
                ),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="text/plain",
    )

    output = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    try:
        out = output.candidates[0].content.parts[0].text
    except:
        out = "Error: No output from model"
    return input_tuples, out


# Process in batches of 100 in parallel
batch_size = 100
batches = list(batched(chinese_tuples, batch_size))

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    future_to_batch = {executor.submit(generate_chinese, batch): batch for batch in batches}

    results_chinese = []
    for future in concurrent.futures.as_completed(future_to_batch):
        batch = future_to_batch[future]
        try:
            input_batch, output = future.result()
            results_chinese.append((input_batch, output))
            print(f"Completed batch of {len(input_batch)} items")
        except Exception as exc:
            print(f"Batch generated an exception: {exc}")

# parse the chinese results
chinese_keep_tuples = []
chinese_bad_tuples = []
for input_batch, output in results_chinese:
    if output.startswith("Error:"):
        print(f"Error in chinese batch")
        continue

    parts = output.split("\n\n")

    if len(parts) == 2:
        keep_part, bad_part = parts
    elif len(parts) == 1:
        keep_part = parts[0]
        bad_part = ""
    else:
        print(f"Unexpected chinese output format: {output}")
        continue
    chinese_keep_tuples.extend(keep_part.strip().split("\n"))
    chinese_bad_tuples.extend(bad_part.strip().split("\n"))

# remove empty strings
chinese_keep_tuples = [t for t in chinese_keep_tuples if t]
chinese_bad_tuples = [t for t in chinese_bad_tuples if t]

# now get a final keep list
final_keep_tuples = keep_tuples + chinese_keep_tuples

# we may have messed up the flipping of the tuples, so let's fix that
final_keep_tuples_fixed = []
for t in final_keep_tuples:
    parts = t.split(",")
    if len(parts) == 2:
        final_keep_tuples_fixed.append(f"{parts[0].strip()},{parts[1].strip()}")
        final_keep_tuples_fixed.append(f"{parts[1].strip()},{parts[0].strip()}")
    else:
        print(f"Unexpected tuple format: {t}")

# dedupe while preserving order
seen = set()
final_keep_tuples_deduped = []
for t in final_keep_tuples_fixed:
    if t not in seen:
        seen.add(t)
        final_keep_tuples_deduped.append(t)

# now we should try to reconstruct the original unnormalized name file
# from the web and then filter it down to the final keep tuples
from s2and.text import normalize_text
from s2and.consts import CONFIG
import os
import urllib.request

url = "https://raw.githubusercontent.com/Christopher-Thornton/hmni/master/dev/name_pairs.txt"
with urllib.request.urlopen(url) as response:
    content = response.read().decode("utf-8")
    original_lines = content.splitlines()

# Step 1: Build inverse dict from normalized name -> unnormalized name
normalized_to_unnormalized = {}

for line in original_lines:
    line_split = line.strip().split(",")
    if len(line_split) > 1:
        a_orig, b_orig = line_split[0].strip(), line_split[1].strip()
        a_norm, b_norm = normalize_text(a_orig), normalize_text(b_orig)
        normalized_to_unnormalized[a_norm] = a_orig
        normalized_to_unnormalized[b_norm] = b_orig
    else:
        print(f"Unexpected line format: {line}")


# Step 2: A bunch of the names in the final_keep_tuples_deduped
# don't appear in the original name_pairs.txt file, so we need to handle that
# with LLMs!
def generate_normalized(input_tuples):
    tuples_str = "\n".join(input_tuples)
    client = genai.Client(
        api_key=API_KEY,
    )

    model = "gemini-2.5-pro"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"""Here are some names that have been normalized to lowercase and various punctuation removed:
{tuples_str}
For each one of these, output the name as it would appear in a real name, with proper capitalization and punctuation (apostraphes and dashes).
Output the names in the same order as they appear in the input, one per line."""
                ),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="text/plain",
    )

    output = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    try:
        out = output.candidates[0].content.parts[0].text
    except:
        out = "Error: No output from model"
    return input_tuples, out


# generate a list of all names that need to be unnormalized
all_normalized_names = set()
for t in final_keep_tuples_deduped:
    a_norm, b_norm = t.split(",")
    all_normalized_names.add(a_norm)
    all_normalized_names.add(b_norm)

# find which names are not in our normalized_to_unnormalized dict
names_to_unnormalize = []
for name in all_normalized_names:
    if name not in normalized_to_unnormalized:
        names_to_unnormalize.append(name)

print(f"Found {len(names_to_unnormalize)} names that need unnormalization")

# get all the normalized names in parallel
batch_size = 100
batches = list(batched(names_to_unnormalize, batch_size))

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    future_to_batch = {executor.submit(generate_normalized, batch): batch for batch in batches}

    results_normalized = []
    for future in concurrent.futures.as_completed(future_to_batch):
        batch = future_to_batch[future]
        try:
            input_batch, output = future.result()
            results_normalized.append((input_batch, output))
            print(f"Completed normalization batch of {len(input_batch)} items")
        except Exception as exc:
            print(f"Normalization batch generated an exception: {exc}")

# parse the normalized results
for input_batch, output in results_normalized:
    if output.startswith("Error:"):
        print(f"Error in normalization batch")
        continue

    output_lines = output.strip().split("\n")
    if len(output_lines) == len(input_batch):
        for normalized_name, unnormalized_name in zip(input_batch, output_lines):
            normalized_to_unnormalized[normalized_name] = unnormalized_name.strip()
    else:
        print(f"Mismatch in batch size: input={len(input_batch)}, output={len(output_lines)}")

# which ones STILL don't have an unnormalized version?
missing_unnormalized = [name for name in names_to_unnormalize if name not in normalized_to_unnormalized]
if missing_unnormalized:
    print(f"Still missing unnormalized versions for {len(missing_unnormalized)} names:")
    for name in missing_unnormalized:
        print(f"  {name}")

# send them back through the LLM to try to get unnormalized versions, unbatched
if missing_unnormalized:
    print(f"Attempting to unnormalize remaining {len(missing_unnormalized)} names...")
    try:
        _, output = generate_normalized(missing_unnormalized)
        if not output.startswith("Error:"):
            output_lines = output.strip().split("\n")
            if len(output_lines) == len(missing_unnormalized):
                for normalized_name, unnormalized_name in zip(missing_unnormalized, output_lines):
                    normalized_to_unnormalized[normalized_name] = unnormalized_name.strip()
                print(f"Successfully unnormalized {len(missing_unnormalized)} additional names")
            else:
                print(f"Mismatch in final batch size: input={len(missing_unnormalized)}, output={len(output_lines)}")
        else:
            print(f"Error in final unnormalization batch: {output}")
    except Exception as exc:
        print(f"Final unnormalization batch generated an exception: {exc}")


# Step 3: Apply this dict to final_keep_tuples_deduped
unnormalized_keep_tuples = []
not_found_tuples = []
for t in final_keep_tuples_deduped:
    a_norm, b_norm = t.split(",")
    a_unnorm = normalized_to_unnormalized.get(a_norm, a_norm)
    b_unnorm = normalized_to_unnormalized.get(b_norm, b_norm)

    # Track tuples where we couldn't find the unnormalized version
    if a_unnorm == a_norm and a_norm not in normalized_to_unnormalized:
        not_found_tuples.append(f"Could not unnormalize: {a_norm}")
    if b_unnorm == b_norm and b_norm not in normalized_to_unnormalized:
        not_found_tuples.append(f"Could not unnormalize: {b_norm}")

    unnormalized_keep_tuples.append(f"{a_unnorm},{b_unnorm}")

# Print stats about unnormalization
print(f"Successfully processed {len(unnormalized_keep_tuples)} tuples")
print(f"Could not unnormalize {len(set(not_found_tuples))} unique names")
if not_found_tuples:
    print("Names that couldn't be unnormalized:")
    for name in sorted(set(not_found_tuples)):
        print(f"  {name}")

# save the matching original lines to a file
import unicodedata


def standardize_punctuation(s):
    # Standardize apostrophes
    for apos in ["’", "‘", "‛", "ʼ", "ʽ", "ʾ", "ʿ", "ˈ", "ˊ", "ˋ", "˴", "ʻ", "ˮ", "＇", "`", "´"]:
        s = s.replace(apos, "'")
    # Standardize dashes/hyphens
    for dash in ["–", "—", "−", "‐", "‒", "―", "﹘", "－", "‑"]:
        s = s.replace(dash, "-")
    # Normalize unicode to NFKC
    s = unicodedata.normalize("NFKC", s)
    return s


output_file_path = os.path.join(CONFIG["main_data_dir"], "s2and_unnormalized_filtered_name_tuples.txt")
with open(output_file_path, "w", encoding="utf-8") as f:
    for line in unnormalized_keep_tuples:
        line_std = standardize_punctuation(line)
        f.write(f"{line_std}\n")
