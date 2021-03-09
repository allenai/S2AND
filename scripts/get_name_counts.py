"""
Note: This script won't run because it relies on an internal Semantic Scholar package
called pys2, and is here for documentation of how the name count features were built.
"""
import pickle
from pys2 import _evaluate_redshift_query
from s2and.text import normalize_text


# this queries our internal databases
query = """
    select concat(concat(nvl(first_name, ''), '|||'), nvl(last_name, '')), count(*)
    from content.authors
    group by concat(concat(nvl(first_name, ''), '|||'), nvl(last_name, ''))
"""
first_last_count = _evaluate_redshift_query(query)

# separate first and last + normalize each
first_last_count["first"] = first_last_count["concat"].apply(lambda s: normalize_text(s.split("|||")[0]))
first_last_count["last"] = first_last_count["concat"].apply(lambda s: normalize_text(s.split("|||")[1]))

# for first name, we only keep the first token, so we have to split that off
first_last_count.loc[:, "first"] = first_last_count["first"].apply(lambda s: s.split(" ")[0])

# other intermediate data
first_last_count["first last"] = first_last_count.apply(lambda row: (row["first"] + " " + row["last"]).strip(), axis=1)
first_last_count["first initial"] = first_last_count["first"].apply(lambda s: s[0] if len(s) > 0 else "")
first_last_count["last first initial"] = first_last_count.apply(
    lambda row: (row["last"] + " " + row["first initial"]).strip(), axis=1
)

# counts
first_last_df = first_last_count.groupby("first last")["count"].sum()
last_df = first_last_count.groupby("last")["count"].sum()
first_df = first_last_count.groupby("first")["count"].sum()
last_first_initial_df = first_last_count.groupby("last first initial")["count"].sum()

# save space by filtering out anything with count = 1 as we can get that by default
first_last_dict = first_last_df[first_last_df > 1].to_dict()
last_dict = last_df[last_df > 1].to_dict()
first_dict = first_df[first_df > 1].to_dict()
last_first_initial_dict = last_first_initial_df[last_first_initial_df > 1].to_dict()

# this ends up in S3
with open("name_counts.pickle", "wb") as f:
    pickle.dump(
        (first_dict, last_dict, first_last_dict, last_first_initial_dict),
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
