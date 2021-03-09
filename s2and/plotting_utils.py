import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
import json
import pandas as pd
import os

CONFIG_LOCATION = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data", "path_config.json"))
with open(CONFIG_LOCATION) as _json_file:
    CONFIG = json.load(_json_file)

EXP_DIR = os.path.join(CONFIG["internal_data_dir"], "experiments/paper_experiments_baseline_save_facets_w_gen_eth/")


def plot_box(s2and_performance: dict, s2_performance: dict, figs_path: str, title: str, total_bins: int = 5):

    b3 = []
    keylist = []
    model = []

    if title == "ethnicity":
        sns.set(rc={"figure.figsize": (15, 7)})
    else:
        sns.set(rc={"figure.figsize": (12, 6)})

    for facet, f1 in s2and_performance.items():
        if title == "gender":
            if facet == "-":
                continue

        if title == "year":
            if int(facet) == 0:
                continue

        for _f1 in f1:
            if title != "gender" and title != "ethnicity":
                keylist.append(float(facet))
            else:
                keylist.append(facet)
            b3.append(_f1)
            model.append("S2AND")

        for _f1 in s2_performance[facet]:
            if title != "gender" and title != "ethnicity":
                keylist.append(float(facet))
            else:
                keylist.append(facet)
            b3.append(_f1)
            model.append("S2")

    if title == "year":
        bins = pd.IntervalIndex.from_tuples(
            [
                (int(min(keylist)), 1960),
                (1960, 1980),
                (1980, 1990),
                (1990, 2000),
                (2000, 2005),
                (2005, 2010),
                (2010, 2015),
                (2015, 2020),
            ]
        )
    elif title == "number of authors":
        bins = pd.IntervalIndex.from_tuples(
            [
                (0, 1),
                (1, 2),
                (2, 5),
                (5, 10),
                (10, 15),
                (15, 25),
                (25, 50),
                (50, 100),
                (100, 1000),
            ]
        )
    elif title == "block size":
        bins = pd.IntervalIndex.from_tuples(
            [(0, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 100), (100, 200), (200, 400), (400, 800), (800, 3000)]
        )
    elif title == "cluster size":
        bins = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 100), (100, 800)])
    elif total_bins > 0:
        bins = np.linspace(
            min(keylist), max(keylist), total_bins + 1
        )  # need a + 1 because of how bins interacts with boxplot

    df = pd.DataFrame({"X": keylist, "Y": b3, "Model": model})
    if total_bins > 0:
        data_cut = pd.cut(df.X, bins)
        df["group"] = data_cut
    else:
        df["group"] = keylist

    ax = sns.boxplot(
        x="group",
        y="Y",
        hue="Model",
        data=df,
        showmeans=True,
        meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "blue"},
    )

    if title != "gender" and title != "ethnicity":
        plt.xlabel(title, fontsize=15)
    else:
        plt.xlabel("", fontsize=15)

    plt.ylabel("B3 F1", fontsize=15)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc="lower left")
    plt.savefig(join(figs_path, title + "_facet.png"), bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_facets(
    union_gender_f1,
    union_ethnicity_f1,
    union_author_num_f1,
    union_year_f1,
    union_block_len_f1,
    union_cluster_len_f1,
    union_homonymity_f1,
    union_synonymity_f1,
    union_s2_gender_f1,
    union_s2_ethnicity_f1,
    union_s2_author_num_f1,
    union_s2_year_f1,
    union_s2_block_len_f1,
    union_s2_cluster_len_f1,
    union_s2_homonymity_f1,
    union_s2_synonymity_f1,
    figs_path,
    gender_ethnicity_available=True,
    save_results=True,
):

    pred_facets = [
        union_gender_f1,
        union_ethnicity_f1,
        union_author_num_f1,
        union_year_f1,
        union_block_len_f1,
        union_cluster_len_f1,
        union_homonymity_f1,
        union_synonymity_f1,
    ]

    s2_facets = [
        union_s2_gender_f1,
        union_s2_ethnicity_f1,
        union_s2_author_num_f1,
        union_s2_year_f1,
        union_s2_block_len_f1,
        union_s2_cluster_len_f1,
        union_s2_homonymity_f1,
        union_s2_synonymity_f1,
    ]

    plot_names = [
        "gender",
        "ethnicity",
        "number of authors",
        "year",
        "block size",
        "cluster size",
        "homonymity",
        "synonymity",
    ]

    num_bins = [
        0,
        0,
        8,
        4,
        8,
        8,
        10,
        10,
    ]

    if not gender_ethnicity_available:
        pred_facets.remove(union_gender_f1)
        pred_facets.remove(union_ethnicity_f1)
        s2_facets.remove(union_s2_gender_f1)
        s2_facets.remove(union_s2_ethnicity_f1)
        plot_names.remove("gender")
        plot_names.remove("ethnicity")
        num_bins.remove(0)
        num_bins.remove(0)

    for pred_facet, s2_facet, plot_name, bin_size in zip(pred_facets, s2_facets, plot_names, num_bins):

        if save_results:
            with open(figs_path + plot_name + "_dict_pred.json", "w") as fp:
                json.dump(pred_facet, fp, indent=4)
            with open(figs_path + plot_name + "_dict_s2.json", "w") as fp:
                json.dump(s2_facet, fp, indent=4)

        plot_box(pred_facet, s2_facet, figs_path, plot_name, total_bins=bin_size)


if __name__ == "__main__":
    TEST_DATA_PATH = EXP_DIR

    with open(TEST_DATA_PATH + "facetsgender_dict_pred.json", "r") as f:
        union_gender_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetsethnicity_dict_pred.json", "r") as f:
        union_ethnicity_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetsnumber of authors_dict_pred.json", "r") as f:
        union_author_num_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetsyear_dict_pred.json", "r") as f:
        union_year_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetsblock size_dict_pred.json", "r") as f:
        union_block_len_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetscluster size_dict_pred.json", "r") as f:
        union_cluster_len_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetshomonymity_dict_pred.json", "r") as f:
        union_homonymity_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetssynonymity_dict_pred.json", "r") as f:
        union_synonymity_f1 = json.load(f)

    with open(TEST_DATA_PATH + "facetsgender_dict_s2.json", "r") as f:
        union_s2_gender_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetsethnicity_dict_s2.json", "r") as f:
        union_s2_ethnicity_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetsnumber of authors_dict_s2.json", "r") as f:
        union_s2_author_num_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetsyear_dict_s2.json", "r") as f:
        union_s2_year_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetsblock size_dict_s2.json", "r") as f:
        union_s2_block_len_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetscluster size_dict_s2.json", "r") as f:
        union_s2_cluster_len_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetshomonymity_dict_s2.json", "r") as f:
        union_s2_homonymity_f1 = json.load(f)
    with open(TEST_DATA_PATH + "facetssynonymity_dict_s2.json", "r") as f:
        union_s2_synonymity_f1 = json.load(f)

    plot_facets(
        union_gender_f1,
        union_ethnicity_f1,
        union_author_num_f1,
        union_year_f1,
        union_block_len_f1,
        union_cluster_len_f1,
        union_homonymity_f1,
        union_synonymity_f1,
        union_s2_gender_f1,
        union_s2_ethnicity_f1,
        union_s2_author_num_f1,
        union_s2_year_f1,
        union_s2_block_len_f1,
        union_s2_cluster_len_f1,
        union_s2_homonymity_f1,
        union_s2_synonymity_f1,
        figs_path=EXP_DIR + "boxplot/",
        gender_ethnicity_available=True,
    )
