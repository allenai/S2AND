from typing import Any, List, Dict

import os
import argparse
import shutil
import json
import re

EMAIL_SPECIAL_CASES = {
    '{"dianwenju@fudan.edu.cn hu.haifeng@sipi.com.cn"}': "hu.haifeng@sipi.com.cn",
    '{"jyuan@tjh.tjmu.edu.cn yanghand@139.com"}': "jyuan@tjh.tjmu.edu.cn",
    '{"shichen@whu.edu.cn lianrong@whu.edu.cn"}': "shichen@whu.edu.cn",
    '{"yuyang5012@hotmail.com yongfan2011@gmail.com xiaofangsun@hotmail.com"}': "yongfan2011@gmail.com",
    '{"dryiminli@vip.163.com lxq1118@126.com"}': "lxq1118@126.com",
    '{"litao@zjnu.cn shenli@bjmu.edu.cn"}': "shenli@bjmu.edu.cn",
    '{"tanruoyun112@vip.sina.com lancetgu@aliyun.com"}': "lancetgu@aliyun.com",
    '{"njmuwzj@qq.com njmuwzj1990@hotmail.com"}': "njmuwzj1990@hotmail.com",
    '{"yczhang@tjh.tjmu.edu.cn jiawei@tjh.tjmu.edu.cn"}': "jiawei@tjh.tjmu.edu.cn",
    '{"Jenny.Wei@astrazeneca.com kzhu@cs.sjtu.edu.cn"}': "Jenny.Wei@astrazeneca.com",
    '{"xjn0906@gmail.com jhw8799@yahoo.com"}': "jhw8799@yahoo.com",
    '{"liu086@126.com muyanshuang@163.com"}': "liu086@126.com",
    '{"liu086@126.com sxx1959@163.com"}': "liu086@126.com",
    '{"wzy5607@sina.com dengkaiyu@yahoo.com"}': "dengkaiyu@yahoo.com",
    '{"zhangjue@pku.edu.cn cjr.wangxiaoying@vip.163.com"}': "zhangjue@pku.edu.cn",
    '{"null yguo_smmu@163.com"}': "yguo_smmu@163.com",
    '{"qmzou2007@163.com zyang@tmmu.edu.cn"}': "zyang@tmmu.edu.cn",
    '{"peijun1020@163.com renyyyyy@126.com"}': "renyyyyy@126.com",
    '{"lixd@chinacdc.cn yuchua@163.com"}': "lixd@chinacdc.cn",
    '{"guopeng661@mail.xjtu.edu.cn dalinhexjtu@163.com"}': "guopeng661@mail.xjtu.edu.cn",
    '{"pguo@imaplad.ac.cn whlin@bjmu.edu.cn"}': "pguo@imaplad.ac.cn",
    '{"wangzq@zzu.edu.cn cuij@zzu.edu.cn"}': "cuij@zzu.edu.cn",
    '{"hjtsai@ntu.edu.tw ahjwang@gate.sinica.edu.tw"}': "ahjwang@gate.sinica.edu.tw",
    '{"cailei2010@126.com helinhelin3@gmail.com"}': "cailei2010@126.com",
    '{"chendi@nicemice.cn kjia@fau.edu"}': "chendi@nicemice.cn",
    '{"liusw@smu.edu.cn jianhe@smu.edu.cn"}': "liusw@smu.edu.cn",
    '{"wangyanrang2014@126.com qinggu1118@126.com"}': "qinggu1118@126.com",
    '{"kqtanjg@bjmu.edu.cn liheping@tsinghua.edu.cn"}': "liheping@tsinghua.edu.cn",
    '{"scnubip@gmail.com xingda@scnu.edu.cn"}': "xingda@scnu.edu.cn",
    '{"qinxue919@126.com lis8858@126.com"}': "qinxue919@126.com",
    '{"zhaogang@fmmu.edu.cn biomidas@fmmu.edu.cn"}': "biomidas@fmmu.edu.cn",
    '{"hrmaskf@hku.hk yxliang@hku.hk ytf0707@126.com"}': "hrmaskf@hku.hk",
    '{"njzhaxm@qq.com ws0801@hotmail.com"}': "ws0801@hotmail.com",
    '{"wangzy@nju.edu.cn sunping@nju.edu.cn"}': "sunping@nju.edu.cn",
    '{"preecer@uah.edu James.Burgess@uah.edu charles.dermer@nrl.navy.mil nicola.omodei@stanford.edu azk@mpe"}': "charles.dermer@nrl.navy.mil",
    '{"sjzhu@umd.edu jchiang@slac.stanford.edu charles.dermer@nrl.navy.mil nicola.omodei@stanford.edu giaco"}': "charles.dermer@nrl.navy.mil",
    '{"bjbohr@nbi,dk"}': "bjbohr@nbi.dk",
    '{"31848346@qq.com xiaofangsun@hotmail.com yongfan011@gzhmu.edu.cn"}': "yongfan011@gzhmu.edu.cn",
}

AFFILIATION_SPECIAL_CASES = {
    'National Centre of Scientific Research "Demokritos"': "National Centre of Scientific Research Demokritos"
}


def possibly_copy_file(full_file_name_source: str, full_file_name_target: str, write_files: bool):
    if write_files:
        print(f"Copying {full_file_name_source} to {full_file_name_target}")
        shutil.copy(full_file_name_source, full_file_name_target)
    else:
        print(f"Would copy {full_file_name_source} to {full_file_name_target}")


def possibly_write_json(output: Any, full_file_name_target: str, write_files: bool):
    if write_files:
        print(f"Writing output of length {len(output)} to {full_file_name_target}")
        with open(full_file_name_target, "w") as _json_file:
            json.dump(output, _json_file)
    else:
        print(f"Would write output of length {len(output)} to {full_file_name_target}")


def transform_cluster_file(full_file_name_source: str):
    with open(full_file_name_source) as _json_file:
        input_data = json.load(_json_file)

    output_data = {}
    for cluster_id, signature_ids in input_data.items():
        output_row = {}
        output_row["cluster_id"] = cluster_id
        output_row["signature_ids"] = [str(signature_id) for signature_id in signature_ids]
        output_row["model_version"] = -1  # using -1 to signify gold
        output_data[cluster_id] = output_row

    return output_data


def is_empty_value(input_data) -> bool:
    if input_data is None:
        return True
    if input_data == "":
        return True
    if input_data.lower() == "nan":
        return True

    return False


def transform_paper_file(full_file_name_source: str):
    with open(full_file_name_source) as _json_file:
        input_data = json.load(_json_file)

    output_data = {}
    for paper_id, paper_info in input_data.items():
        output_row: Dict[str, Any] = {}
        output_row["paper_id"] = int(paper_id)
        output_row["title"] = paper_info["title"] if not is_empty_value(paper_info["title"]) else None
        output_row["abstract"] = paper_info["abstract"] if not is_empty_value(paper_info["abstract"]) else None
        output_row["journal_name"] = (
            paper_info["journal_name"] if not is_empty_value(paper_info["journal_name"]) else None
        )
        output_row["venue"] = paper_info["venue"] if not is_empty_value(paper_info["venue"]) else None

        assert type(paper_info["year"]) == int
        output_row["year"] = paper_info["year"]

        assert paper_info["sources"].startswith("{")
        assert paper_info["sources"].endswith("}")
        output_row["sources"] = paper_info["sources"][1:-1].split(",")

        if is_empty_value(paper_info["fields_of_study"]):
            output_row["fields_of_study"] = []
        else:
            assert paper_info["fields_of_study"].startswith("{")
            assert paper_info["fields_of_study"].endswith("}")
            output_row["fields_of_study"] = paper_info["fields_of_study"][1:-1].split(",")

        output_row["authors"] = [{"position": author[0], "author_name": author[1]} for author in paper_info["authors"]]
        output_row["references"] = (
            [int(reference_id) for reference_id in paper_info["references"]] if "references" in paper_info else None
        )
        output_data[int(paper_id)] = output_row
    return output_data


def transform_signature_file(full_file_name_source: str):
    with open(full_file_name_source) as _json_file:
        input_data = json.load(_json_file)

    output_data = {}
    for signature_id, signature_info in input_data.items():
        output_row: Dict[str, Any] = {}
        output_row["signature_id"] = str(signature_id)
        output_row["paper_id"] = int(signature_info["paperid"])

        author_info = signature_info["authorinfo"]
        output_author_info: Dict[str, Any] = {}
        output_author_info["position"] = int(author_info["position"])
        output_author_info["block"] = author_info["block"]
        output_author_info["first"] = author_info["first"] if not is_empty_value(author_info["first"]) else None
        output_author_info["middle"] = author_info["middle"] if not is_empty_value(author_info["middle"]) else None
        output_author_info["last"] = author_info["last"]
        output_author_info["suffix"] = author_info["suffix"] if not is_empty_value(author_info["suffix"]) else None

        if is_empty_value(author_info["emails"]):
            output_author_info["email"] = None
        else:
            assert author_info["emails"].startswith("{")
            assert author_info["emails"].endswith("}")
            emails = EMAIL_SPECIAL_CASES.get(author_info["emails"], author_info["emails"])
            emails_list: List[str] = list(set(re.split(r"\s|,", emails.strip('{"').strip('"}'))))
            assert not any(re.search(r"[^\\]\"", emails) for emails in emails_list), emails_list
            if len(emails_list) != 1:
                print(
                    f"WARNING: skipping poorly formatted email {author_info['emails']} for {author_info['first']} {author_info['last']} on {signature_info['paperid']}"
                )
            output_author_info["email"] = emails_list[0]

        if is_empty_value(author_info["affiliations"]):
            output_author_info["affiliations"] = []
        else:
            affiliations = AFFILIATION_SPECIAL_CASES.get(author_info["affiliations"], author_info["affiliations"])
            affiliations_list: List[str] = list(set(affiliations.strip('{"').strip('"}').split('","')))
            assert len(affiliations_list) >= 1
            if any(re.search(r"[^\\]\"", affiliation) for affiliation in affiliations_list):
                print(f"WARNING: affiliation has a quote {affiliations_list}")
            output_author_info["affiliations"] = affiliations_list

        output_author_info["given_block"] = (
            author_info.get("given-block", None) if not is_empty_value(author_info.get("given-block", None)) else None
        )
        output_author_info["given_name"] = (
            signature_info.get("actual_name", None)
            if not is_empty_value(signature_info.get("actual_name", None))
            else None
        )

        output_author_info["estimated_ethnicity"] = (
            author_info["ethnicity"] if not is_empty_value(author_info["ethnicity"]) else None
        )
        output_author_info["estimated_gender"] = (
            author_info["gender"] if not is_empty_value(author_info["gender"]) else None
        )

        output_row["author_info"] = output_author_info
        output_data[str(signature_id)] = output_row

    return output_data


def take_outer_keys(full_file_name_source: str):
    with open(full_file_name_source) as _json_file:
        input_data = json.load(_json_file)

    return list(input_data.keys())


def main(base_dir_in: str, base_dir_out: str, write_files: bool):
    """
    This script transforms an older format of the dataset files to the final
    output format
    """
    if not os.path.exists(base_dir_out):
        os.mkdir(base_dir_out)

    for dataset in os.listdir(base_dir_in):
        print()
        print(f"Iterating over {dataset}:")
        if not os.path.exists(os.path.join(base_dir_out, dataset)):
            os.mkdir(os.path.join(base_dir_out, dataset))

        for file_name in os.listdir(os.path.join(base_dir_in, dataset)):
            if "specter" in file_name:
                possibly_copy_file(
                    os.path.join(base_dir_in, dataset, file_name),
                    os.path.join(base_dir_out, dataset, dataset + "_specter.pickle"),
                    write_files,
                )
            elif "cluster" in file_name:
                output = transform_cluster_file(os.path.join(base_dir_in, dataset, file_name))
                possibly_write_json(
                    output,
                    os.path.join(base_dir_out, dataset, dataset + "_clusters.json"),
                    write_files,
                )
            elif "paper" in file_name:
                output = transform_paper_file(os.path.join(base_dir_in, dataset, file_name))
                possibly_write_json(
                    output,
                    os.path.join(base_dir_out, dataset, dataset + "_papers.json"),
                    write_files,
                )
            elif "signature" in file_name:
                output = transform_signature_file(os.path.join(base_dir_in, dataset, file_name))
                possibly_write_json(
                    output,
                    os.path.join(base_dir_out, dataset, dataset + "_signatures.json"),
                    write_files,
                )
            elif dataset == "medline" and file_name == "medtestpairs_pipe_fresh_cleaned.csv":
                possibly_copy_file(
                    os.path.join(base_dir_in, dataset, file_name),
                    os.path.join(base_dir_out, dataset, "test_pairs.csv"),
                    write_files,
                )
            elif dataset == "medline" and file_name == "medtrainpairs_pipe_fresh_cleaned.csv":
                possibly_copy_file(
                    os.path.join(base_dir_in, dataset, file_name),
                    os.path.join(base_dir_out, dataset, "train_pairs.csv"),
                    write_files,
                )
            elif dataset == "aminer" and file_name == "name_to_pubs_test_100.json":
                output = take_outer_keys(os.path.join(base_dir_in, dataset, file_name))
                possibly_write_json(
                    output,
                    os.path.join(base_dir_out, dataset, "test_keys.json"),
                    write_files,
                )
            elif dataset == "aminer" and file_name == "name_to_pubs_train_500.json":
                output = take_outer_keys(os.path.join(base_dir_in, dataset, file_name))
                possibly_write_json(
                    output,
                    os.path.join(base_dir_out, dataset, "train_keys.json"),
                    write_files,
                )

            else:
                print(f"WARNING: Ignoring {file_name} in {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir_in",
        help="Path to the directory with the previous data format",
    )
    parser.add_argument(
        "--base_dir_out",
        help="Path to the directory to write the new data format to",
    )
    parser.add_argument(
        "--write_files",
        help="Whether to actually write the files or just print some info",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    main(args.base_dir_in, args.base_dir_out, args.write_files)
