import argparse
import collections
import json
import logging
import os
import sqlite3
import string
import unicodedata

import nltk

from .StanfordNLP import StanfordNLP

corenlp = StanfordNLP()
from tqdm import tqdm

from src.tools.monitor_utils import ProcessMonitor


def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    return c


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_split",
        default=None,
        type=str,
        required=True,
        help="[train | dev | test]",
    )
    parser.add_argument(
        "--sent_retrieve_range",
        default=5,
        type=int,
        help="Top k retrieved documents to be used in sentence selection.",
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        type=str,
    )
    parser.add_argument("--dataset_name", default="hover", type=str)

    parser.add_argument(
        "--setting",
        default=None,
        type=str,
        help="Name of the db file inside data/db_files folder",
    )

    parser.add_argument(
        "--doc_retrieval_output_dir",
        default="exp1.0",
        type=str,
    )
    parser.add_argument("--doc_retrieval_model_global_step", default=900, type=int)
    parser.add_argument("--oracle", action="store_true")

    args = parser.parse_args()
    with ProcessMonitor(dataset=args.dataset_name) as pm:
        pm.start()

        if args.setting:
            wiki_db = connect_to_db(
                os.path.join(args.data_dir, "db_files", args.setting + ".db")
            )
        else:
            wiki_db = connect_to_db(os.path.join(args.data_dir, "wiki_wo_links.db"))

        args.data_dir = os.path.join(args.data_dir, args.dataset_name)
        hover_data = json.load(
            open(
                os.path.join(
                    args.data_dir,
                    args.dataset_name + "_" + args.data_split + "_release_v1.1.json",
                )
            )
        )

        args.doc_retrieval_output_dir = os.path.join(
            "out",
            args.dataset_name,
            args.doc_retrieval_output_dir,
            "doc_retrieval",
            "checkpoint-" + str(args.doc_retrieval_model_global_step),
        )
        doc_retrieval_predictions = json.load(
            open(
                os.path.join(
                    args.doc_retrieval_output_dir,
                    args.data_split + "_predictions_.json",
                )
            )
        )

        uid_to_supporting_fact = {}
        for e in hover_data:
            uid, claim, supporting_facts = e["uid"], e["claim"], e["supporting_facts"]
            assert uid not in uid_to_supporting_fact
            uid_to_supporting_fact[uid] = [claim, supporting_facts]

        data_for_sent_ret = []
        for uid in tqdm(doc_retrieval_predictions.keys()):
            pred_obj = doc_retrieval_predictions[uid]
            sorted_titles, sorted_probs = (
                pred_obj["sorted_titles"],
                pred_obj["sorted_probs"],
            )
            pred_titles = []
            for idx in range(min(len(sorted_titles), args.sent_retrieve_range)):
                pred_titles.append(sorted_titles[idx])

            context = []

            if uid in uid_to_supporting_fact:
                claim, supporting_facts = uid_to_supporting_fact[uid]
                sp_title_to_sp = {}
                for _sp in supporting_facts:
                    if _sp[0] not in sp_title_to_sp:
                        sp_title_to_sp[_sp[0]] = [_sp[1]]
                    else:
                        sp_title_to_sp[_sp[0]].append(_sp[1])

                for title in pred_titles:
                    try:
                        para = wiki_db.execute(
                            "SELECT id, text FROM documents WHERE id=(?)",
                            (unicodedata.normalize("NFD", title),),
                        ).fetchall()[0]
                        para_title, para_text = list(para)
                        para_text = " ".join(para_text.split("[SENT]"))
                        if para_text.split():
                            para_parse = corenlp.annotate(para_text)
                            para_sents = []
                            for sent_parse in para_parse["sentences"]:
                                start_idx = sent_parse["tokens"][0][
                                    "characterOffsetBegin"
                                ]
                                end_idx = sent_parse["tokens"][-1]["characterOffsetEnd"]
                                sent = para_text[start_idx:end_idx]
                                para_sents.append(sent)
                            context.append([title, para_sents])
                    except ValueError:
                        continue

                dp = {
                    "id": uid,
                    "claim": claim,
                    "context": context,
                    "supporting_facts": supporting_facts,
                }
                data_for_sent_ret.append(dp)
                # data_for_sent_ret_dict[uid] = dp
            else:
                print(uid)
                print(claim)
                assert False

        logging.info("Saving prepared data ...")
        if args.oracle:
            with open(
                os.path.join(
                    args.data_dir,
                    "sent_retrieval",
                    "hover_" + args.data_split + "_sent_retrieval_oracle.json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(data_for_sent_ret, f)
        else:
            with open(
                os.path.join(
                    args.data_dir,
                    "sent_retrieval",
                    "hover_" + args.data_split + "_sent_retrieval.json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(data_for_sent_ret, f)


if __name__ == "__main__":
    main()
