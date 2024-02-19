import argparse
import bz2
import csv
import json
import os
import random
import re
import sys

from itertools import chain
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(sys.path[0] + "/../..".replace("/", os.path.sep))
from scripts.text_processing_utils import remove_html_tags, tqdm_joblib, search_file_paths
csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    default=None,
    type=str,
    required=True,
    help="[hover | wice]",
)

parser.add_argument(
    "--enwiki_name",
    default=None,
    type=str,
    required=True,
    help="[enwiki-2017-original | enwiki-2023-cite]",
)
args = parser.parse_args()


HOVER_PATH = os.path.join("..", "hover", "data")
JPQ_PATH = os.path.join("data", "doc", "enwiki-dataset")
if not os.path.exists(JPQ_PATH):
    os.makedirs(JPQ_PATH, exist_ok=True)

def jpq_data_format(file_path, data_location) -> List:
    """
    Format data from ENWIKI bz2 files to JPQ format.

    Args:
        file_path (str): Individual file to format.
        data_location (str): Location of the HoVer English wikipedia bz2 files.

    Returns:
        List: List containing formatted data.
    """
    file_data = []
    with bz2.open(data_location + file_path, "rt") as file:
        for line in file:
            wiki_article = json.loads(line)
            wiki_text = wiki_article['text'][1:]
            paragraphs = list(chain.from_iterable(wiki_text))
            text = " ".join(remove_html_tags(paragraphs))
            article = [wiki_article['id'], wiki_article['url'], wiki_article['title'], text]
            file_data.append(article)
    return file_data

def format_docs_data() -> Tuple[Dict, List] :
    """
    Format documents data.

    Returns:
        Tuple[Dict, List]: A tuple containing lookup dictionary and document IDs.
    """
    doc_loc = os.path.join(HOVER_PATH, "enwiki_files", args.enwiki_name)
    tsv_loc = os.path.join(JPQ_PATH, "enwiki-docs.tsv")

    file_paths = search_file_paths(doc_loc)
    with tqdm_joblib(tqdm(desc="Process for jpq", total=len(file_paths))) as progress_bar:
        results = Parallel(n_jobs=16)(
            delayed(jpq_data_format)(bz2_filepath, doc_loc) for bz2_filepath in file_paths
        )
    results = [article for file_data in results for article in file_data]
    with open(tsv_loc, "w") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")
        for article in tqdm(results, total=len(results), desc="Save to tsv"):
            writer.writerow(article)

    doc_ids, _, doc_titles, _ = zip(*results)
    lookup_dict = dict(zip(doc_titles, range(0,len(doc_titles))))
    return lookup_dict, doc_ids

def format_queries_data(lookup_dict, doc_ids) -> None:
    """
    Format queries data.

    Args:
        lookup_dict (Dict): Dictionary for looking up document titles.
        doc_ids (List): List of document IDs.
    """
    last_id = 0
    hover_data_path = os.path.join(HOVER_PATH, args.dataset_name)
    with open(os.path.join(hover_data_path, "hover_train_release_v1.1.json"), 'r') as json_file:
        claim_json = json.load(json_file)
        train_claims = [(claim_id, re.sub('\s+',' ', claim['claim']), claim['supporting_facts']) for claim_id, claim in enumerate(claim_json)]
        last_id = train_claims[-1][0] + 1
        random.shuffle(train_claims)

    with open(os.path.join(hover_data_path, "hover_dev_release_v1.1.json"), 'r') as json_file:
        claim_json = json.load(json_file)
        dev_claims = [(last_id+claim_id, re.sub('\s+',' ', claim['claim']), claim['supporting_facts']) for claim_id, claim in enumerate(claim_json)]
        last_id = dev_claims[-1][0] + 1
        random.shuffle(dev_claims)

    with open(os.path.join(hover_data_path, "hover_test_release_v1.1.json"), 'r') as json_file:
        claim_json = json.load(json_file)
        lead_claims = [(last_id+claim_id, re.sub('\s+',' ', claim['claim'])) for claim_id, claim in enumerate(claim_json)]
        random.shuffle(dev_claims)

    split_size = (len(train_claims) // 10) * 7
    test_claims = train_claims[split_size:]
    train_claims = train_claims[:split_size]

    for split, vals in tqdm([("train", train_claims), ("dev", dev_claims), ("test", test_claims), ("lead", lead_claims)], total=4):
        # Save Queries
        tsv_loc = os.path.join(JPQ_PATH, f"enwiki-doc{split}-queries.tsv")
        with open(tsv_loc, "w") as tsv_file:
            writer = csv.writer(tsv_file, delimiter="\t")
            for article in vals:
                article_claim = [article[0], article[1]]
                writer.writerow(article_claim)

        # Save Qrels
        if split != "lead":
            support_titles = [[doc[0] for doc in doc_list[2]] for doc_list in vals]
            support_ids = [[doc_ids[lookup_dict[title]] for title in doc_list] for doc_list in tqdm(support_titles)]
            qrels = []
            for idx, support_id in enumerate(support_ids):
                claim_id = vals[idx][0]
                for s_id in support_id:
                    qrels.append([claim_id, 0, s_id, 1])

            tsv_loc = os.path.join(JPQ_PATH, f"enwiki-doc{split}-qrels.tsv")
            with open(tsv_loc, "w") as tsv_file:
                writer = csv.writer(tsv_file, delimiter=" ")
                for qrel in qrels:
                    writer.writerow(qrel)

def main():
    lookup_dict, doc_ids = format_docs_data()
    format_queries_data(lookup_dict=lookup_dict, doc_ids=doc_ids)

if __name__ == "__main__":
    main()
