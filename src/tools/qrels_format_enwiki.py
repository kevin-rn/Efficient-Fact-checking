import argparse
import bz2
import csv
import json
import os
import pandas as pd
import random
import re
import sys

from itertools import chain
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
from tqdm import tqdm
from .text_processing_utils import remove_html_tags, tqdm_joblib, search_file_paths
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
    "--setting",
    default=None,
    type=str,
    required=True,
    help="[enwiki-2017-original | enwiki-2023-cite]",
)
parser.add_argument(
    "--first_paragraph_only",
    action='store_true',
    help="For given experiment setting, only use the first available paragraph text"
)
args = parser.parse_args()

def jpq_data_format(file_path, data_location, text_key) -> List:
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
            wiki_text = wiki_article[text_key][1:]

            if args.first_paragraph_only:
                paragraphs = []
                for para in wiki_text:
                    if para:
                        paragraphs = para
                        break
            else:
                paragraphs = list(chain.from_iterable(wiki_text))
            text = " ".join(remove_html_tags(paragraphs))
            article = [wiki_article['id'], wiki_article['url'], wiki_article['title'], re.sub('\s+',' ', text)]
            file_data.append(article)
    return file_data

def format_docs_data(jpq_path) -> Tuple[Dict, List] :
    """
    Format documents data.

    Returns:
        Tuple[Dict, List]: A tuple containing lookup dictionary and document IDs.
    """
    doc_loc = os.path.join("data", f"enwiki_files", args.setting)
    tsv_loc = os.path.join(jpq_path, args.setting + "-docs.tsv")
    if os.path.isfile(tsv_loc):
        tsv_file = pd.read_csv(tsv_loc, delimiter="\t", names=['doc_id', 'url', 'doc_title', 'doc_text'])
        doc_ids = tsv_file['doc_id'].tolist()
        doc_titles = tsv_file['doc_title'].tolist()
    else:
        text_key = "text" if "original" in args.setting else "fact_text"
        file_paths = search_file_paths(doc_loc)
        with tqdm_joblib(tqdm(desc="Process for jpq", total=len(file_paths))) as progress_bar:
            results = Parallel(n_jobs=16)(
                delayed(jpq_data_format)(bz2_filepath, doc_loc, text_key) for bz2_filepath in file_paths
            )
        results = [article for file_data in results for article in file_data]
        with open(tsv_loc, "w") as tsv_file:
            writer = csv.writer(tsv_file, delimiter="\t")
            for article in tqdm(results, total=len(results), desc="Save to tsv"):
                writer.writerow(article)
        doc_ids, _, doc_titles, _ = zip(*results)
    lookup_dict = dict(zip(doc_titles, range(0,len(doc_titles))))
    return lookup_dict, doc_ids

def check_query_files_present(dir_loc):
    files = os.listdir(dir_loc)
    qrels = [f"enwiki-doc{split}-qrels.tsv" for split in ["train", "dev", "test"]]
    queries = [f"enwiki-doc{split}-queries.tsv" for split in ["train", "dev", "test", "lead"]]
    qq_files = qrels + queries
    
    missing_files = []
    for file_name in qq_files:
        if file_name not in files:
            missing_files.append(file_name)
    
    if len(qq_files) == len(missing_files):
        return False
    elif len(missing_files) == 0:
        return True
    else:
        print("Missing some files: ", missing_files)
        return True


def format_queries_data(lookup_dict, doc_ids, jpq_path) -> None:
    """
    Format queries data.

    Args:
        lookup_dict (Dict): Dictionary for looking up document titles.
        doc_ids (List): List of document IDs.
    """
    last_id = 0
    hover_data_path = os.path.join("data", args.dataset_name)
    with open(os.path.join(hover_data_path, f"{args.dataset_name}_train_release_v1.1.json"), 'r') as json_file:
        claim_json = json.load(json_file)
        train_claims = [(claim_id, re.sub('\s+',' ', claim['claim']), claim['supporting_facts']) for claim_id, claim in enumerate(claim_json)]
        last_id = train_claims[-1][0] + 1
        random.shuffle(train_claims)

    with open(os.path.join(hover_data_path, f"{args.dataset_name}_dev_release_v1.1.json"), 'r') as json_file:
        claim_json = json.load(json_file)
        dev_claims = [(last_id+claim_id, re.sub('\s+',' ', claim['claim']), claim['supporting_facts']) for claim_id, claim in enumerate(claim_json)]
        last_id = dev_claims[-1][0] + 1
        random.shuffle(dev_claims)

    with open(os.path.join(hover_data_path, f"{args.dataset_name}_test_release_v1.1.json"), 'r') as json_file:
        claim_json = json.load(json_file)
        lead_claims = [(last_id+claim_id, re.sub('\s+',' ', claim['claim'])) for claim_id, claim in enumerate(claim_json)]
        random.shuffle(dev_claims)

    split_size = (len(train_claims) // 10) * 7
    test_claims = train_claims[split_size:]
    train_claims = train_claims[:split_size]

    for split, vals in tqdm([("train", train_claims), ("dev", dev_claims), ("test", test_claims), ("lead", lead_claims)], total=4):
        # Save Queries
        tsv_loc = os.path.join(jpq_path, f"enwiki-doc{split}-queries.tsv")
        with open(tsv_loc, "w") as tsv_file:
            writer = csv.writer(tsv_file, delimiter="\t")
            for article in vals:
                article_claim = [article[0], article[1]]
                writer.writerow(article_claim)

        # Save Qrels
        if split != "lead":
            claim_id_support = [[(doc_list[0], lookup_dict.get(doc[0], None)) for doc in doc_list[2]] for doc_list in vals]
            qrels = [[claim_id, 0, doc_ids[title_id], 1] 
                     for doc_list in tqdm(claim_id_support) 
                     for claim_id, title_id in doc_list if title_id]
            tsv_loc = os.path.join(jpq_path, f"enwiki-doc{split}-qrels.tsv")
            with open(tsv_loc, "w") as tsv_file:
                writer = csv.writer(tsv_file, delimiter=" ")
                for qrel in qrels:
                    writer.writerow(qrel)

def main():
    JPQ_PATH = os.path.join("data", "jpq_doc", f"enwiki-{args.dataset_name}-dataset")
    if not os.path.exists(JPQ_PATH):
        os.makedirs(JPQ_PATH, exist_ok=True)
    if not check_query_files_present(JPQ_PATH):
        lookup_dict, doc_ids = format_docs_data(JPQ_PATH)
        format_queries_data(lookup_dict=lookup_dict, doc_ids=doc_ids, jpq_path=JPQ_PATH)
    else:
        print("Skip file enwiki data formatting")

if __name__ == "__main__":
    main()
