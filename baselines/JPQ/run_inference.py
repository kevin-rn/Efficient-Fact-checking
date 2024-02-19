import argparse
from jpq.model import DenseRetrievalJPQSearch, JPQDualEncoder
import torch
import json
import re
import pandas as pd
import os

import sys
sys.path.append(sys.path[0] + "/../..".replace("/", os.path.sep))
from scripts.monitor_utils import ProcessMonitor, format_size

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument(
    "--dataset_name",
    default=None,
    type=str,
    required=True,
    help="[hover | wice]",
)

parser.add_argument(
    "--data_split",
    default=None,
    type=str,
    required=True,
    help="[train | dev | test]",
)

parser.add_argument(
    "--subvectors_num",
    default=96,
    type=int,
    help="Subvector num setting",
)

parser.add_argument(
    "--batch_size",
    default=128,
    type=int,
    help="Batch size for inference",
)

parser.add_argument(
    "--top_k_nn",
    default=5,
    type=int,
    help="Number of nearest neighbour documents to retrieve"
)
args = parser.parse_args()

def load_documents(tsv_file):
    tsv_path = os.path.join("data", "doc", "enwiki-dataset", tsv_file)
    tsv_file = pd.read_csv(tsv_path, delimiter="\t", names=['doc_id', 'url', 'doc_title', 'doc_text'])
    doc_ids = tsv_file['doc_id'].tolist()
    doc_titles = tsv_file['doc_title'].tolist()
    doc_texts = tsv_file['doc_text'].tolist()
    corpus = {}
    for id, title, text in zip(doc_ids, doc_titles, doc_texts): 
        corpus[str(id)] = {"title": str(title), "text": str(text)}
    return corpus

def load_queries(claim_file):
    claims_path = os.path.join("..", "hover", "data", args.dataset_name, claim_file)
    with open(claims_path, "r") as f:
        claim_json = json.load(f)
    queries = {}
    for claim in claim_json:
        queries[claim['uid']] = re.sub('\s+', ' ', claim['claim'])
    return claim_json, queries

def get_label_by_uid(uid, claim_json):
    for item in claim_json:
        if item['uid'] == uid:
            return item['label']
    return None

def search_and_generate_results(model, corpus, queries, claim_json, device):
    dr_jpq = DenseRetrievalJPQSearch(dataset_name=args.dataset_name,
                                     model=model, 
                                     subvectors_num=args.subvectors_num,
                                     batch_size=args.batch_size)
    # Create or load corpus index
    index_path = os.path.join("data","eval",f"OPQ{args.subvectors_num},IVF1,PQ{args.subvectors_num}x8.index")
    with ProcessMonitor(dataset=args.dataset_name) as pm:
        pm.start()
        dr_jpq.index_corpus(corpus=corpus, score_function="dot")

    vecs_size = os.path.getsize(index_path)
    total_vecs = dr_jpq.corpus_index.ntotal
    print(f"{total_vecs} Vectors, ", 
            f"Total size: {format_size(vecs_size)}, ", 
            f"Vector size: {format_size(vecs_size/total_vecs)}")

    # Perform Retrieval
    with ProcessMonitor(dataset=args.dataset_name) as pm:
        pm.start()
        results = dr_jpq.search(corpus=corpus, queries=queries, 
                                top_k=args.top_k_nn, score_function="dot", device=device)
        claim_results = []
        for query_id, docs_data in results.items():
            context = " ".join(corpus[doc_id]["text"] for doc_id, _ in docs_data.items())
            json_result = {'id': query_id, 
                            'claim': queries[query_id], 
                            'context': context, 
                            'label': get_label_by_uid(query_id, claim_json)}
            claim_results.append(json_result)

        retrieve_file = os.path.join("data","eval", args.data_split + ".json")
        with open(retrieve_file, 'w', encoding="utf-8") as f:
            json.dump(results, f)

def main():
    jpq_path = (f"data/doc/eval/m{args.subvectors_num}/doc_encoder", 
                f"data/doc/eval/m{args.subvectors_num}/query_encoder")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JPQDualEncoder(model_path=jpq_path)
    corpus = load_documents("enwiki-docs.tsv")
    claim_json, queries = load_queries(f"hover_{args.data_split}_release_v1.1.json")
    search_and_generate_results(model, corpus, queries, claim_json, device)
    print("Finished inference JPQ")

if __name__ == "__main__":
    main()
