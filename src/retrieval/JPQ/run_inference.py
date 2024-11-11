import argparse
import json
import os
import re
import sqlite3
import unicodedata
import uuid

import faiss
import pandas as pd
from typing import Dict, List, Tuple

from src.tools.monitor_utils import ProcessMonitor, format_size

from .jpq.model import DenseRetrievalJPQSearch, JPQDualEncoder

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
    "--setting",
    default=None,
    type=str,
    required=True,
    help="[enwiki-2017-original | enwiki-2023-cite]",
)

parser.add_argument(
    "--data_split",
    default=None,
    type=str,
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
    help="Number of nearest neighbour documents to retrieve",
)

parser.add_argument(
    "--use_gpu", 
    action="store_true", 
    help="Use Faiss-gpu instead of cpu"
)


parser.add_argument(
    "--sent_select",
    action="store_true",
    help="Format data for Sent Retrieval stage of HoVer",
)

args = parser.parse_args()

DB_PATH = os.path.join("data", "db_files", args.setting + ".db")

def load_documents() -> Dict[str, str]:
    """
    Load documents from the dataset.

    Returns:
        dict: A dictionary containing document IDs as keys and document titles and texts as values.
    """
    tsv_path = os.path.join(
        "data", "jpq_doc", f"enwiki-{args.dataset_name}-dataset", args.setting + "-docs.tsv"
    )
    # Check if preprocessing happened, otherwise use the HoVer db file
    if os.path.isfile(tsv_path):
        tsv_file = pd.read_csv(
            tsv_path, delimiter="\t", names=["doc_id", "url", "doc_title", "doc_text"]
        )
        doc_ids = tsv_file["doc_id"].tolist()
        doc_titles = tsv_file["doc_title"].tolist()
        doc_texts = tsv_file["doc_text"].tolist()
    else:
        conn = sqlite3.connect(DB_PATH)
        wiki_db = conn.cursor()
        results = wiki_db.execute(
            "SELECT id, text FROM documents ORDER BY id COLLATE NOCASE ASC"
        ).fetchall()
        conn.close()
        doc_titles, doc_texts = zip(*results)
        doc_ids = [str(uuid.uuid5(uuid.NAMESPACE_OID, str(id))) for id in doc_titles]

    corpus = {}
    for id, title, text in zip(doc_ids, doc_titles, doc_texts):
        corpus[str(id)] = {"title": str(title), "text": str(text)}
    return corpus


def load_queries(claim_file: str) -> Tuple[List, Dict]:
    """
    Load queries from a JSON file.

    Args:
        claim_file (str): The filename of the JSON file containing queries.

    Returns:
        tuple: A tuple containing a list of query items and a dictionary of queries.
    """
    claims_path = os.path.join("data", args.dataset_name, claim_file)
    with open(claims_path, "r") as f:
        claim_json = json.load(f)
    queries = {}
    for claim in claim_json:
        queries[claim["uid"]] = re.sub("\s+", " ", claim["claim"])
    return claim_json, queries


def format_for_claim_verification(
    results: Dict, queries: Dict, corpus: Dict, claim_json: List, data_split: str
) -> None:
    """
    Format retrieval results for claim verification stage.

    Args:
        results (dict): Dictionary containing retrieval results.
        queries (dict): Dictionary containing queries.
        corpus (dict): Dictionary containing document corpus.
        claim_json (list): List of JSON objects containing claims.
        data_split (str): Data split identifier.

    Returns:
        None
    """

    def get_label_by_uid(uid, claim_json):
        for item in claim_json:
            if item["uid"] == uid:
                return item["label"]
        return None

    claim_results = []
    for query_id, docs_data in results.items():
        context = " ".join(corpus[doc_id]["text"] for doc_id, _ in docs_data.items())
        json_result = {
            "id": query_id,
            "claim": queries[query_id],
            "context": context,
            "label": get_label_by_uid(uid=query_id, claim_json=claim_json),
        }
        claim_results.append(json_result)

    retrieve_file = os.path.join(
        "data",
        args.dataset_name,
        "claim_verification",
        f"hover_{data_split}_claim_verification.json",
    )
    with open(retrieve_file, "w", encoding="utf-8") as f:
        json.dump(claim_results, f)


def format_for_sent_retrieval(results: Dict, queries: Dict, corpus: Dict, claim_json: List, data_split: str) -> None:
    """
    Format retrieval results for sentence retrieval stage.

    Args:
        results (dict): Dictionary containing retrieval results.
        queries (dict): Dictionary containing queries.
        corpus (dict): Dictionary containing document corpus.
        claim_json (list): List of JSON objects containing claims.
        data_split (str): Data split identifier.

    Returns:
        None
    """
    def get_supp_facts_by_uid(uid, claim_json):
        for item in claim_json:
            if item["uid"] == uid:
                return item["supporting_facts"]
        return None

    conn = sqlite3.connect(DB_PATH)
    wiki_db = conn.cursor()
    claim_results = []
    for query_id, docs_data in results.items():
        context = []
        for doc_id, _ in docs_data.items():
            title = corpus[doc_id]["title"]
            doc_text = wiki_db.execute(
                "SELECT text FROM documents WHERE id = ?",
                (unicodedata.normalize("NFD", title),),
            ).fetchone()[0]
            context_sents = [title, doc_text.split("[SENT]")]
            context.append(context_sents)

        json_result = {
            "id": query_id,
            "claim": queries[query_id],
            "context": context,
            "supporting_facts": get_supp_facts_by_uid(
                uid=query_id, claim_json=claim_json
            ),
        }
        claim_results.append(json_result)
    conn.close()
    retrieve_file = os.path.join(
        "data",
        args.dataset_name,
        "sent_retrieval",
        f"hover_{data_split}_sent_retrieval.json",
    )
    with open(retrieve_file, "w", encoding="utf-8") as f:
        json.dump(claim_results, f)


def search_and_generate_results(
    model: JPQDualEncoder, corpus: Dict, queries: Dict, claim_json: List, batch_size: int, data_split: str
):
    """
    Search for relevant documents and generate retrieval results.

    Args:
        model: The JPQDualEncoder model instance.
        corpus (dict): A dictionary containing document IDs as keys and document titles and texts as values.
        queries (dict): A dictionary containing query IDs as keys and query texts as values.
        claim_json (list): A list of JSON objects containing claims.
        batch_size (int): Batch size for inference.
        data_split (str): Data split identifier.

    Returns:
        None
    """
    dr_jpq = DenseRetrievalJPQSearch(
        dataset_name=args.dataset_name,
        model=model,
        subvectors_num=args.subvectors_num,
        batch_size=batch_size,
        use_gpu=args.use_gpu,
    )
    # Create or load corpus index
    index_path = os.path.join(
        "data", "indices", "jpq", f"{args.setting}-{args.subvectors_num}x8.index"
    )
    if os.path.isfile(index_path):
        dr_jpq.corpus_index = faiss.read_index(index_path)
    else:
        with ProcessMonitor(dataset=args.dataset_name) as pm:
            pm.start()
            dr_jpq.index_corpus(
                corpus=corpus, score_function="dot", index_path=index_path
            )

        vecs_size = os.path.getsize(index_path)
        total_vecs = dr_jpq.corpus_index.ntotal
        print(
            f"{total_vecs} Vectors, ",
            f"Total size: {format_size(vecs_size)}, ",
            f"Vector size: {format_size(vecs_size/total_vecs)}",
        )

    # Perform Retrieval
    with ProcessMonitor(dataset=args.dataset_name) as pm:
        pm.start()
        results = dr_jpq.search(
            corpus=corpus, queries=queries, top_k=args.top_k_nn, score_function="dot"
        )
        if args.sent_select:
            format_for_sent_retrieval(results, queries, corpus, claim_json, data_split)
        else:
            format_for_claim_verification(
                results, queries, corpus, claim_json, data_split
            )


def main():
    setting = "-".join(str(args.setting).split("-")[:2])
    jpq_path = (
        f"models/{setting}/m{args.subvectors_num}/doc_encoder",
        f"models/{setting}/m{args.subvectors_num}/query_encoder",
    )
    model = JPQDualEncoder(model_path=jpq_path)
    corpus = load_documents()
    if args.data_split:
        claim_json, queries = load_queries(
            f"{args.dataset_name}_{args.data_split}_release_v1.1.json"
        )
        search_and_generate_results(
            model, corpus, queries, claim_json, args.batch_size, args.data_split
        )
    else:
        claim_json, queries = load_queries(f"{args.dataset_name}_train_release_v1.1.json")
        search_and_generate_results(model, corpus, queries, claim_json, 128, "train")
        claim_json, queries = load_queries(f"{args.dataset_name}_dev_release_v1.1.json")
        search_and_generate_results(model, corpus, queries, claim_json, 1, "dev")


if __name__ == "__main__":
    main()
