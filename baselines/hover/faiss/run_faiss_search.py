from argparse import ArgumentParser
from itertools import chain
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import torch
from typing import List, Tuple
from faiss_search import FaissSearch
import sys
sys.path.append(sys.path[0] + '/../..')
from scripts.monitor_utils import ProcessMonitor

### Argument parsing
### e.g. python faiss/run_faiss_search.py --rerank_mode=none --n_neighbours=5 --n_rerank=5
parser = ArgumentParser()

parser.add_argument(
    "--setting",
    type=str,
    default=None,
    help="Name of the setting to run"
)

parser.add_argument(
    "--rerank_mode",
    default="none",
    type=str,
    help="[none | within | between]"
)

parser.add_argument(
    "--n_neighbours",
    default=5,
    type=int,
    help="Top-k documents to retrieve for nearest neighbours FAISS"
)
parser.add_argument(
    "--n_rerank",
    default=5,
    type=int,
    help="Top-k sentences to retrieve for reranking"
)
args = parser.parse_args()

if args.setting:
    ENWIKI_DB = os.path.join("data", "db_files", f"wiki_wo_links-{args.setting}.db")  
else:
    ENWIKI_DB = os.path.join("data", "wiki_wo_links.db")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device=device
)

### Prepare HoVer claim verification data methods ###
def setup_faiss_search() -> FaissSearch:
    """
    Creates or loads FaissSearch object with document titles and indexed document text embeddings.

    Returns:
        FaissSearch - used for dense retrieval
    """
    # Retrieve doc titles and document text from database.
    conn = sqlite3.connect(ENWIKI_DB)
    wiki_db = conn.cursor()
    results = wiki_db.execute("SELECT id, text FROM documents ORDER BY id COLLATE NOCASE ASC").fetchall()
    title_list = np.array([doc[0] for doc in results])
    doc_list = [str(doc[1]).replace("[SENT]", "") for doc in results]

    conn.close()

    # Create or load FAISS index.
    print("-- Creating FAISS index --")
    faiss_search = FaissSearch(data=title_list, 
                            emb_dim=encoder.get_sentence_embedding_dimension(),
                            idx_dir=os.path.join("faiss", "indices"),
                            encoder=encoder,
                            setting=args.setting.replace("-embed", ""),
                            use_gpu=False)

    if not faiss_search.load_index_if_available():
        pm = ProcessMonitor()
        pm.start()
        with torch.no_grad():
            emb_list = encoder.encode(doc_list, batch_size=128, show_progress_bar=True)
        faiss_search.create_index(emb_list)
        faiss_search.print_size_info()
        pm.stop()
    return faiss_search

def rerank_nn(claim_emb, doc_list: List[str], top_k: int, rerank_mode:str) -> str:
    """
    For a given claim, only get top-k sentences (cosine similarity) within or between each nearest neighbour document text.

    Parameters:
        - claim_emb: embedding of the claim to compare against.
        - doc_list (List[str]): Retrieved list of document texts for a claim.
        - top_k (int): how many of the top mmr scored sentences to keep.
        - rerank_mode (str): Options for reranking the document sentences
            * none: No reranking nearest-neighbour document sentences for a given claim, just plain concatenation.
            * within: Rerank on cosine similarity between sentences only within the individual NN documents for a given claim.
            * between: Rerank on cosine similarity between sentences of all NN documents for a given claim.
    
    Returns:
        str: Evidence text composed of a joint top-k sentences per retrieved document for a given claim.
    """
    evidence_text = []
    match rerank_mode:
        case "none":
            evidence_text = [sent for doc_text in doc_list for sent in doc_text.split('[SENT]') if sent.strip()]
        case "within":
            for doc_text in doc_list:
                # Create embeddings for document sentences and compute cosine similarity between these and the original claim.
                doc_sents = [sent for sent in doc_text.split('[SENT]') if sent.strip()]
                if doc_sents:
                    with torch.no_grad():
                        doc_sents_embeds = encoder.encode(doc_sents, batch_size=1, show_progress_bar=True)
                    similarity_scores = cosine_similarity([claim_emb], doc_sents_embeds)[0]
                    # sort on similarity score and afterwards sort top-k sentences back in original order.
                    sorted_sents = [sent for _, sent in sorted(zip(similarity_scores, enumerate(doc_sents)), key=lambda x: x[0], reverse=True)]
                    top_k_sents = sorted_sents[:top_k]
                    top_k_sents = " ".join([sent for sent in doc_sents if sent in top_k_sents])
                    evidence_text.append(top_k_sents)
        case "between":
            doc_sents = [sent for doc_text in doc_list for sent in doc_text.split('[SENT]') if sent.strip()]
            if doc_sents:
                with torch.no_grad():
                    doc_sents_embeds = encoder.encode(doc_sents, batch_size=1, show_progress_bar=True)
                similarity_scores = cosine_similarity([claim_emb], doc_sents_embeds)[0]
                # Get the top-k document sentences based on similarity score
                sorted_sents = [sent for _, sent in sorted(zip(similarity_scores, 
                                                               doc_sents), 
                                                               key=lambda x: x[0], 
                                                               reverse=True)]
                top_k_sents = sorted_sents[:top_k]
                evidence_text = [sent for sent in doc_sents if sent in top_k_sents]
    return " ".join(evidence_text)

def claim_retrieval(faiss_search: FaissSearch, datasplit: str, batched: bool) -> None:
    """
    Retrieves for all the claims of a HoVer datasplit all top-k nearest neighbour texts 
    and formats it for claim verification.

    Parameters:
        - faiss_search (FaissSearch): FAISS object used for dense retrieval.
    """
    # Embed claim
    print(f"-- Retrieval for {datasplit} set --")
    pm = ProcessMonitor()
    pm.start()
    claims_file = os.path.join("data", "hover", 
                                f"hover_{datasplit}_release_v1.1.json")
    with open(claims_file, 'r') as json_file:
        claim_json = json.load(json_file)
    claims = [claim_obj['claim'] for claim_obj in claim_json]

    # Retrieve for each claim the top-5 nearest neighbour documents.
    topk_nn = faiss_search.get_top_n_neighbours(query_list=claims, 
                                                top_k=args.n_neighbours, 
                                                batched=batched,
                                                doc_database=ENWIKI_DB)
    pm.stop()

    # Format output file
    results = []
    docs = topk_nn['docs']
    claims_emb = topk_nn['embeds']
    for idx in range(len(docs)):
        claim_obj = claim_json[idx]
        evidence_text = rerank_nn(claim_emb=claims_emb[idx], 
                                  doc_list=docs[idx], 
                                  top_k=args.n_rerank, 
                                  rerank_mode=args.rerank_mode)

        json_result = {'id': claim_obj['uid'], 'claim': claim_obj['claim'], 
                       'context': evidence_text, 'label': claim_obj['label']}
        results.append(json_result)

    assert len(results) == len(docs)

    # Save to file
    retrieve_file = os.path.join("data", "hover", "claim_verification", 
                            f"hover_{datasplit}_claim_verification.json")
    with open(retrieve_file, 'w', encoding="utf-8") as f:
        json.dump(results, f)

def main():
    faiss = setup_faiss_search()
    claim_retrieval(faiss_search=faiss, datasplit="train", batched=True)
    claim_retrieval(faiss_search=faiss, datasplit="dev", batched=True)
    # faiss.remove_index()

if __name__ == "__main__":
    main()