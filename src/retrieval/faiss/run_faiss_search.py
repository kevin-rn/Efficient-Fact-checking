from argparse import ArgumentParser
import json
import numpy as np
import os
import h5py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import torch
from tqdm import tqdm
from typing import List, Dict
from .faiss_search import FaissSearch
import re
from src.tools.monitor_utils import ProcessMonitor

### Argument parsing
### e.g. python faiss/run_faiss_search.py --hover_stage=claim_verification --rerank_mode=none --topk_nn=5 \
### --n_rerank=5 --use_gpu --precompute_embed --compress_embed
parser = ArgumentParser()

parser.add_argument(
    "--setting",
    type=str,
    default=None,
    help="Name of the setting to run"
)

parser.add_argument(
   "--dataset_name",
   type=str,
   default="hover",
   help="Name of the claim data to evaluate on [hover | wice]"
)
parser.add_argument(
    "--hover_stage",
    default="claim_verification",
    type=str,
    help="[claim_verification | sent_retrieval]"
)

parser.add_argument(
    "--rerank_mode",
    default="none",
    type=str,
    help="[none | within | between]"
)

parser.add_argument(
    "--topk_nn",
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
parser.add_argument(
    "--use_gpu",
    action='store_true',
    help="Use Faiss-gpu instead of cpu"
)

parser.add_argument(
    "--precompute_embed",
    action='store_true',
    help="Use precomputed embeds instead of calculating on the fly"
)

parser.add_argument(
    "--compress_embed",
    action='store_true',
    help="Use Unsupervised Neural Quantitazation to compress embedding size"
)
args = parser.parse_args()

if args.setting:
    ENWIKI_DB = os.path.join("data", "db_files", args.setting + ".db")  
else:
    ENWIKI_DB = os.path.join("data", "wiki_wo_links.db")

UNQ_CHECKPOINT = os.path.join("src", "retrieval", "unq", "notebooks", "logs", 
                              "enwiki_claim_unq_16b_original-full", "checkpoint_best.pth")
EMBED_FILE = os.path.join("data", "embed_files", f"{args.setting}.h5")

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
    conn.close()
    title_list = np.array([doc[0] for doc in results])
    doc_list = [" ".join(doc[1].split("[SENT]")) for doc in tqdm(results, desc="Get document text")]

    print("-- Create FAISS index --")
    emb_dim, n_codebooks = encoder.get_sentence_embedding_dimension(), 16
    faiss_search = FaissSearch(data=title_list, 
                            emb_dim=n_codebooks if args.compress_embed else emb_dim,
                            idx_dir=os.path.join("data", "indices", "faiss"),
                            encoder=encoder,
                            setting=args.setting,
                            use_gpu=args.use_gpu,
                            use_compress=args.compress_embed)
    if args.compress_embed:
        faiss_search.setup_compression_model(checkpoint_path=UNQ_CHECKPOINT, vect_dim=emb_dim, n_codebooks=n_codebooks)

    if not faiss_search.load_index_if_available():
        with ProcessMonitor(dataset=args.dataset_name) as pm, torch.no_grad():
            pm.start()
            if args.precompute_embed and os.path.exists(EMBED_FILE):
                    emb_list = []
                    with h5py.File(EMBED_FILE, 'r') as hf:
                        for group_name in hf.keys():
                            # emb_list = np.append(emb_list, hf[group_name][:])
                            emb_list = hf[group_name][:]
            else:
                emb_list = encoder.encode(doc_list, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
            faiss_search.create_index(emb_list)
            faiss_search.print_size_info()
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
            evidence_text = [sent for doc_text in doc_list for sent in doc_text.split('[SENT]')]
        case "within":
            for doc_text in doc_list:
                # Create embeddings for document sentences and compute cosine similarity between these and the original claim.
                doc_sents = doc_text.split('[SENT]')
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
            doc_sents = [sent for doc_text in doc_list for sent in doc_text.split('[SENT]')]
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

def format_for_claim_verification(claim_args: dict) -> None:
    """
    Formats retrieved neighbours into data format for Claim Verification stage of the HoVer pipeline.

    Args:
        - top_nn_results (Dict): Top-nn retrieved results containing document information.
        - claims (Dict): Dict object containing information for the claims.
        - datasplit (str): Name of the claim datasplit to process for.
    """
    results = []
    docs = claim_args['topk_nn']['docs']
    claims_emb = claim_args['topk_nn']['claim_embeds']
    for idx in range(len(docs)):
        claim_obj = claim_args['claims'][idx]
        evidence_text = rerank_nn(claim_emb=claims_emb[idx], 
                                doc_list=docs[idx], 
                                top_k=args.n_rerank, 
                                rerank_mode=args.rerank_mode)
        json_result = {'id': claim_obj['uid'], 
                       'claim': claim_obj['claim'], 
                       'context': evidence_text, 
                       'label': claim_obj['label']}
        results.append(json_result)
    retrieve_file = os.path.join("data", claim_args['dataset_name'], "claim_verification", 
                            f"hover_{claim_args['datasplit']}_claim_verification.json")
    with open(retrieve_file, 'w', encoding="utf-8") as f:
        json.dump(results, f)

def format_for_sent_retrieval(claim_args: dict) -> None:
    """
    Formats retrieved neighbours into data format for Sent Retrieval stage of the HoVer pipeline.

    Args:
        - top_nn_results (Dict): Top-nn retrieved results containing document information.
        - claims (Dict): Dict object containing information for the claims.
        - datasplit (str): Name of the claim datasplit to process for.
    """
    results = []
    doc_titles = claim_args['topk_nn']['doc_titles']
    docs = claim_args['topk_nn']['docs']
    for idx in range(len(docs)):
        claim_obj = claim_args['claims'][idx]
        context_data = [[doc_title, doc_text.split('[SENT]')] for doc_title, doc_text in zip(doc_titles[idx], docs[idx])]
        json_result = {'id': claim_obj['uid'], 
                       'claim': claim_obj['claim'], 
                       'context': context_data, 
                       'supporting_facts': claim_obj['supporting_facts']}
        results.append(json_result)
    retrieve_file = os.path.join("data", claim_args['dataset_name'], "sent_retrieval", 
                        f"hover_{claim_args['datasplit']}_sent_retrieval.json")
    with open(retrieve_file, 'w', encoding="utf-8") as f:
        json.dump(results, f)

def claim_retrieval(faiss_search: FaissSearch, dataset_name:str, data_split: str, batched: bool) -> None:
    """
    Retrieves for all the claims of a HoVer datasplit all top-k nearest neighbour texts 
    and formats it for claim verification.

    Parameters:
        - faiss_search (FaissSearch): FAISS object used for dense retrieval.
    """
    # Embed claim
    print(f"-- Retrieval for {data_split} set --")

    claims_file = os.path.join("data", dataset_name, f"{dataset_name}_{data_split}_release_v1.1.json")
    with ProcessMonitor(dataset=dataset_name) as pm:
        pm.start()
        # Retrieve for each claim the top-5 nearest neighbour documents.
        with open(claims_file, 'r') as json_file:
            claim_json = json.load(json_file)
        claims = [re.sub("\s+", " ", claim_obj['claim']) for claim_obj in claim_json]
        topk_nn = faiss_search.get_top_n_neighbours(query_list=claims, 
                                                    top_k=args.topk_nn, 
                                                    batched=batched,
                                                    doc_database=ENWIKI_DB)
        claim_args = {"topk_nn": topk_nn,
                      "claims": claim_json,
                      "dataset_name": dataset_name,
                      "datasplit": data_split}

        # Format output file
        if args.hover_stage == "claim_verification":
            format_for_claim_verification(claim_args=claim_args)
        else:
            format_for_sent_retrieval(claim_args=claim_args)

def main():
    faiss = setup_faiss_search()
    claim_retrieval(faiss_search=faiss, dataset_name=args.dataset_name, data_split="train", batched=True)
    claim_retrieval(faiss_search=faiss, dataset_name=args.dataset_name, data_split="dev", batched=False)
    # faiss.remove_index()

if __name__ == "__main__":
    main()