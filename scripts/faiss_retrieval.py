from argparse import ArgumentParser
import faiss
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import torch
from typing import List, Dict
from tqdm import tqdm, trange
from monitor_utils import monitor, format_size
import unicodedata

### Argument parsing
### e.g. python faiss_retrieval.py --data_split=train --setting=cite --rerank_mode=none --top_k_nn=5 --top_k_rerank=5
parser = ArgumentParser()

parser.add_argument(
    "--data_split",
    default=None,
    type=str,
    required=True,
    help="[train | dev]",
)

parser.add_argument(
    "--setting",
    default=None,
    type=str,
    required=True,
    help="Name of the corresponding setting to retrieve documents from e.g. cite-embed"
)

parser.add_argument(
    "--rerank_mode",
    default="none",
    type=str,
    help="[none | within | between]"
)

parser.add_argument(
    "--top_k_nn",
    default=5,
    type=int,
    help="Top-k documents to retrieve for nearest neighbours FAISS"
)

parser.add_argument(
    "--top_k_rerank",
    default=5,
    type=int,
    help="Top-k sentences to retrieve for reranking"
)

args = parser.parse_args()


BASE_PATH = os.path.join("..", "baselines", "hover", "data")
ENWIKI_EMBED_DB = os.path.join(BASE_PATH, "db_files", 
                               f"wiki_wo_links-{args.setting}.db")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device=device
)

### FAISS Search Class
class FaissSearch:
    def __init__(self, data: List[str] = None, emb_dim=None, idx_dir=None):
        """
        Initialize a FaissSearch object.

        Parameters:
        - data (List[str], optional): A list of data (passages or text) to create the index.
        - emb_dim (int): The embedding dimension.

        This class is used to create and query a Faiss index for similarity search.

        """
        self.emb_dim = emb_dim
        self.data = data
        self.ann = faiss.IndexFlatIP(self.emb_dim)
        self.dir = idx_dir if idx_dir else os.path.join(BASE_PATH, "indices")
        self.index_path = os.path.join(self.dir, f"index_faiss_{args.setting.replace('-embed', '')}")
    
    def print_size_info(self) -> None:
        """
        Prints information on total amount of entries, total size and individual vector size.
        """
        vecs_size = os.path.getsize(self.index_path)
        total_vecs = self.ann.ntotal
        print(f"{total_vecs} Vectors, ", 
                f"Total size: {format_size(vecs_size)}, ", 
                f"Vector size: {format_size(vecs_size/total_vecs)}")

    def create_index(self, passage_vectors):
        """
        Create a Faiss index with the given passage vectors.

        Parameters:
        - passage_vectors: A collection of passage vectors to build the index.

        """
        self.ann.add(passage_vectors)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        faiss.write_index(self.ann, self.index_path)

    def load_index_if_available(self) -> bool:
        """
        Load a pre-existing Faiss index if available.

        Returns:
        bool: True if the index was successfully loaded, False if no index is found.

        """
        if os.path.exists(self.index_path):
            self.ann = faiss.read_index(self.index_path)
            return True
        else:
            return False

    def get_top_n_neighbours(self, query_vector: torch.Tensor, top_k: int) -> Dict:
        """
        Search for the top nearest neighbors to a given query vector.

        Parameters:
        - query_vector (Tensor): The query vector for which nearest neighbors are to be found.
        - top_k (int): The number of nearest neighbors to retrieve.

        Returns:
        A dictionary with the following keys:
        - "docs" (List[List[str]]): List of nearest neighbors document text for each query vector. 
            e.g. [["doctext[SENT]doctext", "doctext2[SENT]doctext2"], [..]]
        - "distances" (Tensor): The corresponding distances to the nearest neighbors.
        """
        distances, indices = self.ann.search(query_vector, top_k)
        doc_titles = [self.data[idxs].tolist() for idxs in indices]
        # Connect to db
        conn = sqlite3.connect(ENWIKI_EMBED_DB)
        wiki_db = conn.cursor()

        # Retrieve top-5 document text for each claim
        doc_list = []
        for claim_doc_titles in tqdm(doc_titles, desc="Retrieve top-k docs"):
            claim_doc_texts = []
            for title in claim_doc_titles:
                doc_text = wiki_db.execute("SELECT text FROM documents WHERE id = ?", 
                                            (unicodedata.normalize('NFD', title),)).fetchone()[0]
                claim_doc_texts.append(doc_text)
            doc_list.append(claim_doc_texts)
        conn.close()
        return {"docs": doc_list, "distances": distances}

### Prepare HoVer claim verification data methods ###
def setup_faiss_search() -> FaissSearch:
    """
    Creates or loads FaissSearch object with document titles and indexed document text embeddings.

    Returns:
        FaissSearch - used for dense retrieval
    """
    conn = sqlite3.connect(ENWIKI_EMBED_DB)
    wiki_db = conn.cursor()

    # Retrieve doc titles and embeds from database.
    results = wiki_db.execute("SELECT id FROM documents ORDER BY id COLLATE NOCASE ASC").fetchall()
    doc_list = np.array([doc[0] for doc in results])

    # Create or load FAISS index.
    faiss_search = FaissSearch(doc_list, encoder.get_sentence_embedding_dimension())
    if not faiss_search.load_index_if_available():
        results = wiki_db.execute("SELECT embed FROM documents ORDER BY id COLLATE NOCASE ASC").fetchall()
        emb_list = np.array([np.frombuffer(embed[0]) for embed in results])
        print("-- Creating FAISS index --")
        faiss_search.create_index(emb_list)
    conn.close()
    print("-- FAISS Instantiated --")
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
            evidence_text = [sent for doc_text in doc_list for sent in doc_text.split('[SENT]') if sent.strip()]
        case "within":
            for doc_text in doc_list:
                # Create embeddings for document sentences and compute cosine similarity between these and the original claim.
                doc_sents = [sent for sent in doc_text.split('[SENT]') if sent.strip()]
                if doc_sents:
                    with torch.no_grad():
                        doc_sents_embeds = encoder.encode(doc_sents)
                    similarity_scores = cosine_similarity([claim_emb], doc_sents_embeds)[0]
                    # sort on similarity score and afterwards sort top-k sentences back in original order.
                    sorted_sents = [sent for _, sent in sorted(zip(similarity_scores, enumerate(doc_sents)), key=lambda x: x[0], reverse=True)]
                    top_k_sents = " ".join([sent for _, sent in sorted(sorted_sents[:top_k], key=lambda x: x[0])])
                    evidence_text.append(top_k_sents)
        case "between":
            doc_sents = [sent for doc_text in doc_list for sent in doc_text.split('[SENT]') if sent.strip()]
            if doc_sents:
                with torch.no_grad():
                    doc_sents_embeds = encoder.encode(doc_sents)
                similarity_scores = cosine_similarity([claim_emb], doc_sents_embeds)[0]
                # sort on similarity score and afterwards sort top-k sentences back in original order.
                sorted_sents = [sent for _, sent in sorted(zip(similarity_scores, enumerate(doc_sents)), key=lambda x: x[0], reverse=True)]
                evidence_text = [sent for _, sent in sorted(sorted_sents[:top_k], key=lambda x: x[0])]

    return " ".join(evidence_text)
    

def claim_retrieval(faiss_search: FaissSearch) -> None:
    """
    Retrieves for all the claims of a HoVer datasplit all top-k nearest neighbour texts 
    and formats it for claim verification.

    Parameters:
        - faiss_search (FaissSearch): FAISS object used for dense retrieval.
    """
    # Embed claim
    claims_file = os.path.join(BASE_PATH, "hover", 
                                f"hover_{args.data_split}_release_v1.1.json")
    with open(claims_file, 'r') as json_file:
        claim_json = json.load(json_file)
    claims = [claim_obj['claim'] for claim_obj in claim_json]
    with torch.no_grad():
        claims_emb = encoder.encode(claims, show_progress_bar=True)
    print(f"-- Claim embedding completed for {args.data_split} set with {len(claims)} entries --")

    # Retrieve for each claim the top-5 nearest neighbour documents.
    topk_nn = faiss_search.get_top_n_neighbours(query_vector=claims_emb, top_k=args.top_k_nn)
    print(f"-- Top-5 Nearest Neighbours retrieved for {args.data_split} set --")

    results = []
    docs = topk_nn['docs']
    for idx in trange(len(docs)):
        claim_obj = claim_json[idx]
        evidence_text = rerank_nn(claim_emb=claims_emb[idx], doc_list=docs[idx], 
                                  top_k=args.top_k_rerank, rerank_mode=args.rerank_mode)
        json_result = {'id': claim_obj['uid'], 'claim': claim_obj['claim'], 
                    'context': evidence_text, 'label': claim_obj['label']}
        results.append(json_result)

    assert len(results) == len(docs)

    # Save to file
    retrieve_file = os.path.join(BASE_PATH, f"hover", "claim_verification", 
                            "hover_"+args.data_split+"_claim_verification.json")
    with open(retrieve_file, 'w', encoding="utf-8") as f:
        json.dump(results, f)

    print(f"-- Processed {args.data_split} file --")

def main():
    faiss = setup_faiss_search()
    claim_retrieval(faiss_search=faiss)

if __name__ == "__main__":
    # To avoid also timing the faiss index creation, only time the loading of it.
    setup_faiss_search() 
    monitor(main)