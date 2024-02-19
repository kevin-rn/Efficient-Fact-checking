import faiss
from functools import partial
import numpy as np
from typing import List, Dict, Any
import os
import torch
import sqlite3
import sys
import unicodedata
import time
from torch import Tensor
from tqdm import tqdm

sys.path.append(sys.path[0] + "/../..".replace("/", os.path.sep))
from unq.lib import UNQModel, UNQSearch, Trainer
sys.path.append(sys.path[0] + "/../../..".replace("/", os.path.sep))
from scripts.monitor_utils import format_size


def sleep(seconds):
    if seconds: time.sleep(seconds)
class FaissSearch:
    def __init__(self,
                data: List[str],
                emb_dim: Any, 
                setting: str,
                encoder: Any,
                use_gpu: bool,
                idx_dir: str="indices",
                sleep_for: int = 2,
                use_compress: bool = False,
                ):
        """
        Initialize a FaissSearch object.

        Parameters:
        - data (List[str], optional): A list of data (passages or text) to create the index.
        - emb_dim (int): The embedding dimension.

        This class is used to create and query a Faiss index for similarity search.

        """
        self.emb_dim = emb_dim
        self.data = data
        self.sleep_for = sleep_for
        if use_gpu:
            self.gpu_res = faiss.StandardGpuResources()
            flat_index = faiss.IndexFlatIP(self.emb_dim)
            self.ann = faiss.index_cpu_to_gpu(self.gpu_res, 0, flat_index)
        else:
            self.ann = faiss.IndexFlatIP(self.emb_dim)
        self.dir = idx_dir
        suffix = "_compressed" if use_compress else ""
        self.use_compress = use_compress
        self.index_path = os.path.join(self.dir, setting+suffix+".index")
        self.encoder = encoder
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_compression_model(self, checkpoint_path: str, vect_dim: int, n_codebooks: int):
        device_ids = [0]
        model = UNQModel(input_dim=vect_dim, num_codebooks=n_codebooks)
        trainer = Trainer(
                model=model, experiment_name='debug', device_ids=device_ids, loss_opts={},
                LearnedSimilaritySearch=partial(UNQSearch, model=model, rerank_k=500, batch_size=1000,
                                                reorder_batch_size=250, device_ids=device_ids),
                NegativeSimilaritySearch=partial(UNQSearch, model=model, rerank_k=1, batch_size=1000,
                                                reorder_batch_size=250, device_ids=device_ids),
            )
        trainer.load_checkpoint(path=checkpoint_path)
        trainer.model.eval()
        self.unq = trainer.model
    
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
        if self.use_compress:
            passage_vectors = self.index_compress(passage_vectors)

        self.ann.add(passage_vectors)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        if self.use_gpu:
            flat_index = faiss.index_gpu_to_cpu(self.ann)
            faiss.write_index(flat_index, self.index_path)
        else:
            faiss.write_index(self.ann, self.index_path)

        sleep(self.sleep_for)

    def load_index_if_available(self) -> bool:
        """
        Load a pre-existing Faiss index if available.

        Returns:
        bool: True if the index was successfully loaded, False if no index is found.

        """
        if os.path.exists(self.index_path):
            if self.use_gpu:
                flat_index = faiss.read_index(self.index_path)
                self.ann = faiss.index_cpu_to_gpu(self.gpu_res, 0, flat_index)
            else:
                self.ann = faiss.read_index(self.index_path)
            return True
        else:
            return False
    
    def remove_index(self) -> None:
        """
        Removes a pre-existing Faiss index if available.
        """
        if os.path.exists(self.index_path):
            os.remove(self.index_path)

    def index_compress(self, embeddings: np.ndarray) -> List[Tensor]:
        """
        Perform index compression (in batches) using the UNQ model (see baselines/unq folder).
        
        Args:
            - embeddings (Tensor): Index embeddings to compress.
            - emb_dim (int): dimensionality of the embeddings.
        
        Returns:
            List of compressed index embedding tensors
        """
        emb_list = []
        arr_chunks = np.array_split(embeddings, np.ceil(len(embeddings)/128))
        for emb_chunk in arr_chunks:
            emb_tensor = torch.from_numpy(emb_chunk).to(self.device)
            compressed_codes = self.unq.get_codes(emb_tensor)
            compressed_codes = compressed_codes.cpu().detach()
            emb_list.extend(list(map(torch.squeeze, compressed_codes)))
        return np.array(emb_list)

    def get_top_n_neighbours(self, query_list: List[Any], top_k: int, batched: bool, doc_database: str) -> Dict:
        """
        Search for the top nearest neighbors to a given query vector.

        Parameters:
        - query_vector (List): The query vector for which nearest neighbors are to be found.
        - top_k (int): The number of nearest neighbors to retrieve.

        Returns:
        A dictionary with the following keys:
        - "docs" (List[List[str]]): List of nearest neighbors document text for each query vector. 
            e.g. [["doctext[SENT]doctext", "doctext2[SENT]doctext2"], [..]]
        - "distances" (Tensor): The corresponding distances to the nearest neighbors.
        - "claim_embeds": embeddings for each claim
        """
        # Connect to db
        conn = sqlite3.connect(doc_database)
        wiki_db = conn.cursor()

        if batched:
            with torch.no_grad():
                embeds = self.encoder.encode(query_list, batch_size=128, show_progress_bar=True)
                if self.use_compress:
                    embeds = self.index_compress(embeddings=embeds)
            distances, indices = self.ann.search(embeds, top_k)
            doc_titles = [self.data[idxs].tolist() for idxs in indices]

            # Retrieve top-5 document text for each claim
            doc_list = []
            pbar = tqdm(total=len(query_list))
            for claim_doc_titles in doc_titles:
                claim_doc_texts = []
                for title in claim_doc_titles:
                    doc_text = wiki_db.execute("SELECT text FROM documents WHERE id = ?", 
                                                (unicodedata.normalize('NFD',title),)).fetchone()[0]
                    claim_doc_texts.append(doc_text)
                doc_list.append(claim_doc_texts)
                pbar.update(1)
            pbar.close()
        else:
            distances, doc_list, embeds, doc_titles = [], [], [], []
            for query in tqdm(query_list, desc="Sequential retrieval"):
                with torch.no_grad():
                    claim_embed = self.encoder.encode([query])
                    if self.use_compress:
                        claim_embed = self.index_compress(claim_embed)
                D, I = self.ann.search(claim_embed, top_k)
                distances.extend(D)
                embeds.extend(claim_embed)

                # Retrieve top-5 document text for each claim
                titles = self.data[I[0]].tolist()
                doc_titles.append(titles)
                claim_doc_texts = []
                for title in titles:
                    doc_text = wiki_db.execute("SELECT text FROM documents WHERE id = ?", 
                                                (unicodedata.normalize('NFD', title),)).fetchone()[0]
                    claim_doc_texts.append(doc_text)
                doc_list.append(claim_doc_texts)
        conn.close()

        top_nn = {"doc_titles": doc_titles, 
                "docs": doc_list, 
                "distances": distances, 
                "claim_embeds": embeds}
        return top_nn
