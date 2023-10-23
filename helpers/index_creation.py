import os
from datasets import load_dataset
import faiss
import logging
import pandas as pd
from typing import List, Dict
from torch import Tensor

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))


class FaissSearch:
    def __init__(self, data: List[str] = None, emb_dim=None):
        self.emb_dim = emb_dim
        self.data = data
        self.ann = faiss.IndexFlatIP(self.emb_dim)

    def get_top_n_neighbours(self, query_vector: Tensor, top_k: int) -> Dict:
        distances, indices = self.ann.search(query_vector, top_k)
        assert indices.shape == distances.shape == (query_vector.shape[0], 100)
        indices = indices.tolist()
        # passages = [self.data[idx] for passage_ids in indices for idx in passage_ids]
        return {"ids": indices, "distances": distances}

    def load_index_if_available(self) -> None:
        if os.path.exists("indices/faiss/index_faiss"):
            self.ann.load("indices/faiss/index_faiss")
            return True
        else:
            return False

    def create_index(self, passage_vectors):
        self.ann.add(passage_vectors)
        if not os.path.exists("indices/faiss"):
            os.makedirs("indices/faiss")
        faiss.write_index(self.ann, "indices/faiss/index_faiss")


def main():
    # The embedding dimension of Cohere’s multilingual-22-12 embedding model
    emb_dim = 768

    # Download Wikipedia data if not available
    if not os.path.exists("data/cohore_wikipedia.csv"):
        dataset = load_dataset("Cohere/wikipedia-22-12-en-embeddings")
        df_wiki = pd.DataFrame(dataset["train"])
        df_wiki.to_csv("data/cohore_wikipedia.csv")
    else:
        df_wiki = pd.read_csv("data/cohore_wikipedia.csv")

    faiss_search = FaissSearch(df_wiki, emb_dim)

    faiss_search.load_index_if_available()
    if not faiss_search.ann.is_trained:
        faiss_search.create_index(df_wiki)


if __name__ == "__main__":
    main()
