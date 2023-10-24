import os
from datasets import load_dataset, load_from_disk
import faiss
import logging
import numpy as np
from typing import List, Dict
from torch import Tensor

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_faiss_idx = "../data/indices/faiss"


class FaissSearch:
    def __init__(self, data: List[str] = None, emb_dim=None):
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

    def get_top_n_neighbours(self, query_vector: Tensor, top_k: int) -> Dict:
        """
        Search for the top nearest neighbors to a given query vector.

        Parameters:
        - query_vector (Tensor): The query vector for which nearest neighbors are to be found.
        - top_k (int): The number of nearest neighbors to retrieve.

        Returns:
        A dictionary with the following keys:
        - "ids" (List[List[int]]): List of lists containing the indices of nearest neighbors for each query vector.
        - "distances" (Tensor): The corresponding distances to the nearest neighbors.

        """
        distances, indices = self.ann.search(query_vector, top_k)
        assert indices.shape == distances.shape == (query_vector.shape[0], 100)
        indices = indices.tolist()
        return {"ids": indices, "distances": distances}

    def load_index_if_available(self) -> bool:
        """
        Load a pre-existing Faiss index if available.

        Returns:
        bool: True if the index was successfully loaded, False if no index is found.

        """
        if os.path.exists(f"{dir_faiss_idx}/index_faiss"):
            self.ann.load(f"{dir_faiss_idx}/index_faiss")
            return True
        else:
            return False

    def create_index(self, passage_vectors):
        """
        Create a Faiss index with the given passage vectors.

        Parameters:
        - passage_vectors: A collection of passage vectors to build the index.

        """
        self.ann.add(passage_vectors)
        if not os.path.exists(dir_faiss_idx):
            os.makedirs(dir_faiss_idx)
        faiss.write_index(self.ann, f"{dir_faiss_idx}/index_faiss")


def main():
    """
    The main function for setting up and using the FaissSearch class.

    This function does the following:
    1. Downloads Wikipedia data if not available locally.
    2. Initializes a FaissSearch object using the embedding dimension.
    3. Attempts to load an existing Faiss index, and if not available, creates one.

    """

    # Download Wikipedia data if not available
    if not os.path.exists("../data/cohore_wiki_dataset"):
        dataset = load_dataset(
            "Cohere/wikipedia-22-12-en-embeddings", cache_dir="../data/cache"
        )
        dataset.save_to_disk("../data/cohore_wiki_dataset")
        dataset.cleanup_cache_files()
    else:
        dataset = load_from_disk("../data/cohore_wiki_dataset")

    dataset = dataset["train"]
    text_data = dataset["text"]
    passage_vectors = [np.array(row) for row in dataset["emb"]]
    passage_vectors = np.vstack(passage_vectors)

    emb_dim = passage_vectors.shape[1]
    faiss_search = FaissSearch(text_data, emb_dim)
    if not faiss_search.load_index_if_available():
        faiss_search.create_index(passage_vectors)


if __name__ == "__main__":
    main()
