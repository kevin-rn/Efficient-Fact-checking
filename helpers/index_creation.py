import os
import csv
from datasets import load_dataset
import faiss
import json
import logging
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_cohore_wiki = "../data/cohore_wiki"
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

    def get_top_n_neighbours(self, query_vector: torch.Tensor, top_k: int) -> Dict:
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


def cohore_wiki_docs() -> Tuple[List, List]:
    # Create the directory if it doesn't exist
    if not os.path.exists(dir_cohore_wiki):
        os.makedirs(dir_cohore_wiki)

    docs_stream = load_dataset(
        "Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True
    )

    doc_list = []
    doc_embeds = []

    try:
        for doc in docs_stream:
            doc_list.append(doc["text"])
            doc_embeds.append(doc["emb"])

        with open(f"{dir_cohore_wiki}/wiki_docs.json", "w+") as file:
            json.dump(doc_list, file)

    except Exception as e:
        print("exception: ", e, "at ", doc["id"])

    doc_embeds = torch.tensor(doc_embeds)
    torch.save(doc_embeds, f"{dir_cohore_wiki}/wiki_embeddings.pt")

    return doc_list, doc_embeds


def main():
    """
    The main function for setting up and using the FaissSearch class.

    This function does the following:
    1. Downloads Wikipedia data if not available locally.
    2. Initializes a FaissSearch object using the embedding dimension.
    3. Attempts to load an existing Faiss index, and if not available, creates one.

    """
    # Download Wikipedia data if not available
    if not os.path.exists(f"{dir_cohore_wiki}/wiki_docs.json"):
        doc_list, doc_emb = cohore_wiki_docs()
    else:
        with open(f"{dir_cohore_wiki}/wiki_docs.json") as file:
            doc_list = json.load(file)
        doc_emb = torch.load(f"{dir_cohore_wiki}/wiki_embeddings.pt")

    # docs_stream = load_dataset("Cohere/wikipedia-22-12-en-embeddings")
    # dataset = dataset["train"]
    # text_data = dataset["text"]
    # passage_vectors = [np.array(row) for row in dataset["emb"]]
    # passage_vectors = np.vstack(passage_vectors)

    emb_dim = doc_emb.shape[-1]

    faiss_search = FaissSearch(doc_list, emb_dim)
    if not faiss_search.load_index_if_available():
        faiss_search.create_index(doc_emb)


if __name__ == "__main__":
    main()
