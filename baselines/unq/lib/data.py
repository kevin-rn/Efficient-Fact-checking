import os
import warnings

import json
import numpy as np
import torch
import random
from .utils import fvecs_read, download
import os.path as osp
import sqlite3
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import unicodedata

DATA_PATH = os.path.join("..", "..", "hover", "data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device=device
)

class Dataset:

    def __init__(self, dataset, data_path='./data', normalize=False, random_state=50, **kwargs):
        """
        Dataset is a bunch of tensors with all the learning and evaluation data required for an experiment
        :param dataset: a pre-defined dataset name (see DATSETS) or a custom dataset
            Your dataset should be at (or will be downloaded into) {data_path}/{dataset}
        :param data_path: a shared data folder path where the dataset is stored (or will be downloaded into)
        :param random_state: global random seed for an experiment
        :param normalize: if True, divides all data points by an average l2 norm of train_vectors
        :param kwargs: depending on the dataset, you may select train size, test size or other params
            If dataset is not in DATASETS, provide three keys: train_vectors, test_vectors and query_vectors

        """
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)

        if dataset in DATASETS:
            data_dict = DATASETS[dataset](osp.join(data_path, dataset), **kwargs)
        else:
            assert all(key in kwargs for key in ('train_vectors', 'test_vectors', 'query_vectors'))
            data_dict = kwargs

        self.train_vectors = torch.as_tensor(data_dict['train_vectors'])
        self.test_vectors = torch.as_tensor(data_dict['test_vectors'])
        self.query_vectors = torch.as_tensor(data_dict['query_vectors'])
        assert self.train_vectors.shape[1] == self.test_vectors.shape[1] == self.query_vectors.shape[1]
        self.vector_dim = self.train_vectors.shape[1]

        mean_norm = self.train_vectors.norm(p=2, dim=-1).mean().item()
        if normalize:
            self.train_vectors /= mean_norm
            self.test_vectors /= mean_norm
            self.query_vectors /= mean_norm
        else:
            if mean_norm < 0.1 or mean_norm > 10.0:
                warnings.warn("Mean train_vectors norm is {}, consider normalizing")


def fetch_DEEP1M(path, train_size=5 * 10 ** 5, test_size=10 ** 6, ):
    base_path = osp.join(path, 'deep_base1M.fvecs')
    learn_path = osp.join(path, 'deep_learn500k.fvecs')
    query_path = osp.join(path, 'deep_query10k.fvecs')
    if not all(os.path.exists(fname) for fname in (base_path, learn_path, query_path)):
        os.makedirs(path, exist_ok=True)
        download("https://www.dropbox.com/s/e23sdc3twwn9syk/deep_base1M.fvecs?dl=1", base_path,
                 chunk_size=4 * 1024 ** 2)
        download("https://www.dropbox.com/s/4i0c5o8jzvuloxy/deep_learn500k.fvecs?dl=1", learn_path,
                 chunk_size=4 * 1024 ** 2)
        download("https://www.dropbox.com/s/5z087cxqh61n144/deep_query10k.fvecs?dl=1", query_path)
    return dict(
        train_vectors=fvecs_read(learn_path)[:train_size],
        test_vectors=fvecs_read(base_path)[:test_size],
        query_vectors=fvecs_read(query_path)
    )


def fetch_BIGANN1M(path, train_size=None, test_size=None):
    base_path = osp.join(path, 'bigann_base1M.fvecs')
    learn_path = osp.join(path, 'bigann_learn500k.fvecs')
    query_path = osp.join(path, 'bigann_query10k.fvecs')
    if not all(os.path.exists(fname) for fname in (base_path, learn_path, query_path)):
        os.makedirs(path, exist_ok=True)
        download("https://www.dropbox.com/s/zcnvsy7mlogj4g0/bigann_base1M.fvecs?dl=1", base_path,
                 chunk_size=4 * 1024 ** 2)
        download("https://www.dropbox.com/s/dviygi2zhk57p9m/bigann_learn500k.fvecs?dl=1", learn_path,
                 chunk_size=4 * 1024 ** 2)
        download("https://www.dropbox.com/s/is6anxwon6g5bpe/bigann_query10k.fvecs?dl=1", query_path)
    return dict(
        train_vectors=fvecs_read(learn_path)[:train_size],
        test_vectors=fvecs_read(base_path)[:test_size],
        query_vectors=fvecs_read(query_path)
    )

def fetch_ENWIKI(path, train_size=5 * 10 ** 5, test_size=10 ** 6, setting="original-full"):
    data_train_path = os.path.join("data", "ENWIKI", f"{setting}_train_dataset.pt")
    data_test_path = os.path.join("data", "ENWIKI", f"{setting}_test_dataset.pt")
    query_path = os.path.join("data", "ENWIKI", f"{setting}_query_dataset.pt")

    if os.path.exists(query_path) and os.path.exists(data_train_path) and os.path.exists(data_test_path):
        print("load tensor")
        query_dataset = torch.load(query_path).to(torch.float)
        train_dataset = torch.load(data_train_path).to(torch.float)
        test_dataset = torch.load(data_test_path).to(torch.float)
    else:
        print("create tensor")   
        train_titles, dev_titles, claims = [], [], []
        hover_train_path = os.path.join(DATA_PATH, "hover", "hover_train_release_v1.1.json")
        hover_dev_path = os.path.join(DATA_PATH, "hover", "hover_dev_release_v1.1.json")
            
        with open(hover_train_path, 'r') as json_file:
            claim_json = json.load(json_file)
            claims = [claim for claim in claim_json]
            train_titles = list(set([unicodedata.normalize('NFD', supporting[0]) for doc in claim_json for supporting in doc['supporting_facts']]))
            with torch.no_grad():
                query_dataset = encoder.encode(claims, batch_size=128, 
                                            convert_to_tensor=True, 
                                            show_progress_bar=True).to(torch.float)

        with open(hover_dev_path, 'r') as json_file:
            claim_json = json.load(json_file)
            dev_titles = list(set([supporting[0] for doc in claim_json for supporting in doc['supporting_facts']]))
        db_path = os.path.join(DATA_PATH, "db_files", f"wiki_wo_links-{setting}.db")
        conn = sqlite3.connect(db_path)
        wiki_db = conn.cursor()

        pre_compute_path = os.path.join(DATA_PATH, "embed_files", setting+".npy")
        if os.path.exists(pre_compute_path):
            print("Load pre-computed embeddings")
            embed_list = np.load(pre_compute_path)
            title_list = wiki_db.execute("SELECT id FROM documents ORDER BY id COLLATE NOCASE ASC").fetchall()
        else:
            results = wiki_db.execute("SELECT id, text FROM documents ORDER BY id COLLATE NOCASE ASC").fetchall()
            title_list = [title[0] for title in results]
            text_list = [text[1] for text in results]
            with torch.no_grad():
                embed_list = encoder.encode(text_list, batch_size=128, show_progress_bar=True)
        conn.close()

        index_list, support_embeds = [], []
        for doc_titles in [train_titles, dev_titles]:
            indices = []
            for doc_title in tqdm(doc_titles):
                try:
                    id = title_list.index(doc_title)
                    indices.append(id)
                except ValueError:
                    pass
        
            # Generate arrays containing and without the supporting facts
            support_embeds.append(embed_list[indices])
            index_list.extend(indices)

        # Fill the dataset with other non-supporting documents to increase size.
        non_support_embeds = np.delete(embed_list, list(set(index_list)), axis=0)
        non_support_indices = np.arange(len(non_support_embeds))
        train_fill = np.random.choice(non_support_indices, size=5*10**5 - len(support_embeds[0]))
        non_support_indices = np.delete(non_support_indices, train_fill)
        dev_vals = np.random.choice(non_support_indices, size=10**6 - len(support_embeds[1]))
        train_dataset = torch.from_numpy(np.concatenate([support_embeds[0], non_support_embeds[train_fill]])).to(torch.float)
        dev_dataset = torch.from_numpy(np.concatenate([support_embeds[1], non_support_embeds[dev_vals]])).to(torch.float)
        train_dataset=train_dataset[torch.randperm(train_dataset.size()[0])]
        test_dataset=test_dataset[torch.randperm(test_dataset.size()[0])]

        # Save to disk
        torch.save(query_dataset, query_path)
        torch.save(train_dataset, data_train_path)
        torch.save(test_dataset, data_test_path)

    return dict(
        train_vectors=train_dataset,
        test_vectors=test_dataset,
        query_vectors=query_dataset
    )

DATASETS = {
    'DEEP1M': fetch_DEEP1M,
    'BIGANN1M': fetch_BIGANN1M,
    'ENWIKI': fetch_ENWIKI,
}
