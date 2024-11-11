import contextlib
import json
import os
import os.path as osp
import random
import sqlite3
import unicodedata
import warnings

import h5py
import joblib
import numpy as np
import torch
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


class Dataset:

    def __init__(
        self, dataset, data_path="./data/unq", normalize=False, random_state=50, **kwargs
    ):
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
            assert all(
                key in kwargs
                for key in ("train_vectors", "test_vectors", "query_vectors")
            )
            data_dict = kwargs

        if "ENWIKI" == dataset:
            self.train_query_vectors = torch.as_tensor(data_dict["train_query_vectors"])
            self.train_data_vectors = torch.as_tensor(data_dict["train_data_vectors"])
            self.train_nn_indices = torch.as_tensor(data_dict["train_nn_indices"])
            self.test_query_vectors = torch.as_tensor(data_dict["test_query_vectors"])
            self.test_data_vectors = torch.as_tensor(data_dict["test_data_vectors"])
            self.test_nn_indices = torch.as_tensor(data_dict["train_nn_indices"])

            self.vector_dim = self.train_query_vectors.shape[1]

            mean_norm = self.train_data_vectors.norm(p=2, dim=-1).mean().item()
            if normalize:
                self.train_query_vectors /= mean_norm
                self.train_data_vectors /= mean_norm
                self.test_query_vectors /= mean_norm
                self.test_data_vectors /= mean_norm
            else:
                if mean_norm < 0.1 or mean_norm > 10.0:
                    warnings.warn("Mean train_vectors norm is {}, consider normalizing")

        else:
            self.train_vectors = torch.as_tensor(data_dict["train_vectors"])
            self.test_vectors = torch.as_tensor(data_dict["test_vectors"])
            self.query_vectors = torch.as_tensor(data_dict["query_vectors"])

            assert (
                self.train_vectors.shape[1]
                == self.test_vectors.shape[1]
                == self.query_vectors.shape[1]
            )
            self.vector_dim = self.train_vectors.shape[1]

            mean_norm = self.train_vectors.norm(p=2, dim=-1).mean().item()
            if normalize:
                self.train_vectors /= mean_norm
                self.test_vectors /= mean_norm
                self.query_vectors /= mean_norm
            else:
                if mean_norm < 0.1 or mean_norm > 10.0:
                    warnings.warn("Mean train_vectors norm is {}, consider normalizing")


def fetch_DEEP1M(
    path,
    train_size=5 * 10**5,
    test_size=10**6,
):
    base_path = osp.join(path, "deep_base1M.fvecs")
    learn_path = osp.join(path, "deep_learn500k.fvecs")
    query_path = osp.join(path, "deep_query10k.fvecs")
    if not all(os.path.exists(fname) for fname in (base_path, learn_path, query_path)):
        os.makedirs(path, exist_ok=True)
        download(
            "https://www.dropbox.com/s/e23sdc3twwn9syk/deep_base1M.fvecs?dl=1",
            base_path,
            chunk_size=4 * 1024**2,
        )
        download(
            "https://www.dropbox.com/s/4i0c5o8jzvuloxy/deep_learn500k.fvecs?dl=1",
            learn_path,
            chunk_size=4 * 1024**2,
        )
        download(
            "https://www.dropbox.com/s/5z087cxqh61n144/deep_query10k.fvecs?dl=1",
            query_path,
        )
    return dict(
        train_vectors=fvecs_read(learn_path)[:train_size],
        test_vectors=fvecs_read(base_path)[:test_size],
        query_vectors=fvecs_read(query_path),
    )


def fetch_BIGANN1M(path, train_size=None, test_size=None):
    base_path = osp.join(path, "bigann_base1M.fvecs")
    learn_path = osp.join(path, "bigann_learn500k.fvecs")
    query_path = osp.join(path, "bigann_query10k.fvecs")
    if not all(os.path.exists(fname) for fname in (base_path, learn_path, query_path)):
        os.makedirs(path, exist_ok=True)
        download(
            "https://www.dropbox.com/s/zcnvsy7mlogj4g0/bigann_base1M.fvecs?dl=1",
            base_path,
            chunk_size=4 * 1024**2,
        )
        download(
            "https://www.dropbox.com/s/dviygi2zhk57p9m/bigann_learn500k.fvecs?dl=1",
            learn_path,
            chunk_size=4 * 1024**2,
        )
        download(
            "https://www.dropbox.com/s/is6anxwon6g5bpe/bigann_query10k.fvecs?dl=1",
            query_path,
        )
    return dict(
        train_vectors=fvecs_read(learn_path)[:train_size],
        test_vectors=fvecs_read(base_path)[:test_size],
        query_vectors=fvecs_read(query_path),
    )


def get_claim_splits():
    """
    Get claim and supporting titles and splits them in a 7-3 split
    """
    train_titles, claims = [], []
    hover_train_path = os.path.join("data", "hover", "hover_train_release_v1.1.json")
    with open(hover_train_path, "r") as json_file:
        claim_json = json.load(json_file)
        claims = [claim for claim in claim_json]
        train_titles = [
            list(
                set(
                    [
                        unicodedata.normalize("NFD", supporting[0])
                        for supporting in doc["supporting_facts"]
                    ]
                )
            )
            for doc in claim_json
        ]
        with torch.no_grad():
            query_dataset = encoder.encode(
                claims, batch_size=128, convert_to_numpy=True, show_progress_bar=True
            )

    values = list(zip(train_titles, query_dataset))
    split_size = (len(values) // 10) * 7
    train_values = values[:split_size]
    test_values = values[split_size:]
    return train_values, test_values


def retrieve_doc_embeds(setting):
    """
    Get all titles and document embeddings
    """
    db_path = os.path.join("data", "db_files", f"wiki_wo_links-{setting}.db")
    pre_compute_path = os.path.join("data", "embed_files", setting + ".h5")
    conn = sqlite3.connect(db_path)
    wiki_db = conn.cursor()
    if os.path.exists(pre_compute_path):
        embed_list = []
        with h5py.File(pre_compute_path, "r") as hf:
            for group_name in hf.keys():
                embed_list = hf[group_name][:]
        title_list = wiki_db.execute(
            "SELECT id FROM documents ORDER BY id COLLATE NOCASE ASC"
        ).fetchall()
        title_list = [title[0] for title in title_list]
    else:
        title_list, text_list = wiki_db.execute(
            "SELECT id, text FROM documents ORDER BY id COLLATE NOCASE ASC"
        ).fetchall()
        with torch.no_grad():
            embed_list = encoder.encode(
                text_list, batch_size=128, show_progress_bar=True
            )
    conn.close()
    return title_list, embed_list


def form_datasets(
    train_values, test_values, title_list, embed_list, train_size, test_size
):
    """
    Forms the train and test datasets concisting of supporting and non-supporting documents.
    """
    # Get all supporting documents for train and test
    lookup_dict = dict(zip(title_list, range(0, len(title_list))))
    support_indices, train_support, train_query, train_dataset = [], [], [], []
    for support_titles, claim_embed in tqdm(train_values, desc="train_values"):
        train_query.append(claim_embed)
        indices = [lookup_dict[title] for title in support_titles]
        support_indices.extend(indices)
        embed_vals = embed_list[indices]
        train_support.append(embed_vals)
        train_dataset.extend(embed_vals)

    test_support, test_query, test_dataset = [], [], []
    for support_titles, claim_embed in tqdm(test_values, desc="test_values"):
        test_query.append(claim_embed)
        indices = [lookup_dict[title] for title in support_titles]
        support_indices.extend(indices)
        embed_vals = embed_list[indices]
        test_support.append(embed_vals)
        test_dataset.extend(embed_vals)

    train_dataset = np.unique(train_dataset, axis=0)
    test_dataset = np.unique(test_dataset, axis=0)
    support_indices = np.unique(support_indices)

    # Fill the dataset with non-supporting documents.
    indices, non_support_embeds = zip(
        *enumerate(np.delete(embed_list, support_indices, axis=0))
    )
    fill_indices = np.random.choice(indices, size=train_size - len(train_dataset))
    non_support_embeds = np.array(non_support_embeds)
    train_dataset = np.concatenate([train_dataset, non_support_embeds[fill_indices]])
    indices, non_support_embeds = zip(
        *enumerate(np.delete(non_support_embeds, fill_indices, axis=0))
    )
    fill_indices = np.random.choice(indices, size=test_size - len(test_dataset))
    non_support_embeds = np.array(non_support_embeds)
    test_dataset = np.concatenate([test_dataset, non_support_embeds[fill_indices]])
    np.random.shuffle(train_dataset)
    np.random.shuffle(test_dataset)

    # Get support indices from new dataset splits and return final dataset values.
    with tqdm_joblib(
        tqdm(desc="Retrieval train support", total=len(train_support))
    ) as progress_bar:
        train_nn_indices = Parallel(n_jobs=16)(
            delayed(find_indices)(support_embeds, train_dataset)
            for support_embeds in train_support
        )

    with tqdm_joblib(
        tqdm(desc="Retrieval test support", total=len(test_support))
    ) as progress_bar:
        test_nn_indices = Parallel(n_jobs=16)(
            delayed(find_indices)(support_embeds, test_dataset)
            for support_embeds in test_support
        )

    # Pad with -1 as the amount of supporting documents are between 2-4
    train_nn_indices = np.array(
        [sublist.tolist() + [-1] * (4 - len(sublist)) for sublist in train_nn_indices],
        dtype=int,
    )
    test_nn_indices = np.array(
        [sublist.tolist() + [-1] * (4 - len(sublist)) for sublist in test_nn_indices],
        dtype=int,
    )

    values = [
        train_query,
        train_dataset,
        train_nn_indices,
        test_query,
        test_dataset,
        test_nn_indices,
    ]
    return values


def fetch_ENWIKI(path, train_size=5 * 10**5, test_size=10**6, setting="original-full"):
    keys = [
        "train_query_vectors",
        "train_data_vectors",
        "train_nn_indices",
        "test_query_vectors",
        "test_data_vectors",
        "test_nn_indices",
    ]
    file_path = os.path.join(path, setting + ".h5")
    if os.path.isfile(file_path):
        values = []
        with h5py.File(file_path, "r") as hf:
            for group_name in keys:
                vectors = hf[group_name][:]
                values.append(vectors)
    else:
        train_values, test_values = get_claim_splits()
        title_list, embed_list = retrieve_doc_embeds(setting=setting)
        values = form_datasets(
            train_values=train_values,
            test_values=test_values,
            title_list=title_list,
            embed_list=embed_list,
            train_size=train_size,
            test_size=test_size,
        )

        for key, value in zip(keys, values):
            with h5py.File(file_path, "a") as hf:
                hf.create_dataset(key, data=value)

    return dict(zip(keys, values))


DATASETS = {
    "DEEP1M": fetch_DEEP1M,
    "BIGANN1M": fetch_BIGANN1M,
    "ENWIKI": fetch_ENWIKI,
}


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    see https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def find_indices(values, dataset):
    indices = [
        np.where(np.all(np.equal(dataset, value), axis=1))[0][0] for value in values
    ]
    return np.array(indices)
