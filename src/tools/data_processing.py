import bz2
import json
import os
import sqlite3
import unicodedata
from argparse import ArgumentParser
from itertools import chain
from typing import Any, List

import h5py
import jsonlines
import torch
from joblib import Parallel, delayed
from monitor_utils import ProcessMonitor, format_size
from sentence_transformers import SentenceTransformer
from text_processing_utils import (remove_html_tags, search_file_paths,
                                   tqdm_joblib)
from tqdm import tqdm

### Argument parsing
### E.g. python data_processing.py --setting=cite --split_sent --first_para_only --pre_compute_embed
parser = ArgumentParser()
parser.add_argument(
    "--setting",
    type=str,
    required=True,
    help="Name of the corresponding setting to retrieve documents from e.g. cite-first, cite-full, cite-first-embed",
)

parser.add_argument(
    "--split_sent",
    action="store_true",
    help="Store every sentence as separate entry instead per whole document text",
)

parser.add_argument(
    "--first_para_only",
    action="store_true",
    help="Only use text from the first paragraph instead of whole document",
)

parser.add_argument(
    "--pre_compute_embed",
    action="store_true",
    help="Pre-compute vector embeddings for FAISS retrieval",
)

parser.add_argument(
    "--store_original",
    action="store_true",
    help="Use original text instead of just the supporting facts",
)

args = parser.parse_args()

# # ### DATA ###
BASE_PATH = os.path.join(os.path.abspath(os.curdir), "data")
ENWIKI_FOLDER = os.path.join(BASE_PATH, "enwiki_files")
EMBED_FOLDER = os.path.join(BASE_PATH, "embed_files")
is_full = "-first" if args.first_para_only else "-full"
sent_split = "-sent" if args.split_sent else ""
FILENAME = args.setting + is_full + sent_split
DB_PATH = os.path.join(BASE_PATH, "db_files", f"{FILENAME}.db")

### MODEL ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


def read_bz2_file(file_path: str, args: dict):
    bz2_path = args["path_location"] + file_path
    wiki_values = []
    with bz2.open(bz2_path, "rt") as file:
        for line in file:
            wiki_article = json.loads(line)
            title = wiki_article["title"]
            doc_title = unicodedata.normalize("NFD", title)

            # Insert full document text or just the first paragraph.
            wiki_text = wiki_article[args["text"]][1:]
            if args["first_only"]:
                paragraphs = []
                for para in wiki_text:
                    if para:
                        paragraphs = remove_html_tags(para)
                        break
            else:
                paragraphs = list(chain.from_iterable(wiki_text))
                paragraphs = remove_html_tags(paragraphs)

            # Store sentences as joint text or separate.
            if args["split_sent"]:
                for idx, sent in enumerate(paragraphs):
                    if sent.strip():
                        sent_title = f"{doc_title}_{idx}"
                        wiki_values.append((sent_title, sent.strip()))
            else:
                doc_text = "[SENT]".join(
                    [sent.strip() for sent in paragraphs if sent.strip()]
                )
                wiki_values.append((doc_title, doc_text))
    return wiki_values


def construct_db_file(db_args: dict) -> None:
    """
    Generates sqlite db file from enwiki bz2 folders
    Columns are (id, text, embed) with embed else  (id, text)
    """
    file_paths = search_file_paths(db_args["path_location"])
    with tqdm_joblib(
        tqdm(desc=f"Construct {args.setting} database", total=len(file_paths))
    ) as progress_bar:
        results = Parallel(n_jobs=16)(
            delayed(read_bz2_file)(bz2_filepath, db_args) for bz2_filepath in file_paths
        )
    wiki_values = [article for file_data in results for article in file_data]

    # Create SQL table
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        text TEXT
                    )
                    """
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO documents (id, text) VALUES (?, ?)", wiki_values
    )
    conn.commit()

    # Rebuilds the database file, repacking it into a minimal amount of disk space
    cursor.execute("VACUUM;")
    conn.commit()
    conn.close()
    print(f"Finished {FILENAME} database")


def pre_compute_embed() -> None:
    """
    Pre-compute embeddings for all document text in database.
    """

    def divide_n_chunks(docs: List[Any], n: int) -> List[List[Any]]:
        "Divides List into n-chunks"
        k, m = divmod(len(docs), n)
        n_chunks = list(
            (docs[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))
        )
        return n_chunks

    # Retrieve documents from database file.
    conn = sqlite3.connect(DB_PATH)
    wiki_db = conn.cursor()
    results = wiki_db.execute(
        "SELECT text FROM documents ORDER BY id COLLATE NOCASE ASC"
    ).fetchall()
    doc_text_list = [
        " ".join(doc[0].split("[SENT]"))
        for doc in tqdm(results, desc="Get document text")
    ]
    conn.close()

    # Chunk list if larger than 10 million entries
    documents = (
        divide_n_chunks(doc_text_list, 8)
        if len(doc_text_list) > 10**7
        else [doc_text_list]
    )

    # Encode document text and store them to h5 file.
    embed_path = os.path.join(EMBED_FOLDER, FILENAME + ".h5")
    with torch.no_grad(), ProcessMonitor() as pm:
        pm.start()
        for idx, doc_text in enumerate(documents):
            embed_arr = encoder.encode(
                doc_text, batch_size=128, show_progress_bar=True, convert_to_numpy=True
            )
            with h5py.File(embed_path, "a") as hf:
                hf.create_dataset(f"group_{idx}", data=embed_arr)
            del embed_arr
    print(f"Finished {FILENAME} pre-computed embeddings")


def get_raw_size() -> None:
    """
    Get raw corpus size of a setting
    """
    # Retrieve data from database.
    conn = sqlite3.connect(DB_PATH)
    wiki_db = conn.cursor()
    results = wiki_db.execute(
        "SELECT id, text FROM documents ORDER BY id COLLATE NOCASE ASC"
    ).fetchall()
    conn.close()

    # Create dict object for each entry and count total number of sentences.
    json_obj = []
    total_sents = 0
    for title, text in tqdm(results, desc="Raw files"):
        wiki_store = {"title": title, "text": " ".join(text.split("[SENT]"))}
        json_obj.append(wiki_store)
        if text.strip():
            total_sents += len([sent for sent in text.split("[SENT]") if sent.strip()])

    # Store to jsonlines file.
    raw_path = os.path.join("..", "data")
    raw_file = os.path.join(raw_path, FILENAME + ".jsonl")
    if not os.path.exists(raw_path):
        os.makedirs(raw_path, exist_ok=True)

    with jsonlines.open(raw_file, "a") as writer:
        writer.write_all(json_obj)

    # Measure size.
    size = os.path.getsize(raw_file)
    print(f"{FILENAME} Corpus size: {format_size(size)}, {total_sents} sentences")

    # Cleanup
    os.remove(raw_file)


def main():
    db_args = {
        "path_location": os.path.join(ENWIKI_FOLDER, args.setting),
        "text": "text" if args.store_original else "fact_text",
        "split_sent": args.split_sent,
        "first_only": args.first_para_only,
    }
    if os.path.isfile(DB_PATH):
        print("DB file already exists")
    else:
        construct_db_file(db_args)

    # Pre-computes vector embeddings for Faiss Retrieval setup
    if args.pre_compute_embed:
        pre_compute_embed()

    # Calculates Raw corpus size
    get_raw_size()


if __name__ == "__main__":
    main()
