from argparse import ArgumentParser
import bz2
from itertools import chain
import json
import jsonlines
from typing import List, Any
from monitor_utils import format_size
from npy_append_array import NpyAppendArray
import numpy as np
import pandas as pd
import os
import sqlite3
from tqdm import tqdm
import torch
import unicodedata
from text_processing_utils import search_file_paths, remove_html_tags
from sentence_transformers import SentenceTransformer

### Argument parsing
### E.g. python data_processing.py --setting=cite --split_sent --first_para_only --pre_compute_embed
parser = ArgumentParser()
parser.add_argument(
    "--setting",
    type=str,
    required=True,
    help="Name of the corresponding setting to retrieve documents from e.g. cite-first, cite-full, cite-first-embed"
)

parser.add_argument(
    "--split_sent",
    action='store_true',
    help="Store every sentence as separate entry instead per whole document text"
)

parser.add_argument(
    "--first_para_only",
    action='store_true',
    help="Only use text from the first paragraph instead of whole document"
)

parser.add_argument(
    "--pre_compute_embed",
    action='store_true',
    help="Pre-compute embeddings for a setting"
)

args = parser.parse_args()

### DATA ###
BASE_PATH = os.path.join("..", "baselines", "hover", "data")
ENWIKI_FOLDER= os.path.join(BASE_PATH, "enwiki_files")
EMBED_FOLDER = os.path.join(BASE_PATH, "embed_files")
suffix = "-first" if args.first_para_only else "-full" 
sent_split = "-sent" if args.split_sent else ""
FILENAME = args.setting + suffix + sent_split
DB_PATH = os.path.join(BASE_PATH, "db_files", f"wiki_wo_links-{FILENAME}.db")

### MODEL ###
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device=device
)

class CorpusCreator:
    def __init__(self, data_dir="data/QA-dataset/data"):
        """
        Initialize a CorpusCreator object with the specified data directory.

        Args:
            data_dir (str): The directory where the data files are located.
        """
        self.data_dir = data_dir
        self.df_wiki_corpus = None
        self.df_musique_corpus = None

    def __read_wiki_data(self):
        """
        Load 2WikiMultiHopQA data and preprocess it.
        """
        df_wiki_train = pd.read_json(f"{self.data_dir}/2wikimultihopQA/train.json")
        df_wiki_dev = pd.read_json(f"{self.data_dir}/2wikimultihopQA/dev.json")
        df_wiki_test = pd.read_json(f"{self.data_dir}/2wikimultihopQA/test.json")

        df_wiki = pd.concat([df_wiki_train, df_wiki_dev, df_wiki_test])
        df_wiki.drop(columns=["type", "supporting_facts", "evidences"], inplace=True)

        df_wiki = df_wiki.explode("context", ignore_index=True)
        self.df_wiki_corpus = df_wiki["context"].apply(
            lambda row: row[0] + " - " + " ".join(row[1])
        )
        self.df_wiki_corpus.drop_duplicates(ignore_index=True, inplace=True)

    def __read_musique_data(self):
        """
        Load MuSiQueQA data and preprocess it.
        """
        df_musique_train = pd.read_json(
            f"{self.data_dir}/musique/musique_full_v1.0_train.jsonl", lines=True
        )
        df_musique_dev = pd.read_json(
            f"{self.data_dir}/musique/musique_full_v1.0_dev.jsonl", lines=True
        )
        df_musique_test = pd.read_json(
            f"{self.data_dir}/musique/musique_full_v1.0_test.jsonl", lines=True
        )

        df_musique = pd.concat([df_musique_train, df_musique_dev, df_musique_test])
        df_musique.drop(
            columns=["question_decomposition", "answer_aliases", "answerable"],
            inplace=True,
        )

        df_musique = df_musique.explode("paragraphs", ignore_index=True)
        self.df_musique_corpus = df_musique["paragraphs"].apply(
            lambda row: row["title"] + " - " + row["paragraph_text"]
        )
        self.df_musique_corpus.drop_duplicates(ignore_index=True, inplace=True)

    def create_corpus(self):
        """
        Combine the two corpora into a single corpus, shuffle it, and save it as an npy file.
        """
        self.__read_wiki_data()
        self.__read_musique_data()
        df_corpus = pd.concat(
            [self.df_wiki_corpus.astype(str), self.df_musique_corpus.astype(str)]
        )
        df_corpus = df_corpus.sample(frac=1, random_state=42)
        df_corpus = df_corpus.reset_index(drop=True)

        np_corpus = df_corpus.to_numpy()
        for i in range(len(np_corpus)):
            np_corpus[i] = np_corpus[i].encode()

        np.save("data/wikimusique_corpus.npy", np_corpus)

def construct_db_file(setting: str, split_sent: bool) -> None:
    """
    Generates sqlite db file from enwiki bz2 folders
    Columns are (id, text, embed) with embed else  (id, text)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create SQL table
    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        text TEXT
                    )
                    """)

    path_location = os.path.join(ENWIKI_FOLDER, "enwiki-2017-" + args.setting)
    text = "text" if "original" in setting else "fact_text"

    file_paths = search_file_paths(path_location)
    for file_path in tqdm(file_paths):
        bz2_path = path_location + file_path
        wiki_values = []
        with bz2.open(bz2_path, "rt") as file:
            for line in file:
                wiki_article = json.loads(line)
                title = wiki_article["title"]
                doc_title = unicodedata.normalize('NFD', title)

                wiki_text = wiki_article[text][1:]
                if args.first_para_only:
                    paragraphs = []
                    for para in wiki_text:
                        if para:
                            paragraphs = remove_html_tags(para)
                            break
                else:
                    paragraphs = list(chain.from_iterable(wiki_text))
                    paragraphs = remove_html_tags(paragraphs)
                
                if split_sent:
                    for idx, sent in enumerate(paragraphs):
                        sent_title = f"{doc_title}_{idx}"
                        wiki_values.append((sent_title, sent))
                else:
                    doc_text = "[SENT]".join(paragraphs)
                    wiki_values.append((doc_title, doc_text))
            cursor.executemany("INSERT OR IGNORE INTO documents (id, text) VALUES (?, ?)", wiki_values)
        conn.commit()

    # Rebuilds the database file, repacking it into a minimal amount of disk space
    cursor.execute("VACUUM;")
    conn.commit()
    conn.close()

def pre_compute_embed(setting: str) -> None:
    """
    Pre-compute embeddings for all document text in database.
    """
    def divide_n_chunks(docs: List[Any], n: int):
        "Divides List into n-chunks"
        k, m = divmod(len(docs), n)
        return (docs[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    # Retrieve documents from database file.
    conn = sqlite3.connect(DB_PATH)
    wiki_db = conn.cursor()
    results = wiki_db.execute("SELECT text FROM documents ORDER BY id COLLATE NOCASE ASC").fetchall()
    doc_text_list = [str(doc[0]).replace("[SENT]", "") for doc in results]
    conn.close()

    # Chunk list if larger than 10 million entries
    documents = list(divide_n_chunks(doc_text_list, 8)) if len(doc_text_list) > 10**7 else [doc_text_list]

    # Encode document text and store them to numpy file.
    embed_path = os.path.join(EMBED_FOLDER, FILENAME + ".npy")
    
    with torch.no_grad():
        for doc_text in documents:
            embed_arr = encoder.encode(doc_text, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
        with NpyAppendArray(embed_path) as file:
            file.append(embed_arr)
    # data = np.load(filename, mmap_mode="r")

def get_raw_size(setting: str) -> None:
    """
    Get raw corpus size of a setting
    """
    # Retrieve data from database.
    conn = sqlite3.connect(DB_PATH)
    wiki_db = conn.cursor()
    results = wiki_db.execute("SELECT id, text FROM documents ORDER BY id COLLATE NOCASE ASC").fetchall()
    conn.close()

    # Create dict object for each entry and count total number of sentences.
    json_obj = []
    total_sents = 0
    for title, text in tqdm(results, desc="Entry loop"):
        wiki_store = {"title": title, "text": text.replace("[SENT]", "")}
        json_obj.append(wiki_store)
        total_sents += len(text.split("[SENT]"))

    # Store to jsonlines file.
    raw_path = os.path.join(BASE_PATH, "raw_files", FILENAME + "-raw.jsonl")
    with jsonlines.open(raw_path, 'a') as writer:
        writer.write_all(json_obj)

    # Measure size.
    size = os.path.getsize(raw_path)
    print(f"{setting} Corpus size: {format_size(size)}, {total_sents} sentences")

    # Cleanup
    # os.remove(raw_path)

def just_cite():
    original_path = os.path.join(BASE_PATH, "db_files", "wiki_wo_links-original-full.db")
    conn = sqlite3.connect(original_path)
    wiki_db = conn.cursor()
    results = wiki_db.execute("SELECT id, text FROM documents ORDER BY id COLLATE NOCASE ASC").fetchall()
    conn.close()

    cite_path = os.path.join(BASE_PATH, "db_files", "wiki_wo_links-original-full.db")
    conn = sqlite3.connect(cite_path)
    wiki_db = conn.cursor()

    no_changes, changes = 0, 0
    for title, original_text in results:
        title = unicodedata.normalize('NFD', title)
        cite_text = wiki_db.execute("SELECT text FROM documents WHERE id = ?", 
                                              (title ,)).fetchone()[0]
        if original_text == cite_text:
            changes += 1
        else:
            no_changes += 1

    conn.close()

def main():
    run_setting=args.setting
    # construct_db_file(setting=run_setting, split_sent=args.split_sent)
    if args.pre_compute_embed:
        pre_compute_embed(setting=run_setting)
    # get_raw_size(setting=run_setting)

if __name__ == "__main__":
    main()