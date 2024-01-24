from argparse import ArgumentParser
import bz2
from itertools import chain
import json
import jsonlines
import h5py
from typing import List, Any
from monitor_utils import format_size
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

# # ### DATA ###
BASE_PATH = os.path.join("..", "baselines", "hover", "data")
ENWIKI_FOLDER= os.path.join(BASE_PATH, "enwiki_files")
EMBED_FOLDER = os.path.join(BASE_PATH, "embed_files")
suffix = "-first" if args.first_para_only else "-full" 
sent_split = "-sent" if args.split_sent else ""
FILENAME = args.setting + suffix + sent_split
FILENAME = "claim-full-sent"
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
    for idx, file_path in tqdm(enumerate(file_paths), total=len(file_paths)):
        bz2_path = path_location + file_path
        wiki_values = []
        with bz2.open(bz2_path, "rt") as file:
            for line in file:
                wiki_article = json.loads(line)
                title = wiki_article["title"]
                doc_title = unicodedata.normalize('NFD', title)

                # Insert full document text or just the first paragraph.
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
                
                # Store sentences as joint text or separate.
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

def pre_compute_embed() -> None:
    """
    Pre-compute embeddings for all document text in database.
    """
    def divide_n_chunks(docs: List[Any], n: int) -> List[List[Any]]:
        "Divides List into n-chunks"
        k, m = divmod(len(docs), n)
        n_chunks = list((docs[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)))
        return n_chunks

    # Retrieve documents from database file.
    conn = sqlite3.connect(DB_PATH)
    wiki_db = conn.cursor()
    results = wiki_db.execute("SELECT text FROM documents ORDER BY id COLLATE NOCASE ASC").fetchall()
    doc_text_list = [str(doc[0]).replace("[SENT]", " ") for doc in results]
    conn.close()

    # Chunk list if larger than 10 million entries
    documents = divide_n_chunks(doc_text_list, 8) if len(doc_text_list) > 10**7 else [doc_text_list]

    # Encode document text and store them to h5 file.
    embed_path = os.path.join(EMBED_FOLDER, FILENAME+".h5")
    with torch.no_grad():
        for idx, doc_text in enumerate(documents):
            embed_arr = encoder.encode(doc_text, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
            with h5py.File(embed_path, 'a') as hf:
                hf.create_dataset(f"group_{idx}",  data=embed_arr)
            del embed_arr

def get_raw_size() -> None:
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
        wiki_store = {"title": title, "text": text.replace("[SENT]", " ")}
        json_obj.append(wiki_store)
        if text.strip():
            total_sents += len([sent for sent in text.split("[SENT]") if sent.strip()])

    # Store to jsonlines file.
    raw_path = os.path.join(BASE_PATH, "raw_files", FILENAME + ".jsonl")
    with jsonlines.open(raw_path, 'a') as writer:
        writer.write_all(json_obj)

    # Measure size.
    size = os.path.getsize(raw_path)
    print(f"Corpus size: {format_size(size)}, {total_sents} sentences")

    # Cleanup
    # os.remove(raw_path)

def cite_fusion():
    cite_path = os.path.join(BASE_PATH, "db_files", "wiki_wo_links-just-cite-full.db")
    conn = sqlite3.connect(cite_path)
    wiki_db = conn.cursor()
    docs = wiki_db.execute("SELECT id, text FROM documents ORDER BY id COLLATE NOCASE ASC").fetchall()
    conn.close()

    claim_path = os.path.join(BASE_PATH, "db_files", "wiki_wo_links-claim-full.db")
    conn = sqlite3.connect(claim_path)
    wiki_db = conn.cursor()
    titles, cited_texts, claim_texts = [], [], []
    for title, cite_text in tqdm(docs):
        claim_text = wiki_db.execute("SELECT text FROM documents WHERE id = ?", 
                                              (title,)).fetchone()[0]
        titles.append(title)
        claim_texts.append(claim_text)
        cited_texts.append(cite_text)
    conn.close()

    fusion_path = os.path.join(BASE_PATH, "db_files", "wiki_wo_links-cite-claim-full.db")
    conn = sqlite3.connect(fusion_path)
    wiki_db = conn.cursor()
    wiki_db.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        text TEXT
                    )
                    """)
    count = 0
    wiki_values = []
    for title, cite_text, claim_text in tqdm(zip(titles, cited_texts, claim_texts), total=len(titles)):
        # Add claim detected sentence if there are any and if it doesn't contain any citations yet.
        if not cite_text.strip() and claim_text.strip():
            wiki_values.append((title, claim_text))
            count += 1
        else:
            wiki_values.append((title, cite_text))
    wiki_db.executemany("INSERT OR IGNORE INTO documents (id, text) VALUES (?, ?)", wiki_values)
    conn.commit()

    wiki_db.execute("VACUUM;")
    conn.commit()
    conn.close()
    print(f"Total length: {len(titles)}, count original: {count}")

def main():
    construct_db_file(setting=args.setting, split_sent=args.split_sent)
    if args.pre_compute_embed:
        pre_compute_embed()
    # cite_fusion()
    get_raw_size()

if __name__ == "__main__":
    main()
