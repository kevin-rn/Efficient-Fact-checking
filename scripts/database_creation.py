from argparse import ArgumentParser
import bz2
from itertools import chain
import json
import numpy as np
import os
import sqlite3
from tqdm import tqdm
import unicodedata
from text_processing_utils import search_file_paths, remove_html_tags

### Argument parsing
### E.g. python database_creation.py --setting=cite --db_function=with_embed
parser = ArgumentParser()

parser.add_argument(
    "--setting",
    type=str,
    help="Name of the corresponding setting to retrieve documents from e.g. cite-first, cite-full, cite-first-embed"
)

parser.add_argument(
    "--db_function",
    type=str,
    required=True,
    help="[full_original | construct_db | drop_embed]"
)

parser.add_argument(
    "--first_paragraph_only",
    type=bool,
    default=False,
    help="Only use text from the first paragraph instead of whole document"
)

parser.add_argument(
    "--include_embed",
    type=bool,
    default=False,
    help="Whether to include text embedding into the database or not"
)

args = parser.parse_args()


### SQL commands for Database Creation ###
TABLE_WITH_EMBED = """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        text TEXT,
        embed BLOB
    )
"""
TABLE_WITHOUT_EMBED = """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        text TEXT
    )
"""

### DATA ###
ENWIKI_FOLDER= os.path.join("..", "baselines", "hover", "data", "enwiki_files")


def construct_db_full_original() -> None:
    """
    Generates sqlite db file (id, text) from the original HotPotQA enwiki bz2 folders.
    """
    db_path = os.path.join("..", "baselines", "hover", "data", "db_files", "original-full.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(TABLE_WITHOUT_EMBED)

    # Processed English Wikipedia dump from HotPotQA
    ORIGINAL_ENWIKI = os.path.join(ENWIKI_FOLDER, "enwiki-2017-original") 
    file_paths = search_file_paths(ORIGINAL_ENWIKI)
    for file_path in tqdm(file_paths):
        bz2_path = ORIGINAL_ENWIKI + file_path
        with bz2.open(bz2_path, "rt") as file:
            for line in file:
                wiki_article = json.loads(line)
                title = wiki_article["title"]
                wiki_text = wiki_article["text"]
                paragraphs = list(chain.from_iterable(wiki_text[1:]))
                paragraphs = remove_html_tags(paragraphs)
                doc_text = " ".join(paragraphs)
                cursor.execute(
                    """INSERT OR IGNORE INTO documents (id, text) VALUES (?, ?)""",
                    (unicodedata.normalize('NFD', title), doc_text),
                )
        conn.commit()
    conn.close()

def construct_db_file(first_paragraph_only: bool = False, include_embed: bool = False) -> None:
    """
    Generates sqlite db file from enwiki bz2 folders
    Columns are (id, text, embed) with embed else  (id, text)
    """
    db_path = os.path.join("..", "baselines", "hover", "data", 
                           "db_files", f"wiki_wo_links-{args.setting}.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if include_embed:
        cursor.execute(TABLE_WITH_EMBED)
    else:
        cursor.execute(TABLE_WITHOUT_EMBED)

    path_location = os.path.join(ENWIKI_FOLDER, "enwiki-2017-" + args.setting)
    file_paths = search_file_paths(path_location)
    for file_path in tqdm(file_paths):
        bz2_path = path_location + file_path
        try:
            with bz2.open(bz2_path, "rt") as file:
                for line in file:
                    wiki_article = json.loads(line)
                    title = wiki_article["title"]
                    fact_text = wiki_article["fact_text"]

                    if first_paragraph_only:
                        paragraphs = []
                        for para in fact_text[1:]:
                            if para:
                                paragraphs = para
                                break
                    else:
                        paragraphs = list(chain.from_iterable(fact_text[1:]))
                    doc_text = "[SENT]".join(paragraphs)

                    if include_embed:
                        text_embed = np.array(wiki_article["embed"])
                        blob_embed = text_embed.tobytes()
                        cursor.execute(
                            """INSERT OR IGNORE INTO documents (id, text, embed) VALUES (?, ?, ?)""",
                            (unicodedata.normalize('NFD', title), doc_text, blob_embed),
                        )
                    else:
                        cursor.execute(
                            """INSERT OR IGNORE INTO documents (id, text) VALUES (?, ?)""",
                            (unicodedata.normalize('NFD', title), doc_text),
                        )
        except Exception as e:
            print(f"Error: {e}, file: {bz2_path}")
            break
        conn.commit()
    conn.close()

def drop_embed_db() -> None:
    """
    Alters sqlite db file to drop embed column if exists
    (id, text, embed) --> (id, text)
    """
    try:
        db_path = os.path.join("..", "baselines", "hover", "data", 
                               "db_files", f"wiki_wo_links-{args.setting}.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE documents DROP COLUMN embed;")
        cursor.execute("VACUUM;")
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)

def main():
    match args.db_function:
        case "full_original":
            construct_db_full_original()
        case "construct_db":
            construct_db_file(first_paragraph_only=args.first_paragraph_only, 
                              include_embed=args.include_embed)
        case "drop_embed":
           drop_embed_db()
        case _:
            print("Incorrect function passed for:\n" +
            "--db_function [full_original | construct_db | drop_embed]")

if __name__ == "__main__":
    main()