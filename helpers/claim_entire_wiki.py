import bz2
import cchardet  # speed up lxml (html parsing) just by importing
import contextlib
import json
import lxml
import os
import re
import requests
import spacy
import sqlite3
import unicodedata

from bs4 import BeautifulSoup
import joblib
from joblib import Parallel, delayed
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# from setfit import SetFitModel
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

spacy.require_gpu()
nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "ner", "lemmatizer"])
nlp.add_pipe("sentencizer")
# uncomment when n_process > 1 in nlp.pipe(..., n_process)
# mp.set_start_method('spawn', force=True)

os.environ["PYTHONWARNINGS"] = "ignore"
cite_pattern = re.compile("\[\d+\]|\[nb\s*\d+\]")

### data paths ###
BASE_PATH = os.path.join("..", "baselines", "hover", "data")
ORIGINAL_ENWIKI = os.path.join(
    BASE_PATH, "enwiki-20171001-pages-meta-current-withlinks-processed"
)  # original wiki dump used by HOVER from HotPotQA
# CLAIM_ENWIKI = os.path.join(BASE_PATH, "enwiki-2017-pretrained_claim-detection")
# WICE_ENWIKI = os.path.join(BASE_PATH, "enwiki-2017-wice-binary-selected")
CITED_ENWIKI = os.path.join(BASE_PATH, "enwiki-2017-citated")

### models ###
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# claim_tokenizer = AutoTokenizer.from_pretrained("Nithiwat/bert-base_claimbuster")
# claim_model = AutoModelForSequenceClassification.from_pretrained(
#     "Nithiwat/bert-base_claimbuster"
# ).to(device)
# binary_claim_model = SetFitModel._from_pretrained(
# "../models/setfit/wice_classifier_0.634_full_sklearn"
# ).to(device)
# embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2",
#                                       device=device)
# lambda_val = 0.7


### Helper methods ###
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
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


def search_file_paths(dir: str):
    """
    Retrieves all bz2 file paths within a specified directory.

    Arguments:
        dir (str): directory to search through.
    """
    file_paths = []
    for subdir, _, files in os.walk(dir):
        for file in files:
            bz2_filepath = os.path.join(subdir, file)
            if bz2_filepath.endswith(".bz2"):
                file_paths.append(bz2_filepath[len(dir) :])
    return file_paths


def save_file_to_path(json_list, dir, filepath):
    """
    Writes json objects to a bz2 file for a given filepath.
    """
    folderpath = dir + os.path.split(filepath)[0]
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    with bz2.BZ2File(dir + filepath, "wb") as bz2_f:
        for j_obj in json_list:
            json_data = json.dumps(j_obj)
            bz2_f.write(json_data.encode("utf-8"))
            bz2_f.write(b"\n")


def multiprocess_bz2(
    func, start_location, end_location, n_processes=16, process_style="loky"
):
    """
    Performs multiprocessing for a given function and its filepaths.
    """

    # Get all filepaths to still process for.
    file_paths = search_file_paths(start_location)
    exclude_paths = search_file_paths(end_location)
    search_paths = list(set(file_paths).symmetric_difference(set(exclude_paths)))
    print(f"total files: {len(file_paths)}, pending: {len(search_paths)}")

    # Start Multiprocessing using joblib.
    with tqdm_joblib(
        tqdm(desc="Process bz2 file", total=len(search_paths))
    ) as progress_bar:
        Parallel(n_jobs=n_processes, prefer=process_style)(
            delayed(func)(bz2_filepath, start_location, end_location)
            for bz2_filepath in search_paths
        )


def construct_db_file(path_location: str) -> None:
    """
    Generates sqlite db file from enwiki bz2 folders

    Arguments:
        path_location (str): path location to convert to db file.
    """
    conn = sqlite3.connect("../data/wiki_wo_links.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            text TEXT
        )
        """
    )

    file_paths = search_file_paths(path_location)
    for file_path in tqdm(file_paths):
        bz2_path = path_location + file_path
        with bz2.open(bz2_path, "rt") as file:
            for line in file:
                wiki_article = json.loads(line)
                title = wiki_article["title"]
                fact_text = wiki_article["fact_text"]

                paragraphs = []
                for para in fact_text[1:]:
                    if para:
                        paragraphs = para
                        break

                doc_text = "[SENT]".join(paragraphs)
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO documents (id, text) VALUES (?, ?)
                    """,
                    (title, doc_text),
                )

        conn.commit()
    conn.close()


### Main functions ###
def extract_claimworthy(sentences):
    """
    Performs claim detection on list of strings and returns the salient ones.
    """

    def remove_html_tags(sentences):
        """
        Removes html tags from string.
        """
        soup = BeautifulSoup(sentences, features="html.parser")
        return soup.get_text()

    raw_sentences = [remove_html_tags(sent) for sent in sentences]
    tokenized_inputs = claim_tokenizer(
        raw_sentences, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = claim_model(**tokenized_inputs).logits
        logits = logits.cpu()
    label_indices = torch.nonzero(logits.argmax(dim=1) == 1).squeeze().cpu()
    # Prevent looping over 0d-tensor error.
    if label_indices.dim() == 0:
        label_indices = label_indices.unsqueeze(0)
    claimworthy = [raw_sentences[idx] for idx in label_indices]

    return claimworthy


def process_file(filepath, start_location, end_location):
    #  Read bz2 file and add fact_text field containing claim detected sentences
    json_obj = []
    with bz2.open(start_location + filepath, "rt") as file:
        for line in file:
            wiki_article = json.loads(line)
            article_text = wiki_article["text"]

            fact_text = [article_text[0]]  # skip title

            # Loop through paragraphs
            for i in range(1, len(article_text)):
                paragraph_sents = article_text[i]
                if paragraph_sents:
                    fact_sents = extract_claimworthy(paragraph_sents)
                    fact_text.append(
                        fact_sents
                    )  # may append empty paragraph array (no claims)
            wiki_article["fact_text"] = fact_text
            json_obj.append(wiki_article)

    save_file_to_path(json_list=json_obj, dir=end_location, filepath=filepath)


def mmr_claim_wiki(filepath, start_location, end_location):
    json_obj = []
    bz2_path = start_location + filepath
    with bz2.open(bz2_path, "rt") as file:
        for line in file:
            wiki_article = json.loads(line)
            wiki_text = wiki_article["text"]
            title = wiki_text[0]
            wiki_paragraphs = [
                item for sublist in wiki_article["text"][1:] for item in sublist
            ]
            wiki_doc_emb = embedding_model.encode(" ".join(wiki_paragraphs))
            wiki_sents_emb = embedding_model.encode(wiki_paragraphs)

            # MMR retrieval
            mmr_scores = []
            for sent_emb in wiki_sents_emb:
                sim_with_doc = cosine_similarity([wiki_doc_emb], [sent_emb])[0][0]
                mmr_score = lambda_val * sim_with_doc - (1 - lambda_val) * max(
                    mmr_scores, default=0
                )
                mmr_scores.append(mmr_score)
            sorted_sents = [
                sent
                for _, sent in sorted(
                    zip(mmr_scores, enumerate(wiki_paragraphs)),
                    key=lambda x: x[0],
                    reverse=True,
                )
            ]
            top_k = [
                sent for _, sent in sorted(sorted_sents[:5], key=lambda x: x[0])
            ]  # sort back in original order

            # Binary wice classification
            claim_worthy = []
            if top_k:
                preds = binary_claim_model(top_k)
                preds = preds.to("cpu").numpy()
                claim_worthy = [
                    sentence
                    for sentence, class_label in zip(top_k, preds)
                    if class_label == 1
                ]

            wiki_article["fact_text"] = [title, claim_worthy]
            wiki_article["text"] = [title, top_k]
            json_obj.append(wiki_article)

    save_file_to_path(json_list=json_obj, dir=end_location, filepath=filepath)


def extract_citations(url):
    raw_page = requests.get(url=url)
    soup = BeautifulSoup(raw_page.text, "lxml")

    # Find all citation tags
    cite_tags = soup.find_all("sup", {"class": "reference"})

    sentences, texts_to_process = [], []
    for cite_tag in cite_tags:
        # Find the parent tag and extract text
        parent_tag = cite_tag.find_parent().get_text()
        cite_text = cite_tag.get_text()
        if cite_text:
            citated_text = parent_tag.split(cite_text)

            # Multiple citations can be within a parent tag,
            # so only skip the last split
            cited = ""
            for c in citated_text[:-1]:
                cited += c

                # Remove citation numbers from the text
                cleaned_text = cite_pattern.sub(
                    "", unicodedata.normalize("NFKD", cited)
                ).strip()
                texts_to_process.append(cleaned_text)

    # Process the cleaned text to get the actual last sentence before citation reference
    docs = nlp.pipe(texts_to_process, batch_size=256, n_process=1)
    for doc in docs:
        last_sentence = list(doc.sents)
        if last_sentence:
            sentences.append(last_sentence[-1].text)

    # Return non-duplicate sentences
    non_duplicates = list(dict.fromkeys(sentences))
    filtered_sentences = [
        sentence
        for sentence in non_duplicates
        if not any(
            sentence in other_sentence
            for other_sentence in non_duplicates
            if sentence != other_sentence
        )
    ]
    return filtered_sentences


def cite_wiki(filepath: str, start_location: str, end_location: str):
    json_obj = []

    # Get total amount of wiki articles stored in bz2 file
    with bz2.open(start_location + filepath, "rt") as file:
        file_size = sum(1 for _ in file)

    # Actual loop extracting citations for each wiki article in a bz2 file.
    with bz2.open(start_location + filepath, "rt") as file:
        for line in tqdm(
            file, desc=f"file loop - {filepath}", leave=False, total=file_size
        ):
            wiki_article = json.loads(line)
            wiki_url = wiki_article["url"]
            sentences = extract_citations(url=wiki_url)
            wiki_article["fact_text"] = [wiki_article["title"], sentences]
            json_obj.append(wiki_article)

    # save in end location folder
    save_file_to_path(json_list=json_obj, dir=end_location, filepath=filepath)


if __name__ == "__main__":
    # Pre-trained claim detection
    # multiprocess_bz2(func=process_file, start_location=ORIGINAL_ENWIKI, end_location=CLAIM_ENWIKI)

    # MMR retrieval + Binary claim detection classifier (WiCE)
    # multiprocess_bz2(func=mmr_claim_wiki, start_location=ORIGINAL_ENWIKI, end_location=WICE_ENWIKI)

    # Extract citations from html pages of enwiki data
    multiprocess_bz2(
        func=cite_wiki,
        start_location=ORIGINAL_ENWIKI,
        end_location=CITED_ENWIKI,
        n_processes=os.cpu_count() - 1,
        process_style="threads",
    )

    # Constructs DB file used in HOVER
    # construct_db_file(CITED_ENWIKI)
