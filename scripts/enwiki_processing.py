import bz2
import cchardet # speed up lxml (html parsing) just by importing
import json
import lxml
import numpy as np
import os
import re
import requests
import spacy
import time
import unicodedata

from argparse import ArgumentParser
from bs4 import BeautifulSoup
from itertools import chain
from text_processing_utils import *
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


parser = ArgumentParser()
parser.add_argument(
    "--start_loc",
    type=str,
    required=True,
    help="Name of the corresponding setting to retrieve documents from e.g. cite, wice"
)

parser.add_argument(
    "--end_loc",
    type=str,
    required=True,
    help="Name of the corresponding setting to store documents for e.g. cite, wice"
)

parser.add_argument(
    "--process_function",
    type=str,
    required=True,
    help="[claimbuster | claimbuster_current | wice | cite ]"
)
parser.add_argument(
    "--first_paragraph_only",
    action='store_true',
    help="Only use text from the first paragraph instead of whole document"
)
parser.add_argument(
    "--do_mmr",
    action='store_true',
    help="Perform MMR-Retrieval before Binary claim classification."
)

args = parser.parse_args()
cite_pattern = re.compile("\[\d+\]|\[nb\s*\d+\]")   # Citation pattern e.g. [1] or [nb]

### DATA ###
BASE_PATH = os.path.join("..", "baselines", "hover", "data", "enwiki_files")
START_ENWIKI = os.path.join(BASE_PATH, "enwiki-2017-" + args.start_loc)
END_ENWIKI = os.path.join(BASE_PATH, "enwiki-2017-" + args.end_loc)

### MODEL ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
match args.process_function: 
    case "claimbuster":
        claim_tokenizer = AutoTokenizer.from_pretrained("Nithiwat/bert-base_claimbuster")
        claim_model = AutoModelForSequenceClassification.from_pretrained(
            "Nithiwat/bert-base_claimbuster"
        ).to(device)
    case "claimbuster_current":
        claim_tokenizer = AutoTokenizer.from_pretrained("Nithiwat/bert-base_claimbuster")
        claim_model = AutoModelForSequenceClassification.from_pretrained(
            "Nithiwat/bert-base_claimbuster"
        ).to(device)
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_lg", disable=['tagger', 'parser', 'ner', 'lemmatizer'])
    case "wice":
        binary_claim_model = SetFitModel._from_pretrained(
        "../models/setfit/wice_classifier_0.634_full_sklearn"
        ).to(device)
        encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )
    case "cite":
        spacy.prefer_gpu() # alternatively use: spacy.require_gpu()
        nlp = spacy.load("en_core_web_lg", disable=['tagger', 'parser', 'ner', 'lemmatizer'])
        nlp.add_pipe('sentencizer')

def __claim_detection(wiki_paras: List[str]) -> List[List[str]]:
    results = []
    for sents in wiki_paras:
        if sents:
            # Performs claim detection on list of strings and returns the salient ones.
            tokenized_inputs = claim_tokenizer(sents, 
                                            padding=True, 
                                            truncation=True, 
                                            return_tensors="pt"
                                            ).to(device)
            with torch.no_grad():
                logits = claim_model(**tokenized_inputs).logits
                logits = logits.cpu()
            label_indices = torch.nonzero(logits.argmax(dim=1) == 1).squeeze().cpu()
            # Prevent looping over 0d-tensor error.
            if label_indices.dim() == 0:
                label_indices = label_indices.unsqueeze(0)
            claimworthy = [sents[idx] for idx in label_indices]
            results.append(claimworthy)
    return results

def claimbuster_enwiki(filepath: str, start_loc: str, end_loc: str) -> None:
    """
    Read bz2 file and add fact_text field containing claim detected sentences

    Args:
        - filepath (str): bz2 filepath to process for.
        - start_loc (str): The enwiki bz2 folder location to process.
        - end_loc (str): The enwiki bz2 folder location to create.
    """
    json_obj = []
    with bz2.open(start_loc + filepath, "rt") as file:
        for line in file:
            wiki_article = json.loads(line)
            wiki_text = wiki_article["text"]

            # Loop through paragraphs
            wiki_paragraphs = [remove_html_tags(para) for para in wiki_text[1:]]
            fact_text = __claim_detection(wiki_paras=wiki_paragraphs)
            fact_text.insert(0, wiki_text[0]) # Insert title at start
            wiki_article["fact_text"] = fact_text
            json_obj.append(wiki_article)
    save_file_to_path(json_list=json_obj, dir=end_loc, filepath=filepath)

def claimbuster_enwiki_current(filepath: str, start_loc: str, end_loc: str) -> None:
    """
    Read bz2 file and add fact_text field containing claim detected sentences

    Args:
        - filepath (str): bz2 filepath to process for.
        - start_loc (str): The enwiki bz2 folder location to process.
        - end_loc (str): The enwiki bz2 folder location to create.
    """
    json_obj = []
    with bz2.open(start_loc + filepath, "rt") as file:
        for line in file:
            wiki_article = json.loads(line)
            wiki_text = wiki_article['text']
            wiki_url = wiki_article['url']
            hasRetrieved = False
            
            # retry to retrieve html page for 3 times
            for _ in range(3):
                try:
                    raw_page = requests.get(url=wiki_url)
                    hasRetrieved = True
                    break # stop retry loop once succesfull
                except requests.exceptions.ChunkedEncodingError:
                    time.sleep(1)
                except requests.exceptions.ConnectionError:
                    print("Connection refused for ", wiki_url)
                    exit()

            if hasRetrieved:
                soup = BeautifulSoup(raw_page.text, 'lxml')
                # used only when extracting just the first paragraph
                with torch.no_grad():
                    if args.first_paragraph_only: 
                        first_para = soup.find('p', class_=None)
                        # Check if first paragraph has content
                        if not first_para:
                            wiki_sents = []
                            wiki_article['hasRetrieved'] = False
                        else:
                            wiki_sents = [sent.text for sent in nlp(first_para.get_text()) if sent.text.strip()]
                            wiki_article['hasRetrieved'] = hasRetrieved
                    else:
                        wiki_sents = [sent.text for sent in nlp(soup.get_text()) if sent.text.strip()]
                        wiki_article['hasRetrieved'] = hasRetrieved
            else:
                wiki_sents = []
                wiki_article['hasRetrieved'] = False
            fact_text = __claim_detection(wiki_paras=[wiki_sents])
            fact_text.insert(0, wiki_text[0])
            wiki_article['fact_text'] = fact_text
            json_obj.append(wiki_article)

    save_file_to_path(json_list=json_obj, dir=end_loc, filepath=filepath)

def wice_enwiki(filepath: str, start_loc: str, end_loc: str) -> None:
    """
    MMR Retrieval to get top-5 sentences per wikipedia document. 
    Afterwards perform binary claim classification using a SetFit model trained on WiCE data.

    Args:
        - filepath (str): bz2 filepath to process for.
        - start_loc (str): The enwiki bz2 folder location to process.
        - end_loc (str): The enwiki bz2 folder location to create.
        - do_mmr (bool): Whether to perform the MMR Retrieval step or not.

    """
    json_obj = []
    top_k=5
    lambda_val = 0.7
    bz2_path = start_loc + filepath

    with bz2.open(bz2_path, "rt") as file:
        for line in file:
            wiki_article = json.loads(line)
            wiki_text = wiki_article['text']
            title = wiki_text[0]

            wiki_paragraphs = [remove_html_tags(sent) for sent in wiki_text[1:]]
            if args.do_mmr:
                # Perform MMR retrieval of top k sentences within a document
                doc_sentences = list(chain.from_iterable(wiki_paragraphs))
                wiki_doc_emb = encoder.encode(' '.join(doc_sentences))  # embed entire doc text
                wiki_sents_emb = encoder.encode(doc_sentences)          # embed individual sentences

                mmr_scores = []
                for sent_emb in wiki_sents_emb:
                    sim_with_doc = cosine_similarity([wiki_doc_emb], [sent_emb])[0][0]
                    mmr_score = lambda_val * sim_with_doc - (1 - lambda_val) * max(mmr_scores, default=0)
                    mmr_scores.append(mmr_score)

                # sort on mmr score and afterwards back in original order using enumeration
                sorted_sents = [sent for _, sent in sorted(zip(mmr_scores, enumerate(doc_sentences)), key=lambda x: x[0], reverse=True)]
                wiki_paragraphs = [sent for _, sent in sorted(sorted_sents[:top_k], key=lambda x: x[0])] 

            claim_worthy = []
            for paragraph_sents in wiki_paragraphs:
                if paragraph_sents:
                    np_sents = np.array(paragraph_sents)
                    with torch.no_grad():
                        preds = binary_claim_model(np_sents) 
                        preds = preds.to('cpu').bool().numpy()
                        claim_worthy.append(list(np_sents[preds]))
                    break # only add first available paragraph to save time

            claim_worthy.insert(0, title)
            wiki_article['fact_text'] = claim_worthy # [[title], [paragraph1], [paragraph2]]
            json_obj.append(wiki_article)
    save_file_to_path(json_list=json_obj, dir=end_loc, filepath=filepath)



def citation_enwiki(filepath: str, start_loc: str, end_loc: str) -> None:
    """
    Extracting citations for each wiki article in a bz2 file.

    Args:
        - filepath (str): bz2 filepath to process for.
        - start_loc (str): The enwiki bz2 folder location to process.
        - end_loc (str): The enwiki bz2 folder location to create.
    """
    json_obj = []
    with bz2.open(start_loc + filepath, "rt") as file:
        for line in file:
            wiki_article = json.loads(line)
            wiki_text = wiki_article['text']
            wiki_url = wiki_article['url']
            hasRetrieved = False
            
            # retry to retrieve html page for 3 times
            for _ in range(3):
                try:
                    raw_page = requests.get(url=wiki_url)
                    hasRetrieved = True
                    break # stop retry loop once succesfull
                except requests.exceptions.ChunkedEncodingError:
                    time.sleep(1)
                except requests.exceptions.ConnectionError:
                    print("Connection refused for ", wiki_url)
                    exit()

            if hasRetrieved:
                soup = BeautifulSoup(raw_page.text, 'lxml')
                # used only when extracting just the first paragraph
                cite_texts, parent_paras = [], []
                parent_tags = soup.find_all('p', class_=None)

                for parent_tag in parent_tags:
                    parent_text = parent_tag.get_text()
                    paragraph_text = nlp(cite_pattern.sub('', unicodedata.normalize('NFD', parent_text)).strip())
                    parent_paras.append([sent.text.strip() for sent in paragraph_text.sents])

                    # Find all citation tags in paragraph
                    cite_tags = parent_tag.find_all('sup', {'class': 'reference'})
                    for cite_tag in cite_tags:
                        # extract text up until the citation
                        cite_text = cite_tag.get_text()
                        if cite_text:
                            citated_text = parent_text.split(cite_text)

                            # Multiple citations can occur in a paragraph so concatenate previous parts
                            cited = ''
                            for c in citated_text[:-1]:
                                cited += c
                                # Remove citation numbers e.g. [1] [nb] from the text
                                cleaned_text = cite_pattern.sub('', unicodedata.normalize('NFD', cited)).strip()
                                cite_texts.append(cleaned_text)

                # Get actual last sentence and remove non-duplicates (exact match and sub-strings)
                docs = nlp.pipe(cite_texts, batch_size=128, n_process=1)
                sentences = [list(doc.sents)[-1].text for doc in docs if list(doc.sents)]
                non_dupes = list(dict.fromkeys(sentences))
                filtered_sentences = [sent for sent in non_dupes if not any(sent in other_sent for other_sent in non_dupes if sent != other_sent)]

                # [[title], [citation sentences], [citation sentences], ...]
                results = [wiki_text[0]]
                for paragraph in parent_paras:
                    para_sents = [p_sent for c_sent in filtered_sentences for p_sent in paragraph if c_sent in p_sent]
                    results.append(para_sents)
                wiki_article['fact_text'] = results 
            else:
                # [[title], []]
                wiki_article['fact_text'] = [wiki_text[0], []] 
            wiki_article['hasRetrieved'] = hasRetrieved
            json_obj.append(wiki_article)

    save_file_to_path(json_list=json_obj, dir=end_loc, filepath=filepath)

def main():
    # create directory if doesn't exist.
    if not os.path.exists(END_ENWIKI):
        os.makedirs(END_ENWIKI)

    match args.process_function:
        case "claimbuster":
            multiprocess_bz2(func=claimbuster_enwiki,
                            start_location=START_ENWIKI,
                            end_location=END_ENWIKI,
                            n_processes=16)
        case "claimbuster_current":
            multiprocess_bz2(func=claimbuster_enwiki_current,
                            start_location=START_ENWIKI,
                            end_location=END_ENWIKI,
                            n_processes=16,
                            process_style="threads")
        case "wice":
            multiprocess_bz2(func=wice_enwiki,
                            start_location=START_ENWIKI,
                            end_location=END_ENWIKI,
                            n_processes=16)
        case "cite":
            multiprocess_bz2(func=citation_enwiki,
                            start_location=START_ENWIKI,
                            end_location=END_ENWIKI,
                            n_processes=16,
                            process_style="threads")
        case _:
            print("Incorrect function passed for:\n" +
            "--process_function [claimbuster | claimbuster_current | wice | cite]")

if __name__ == "__main__":
    main()

