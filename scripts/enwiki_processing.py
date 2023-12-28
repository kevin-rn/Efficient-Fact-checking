import bz2
import cchardet # speed up lxml (html parsing) just by importing
import json
import jsonlines
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
    "--start_location",
    type=str,
    required=True,
    help="Name of the corresponding setting to retrieve documents from e.g. cite, wice"
)

parser.add_argument(
    "--end_location",
    type=str,
    required=True,
    help="Name of the corresponding setting to retrieve documents from e.g. cite, wice"
)

parser.add_argument(
    "--process_function",
    type=str,
    required=True,
    help="[generate_embed | generate_embed_original | claimbuster | claimbuster_current | wice | cite | store_raw]"
)
parser.add_argument(
    "--first_paragraph_only",
    type=bool,
    default=False,
    help="Only use text from the first paragraph instead of whole document"
)
parser.add_argument(
    "--do_mmr",
    type=bool,
    default=False,
    help="Perform MMR-Retrieval before Binary claim classification."
)

args = parser.parse_args()
cite_pattern = re.compile("\[\d+\]|\[nb\s*\d+\]")   # Citation pattern e.g. [1] or [nb]

### DATA ###
BASE_PATH = os.path.join("..", "baselines", "hover", "data", "enwiki_files")
ORIGINAL_ENWIKI = os.path.join(BASE_PATH, "enwiki-2017-original") # from HotPotQA
START_ENWIKI = os.path.join(BASE_PATH, "enwiki-2017-" + args.start_location)
END_ENWIKI = os.path.join(BASE_PATH, "enwiki-2017-" + args.end_location)

### MODEL ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
match args.process_function: 
    case "generate_embed" | "generate_embed_original":
        encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )
        END_ENWIKI = END_ENWIKI + "-embed"
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
    case "cite":
        spacy.prefer_gpu() # alternatively use: spacy.require_gpu()
        nlp = spacy.load("en_core_web_lg", disable=['tagger', 'parser', 'ner', 'lemmatizer'])
        nlp.add_pipe('sentencizer')


def generate_embed(filepath: str, start_location: str, end_location: str) -> None:
    """
    Calculate sentence embeddings for text.

    Args:
        - filepath (str): bz2 filepath to process for.
        - start_location (str): The enwiki bz2 folder location to process.
        - end_location (str): The enwiki bz2 folder location to create.
    """
    json_obj = []
    with bz2.open(start_location + filepath, "rt") as file:
        for line in file:
            wiki_article = json.loads(line)
            wiki_fact = wiki_article['fact_text']
            if args.first_paragraph_only:
                # Only keep first paragraph
                paragraphs = []
                for para in wiki_fact[1:]:
                    if para:
                        paragraphs = remove_html_tags(para)
                        break
                wiki_article['fact_text'] = [wiki_fact[0], paragraphs]
            else:
                paragraphs = remove_html_tags(list(chain.from_iterable(wiki_fact[1:])))

            emb_text = " ".join(paragraphs)
            embed = encoder.encode(emb_text)
            wiki_article['embed'] = embed.tolist()

            json_obj.append(wiki_article)

    save_file_to_path(json_list=json_obj, dir=end_location, filepath=filepath)


def generate_embed_original(filepath: str, start_location: str, end_location: str) -> None:
    """
    Calculate sentence embeddings for text.

    Args:
        - filepath (str): bz2 filepath to process for.
        - start_location (str): The enwiki bz2 folder location to process.
        - end_location (str): The enwiki bz2 folder location to create.
    """
    json_obj = []
    with bz2.open(start_location + filepath, "rt") as file:
        for line in file:
            wiki_article = json.loads(line)
            wiki_text = wiki_article['text']

            if args.first_paragraph_only:
                paragraphs = []
                for para in wiki_text[1:]:
                    if para:
                        paragraphs = remove_html_tags(para)
                        break
            else:
                paragraphs = remove_html_tags([sent for para in wiki_text[1:] for sent in para if sent.strip()])

            emb_text = " ".join(paragraphs)
            embed = encoder.encode(emb_text)
            wiki_article['embed'] = embed.tolist()
            wiki_article['fact_text'] = [wiki_article['title'], paragraphs]
            json_obj.append(wiki_article)

    save_file_to_path(json_list=json_obj, dir=end_location, filepath=filepath)

def __use_original_text(doc_text: str) -> List[str]:
    """
    Helper method for citation extraction.
    In case url is unreachable or no text to process for, take original text sentences.

    Args:
        - doc_text (str): original doc_text to retrieve.
    
    Returns:
        List of document sentences.
    """
    if args.first_paragraph_only:
        sentences = []
        for para in doc_text:
            if para:
                sentences = remove_html_tags(para)
                break
    else:
        sentences = list(chain.from_iterable(doc_text))
    sentences = remove_html_tags(sentences)
    return sentences

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

def claimbuster_enwiki(filepath: str, start_location: str, end_location: str) -> None:
    """
    Read bz2 file and add fact_text field containing claim detected sentences

    Args:
        - filepath (str): bz2 filepath to process for.
        - start_location (str): The enwiki bz2 folder location to process.
        - end_location (str): The enwiki bz2 folder location to create.
    """
    json_obj = []
    with bz2.open(start_location + filepath, "rt") as file:
        for line in file:
            wiki_article = json.loads(line)
            wiki_text = wiki_article["text"]

            # Loop through paragraphs
            wiki_paragraphs = [remove_html_tags(para) for para in wiki_text[1:]]
            fact_text = __claim_detection(wiki_paras=wiki_paragraphs)
            fact_text.insert(0, wiki_text[0]) # Insert title at start
            wiki_article["fact_text"] = fact_text
            json_obj.append(wiki_article)
    save_file_to_path(json_list=json_obj, dir=end_location, filepath=filepath)

def claimbuster_enwiki_current(filepath: str, start_location: str, end_location: str) -> None:
    """
    Read bz2 file and add fact_text field containing claim detected sentences

    Args:
        - filepath (str): bz2 filepath to process for.
        - start_location (str): The enwiki bz2 folder location to process.
        - end_location (str): The enwiki bz2 folder location to create.
    """
    json_obj = []
    with bz2.open(start_location + filepath, "rt") as file:
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
                            wiki_sents = __use_original_text(wiki_text[1:])
                            wiki_article['hasRetrieved'] = False
                        else:
                            wiki_sents = [sent.text for sent in nlp(first_para.get_text()) if sent.text.strip()]
                            wiki_article['hasRetrieved'] = hasRetrieved
                    else:
                        wiki_sents = [sent.text for sent in nlp(soup.get_text()) if sent.text.strip()]
                        wiki_article['hasRetrieved'] = hasRetrieved
            else:
                wiki_sents = __use_original_text(wiki_text[1:])
                wiki_article['hasRetrieved'] = False
            fact_text = __claim_detection(wiki_paras=[wiki_sents])
            fact_text.insert(0, wiki_text[0])
            wiki_article['fact_text'] = fact_text
            json_obj.append(wiki_article)

    save_file_to_path(json_list=json_obj, dir=end_location, filepath=filepath)

def wice_enwiki(filepath: str, start_location: str, end_location: str) -> None:
    """
    MMR Retrieval to get top-5 sentences per wikipedia document. 
    Afterwards perform binary claim classification using a SetFit model trained on WiCE data.

    Args:
        - filepath (str): bz2 filepath to process for.
        - start_location (str): The enwiki bz2 folder location to process.
        - end_location (str): The enwiki bz2 folder location to create.
        - do_mmr (bool): Whether to perform the MMR Retrieval step or not.

    """
    json_obj = []
    top_k=5
    lambda_val = 0.7
    bz2_path = start_location + filepath

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
    save_file_to_path(json_list=json_obj, dir=end_location, filepath=filepath)



def citation_enwiki(filepath: str, start_location: str, end_location: str) -> None:
    """
    Extracting citations for each wiki article in a bz2 file.

    Args:
        - filepath (str): bz2 filepath to process for.
        - start_location (str): The enwiki bz2 folder location to process.
        - end_location (str): The enwiki bz2 folder location to create.
    """
    json_obj = []
    with bz2.open(start_location + filepath, "rt") as file:
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
                first_para = soup.find('p', class_=None)
                if not first_para:
                    wiki_article['fact_text'] = [wiki_text[0], __use_original_text(wiki_text[1:])]
                    wiki_article['hasRetrieved'] = False
                    json_obj.append(wiki_article)
                    continue
                else:
                    first_para = first_para.get_text()

                cite_tags = soup.find_all('sup', {'class': 'reference'})
                # Find all citation tags
                texts_to_process, parent_texts = [], []
                for cite_tag in cite_tags:
                    # Find the parent tag and extract text the everything up until the citation
                    parent_tag = cite_tag.find_parent().get_text()

                    # Skip citations that are not in first paragraph
                    if args.first_paragraph_only and parent_tag not in first_para:
                        continue

                    cite_text = cite_tag.get_text()
                    if cite_text:
                        citated_text = parent_tag.split(cite_text)

                        # Multiple citations can occur so concatenate (possible) previous ones
                        cited = ''
                        for c in citated_text[:-1]:
                            cited += c
                            # Remove citation numbers e.g. [1] [nb] from the text
                            cleaned_text = cite_pattern.sub('', unicodedata.normalize('NFKC', cited)).strip()
                            texts_to_process.append(cleaned_text)

                            p_text = cite_pattern.sub('', unicodedata.normalize('NFKC', parent_tag)).strip()
                            if p_text not in parent_texts:
                                parent_texts.append(p_text)

                # Get actual last sentence and remove non-duplicates
                docs = nlp.pipe(texts_to_process, batch_size=128, n_process=1)
                sentences = [list(doc.sents)[-1].text for doc in docs if list(doc.sents)]
                non_duplicates = list(dict.fromkeys(sentences))
                filtered_sentences = [sentence for sentence in non_duplicates 
                                      if not any(sentence in other_sentence for other_sentence in non_duplicates if sentence != other_sentence)]

                # To get the whole sentences 
                # e.g. "This is an example[1], of a full sentence." instead of "This is an example[1]"
                docs = nlp.pipe(parent_texts, batch_size=128, n_process=1)
                parent_sents = [sent.text.strip() for doc in docs if list(doc.sents) for sent in doc.sents]
                results = [parent_sent for cite_sent in filtered_sentences for parent_sent in parent_sents if cite_sent in parent_sent]

                wiki_article['fact_text'] = [wiki_text[0], results] # [[title], [citation sentences]]
            else:
                wiki_article['fact_text'] = [wiki_text[0], __use_original_text(wiki_text[1:])] # [[title], [citation sentences]]

            wiki_article['hasRetrieved'] = hasRetrieved
            json_obj.append(wiki_article)

    save_file_to_path(json_list=json_obj, dir=end_location, filepath=filepath)

def store_raw(filepath: str, start_location: str, end_location: str) -> None:
    """
    Convert bz2 file to raw jsonlines containing only 
    jsonobjects with the wikipedia article title and paragraphs.

    Args:
        - filepath (str): bz2 filepath to process for.
        - start_location (str): The enwiki bz2 folder location to process.
        - end_location (str): The enwiki bz2 folder location to create.
    """
    json_obj = []
    with bz2.open(start_location + filepath, "rt") as file:
        for line in file:
            wiki_article = json.loads(line)
            wiki_store = {"title": wiki_article['title'], "text": wiki_article['fact_text']}
            json_obj.append(wiki_store)

    folderpath = end_location + os.path.split(filepath)[0]
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    folderpath = end_location + filepath.replace(".bz2", ".jsonl")
    with jsonlines.open(folderpath, 'w') as writer:
        writer.write_all(json_obj)

def main():
    # create directory if doesn't exist.
    if not os.path.exists(END_ENWIKI):
        os.makedirs(END_ENWIKI)

    match args.process_function:
        case "generate_embed":
            multiprocess_bz2(func=generate_embed,
                            start_location=START_ENWIKI,
                            end_location=END_ENWIKI,
                            n_processes=16)
        case "generate_embed_original":
            multiprocess_bz2(func=generate_embed_original,
                            start_location=ORIGINAL_ENWIKI,
                            end_location=END_ENWIKI,
                            n_processes=16)
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
        case "store_raw":
            multiprocess_bz2(func=store_raw,
                             start_location=START_ENWIKI,
                             end_location=END_ENWIKI,
                             n_processes=16)
        case _:
            print("Incorrect function passed for:\n" +
            "--process_function [generate_embed | generate_embed_original | claimbuster | claimbuster_current | wice | cite]")

if __name__ == "__main__":
    main()
