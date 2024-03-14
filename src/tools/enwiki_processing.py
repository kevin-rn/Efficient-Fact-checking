import asyncio

import aiohttp
import nest_asyncio

nest_asyncio.apply()

import bz2
import json
import os
import re
import unicodedata
from argparse import ArgumentParser
from itertools import chain

import cchardet  # speed up lxml (html parsing) just by importing
import lxml
import numpy as np
import torch
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from text_processing_utils import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = ArgumentParser()
parser.add_argument(
    "--first_loc",
    type=str,
    required=True,
    help="Name of the corresponding setting to retrieve documents from e.g. enwiki-2017-cite, enwiki-2023-wice",
)

parser.add_argument(
    "--second_loc",
    type=str,
    required=True,
    help="Name of the corresponding setting to store documents for e.g. enwiki-2017-cite, enwiki-2023-wice",
)

parser.add_argument(
    "--process_function",
    type=str,
    required=True,
    help="[preprocess | claim_detect | wice | cite | fusion ]",
)
parser.add_argument(
    "--do_mmr",
    action="store_true",
    help="Perform MMR-Retrieval before Binary claim classification.",
)
parser.add_argument(
    "--use_spacy",
    action="store_true",
    help="Use SpaCy for sentence splitting instead of StanfordCoreNLP.",
)

args = parser.parse_args()

# Citation pattern e.g. [1] or [nb]
cite_pattern = re.compile("\[\d+\]|\[nb\s*\d+\]")

### DATA ###
BASE_PATH = os.path.join(os.path.abspath(os.curdir), "data", "enwiki_files")
FIRST_ENWIKI = os.path.join(BASE_PATH, args.first_loc)
SECOND_ENWIKI = os.path.join(BASE_PATH, args.second_loc)

### MODEL ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
match args.process_function:
    case "claim_detect":
        claim_tokenizer = AutoTokenizer.from_pretrained(
            "Nithiwat/bert-base_claimbuster"
        )
        claim_model = AutoModelForSequenceClassification.from_pretrained(
            "Nithiwat/bert-base_claimbuster"
        ).to(device)
    case "wice":
        from setfit import SetFitModel

        binary_claim_model = SetFitModel._from_pretrained(
            "../models/setfit/wice_classifier_0.634_full_sklearn"
        ).to(device)
        encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )
    case "preprocess":
        if args.use_spacy:
            import spacy

            spacy_gpu = spacy.prefer_gpu(
                gpu_id=0
            )  # alternatively use: spacy.require_gpu()
            print("Using Spacy-GPU" if spacy_gpu else "Using Spacy-CPU")
            nlp = spacy.load(
                "en_core_web_lg", disable=["tagger", "parser", "ner", "lemmatizer"]
            )
            nlp.add_pipe("sentencizer")
        else:
            import sys

            sys.path.append(sys.path[0] + "/..")
            from baselines.hover.StanfordNLP import StanfordNLP

            corenlp = StanfordNLP(port=9000)
            print("Using StanfordNLP")
    case "fusion":
        fusion_dir = re.sub(r"(enwiki-\d+).*", r"\1", args.first_loc) + "-fusion"
        FUSION_ENWIKI = os.path.join(BASE_PATH, fusion_dir)


def preprocess_enwiki(filepath: str, start_loc: str, end_loc: str) -> None:
    """
    Read bz2 file and update text field to list of paragraphs
    where each paragraph is a list of sentences.

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
            wiki_text = [para for para in wiki_text.split("\n\n")]
            # preprocessed = [[sent.text for sent in nlp(para) if sent.text.strip()] for para in wiki_text]
            preprocessed = []
            for para_text in wiki_text:
                para_parse = corenlp.annotate(para_text)
                para_sents = []
                for sent_parse in para_parse["sentences"]:
                    start_idx = sent_parse["tokens"][0]["characterOffsetBegin"]
                    end_idx = sent_parse["tokens"][-1]["characterOffsetEnd"]
                    sent = para_text[start_idx:end_idx]
                    para_sents.append(sent)
                preprocessed.append(para_sents)
            wiki_article["text"] = preprocessed
            json_obj.append(wiki_article)

    save_file_to_path(json_list=json_obj, dir=end_loc, filepath=filepath)


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
            fact_text = [wiki_text[0]]
            for para in wiki_text[1:]:
                sents = remove_html_tags(para)
                if sents:
                    # Performs claim detection on list of strings and returns the salient ones.
                    tokenized_inputs = claim_tokenizer(
                        sents, padding=True, truncation=True, return_tensors="pt"
                    ).to(device)
                    with torch.no_grad():
                        logits = claim_model(**tokenized_inputs).logits
                        logits = logits.cpu()
                    label_indices = (
                        torch.nonzero(logits.argmax(dim=1) == 1).squeeze().cpu()
                    )
                    # Prevent looping over 0d-tensor error.
                    if label_indices.dim() == 0:
                        label_indices = label_indices.unsqueeze(0)
                    claimworthy = [sents[idx] for idx in label_indices]
                    fact_text.append(claimworthy)

            wiki_article["fact_text"] = fact_text
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
    top_k = 5
    lambda_val = 0.7
    bz2_path = start_loc + filepath

    with bz2.open(bz2_path, "rt") as file:
        for line in file:
            wiki_article = json.loads(line)
            wiki_text = wiki_article["text"]
            title = wiki_text[0]

            wiki_paragraphs = [remove_html_tags(sent) for sent in wiki_text[1:]]
            if args.do_mmr:
                # Perform MMR retrieval of top k sentences within a document
                doc_sentences = list(chain.from_iterable(wiki_paragraphs))
                wiki_doc_emb = encoder.encode(
                    " ".join(doc_sentences)
                )  # embed entire doc text
                wiki_sents_emb = encoder.encode(
                    doc_sentences
                )  # embed individual sentences

                mmr_scores = []
                for sent_emb in wiki_sents_emb:
                    sim_with_doc = cosine_similarity([wiki_doc_emb], [sent_emb])[0][0]
                    mmr_score = lambda_val * sim_with_doc - (1 - lambda_val) * max(
                        mmr_scores, default=0
                    )
                    mmr_scores.append(mmr_score)

                # sort on mmr score and afterwards back in original order using enumeration
                sorted_sents = [
                    sent
                    for _, sent in sorted(
                        zip(mmr_scores, enumerate(doc_sentences)),
                        key=lambda x: x[0],
                        reverse=True,
                    )
                ]
                wiki_paragraphs = [
                    sent for _, sent in sorted(sorted_sents[:top_k], key=lambda x: x[0])
                ]

            claim_worthy = []
            for paragraph_sents in wiki_paragraphs:
                if paragraph_sents:
                    np_sents = np.array(paragraph_sents)
                    with torch.no_grad():
                        preds = binary_claim_model(np_sents)
                        preds = preds.to("cpu").bool().numpy()
                        claim_worthy.append(list(np_sents[preds]))
                    break  # only add first available paragraph to save time

            claim_worthy.insert(0, title)
            wiki_article["fact_text"] = (
                claim_worthy  # [[title], [paragraph1], [paragraph2]]
            )
            json_obj.append(wiki_article)
    save_file_to_path(json_list=json_obj, dir=end_loc, filepath=filepath)


async def citation_enwiki(filepath: str, start_loc: str, end_loc: str) -> None:
    """
    Extracting citations for each wiki article in a bz2 file.

    Args:
        - filepath (str): bz2 filepath to process for.
        - start_loc (str): The enwiki bz2 folder location to process.
        - end_loc (str): The enwiki bz2 folder location to create.
    """
    if args.use_spacy:
        import spacy

        spacy_gpu = spacy.prefer_gpu(gpu_id=0)  # alternatively use: spacy.require_gpu()
        # print("Using Spacy-GPU" if spacy_gpu else "Using Spacy-CPU")
        nlp = spacy.load(
            "en_core_web_lg", disable=["tagger", "parser", "ner", "lemmatizer"]
        )
        nlp.add_pipe("sentencizer")
    else:
        import sys

        sys.path.append(sys.path[0] + "/..")
        from baselines.hover.StanfordNLP import StanfordNLP

        corenlp = StanfordNLP(port=9000)
        print("Using StanfordNLP")

    json_obj = []
    with bz2.open(start_loc + filepath, "rt") as file:
        # file_iter = get_file_iter(file=file, filepath=filepath)
        connector = aiohttp.TCPConnector(force_close=True)
        async with aiohttp.ClientSession(
            connector=connector, trust_env=True
        ) as session:
            for line in file:
                wiki_article = json.loads(line)
                wiki_text = wiki_article["text"]
                wiki_url = wiki_article["url"]
                wiki_article["hasRetrieved"] = False

                raw_page, response = await get_url(wiki_url, session)
                if not response:
                    exit()
                else:
                    wiki_article["hasRetrieved"] = True

                if not wiki_article["hasRetrieved"]:
                    wiki_article["fact_text"] = [wiki_text[0], []]
                else:
                    soup = BeautifulSoup(raw_page, "lxml")
                    # used only when extracting just the first paragraph
                    cite_texts, parent_paras = [], []
                    parent_tags = soup.find_all(["p", "ul"], class_=None)
                    for parent_tag in parent_tags:
                        # Store Parent sentences per paragraph sub-list
                        parent_text = cite_pattern.sub(
                            "",
                            unicodedata.normalize("NFD", parent_tag.get_text()).strip(),
                        )
                        if not parent_text:
                            continue

                        if args.use_spacy:
                            parent_sents = [
                                sent
                                for p_sent in nlp(parent_text).sents
                                for sent in p_sent.text.strip().split("\n")
                            ]
                            parent_paras.append(parent_sents)
                        else:
                            corenlp_sents = []
                            para_parse = corenlp.annotate(parent_text)
                            for sent_parse in para_parse["sentences"]:
                                start_idx = sent_parse["tokens"][0][
                                    "characterOffsetBegin"
                                ]
                                end_idx = sent_parse["tokens"][-1]["characterOffsetEnd"]
                                corenlp_sents.extend(
                                    parent_text[start_idx:end_idx].strip().split("\n")
                                )
                            parent_paras.append(corenlp_sents)

                        # Find all citation tags in paragraph and extract text up to each citation tag.
                        cite_tags = parent_tag.find_all("sup", {"class": "reference"})
                        for cite_tag in cite_tags:
                            cite_text = cite_tag.get_text()
                            if cite_text:
                                citated_text = parent_tag.get_text().split(cite_text)[
                                    :-1
                                ]
                                cited = ""
                                # Multiple citations can occur in a paragraph (or sentence) so concatenate previous parts
                                for c in citated_text:
                                    cited += c
                                    cleaned_text = cite_pattern.sub(
                                        "",
                                        unicodedata.normalize(
                                            "NFD", cited.split("\n")[-1]
                                        ),
                                    ).strip()
                                    if cleaned_text and cleaned_text not in cite_texts:
                                        cite_texts.append(cleaned_text)

                    # Get actual last sentence and remove non-duplicates (exact match and sub-strings)
                    if args.use_spacy:
                        docs = nlp.pipe(cite_texts, batch_size=128, n_process=1)
                        sentences = [
                            list(doc.sents)[-1].text for doc in docs if list(doc.sents)
                        ]
                        non_dupes = list(dict.fromkeys(sentences))
                    else:
                        non_dupes = []
                        for para_text in cite_texts:
                            para_parse = corenlp.annotate(para_text)
                            sent_parse = para_parse["sentences"][-1]
                            start_idx = sent_parse["tokens"][0]["characterOffsetBegin"]
                            sent = para_text[start_idx:]
                            if sent not in non_dupes:
                                non_dupes.append(sent)
                    filtered_sentences = [
                        sent
                        for sent in non_dupes
                        if not any(
                            sent in other_sent
                            for other_sent in non_dupes
                            if sent != other_sent
                        )
                    ]

                    # Get the full sentence (possibly sentence cut-off due to citation in middle of sentence).
                    results = [wiki_text[0]]
                    for paragraph in parent_paras:
                        para_sents = [
                            p_sent
                            for c_sent in filtered_sentences
                            for p_sent in paragraph
                            if c_sent in p_sent
                        ]
                        if para_sents:
                            results.append(para_sents)
                    wiki_article["fact_text"] = (
                        results if len(results) >= 2 else [wiki_text[0], []]
                    )
                json_obj.append(wiki_article)

    save_file_to_path(json_list=json_obj, dir=end_loc, filepath=filepath)


def citation_collect(filepath: str, start_loc: str, end_loc: str):
    return asyncio.run(citation_enwiki(filepath, start_loc, end_loc))


def fusion_enwiki(filepath: str, first_loc: str, second_loc: str):
    """
    Adds claim detected sentences into the cited sentences corpus
    """
    json_obj = []
    with (
        bz2.open(first_loc + filepath, "rt") as file_1,
        bz2.open(second_loc + filepath, "rt") as file_2,
    ):
        for line_1, line_2 in zip(file_1, file_2):
            wiki_article_1 = json.loads(line_1)
            paragraphs_1 = remove_html_tags(
                list(chain.from_iterable(wiki_article_1["fact_text"][1:]))
            )
            wiki_text_1 = " ".join(
                [sent.strip() for sent in paragraphs_1 if sent.strip()]
            )

            wiki_article_2 = json.loads(line_2)
            paragraphs_1 = remove_html_tags(
                list(chain.from_iterable(wiki_article_2["fact_text"][1:]))
            )
            wiki_text_2 = " ".join(
                [sent.strip() for sent in paragraphs_1 if sent.strip()]
            )

            # Check if first location has fact text otherwise use second location to fill in.
            if not wiki_text_1.strip() and wiki_text_2.strip():
                wiki_article_1["fact_text"] = wiki_article_2["fact_text"]
            json_obj.append(wiki_article_1)

    save_file_to_path(json_list=json_obj, dir=FUSION_ENWIKI, filepath=filepath)


def main():
    # create directory if doesn't exist.
    if not os.path.exists(SECOND_ENWIKI):
        os.makedirs(SECOND_ENWIKI)

    match args.process_function:
        case "preprocess":
            multiprocess_bz2(
                func=preprocess_enwiki,
                first_loc=FIRST_ENWIKI,
                second_loc=SECOND_ENWIKI,
                n_processes=16,
                process_style="threads" if args.use_spacy else None,
            )
        case "claim_detect":
            multiprocess_bz2(
                func=claimbuster_enwiki,
                first_loc=FIRST_ENWIKI,
                second_loc=SECOND_ENWIKI,
                n_processes=8,
            )
        case "wice":
            multiprocess_bz2(
                func=wice_enwiki,
                first_loc=FIRST_ENWIKI,
                second_loc=SECOND_ENWIKI,
                n_processes=16,
            )
        case "cite":
            multiprocess_bz2(
                func=citation_collect,
                first_loc=FIRST_ENWIKI,
                second_loc=SECOND_ENWIKI,
                n_processes=8,
                process_style=None,
            )
        case "fusion":
            if not os.path.exists(FUSION_ENWIKI):
                os.makedirs(FUSION_ENWIKI)

            multiprocess_bz2(
                func=fusion_enwiki,
                first_loc=FIRST_ENWIKI,
                second_loc=SECOND_ENWIKI,
                third_loc=FUSION_ENWIKI,
                n_processes=16,
            )

        case _:
            print(
                "Incorrect function passed for:\n"
                + "--process_function [preprocess | claim_detect | claimbuster_current | wice | cite]"
            )


if __name__ == "__main__":
    main()
