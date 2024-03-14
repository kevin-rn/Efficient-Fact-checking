import json
import os
from itertools import chain

import jsonlines
import wget

# Claims
URLS = [
    "https://raw.githubusercontent.com/ryokamoi/wice/main/data/entailment_retrieval/claim/train.jsonl",
    "https://raw.githubusercontent.com/ryokamoi/wice/main/data/entailment_retrieval/claim/dev.jsonl",
    "https://raw.githubusercontent.com/ryokamoi/wice/main/data/entailment_retrieval/claim/test.jsonl",
]

# Subclaims
# URLS = ["https://raw.githubusercontent.com/ryokamoi/wice/main/data/entailment_retrieval/subclaim/train.jsonl",
#         "https://raw.githubusercontent.com/ryokamoi/wice/main/data/entailment_retrieval/subclaim/dev.jsonl",
#         "https://raw.githubusercontent.com/ryokamoi/wice/main/data/entailment_retrieval/subclaim/test.jsonl"]


WICE_FOLDER = os.path.join("data", "wice")


def download_wice() -> None:
    """
    Downloads the WICE dataset into the data folder.
    """
    if not os.path.exists(WICE_FOLDER):
        os.makedirs(WICE_FOLDER)

    for url in URLS:
        wget.download(url, out=WICE_FOLDER)


def convert_wice() -> None:
    """
    Converts the downloaded WICE dataset into HoVer format.
    """
    for data_split in ["train", "dev", "test"]:
        query_path = os.path.join(WICE_FOLDER, f"{data_split}.jsonl")
        with jsonlines.open(query_path, "r") as jsonl_file:
            claim_json = [obj for obj in jsonl_file]

        if data_split == "test":
            hover_format_claims = [
                {"uid": claim_obj["meta"]["id"], "claim": claim_obj["claim"]}
                for claim_obj in claim_json
            ]
        else:
            hover_format_claims = []
            for claim_obj in claim_json:
                uid = claim_obj["meta"]["id"]
                claim = claim_obj["claim"]
                article_title = claim_obj["meta"]["claim_title"]
                support_sents = list(
                    dict.fromkeys(
                        chain.from_iterable(claim_obj["supporting_sentences"])
                    )
                )
                supporting_facts = [
                    [article_title, sent_no] for sent_no in support_sents
                ]
                label = (
                    "SUPPORTED"
                    if claim_obj["label"] == "supported"
                    else "NOT_SUPPORTED"
                )
                hover_format_claims.append(
                    {
                        "uid": uid,
                        "claim": claim,
                        "supporting_facts": supporting_facts,
                        "label": label,
                    }
                )

        json_path = os.path.join(WICE_FOLDER, f"wice_{data_split}_release_v1.1.json")
        with open(json_path, "w") as outfile:
            json.dump(hover_format_claims, outfile)

        # Cleanup
        os.remove(query_path)


def main():
    """
    Downloads claim files from https://github.com/ryokamoi/wice
    and converts it to HoVer claim data format.
    """
    download_wice()
    convert_wice()


if __name__ == "__main__":
    main()
