import argparse
import json
import os
import re
from typing import Any, List

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def read_data(predict_file: str, label_file: str) -> Any:
    """
    Args:
        - predict_file (str): File containing the predicted labels for each claim.
        - label_file (str): File containing the true labels for each claim.

    Returns:
        tuple containing a list of true labels and a list of predicted labels
        denoted with 0s (NOT SUPPORTED) and 1s (SUPPORTED).
    """

    def convert_label_to_int(value) -> int:
        return 1 if value == "SUPPORTED" else 0

    with open(predict_file, "rt") as file:
        data_dict = json.load(file)
        df_pred = pd.DataFrame(list(data_dict.values()))
        df_pred["uid"] = data_dict.keys()

    df_labels = pd.read_json(label_file)
    df_labels = df_labels[["uid", "claim", "label"]]
    df_labels["claim"] = df_labels["claim"].apply(lambda x: re.sub("\s+", " ", x))
    merge_df = pd.merge(df_pred, df_labels, on=["uid", "claim"])

    # Convert labels to list concisting of 0's (Not supported) or 1's (Supported)
    true_labels = merge_df["label"].apply(convert_label_to_int).to_list()
    pred_labels = merge_df["predicted_label"].apply(convert_label_to_int).to_list()

    return true_labels, pred_labels


def metrics(true_labels: List[int], pred_labels: List[int]) -> None:
    """
    Prints out metrics of accuracy, f1-score, precision-score and recall-score.

    Args:
        - true_labels (List[int]): list containing true labels for the claims.
        - pred_labels (List[int]): list containing predicted labels for the claims.
    """
    acc = accuracy_score(true_labels, pred_labels)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    results = {
        "acc": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "prec_weighted": prec_weighted,
        "prec_macro": prec_macro,
        "rec_weighted": rec_weighted,
        "rec_macro": rec_macro,
    }
    results = {k: v * 100 for k, v in results.items()}
    return results


def main():
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="hover", type=str)

    parser.add_argument("--out_dir", default="exp1.0", type=str)

    parser.add_argument(
        "--verbose", action="store_true", help="Print metrics for all checkpoints"
    )
    args = parser.parse_args()

    claim_prediction_dir = os.path.join(
        "out", args.dataset_name, args.out_dir, "claim_verification"
    )
    checkpoints = [x[0] for x in os.walk(claim_prediction_dir) if "checkpoint" in x[0]]
    labels = os.path.join(
        "data", args.dataset_name, f"{args.dataset_name}_dev_release_v1.1.json"
    )

    results = []
    for check in checkpoints:
        predict = os.path.join(check, "dev_predictions_.json")
        checkpoint_name = check.split(os.path.sep)[-1]
        true_labels, pred_labels = read_data(predict_file=predict, label_file=labels)
        check_results = metrics(true_labels=true_labels, pred_labels=pred_labels)
        results.append(check_results)
        if args.verbose:
            print(
                checkpoint_name,
                ",\t".join([f"{k}: {v:.2f}%" for k, v in check_results.items()]),
            )

    if results:
        key_metric = "acc" if args.dataset_name == "hover" else "f1_weighted"
        high_score = max(results, key=lambda x: x[key_metric])
        checkpoint = checkpoints[results.index(high_score)]
        print(
            f"\n{checkpoint.split(os.path.sep)[-1]}",
            ",    ".join([f"{k}: {v:.2f}%" for k, v in high_score.items()]),
        )
    else:
        print("No checkpoint found")


if __name__ == "__main__":
    main()
