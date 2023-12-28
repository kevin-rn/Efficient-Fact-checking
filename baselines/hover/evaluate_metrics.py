from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import os
from typing import List, Dict
import pandas as pd

def read_data(predict_file: str, label_file: str) -> tuple[List[int], List[int]]:
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

    with open(predict_file, 'rt') as file:
        data_dict = json.load(file)
        df_pred = pd.DataFrame(list(data_dict.values()))
        df_pred['uid'] = data_dict.keys()

    df_labels = pd.read_json(label_file)
    df_labels = df_labels[['uid', 'claim', 'label', 'context']]
    merge_df = pd.merge(df_pred, df_labels, on=['uid', 'claim'])

    # Convert labels to list concisting of 0's (Not supported) or 1's (Supported)
    true_labels = merge_df['label'].apply(convert_label_to_int).to_list()
    pred_labels = merge_df['predicted_label'].apply(convert_label_to_int).to_list()

    return true_labels, pred_labels

def metrics(true_labels: List[int], pred_labels: List[int]) -> None:
    """
    Prints out metrics of accuracy, f1-score, precision-score and recall-score.

    Args:
        - true_labels (List[int]): list containing true labels for the claims.
        - pred_labels (List[int]): list containing predicted labels for the claims.
    """
    acc = accuracy_score(true_labels, pred_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="binary")
    results = {"acc": acc, "f1": f1, "prec": prec, "rec": rec}
    results = {k: f"{v*100:.02f}" for k, v in results.items()}
    print(',   '.join([f'{k}: {v}%' for k,v in results.items()]))


def main():
    # checkpoints = [os.path.join(args.output_dir, 'checkpoint-'+args.ckpt_to_evaluate)]
    predict = os.path.join("out", "hover", "exp1.0", "claim_verification", "checkpoint-2000", "dev_predictions_.json")
    labels = os.path.join("data", "hover_files", "claim_detect", "hover-cite-full", "doc_retrieval", "hover_dev_doc_retrieval.json")
    true_labels, pred_labels = read_data(predict_file=predict, label_file=labels)
    metrics(true_labels=true_labels, pred_labels=pred_labels)

if __name__ == "__main__":
    main()
