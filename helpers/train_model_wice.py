from datasets import Dataset
import torch
import os
import pandas as pd
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from typing import List, Any

dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
save_dir = str(os.path.join(dir_path, "models", "setfit", "wice_classifier"))


def binary_indices(row_data: pd.Series) -> List:
    """
    Convert list of indices to list of binary values indicating whether the corresponding
    evidence sentence supports or not.

    Args:
        row_data (Series): row of WiCE dataset to apply changes to.

    Returns:
        List: list of binary values corresponding to the list of sentences in the 'evidence' column,
        where 0's are for 'not support' sentences and 1's for 'support' sentences.
    """
    indices = sorted(
        list(
            set(
                item for sublist in row_data["supporting_sentences"] for item in sublist
            )
        )
    )
    binary_array = [int(i in indices) for i in range(len(row_data["evidence"]))]
    return binary_array


# Copy train and dev files from `https://github.com/ryokamoi/wice/data/entailment_retrieval/claims`
def prepare_wice_data() -> (pd.DataFrame, pd.DataFrame):
    """
    Loads WiCE train and dev data and formats the data for model training.

    Returns:
        (DataFrame, DataFrame): train and dev dataframes with just the evidence and supporting sentences as columns.
    """
    df_train = pd.read_json("../data/wice/train.jsonl", lines=True)
    df_train["label"] = df_train.apply(binary_indices, axis=1)
    df_train.drop(["supporting_sentences", "claim", "meta"], axis=1, inplace=True)

    df_dev = pd.read_json("../data/wice/dev.jsonl", lines=True)
    df_dev["label"] = df_dev.apply(binary_indices, axis=1)
    df_dev.drop(["supporting_sentences", "claim", "meta"], axis=1, inplace=True)

    df_train = df_train.explode(["label", "evidence"], ignore_index=True)
    df_dev = df_dev.explode(["label", "evidence"], ignore_index=True)

    return df_train, df_dev


def train_setfit(df_train: pd.DataFrame, df_dev: pd.DataFrame) -> None:
    """
    Trains SetFit model on WiCE dataset and saves it to disk

    Args:
        df_train (DataFrame): training data of WiCE.
        df_dev (DataFrame): dev data of WiCE.
    """

    # Take amount of train labels corresponding to support (imbalance) as the total number of samples.
    nsamples = len(df_train[df_train["label"] == 1])
    train_dataset = Dataset.from_pandas(df=df_train)
    train_dataset = sample_dataset(
        train_dataset, label_column="label", num_samples=nsamples
    )
    val_dataset = Dataset.from_pandas(df_dev)

    # Load SetFit model.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = "sentence-transformers/paraphrase-mpnet-base-v2"
    model = SetFitModel.from_pretrained(checkpoint)
    model.to(device=device)

    # train, evaluate and save the model.
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        train_batch=32,
        column_mapping={"evidence": "text", "label": "label"},
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.model._save_pretrained(save_directory=save_dir)
    print(metrics)


train_data, dev_data = prepare_wice_data()
train_setfit(df_train=train_data, df_dev=dev_data)

# saved_model = SetFitModel._from_pretrained(save_dir)


"""
output

***** Running training *****
  Num examples = 354160
  Num epochs = 1
  Total optimization steps = 11068
  Total train batch size = 32
Iteration: 100%|██████████████████████████| 11068/11068 [9:27:24]

***** Running evaluation *****
{'accuracy': 0.5990647482014388}
"""
