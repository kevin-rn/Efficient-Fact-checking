import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BatchEncoding
from collections import Counter
    
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from typing import Any, Dict, List
import json
from tqdm import tqdm
from .monitor_utils import ProcessMonitor

class StanceInput:
    def __init__(self, claim, evidence):
        self.claim = claim
        self.evidence = evidence

class StanceModel:
    def __init__(self, model_name, device, local_files_only=False):
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_files_only)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self._model.to(device)
        self._device = device
        self.softmax = torch.nn.Softmax(dim=1)

    def _encode(self, input: StanceInput) -> BatchEncoding:
        """Encodes the input into a Batch."""
        inputs = self._tokenizer.batch_encode_plus(
            [[input.claim, doc['snippet']] for doc in input.evidence],
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        return inputs

    def predict(self, stance_inp: StanceInput):
        """Generates predictions based on the input."""
        inputs = self._encode(stance_inp).to(self._device)  # Move to device
        with torch.no_grad():
            logits = self._model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"]
            ).logits
            softmax = self.softmax(logits).cpu().numpy().tolist()  # Move back to CPU and convert to list
            predictions = np.argmax(softmax, axis=1).flatten().tolist()  # Get predictions

        stance_counts = Counter(predictions)    # Count each stance type
        final_stance = stance_counts.most_common(1)[0][0]  # Get the most common stance
        return final_stance


def metrics(true_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
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

    print(classification_report(true_labels, pred_labels))
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
    print(results)


def predict_for_file(file_path):
    with ProcessMonitor(dataset="elections") as pm:
        pm.start()

        model_path = "./models/xlmr_model"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stance_model = StanceModel(model_path, device, local_files_only=True)

        with open(file_path, 'r') as file:
            data = json.load(file)

        def convert_label_to_int(value) -> int:
            return 1 if value == "SUPPORTED" else 0
            
        predictions, true_labels = [], []
        for item in tqdm(data):
            evidence = [{"snippet": item['context']}]
            st_input = StanceInput(item['claim'], evidence)
            predictions.append(stance_model.predict(st_input))
            true_labels.append(convert_label_to_int(item['label']))
        metrics(true_labels=true_labels, pred_labels=predictions)

if __name__ == '__main__':
    predict_for_file(file_path="data/elections/claim_verification/hover_dev_claim_verification.json")
