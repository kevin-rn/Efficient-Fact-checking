import pandas as pd
import numpy as np
from transformers import AutoTokenizer, RealmForOpenQA, RealmRetriever, RealmConfig
import torch
from tqdm import tqdm


def perform_realm_openqa(question_data, corpus_data):
    """Process Wikipedia data with the REALM model.

    Args:
        question_data (pd.DataFrame): DataFrame containing question-answer pairs.
        corpus_data (np.ndarray): Numpy array of byte strings representing external knowledge paragraphs.

    Returns:
        pd.DataFrame: DataFrame with predicted values and loss.
    """

    model_name = "google/realm-orqa-nq-openqa"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    retriever = RealmRetriever(block_records=corpus_data, tokenizer=tokenizer)
    config = RealmConfig()
    config.num_block_records = len(corpus_data)
    model = RealmForOpenQA.from_pretrained(
        model_name,
        config=config,
        retriever=retriever,
        ignore_mismatched_sizes=True,
    )

    predicted_vals = []
    loss_vals = []

    for i in tqdm(range(len(question_data))):
        question = question_data["question"][i]
        question_ids = tokenizer([question], return_tensors="pt")
        answer_ids = tokenizer(
            [question_data["answer"][i]],
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        ).input_ids

        try:
            with torch.no_grad():
                reader_output, predicted_answer_ids = model(
                    **question_ids, answer_ids=answer_ids, return_dict=False
                )
                predicted_answer = tokenizer.decode(predicted_answer_ids)
                loss = reader_output.loss

                predicted_vals.append(predicted_answer)
                loss_vals.append(loss)

        except Exception as e:
            print(f"Error processing question: {question}, with error: {e}")

    question_data["predicted"] = predicted_vals
    question_data["loss"] = loss_vals

    return question_data


# Load in data files
df_wiki_dev = pd.read_json("dev.json")
wikimusique_corpus = np.load("wikimusique_corpus.npy", allow_pickle=True)

# Perform inference
result_df = perform_realm_openqa(df_wiki_dev, wikimusique_corpus)
result_df.to_csv("wikimultihop_output.csv")
