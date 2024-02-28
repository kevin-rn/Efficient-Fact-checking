import pandas as pd
import numpy as np
from transformers import AutoTokenizer, RealmForOpenQA, RealmRetriever, RealmConfig
import torch
from tqdm import tqdm

class CorpusCreator:
    def __init__(self, data_dir="data/QA-dataset/data"):
        """
        Initialize a CorpusCreator object with the specified data directory.

        Args:
            data_dir (str): The directory where the data files are located.
        """
        self.data_dir = data_dir
        self.df_wiki_corpus = None
        self.df_musique_corpus = None

    def __read_wiki_data(self):
        """
        Load 2WikiMultiHopQA data and preprocess it.
        """
        df_wiki_train = pd.read_json(f"{self.data_dir}/2wikimultihopQA/train.json")
        df_wiki_dev = pd.read_json(f"{self.data_dir}/2wikimultihopQA/dev.json")
        df_wiki_test = pd.read_json(f"{self.data_dir}/2wikimultihopQA/test.json")

        df_wiki = pd.concat([df_wiki_train, df_wiki_dev, df_wiki_test])
        df_wiki.drop(columns=["type", "supporting_facts", "evidences"], inplace=True)

        df_wiki = df_wiki.explode("context", ignore_index=True)
        self.df_wiki_corpus = df_wiki["context"].apply(
            lambda row: row[0] + " - " + " ".join(row[1])
        )
        self.df_wiki_corpus.drop_duplicates(ignore_index=True, inplace=True)

    def __read_musique_data(self):
        """
        Load MuSiQueQA data and preprocess it.
        """
        df_musique_train = pd.read_json(
            f"{self.data_dir}/musique/musique_full_v1.0_train.jsonl", lines=True
        )
        df_musique_dev = pd.read_json(
            f"{self.data_dir}/musique/musique_full_v1.0_dev.jsonl", lines=True
        )
        df_musique_test = pd.read_json(
            f"{self.data_dir}/musique/musique_full_v1.0_test.jsonl", lines=True
        )

        df_musique = pd.concat([df_musique_train, df_musique_dev, df_musique_test])
        df_musique.drop(
            columns=["question_decomposition", "answer_aliases", "answerable"],
            inplace=True,
        )

        df_musique = df_musique.explode("paragraphs", ignore_index=True)
        self.df_musique_corpus = df_musique["paragraphs"].apply(
            lambda row: row["title"] + " - " + row["paragraph_text"]
        )
        self.df_musique_corpus.drop_duplicates(ignore_index=True, inplace=True)

    def create_corpus(self):
        """
        Combine the two corpora into a single corpus, shuffle it, and save it as an npy file.
        """
        self.__read_wiki_data()
        self.__read_musique_data()
        df_corpus = pd.concat(
            [self.df_wiki_corpus.astype(str), self.df_musique_corpus.astype(str)]
        )
        df_corpus = df_corpus.sample(frac=1, random_state=42)
        df_corpus = df_corpus.reset_index(drop=True)

        np_corpus = df_corpus.to_numpy()
        for i in range(len(np_corpus)):
            np_corpus[i] = np_corpus[i].encode()

        np.save("data/wikimusique_corpus.npy", np_corpus)


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
