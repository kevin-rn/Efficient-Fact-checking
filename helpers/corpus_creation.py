import pandas as pd
import numpy as np

class CorpusCreator:
    def __init__(self, data_dir):
        """
        Initialize a CorpusCreator object with the specified data directory.

        Args:
            data_dir (str): The directory where the data files are located.
        """
        self.data_dir = data_dir
        self.df_wiki_corpus = None
        self.df_musique_corpus = None

    def read_wiki_data(self):
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

    def read_musique_data(self):
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
        if self.df_wiki_corpus is not None and self.df_musique_corpus is not None:
            df_corpus = pd.concat(
                [self.df_wiki_corpus.astype(str), self.df_musique_corpus.astype(str)]
            )
            df_corpus = df_corpus.sample(frac=1, random_state=42)
            df_corpus = df_corpus.reset_index(drop=True)

            np_corpus = df_corpus.to_numpy()
            for i in range(len(np_corpus):
                np_corpus[i] = np_corpus[i].encode()

            np.save("data/wikimusique_corpus.npy", np_corpus)

def main():
    ccreator = CorpusCreator(data_dir="data/QA-dataset/data")
    ccreator.read_wiki_data()
    ccreator.read_musique_data()
    ccreator.create_corpus()
    print("Finished creating the corpus")

if __name__ == "__main__":
    main()
