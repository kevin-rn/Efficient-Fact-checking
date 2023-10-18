import pandas as pd
import numpy as np
from transformers import AutoTokenizer, RealmForOpenQA, RealmRetriever, RealmConfig
from tqdm import tqdm

df_wiki_dev = pd.read_json('dev.json')
corpus_data = np.load('wikimusique_corpus.npy', allow_pickle=True)

tokenizer = AutoTokenizer.from_pretrained("google/realm-orqa-nq-openqa")
retriever = RealmRetriever.from_pretrained("google/realm-orqa-nq-openqa", block_records=corpus_data)
config = RealmConfig()
config.num_block_records = len(corpus_data)
model = RealmForOpenQA.from_pretrained("google/realm-orqa-nq-openqa", config=config, retriever=retriever, ignore_mismatched_sizes=True)


predicted_vals = []
loss_vals = []

for i in tqdm(range(len(df_wiki_dev))):
    question = df_wiki_dev['question'][i]
    question_ids = tokenizer([question], return_tensors="pt")
    answer_ids = tokenizer(
        [df_wiki_dev['answer'][i]],
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False,
    ).input_ids

    reader_output, predicted_answer_ids = model(**question_ids, answer_ids=answer_ids, return_dict=False)
    predicted_answer = tokenizer.decode(predicted_answer_ids)
    loss = reader_output.loss

    predicted_vals.append(predicted_answer)
    loss_vals.append(loss)

df_wiki_dev['predicted'] = predicted_vals
df_wiki_dev['loss'] = loss_vals
df_wiki_dev.to_csv('wikimultihop_output.csv')
