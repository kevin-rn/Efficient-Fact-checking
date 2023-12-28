from argparse import ArgumentParser
import json
from bm25_search import BM25Search
from typing import List, Dict
import os
import sqlite3
import uuid
import sys

sys.path.append(sys.path[0] + '/../../..')
from scripts.monitor_utils import monitor

parser = ArgumentParser()

parser.add_argument(
   "--data_split",
   default=None,
   type=str,
   required=True,
   help="[train | dev ]",
)

args = parser.parse_args()

def get_corpus(db_path: str) -> tuple[Dict[str, Dict[str,str]], List[Dict[str, str]]]:
   """
   Retrieves document data from database and creates dictionary for each document.

   Parameters:
      - db_path (str): database path to use.

   Returns:
      Dictionary of the format { "id": {"title": _ , "text": _}, "id": {"title": _, "text": _}, .. }
   """
   conn = sqlite3.connect(db_path)
   wiki_db = conn.cursor()
   results = wiki_db.execute("SELECT id, text FROM documents ORDER BY id COLLATE NOCASE ASC").fetchall()

   corpus = {} 
   for id, text in results:
      uid = str(uuid.uuid5(uuid.NAMESPACE_OID, str(id)))
      doc_dict = {uid: {"title": str(id), "text": str(text).replace("[SENT]", " ")}}
      corpus.update(doc_dict)
   conn.close()
   return corpus

def get_queries(hover_claim_path) -> Dict[str, str]:
   """
   Retrieves claim information from hover and creates a dictionary of claims.

   Parameters:
      - hover_claim_path (str): Path to HoVer claim file.

   Returns:
      Dictionary of the format {"id": "text", "id": "text", "id": ..}
   """
   queries = {} 
   with open(hover_claim_path, 'r') as json_file:
      claim_json = json.load(json_file)
   for claim in claim_json:
      queries.update({claim['uid']: claim['claim']})
      #print(queries)
   return queries, claim_json

def search_and_retrieve(corpus_path: str, data_split: str) -> None:
   """
   Retrieves top-100 documents for a given claim using BM-25 (ElasticSearch).

   Parameters:
      - corpus_path (str): Path location to the wikipedia document corpus.
      - data_split (str): Name of the datasplit to process for e.g. train or dev.
   """
   query_path = os.path.join("..", "data", "hover", f"hover_{data_split}_release_v1.1.json")
   corpus = get_corpus(corpus_path)
   queries, claim_list = get_queries(query_path)

   # Batch size 1 to represent real-world inference time
   bm25_search = BM25Search(index_name="hover",initialize=True, batch_size=1)
   response = bm25_search.retrieve(corpus, queries, 100)

   bm25_doc_results = []
   for idx, (claim_uid, doc_results) in enumerate(response.items()):
      # Get claim information
      current_claim = claim_list[idx]
      assert current_claim['uid'] == claim_uid

      # get bm25 document retrieval results
      doc_list, prob_list = [], []
      for doc_uid, doc_prob in doc_results.items():
         doc_list.append(corpus[doc_uid]['title'])
         prob_list.append(doc_prob)

      # Check if supporting titles are in the list
      supporting_titles = [title[0] for title in current_claim['supporting_facts']]
      support = 1 if all(title in doc_list for title in supporting_titles) else 0

      # Convert to HoVer format
      json_claim = {"id": claim_uid, 
                    "verifiable": "VERIFIABLE", 
                    "label": current_claim['label'], 
                    "claim": current_claim['claim'], 
                    "evidence": [current_claim['supporting_facts']], 
                    "doc_retrieval_results": [[doc_list, prob_list], support]}
      bm25_doc_results.append(json_claim)

   result_path = os.path.join("..", "data", "hover", "bm25_retrieved", f"{data_split}_bm25_doc_retrieval_results.json")
   with open(result_path, 'w', encoding="utf-8") as f:
      json.dump(bm25_doc_results, f)


def main():
   # try:
   #    enwiki_db = os.path.join("..", "data", "db_files", "cite", "wiki_wo_links-cite-full.db")
   #    search_and_retrieve(enwiki_db, args.data_split)
   # except Exception as e:
   #    print(e)
   j = 0
   for i in range(5000):
      j += i
   print(j)

if __name__ == "__main__":
   # config = config_instance.get_all_params()
   monitor(main)