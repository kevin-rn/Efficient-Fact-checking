from argparse import ArgumentParser
import json
from bm25_search import BM25Search
from typing import List, Dict, Tuple, Any
import os
import sqlite3
import uuid
import sys

sys.path.append(sys.path[0] + '/../..')
from scripts.monitor_utils import ProcessMonitor

parser = ArgumentParser()
parser.add_argument(
    "--db_name",
    type=str,
    default=None,
    help="Name of the database files"
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

def get_queries(hover_claim_path) -> Tuple[Dict[str, str], Any]:
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
   return queries, claim_json

def search_and_retrieve(data_split: str, init_index: bool, batched: bool) -> None:
   """
   Retrieves top-100 documents for a given claim using BM-25 (ElasticSearch).

   Parameters:
      - data_split (str): Name of the datasplit to process for e.g. train or dev.
   """
   if args.db_name:
      corpus_path = os.path.join("data", "db_files", f"wiki_wo_links-{args.db_name}.db")
   else:
      corpus_path = os.path.join("data", "wiki_wo_links.db")
   query_path = os.path.join("data", "hover", f"hover_{data_split}_release_v1.1.json")

   corpus = get_corpus(corpus_path)
   queries, claim_list = get_queries(query_path)

   # Perform Indexing and retrieval
   batch_size = 32 if batched else 1
   bm25_search = BM25Search(index_name="hover", 
                            initialize=init_index, 
                            batch_size=batch_size, 
                            corpus=corpus)
   pm = ProcessMonitor()
   pm.start()
   response = bm25_search.retrieve(queries, 100)
   pm.stop()
   # Save to file
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

   result_path = os.path.join("data", "hover", "bm25_retrieved", f"{data_split}_bm25_doc_retrieval_results.json")
   with open(result_path, 'w', encoding="utf-8") as f:
      json.dump(bm25_doc_results, f)
   


def main():
   try:
      search_and_retrieve("train", init_index=True, batched=True)
      search_and_retrieve("dev", init_index=False, batched=False)
   except Exception as e:
      print(e)

if __name__ == "__main__":
   main()