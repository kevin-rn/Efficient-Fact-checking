#!/bin/bash
set -eu -o pipefail

eval "$(conda shell.bash hook)"
db_name=$1

echo -e "BM25\n"
conda activate grounding
python3 bm25/run_bm25_search.py --db_name=$db_name

echo -e "\nDOC RETRIEVAL\n"
conda activate hover
python3 prepare_data_for_doc_retrieval.py --data_split=train --doc_retrieve_range=20 --db_name=$db_name
python3 prepare_data_for_doc_retrieval.py --data_split=dev --doc_retrieve_range=20 --db_name=$db_name
./train_scripts/train_doc_retrieval.sh
./eval_scripts/eval_doc_retrieval_on_train.sh
./eval_scripts/eval_doc_retrieval_on_dev.sh

echo -e "\nSENTENCE RETRIEVAL\n"
python3 prepare_data_for_sent_retrieval.py --data_split=train --sent_retrieve_range=5
python3 prepare_data_for_sent_retrieval.py --data_split=dev --sent_retrieve_range=5
./train_scripts/train_sent_retrieval.sh
./eval_scripts/eval_sent_retrieval_on_train.sh
./eval_scripts/eval_sent_retrieval_on_dev.sh

echo -e "\nCLAIM VERIFICATION\n"
python3 prepare_data_for_claim_verification.py --data_split=train
python3 prepare_data_for_claim_verification.py --data_split=dev
./train_scripts/train_claim_verification.sh
./eval_scripts/eval_claim_verification_on_dev.sh