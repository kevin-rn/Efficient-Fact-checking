#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
claim_name=$1
setting=$2
bm25_type=$3
folder_name=$(echo "$setting" | sed 's/enwiki-[^-]*-//')

echo -e "BM25"
conda activate grounding
python3 bm25/run_bm25_search.py \
  --dataset_name=$claim_name \
  --setting=$setting

echo -e "\nDOC RETRIEVAL"
cd hover
conda activate hover
python3 prepare_data_for_doc_retrieval.py \
  --dataset_name=$claim_name \
  --data_split=train \
  --doc_retrieve_range=20 \
  --setting=$setting
python3 prepare_data_for_doc_retrieval.py \
  --dataset_name=$claim_name \
  --data_split=dev \
  --doc_retrieve_range=20 \
  --setting=$setting
./cmds/train_scripts/train_doc_retrieval.sh $claim_name
./cmds/eval_scripts/eval_doc_retrieval_on_train.sh $claim_name
./cmds/eval_scripts/eval_doc_retrieval_on_dev.sh $claim_name

if [ "$bm25_type" = "original" ]
  then
    echo -e "\nSENTENCE RETRIEVAL"
    latest="$(ls -td  out/$claim_name/exp1.0/doc_retrieval/checkpoint-*/ | head -1)"
    doc_model=$(basename "$latest" | sed 's/checkpoint-//')
    python3 prepare_data_for_sent_retrieval.py \
      --data_split=train \
      --sent_retrieve_range=5 \
      --doc_retrieval_model_global_step=$doc_model \
      --dataset_name=$claim_name \
      --setting=$setting
    python3 prepare_data_for_sent_retrieval.py \
      --data_split=dev \
      --sent_retrieve_range=5 \
      --doc_retrieval_model_global_step=$doc_model \
      --dataset_name=$claim_name \
      --setting=$setting
    ./cmds/train_scripts/train_sent_retrieval.sh $claim_name
    ./cmds/eval_scripts/eval_sent_retrieval_on_train.sh $claim_name
    ./cmds/eval_scripts/eval_sent_retrieval_on_dev.sh $claim_name

    echo -e "\nCLAIM VERIFICATION"
    latest="$(ls -td  out/$claim_name/exp1.0/sent_retrieval/checkpoint-*/ | head -1)"
    sent_model=$(basename "$latest" | sed 's/checkpoint-//')
    python3 prepare_data_for_claim_verification.py \
      --dataset_name=$claim_name \
      --data_split=train \
      --sent_retrieval_model_global_step=$sent_model
    python3 prepare_data_for_claim_verification.py \
      --dataset_name=$claim_name \
      --data_split=dev \
      --sent_retrieval_model_global_step=$sent_model
    ./cmds/train_scripts/train_claim_verification.sh $claim_name
    ./cmds/eval_scripts/eval_claim_verification_on_dev.sh $claim_name

    mkdir data/hover_files/$claim_name/bm25/$folder_name
    mv data/$claim_name/bm25_retrieved data/hover_files/$claim_name/bm25/$folder_name
    mv data/$claim_name/claim_verification data/hover_files/$claim_name/bm25/$folder_name
    mv data/$claim_name/doc_retrieval data/hover_files/$claim_name/bm25/$folder_name
    mv data/$claim_name/sent_retrieval data/hover_files/$claim_name/bm25/$folder_name
    mkdir data/$claim_name/bm25_retrieved
    mkdir data/$claim_name/claim_verification
    mkdir data/$claim_name/doc_retrieval
    mkdir data/$claim_name/sent_retrieval

    mkdir  out/$claim_name/bm25/$folder_name
    mv  out/$claim_name/exp1.0/claim_verification out/$claim_name/bm25/$folder_name
    mv  out/$claim_name/exp1.0/doc_retrieval out/$claim_name/bm25/$folder_name
    mv  out/$claim_name/exp1.0/sent_retrieval out/$claim_name/bm25/$folder_name
  else
    echo -e "\nCLAIM VERIFICATION"
    conda activate grounding
    python3 prepare_doc_retrieve_for_claim_verification.py \
      --dataset_name=$claim_name \
      --data_split=train \
      --doc_retrieve_range=5
    python3 prepare_doc_retrieve_for_claim_verification.py \
      --dataset_name=$claim_name \
      --data_split=dev \
      --doc_retrieve_range=5

    conda activate hover
    ./cmds/train_scripts/train_claim_verification.sh $claim_name
    ./cmds/eval_scripts/eval_claim_verification_on_dev.sh $claim_name

    mkdir data/hover_files/$claim_name/bm25/$folder_name
    mv data/$claim_name/bm25_retrieved data/hover_files/$claim_name/bm25/$folder_name
    mv data/$claim_name/claim_verification data/hover_files/$claim_name/bm25/$folder_name
    mv data/$claim_name/doc_retrieval data/hover_files/$claim_name/bm25/$folder_name
    mkdir data/$claim_name/bm25_retrieved
    mkdir data/$claim_name/claim_verification
    mkdir data/$claim_name/doc_retrieval

    mkdir  out/$claim_name/bm25/$folder_name
    mv  out/$claim_name/exp1.0/claim_verification out/$claim_name/bm25/$folder_name
    mv  out/$claim_name/exp1.0/doc_retrieval out/$claim_name/bm25/$folder_name
fi

cd ..