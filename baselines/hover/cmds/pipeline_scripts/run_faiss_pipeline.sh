#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
claim_name=$1
db_name=$2
faiss_type=$3

echo -e "FAISS\n"
conda activate grounding
if [ "$faiss_type" = "compress" ]
  then
    python faiss/run_faiss_search.py \
    --dataset_name=$claim_name \
    --setting=$db_name \
    --precompute_embed \
    --compress_embed

    echo -e "\nCLAIM VERIFICATION\n"
    conda activate hover
    ./cmds/train_scripts/train_claim_verification.sh $claim_name
    ./cmds/eval_scripts/eval_claim_verification_on_dev.sh $claim_name

    mkdir data/hover_files/faiss/$claim_name-$db_name-compress
    mv data/$claim_name/claim_verification data/hover_files/faiss/$claim_name-$db_name-compress
    mv data/$claim_name/sent_retrieval data/hover_files/faiss/$claim_name-$db_name-compress
    mkdir data/$claim_name/claim_verification
    mkdir data/$claim_name/sent_retrieval

    mkdir out/$claim_name/faiss/$claim_name-$db_name-compress
    mv out/$claim_name/exp1.0/claim_verification out/$claim_name/faiss/$claim_name-$db_name-compress

elif [ "$faiss_type" = "select" ]
  then
    python faiss/run_faiss_search.py \
    --dataset_name=$claim_name \
    --setting=$db_name \
    --precompute_embed \
    --hover_stage=sent_retrieval

    echo -e "\nSENTENCE SELECTION\n"
    conda activate hover
    ./cmds/train_scripts/train_sent_retrieval.sh $claim_name
    ./cmds/eval_scripts/eval_sent_retrieval_on_train.sh $claim_name
    ./cmds/eval_scripts/eval_sent_retrieval_on_dev.sh $claim_name

    echo -e "\nCLAIM VERIFICATION\n"
    python3 prepare_data_for_claim_verification.py --dataset_name=$claim_name --data_split=train
    python3 prepare_data_for_claim_verification.py --dataset_name=$claim_name --data_split=dev
    ./cmds/train_scripts/train_claim_verification.sh $claim_name
    ./cmds/eval_scripts/eval_claim_verification_on_dev.sh $claim_name

    mkdir data/hover_files/faiss/$claim_name-$db_name-select
    mv data/$claim_name/claim_verification data/hover_files/faiss/$claim_name-$db_name-select
    mv data/$claim_name/sent_retrieval data/hover_files/faiss/$claim_name-$db_name-select
    mkdir data/$claim_name/claim_verification
    mkdir data/$claim_name/sent_retrieval

    mkdir out/$claim_name/faiss/$claim_name-$db_name-select
    mv out/$claim_name/exp1.0/claim_verification out/$claim_name/faiss/$claim_name-$db_name-select
    mv out/$claim_name/exp1.0/sentence_retrieval out/$claim_name/faiss/$claim_name-$db_name-select
  else
    python faiss/run_faiss_search.py \
    --setting=$db_name \
    --dataset_name=$claim_name \
    --precompute_embed

    echo -e "\nCLAIM VERIFICATION\n"
    conda activate hover
    ./cmds/train_scripts/train_claim_verification.sh $claim_name
    ./cmds/eval_scripts/eval_claim_verification_on_dev.sh $claim_name

    mkdir data/hover_files/faiss/$claim_name-$db_name
    mv data/$claim_name/claim_verification data/hover_files/faiss/$claim_name-$db_name
    mv data/$claim_name/sent_retrieval data/hover_files/faiss/$claim_name-$db_name
    mkdir data/$claim_name/claim_verification
    mkdir data/$claim_name/sent_retrieval

    mkdir out/$claim_name/faiss/$claim_name-$db_name
    mv out/$claim_name/exp1.0/claim_verification out/$claim_name/faiss/$claim_name-$db_name
fi