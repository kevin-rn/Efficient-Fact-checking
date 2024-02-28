#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
claim_name=$1
setting=$2
hover_stage=$3

folder_name=$(echo "$setting" | sed 's/enwiki-[0-9]*-//')

# echo -e "FAISS"
conda activate grounding
# python3 faiss/run_faiss_search.py \
# --dataset_name=$claim_name \
# --setting=$setting \
# --compress_embed
echo -e "\nIndex Compression"
cd JPQ
./cmds/jpq_train_inference.sh $claim_name $setting 96 $hover_stage
cd ../hover

if [ "$hover_stage" = "sent_select" ]
  then
    echo -e "\nSENTENCE SELECTION"
    conda activate hover
    ./cmds/train_scripts/train_sent_retrieval.sh $claim_name
    ./cmds/eval_scripts/eval_sent_retrieval_on_train.sh $claim_name
    ./cmds/eval_scripts/eval_sent_retrieval_on_dev.sh $claim_name

    echo -e "\nCLAIM VERIFICATION"
    latest="$(ls -td out/$claim_name/exp1.0/sent_retrieval/checkpoint-*/ | head -1)"
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

    mkdir data/hover_files/$claim_name/faiss/$folder_name-compress-select
    mv data/$claim_name/claim_verification data/hover_files/$claim_name/faiss/$folder_name-compress-select
    mv data/$claim_name/sent_retrieval data/hover_files/$claim_name/faiss/$folder_name-compress-select
    mkdir data/$claim_name/claim_verification
    mkdir data/$claim_name/sent_retrieval

    mkdir out/$claim_name/faiss/$folder_name-compress-select
    mv out/$claim_name/exp1.0/claim_verification out/$claim_name/faiss/$folder_name-compress-select
    mv out/$claim_name/exp1.0/sent_retrieval out/$claim_name/faiss/$folder_name-compress-select
else
    echo -e "\nCLAIM VERIFICATION"
    conda activate hover
    ./cmds/train_scripts/train_claim_verification.sh $claim_name
    ./cmds/eval_scripts/eval_claim_verification_on_dev.sh $claim_name

    mkdir data/hover_files/$claim_name/faiss/$folder_name-compress
    mv data/$claim_name/claim_verification data/hover_files/$claim_name/faiss/$folder_name-compress
    mkdir data/$claim_name/claim_verification

    mkdir out/$claim_name/faiss/$folder_name-compress
    mv out/$claim_name/exp1.0/claim_verification out/$claim_name/faiss/$folder_name-compress
fi
cd ..