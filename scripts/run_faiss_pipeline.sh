#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
claim_name=$1
setting=$2
hover_stage=${3:-"claim_verification"}
retrieval_mode=${4:-"cpu"}
folder_name=$(echo "$setting" | sed 's/enwiki-[0-9]*-//')

echo -e "FAISS\n"
conda activate grounding
if [ "$hover_stage" = "sent_select" ]
  then
    if [ "$retrieval_mode" = "gpu" ]
      then
        python3 -m src.retrieval.faiss.run_faiss_search \
        --dataset_name=$claim_name \
        --setting=$setting \
        --hover_stage=sent_retrieval \
        --precompute_embed \
        --use_gpu
      else
        python3 -m src.retrieval.faiss.run_faiss_search \
        --dataset_name=$claim_name \
        --setting=$setting \
        --hover_stage=sent_retrieval \
        --precompute_embed
      fi

    echo -e "\nSENTENCE SELECTION\n"
    conda activate hover
    ./src/hover/train_scripts/train_sent_retrieval.sh $claim_name
    ./src/hover/eval_scripts/eval_sent_retrieval_on_train.sh $claim_name
    ./src/hover/eval_scripts/eval_sent_retrieval_on_dev.sh $claim_name

    echo -e "\nCLAIM VERIFICATION\n"
    latest="$(ls -td out/$claim_name/exp1.0/sent_retrieval/checkpoint-*/ | head -1)"
    sent_model=$(basename "$latest" | sed 's/checkpoint-//')
    python3 -m src.hover.prepare_data_for_claim_verification \
      --dataset_name=$claim_name \
      --data_split=train \
      --sent_retrieval_model_global_step=$sent_model
    python3 -m src.hover.prepare_data_for_claim_verification \
      --dataset_name=$claim_name \
      --data_split=dev \
      --sent_retrieval_model_global_step=$sent_model
    ./src/hover/train_scripts/train_claim_verification.sh $claim_name
    ./src/hover/eval_scripts/eval_claim_verification_on_dev.sh $claim_name
    python3 -m src.tools.evaluate_metrics --dataset_name $claim_name

    mkdir data/hover_files/$claim_name/faiss/$folder_name-select
    mv data/$claim_name/claim_verification data/hover_files/$claim_name/faiss/$folder_name-select
    mv data/$claim_name/sent_retrieval data/hover_files/$claim_name/faiss/$folder_name-select
    mkdir data/$claim_name/claim_verification
    mkdir data/$claim_name/sent_retrieval

    mkdir out/$claim_name/faiss/$folder_name-select
    mv out/$claim_name/exp1.0/claim_verification out/$claim_name/faiss/$folder_name-select
    mv out/$claim_name/exp1.0/sent_retrieval out/$claim_name/faiss/$folder_name-select
  else
      if [ "$retrieval_mode" = "gpu" ]
        then
          python3 -m src.retrieval.faiss.run_faiss_search \
          --setting=$setting \
          --dataset_name=$claim_name \
          --precompute_embed \
          --use_gpu
        else
          python3 -m src.retrieval.faiss.run_faiss_search \
          --setting=$setting \
          --dataset_name=$claim_name \
          --precompute_embed
      fi

    echo -e "\nCLAIM VERIFICATION\n"
    conda activate hover
    ./src/hover/train_scripts/train_claim_verification.sh $claim_name
    ./src/hover/eval_scripts/eval_claim_verification_on_dev.sh $claim_name
    python3 -m src.tools.evaluate_metrics \
      --dataset_name $claim_name

    mkdir data/hover_files/$claim_name/faiss/$folder_name
    mv data/$claim_name/claim_verification data/hover_files/$claim_name/faiss/$folder_name
    mkdir data/$claim_name/claim_verification

    mkdir out/$claim_name/faiss/$folder_name
    mv out/$claim_name/exp1.0/claim_verification out/$claim_name/faiss/$folder_name
fi
cd ..