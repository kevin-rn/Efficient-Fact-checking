#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
claim_name=$1
setting=$2
hover_stage=$3
retrieval_mode=${4:-"cpu"}
subvectors=${5:-96}

folder_name=$(echo "$setting" | sed 's/enwiki-[0-9]*-//')

conda activate grounding

# echo -e "FAISS"
# python3 -m src.retrieval.faiss.run_faiss_search \
# --dataset_name=$claim_name \
# --setting=$setting \
# --compress_embed

echo -e "\nIndex Compression"
# For different reranking datasets setup the jpq format data
python -m src.tools.qrels_format_enwiki \
  --dataset_name $claim_name \
  --setting $setting

model_setting=$(echo "$setting" | sed 's/-[^-]*$//')
if ! [ -d "./models/$model_setting" ]; 
  then
    # Skip JPQ model training if encoders already exists
    echo -e "Preprocess data\n"
    python -m src.retrieval.JPQ.jpq.preprocess \
      --data_type 0 \
      --dataset $claim_name \
      --enwiki_name $setting

    echo -e "\nEncode Documents\n"
    python -m src.retrieval.JPQ.jpq.run_init \
      --preprocess_dir ./data/jpq/doc/preprocess/ \
      --model_dir ./data/jpq/doc/star \
      --max_doc_length 512 \
      --output_dir ./data/jpq/doc/init/$subvectors \
      --subvector_num $subvectors

    echo -e "\nEncode Query\n"
    python -m src.retrieval.JPQ.jpq.run_train \
        --preprocess_dir ./data/jpq/doc/preprocess \
        --model_save_dir ./data/jpq/doc/train/m${subvectors}/models \
        --log_dir ./data/jpq/doc/train/m${subvectors}/log \
        --init_index_path ./data/jpq/doc/init/$subvectors/OPQ${subvectors},IVF1,PQ${subvectors}x8.index \
        --init_model_path ./data/jpq/doc/star \
        --lambda_cut 10 \
        --gpu_search \
        --centroid_lr 1e-4 \
        --train_batch_size 32

    echo -e "\nConvert encoders to JPQTower\n"
    python -m src.retrieval.JPQ.jpq.convert_model \
        --data_type 0 \
        --subvector_num=$subvectors \
        --enwiki_name=$setting
fi

echo -e "\nRun Inference"
if [ "$hover_stage" = "sent_select" ]
  then
    if [ "$retrieval_mode" = "gpu" ]
      then
        python -m src.retrieval.JPQ.run_inference \
          --dataset_name $claim_name \
          --setting=$setting \
          --subvectors_num $subvectors \
          --sent_select \
          --use_gpu
    else
        python -m src.retrieval.JPQ.run_inference \
          --dataset_name $claim_name \
          --setting=$setting \
          --subvectors_num $subvectors \
          --sent_select
    fi
else
    if [ "$retrieval_mode" = "gpu" ]
      then
        python -m src.retrieval.JPQ.run_inference \
          --dataset_name $claim_name \
          --setting=$setting \
          --subvectors_num $subvectors \
          --use_gpu
    else
        python -m src.retrieval.JPQ.run_inference \
          --dataset_name $claim_name \
          --setting=$setting \
          --subvectors_num $subvectors
    fi
fi

if [ "$hover_stage" = "sent_select" ]
  then
    echo -e "\nSENTENCE SELECTION"
    conda activate hover
    ./src/hover/train_scripts/train_sent_retrieval.sh $claim_name
    ./src/hover/eval_scripts/eval_sent_retrieval_on_train.sh $claim_name
    ./src/hover/eval_scripts/eval_sent_retrieval_on_dev.sh $claim_name

    echo -e "\nCLAIM VERIFICATION"
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
    ./src/hover/train_scripts/train_claim_verification.sh $claim_name
    ./src/hover/eval_scripts/eval_claim_verification_on_dev.sh $claim_name
    python3 -m src.tools.evaluate_metrics --dataset_name $claim_name

    mkdir data/hover_files/$claim_name/faiss/$folder_name-compress
    mv data/$claim_name/claim_verification data/hover_files/$claim_name/faiss/$folder_name-compress
    mkdir data/$claim_name/claim_verification

    mkdir out/$claim_name/faiss/$folder_name-compress
    mv out/$claim_name/exp1.0/claim_verification out/$claim_name/faiss/$folder_name-compress
fi
cd ..