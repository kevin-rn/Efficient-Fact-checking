#!/bin/bash
set -e

eval "$(conda shell.bash hook)"

claim_name=$1
setting=$2
subvectors=$3

conda activate grounding

echo -e "Format data\n"
python -m jpq.enwiki_data_formatter \
  --dataset_name $claim_name \
  --enwiki_name $setting

echo -e "Preprocess data\n"
python -m jpq.preprocess \
  --data_type 0 \
  --dataset enwiki

echo -e "\nEncode Documents\n"
python -m jpq.run_init \
  --preprocess_dir ./data/doc/preprocess/ \
  --model_dir ./data/doc/star \
  --max_doc_length 512 \
  --output_dir ./data/doc/init \
  --subvector_num $subvectors

echo -e "\nEncode Query\n"
python -m jpq.run_train \
    --preprocess_dir ./data/doc/preprocess \
    --model_save_dir ./data/doc/train/m${subvectors}/models \
    --log_dir ./data/doc/train/m${subvectors}/log \
    --init_index_path ./data/doc/init/OPQ${subvectors},IVF1,PQ${subvectors}x8.index \
    --init_model_path ./data/doc/star \
    --lambda_cut 10 \
    --gpu_search \
    --centroid_lr 1e-4 \
    --train_batch_size 32

mkdir data/passage/init/m96
mv data/doc/init/OPQ96,IVF1,PQ96x8.index ata/passage/init/m96
mv data/doc/init/doc_embed.memmap

echo -e "\nConvert encoders to JPQTower\n"
python -m jpq.convert_model \
    --data_type 0 \
    --subvector_num=$subvectors

echo -e "\nRun Inference\n"
python -m run_inference.py \
  --dataset_name \
  --data_split train \
  --subvectors_num $subvectors \
  --batch_size 128

python -m run_inference.py \
  --dataset_name \
  --data_split dev \
  --subvectors_num $subvectors \
  --batch_size 1