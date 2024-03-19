#!/bin/bash
set -e

claim_name=${1:-"hover"}

echo -e "\nBM25"
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=bm25/original-full 
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=bm25/claim-full
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=bm25/cite-full
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=bm25/fusion-full

echo -e "\nFAISS"
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=faiss/original-full-select
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=faiss/original-full
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=faiss/claim-full
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=faiss/cite-full
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=faiss/fusion-full

echo -e "\nJPQ"
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=faiss/original-full-compress-select
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=faiss/original-full-compress
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=faiss/claim-full-compress
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=faiss/cite-full-compress
python3 -m src.tools.evaluate_metrics  --dataset_name=$claim_name --out_dir=faiss/fusion-full-compress
