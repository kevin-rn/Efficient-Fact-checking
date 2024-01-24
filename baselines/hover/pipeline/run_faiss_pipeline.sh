#!/bin/bash
set -eu -o pipefail

eval "$(conda shell.bash hook)"
db_name=$1

echo -e "FAISS\n"
conda activate grounding
python faiss/run_faiss_search.py --setting=$db_name --compress_embed

echo -e "\nCLAIM VERIFICATION\n"
conda activate hover
./train_scripts/train_claim_verification.sh
./eval_scripts/eval_claim_verification_on_dev.sh