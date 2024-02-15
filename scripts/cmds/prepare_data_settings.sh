#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate grounding

echo -e "original\n"
python3 data_processing.py --setting=original --pre_compute_embed
python3 data_processing.py --setting=original --first_para_only --pre_compute_embed

echo -e "\ncite\n"
python3 data_processing.py --setting=cite --pre_compute_embed
python3 data_processing.py --setting=cite --first_para_only --pre_compute_embed

echo -e "\nclaim\n"
python3 data_processing.py --setting=claim --pre_compute_embed
python3 data_processing.py --setting=claim --first_para_only --pre_compute_embed

echo -e "\nfusion\n"
python3 data_processing.py --setting=fusion --pre_compute_embed
python3 data_processing.py --setting=fusion --first_para_only --pre_compute_embed