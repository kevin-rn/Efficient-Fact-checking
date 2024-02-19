#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate grounding

wikifolder=$1

echo -e "original\n"
python3 data_processing.py --setting=$wikifolder-original --pre_compute_embed
python3 data_processing.py --setting=$wikifolder-original --first_para_only --pre_compute_embed

echo -e "\ncite\n"
python3 data_processing.py --setting=$wikifolder-cite --pre_compute_embed
python3 data_processing.py --setting=$wikifolder-cite --first_para_only --pre_compute_embed

echo -e "\nclaim\n"
python3 data_processing.py --setting=$wikifolder-claim --pre_compute_embed
python3 data_processing.py --setting=$wikifolder-claim --first_para_only --pre_compute_embed

echo -e "\nfusion\n"
python3 data_processing.py --setting=$wikifolder-fusion --pre_compute_embed
python3 data_processing.py --setting=$wikifolder-fusion --first_para_only --pre_compute_embed