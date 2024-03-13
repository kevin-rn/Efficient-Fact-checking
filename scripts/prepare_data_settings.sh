#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate grounding

wikifolder=$1

echo -e "original"
python3 src/tools/data_processing.py --setting=$wikifolder-original --pre_compute_embed --store_original
# python3 src/tools/data_processing.py --setting=$wikifolder-original --first_para_only --pre_compute_embed --store_original

echo -e "\nclaim"
python3 src/tools/data_processing.py --setting=$wikifolder-claim --pre_compute_embed
# python3 src/tools/data_processing.py --setting=$wikifolder-claim --first_para_only --pre_compute_embed

echo -e "\ncite"
python3 src/tools/data_processing.py --setting=$wikifolder-cite --pre_compute_embed
# python3 src/tools/data_processing.py --setting=$wikifolder-cite --first_para_only --pre_compute_embed

echo -e "\nfusion"
python3 src/tools/data_processing.py --setting=$wikifolder-fusion --pre_compute_embed
# python3 src/tools/data_processing.py --setting=$wikifolder-fusion --first_para_only --pre_compute_embed