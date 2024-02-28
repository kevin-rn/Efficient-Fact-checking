#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate grounding

cd ../wikiextractor
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
nohup python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2 -o ../data/extracted --no-templates -c --json &
cd ..
mv data/extracted baselines/hover/data/enwiki-files/enwiki-2024

cd scripts
echo -e "preprocess\n"
python3 enwiki_processing.py \
    --start_loc=extracted \
    --end_loc=enwiki-2024-original \
    --process_function=preprocess

echo -e "\ndatabase\n"
python3 data_processing.py \
    --setting=enwiki-2024-original \
    --pre_compute_embed