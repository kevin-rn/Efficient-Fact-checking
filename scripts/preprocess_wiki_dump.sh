#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate grounding

wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
python -m src.wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2 -o data/enwiki_files/enwiki-latest --no-templates -c --json

echo -e "preprocess\n"
python3 src/tools/enwiki_processing.py \
    --start_loc=enwiki-latest \
    --end_loc=enwiki-latest-original \
    --process_function=preprocess

rm -rf data/enwiki_files/enwiki-latest