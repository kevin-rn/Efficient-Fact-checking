# !/bin/bash
set -eu -o pipefail

# Create folders
mkdir -p data
mkdir -p models
mkdir -p out

# Setup data folder structure
cd data
mkdir -p db_files
mkdir -p enwiki_files
mkdir -p embed_files
mkdir -p hover_files
# cd db_files
# wget https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db
# cd ..

# Setup 2017 HotPotQA data.
wget https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2
tar -xf enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2 -C enwiki_files
rm enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2
mv enwiki_files/enwiki-20171001-pages-meta-current-withlinks-processed enwiki_files/enwiki-2017-original

# Setup HoVer folders
mkdir -p hover
cd hover
mkdir -p doc_retrieval
mkdir -p sent_retrieval
mkdir -p claim_verification
mkdir -p bm25_retrieved

# Commented out as we recreate these files using our own retrieval methods.
# mkdir -p tfidf_retrieved
# cd tfidf_retrieved
# wget https://nlp.cs.unc.edu/data/hover/train_tfidf_doc_retrieval_results.json
# wget https://nlp.cs.unc.edu/data/hover/dev_tfidf_doc_retrieval_results.json
# wget https://nlp.cs.unc.edu/data/hover/test_tfidf_doc_retrieval_results.json
cd ..

# Setup WiCE folders and claim data
mkdir -p wice
cd wice
mkdir -p doc_retrieval
mkdir -p sent_retrieval
mkdir -p claim_verification
mkdir -p bm25_retrieved
cd ../..

python3 src/tools/setup_wice.py

