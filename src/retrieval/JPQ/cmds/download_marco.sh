set -e 

mkdir -p data/jpq
mkdir -p data/jpq/passage
mkdir -p data/jpq/passage/dataset
cd data/jpq/passage/dataset

# download MSMARCO passage data https://microsoft.github.io/msmarco/Datasets.html
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar xvfz collectionandqueries.tar.gz -C ./

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip ./msmarco-test2019-queries.tsv.gz 

wget https://trec.nist.gov/data/deep/2019qrels-pass.txt

cd ../../../../
mkdir -p data/jpq/doc
mkdir -p data/jpq/doc/dataset
cd data/jpq/doc/dataset

# download MSMARCO doc data
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
gunzip msmarco-docs.tsv.gz

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz
gunzip msmarco-doctrain-queries.tsv.gz

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz
gunzip msmarco-doctrain-qrels.tsv.gz

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip msmarco-test2019-queries.tsv.gz

wget https://trec.nist.gov/data/deep/2019qrels-docs.txt

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz
gunzip msmarco-docdev-queries.tsv.gz

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz
gunzip msmarco-docdev-qrels.tsv.gz

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/docleaderboard-queries.tsv.gz
gunzip docleaderboard-queries.tsv.gz
cd ../../../../