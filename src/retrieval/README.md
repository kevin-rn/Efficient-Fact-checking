# Retrieval methods
For the retrieval setup the following methods are used: BM25 for Sparse retrieval, FAISS for Dense Retrieval and JPQ for Dense Retrieval with Index Compression. Additionally there is also a setup for alternative index compression method UNQ as part of the FAISS method but we did not succeed in succesfully training the encoders for this.

## BM25
For Sparse retrieval we used the BM25 from [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html) (version 8.11.3) and its [python package](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/installation.html) (version 7.9.1). This as alternative to [DrQA](https://github.com/facebookresearch/DrQA) from the original HoVer fact-checking pipeline. Ensure Elasticsearch instance is running and ``bm25/ElasticSearch.py`` contains correct credentials 

```console
usage: run_bm25_search.py [-h] --dataset_name=DATASET_NAME [--setting]

positional arguments:
    --dataset_name          Name of the claim dataset to run for [hover | wice]

optional arguments:
    -h, --help          show this help message and exit

    --setting               Name of the data corpus to run for (name of the database file).
                            This defaults to the standard wiki_wo_links.db file from the original repository.


Example:
foo@bar:~$ python -m src.retrieval.bm25.run_bm25_search --dataset_name=hover --setting=enwiki-2017-original
```
This method essentially retrieves for a train and dev file the top-100 relevant documents and formats it into the HoVer data format before the Neural Document retrieval stage.
```json
{...},
{
    "uid": "e822c12c-0f2c-4cb1-99cb-f4ff41fd436b",
    "claim": "Yakuza Kiwami and Yakuza 0 were released on the PlayStation 2 gaming console.",
    "supporting_facts": [
      [
        "Yakuza Kiwami",
        2
      ],
      [
        "Yakuza 0",
        3
      ]
    ],
    "label": "NOT_SUPPORTED",
    "num_hops": 2,
    "hpqa_id": "5ab29caa554299545a2cf9d3"
  },
{...},
```

Output will be saved to the subfolder ``bm25_retrieved`` inside the claim data folder e.g. for hover this would be ``data/hover/bm25_retrieved``

## FAISS
As an alternative to performing evidence retrieval and reranking by doing HoVer's Sparse retrieval and Neural-based Document retrieval, we suggest to perform Dense Retrieval.
For this we used the [FAISS](https://github.com/facebookresearch/faiss) library. Performing FAISS top-k document retrieval requires the creation of vector embeddings for the document and claim text. Here we leveraged the [all-MiniLM-L6-v2](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) model from [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html).

```console
usage: run_faiss_search.py [-h] --setting=SETTING --dataset_name=DATASET_NAME [--hover_stage=HOVER_STAGE]
                           [--rerank_mode=RERANK_MODE] [--topk_nn=TOPK_NN] [--n_rerank=N_RERANK] [--use_gpu]
                           [--precompute_embed] [--compress_embed]

positional arguments:
  --setting                 Name of the setting to run
  --dataset_name            Name of the claim dataset to run for [hover | wice] (default: hover)

optional arguments:
  -h, --help                show this help message and exit
  --hover_stage             Whether to perform sentence selection stage or immediately go to the claim verification stage (default: claim_verification)
  --topk_nn                 Top-k documents to retrieve for nearest neighbours FAISS (default: 5)
  --rerank_mode             Whether to rerank the sentences of the retrieved documents. This is either done: not at all (default: none), in each document (within) or for all documents together (between). Not used in our experiments
  --n_rerank                Top-k sentences to retrieve for reranking (default: 5). If rerank_mode is none, this is option is not used.
  --use_gpu                 Whether to use Faiss-gpu instead of cpu (default: cpu)
  --precompute_embed        Use precomputed embeds instead of calculating vector embeddings on the fly. See data_processing.py in main README file.
  --compress_embed          Use Unsupervised Neural Quantitazation to compress embedding size. Not used in our experiments

example:
foo@bar:~$ python -m src.retrieval.faiss.run_faiss_search --dataset_name=hover --setting=enwiki-2017-original
```
Given that we replace the entire document retrieval part of the original HoVer pipeline, the data format changes to what the next stage(s) need.
When retrieving for Sentence Selection the format becomes:
```json
{...},
{
    "id": "e822c12c-0f2c-4cb1-99cb-f4ff41fd436b", 
    "claim": "Yakuza Kiwami and Yakuza 0 were released on the PlayStation 2 gaming console.", 
    "context": [["Yakuza Kiwami 2", ["Ry\u016b ga Gotoku: Kiwami 2 (unofficially known as Yakuza Kiwami 2) is an action-adventure video game developed and published by Sega.", "It is a remake of the 2006 video game \"Yakuza 2\", and is the series' second remake title following 2016's \"Yakuza Kiwami\".", ...]], [...], [...]], 
    "supporting_facts": [["Yakuza Kiwami", 2], ["Yakuza 0", 3]]
},
{...}
```
When retrieving for Claim Verification the format becomes:

```json
{...},
{
    "id": "e822c12c-0f2c-4cb1-99cb-f4ff41fd436b", 
    "claim": "Yakuza Kiwami and Yakuza 0 were released on the PlayStation 2 gaming console.", 
    "context": "Yakuza Kiwami is a 2016 action-adventure game developed by Sega for the PlayStation 3 and PlayStation 4. Similarly to \"Yakuza 0\", the prequel installment before it, \"Yakuza Kiwami\" was released exclusively on PlayStation 4 in Europe and North America in August 2017. Yakuza 0 is an action-adventure video game developed and published by Sega.", 
    "label": "NOT_SUPPORTED"
},
{...}
```
Depending on the ``--hover_stage``, output will be saved to the subfolder ``sent_retrieval`` or ``claim_verification`` inside the claim data folder e.g. for hover this would be ``data/hover/sent_retrieval`` or ``data/hover/claim_verification``

## JPQ
For more comphehensive information see the README inside JPQ folder (same as their [official repository](https://github.com/jingtaozhan/JPQ/tree/main)).

First one would need to train the JPQ model. It should be noted that we only trained for document-level encoding of the corpus and not for passage-level encoding. The training requires the [STAR model](https://drive.google.com/drive/folders/18GrqZxeiYFxeMfSs97UxkVHwIhZPVXTc?usp=sharing) from the original authors other work [DRhard](https://github.com/jingtaozhan/DRhard). For this the qrel files first need to be created as follows using our helper script in ``src/tools``:

```console
usage: qrels_format_enwiki.py [-h] --setting=SETTING --dataset_name=DATASET_NAME [--first_paragraph_only]

positional arguments:
    --setting                       Name of the data corpus to run for (name of the enwiki folder files)
    --dataset_name                  Name of the claim dataset to run for [hover | wice]

optional arguments:
    -h, --help                      show this help message and exit
    --first_paragraph_only          Only process for the first paragraph. Not used in our experiments

Example:
foo@bar:~$ python -m src.tools.qrels_format_enwiki --dataset_name=hover --setting=enwiki-2017-original
```
The above essentially creates tsv files for the corpus data and qrels of the claim data splits pointing to their supporting documents of the corpus. All the files are stored in ``data/jpq_doc`` underneath the folder containing the dataset and setting name (e.g. enwiki-hover-dataset). It should be noted that training can take a long time on the gpu setup let alone on cpu. In the case of already trained JPQ model, only the corpus data is required for performing the inference step. When there is no trained model yet, training can be done by following the Preprocess (disregard MSMARCO step) and Training sections from the original README of JPQ. An example on HoVer claim dataset and enwiki-2017-original corpus with 96 subvectors will be executed with the following commands:
```console
foo@bar:~$ python -m src.retrieval.JPQ.jpq.preprocess --data_type 0 --dataset=hover --enwiki_name=enwiki-2017-original

foo@bar:~$ python -m src.retrieval.JPQ.jpq.run_init \
        --preprocess_dir ./data/jpq/doc/preprocess/ \
        --model_dir ./data/jpq/doc/star \
        --max_doc_length 512 \
        --output_dir ./data/jpq/doc/init/$subvectors \
        --subvector_num 96

foo@bar:~$ python -m src.retrieval.JPQ.jpq.run_train \
            --preprocess_dir ./data/jpq/doc/preprocess \
            --model_save_dir ./data/jpq/doc/train/m96/models \
            --log_dir ./data/jpq/doc/train/m96/log \
            --init_index_path ./data/jpq/doc/init/96/OPQ96,IVF1,PQ96x8.index \
            --init_model_path ./data/jpq/doc/star \
            --lambda_cut 10 \
            --gpu_search \
            --centroid_lr 1e-4 \
            --train_batch_size 32
```
As the now trained encoders are still Roberta model format, to convert them to the JPQ model run the following:
```console
usage: convert_model.py [-h] --data_type=DATA_TYPE --subvector_num=SUBVECTOR_NUM --enwiki_name=ENWIKI_NAME

positional arguments:
    --data_type                     Int value to indicate whether to convert for doc (default: 0) or passage (1) encoders.
    -subvector_num                  Amount of subvectors to use (default: 96)
    --enwiki_name                   Name of the data corpus to run for (name of the enwiki folder files)

optional arguments:
    -h, --help                      show this help message and exit


Example:
foo@bar:~$ python -m src.retrieval.JPQ.jpq.convert_model --data_type=0 --subvector_num=96 --enwiki_name=enwiki-2017-original
```
When trying to run inference on it, we created the following script. Do note that due to input changes from the encoders, the previously trained index does not work therefore retraining is needed.


```console
usage: run_inference.py [-h] --setting=SETTING --dataset_name=DATASET_NAME [--data_split] [--batch_size] [--top_k_nn] [--use_gpu] [--first_paragraph_only] [--sent_select]

positional arguments:
    --setting                       Name of the data corpus to run for (name of the enwiki folder files)
    --dataset_name                  Name of the claim dataset to run for [hover | wice]

optional arguments:
    -h, --help                    show this help message and exit
    --data_split                  Datasplit to run inference for [train | dev | test]. In case not used, perform retrieval for train and dev.
    --batch_size                  Batch size for inference (default: 128)
    --top_k_nn                    Number of nearest neighbour documents to retrieve (default: 5)
    --use_gpu                     Use Faiss-gpu instead of cpu (default: cpu)
    --first_paragraph_only        Only process for the first paragraph
    --sent_select                 Format data for Sent Retrieval stage of HoVer

Example:
python -m src.retrieval.JPQ.run_inference --dataset_name=hover --setting=enwiki-2017-original --subvectors_num=96 --use_gpu
```
