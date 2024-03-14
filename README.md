# Efficient Fact-checking through Supporting Fact Extraction from Large Data Collections


## Quick Links

- [Efficient Fact-checking through Supporting Fact Extraction from Large Data Collections](#efficient-fact-checking-through-supporting-fact-extraction-from-large-data-collections)
  - [Quick Links](#quick-links)
  - [Folder structure](#folder-structure)
  - [Dependencies](#dependencies)
  - [Setup](#setup)
  - [Pipeline](#pipeline)
  - [Citation](#citation)
  - [Referenced Work](#referenced-work)
  - [Coding Practices](#coding-practices)


## Folder structure
```
├── *data      
│   ├── db_files
│   ├── embed_files
│   ├── enwiki_files
│   ├── hover_files
│   ├── hover
│   ├── wice
│   ├── jpq_doc
├── *model
├── *out
│   ├── hover
│   │   ├── exp1.0
│   │   ├── bm25
│   │   ├── faiss
│   ├── wice
│   │   ├── exp1.0
│   │   ├── bm25
│   │   ├── faiss
├── scripts
├── src
│   ├── hover
│   ├── retrieval
│   │   ├── bm25
│   │   ├── faiss
│   │   ├── JPQ
│   │   ├── unq
│   ├── tools
│   ├── wikiextractor
├── .gitignore
├── grounding_env.yml
└── hover_env.yml
```

The (sub)folders of data, model, and out are instantiated when setting up the project. The folder structure is laid out as follows:
* data: Should contain all corpus data (wikipedia files), claim data and generally intermediate data for running the pipeline.
* model: Should contain the document and query encoders as well as possible custom models for reranking.
* out: will contain the model checkpoints for each HoVer pipeline stage as well as the predictions for each checkpoint.
* scripts: Bash scripts for running the three retrieval pipeline settings (bm25, faiss, jpq) as well as shell scripts for setting up data folder and downloading wikipedia dump.
* src: Contains the main hover pipeline, retrieval folder containing the retrieval methods, tools for misecellenous helper code and forked [wikiextractor](https://github.com/qipeng/wikiextractor) for processing the wikipedia dump.

## Dependencies
For installing all dependencies, we recommend using Anaconda and installing the ``grounding_env.yml`` file. Please ensure that the environment is named "grounding" as the scripts will explicitly attempt to activate it. Alternatively, rename all instances in the scripts folder.

Since HoVer has a somewhat outdated codebase, and to avoid breaking existing working code, a separate environment YAML file, ``hover_env.yml``, has been created with older dependencies. Similarly to the "grounding" environment, ensure that this environment is named "hover".

```console
foo@bar:~$ conda env create  --file=grounding_env.yml
foo@bar:~$ conda env create  --file=hover_env.yml
```

## Setup
The intial first step is creating the necessary folders to hold data to run the pipeline:
```console
foo@bar:~$ ./scripts/download_data.sh
```

For our experiments we used the following corpus and claim data:
* processed [2017 English wikipedia dump](https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2) provided by [HotPotQA](https://hotpotqa.github.io/wiki-readme.html) with the [HoVer](https://github.com/hover-nlp/hover/tree/main/data/hover) as claim data.
* processed [2023 English Wikipedia dump](https://dumps.wikimedia.org/enwiki/20231201/) from `https://dumps.wikimedia.org/` with the [WiCE](https://github.com/ryokamoi/wice/tree/main/data/entailment_retrieval/claim) as claim data.

```json
{ ... },
{
    "id": 47094971,
    "url": "https://en.wikipedia.org/wiki?curid=47094971",
    "title": "Dark Souls",
    "text": [["Dark Souls"], ["Dark Souls is a series of action role-playing games</a> developed by <a href=\"/wiki/FromSoftware\" >FromSoftware </a> and published by <a href=\"/wiki/Bandai_Namco_Entertainment\">Bandai Namco Entertainment</a>. ", ...], ... ],
},
{ ... },
```

To use a different wikipedia dump, download it from aforementioned wikimedia dump site (for example the [latest one](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2)). To simplify the process we automated it into the following script (requires dependencies as explained further down):
```console
foo@bar:~$ ./scripts/preprocess_wiki_dump.sh
```
The exact steps this script performs are as follows. To use the data for the pipeline first some preprocessing needs to be done using HotPotQA's forked [WikiExtractor](https://github.com/qipeng/wikiextractor) to get the above format. First install the WikiExtractor as pip module:
```console
foo@bar:~$ cd src/wikiextractor
foo@bar:~$ pip install .
foo@bar:~$ cd ../..
```

For preprocessing it, run the following:

```console
foo@bar:~$ python -m src.wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2 -o data/enwiki_files/enwiki-latest --no-templates -c --json
```

Lastly, as the sentences are still contenated some further processing is needed. This requires a NLP model for sentence splitting. In our case we provide implementation to use either [StanfordCoreNLP](https://stanfordnlp.github.io/CoreNLP/) as well as [spaCy's en_core_web_lg model](https://spacy.io/models/en#en_core_web_lg). 

For StanfordCoreNLP run inside the folder on a seperate terminal:
```console
foo@bar:~$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000
```
For spaCY install the [package](https://spacy.io/usage) with the correct model:

```console
foo@bar:~$ python -m spacy download en_core_web_lg
```
Lastly, for splitting the english wikipedia articles sentences (with --use_spacy being an optional argument to pass along):
```console
foo@bar:~$ python -m src.tools.enwiki_processing --start_loc=enwiki-latest --end_loc=enwiki-latest-original --process_function=preprocess
```

## Pipeline
For the reranking setup, one would first need to pre-compute the supporting facts in the data corpus (wikipedia files). This can be invoked with the following command:
```console
usage: enwiki_processing.py [-h] --first_loc=FIRST_LOC --second_loc=SECOND_LOC --process_function=PROCESS_FUNCTION [--first_paragraph_only] [--do_mmr] [--use_spacy]

positional arguments:
  --first_loc           Input Folder containing wiki bz2 files to process for

  --second_loc          Output Folder containing wiki bz2 files to save the processed files to. 
                        For fusion this will be folder to combine from instead.

  --process_function    Type of processing on the corpus:
                            - preprocess: Sentence splitting text of wiki-extracted dump.
                            - claim_detect: Perform claim-detection using Huggingface model.
                            - wice: Perform claim-detection using a custom model. Note: Not 
                            used in our work but serves as example how to integrate in our code.
                            - cite: Extract citations from corpus. Requires online connection 
                            due to asynchronous scraping as well as either the StanfordCoreNLP 
                            being run in the background or downloaded spaCy model in-place.
                            - fusion: Combine two corpus datasets by searching for entries in 
                            the first location that are empty and fill them in with data from 
                            the second location.
                        Note: for all reranking setups, this will create an additional field 
                        for each article 'fact_text' instead of overwriting the 'text' field.

optional arguments:
    --mmr               Store top-k sentences from the supporting fact extraction. Used only in 
                        the custom model code.
    
    --use_spacy         Use spaCy model instead of StanfordCoreNLP for sentence splitting. Used 
                        for process_function preprocess or cite .

Commands for recreating our experiments:
foo@bar:~$ python -m src.tools.enwiki_processing --start_loc=enwiki-latest-original --end_loc=enwiki-latest-claim --process_function=claim_detect

foo@bar:~$ python -m src.tools.enwiki_processing --start_loc=enwiki-latest-original --end_loc=enwiki-latest-cite --process_function=cite --use_spacy

foo@bar:~$ python -m src.tools.enwiki_processing --start_loc=enwiki-latest-cite --end_loc=enwiki-latest-claim --process_function=fusion

```

Before running the pipeline, the following step of converting the data corpus to database files is also required.
```console
usage: data_processing.py [-h] --setting=SETTING [--split_sent] [--first_para_only] [--store_original]

positional arguments:
    --setting           The (reranked) corpus files to process for. Converts the bz2 files 
                        structure into single database file containing title as id and text as 
                        values. For the text, the sentences will be concatenated with [SENT] as 
                        tokenizer in-between. Additionally temporarily creates and measures a 
                        single json file containing all entries for the corpus size.

optional arguments:
    --split_sent        Store invidiual sentences in the database instead of the concatenated 
                        text per wikipedia article.

    --first_para_only   Only process for the first paragraph and saves database file with 
                        `-first` suffix else with `-full` suffix. Not used in our work due to 
                        Wikipedia not having citations in the lead sections of their articles.
    
    --store_original    Stores the 'text' field values of the wikipedia bz2 files into the 
                        database instead of the 'fact_text'. Only used for the non-reranked 
                        data corpus in our experiments.

Commands for recreating our experiments:
foo@bar:~$ python -m src.tools.data_processing --setting=enwiki-latest-original --store_original --pre_compute_embed

foo@bar:~$ python -m src.tools.data_processing --setting=enwiki-latest-claim --pre_compute_embed


foo@bar:~$ python -m src.tools.data_processing --setting=enwiki-latest-cite --pre_compute_embed


foo@bar:~$ python -m src.tools.data_processing --setting=enwiki-latest-fusion --pre_compute_embed

```

For running the entire HoVer pipeline following our three of adjustments, we have created the following scripts to run.

```console
foo@bar:~$ ./scripts/run_bm25_pipeline.sh CLAIM_NAME SETTING BM25_TYPE
Note: Requires Elasticsearch instance to be run in the background. Additionally when using BM25_TYPE 'original', requires sentence splitting to be in-place similar to the supporting fact extraction part

foo@bar:~$ ./scripts/run_faiss_pipeline.sh CLAIM_NAME SETTING HOVER_STAGE RETRIEVAL_MODE
Note: Pre-computing the vector embeddings can speed-up the index construction part. Requires the Sentence tranformers model all-MiniLM-L6-v2.

foo@bar:~$ ./scripts/run_compress_pipeline.sh CLAIM_NAME SETTING HOVER_STAGE RETRIEVAL_MODE SUBVECTORS
Note: Training encoders from scratch is possible, by passing the ensuring there is no encoders folder (setting name) in the data/jpq_doc folder.

arguments:
  CLAIM_NAME            Name of the claim dataset to run for e.g. hover or wice

  SETTING               Name of the data corpus to run for (name of the database file)

  BM25_TYPE             Run for original pipeline setting or reranking (skips sentence 
                        selection stage) e.g. original or -

  HOVER_STAGE           Perform immediate claim verification stage or include sentence 
                        selection e.g. sent_select or -

  RETRIEVAL_MODE        Perform cpu (default) or gpu retrieval e.g. cpu or gpu.

  SUBVECTORS            Amount of subvectors to use e.g. 96 (default)



```

For more information on each HoVer stage, read the README inside ``src/hover``.

## Citation


## Referenced Work
Below follow the work from which we utilised the existing code base and modified certain parts of it.

*"HoVer: A Dataset for Many-Hop Fact Extraction And Claim Verification"* in Findings of EMNLP, 2020. ([paper](https://arxiv.org/abs/2011.03088) | [code](https://github.com/hover-nlp/hover)).  
*"WiCE: Real-World Entailment for Claims in Wikipedia"* in Findings of EMNLP, 2023. ([paper](https://arxiv.org/abs/2303.01432) | [code](https://github.com/ryokamoi/wice)).  
*"Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance"* CIKM, 2021. ([paper](https://arxiv.org/abs/2108.00644) | [code](https://github.com/jingtaozhan/JPQ))  
*"Unsupervised Neural Quantization for Compressed-Domain Similarity Search"*  ([paper](https://arxiv.org/abs/1908.03883) | [code](https://github.com/stanis-morozov/unq))  
*"WikiExtractor" ([code](https://github.com/qipeng/wikiextractor))

## Coding Practices
To ensure consistent code style and readability, this project uses auto-formatting tools such as black and isort.
Additionally for readability and reproducibility purposes, we use [type hints](https://docs.python.org/3/library/typing.html) for majority of the functions we created and use docstrings following the [Google format](https://github.com/NilsJPWerner/autoDocstring/blob/HEAD/docs/google.md).
